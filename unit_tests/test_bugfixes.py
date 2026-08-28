# Call 'python -m unittest' on this folder
"""Targeted tests for the bugs found in the 2.0 audit pass.

Each test names the defect it pins down, so a regression is self-explaining.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri

from spineps.phase_instance import find_prediction_couple
from spineps.phase_labeling import perform_labeling_step
from spineps.phase_post import detect_and_solve_merged_vertebra
from spineps.phase_semantic import semantic_bounding_box_clean
from spineps.seg_pipeline import pipeline_version
from spineps.utils.citation_reminder import OPT_OUT_ENV_VAR, reminder_disabled
from spineps.utils.find_min_cost_path import DEFAULT_REGION_STARTS, find_most_probably_sequence

logger = No_Logger()


def _nii(arr: np.ndarray, zoom: float = 1.0) -> NII:
    """Wrap an array as a ``(P, I, R)`` oriented segmentation without permuting it.

    Axis 0 grows posteriorly, axis 1 inferiorly, axis 2 to the right -- so the index arithmetic in the
    tests below reads the same way the pipeline code does.
    """
    affine = np.array([[0, 0, zoom, 0], [-zoom, 0, 0, 0], [0, -zoom, 0, 0], [0, 0, 0, 1.0]])
    nii = NII(nib.Nifti1Image(arr.astype(np.uint8), affine=affine), seg=True)
    assert nii.orientation == ("P", "I", "R"), nii.orientation
    return nii


class Test_Merged_Vertebra_Background(unittest.TestCase):
    """The IVD components must be offset without dragging the background out of 0.

    A plain ``subreg_cc += OFFSET`` turned every background voxel into one giant phantom "IVD"
    whose center of mass sits in the middle of the volume. When the anatomy lives in the inferior
    half, that phantom sorts *above* every real structure and takes the first slot in the
    height-sorted list -- which is exactly the slot the split-C2 heuristic inspects.
    """

    @staticmethod
    def _fixture() -> tuple[NII, NII]:
        # PIR: axis 1 is the inferior axis, so "high index" == inferior == low in the body.
        shape = (12, 40, 12)
        seg = np.zeros(shape, dtype=np.uint8)
        vert = np.zeros(shape, dtype=np.uint8)
        # Everything sits in the inferior half so the background centroid lands above it all.
        # A small upper vertebra (1) stacked directly onto a large one (2) -> should be merged.
        seg[3:9, 24:27, 3:9] = 49  # Vertebra_Corpus_border, instance 1
        vert[3:9, 24:27, 3:9] = 1
        seg[3:9, 27:37, 3:9] = 49  # instance 2, clearly larger
        vert[3:9, 27:37, 3:9] = 2
        seg[3:9, 37:39, 3:9] = 100  # an IVD below them, so the disc branch has something to find
        return _nii(seg), _nii(vert)

    def test_top_two_instances_are_merged(self):
        seg_nii, vert_nii = self._fixture()
        detect_and_solve_merged_vertebra(seg_nii, vert_nii)
        self.assertNotIn(1, vert_nii.unique(), "the small top instance should have been merged into its neighbour")
        self.assertIn(2, vert_nii.unique())


class Test_Semantic_Bounding_Box_Clean(unittest.TestCase):
    """The region kept must be the union of the incorporated components' boxes, not just the largest."""

    def test_incorporated_component_survives(self):
        shape = (16, 60, 16)
        arr = np.zeros(shape, dtype=np.uint8)
        arr[6:10, 10:40, 6:10] = 49  # largest component
        arr[6:10, 42:52, 6:10] = 49  # second component, below it, within the inferior margin
        seg = _nii(arr, zoom=1.0)
        n_second = int((arr[6:10, 42:52, 6:10] != 0).sum())
        kept = semantic_bounding_box_clean(seg.copy()).get_seg_array()
        # The old code cropped to the largest component's box only, so the tail of an incorporated
        # component beyond that box was silently deleted.
        self.assertEqual(int((kept[6:10, 42:52, 6:10] != 0).sum()), n_second, "an incorporated component must survive whole")
        self.assertGreater(kept[6:10, 10:40, 6:10].sum(), 0)

    def test_far_component_is_dropped(self):
        shape = (16, 60, 16)
        arr = np.zeros(shape, dtype=np.uint8)
        arr[6:10, 4:34, 6:10] = 49  # largest component, superior
        arr[1:3, 56:59, 1:3] = 49  # far away in every axis -> never incorporated
        seg = _nii(arr, zoom=1.0)
        cleaned = semantic_bounding_box_clean(seg.copy())
        self.assertEqual(cleaned.get_seg_array()[1:3, 56:59, 1:3].sum(), 0)


class Test_Prediction_Couple_Partner_Drop(unittest.TestCase):
    """Two partners that agree with the anchor but not with each other cannot both be kept."""

    @staticmethod
    def _sparse(spans: dict) -> dict:
        """Build sparse predictions from ``{(com, label): (start, stop)}`` spans along the first axis."""
        from spineps.phase_instance import SparsePrediction

        return {
            key: SparsePrediction((slice(a, b), slice(0, 1), slice(0, 1)), np.ones((b - a, 1, 1), dtype=bool))
            for key, (a, b) in spans.items()
        }

    def test_overlapping_partners_are_both_kept(self):
        preds = self._sparse({(1, 1): (0, 6), (0, 1): (0, 5), (2, 1): (1, 7)})
        couple, agreement = find_prediction_couple(1, 1, preds, 3)
        self.assertEqual(len(couple), 3, couple)
        self.assertGreater(agreement, 0)

    def test_non_overlapping_partner_is_dropped(self):
        # Both partners clear the Dice threshold against the anchor, but they are disjoint from each
        # other -- so they cannot both be the same vertebra as the anchor.
        preds = self._sparse({(1, 1): (0, 10), (0, 1): (0, 4), (2, 1): (6, 10)})
        couple, _agreement = find_prediction_couple(1, 1, preds, 3)
        self.assertEqual(len(couple), 2, f"the weaker of two disagreeing partners must be dropped, got {couple}")
        self.assertIn((1, 1), couple, "the anchor always stays in its own couple")


class Test_Sparse_Dice_Matches_Dense(unittest.TestCase):
    """`sparse_dice` must agree with `np_dice` on the equivalent full-volume masks, exactly."""

    def test_random_boxes(self):
        from TPTBox.core.np_utils import np_dice

        from spineps.phase_instance import SparsePrediction, sparse_dice

        rng = np.random.default_rng(0)
        shape = (24, 20, 18)
        for _ in range(60):
            preds = []
            for _side in range(2):
                starts = [int(rng.integers(0, s - 4)) for s in shape]
                sizes = [int(rng.integers(2, min(9, s - st))) for st, s in zip(starts, shape)]
                bbox = tuple(slice(st, st + sz) for st, sz in zip(starts, sizes))
                mask = rng.random(tuple(sizes)) < 0.6
                preds.append(SparsePrediction(bbox, mask))
            dense = []
            for pr in preds:
                full = np.zeros(shape, dtype=np.uint8)
                full[pr.bbox] = pr.mask
                dense.append(full)
            self.assertAlmostEqual(sparse_dice(preds[0], preds[1]), float(np_dice(dense[0], dense[1])), places=12)

    def test_disjoint_and_empty(self):
        from spineps.phase_instance import SparsePrediction, sparse_dice

        a = SparsePrediction((slice(0, 2), slice(0, 2), slice(0, 2)), np.ones((2, 2, 2), dtype=bool))
        far = SparsePrediction((slice(9, 11), slice(0, 2), slice(0, 2)), np.ones((2, 2, 2), dtype=bool))
        empty = SparsePrediction((slice(0, 2), slice(0, 2), slice(0, 2)), np.zeros((2, 2, 2), dtype=bool))
        self.assertEqual(sparse_dice(a, far), 0.0)
        self.assertEqual(sparse_dice(empty, empty), 1.0, "np_dice returns 1.0 when both masks are empty")
        self.assertEqual(sparse_dice(a, a), 1.0)


class Test_Min_Cost_Path_Arguments(unittest.TestCase):
    def test_region_skip_without_rel_cost(self):
        """`allow_skip_at_region` used to dereference a `regions_ranges` that was never built."""
        rng = np.random.default_rng(0)
        cost = rng.random((4, 24))
        fcost, fpath, _mcp = find_most_probably_sequence(cost, allow_skip_at_region=[0], region_rel_cost=None)
        self.assertEqual(len(fpath), 4)
        self.assertIsInstance(fcost, float)

    def test_regions_argument_is_not_mutated(self):
        before = list(DEFAULT_REGION_STARTS)
        regions = list(DEFAULT_REGION_STARTS)
        rng = np.random.default_rng(1)
        find_most_probably_sequence(rng.random((3, 24)), regions=regions, region_rel_cost=None)
        self.assertEqual(regions, before, "the caller's region list must not be appended to")
        self.assertEqual(list(DEFAULT_REGION_STARTS), before)


class Test_Labeling_Guards(unittest.TestCase):
    def test_empty_instance_mask_returns_unchanged(self):
        from unit_tests.test_inference_mocked import Labeling_Model_Dummy

        mri, _subreg, vert, _label = get_test_mri()
        empty = vert.set_array(np.zeros_like(vert.get_seg_array()))
        model = Labeling_Model_Dummy().load()
        out = perform_labeling_step(model, mri, empty, subreg_nii=None)
        self.assertEqual(len(out.unique()), 0)

    def test_no_subreg_with_c1_enabled_does_not_crash(self):
        from unittest.mock import MagicMock

        from unit_tests.test_inference_mocked import Labeling_Model_Dummy, _fake_run_all_seg_instances

        mri, _subreg, vert, _label = get_test_mri()
        model = Labeling_Model_Dummy().load()
        model.run_all_seg_instances = MagicMock(side_effect=_fake_run_all_seg_instances)
        # disable_c1=False previously dereferenced the (None) subregion mask.
        out = perform_labeling_step(model, mri, vert.copy(), subreg_nii=None, disable_c1=False)
        self.assertIsInstance(out, NII)


class Test_Process_Dataset_Compatibility(unittest.TestCase):
    def test_incompatible_model_raises(self):
        """`process_dataset` used to log "stop program" and then carry on regardless."""
        from spineps.seg_enums import Acquisition, Modality
        from spineps.seg_run import process_dataset
        from unit_tests.test_inference_mocked import SegmentationModelDummy

        model = SegmentationModelDummy().load()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            # The dummy model is sagittal T2w/SEG/T1w; asking for an axial CT cannot work.
            process_dataset(
                dataset_path=Path(d),
                model_instance=model,
                model_semantic=model,
                modalities=(Modality.CT, Acquisition.ax),
                save_log_data=False,
            )


class Test_Pipeline_Version(unittest.TestCase):
    def test_version_is_not_read_from_the_callers_git_repo(self):
        pipeline_version.cache_clear()
        with patch("spineps.seg_pipeline._package_version", return_value="2.0.0") as m:
            self.assertEqual(pipeline_version(), "2.0.0")
        m.assert_called_once()
        pipeline_version.cache_clear()


class Test_Citation_Opt_Out(unittest.TestCase):
    def test_env_var_disables_reminder(self):
        old = os.environ.get(OPT_OUT_ENV_VAR)
        try:
            os.environ[OPT_OUT_ENV_VAR] = "1"
            self.assertTrue(reminder_disabled())
            os.environ[OPT_OUT_ENV_VAR] = "TRUE"
            self.assertTrue(reminder_disabled())
            os.environ.pop(OPT_OUT_ENV_VAR)
            self.assertFalse(reminder_disabled())
        finally:
            if old is None:
                os.environ.pop(OPT_OUT_ENV_VAR, None)
            else:
                os.environ[OPT_OUT_ENV_VAR] = old


class Test_Separating_Components(unittest.TestCase):
    """`get_separating_components` splits a merged corpus into two parts.

    ``np_erode_msk`` / ``np_dilate_msk`` mutate and return their input. Dilating ``spart``/``tpart`` in
    place therefore grew the very arrays the function returns as "the two separated components", so it
    handed back two overlapping blobs -- and ``get_plane_split`` derived its separating plane from their
    smeared centers of mass.
    """

    def test_dumbbell_splits_into_disjoint_parts(self):
        from spineps.phase_instance import get_separating_components

        arr = np.zeros((24, 12, 12), dtype=np.uint8)
        arr[2:10, 2:10, 2:10] = 1  # first body
        arr[14:22, 2:10, 2:10] = 1  # second body
        arr[10:14, 5:7, 5:7] = 1  # thin bridge that erosion breaks
        spart, tpart, spart_dil, tpart_dil, stpart = get_separating_components(arr, connectivity=3)
        self.assertGreater(spart.sum(), 0)
        self.assertGreater(tpart.sum(), 0)
        self.assertEqual((spart & tpart).sum(), 0, "the two parts must be disjoint")
        self.assertIn(3, np.unique(stpart), "the dilated parts must end up touching")
        # the dilations are strictly larger than the parts they came from
        self.assertGreater(spart_dil.sum(), spart.sum())
        self.assertGreater(tpart_dil.sum(), tpart.sum())

    def test_unsplittable_shape_fails_with_a_readable_error(self):
        """A uniform bar has no waist to split at; the fallback branch used to die with a bare KeyError."""
        from spineps.phase_instance import get_separating_components

        arr = np.zeros((20, 16, 16), dtype=np.uint8)
        arr[4:16, 6:10, 6:10] = 1
        with self.assertRaises(Exception) as ctx:
            get_separating_components(arr, connectivity=3)
        self.assertNotIsInstance(ctx.exception, KeyError, "the failure must name the problem, not blow up on a missing key")
        self.assertTrue(str(ctx.exception), "the exception must carry a message")


class Test_Endplate_Labels_Reach_The_Semantic_Mask(unittest.TestCase):
    """The split superior/inferior endplate labels must survive into the returned semantic mask.

    ``NII.extract_label`` binarises unless ``keep_label=True``, so the final extract in
    ``add_ivd_ep_vert_label`` collapsed the whole split to 1 and the semantic mask came back with
    endplates labelled ``1`` -- a label that means nothing in the subregion space.
    """

    def test_superior_and_inferior_plates_are_present(self):
        from TPTBox import Location

        from unit_tests.test_regression_golden import run_post_phase_synthetic

        seg_cleaned, _vert = run_post_phase_synthetic()
        labels = seg_cleaned.unique()
        self.assertIn(Location.Vertebral_Body_Endplate_Superior.value, labels)
        self.assertIn(Location.Vertebral_Body_Endplate_Inferior.value, labels)
        self.assertNotIn(1, labels, "1 is not a subregion label; it was the binarised endplate mask")


class Test_Fix_Wrong_Posterior_Instance_Label(unittest.TestCase):
    """A detached arcus fragment must be relabelled to the single instance it touches.

    The function now crops to each vertebra's bounding box before running connected components; the
    per-component windows and the write-back have to stay aligned with the full volume.
    """

    def test_detached_arcus_is_reassigned(self):
        from TPTBox import Location

        from spineps.utils.proc_functions import fix_wrong_posterior_instance_label

        shape = (40, 40, 20)
        sem = np.zeros(shape, dtype=np.uint8)
        inst = np.zeros(shape, dtype=np.uint8)

        # instance 1: a corpus high up (small I index == superior)
        sem[6:16, 4:12, 6:14] = Location.Vertebra_Corpus_border.value
        inst[6:16, 4:12, 6:14] = 1
        # instance 2: a corpus below it, with its own arcus
        sem[6:16, 20:30, 6:14] = Location.Vertebra_Corpus_border.value
        sem[16:22, 20:30, 8:12] = Location.Arcus_Vertebrae.value
        inst[6:22, 20:30, 6:14] = 2
        # a stray arcus-only fragment carrying instance 1's label but sitting on instance 2
        sem[22:25, 22:26, 9:11] = Location.Arcus_Vertebrae.value
        inst[22:25, 22:26, 9:11] = 1

        sem_nii = _nii(sem)
        inst_nii = _nii(inst)
        out = fix_wrong_posterior_instance_label(sem_nii, inst_nii, logger=logger).get_seg_array()
        self.assertTrue(np.all(out[22:25, 22:26, 9:11] == 2), "the stray arcus should follow the instance it touches")
        self.assertTrue(np.all(out[6:16, 4:12, 6:14] == 1), "the real instance-1 corpus must be untouched")
