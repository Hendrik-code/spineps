# Call 'python -m unittest' on this folder
"""An input NIfTI with a plain file name must be segmented, not refused.

Two separate gates used to reject anything that is not named ``sub-<id>_<modality>.nii.gz``:

* the output-naming layer raised ``AssertionError: ..._mod-myscan_...`` from inside TPTBox before a single
  voxel was read, and
* the model/input compatibility check read the trailing word of the file name as the modality, so
  ``myscan.nii.gz`` "was" modality ``myscan``, did not match the model, and the whole run returned
  ``ErrCode.COMPATIBILITY`` without writing anything.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.seg_enums import ErrCode, Modality, OutputType
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel
from spineps.seg_run import output_paths_from_input, segment_image
from spineps.seg_utils import check_input_model_compatibility

logger = No_Logger()


def _config(modality: list[str], cutout_size: tuple[int, int, int] = (48, 48, 32)) -> Segmentation_Inference_Config:
    return Segmentation_Inference_Config(
        logger=No_Logger(),
        modality=modality,
        acquisition="sag",
        log_name="DummyModel",
        modeltype="unet",
        model_expected_orientation=("P", "I", "R"),
        available_folds=1,
        inference_augmentation=False,
        resolution_range=[1.5, 1.5, 1.5],  # equals the test fixture zoom -> no rescaling
        default_step_size=0.5,
        labels={1: 1},
        cutout_size=cutout_size,
    )


class _SemanticDummy(SegmentationModel):
    """Echoes the fixture's subregion mask, resampled onto whatever grid it is handed."""

    def __init__(self, subreg: NII) -> None:
        self.logger = No_Logger()
        self._subreg = subreg
        super().__init__(__file__, _config(["T2w", "T1w"]), default_verbose=False, default_allow_tqdm=False)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.predictor = object()
        return self

    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:  # noqa: ARG002
        return {OutputType.seg: self._subreg.copy().resample_from_to(input_nii[0]), OutputType.softmax_logits: None}


class _InstanceDummy(SegmentationModel):
    """Splits each corpus cutout into a 1/2/3 three-vertebra hierarchy, like the real instance model."""

    def __init__(self) -> None:
        self.logger = No_Logger()
        super().__init__(__file__, _config(["SEG"]), default_verbose=False, default_allow_tqdm=False)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.predictor = object()
        return self

    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:  # noqa: ARG002
        return {OutputType.seg: input_nii[0], OutputType.softmax_logits: None}

    def segment_scan(self, input_image, **kwargs):  # noqa: ARG002
        cut_nii: NII = input_image
        arr = cut_nii.get_seg_array()
        out = np.zeros_like(arr)
        corpus = np.argwhere(arr != 0)
        if len(corpus) > 0:
            axis = int(np.argmax(corpus.max(axis=0) - corpus.min(axis=0)))
            lo, hi = corpus[:, axis].min(), corpus[:, axis].max() + 1
            third = max((hi - lo) / 3.0, 1.0)
            for c in corpus:
                out[c[0], c[1], c[2]] = min(int((c[axis] - lo) / third), 2) + 1
        return {OutputType.seg: cut_nii.set_array(out), OutputType.softmax_logits: None}


class Test_Output_Paths_For_Plain_File_Names(unittest.TestCase):
    def _bids_file(self, name: str, directory: Path) -> BIDS_FILE:
        mri, *_ = get_test_mri()
        mri.save(directory / name, verbose=False)
        return BIDS_FILE(str(directory / name), dataset=str(directory), verbose=False)

    def test_plain_file_name_gets_output_paths(self):
        with tempfile.TemporaryDirectory() as td:
            img_ref = self._bids_file("myscan.nii.gz", Path(td))
            paths = output_paths_from_input(img_ref, "derivatives_seg", None, input_format=img_ref.format)
        # The file name becomes the subject id instead of aborting the run.
        self.assertIn("sub-myscan", paths["out_spine"].name)
        self.assertTrue(paths["out_spine"].name.endswith("_msk.nii.gz"))

    def test_bids_conform_name_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            img_ref = self._bids_file("sub-01_T2w.nii.gz", Path(td))
            paths = output_paths_from_input(img_ref, "derivatives_seg", None, input_format=img_ref.format)
        self.assertEqual(paths["out_spine"].name, "sub-01_mod-T2w_seg-spine_msk.nii.gz")


class Test_Unknown_Modality_Is_Not_A_Mismatch(unittest.TestCase):
    def test_untagged_file_name_stays_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            mri, *_ = get_test_mri()
            path = Path(td) / "myscan.nii.gz"
            mri.save(path, verbose=False)
            img_ref = BIDS_FILE(str(path), dataset=td, verbose=False)
            model = _SemanticDummy(mri).load()
            self.assertTrue(check_input_model_compatibility(img_ref, model=model, img_nii=mri, verbose=False))

    def test_wrong_modality_in_the_name_still_fails(self):
        # "ct" IS a known modality key, so a CT file handed to an MR model remains a real mismatch.
        self.assertIn("ct", Modality.known_format_keys())
        with tempfile.TemporaryDirectory() as td:
            mri, *_ = get_test_mri()
            path = Path(td) / "sub-01_ct.nii.gz"
            mri.save(path, verbose=False)
            img_ref = BIDS_FILE(str(path), dataset=td, verbose=False)
            model = _SemanticDummy(mri).load()
            self.assertFalse(check_input_model_compatibility(img_ref, model=model, img_nii=mri, verbose=False))

    def test_known_format_keys_covers_every_defined_modality(self):
        keys = Modality.known_format_keys()
        for expected in ("T2w", "T1w", "ct", "vibe", "msk", "mpr"):
            self.assertIn(expected, keys)


class Test_Segment_Image_With_A_Plain_File_Name(unittest.TestCase):
    def test_full_pipeline_writes_outputs(self):
        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            path = directory / "myscan.nii.gz"
            mri.save(path, verbose=False)
            img_ref = BIDS_FILE(str(path), dataset=str(directory), verbose=False)

            output_paths, errcode = segment_image(
                img_ref,
                model_semantic=_SemanticDummy(subreg).load(),
                model_instance=_InstanceDummy().load(),
                model_labeling=None,
                proc_sem_n4_bias_correction=False,
                verbose=False,
            )
            self.assertEqual(errcode, ErrCode.OK)
            for key in ("out_spine", "out_vert", "out_ctd", "out_snap"):
                self.assertTrue(output_paths[key].is_file(), f"{key} was not written to {output_paths[key]}")


if __name__ == "__main__":
    unittest.main()
