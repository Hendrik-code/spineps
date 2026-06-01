# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
"""Tests for the GPU/model-inference code paths, emulating the networks with unittest.mock.

These tests reuse the established dummy-model + MagicMock pattern from test_semantic.py so the
surrounding orchestration (input preparation, cutout collection, prediction merging, labeling and
output mapping) is exercised without any real model weights or a GPU.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.lab_model import VertLabelingClassifier
from spineps.phase_instance import predict_instance_mask
from spineps.phase_labeling import VERT_CLASSES, perform_labeling_step, run_model_for_vert_labeling
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, Segmentation_Model

logger = No_Logger()


class DummyPredictor:
    """Minimal stand-in for a loaded network predictor."""

    def __init__(self) -> None:
        pass


def _dummy_seg_config(cutout_size: tuple[int, int, int] = (48, 48, 32)) -> Segmentation_Inference_Config:
    """Build an inference config for a dummy segmentation model (no weights needed)."""
    return Segmentation_Inference_Config(
        logger=No_Logger(),
        modality=["T2w", "SEG", "T1w"],
        acquisition="sag",
        log_name="DummySegModel",
        modeltype="unet",
        model_expected_orientation=("P", "I", "R"),
        available_folds=1,
        inference_augmentation=False,
        resolution_range=[1.5, 1.5, 1.5],  # equals the test fixture zoom -> no rescaling
        default_step_size=0.5,
        labels={1: 1},
        cutout_size=cutout_size,
    )


class Segmentation_Model_Dummy(Segmentation_Model):
    """A Segmentation_Model whose load() installs a dummy predictor instead of real weights."""

    def __init__(self, cutout_size: tuple[int, int, int] = (48, 48, 32)) -> None:
        self.logger = No_Logger()
        super().__init__(__file__, _dummy_seg_config(cutout_size), default_verbose=False, default_allow_tqdm=False)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.predictor = DummyPredictor()
        return self

    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:  # noqa: ARG002
        return {OutputType.seg: input_nii[0], OutputType.softmax_logits: None}


class Labeling_Model_Dummy(VertLabelingClassifier):
    """A VertLabelingClassifier whose load() installs a dummy predictor instead of real weights."""

    def __init__(self) -> None:
        self.logger = No_Logger()
        config = Segmentation_Inference_Config(
            logger=self.logger,
            modality=["T2w", "SEG", "T1w"],
            acquisition="sag",
            log_name="DummyLabelModel",
            modeltype="classifier",
            model_expected_orientation=("P", "I", "R"),
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[1.0, 1.0, 1.0],
            default_step_size=0.5,
            labels={1: 1},
        )
        super().__init__(__file__, config, default_verbose=False, default_allow_tqdm=False)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.predictor = DummyPredictor()
        return self


def _vert_softmax(peak_class: int) -> np.ndarray:
    """Return a length-VERT_CLASSES softmax-like vector peaked at ``peak_class``."""
    arr = np.full(VERT_CLASSES, 0.01, dtype=float)
    arr[min(max(peak_class, 0), VERT_CLASSES - 1)] = 0.89
    return arr / arr.sum()


def _fake_run_all_seg_instances(img: NII, seg: NII, *args, **kwargs):  # noqa: ARG001
    """Emulate VertLabelingClassifier.run_all_seg_instances with a deterministic VERT head.

    Returns one prediction per unique vertebra label in ``seg`` (in ascending order), each peaked at a
    consecutive cervical class so the resulting path is a valid, increasing sequence.
    """
    labels = [int(v) for v in seg.unique() if v != 0]
    predictions: dict[int, dict] = {}
    for offset, v in enumerate(sorted(labels)):
        soft = _vert_softmax(1 + offset)  # C2, C3, C4, ...
        predictions[v] = {"soft": {"VERT": soft}, "pred": {"VERT": int(np.argmax(soft))}}
    return predictions


class Test_Labeling_Inference_Mocked(unittest.TestCase):
    def test_run_model_for_vert_labeling(self):
        mri, _subreg, vert, _label = get_test_mri()
        model = Labeling_Model_Dummy().load()
        model.run_all_seg_instances = MagicMock(side_effect=_fake_run_all_seg_instances)

        labelmap, _fcost, _fpath, fpath_post, _costlist, _mcp, predictions = run_model_for_vert_labeling(model, mri, vert)
        # One prediction and one labelmap entry per input vertebra (5, 6, 7).
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(labelmap), 3)
        self.assertEqual(len(fpath_post), 3)
        model.run_all_seg_instances.assert_called()

    def test_perform_labeling_step_relabels(self):
        mri, subreg, vert, _label = get_test_mri()
        model = Labeling_Model_Dummy().load()
        model.run_all_seg_instances = MagicMock(side_effect=_fake_run_all_seg_instances)

        out = perform_labeling_step(model, mri, vert.copy(), subreg_nii=subreg)
        self.assertIsInstance(out, NII)
        # Same spatial frame as the input vertebra mask.
        self.assertTrue(out.assert_affine(other=vert))
        # The instance labels were remapped to the labeling model's output classes.
        self.assertGreater(len(out.unique()), 0)


class Test_Segment_Scan_Mocked(unittest.TestCase):
    def test_segment_scan_padding_round_trip(self):
        mri, subreg, _vert, _label = get_test_mri()
        model = Segmentation_Model_Dummy()
        # run() echoes its (padded, reoriented) input back as the segmentation.
        model.run = MagicMock(side_effect=lambda input_nii, verbose=False: {OutputType.seg: input_nii[0], OutputType.softmax_logits: None})  # noqa: ARG005

        result = model.segment_scan(
            mri,
            pad_size=3,
            resample_to_recommended=False,
            resample_output_to_input_space=True,
            verbose=False,
        )
        seg = result[OutputType.seg]
        self.assertIsInstance(seg, NII)
        # Padding added before inference is removed again -> output matches the input shape.
        self.assertEqual(seg.shape, mri.shape)
        model.run.assert_called_once()

    def test_segment_scan_without_resample_back(self):
        mri, subreg, _vert, _label = get_test_mri()
        model = Segmentation_Model_Dummy()
        model.run = MagicMock(return_value={OutputType.seg: subreg.copy(), OutputType.softmax_logits: None})

        result = model.segment_scan(
            mri,
            pad_size=0,
            resample_to_recommended=False,
            resample_output_to_input_space=False,
            verbose=False,
        )
        self.assertIn(OutputType.seg, result)
        self.assertIsInstance(result[OutputType.seg], NII)


class Test_Instance_Inference_Mocked(unittest.TestCase):
    @staticmethod
    def _fake_segment_scan(cut_nii: NII, **kwargs):  # noqa: ARG004
        """Emulate the instance model: split the cutout's corpus into a 1/2/3 three-vertebra hierarchy."""
        arr = cut_nii.get_seg_array()
        out = np.zeros_like(arr)
        corpus = np.argwhere(arr != 0)
        if len(corpus) > 0:
            # Split along the axis with the largest extent into thirds (above=1, center=2, below=3).
            extents = corpus.max(axis=0) - corpus.min(axis=0)
            axis = int(np.argmax(extents))
            coords = corpus[:, axis]
            lo, hi = coords.min(), coords.max() + 1
            third = max((hi - lo) / 3.0, 1.0)
            for c in corpus:
                bucket = int((c[axis] - lo) / third)
                out[c[0], c[1], c[2]] = min(bucket, 2) + 1
        return {OutputType.seg: cut_nii.set_array(out), OutputType.softmax_logits: None}

    def test_predict_instance_mask_runs(self):
        _mri, subreg, _vert, _label = get_test_mri()
        model = Segmentation_Model_Dummy(cutout_size=(48, 48, 32))
        model.segment_scan = MagicMock(side_effect=self._fake_segment_scan)

        whole_vert_nii, errcode = predict_instance_mask(
            subreg.copy(),
            model,
            debug_data={},
            proc_corpus_clean=False,
            proc_inst_clean_small_cc_artifacts=False,
            verbose=False,
        )
        self.assertEqual(errcode, ErrCode.OK)
        self.assertIsInstance(whole_vert_nii, NII)
        self.assertEqual(whole_vert_nii.shape, subreg.shape)
        # At least one vertebra instance was produced and the model was queried per corpus cutout.
        self.assertGreater(len([v for v in whole_vert_nii.unique() if v != 0]), 0)
        model.segment_scan.assert_called()

    def test_predict_instance_mask_empty_without_corpus(self):
        _mri, subreg, _vert, _label = get_test_mri()
        # Remove the corpus-border label (49) so the instance phase has nothing to work with.
        no_corpus = subreg.copy()
        no_corpus[no_corpus == 49] = 0
        model = Segmentation_Model_Dummy()
        model.segment_scan = MagicMock(side_effect=self._fake_segment_scan)

        result, errcode = predict_instance_mask(no_corpus, model, debug_data={}, proc_corpus_clean=False)
        self.assertIsNone(result)
        self.assertEqual(errcode, ErrCode.EMPTY)


if __name__ == "__main__":
    unittest.main()
