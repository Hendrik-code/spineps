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
from typing import ClassVar
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import torch
from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.lab_model import VertLabelingClassifier
from spineps.phase_instance import predict_instance_mask
from spineps.phase_labeling import VERT_CLASSES, perform_labeling_step, run_model_for_vert_labeling
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel, SegmentationModelUnet3D

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


class SegmentationModelDummy(SegmentationModel):
    """A SegmentationModel whose load() installs a dummy predictor instead of real weights."""

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
        mri, _subreg, _vert, _label = get_test_mri()
        model = SegmentationModelDummy()
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
        model = SegmentationModelDummy()
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
        model = SegmentationModelDummy(cutout_size=(48, 48, 32))
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
        model = SegmentationModelDummy()
        model.segment_scan = MagicMock(side_effect=self._fake_segment_scan)

        result, errcode = predict_instance_mask(no_corpus, model, debug_data={}, proc_corpus_clean=False)
        self.assertIsNone(result)
        self.assertEqual(errcode, ErrCode.EMPTY)


class FakeClassifierPredictor:
    """Stand-in for a loaded PLClassifier: a multi-head network returning deterministic logits.

    Implements the small surface that VertLabelingClassifier._run_array uses: ``eval``, ``to``,
    ``forward`` (returning a per-head logits dict) and ``softmax``. Zero logits give a uniform,
    deterministic softmax, which is all the surrounding code needs.
    """

    HEADS: ClassVar[dict[str, int]] = {"VERT": VERT_CLASSES, "VERTGRP": 12, "REGION": 3, "VERTREL": 6, "VERTT13": 2, "FULLYVISIBLE": 2}

    def eval(self) -> FakeClassifierPredictor:
        return self

    def to(self, device) -> FakeClassifierPredictor:  # noqa: ARG002
        return self

    def softmax(self, v: torch.Tensor) -> torch.Tensor:
        return torch.softmax(v, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        return {name: torch.zeros((batch, n_classes)) for name, n_classes in self.HEADS.items()}

    __call__ = forward


def _make_classifier_with_fake_predictor(cutout: tuple[int, int, int] = (32, 32, 16)) -> Labeling_Model_Dummy:
    """Build a labeling model wired to a FakeClassifierPredictor, running entirely on CPU."""
    from monai.transforms import CenterSpatialCropd, Compose, NormalizeIntensityd

    model = Labeling_Model_Dummy()
    model.device = torch.device("cpu")
    model.final_size = cutout
    model.cutout_size = cutout  # normally set from the checkpoint in load()
    model.transform = Compose(
        [
            NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=False),
            CenterSpatialCropd(keys=["img", "seg"], roi_size=cutout),
        ]
    )
    model.predictor = FakeClassifierPredictor()
    return model


class Test_Classifier_Forward_Mocked(unittest.TestCase):
    def test_run_array(self):
        model = _make_classifier_with_fake_predictor()
        img_arr = np.arange(32 * 32 * 16, dtype=np.float32).reshape(32, 32, 16)
        seg_arr = (img_arr > img_arr.mean()).astype(np.float32)

        logits_soft, pred_cls = model._run_array(img_arr, seg_arr)
        # One entry per network head, with the VERT head holding a full class vector.
        self.assertEqual(set(logits_soft.keys()), set(FakeClassifierPredictor.HEADS.keys()))
        self.assertEqual(logits_soft["VERT"].shape, (VERT_CLASSES,))
        self.assertEqual(set(pred_cls.keys()), set(FakeClassifierPredictor.HEADS.keys()))
        self.assertEqual(np.asarray(pred_cls["VERT"]).ndim, 0)

    def test_run_array_without_seg(self):
        model = _make_classifier_with_fake_predictor()
        img_arr = np.ones((32, 32, 16), dtype=np.float32)
        # seg defaults to a clone of the image when omitted.
        logits_soft, _pred_cls = model._run_array(img_arr)
        self.assertEqual(logits_soft["VERT"].shape, (VERT_CLASSES,))

    def test_run_all_arrays(self):
        model = _make_classifier_with_fake_predictor()
        arrays = {5: np.ones((32, 32, 16), dtype=np.float32), 6: np.ones((32, 32, 16), dtype=np.float32)}
        predictions = model.run_all_arrays(arrays)
        self.assertEqual(set(predictions.keys()), {5, 6})
        for entry in predictions.values():
            self.assertIn("soft", entry)
            self.assertIn("pred", entry)
            self.assertEqual(entry["soft"]["VERT"].shape, (VERT_CLASSES,))

    def test_run_all_seg_instances_full_path(self):
        mri, _subreg, vert, _label = get_test_mri()
        model = _make_classifier_with_fake_predictor()
        # Drives reorient -> per-instance cutout -> _run_array for every label in the mask.
        predictions = model.run_all_seg_instances(mri, vert)
        expected = [int(v) for v in vert.unique() if v != 0]
        self.assertEqual(sorted(predictions.keys()), sorted(expected))
        for entry in predictions.values():
            self.assertEqual(entry["soft"]["VERT"].shape, (VERT_CLASSES,))


class Test_Same_Modelzoom(unittest.TestCase):
    @staticmethod
    def _model_with_zoom(zoom: tuple[float, float, float]) -> SegmentationModelDummy:
        # A fixed (length-3) resolution_range makes calc_recommended_resampling_zoom return it verbatim.
        model = SegmentationModelDummy()
        model.inference_config.resolution_range = tuple(zoom)
        return model

    def test_same_resolution_matches(self):
        a = self._model_with_zoom((1.0, 1.0, 1.0))
        b = self._model_with_zoom((1.0, 1.0, 1.0))
        self.assertTrue(a.same_modelzoom_as_model(b, (1.0, 1.0, 1.0)))

    def test_coarser_other_model_does_not_match(self):
        # Regression: model_zms > self_zms yields a negative per-axis difference that must NOT count
        # as a match (the bug was comparing the signed difference against the tolerance).
        a = self._model_with_zoom((1.0, 1.0, 1.0))
        b = self._model_with_zoom((2.0, 2.0, 2.0))
        self.assertFalse(a.same_modelzoom_as_model(b, (1.0, 1.0, 1.0)))

    def test_finer_other_model_does_not_match(self):
        a = self._model_with_zoom((2.0, 2.0, 2.0))
        b = self._model_with_zoom((1.0, 1.0, 1.0))
        self.assertFalse(a.same_modelzoom_as_model(b, (1.0, 1.0, 1.0)))

    def test_single_coarser_axis_does_not_match(self):
        # Only the inferior axis differs and the other model is coarser there (negative diff).
        a = self._model_with_zoom((1.0, 1.0, 1.0))
        b = self._model_with_zoom((1.0, 1.0, 2.0))
        self.assertFalse(a.same_modelzoom_as_model(b, (1.0, 1.0, 1.0)))


class _FakeUnetNetwork:
    def __init__(self, channels: int) -> None:
        self.channels = channels


class _FakeUnetPredictor:
    """Identity 'network': returns its input as logits, so softmax+argmax recovers the one-hot input label."""

    def __init__(self, channels: int) -> None:
        self.network = _FakeUnetNetwork(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_unet3d_test_model(n_classes: int = 4, cutout: tuple[int, int, int] = (6, 6, 6)) -> SegmentationModelUnet3D:
    """A real SegmentationModelUnet3D wired to an identity predictor (no weights, runs on CPU)."""
    config = Segmentation_Inference_Config(
        logger=No_Logger(),
        modality=["SEG"],
        acquisition="sag",
        log_name="TestUnet3D",
        modeltype="unet",
        model_expected_orientation=("P", "I", "R"),
        available_folds=1,
        inference_augmentation=False,
        resolution_range=[1.5, 1.5, 1.5],
        default_step_size=0.5,
        labels={1: 1, 2: 2, 3: 3},
        expected_inputs=["seg"],
        cutout_size=cutout,
    )
    model = SegmentationModelUnet3D(__file__, config, default_verbose=False, default_allow_tqdm=False)
    model.predictor = _FakeUnetPredictor(n_classes)
    model.device = torch.device("cpu")
    return model


def _seg_nii(arr: np.ndarray) -> NII:
    return NII(nib.Nifti1Image(arr.astype(np.uint8), affine=np.eye(4)), seg=True)


class Test_Unet3D_Batching(unittest.TestCase):
    """The batched instance path must produce exactly the same masks as predicting each cutout on its own."""

    def _cutouts(self, n: int = 5) -> list[NII]:
        rng = np.random.default_rng(0)
        return [_seg_nii(rng.integers(0, 4, size=(6, 6, 6))) for _ in range(n)]

    def test_run_batch_matches_run(self):
        # run_batch over a list must equal calling run() on each cutout individually (golden oracle).
        model = _make_unet3d_test_model()
        cutouts = self._cutouts()
        batched = model.run_batch(cutouts, batch_size=2)
        self.assertEqual(len(batched), len(cutouts))
        for cut, res in zip(cutouts, batched):
            single = model.run([cut])
            np.testing.assert_array_equal(res[OutputType.seg].get_seg_array(), single[OutputType.seg].get_seg_array())
        # distinct inputs must yield distinct outputs (guards against the batch collapsing to one result)
        outs = [r[OutputType.seg].get_seg_array() for r in batched]
        self.assertTrue(any(not np.array_equal(outs[0], o) for o in outs[1:]))

    def test_run_batch_size_invariant(self):
        # The result must not depend on how cutouts are chunked into forward passes.
        model = _make_unet3d_test_model()
        cutouts = self._cutouts()
        for r1, r9 in zip(model.run_batch(cutouts, batch_size=1), model.run_batch(cutouts, batch_size=9)):
            np.testing.assert_array_equal(r1[OutputType.seg].get_seg_array(), r9[OutputType.seg].get_seg_array())

    def test_run_batch_actually_batches_forward_calls(self):
        # 5 cutouts at batch_size 2 must take ceil(5/2)=3 forward passes, not 5 (the whole point of batching).
        model = _make_unet3d_test_model()
        real_forward = model.predictor.forward
        calls = {"n": 0}

        def counting_forward(x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return real_forward(x)

        model.predictor.forward = counting_forward
        model.run_batch(self._cutouts(5), batch_size=2)
        self.assertEqual(calls["n"], 3)

    def test_segment_scan_batch_matches_segment_scan(self):
        # The way the instance phase calls it: no resampling, batched == sequential segment_scan.
        model = _make_unet3d_test_model()
        cutouts = self._cutouts()
        kwargs = {"resample_to_recommended": False, "pad_size": 0, "resample_output_to_input_space": False}
        batched = model.segment_scan_batch(cutouts, batch_size=3, **kwargs)
        for cut, res in zip(cutouts, batched):
            single = model.segment_scan(cut, **kwargs)
            np.testing.assert_array_equal(res[OutputType.seg].get_seg_array(), single[OutputType.seg].get_seg_array())

    def test_set_test_time_augmentation(self):
        model = _make_unet3d_test_model()
        # the instance predictor has no use_mirroring attribute -> no-op, must not raise
        model.set_test_time_augmentation(False)

        class _MirroringPredictor:
            use_mirroring = True

        model.predictor = _MirroringPredictor()
        model.set_test_time_augmentation(False)
        self.assertFalse(model.predictor.use_mirroring)
        model.set_test_time_augmentation(True)
        self.assertTrue(model.predictor.use_mirroring)


if __name__ == "__main__":
    unittest.main()
