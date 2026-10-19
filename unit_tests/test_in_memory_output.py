# Call 'python -m unittest' on this folder
"""``output_in_memory=True`` must return the masks and leave the input folder alone.

Two defects made the in-memory API unusable on a real path input:

* the centroid file was saved unconditionally, so an in-memory run still wrote into the user's dataset --
  and crashed with ``FileNotFoundError`` on a fresh input, because the derivatives folder is only created
  by the mask saves that in-memory mode skips, while ``POI.save`` does not create parents, and
* if the derivatives were already there from an earlier run, the "all done" shortcut returned
  ``(output_paths, ErrCode.ALL_DONE)``, i.e. ``SpinepsResult.success`` was True with
  ``result.semantic is None``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, NII, POI, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel
from spineps.seg_run import segment_image

logger = No_Logger()


def _config(modality: list[str]) -> Segmentation_Inference_Config:
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
        cutout_size=(48, 48, 32),
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


def _input_file(directory: Path, mri: NII, name: str = "sub-01_T2w.nii.gz") -> BIDS_FILE:
    mri.save(directory / name, verbose=False)
    return BIDS_FILE(str(directory / name), dataset=str(directory), verbose=False)


class Test_In_Memory_Output(unittest.TestCase):
    def test_returns_masks_and_writes_nothing(self):
        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            img_ref = _input_file(directory, mri)
            result = segment_image(
                img_ref,
                model_semantic=_SemanticDummy(subreg).load(),
                model_instance=_InstanceDummy().load(),
                model_labeling=None,
                proc_sem_n4_bias_correction=False,
                return_output_instead_of_save=True,
                verbose=False,
            )
            self.assertEqual(len(result), 4, f"expected (seg, vert, ctd, errcode), got {result}")
            seg_nii, vert_nii, ctd, errcode = result
            self.assertEqual(errcode, ErrCode.OK)
            self.assertIsInstance(seg_nii, NII)
            self.assertIsInstance(vert_nii, NII)
            self.assertIsInstance(ctd, POI)
            written = sorted(p.name for p in directory.rglob("*") if p.is_file())
            self.assertEqual(written, ["sub-01_T2w.nii.gz"], f"in-memory mode wrote files: {written}")

    def test_already_processed_input_still_returns_masks(self):
        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            img_ref = _input_file(directory, mri)
            kwargs = {
                "model_semantic": _SemanticDummy(subreg).load(),
                "model_instance": _InstanceDummy().load(),
                "model_labeling": None,
                "proc_sem_n4_bias_correction": False,
                "verbose": False,
            }
            # First run writes a full derivatives folder ...
            _paths, errcode = segment_image(img_ref, **kwargs)
            self.assertEqual(errcode, ErrCode.OK)
            # ... which must not turn the next in-memory call into an empty "all done".
            result = segment_image(img_ref, return_output_instead_of_save=True, **kwargs)
            self.assertEqual(len(result), 4, f"expected (seg, vert, ctd, errcode), got {result}")
            self.assertIsInstance(result[0], NII)
            self.assertIsInstance(result[1], NII)
            self.assertEqual(result[3], ErrCode.OK)

    def test_save_mode_still_short_circuits(self):
        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            img_ref = _input_file(directory, mri)
            kwargs = {
                "model_semantic": _SemanticDummy(subreg).load(),
                "model_instance": _InstanceDummy().load(),
                "model_labeling": None,
                "proc_sem_n4_bias_correction": False,
                "verbose": False,
            }
            segment_image(img_ref, **kwargs)
            _paths, errcode = segment_image(img_ref, **kwargs)
            self.assertEqual(errcode, ErrCode.ALL_DONE)


class Test_Segment_Api_In_Memory(unittest.TestCase):
    def test_segment_on_a_path_returns_masks(self):
        from spineps.api import SpinepsPipeline

        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            mri.save(directory / "sub-01_T2w.nii.gz", verbose=False)
            pipeline = SpinepsPipeline.__new__(SpinepsPipeline)  # bypass model downloading
            pipeline.model_semantic = _SemanticDummy(subreg).load()
            pipeline.model_instance = _InstanceDummy().load()
            pipeline.model_labeling = None

            result = pipeline.segment(str(directory / "sub-01_T2w.nii.gz"), output_in_memory=True)
            self.assertTrue(result.success)
            self.assertIsInstance(result.semantic, NII)
            self.assertIsInstance(result.vertebra, NII)
            self.assertIsInstance(result.centroids, POI)


if __name__ == "__main__":
    unittest.main()
