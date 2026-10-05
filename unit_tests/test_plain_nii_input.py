# Call 'python -m unittest' on this folder
"""An uncompressed ``.nii`` input must be segmented like a ``.nii.gz`` one.

Plain ``.nii`` is what `dcm2niix` writes without `-z y`, and what a good share of public datasets ship.
SPINEPS accepted only ``.nii.gz``: the CLI appended the suffix to whatever it was given, `BIDS_FILE.file`
is keyed by extension so the three `file["nii.gz"]` lookups raised `KeyError`, and the dataset query
filtered on that one extension. Outputs are written as ``.nii.gz`` either way.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel
from spineps.seg_run import segment_image
from spineps.seg_utils import NIFTI_FILE_TYPES, check_input_model_compatibility, input_image_path

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


class Test_Input_Image_Path(unittest.TestCase):
    def test_both_extensions_are_accepted(self):
        self.assertEqual(NIFTI_FILE_TYPES, ("nii.gz", "nii"))
        mri, *_ = get_test_mri()
        for name in ("sub-01_T2w.nii.gz", "sub-01_T2w.nii"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                path = Path(td) / name
                mri.save(path, verbose=False)
                img_ref = BIDS_FILE(str(path), dataset=td, verbose=False)
                self.assertEqual(input_image_path(img_ref), path)

    def test_compatibility_check_works_on_plain_nii(self):
        mri, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub-01_T2w.nii"
            mri.save(path, verbose=False)
            img_ref = BIDS_FILE(str(path), dataset=td, verbose=False)
            model = _SemanticDummy(mri).load()
            self.assertTrue(check_input_model_compatibility(img_ref, model=model, img_nii=mri, verbose=False))


class Test_Segment_Plain_Nii(unittest.TestCase):
    def test_full_pipeline_on_a_plain_nii(self):
        mri, subreg, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            path = directory / "sub-01_T2w.nii"
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
            # the masks are written compressed regardless of how the input was stored
            self.assertTrue(output_paths["out_spine"].name.endswith(".nii.gz"), output_paths["out_spine"].name)


class Test_Cli_Accepts_Plain_Nii(unittest.TestCase):
    def test_run_sample_does_not_reject_nii(self):
        import argparse
        from unittest import mock

        from spineps import entrypoint

        self.assertEqual(entrypoint.NIFTI_SUFFIXES, (".nii.gz", ".nii"))
        mri, *_ = get_test_mri()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub-01_T2w.nii"
            mri.save(path, verbose=False)
            parser = argparse.ArgumentParser()
            entrypoint.parser_arguments(parser)
            opt = parser.parse_args([])
            opt.input = str(path)
            opt.model_semantic, opt.model_instance, opt.model_labeling = "t2w", "instance", "none"
            with (
                mock.patch.object(entrypoint, "get_semantic_model"),
                mock.patch.object(entrypoint, "get_instance_model"),
                mock.patch.object(entrypoint, "segment_image", return_value=({"out_spine": path}, ErrCode.OK)),
            ):
                entrypoint.run_sample(opt)  # must not raise


if __name__ == "__main__":
    unittest.main()
