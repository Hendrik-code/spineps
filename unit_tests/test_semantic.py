# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.phase_pre import preprocess_input
from spineps.phase_semantic import predict_semantic_mask
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, Segmentation_Model, run_inference
from spineps.seg_utils import check_input_model_compatibility, check_model_modality_acquisition

logger = No_Logger()


class DummyPredictor:
    def __init__(self) -> None:
        pass

    def predict_single_npy_array(self, arr: np.ndarray):
        return arr


class Segmentation_Model_Dummy(Segmentation_Model):
    def __init__(
        self,
        model_folder: str | Path = __file__,
        inference_config: Segmentation_Inference_Config | None = None,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        self.logger = No_Logger()
        inference_config = Segmentation_Inference_Config(
            logger=self.logger,
            modality=["T2w", "SEG", "T1w"],
            acquisition="sag",
            log_name="DummySegModel",
            modeltype="unet",
            model_expected_orientation=("P", "I", "R"),
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[0.75, 0.75, 1.65],
            default_step_size=0.5,
            labels={1: 1},
        )
        super().__init__(model_folder, inference_config, default_verbose, default_allow_tqdm)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.print("Model loaded from", self.model_folder, verbose=True)
        self.predictor = DummyPredictor()
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,  # noqa: ARG002
    ) -> dict[OutputType, NII | None]:
        assert len(input_nii) == 1, "Unet3D does not support more than one input"
        return {OutputType.seg: input_nii}


class Test_Semantic_Phase(unittest.TestCase):
    def test_compatibility(self):
        from TPTBox import BIDS_FILE
        from TPTBox.tests.test_utils import get_tests_dir

        from spineps.seg_enums import Acquisition, InputType, Modality

        mri, subreg, vert, label = get_test_mri()
        input_path = get_tests_dir().joinpath("sample_mri", "sub-mri_label-6_T2w.nii.gz")
        model = Segmentation_Model_Dummy()

        print(input_path)
        bf = BIDS_FILE(input_path, dataset=input_path.parent)
        compatible = check_input_model_compatibility(bf, model)
        self.assertFalse(compatible)

        compatible = check_input_model_compatibility(bf, model, ignore_labelkey=True)
        self.assertTrue(compatible)

        compatible = check_model_modality_acquisition(model, mod_pair=(Modality.CT, Acquisition.ax))
        self.assertTrue(not compatible)

        # change model artifically
        print(mri.get_plane())
        # mri.get_plane = MagicMock(return_value="sag")
        model.inference_config.acquisition = Acquisition.ax
        model.inference_config.modalities = [Modality.CT]
        compatible = check_input_model_compatibility(bf, model, ignore_labelkey=True)
        self.assertFalse(compatible)

    def test_phase_preprocess(self):
        mri, subreg, vert, label = get_test_mri()
        for pad_size in range(7):
            origin_diff = max([d * float(pad_size) for d in mri.zoom]) + 1e-4
            # print(origin_diff)
            preprossed_input, errcode = preprocess_input(mri, debug_data={}, pad_size=pad_size, verbose=True)
            print(mri)
            print(preprossed_input)
            self.assertTrue(preprossed_input.assert_affine(origin=mri.origin, origin_tolerance=origin_diff))
            self.assertTrue(preprossed_input.assert_affine(rotation=mri.rotation, orientation=mri.orientation, zoom=mri.zoom))
            self.assertEqual(errcode, ErrCode.OK)
            for idx, s in enumerate(mri.shape):
                self.assertEqual(s + (2 * pad_size), preprossed_input.shape[idx])

    def test_segment_scan(self):
        mri, subreg, vert, label = get_test_mri()
        model = Segmentation_Model_Dummy()
        model.run = MagicMock(return_value={OutputType.seg: subreg, OutputType.softmax_logits: None})
        debug_data = {}
        seg_nii, softmax_logits, errcode = predict_semantic_mask(
            mri,
            model,
            debug_data=debug_data,
            verbose=True,
            proc_clean_small_cc_artifacts=False,
            proc_remove_inferior_beyond_canal=True,
        )
        predicted_volumes = seg_nii.volumes()
        ref_volumes = subreg.volumes()
        print(predicted_volumes)
        print(ref_volumes)
        for i, v in ref_volumes.items():
            self.assertEqual(v, predicted_volumes[i])
        self.assertEqual(errcode, ErrCode.OK)

    def test_run_inference(self):
        mri, subreg, vert, label = get_test_mri()
        model = Segmentation_Model_Dummy().load()
        s_arr = subreg.get_seg_array()
        model.predictor.predict_single_npy_array = MagicMock(return_value=(s_arr, s_arr[np.newaxis, :]))

        seg_arr, _ = run_inference(mri, model.predictor)
        seg_nii = subreg.set_array(seg_arr)
        # debug_data = {}
        predicted_volumes = seg_nii.volumes()
        ref_volumes = subreg.volumes()
        print(predicted_volumes)
        print(ref_volumes)
        for i, v in ref_volumes.items():
            self.assertEqual(v, predicted_volumes[i])


#
# seg_nii = model.run([mri])[OutputType.seg]
#
# predicted_volumes = seg_nii.volumes()
# ref_volumes = subreg.volumes()
# print(predicted_volumes)
# print(ref_volumes)
# for i, v in ref_volumes.items():
#    self.assertEqual(v, predicted_volumes[i])
