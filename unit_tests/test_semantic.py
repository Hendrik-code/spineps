# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.phase_pre import preprocess_input
from spineps.phase_semantic import predict_semantic_mask
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, Segmentation_Model

logger = No_Logger()


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
            model_expected_orientation=["P", "I", "R"],
            available_folds=1,
            inference_augmentation=False,
            resolution_range=[0.75, 0.75, 1.65],
            default_step_size=0.5,
            labels={1: 1},
        )
        super().__init__(model_folder, inference_config, default_verbose, default_allow_tqdm)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.print("Model loaded from", self.model_folder, verbose=True)
        self.predictor = object()
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,  # noqa: ARG002
    ) -> dict[OutputType, NII | None]:
        assert len(input_nii) == 1, "Unet3D does not support more than one input"
        return {OutputType.seg: input_nii}


class Test_Semantic_Phase(unittest.TestCase):
    def test_phase_preprocess(self):
        mri, subreg, vert, label = get_test_mri()
        for pad_size in range(7):
            origin_diff = max([d * float(pad_size) for d in mri.zoom]) + 1e-4
            # print(origin_diff)
            preprossed_input, errcode = preprocess_input(mri, debug_data={}, pad_size=pad_size, verbose=True)
            print(mri)
            print(preprossed_input)

            # backchanged_origin = tuple(preprossed_input.origin[idx] + origin_diff[idx] for idx in range(3))
            self.assertTrue(preprossed_input.assert_affine(origin=mri.origin, error_tolerance=origin_diff))
            # affine=mri.affine,
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
            mri, model, debug_data=debug_data, verbose=True, proc_clean_small_cc_artifacts=False
        )
        predicted_volumes = seg_nii.volumes()
        ref_volumes = subreg.volumes()
        print(predicted_volumes)
        print(ref_volumes)
        for i, v in ref_volumes.items():
            self.assertEqual(v, predicted_volumes[i])
        self.assertEqual(errcode, ErrCode.OK)
