# Call 'python -m unittest' on this folder
"""``predict_instance_mask(pad_size=n)`` must return a mask in the field of view it was handed.

The padding added before the cutouts were taken was never removed: `whole_vert_nii_uncropped.apply_pad(-pad_size)`
is not an in-place call, and its result was dropped. So any `pad_size > 0` returned a mask that is
`2 * pad_size` voxels larger per axis than the input -- while the docstring promised the padding was
"removed afterwards". Padding also happened *before* the rescale to the model resolution, so even a
rebound `apply_pad(-pad_size)` would have removed the wrong number of voxels on a mask whose zoom differs
from the model's.
"""

from __future__ import annotations

import unittest

import numpy as np
from TPTBox import NII, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.phase_instance import predict_instance_mask
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Inference_Config, SegmentationModel

logger = No_Logger()


class _ThirdsInstanceModel(SegmentationModel):
    """Deterministic stand-in: splits each cutout into three bands along the inferior axis."""

    def __init__(self, zoom: tuple[float, float, float] = (1.5, 1.5, 1.5)) -> None:
        self.logger = No_Logger()
        config = Segmentation_Inference_Config(
            logger=self.logger,
            modality=["SEG"],
            acquisition="sag",
            log_name="ThirdsInstanceModel",
            modeltype="unet",
            model_expected_orientation=("P", "I", "R"),
            available_folds=1,
            inference_augmentation=False,
            resolution_range=list(zoom),
            default_step_size=0.5,
            labels={1: 1, 2: 2, 3: 3},
            expected_inputs=["seg"],
            cutout_size=(24, 24, 24),
        )
        super().__init__(__file__, config, default_verbose=False, default_allow_tqdm=False)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        self.predictor = object()
        return self

    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:  # noqa: ARG002
        nii = input_nii[0]
        arr = nii.get_seg_array()
        out = np.zeros_like(arr)
        height = arr.shape[1]
        for band, lo in enumerate((0, height // 3, 2 * height // 3)):
            hi = height if band == 2 else (band + 1) * height // 3
            band_slice = out[:, lo:hi, :]
            band_slice[arr[:, lo:hi, :] != 0] = band + 1
        return {OutputType.seg: nii.set_array(out), OutputType.softmax_logits: None}


def _run(pad_size: int, zoom: tuple[float, float, float] = (1.5, 1.5, 1.5)) -> NII:
    _mri, subreg, _vert, _label = get_test_mri()
    model = _ThirdsInstanceModel(zoom).load()
    vert_nii, errcode = predict_instance_mask(subreg.copy(), model, debug_data={}, pad_size=pad_size, verbose=False)
    assert errcode == ErrCode.OK, errcode
    assert vert_nii is not None
    return vert_nii


class Test_Instance_Pad_Size(unittest.TestCase):
    def test_padding_is_removed_again(self):
        unpadded = _run(pad_size=0)
        padded = _run(pad_size=4)
        self.assertEqual(padded.shape, unpadded.shape)
        self.assertTrue(padded.assert_affine(other=unpadded), "pad_size changed the field of view")

    def test_padding_does_not_change_the_result(self):
        unpadded = _run(pad_size=0)
        padded = _run(pad_size=4)
        np.testing.assert_array_equal(padded.get_seg_array(), unpadded.get_seg_array())

    def test_padding_is_removed_when_the_model_rescales(self):
        # The fixture is at 1.5mm; a model at 3.0mm halves the grid, so `pad_size` input voxels and
        # `pad_size` model voxels are different distances. The output must still match the unpadded run.
        unpadded = _run(pad_size=0, zoom=(3.0, 3.0, 3.0))
        padded = _run(pad_size=3, zoom=(3.0, 3.0, 3.0))
        self.assertEqual(padded.shape, unpadded.shape)
        self.assertTrue(padded.assert_affine(other=unpadded))


if __name__ == "__main__":
    unittest.main()
