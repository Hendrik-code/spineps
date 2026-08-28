# Call 'python -m unittest' on this folder
"""Golden-output regression tests for the two whole-volume post-model phases.

The instance merge (``phase_instance``) and the combined post-processing (``phase_post``) are the two
places where memory/speed optimisations are most likely to silently change a label. These tests pin
their exact output on a deterministic fixture, so any refactor that is supposed to be behaviour
preserving has to prove it.

If a change here is *intended*, regenerate the digests with::

    python -m unit_tests.test_regression_golden
"""

from __future__ import annotations

import hashlib
import unittest

import nibabel as nib
import numpy as np
from TPTBox import NII, Location, No_Logger
from TPTBox.tests.test_utils import get_test_mri
from typing_extensions import Self

from spineps.phase_instance import predict_instance_mask
from spineps.phase_post import phase_postprocess_combined
from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import SegmentationModel
from spineps.utils.seg_modelconfig import Segmentation_Inference_Config

logger = No_Logger()


def digest(arr: np.ndarray) -> str:
    """Stable content digest of an array, including its shape and dtype kind."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(repr((a.shape, a.dtype.kind)).encode())
    h.update(a.astype(np.int64).tobytes())
    return h.hexdigest()[:32]


class ThirdsInstanceModel(SegmentationModel):
    """Deterministic stand-in for the instance model.

    Splits each cutout into three bands along the inferior axis and labels them 1/2/3, which is the
    shape of a real three-vertebra prediction -- enough to drive the couple search and the merge.
    """

    def __init__(self, cutout_size: tuple[int, int, int] = (24, 24, 24)) -> None:
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
            resolution_range=[1.5, 1.5, 1.5],
            default_step_size=0.5,
            labels={1: 1, 2: 2, 3: 3},
            expected_inputs=["seg"],
            cutout_size=cutout_size,
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


def run_instance_phase() -> NII:
    """Run the instance phase on the shared fixture and return the vertebra mask."""
    _mri, subreg, _vert, _label = get_test_mri()
    model = ThirdsInstanceModel().load()
    vert_nii, errcode = predict_instance_mask(subreg, model, debug_data={}, verbose=False)
    assert errcode == ErrCode.OK, errcode
    assert vert_nii is not None
    return vert_nii


def run_post_phase() -> tuple[NII, NII]:
    """Run the combined post-processing on the shared fixture."""
    mri, subreg, vert, _label = get_test_mri()
    return phase_postprocess_combined(mri, subreg, vert, model_labeling=None, debug_data={})


def synthetic_spine() -> tuple[NII, NII, NII]:
    """A deterministic multi-vertebra spine with discs and endplates, in (P, I, R).

    The shared TPTBox fixture only has three vertebrae, which barely exercises the endplate splitter --
    the loop that dominates post-processing. This one has six, each with its own corpus, arch, disc and
    endplate band, so superior/inferior plate assignment actually has neighbours to disagree about.
    """
    shape = (40, 120, 40)
    affine = np.array([[0, 0, 1.0, 0], [-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])
    seg = np.zeros(shape, dtype=np.uint8)
    vert = np.zeros(shape, dtype=np.uint8)
    img = np.zeros(shape, dtype=np.float32)

    pitch = 18  # vertebra + disc period along the inferior axis
    for n in range(6):
        top = 6 + n * pitch
        body = slice(top, top + 12)
        seg[10:26, body, 12:28] = Location.Vertebra_Corpus_border.value
        seg[26:32, body, 16:24] = Location.Arcus_Vertebrae.value  # posterior elements
        vert[10:32, body, 12:28] = n + 1
        # endplate band directly below the corpus, then the disc below that
        seg[10:26, top + 12 : top + 14, 12:28] = Location.Endplate.value
        seg[10:26, top + 14 : top + 18, 12:28] = Location.Vertebra_Disc.value
    seg[30:34, 4:112, 18:22] = Location.Spinal_Canal.value
    img[seg != 0] = 800.0

    def wrap(arr, is_seg):
        return NII(nib.Nifti1Image(arr, affine=affine), seg=is_seg)

    return wrap(img, False), wrap(seg, True), wrap(vert, True)


def run_post_phase_synthetic() -> tuple[NII, NII]:
    """Run the combined post-processing on the six-vertebra synthetic spine."""
    mri, subreg, vert = synthetic_spine()
    return phase_postprocess_combined(mri, subreg, vert, model_labeling=None, debug_data={})


# Regenerate with `python -m unit_tests.test_regression_golden` after an intentional change.
GOLDEN_INSTANCE = "d03a9f8081523184108c6f2698737ed5"
GOLDEN_POST_SEG = "98d67f4a9a4ad6c565e0d647b0466441"
GOLDEN_POST_VERT = "8928f04c5539c1eff12672743df80f61"
GOLDEN_SYNTH_SEG = "ab6a357788a70bbe4c53b9dc2a253e3e"
GOLDEN_SYNTH_VERT = "f0850727dd21c001bd661814103ca770"


class Test_Golden_Instance_Phase(unittest.TestCase):
    def test_instance_mask_unchanged(self):
        vert_nii = run_instance_phase()
        self.assertEqual(digest(vert_nii.get_seg_array()), GOLDEN_INSTANCE)


class Test_Golden_Post_Phase(unittest.TestCase):
    def test_postprocess_output_unchanged(self):
        seg_cleaned, vert_cleaned = run_post_phase()
        self.assertEqual(digest(seg_cleaned.get_seg_array()), GOLDEN_POST_SEG)
        self.assertEqual(digest(vert_cleaned.get_seg_array()), GOLDEN_POST_VERT)

    def test_postprocess_synthetic_spine_unchanged(self):
        seg_cleaned, vert_cleaned = run_post_phase_synthetic()
        self.assertEqual(digest(seg_cleaned.get_seg_array()), GOLDEN_SYNTH_SEG)
        self.assertEqual(digest(vert_cleaned.get_seg_array()), GOLDEN_SYNTH_VERT)

    def test_synthetic_spine_splits_endplates(self):
        """Sanity check on the fixture itself: the splitter must produce both plate labels."""
        seg_cleaned, vert_cleaned = run_post_phase_synthetic()
        labels = seg_cleaned.unique()
        self.assertIn(Location.Vertebral_Body_Endplate_Inferior.value, labels)
        self.assertIn(Location.Vertebral_Body_Endplate_Superior.value, labels)
        self.assertGreaterEqual(len([v for v in vert_cleaned.unique() if v < 40]), 5)


if __name__ == "__main__":
    seg_c, vert_c = run_post_phase()
    seg_s, vert_s = run_post_phase_synthetic()
    print(f'GOLDEN_INSTANCE = "{digest(run_instance_phase().get_seg_array())}"')
    print(f'GOLDEN_POST_SEG = "{digest(seg_c.get_seg_array())}"')
    print(f'GOLDEN_POST_VERT = "{digest(vert_c.get_seg_array())}"')
    print(f'GOLDEN_SYNTH_SEG = "{digest(seg_s.get_seg_array())}"')
    print(f'GOLDEN_SYNTH_VERT = "{digest(vert_s.get_seg_array())}"')
