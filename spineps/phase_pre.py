"""Pre-processing phase: crop, N4 bias correction, and intensity normalization of the input MRI before segmentation."""

from __future__ import annotations

import inspect

# from utils.predictor import nnUNetPredictor
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from TPTBox import NII, Log_Type, to_nii

if TYPE_CHECKING:
    from pathlib import Path

from spineps.seg_enums import ErrCode
from spineps.seg_pipeline import logger
from spineps.utils.proc_functions import n4_bias
from spineps.utils.resolution import REFERENCE_ZOOM

# Input intensities are rescaled into this range before segmentation.
NORMALIZE_MIN_VALUE = 0
NORMALIZE_MAX_VALUE = 1500
# Physical margin kept around the detected spine when cropping a Vibe scan.
VIBE_CROP_MARGIN_MM = 25 * min(REFERENCE_ZOOM)


def _has_logger_arg(func) -> bool:
    """Check whether a callable accepts a ``logger`` keyword argument.

    Args:
        func (Callable): The function whose signature is inspected.

    Returns:
        bool: True if ``logger`` is among the function's parameters, else False.
    """
    return "logger" in inspect.signature(func).parameters


def compute_crop(
    nii: NII, out_file: str | Path, dataset_id=100, ddevice: Literal["cpu", "cuda", "mps"] = "cuda", gpu=0, max_folds=None, logger=None
) -> tuple[slice, slice, slice]:
    """Run the Vibe whole-body segmentation and compute a crop region around the spine.

    Segments the input with ``run_vibeseg``, keeps only the spine-relevant labels (IVD, vertebra body,
    vertebra posterior elements, and sacrum), and returns a bounding-box crop expanded by ``VIBE_CROP_MARGIN_MM``.

    Args:
        nii (NII): Input MRI image to segment and crop.
        out_file: Path where the Vibe segmentation output is written.
        dataset_id (int, optional): Vibe model/dataset identifier passed to ``run_vibeseg``. Defaults to 100.
        ddevice (Literal["cpu", "cuda", "mps"], optional): Compute device for inference. Defaults to "cuda".
        gpu (int, optional): GPU index used when running on CUDA. Defaults to 0.
        max_folds (int | None, optional): Maximum number of model folds to ensemble. Defaults to None (all folds).
        logger (optional): Logger forwarded to ``run_vibeseg`` when that version supports it. Defaults to None.

    Returns:
        tuple[slice, slice, slice]: The crop slices around the segmented spine, with a ``VIBE_CROP_MARGIN_MM`` margin.
    """
    from TPTBox.core.vert_constants import Full_Body_Instance_Vibe
    from TPTBox.segmentation import run_vibeseg

    if _has_logger_arg(run_vibeseg):
        out = run_vibeseg(nii, out_file, dataset_id=dataset_id, ddevice=ddevice, gpu=gpu, max_folds=max_folds, logger=logger)
    else:  # backwards compatibility, can be removed if we force to a new version of TPTBox than 30.Apr.26
        out = run_vibeseg(nii, out_file, dataset_id=dataset_id, ddevice=ddevice, gpu=gpu, max_folds=max_folds)
    seg = to_nii(out, True)
    seg.extract_label_(
        [
            Full_Body_Instance_Vibe.IVD,
            Full_Body_Instance_Vibe.vertebra_body,
            Full_Body_Instance_Vibe.vertebra_posterior_elements,
            Full_Body_Instance_Vibe.sacrum,
        ]
    )
    return seg.compute_crop(0, dist=VIBE_CROP_MARGIN_MM / min(seg.zoom))


def preprocess_input(
    mri_nii: NII,
    debug_data: dict,  # noqa: ARG001
    pad_size: int = 4,
    proc_normalize_input: bool = True,
    proc_do_n4_bias_correction: bool = True,
    proc_crop_input: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, ErrCode]:
    """Pre-process an input MRI for segmentation: normalize, crop, N4-correct, and pad.

    Optionally rescales intensities to ``[NORMALIZE_MIN_VALUE, NORMALIZE_MAX_VALUE]``, crops away empty
    background to speed up computation, applies N4 bias field correction on the crop, re-normalizes,
    writes the processed crop back into the full image, and finally pads the volume by ``pad_size`` on every side.

    Args:
        mri_nii (NII): Input grayscale MRI image.
        debug_data (dict): Dictionary for collecting intermediate results (unused here, reserved for parity).
        pad_size (int, optional): Number of voxels of edge padding added on each side per axis. Defaults to 4.
        proc_normalize_input (bool, optional): Whether to rescale intensities into the normalization range. Defaults to True.
        proc_do_n4_bias_correction (bool, optional): Whether to apply N4 bias field correction. Defaults to True.
        proc_crop_input (bool, optional): Whether to crop away background before processing. Defaults to True.
        verbose (bool, optional): Emit additional progress logging. Defaults to False.

    Returns:
        tuple[NII | None, ErrCode]: The padded, pre-processed image and ``ErrCode.OK``; or ``(None, ErrCode.EMPTY)``
        if the input image is empty.
    """
    logger.print("Prepare input image", Log_Type.STAGE)
    mri_nii = mri_nii.copy()
    with logger:
        # Crop Down
        try:
            # Enforce to range [0, 1500]
            if proc_normalize_input:
                mri_nii.normalize_to_range_(min_value=NORMALIZE_MIN_VALUE, max_value=NORMALIZE_MAX_VALUE, verbose=False)
                crop = mri_nii.compute_crop(dist=0) if proc_crop_input else (slice(None, None), slice(None, None), slice(None, None))
            else:
                crop = (
                    mri_nii.compute_crop(minimum=mri_nii.min(), dist=0)
                    if proc_crop_input
                    else (slice(None, None), slice(None, None), slice(None, None))
                )
        except ValueError:
            logger.print("Image Nifty is empty, skip this", Log_Type.FAIL)
            return None, ErrCode.EMPTY

        cropped_nii = mri_nii.apply_crop(crop)
        logger.print(f"Crop down from {mri_nii.shape} to {cropped_nii.shape}", verbose=verbose)

        # N4 Bias field correction
        if proc_do_n4_bias_correction:
            n4_start = perf_counter()
            cropped_nii, _ = n4_bias(cropped_nii)  # PIR
            logger.print(f"N4 Bias field correction done in {perf_counter() - n4_start} sec", verbose=True)

        # Enforce to range [NORMALIZE_MIN_VALUE, NORMALIZE_MAX_VALUE]
        if proc_normalize_input:
            cropped_nii.normalize_to_range_(min_value=NORMALIZE_MIN_VALUE, max_value=NORMALIZE_MAX_VALUE, verbose=logger)

        # Uncrop again
        # uncropped_input[crop] = cropped_nii.get_array()
        mri_nii[crop] = cropped_nii
        logger.print(f"Uncrop back from {cropped_nii.shape} to {mri_nii.shape}", verbose=verbose)

        # Apply padding
        padded_nii = mri_nii.pad_to(tuple(mri_nii.shape[i] + (2 * pad_size) for i in range(3)))
        logger.print(f"Padded from {mri_nii.shape} to {padded_nii.shape}", verbose=verbose)

    # Return pre-processed
    return padded_nii, ErrCode.OK
