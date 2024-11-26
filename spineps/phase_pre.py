# from utils.predictor import nnUNetPredictor
from time import perf_counter

import numpy as np
from TPTBox import NII, Location, Log_Type

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import fill_holes_labels, logger
from spineps.utils.proc_functions import clean_cc_artifacts, n4_bias


def preprocess_input(
    mri_nii: NII,
    debug_data: dict,  # noqa: ARG001
    pad_size: int = 4,
    proc_normalize_input: bool = True,
    proc_do_n4_bias_correction: bool = True,
    proc_crop_input: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, ErrCode]:
    logger.print("Prepare input image", Log_Type.STAGE)
    mri_nii = mri_nii.copy()
    with logger:
        # Crop Down
        try:
            # Enforce to range [0, 1500]
            if proc_normalize_input:
                mri_nii.normalize_to_range_(min_value=0, max_value=9000, verbose=False)
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

        # Enforce to range [0, 1500]
        if proc_normalize_input:
            cropped_nii.normalize_to_range_(min_value=0, max_value=1500, verbose=logger)

        # Uncrop again
        # uncropped_input[crop] = cropped_nii.get_array()
        mri_nii[crop] = cropped_nii
        logger.print(f"Uncrop back from {cropped_nii.shape} to {mri_nii.shape}", verbose=verbose)

        # Apply padding
        padded_nii = mri_nii.pad_to(tuple(mri_nii.shape[i] + (2 * pad_size) for i in range(3)))
        logger.print(f"Padded from {mri_nii.shape} to {padded_nii.shape}", verbose=verbose)

    # Return pre-processed
    return padded_nii, ErrCode.OK
