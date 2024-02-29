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
    do_n4: bool = True,
    do_crop: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, ErrCode]:
    logger.print("Prepare input image", Log_Type.STAGE)
    with logger:
        # Crop Down
        uncropped_input = np.zeros(mri_nii.shape)
        try:
            # Enforce to range [0, 1500]
            mri_nii.normalize_to_range_(min_value=0, max_value=9000, verbose=logger)
            crop = mri_nii.compute_crop(dist=0) if do_crop else (slice(None, None), slice(None, None), slice(None, None))
        except ValueError:
            logger.print("Image Nifty is empty, skip this", Log_Type.FAIL)
            return None, ErrCode.EMPTY

        cropped_nii = mri_nii.apply_crop(crop)
        logger.print(f"Crop down from {mri_nii.shape} to {cropped_nii.shape}", verbose=verbose)

        # N4 Bias field correction
        if do_n4:
            n4_start = perf_counter()
            cropped_nii, _ = n4_bias(cropped_nii)  # PIR
            logger.print(f"N4 Bias field correction done in {perf_counter() - n4_start} sec", verbose=True)

        # Enforce to range [0, 1500]
        cropped_nii.normalize_to_range_(min_value=0, max_value=1500, verbose=logger)

        # Uncrop again
        uncropped_input[crop] = cropped_nii.get_array()
        logger.print(f"Uncrop back from {cropped_nii.shape} to {uncropped_input.shape}", verbose=verbose)

        # Apply padding
        padded_input = np.pad(uncropped_input, pad_width=pad_size, mode="edge")
        logger.print(f"Padded from {uncropped_input.shape} to {padded_input.shape}", verbose=verbose)

    # Return pre-processed
    return mri_nii.set_array(padded_input), ErrCode.OK
