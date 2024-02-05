# from utils.predictor import nnUNetPredictor
from TPTBox import NII, Location, Log_Type
import numpy as np
from spineps.utils.proc_functions import clean_cc_artifacts, n4_bias
from spineps.seg_model import Segmentation_Model
from spineps.seg_enums import ErrCode, OutputType
from time import perf_counter

from spineps.seg_pipeline import logger, fill_holes_labels


def predict_semantic_mask(
    mri_nii: NII,
    model: Segmentation_Model,
    debug_data: dict,
    do_n4: bool = True,
    fill_holes: bool = True,
    clean_artifacts: bool = True,
    do_crop: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, NII | None, NII | None, np.ndarray | None, ErrCode]:
    """Predicts the semantic mask, takes care of rescaling, and back

    Args:
        mri_nii (NII): input mri image (grayscal, must be in range 0 -> ?)
        model (Segmentation_Model): Model to semantically segment with
        do_n4 (bool, optional): Wheter to apply n4 bias field correction. Defaults to True.
        fill_holes (bool, optional): Whether to fill holes in the output mask. Defaults to True.
        clean_artifacts (bool, optional): Whether to try and clean possible artifacts. Defaults to True.
        do_crop (bool, optional): Whether to apply cropping in order to speedup computation (min value in scan must be 0!). Defaults to True.
        verbose (bool, optional): If you want some more infos on whats happening. Defaults to False.

    Returns:
        tuple[NII | None, NII | None, NII | None, np.ndarray, ErrCode]: seg_nii, seg_nii_modelres, unc_nii, softmax_logits, ErrCode
    """
    logger.print("Predict Semantic Mask", Log_Type.STAGE)
    with logger:
        mri_nii = mri_nii.copy()
        uncropped_subregion_mask = np.zeros(mri_nii.shape)
        uncropped_unc_image = np.zeros(mri_nii.shape)
        uncropped_input_image = np.zeros(mri_nii.shape)
        mri_nii_rdy = mri_nii
        # orientation = mri_nii.orientation
        # mri_nii_rdy = mri_nii.reorient(verbose=logger)
        # shp = mri_nii_rdy.shape
        # zms = mri_nii_rdy.zoom
        # uncropped_subregion_mask = np.zeros(mri_nii_rdy.shape)
        # uncropped_unc_image = np.zeros(mri_nii_rdy.shape)
        try:
            crop = mri_nii_rdy.compute_crop_slice(dist=5) if do_crop else (slice(None, None), slice(None, None), slice(None, None))
        except ValueError as e:
            logger.print("Image Nifty is empty, skip this", Log_Type.FAIL)
            return None, None, None, None, ErrCode.EMPTY
        mri_nii_rdy.apply_crop_slice_(crop)
        logger.print(f"Crop down from {uncropped_subregion_mask.shape} to {mri_nii_rdy.shape}", verbose=verbose)

        if do_n4:
            n4_start = perf_counter()
            mri_nii_rdy, _ = n4_bias(mri_nii_rdy)  # PIR
            logger.print(f"N4 Bias field correction done in {perf_counter() - n4_start} sec", verbose=True)

        # Normalize to [0,1500]
        mri_nii_rdy += -mri_nii_rdy.min()  # min = 0
        mri_dtype = mri_nii_rdy.dtype
        max_value = mri_nii_rdy.max()
        if max_value > 1500:
            mri_nii_rdy *= 1500 / max_value
            mri_nii_rdy.set_dtype_(mri_dtype)

        results = model.segment_scan(
            mri_nii_rdy,
            pad_size=2,
            resample_to_recommended=True,
            verbose=verbose,
        )  # type:ignore
        seg_nii = results[OutputType.seg]
        unc_nii = results[OutputType.unc] if OutputType.unc in results else None
        seg_nii_modelres = results[OutputType.seg_modelres]
        softmax_logits = results[OutputType.softmax_logits]

        if len(seg_nii.unique()) == 0:
            logger.print("Subregion mask is empty, skip this", Log_Type.FAIL)
            return seg_nii, seg_nii_modelres, unc_nii, softmax_logits, ErrCode.EMPTY
        if clean_artifacts:
            seg_nii.set_array_(
                clean_cc_artifacts(
                    seg_nii,
                    logger=logger,
                    verbose=False,
                    labels=[
                        Location.Arcus_Vertebrae.value,
                        Location.Spinosus_Process.value,
                        Location.Costal_Process_Left.value,
                        Location.Costal_Process_Right.value,
                        Location.Superior_Articular_Left.value,
                        Location.Superior_Articular_Right.value,
                        Location.Inferior_Articular_Left.value,
                        Location.Inferior_Articular_Right.value,
                        Location.Vertebra_Corpus_border.value,
                        Location.Spinal_Canal.value,
                        Location.Vertebra_Disc.value,
                    ],
                    only_delete=True,
                    ignore_missing_labels=True,
                    cc_size_threshold=30,  # [
                ),
                verbose=verbose,
            )
        if fill_holes:
            seg_nii = seg_nii.fill_holes_(fill_holes_labels, verbose=logger)

        # Uncrop
        uncropped_subregion_mask[crop] = seg_nii.get_seg_array()
        uncropped_input_image[crop] = mri_nii_rdy.get_array()
        debug_data["a_input_preprocessed"] = mri_nii_rdy.set_array(uncropped_input_image)
        logger.print(f"Uncrop back from {seg_nii.shape} to {uncropped_subregion_mask.shape}", verbose=verbose)
        if isinstance(unc_nii, NII):
            uncropped_unc_image[crop] = unc_nii.get_array()
            unc_nii.set_array_(uncropped_unc_image)
        seg_nii.set_array_(uncropped_subregion_mask)

    return seg_nii, seg_nii_modelres, unc_nii, softmax_logits, ErrCode.OK
