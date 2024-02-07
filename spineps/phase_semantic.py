# from utils.predictor import nnUNetPredictor
from time import perf_counter

import numpy as np
from TPTBox import NII, Location, Log_Type

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import fill_holes_labels, logger
from spineps.utils.proc_functions import clean_cc_artifacts, n4_bias


def predict_semantic_mask(
    mri_nii: NII,
    model: Segmentation_Model,
    debug_data: dict,
    fill_holes: bool = True,
    clean_artifacts: bool = True,
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
        results = model.segment_scan(
            mri_nii,
            pad_size=0,
            resample_to_recommended=True,
            resample_output_to_input_space=False,
            verbose=verbose,
        )  # type:ignore
        seg_nii = results[OutputType.seg]
        unc_nii = results[OutputType.unc] if OutputType.unc in results else None
        softmax_logits = results[OutputType.softmax_logits]

        if len(seg_nii.unique()) == 0:
            logger.print("Subregion mask is empty, skip this", Log_Type.FAIL)
            return seg_nii, unc_nii, softmax_logits, ErrCode.EMPTY
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

    return seg_nii, unc_nii, softmax_logits, ErrCode.OK
