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
    debug_data: dict,  # noqa: ARG001
    fill_holes: bool = True,
    clean_artifacts: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, NII | None, np.ndarray | None, ErrCode]:
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
        unc_nii = results.get(OutputType.unc, None)
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

    seg_nii = semantic_bounding_box_clean(seg_nii=seg_nii.copy())

    return seg_nii, unc_nii, softmax_logits, ErrCode.OK


def semantic_bounding_box_clean(seg_nii: NII):
    ori = seg_nii.orientation
    seg_binary = seg_nii.reorient_().extract_label(list(seg_nii.unique()))  # whole thing binary
    seg_bin_largest_k_cc_nii = seg_binary.get_largest_k_segmentation_connected_components(
        k=20, labels=1, connectivity=3, return_original_labels=False
    )
    max_k = int(seg_bin_largest_k_cc_nii.max())
    if max_k > 3:
        logger.print(f"Found {max_k} unique connected components in semantic mask", Log_Type.STRANGE)
    # PIR
    largest_nii = seg_bin_largest_k_cc_nii.extract_label(1)
    # width fixed, and heigh include all connected components within bounding box, then repeat
    p_slice, i_slice, r_slice = largest_nii.compute_crop(dist=5)
    # PIR -> fixed, extendable, extendable
    incorporated = [1]
    changed = True
    while changed:
        changed = False
        for k in [l for l in range(2, max_k + 1) if l not in incorporated]:
            k_nii = seg_bin_largest_k_cc_nii.extract_label(k)
            p, i, r = k_nii.compute_crop(dist=3)
            i_slice_compare = slice(
                max(i_slice.start - 10, 0), i_slice.stop + 10
            )  # more margin in inferior direction (allows for gaps in spine)
            if overlap_slice(p_slice, p) and overlap_slice(i_slice_compare, i) and overlap_slice(r_slice, r):
                # extend bbox
                i_slice = slice(min(i_slice.start, i.start), max(i_slice.stop, i.stop))
                r_slice = slice(min(r_slice.start, r.start), max(r_slice.stop, r.stop))
                incorporated.append(k)
                changed = True

    seg_bin_arr = seg_binary.get_seg_array()
    crop = (p_slice, i_slice, r_slice)
    seg_bin_clean_arr = np.zeros(seg_bin_arr.shape)
    seg_bin_clean_arr[crop] = 1

    seg_arr = seg_nii.get_seg_array()
    # logger.print(seg_nii.volumes())
    seg_arr[seg_bin_clean_arr != 1] = 0
    seg_nii.set_array_(seg_arr)
    seg_nii.reorient_(ori)
    cleaned_ks = [l for l in range(2, max_k + 1) if l not in incorporated]
    if len(cleaned_ks) > 0:
        logger.print("semantic_bounding_box_clean", f"got rid of connected components k={cleaned_ks}")
    else:
        logger.print("semantic_bounding_box_clean", "did not remove anything")
    return seg_nii


def overlap_slice(slice1: slice, slice2: slice):
    """checks if two ranges defined by slices overlapping (including border!)

    Args:
        slice1 (slice): _description_
        slice2 (slice): _description_
    """
    slice1s = slice1.start
    slice1e = slice1.stop

    slice2s = slice2.start
    slice2e = slice2.stop

    if slice1s in (slice2s, slice2e) or slice1e in (slice2s, slice2e):
        return True

    if slice2s > slice1s and slice2s <= slice1e:
        return True

    if slice2s < slice1s and slice2e >= slice1s:
        return True
    return False
