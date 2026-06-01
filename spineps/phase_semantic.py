"""Semantic phase: predict and post-process the subregion (semantic) segmentation mask of the spine."""

from __future__ import annotations

# from utils.predictor import nnUNetPredictor
import numpy as np
from TPTBox import NII, Location, Log_Type

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import fill_holes_labels, logger
from spineps.utils.proc_functions import clean_cc_artifacts
from spineps.utils.resolution import REFERENCE_VOXEL_VOLUME_MM3, REFERENCE_ZOOM, mm3_to_voxels, mm_to_voxels

# Connected-component artifacts smaller than this physical volume are removed from the semantic mask.
SMALL_CC_SIZE_THRESHOLD_MM3 = 30 * REFERENCE_VOXEL_VOLUME_MM3
# Vertical (inferior) extent in millimeters kept around the spinal canal.
CANAL_HEIGHT_MARGIN_MM = 64
# Semantic label of S1, i.e. the sacrum.
SACRUM_LABEL = 26
# More connected components than this in the semantic mask is unexpected and gets logged.
MAX_EXPECTED_SEMANTIC_CC = 3
# Physical margin used when cropping around connected components in the bounding-box clean step.
CC_BBOX_MARGIN_MM = 4 * min(REFERENCE_ZOOM)


def predict_semantic_mask(
    mri_nii: NII,
    model: Segmentation_Model,
    debug_data: dict,
    proc_fill_3d_holes: bool = True,
    proc_clean_beyond_largest_bounding_box: bool = True,
    proc_remove_inferior_beyond_canal: bool = False,
    proc_clean_small_cc_artifacts: bool = True,
    verbose: bool = False,
) -> tuple[NII | None, NII | None, ErrCode]:
    """Predict the semantic (subregion) segmentation mask and run post-processing on it.

    Runs the model on the input MRI (resampling to the model's recommended zoom), then optionally removes
    structures beyond the spinal-canal height, cleans small connected-component artifacts, restricts the mask
    to the largest bounding box of connected components, and fills 3D holes.

    Args:
        mri_nii (NII): Input grayscale MRI image (intensities must start at 0).
        model (Segmentation_Model): Model used to produce the semantic segmentation.
        debug_data (dict): Dictionary for collecting intermediate results (e.g. the raw segmentation).
        proc_fill_3d_holes (bool, optional): Whether to fill 3D holes in the output mask. Defaults to True.
        proc_clean_beyond_largest_bounding_box (bool, optional): Whether to keep only connected components within
            the largest bounding box. Defaults to True.
        proc_remove_inferior_beyond_canal (bool, optional): Whether to remove non-sacrum structures below the
            spinal-canal height. Defaults to False.
        proc_clean_small_cc_artifacts (bool, optional): Whether to delete small connected-component artifacts. Defaults to True.
        verbose (bool, optional): Emit additional progress logging. Defaults to False.

    Returns:
        tuple[NII | None, NII | None, ErrCode]: The post-processed semantic mask, the softmax logits, and an
        error code (``ErrCode.OK`` on success, ``ErrCode.EMPTY`` if the predicted mask is empty).
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
        # unc_nii = results.get(OutputType.unc, None)
        softmax_logits = results[OutputType.softmax_logits]

        logger.print("Post-process semantic mask...")

        debug_data["sem_raw"] = seg_nii.copy()

        if seg_nii.is_empty:
            logger.print("Subregion mask is empty, skip this", Log_Type.FAIL)
            return seg_nii, softmax_logits, ErrCode.EMPTY

        if proc_remove_inferior_beyond_canal:
            seg_nii = remove_nonsacrum_beyond_canal_height(seg_nii=seg_nii.copy())

        if proc_clean_small_cc_artifacts:
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
                    cc_size_threshold=mm3_to_voxels(SMALL_CC_SIZE_THRESHOLD_MM3, seg_nii.zoom),
                ),
                verbose=verbose,
            )

        # Do two iterations of both processing if enabled to make sure
        if proc_remove_inferior_beyond_canal:
            seg_nii = remove_nonsacrum_beyond_canal_height(seg_nii=seg_nii.copy())

        if proc_clean_beyond_largest_bounding_box:
            seg_nii = semantic_bounding_box_clean(seg_nii=seg_nii.copy())

        if proc_remove_inferior_beyond_canal and proc_clean_beyond_largest_bounding_box:
            seg_nii = remove_nonsacrum_beyond_canal_height(seg_nii=seg_nii.copy())
            seg_nii = semantic_bounding_box_clean(seg_nii=seg_nii.copy())

        if proc_fill_3d_holes:
            seg_nii = seg_nii.fill_holes_(fill_holes_labels, verbose=logger)
            # seg_fh = seg_nii.extract_label(fill_holes_labels, keep_label=True)
            # seg_fh = seg_fh.fill_holes_global_with_majority_voting(verbose=logger)
            # seg_nii[seg_nii == 0] = seg_fh[seg_nii == 0]

    return seg_nii, softmax_logits, ErrCode.OK


def remove_nonsacrum_beyond_canal_height(seg_nii: NII) -> NII:
    """Remove non-sacrum labels that lie above or below the spinal-canal extent.

    Computes the inferior-axis (I) extent of the spinal canal/cord, expanded by ``CANAL_HEIGHT_MARGIN_MM``,
    and zeroes out everything outside that range. The sacrum (``SACRUM_LABEL``) is kept regardless of position.
    If no canal/cord is present, the mask is returned unchanged.

    Args:
        seg_nii (NII): Semantic segmentation mask in ("P", "I", "R") orientation.

    Returns:
        NII: The segmentation mask with structures beyond the canal height removed (sacrum preserved).

    Raises:
        AssertionError: If ``seg_nii`` is not in ("P", "I", "R") orientation.
    """
    seg_nii.assert_affine(orientation=("P", "I", "R"))
    canal_nii = seg_nii.extract_label([Location.Spinal_Canal.value, Location.Spinal_Cord.value])
    if canal_nii.sum() == 0:
        return seg_nii
    crop_i = canal_nii.compute_crop(dist=CANAL_HEIGHT_MARGIN_MM / seg_nii.zoom[1])[1]
    seg_arr = seg_nii.get_seg_array()
    sacrum_arr = seg_nii.extract_label(SACRUM_LABEL).get_seg_array()
    seg_arr[:, 0 : crop_i.start, :] = 0
    seg_arr[:, crop_i.stop :, :] = 0
    seg_arr[sacrum_arr == 1] = SACRUM_LABEL
    return seg_nii.set_array_(seg_arr)


def semantic_bounding_box_clean(seg_nii: NII) -> NII:
    """Keep only connected components that fall within the spine's growing bounding box.

    Binarizes the mask and labels its connected components. Starting from the largest component's bounding box
    (expanded by ``CC_BBOX_MARGIN_MM``), it iteratively merges in any other component whose bounding box overlaps
    the current region in all three axes (with extra inferior margin to tolerate gaps in the spine). Voxels
    outside the resulting region, and any non-incorporated components, are removed. Components are dropped if the
    binary mask has more than ``MAX_EXPECTED_SEMANTIC_CC`` parts (logged as strange).

    Args:
        seg_nii (NII): Semantic segmentation mask to clean.

    Returns:
        NII: The cleaned segmentation mask, restored to its original orientation.
    """
    ori = seg_nii.orientation
    seg_binary = seg_nii.reorient_().extract_label(list(seg_nii.unique()))  # whole thing binary
    # Resolution-aware bounding-box margin (mm -> voxels at the current zoom).
    bbox_margin_dist = CC_BBOX_MARGIN_MM / min(seg_nii.zoom)
    bbox_margin_vox = mm_to_voxels(CC_BBOX_MARGIN_MM, seg_nii.zoom)
    seg_bin_largest_k_cc_nii: NII = seg_binary.filter_connected_components(
        max_count_component=None, labels=1, connectivity=3, keep_label=False
    )
    max_k = int(seg_bin_largest_k_cc_nii.max())
    if max_k > MAX_EXPECTED_SEMANTIC_CC:
        logger.print(f"Found {max_k} unique connected components in semantic mask", Log_Type.STRANGE)
    # PIR
    largest_nii = seg_bin_largest_k_cc_nii.extract_label(1)
    # width fixed, and heigh include all connected components within bounding box, then repeat
    p_slice, i_slice, r_slice = largest_nii.compute_crop(dist=bbox_margin_dist)
    bboxes = [(p_slice, i_slice, r_slice)]

    # PIR -> fixed, extendable, extendable
    incorporated = [1]
    changed = True
    while changed:
        changed = False
        for k in [l for l in range(2, max_k + 1) if l not in incorporated]:
            k_nii = seg_bin_largest_k_cc_nii.extract_label(k)
            p, i, r = k_nii.compute_crop(dist=bbox_margin_dist)

            for bbox in bboxes:
                i_slice_compare = slice(
                    max(bbox[1].start - bbox_margin_vox, 0), bbox[1].stop + bbox_margin_vox
                )  # more margin in inferior direction (allows for gaps in spine)
                if overlap_slice(bbox[0], p) and overlap_slice(i_slice_compare, i) and overlap_slice(bbox[2], r):
                    # extend bbox
                    bboxes.append((p, i, r))
                    incorporated.append(k)
                    changed = True
                    break

    seg_bin_arr = seg_binary.get_seg_array()
    crop = (p_slice, i_slice, r_slice)
    seg_bin_clean_arr = np.zeros(seg_bin_arr.shape)
    seg_bin_clean_arr[crop] = 1

    # everything below biggest k get always removed
    largest_k_arr = seg_bin_largest_k_cc_nii.get_seg_array()
    seg_bin_clean_arr[largest_k_arr == 0] = 0

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


def overlap_slice(slice1: slice, slice2: slice) -> bool:
    """Check whether two ranges defined by slices overlap (borders inclusive).

    Args:
        slice1 (slice): First range, using its ``start`` and ``stop`` bounds.
        slice2 (slice): Second range, using its ``start`` and ``stop`` bounds.

    Returns:
        bool: True if the two ranges overlap or touch at a border, else False.
    """
    slice1s = slice1.start
    slice1e = slice1.stop

    slice2s = slice2.start
    slice2e = slice2.stop

    if slice1s in (slice2s, slice2e) or slice1e in (slice2s, slice2e):
        return True

    if slice2s > slice1s and slice2s <= slice1e:
        return True

    return bool(slice2s < slice1s and slice2e >= slice1s)
