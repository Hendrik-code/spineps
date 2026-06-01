"""Segmentation post-processing helpers: n4 bias correction, connected-component cleaning and instance fixes."""

from __future__ import annotations

import cc3d
import numpy as np
from scipy.ndimage import center_of_mass
from TPTBox import NII, Location, Logger_Interface
from TPTBox.core.np_utils import (
    np_bbox_binary,
    np_connected_components_per_label,
    np_count_nonzero,
    np_dilate_msk,
    np_unique,
    np_unique_withoutzero,
    np_volume,
)
from tqdm import tqdm

# Vertebra instance labels span 1..25 (cervical, thoracic and lumbar); 26 denotes the sacrum.
MAX_VERTEBRA_INSTANCE_LABEL = 25


def n4_bias(
    nii: NII,
    threshold: int = 60,
    spline_param: int = 100,
    dtype2nii: bool = False,
    norm: int = -1,
):
    """Apply N4 bias field correction to a NIfTI image.

    Builds a foreground mask by thresholding (and filling its bounding box), runs N4 correction restricted to
    that mask, optionally rescales the result to a target maximum and optionally casts back to the input dtype.

    Args:
        nii (NII): Input image to correct.
        threshold (int, optional): Intensity threshold for the foreground mask; voxels below it are excluded.
            Defaults to 60.
        spline_param (int, optional): Spline distance parameter passed to the N4 correction. Defaults to 100.
        dtype2nii (bool, optional): If True, cast the corrected image back to the input image's dtype. Defaults
            to False.
        norm (int, optional): If not -1, rescale the corrected image so its maximum equals this value. Defaults
            to -1.

    Returns:
        tuple[NII, NII]: The bias-corrected image and the binary foreground mask used for correction.
    """
    from ants.utils.convert_nibabel import from_nibabel  # they keep renaming that thing. (version 0.4.2)

    # print("n4 bias", nii.dtype)
    mask = nii.get_array()
    mask[mask < threshold] = 0
    mask[mask != 0] = 1
    slices = np_bbox_binary(mask)
    mask[slices] = 1
    mask_nii = nii.set_array(mask)
    mask_nii.seg = True
    n4: NII = nii.n4_bias_field_correction(threshold=0, mask=from_nibabel(mask_nii.nii), spline_param=spline_param)
    if norm != -1:
        n4 *= norm / n4.max()
    if dtype2nii:
        n4.set_dtype_(nii.dtype)
    return n4, mask_nii


def clean_cc_artifacts(
    mask: NII | np.ndarray,
    logger: Logger_Interface,
    labels: list[int] = [1, 2, 3],  # noqa: B006
    cc_size_threshold: int | list[int] = 100,
    neighbor_factor_2_delete: float = 0.1,
    verbose: bool = True,
    only_delete: bool = False,
    ignore_missing_labels: bool = False,
) -> np.ndarray:
    """Clean small connected-component artifacts in a segmentation mask.

    For each requested label, finds connected components below the size threshold and either deletes them or, if
    they border enough other foreground voxels, relabels them by majority vote of their dilated neighborhood.

    Args:
        mask (NII | np.ndarray): Input segmentation mask.
        logger (Logger_Interface): Logger for progress and cleaning reports.
        labels (list[int], optional): Labels to analyze. Defaults to [1, 2, 3].
        cc_size_threshold (int | list[int], optional): Minimum component size in voxels; a single value applies to
            all labels, or one value per label. Defaults to 100.
        neighbor_factor_2_delete (float, optional): Fraction of neighboring foreground voxels below which a
            component is deleted instead of relabeled. Defaults to 0.1.
        verbose (bool, optional): If True, log per-component details and show a progress bar. Defaults to True.
        only_delete (bool, optional): If True, delete every analyzed component without majority-vote relabeling.
            Defaults to False.
        ignore_missing_labels (bool, optional): If True, skip labels not present instead of asserting. Defaults to
            False.

    Returns:
        np.ndarray: The cleaned segmentation array.

    Raises:
        AssertionError: If requested labels are missing (when ``ignore_missing_labels`` is False) or the length of
            ``cc_size_threshold`` does not match the number of labels.
    """
    mask_arr = mask.get_seg_array() if isinstance(mask, NII) else mask.copy()
    result_arr = mask_arr.copy()

    mask_labels = np_unique(result_arr)
    if 0 not in mask_labels:
        logger.print("No zero in mask? Skip cleaning")
        return mask_arr

    if not ignore_missing_labels:
        assert np.all([l in mask_labels for l in labels]), (
            f"specified labels not found in mask, got labels {labels} and mask labels {mask_labels}"
        )
    else:
        labelsnew = []
        sizes = []
        for idx, l in enumerate(labels):
            if l in mask_labels:
                labelsnew.append(l)
                sizes.append(cc_size_threshold[idx] if isinstance(cc_size_threshold, list) else cc_size_threshold)
        labels = labelsnew
        cc_size_threshold = sizes

    if not isinstance(cc_size_threshold, list):
        cc_size_threshold = [cc_size_threshold for i in range(len(labels))]
    assert len(cc_size_threshold) == len(labels), (
        f"cc_size_threshold size does not match number of given labels to clean, got {len(labels)} and {len(cc_size_threshold)}. Specifiy only an int for cc_size_threshold to use it for all labels"
    )

    subreg_cc, subreg_cc_stats = connected_components_3d(result_arr, connectivity=1)

    cc_to_clean = {}
    for lidx, label in enumerate(tqdm(labels, desc=f"{logger._get_logger_prefix()} cleaning...", disable=not verbose)):
        # print(l, subreg_cc_stats[l]["voxel_counts"])
        idx = [i for i, v in enumerate(subreg_cc_stats[label]["voxel_counts"]) if v < cc_size_threshold[lidx] and v > 0]
        if len(idx) > 0:
            cc_to_clean[label] = idx

        for cc_idx in idx:
            # extract cc label
            mask_cc = subreg_cc[label]
            mask_cc_l = mask_cc.copy()
            mask_cc_l[mask_cc_l != cc_idx] = 0
            log_string = ""
            if verbose:
                cc_volume = np_count_nonzero(mask_cc_l)
                cc_centroid = center_of_mass(mask_cc_l)
                cc_centroid = [int(c) + 1 for c in cc_centroid]  # type: ignore
                log_string = f"Label {label}, cc{cc_idx}, at {cc_centroid}, volume {cc_volume}: "
            if only_delete:
                logger.print(log_string + "deleted") if verbose else None
                # dilated mask nothing in original mask, just delete it
                result_arr[mask_cc_l != 0] = 0
                continue
            dilated_m = np_dilate_msk(mask_cc_l, n_pixel=1)
            dilated_m[mask_cc_l != 0] = 0
            neighbor_voxel_count = np_count_nonzero(dilated_m)
            # print(subreg_cc_stats[label])

            mult = mask_arr * dilated_m
            if np_count_nonzero(mult) <= int(neighbor_voxel_count * neighbor_factor_2_delete):
                logger.print(log_string + "deleted") if verbose else None
                # dilated mask nothing in original mask, just delete it
                result_arr[mask_cc_l != 0] = 0
            else:
                # majority voting
                dilated_m[dilated_m != 0] = 1
                mult = mask_arr * dilated_m
                volumes = np_volume(mult)
                nlabels = list(volumes.keys())
                volumes_values = list(volumes.values())
                newlabel = nlabels[np.argmax(volumes_values)]  # type: ignore
                result_arr[mask_cc_l != 0] = newlabel
                logger.print(log_string + f"labeled as {newlabel}") if verbose else None
                # print(labels, count)
    n_to_clean = {k: len(v) for k, v in cc_to_clean.items()}
    # By clearning: look at surrounding neighbor pixels. If too few, remove cc. otherwise, do majority voting
    if len(n_to_clean) != 0:
        logger.print(f"Cleaned (label, n_components) {n_to_clean}")
    return result_arr


def connected_components_3d(mask_image: np.ndarray, connectivity: int = 3, verbose: bool = False) -> tuple[dict, dict]:  # noqa: ARG001
    """Compute 3D connected components per label together with their statistics.

    Args:
        mask_image (np.ndarray): Input (multi-label) mask.
        connectivity (int, optional): Voxel connectivity in range [1, 3]. For 2D images 2 and 3 are equivalent.
            Defaults to 3.
        verbose (bool, optional): Currently unused. Defaults to False.

    Returns:
        tuple[dict, dict]: A dict mapping each label to its connected-component array, and a dict mapping each
        label to its ``cc3d`` component statistics.
    """
    subreg_cc = np_connected_components_per_label(
        mask_image,
        connectivity=connectivity,
    )
    subreg_cc_stats = {k: cc3d.statistics(v) for k, v in subreg_cc.items()}
    return subreg_cc, subreg_cc_stats


def fix_wrong_posterior_instance_label(seg_sem: NII, seg_inst: NII, logger) -> NII:
    """Reassign misattributed posterior vertebra fragments to the correct instance label.

    For every vertebra instance that splits into multiple connected components, each extra component consisting
    only of posterior elements (arcus vertebrae and/or spinous process) is relabeled to the single neighboring
    instance it touches, if any. Operates on copies and restores the original orientation before returning.

    Args:
        seg_sem (NII): Semantic segmentation (subregion labels) aligned with ``seg_inst``.
        seg_inst (NII): Vertebra instance segmentation to correct.
        logger: Logger used to report each relabeling decision.

    Returns:
        NII: The corrected instance segmentation in the original orientation.

    Raises:
        AssertionError: If ``seg_sem`` and ``seg_inst`` do not share the same affine.
    """
    seg_sem = seg_sem.copy()
    seg_inst = seg_inst.copy()
    orientation = seg_sem.orientation
    seg_sem.assert_affine(other=seg_inst)
    seg_sem.reorient_()
    seg_inst.reorient_()

    seg_inst_arr_proc = seg_inst.get_seg_array()

    instance_labels = [i for i in seg_inst.unique() if 1 <= i <= MAX_VERTEBRA_INSTANCE_LABEL]

    for vert in instance_labels:
        inst_vert = seg_inst.extract_label(vert)
        # sem_vert = seg_sem.apply_mask(inst_vert)

        # Check if multiple CC exist
        inst_vert_cc: NII = inst_vert.filter_connected_components(max_count_component=3, keep_label=False)
        inst_vert_cc_n = int(inst_vert_cc.max())
        if inst_vert_cc_n == 1:
            continue
        #
        # inst_vert_cc is labeled 1 to 3
        for i in range(2, inst_vert_cc_n + 1):
            inst_vert_cc_i = inst_vert_cc.extract_label(i)

            crop = inst_vert_cc_i.compute_crop(dist=1)
            inst_vert_cc_i_c = inst_vert_cc_i.apply_crop(crop)

            cc_sem_vert = seg_sem.apply_crop(crop).apply_mask(inst_vert_cc_i_c)
            # cc_vert is semantic mask of only that cc of instance

            cc_sem_vert_labels = cc_sem_vert.unique()
            # is that cc only arcus and spinosus?
            if len(cc_sem_vert_labels) <= 2 and np.all(
                [i in [Location.Arcus_Vertebrae.value, Location.Spinosus_Process.value] for i in cc_sem_vert_labels]
            ):
                # neighbor that have non arcus/spinosus label?
                neighbor_instance_labels = seg_inst.apply_crop(crop).get_seg_array()
                neighbor_instance_labels[inst_vert_cc_i_c.get_seg_array() == 1] = 0
                neighbor_instance_labels = np_unique_withoutzero(neighbor_instance_labels)
                # which instance labels does it touch
                logger.print(f"vert {vert}, cc_k {i} has instance neighbors {neighbor_instance_labels}")
                # is it touching only one other instance label?
                if len(neighbor_instance_labels) == 1 and neighbor_instance_labels[0] != vert:
                    to_label = neighbor_instance_labels[0]
                    logger.print(f"vert {vert}, cc_k {i} relabel to instance {to_label}")
                    seg_inst_arr_proc[inst_vert_cc_i.get_seg_array() == 1] = to_label

    seg_inst_proc = seg_inst.set_array(seg_inst_arr_proc).reorient_(orientation)
    return seg_inst_proc
