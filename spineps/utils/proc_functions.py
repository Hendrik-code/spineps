import cc3d
import numpy as np
from ants.utils.convert_nibabel import from_nibabel
from scipy.ndimage import center_of_mass
from TPTBox import NII, Logger_Interface
from TPTBox.core.np_utils import np_bbox_binary, np_count_nonzero, np_dilate_msk, np_unique, np_volume
from tqdm import tqdm


def n4_bias(
    nii: NII,
    threshold: int = 60,
    spline_param: int = 100,
    dtype2nii: bool = False,
    norm: int = -1,
):
    """Applies n4 bias field correction to a nifty

    Args:
        nii (NII): Input nifty
        threshold (int, optional): Threshold to use for masking, every input value < threshold is used. Defaults to 60.
        spline_param (int, optional): _description_. Defaults to 200.
        dtype2nii (bool, optional): _description_. Defaults to False.
        norm (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
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
    """Cleans artifacts based on connected components analysis

    Args:
        mask (NII | np.ndarray): input segmentation mask
        logger (Logger_Interface): logger
        labels (list[int], optional): labels to analyze in the input. Defaults to [1, 2, 3].
        cc_size_threshold (int | list[int], optional): threshold on which to clean, can be a number for all labels or a list of values for each different label. Defaults to 100.
        neighbor_factor_2_delete (float, optional): Percentage of existing neighbor pixels to not just delete the CC. Defaults to 0.1.
        verbose (bool, optional): _description_. Defaults to True.
        only_delete (bool, optional): If set, will delete each analyse CC. Defaults to False.
        ignore_missing_labels (bool, optional): If true, will not crash if some labels are not found. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    mask_arr = mask.get_seg_array() if isinstance(mask, NII) else mask.copy()
    result_arr = mask_arr.copy()

    mask_labels = np_unique(result_arr)
    if 0 not in mask_labels:
        logger.print("No zero in mask? Skip cleaning")
        return mask_arr

    if not ignore_missing_labels:
        assert np.all(
            [l in mask_labels for l in labels]
        ), f"specified labels not found in mask, got labels {labels} and mask labels {mask_labels}"
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
    assert (
        len(cc_size_threshold) == len(labels)
    ), f"cc_size_threshold size does not match number of given labels to clean, got {len(labels)} and {len(cc_size_threshold)}. Specifiy only an int for cc_size_threshold to use it for all labels"

    subreg_cc, subreg_cc_stats = connected_components_3d(result_arr, connectivity=1)

    cc_to_clean = {}
    for lidx, label in enumerate(tqdm(labels, desc=f"{logger._get_logger_prefix()} cleaning...", disable=not verbose)):
        # print(l, subreg_cc_stats[l]["voxel_counts"])
        idx = [i for i, v in enumerate(subreg_cc_stats[label]["voxel_counts"]) if v < cc_size_threshold[lidx]]
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
            #
            dilated_m = np_dilate_msk(mask_cc_l, mm=1)
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


def connected_components_3d(mask_image: np.ndarray, connectivity: int = 3, verbose: bool = False) -> tuple[dict, dict]:
    """Applies 3d connected components

    Args:
        mask_image: input mask
        connectivity: in range [1,3]. For 2D images, 2 and 3 is the same.
        verbose:

    Returns:

    """
    assert 2 <= mask_image.ndim <= 3, f"expected 2D or 3D, but got {mask_image.ndim}"
    assert 1 <= connectivity <= 3, f"expected connectivity in [1,3], but got {connectivity}"
    if mask_image.ndim == 2:  # noqa: SIM108
        connectivity = min(connectivity * 2, 8)  # 1:4, 2:8, 3:8
    else:
        connectivity = 6 if connectivity == 1 else 18 if connectivity == 2 else 26

    subreg_cc = {}
    subreg_cc_stats = {}
    regions = np_unique(mask_image)[1:]
    for subreg in regions:
        img_subreg = mask_image.copy()
        img_subreg[img_subreg != subreg] = 0
        # labels_out = cc3d.dust(img_subreg, threshold=400, in_place=False)
        labels_out, n = cc3d.connected_components(img_subreg, connectivity=connectivity, return_N=True)
        # labels_out, N = cc3d.largest_k(img_subreg, k=10, return_N=True)
        subreg_cc[subreg] = labels_out
        subreg_cc_stats[subreg] = cc3d.statistics(labels_out)
        if (n) != 1:  # zero is a label
            print(f"subreg {subreg} does not have one CC (not counting zeros), got {n}") if verbose else None
    return subreg_cc, subreg_cc_stats
