# from utils.predictor import nnUNetPredictor
import numpy as np
from scipy.ndimage import center_of_mass
from TPTBox import NII, Location, Log_Type, v_idx2name, v_name2idx
from TPTBox.core.np_utils import np_approx_center_of_mass, np_bbox_nd, np_connected_components, np_dilate_msk, np_map_labels

from spineps.seg_pipeline import logger, vertebra_subreg_labels


def phase_postprocess_combined(
    seg_nii: NII,
    vert_nii: NII,
    debug_data: dict | None,
    labeling_offset: int = 0,
    proc_assign_missing_cc: bool = True,
    n_vert_bodies: int | None = None,
    verbose: bool = False,
) -> tuple[NII, NII]:
    logger.print("Post process", Log_Type.STAGE)
    with logger:
        seg_nii.assert_affine(shape=vert_nii.shape)
        # Post process semantic mask
        ###################
        seg_nii = semantic_bounding_box_clean(seg_nii=seg_nii.copy())
        ###################
        vert_nii = vert_nii.copy()
        if n_vert_bodies is None:
            n_vert_bodies = len(vert_nii.unique())
        if debug_data is None:
            debug_data = {}
            #
        vert_nii.apply_mask(seg_nii, inplace=True)
        crop_slices = seg_nii.compute_crop_slice(dist=3)
        vert_uncropped_arr = np.zeros(vert_nii.shape, dtype=seg_nii.dtype)
        seg_uncropped_arr = np.zeros(vert_nii.shape, dtype=seg_nii.dtype)

        # Crop down
        vert_nii.apply_crop_slice_(crop_slices)
        seg_nii.apply_crop_slice_(crop_slices)

        # Post processing both
        ###################
        whole_vert_nii_cleaned, seg_nii_cleaned = mask_cleaning_other(
            whole_vert_nii=vert_nii,
            seg_nii=seg_nii,
            n_vert_bodies=n_vert_bodies,
            proc_assign_missing_cc=proc_assign_missing_cc,
            verbose=verbose,
        )

        # Label vertebra top -> down
        whole_vert_nii_cleaned, vert_labels = label_instance_top_to_bottom(whole_vert_nii_cleaned)
        if labeling_offset != 0:
            whole_vert_nii_cleaned.map_labels_({i: i + 1 for i in vert_labels if i != 0}, verbose=verbose)
        logger.print(f"Labeled {len(vert_labels)} vertebra instances from top to bottom")

        vert_arr_cleaned = add_ivd_ep_vert_label(whole_vert_nii_cleaned, seg_nii_cleaned)
        vert_arr_cleaned[seg_nii_cleaned.get_seg_array() == v_name2idx["S1"]] = v_name2idx["S1"]
        ###############
        # Uncrop

        vert_uncropped_arr[crop_slices] = vert_arr_cleaned
        whole_vert_nii_cleaned.set_array_(vert_uncropped_arr, verbose=False)
        #
        seg_uncropped_arr[crop_slices] = seg_nii_cleaned.get_seg_array()

        seg_nii_cleaned.set_array_(seg_uncropped_arr, verbose=False)
        #
        debug_data["vert_arr_crop_e_addivd"] = whole_vert_nii_cleaned.copy()

        # subreg_nii_cleaned = vert_nii_cleaned.set_array(subreg_arr_cleaned, verbose=False)
        logger.print(
            "Vertebra whole_vert_nii_uncropped_backsampled",
            whole_vert_nii_cleaned.zoom,
            whole_vert_nii_cleaned.orientation,
            whole_vert_nii_cleaned.shape,
            verbose=verbose,
        )
        debug_data["vert_arr_return_final"] = whole_vert_nii_cleaned.copy()
    return seg_nii_cleaned, whole_vert_nii_cleaned


def mask_cleaning_other(
    whole_vert_nii: NII,
    seg_nii: NII,
    n_vert_bodies: int,
    proc_assign_missing_cc: bool = False,
    verbose: bool = False,
) -> tuple[NII, NII]:
    # make copy where both masks clean each other
    vert_arr_cleaned = whole_vert_nii.get_seg_array()
    subreg_vert_nii = seg_nii.extract_label(vertebra_subreg_labels)
    subreg_vert_arr = subreg_vert_nii.get_seg_array()
    # if dilation_fill:
    #    vert_arr_cleaned = np_dilate_msk(vert_arr_cleaned, label_ref=vert_labels, mm=5)  # , mask=subreg_vert_arr
    subreg_arr = seg_nii.get_seg_array()

    if proc_assign_missing_cc:
        vert_arr_cleaned, subreg_vert_arr, deletion_map = assign_missing_cc(
            vert_arr_cleaned,
            subreg_vert_arr,
            verbose=False,
        )
        subreg_vert_nii.set_array_(subreg_vert_arr)
        vert_arr_cleaned[subreg_vert_arr == 0] = 0
        subreg_arr[deletion_map == 1] = 0

    n_vert_pixels = np.count_nonzero(vert_arr_cleaned)
    n_subreg_vert_pixels = subreg_vert_nii.volumes()[1]
    n_vert_pixel_per_vertebra = n_subreg_vert_pixels / n_vert_bodies
    n_difference_pixels = n_subreg_vert_pixels - n_vert_pixels
    if n_difference_pixels > 0:
        logger.print("n_subreg_vert_px - n_vert_px", n_subreg_vert_pixels - n_vert_pixels, Log_Type.STRANGE)

    n_vert_pixels_rel_diff = round((n_subreg_vert_pixels - n_vert_pixels) / n_vert_pixel_per_vertebra, 3)
    if n_vert_pixels_rel_diff < 0:
        if proc_assign_missing_cc:
            assert n_vert_pixels_rel_diff >= 0, "Less subreg vertebra pixels than in vert mask cannot happen with assign_missing_cc"
        else:
            logger.print(
                f"A volume of {n_vert_pixels_rel_diff} * avg_vertebra_volume in vertebra not matched in semantic mask, set proc_assign_missing_cc=TRUE to circumvent this",
                Log_Type.WARNING,
            )
    elif n_vert_pixels_rel_diff > 0.5:
        logger.print(f"A volume of {n_vert_pixels_rel_diff} * avg_vertebra_volume in subreg not matched by vertebra mask", Log_Type.WARNING)

    return whole_vert_nii.set_array(vert_arr_cleaned), seg_nii.set_array(subreg_arr)


def assign_missing_cc(
    target_arr: np.ndarray,
    reference_arr: np.ndarray,
    verbose: bool = False,
    verbose_deletion: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pipeline: target = vert, reference = subregion
    assert target_arr.shape == reference_arr.shape
    subreg_arr_vert_rest = reference_arr.copy()
    subreg_arr_vert_rest[target_arr != 0] = 0
    deletion_map = np.zeros(reference_arr.shape)

    label_rest = np.unique(subreg_arr_vert_rest)
    if len(label_rest) == 1 and label_rest[0] == 0:
        logger.print("No CC had to be assigned", Log_Type.OK, verbose=verbose)
        return target_arr, reference_arr, deletion_map
    # subreg_arr_vert_rest is not hit pixels bei vertebra prediction
    subreg_cc, _ = np_connected_components(subreg_arr_vert_rest, connectivity=2)
    loop_counts = 0
    # for label, for each cc
    for label, subreg_cc_map in subreg_cc.items():
        if label == 0:
            continue
        cc_labels = np.unique(subreg_cc_map)[1:]
        loop_counts += len(cc_labels)
        # print(cc_labels)
        for cc_l in cc_labels:
            cc_map = subreg_cc_map.copy()
            cc_map[cc_map != cc_l] = 0
            cc_bbox = np_bbox_nd(cc_map, px_dist=2)
            vert_arr_c = target_arr.copy()[cc_bbox]
            cc_map_c = cc_map[cc_bbox]
            cc_map_c[cc_map_c != 0] = 1
            # print("cc_map_c\n", cc_map_c)
            # print("vert_arr_c\n", vert_arr_c)
            cc_map_dilated = np_dilate_msk(cc_map_c, 1, mm=1, connectivity=2)
            # cc_map_dilated[vert_arr_c == 0] = 0
            # print("cc_map_dilated\n", cc_map_dilated)
            # majority voting
            # print("vert_arr_c", np.unique(vert_arr_c))
            mult = vert_arr_c * cc_map_dilated
            # print("mult", np.unique(mult), "\n", mult)
            labels, count = np.unique(mult, return_counts=True)
            labels = labels[1:]
            count = count[1:]
            if len(labels) > 0:
                newlabel = labels[np.argmax(count)]
                logger.print(f"Assign {label, cc_l} to {newlabel}, Location at {cc_bbox}", verbose=verbose)
                vert_arr_c[cc_map_c != 0] = newlabel
                target_arr[cc_bbox] = vert_arr_c
            else:
                logger.print(f"Assign {label, cc_l} to EMPTY, Location at {cc_bbox}", verbose=verbose or verbose_deletion)
                reference_arr[cc_bbox][cc_map_c == 1] = 0
                deletion_map[cc_bbox][cc_map_c == 1] = 1
            # print("vert_arr\n", vert_arr)
            # print()
    logger.print(f"Assign missing cc: Processed {loop_counts} missed ccs")

    return target_arr, reference_arr, deletion_map


def add_ivd_ep_vert_label(whole_vert_nii: NII, seg_nii: NII):
    # PIR
    orientation = whole_vert_nii.orientation
    vert_t = whole_vert_nii.reorient()
    seg_t = seg_nii.reorient()
    vert_labels = vert_t.unique()  # without zero
    vert_arr = vert_t.get_seg_array()

    subreg_arr = seg_t.get_seg_array()

    coms_vert_dict = {}
    for l in vert_labels:
        vert_l = vert_arr.copy()
        vert_l[vert_l != l] = 0
        vert_l[subreg_arr != 49] = 0  # com of corpus region
        vert_l[vert_l != 0] = 1
        if np.count_nonzero(vert_l) > 0:
            coms_vert_dict[l] = np_approx_center_of_mass(vert_l, label_ref=1)[1][1]  # center_of_mass(vert_l)[1]
        else:
            coms_vert_dict[l] = 0

    coms_vert_y = list(coms_vert_dict.values())
    coms_vert_labels = list(coms_vert_dict.keys())

    n_ivds = 0
    if Location.Vertebra_Disc.value in seg_t.unique():
        # Map IVDS
        subreg_cc, _ = seg_t.get_segmentation_connected_components(labels=Location.Vertebra_Disc.value)
        subreg_cc = subreg_cc[Location.Vertebra_Disc.value]
        cc_labelset = np.unique(subreg_cc)
        mapping_cc_to_vert_label = {}

        coms_ivd_dict = {}
        for c in cc_labelset:
            if c == 0:
                continue
            c_l = subreg_cc.copy()
            c_l[c_l != c] = 0
            com_y = np_approx_center_of_mass(c_l, label_ref=c)[c][1]  # center_of_mass(c_l)[1]

            if com_y < min(coms_vert_y):
                label = min(coms_vert_labels) - 1
            else:
                nearest_lower = find_nearest_lower(coms_vert_y, com_y)
                label = [i for i in coms_vert_dict if coms_vert_dict[i] == nearest_lower][0]
            coms_ivd_dict[label] = com_y
            mapping_cc_to_vert_label[c] = label
            n_ivds += 1

        # find which vert got how many ivd CCs
        to_mapped_labels = list(mapping_cc_to_vert_label.values())
        for l in vert_labels:
            if l not in to_mapped_labels:
                logger.print(f"Vertebra {v_idx2name[l]} got no IVD component assigned", Log_Type.STRANGE)
            count = to_mapped_labels.count(l)
            if count > 1:
                logger.print(f"Vertebra {v_idx2name[l]} got {count} IVD components assigned", Log_Type.STRANGE)

        subreg_ivd = subreg_cc.copy()
        subreg_ivd = np_map_labels(subreg_ivd, label_map=mapping_cc_to_vert_label)
        subreg_ivd += 100
        subreg_ivd[subreg_ivd == 100] = 0
        vert_arr[subreg_arr == Location.Vertebra_Disc.value] = subreg_ivd[subreg_arr == Location.Vertebra_Disc.value]

    n_eps = 0
    if Location.Endplate.value in seg_t.unique():
        # MAP Endplate
        ep_cc, _ = seg_t.get_segmentation_connected_components(labels=Location.Endplate.value)
        ep_cc = ep_cc[Location.Endplate.value]
        cc_ep_labelset = np.unique(ep_cc)
        mapping_ep_cc_to_vert_label = {}
        coms_ivd_dict = {}
        for c in cc_ep_labelset:
            if c == 0:
                continue
            c_l = ep_cc.copy()
            c_l[c_l != c] = 0
            com_y = np_approx_center_of_mass(c_l, label_ref=c)[c][1]  # center_of_mass(c_l)[1]
            nearest_lower = find_nearest_lower(coms_vert_y, com_y)
            label = [i for i in coms_vert_dict if coms_vert_dict[i] == nearest_lower][0]
            mapping_ep_cc_to_vert_label[c] = label
            n_eps += 1

        subreg_ep = ep_cc.copy()
        subreg_ep = np_map_labels(subreg_ep, label_map=mapping_ep_cc_to_vert_label)
        subreg_ep += 200
        subreg_ep[subreg_ep == 200] = 0
        vert_arr[subreg_arr == Location.Endplate.value] = subreg_ep[subreg_arr == Location.Endplate.value]

    logger.print(f"Labeled {n_ivds} IVDs, and {n_eps} Endplates")
    return vert_t.set_array_(vert_arr).reorient_(orientation).get_seg_array()


def find_nearest_lower(seq, x):
    values_lower = [item for item in seq if item < x]
    if len(values_lower) == 0:
        return min(seq)
    return max(values_lower)


def semantic_bounding_box_clean(seg_nii: NII):
    ori = seg_nii.orientation
    seg_binary = seg_nii.reorient_().extract_label(list(seg_nii.unique()))  # whole thing binary
    seg_bin_largest_k_cc_nii = seg_binary.get_largest_k_segmentation_connected_components(
        k=20, labels=1, connectivity=3, return_original_labels=False
    )
    max_k = seg_bin_largest_k_cc_nii.max()
    if max_k > 3:
        logger.print(f"Found {max_k} unique connected components in semantic mask", Log_Type.STRANGE)
    # PIR
    largest_nii = seg_bin_largest_k_cc_nii.extract_label(1)
    # width fixed, and heigh include all connected components within bounding box, then repeat
    P_slice, I_slice, R_slice = largest_nii.compute_crop_slice(dist=5)
    # PIR -> fixed, extendable, extendable
    incorporated = [1]
    changed = True
    while changed:
        changed = False
        for k in [l for l in range(2, max_k + 1) if l not in incorporated]:
            k_nii = seg_bin_largest_k_cc_nii.extract_label(k)
            p, i, r = k_nii.compute_crop_slice(dist=3)
            I_slice_compare = slice(
                max(I_slice.start - 10, 0), I_slice.stop + 10
            )  # more margin in inferior direction (allows for gaps in spine)
            if overlap_slice(P_slice, p) and overlap_slice(I_slice_compare, i) and overlap_slice(R_slice, r):
                # extend bbox
                I_slice = slice(min(I_slice.start, i.start), max(I_slice.stop, i.stop))
                R_slice = slice(min(R_slice.start, r.start), max(R_slice.stop, r.stop))
                incorporated.append(k)
                changed = True

    seg_bin_arr = seg_binary.get_seg_array()
    crop = (P_slice, I_slice, R_slice)
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


def label_instance_top_to_bottom(vert_nii: NII):
    ori = vert_nii.orientation
    vert_nii.reorient_()
    present_labels = list(vert_nii.unique())
    vert_arr = vert_nii.get_seg_array()
    com_i = np_approx_center_of_mass(vert_arr, present_labels)
    # Old, more precise version (but takes longer)
    # comb = {}
    # for i in present_labels:
    #    arr_i = vert_arr.copy()
    #    arr_i[arr_i != i] = 0
    #    comb[i] = center_of_mass(arr_i)
    comb_l = list(zip(com_i.keys(), com_i.values()))
    comb_l.sort(key=lambda a: a[1][1])  # PIR
    com_map = {comb_l[idx][0]: idx + 1 for idx in range(len(comb_l))}

    vert_nii.map_labels_(com_map, verbose=False)
    vert_nii.reorient_(ori)
    return vert_nii, vert_nii.unique()


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

    if slice1s == slice2s or slice1s == slice2e or slice1e == slice2s or slice1e == slice2e:
        return True

    if slice2s > slice1s and slice2s <= slice1e:
        return True

    if slice2s < slice1s and slice2e >= slice1s:
        return True
    return False


if __name__ == "__main__":
    print(overlap_slice(slice(1, 10), slice(3, 8)))
    print(overlap_slice(slice(1, 8), slice(8, 10)))
    print(overlap_slice(slice(1, 7), slice(8, 10)))
    #
    print(overlap_slice(slice(3, 8), slice(1, 10)))
    #
    print(overlap_slice(slice(3, 8), slice(3, 10)))
