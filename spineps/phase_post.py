# from utils.predictor import nnUNetPredictor
import heapq

import numpy as np
from scipy.ndimage import center_of_mass
from TPTBox import NII, Location, Log_Type, v_idx2name, v_name2idx
from TPTBox.core.np_utils import (
    np_bbox_binary,
    np_center_of_mass,
    np_connected_components,
    np_connected_components_per_label,
    np_contacts,
    np_count_nonzero,
    np_dilate_msk,
    np_extract_label,
    np_map_labels,
    np_unique,
    np_unique_withoutzero,
    np_volume,
)

from spineps.phase_labeling import VertLabelingClassifier, perform_labeling_step
from spineps.seg_pipeline import logger, vertebra_subreg_labels
from spineps.utils.proc_functions import fix_wrong_posterior_instance_label


def phase_postprocess_combined(
    img_nii: NII,
    seg_nii: NII,
    vert_nii: NII,
    model_labeling: VertLabelingClassifier | None,
    debug_data: dict | None,
    labeling_offset: int = 0,
    proc_assign_missing_cc: bool = True,
    proc_clean_inst_by_sem: bool = True,
    n_vert_bodies: int | None = None,
    process_merge_vertebra: bool = True,
    proc_vertebra_inconsistency: bool = True,
    proc_assign_posterior_instance_label: bool = True,
    verbose: bool = False,
) -> tuple[NII, NII]:
    logger.print("Post process", Log_Type.STAGE)
    with logger:
        img_nii.assert_affine(other=seg_nii)
        seg_nii.assert_affine(other=vert_nii)
        # Post process semantic mask
        ###################
        vert_nii = vert_nii.copy()
        if n_vert_bodies is None:
            n_vert_bodies = len(vert_nii.unique())
        if debug_data is None:
            debug_data = {}

        if proc_clean_inst_by_sem:
            vert_nii.apply_mask(seg_nii, inplace=True)
        crop_slices = seg_nii.compute_crop(dist=2)

        # Save uncropped to uncrop later
        vert_uncropped = vert_nii.copy()
        seg_uncropped = seg_nii.copy()

        # Crop down
        img_nii.apply_crop_(crop_slices)
        vert_nii.apply_crop_(crop_slices)
        seg_nii.apply_crop_(crop_slices)

        # Post processing both
        ###################
        whole_vert_nii_cleaned, seg_nii_cleaned = mask_cleaning_other(
            whole_vert_nii=vert_nii,
            seg_nii=seg_nii,
            n_vert_bodies=n_vert_bodies,
            proc_assign_missing_cc=proc_assign_missing_cc,
            verbose=verbose,
        )

        if process_merge_vertebra and Location.Vertebra_Disc.value in seg_nii_cleaned.unique():
            detect_and_solve_merged_vertebra(seg_nii_cleaned, whole_vert_nii_cleaned)

        if proc_assign_posterior_instance_label:
            whole_vert_nii_cleaned = fix_wrong_posterior_instance_label(seg_nii_cleaned, seg_inst=whole_vert_nii_cleaned, logger=logger)

        if proc_vertebra_inconsistency:
            # Assigns superior/inferior based on instance label overlap
            assign_vertebra_inconsistency(seg_nii_cleaned, whole_vert_nii_cleaned)

        # Label vertebra top -> down
        whole_vert_nii_cleaned, vert_labels = label_instance_top_to_bottom(whole_vert_nii_cleaned, labeling_offset=labeling_offset)
        logger.print(f"Labeled {len(vert_labels)} vertebra instances from top to bottom")
        if model_labeling is not None:
            whole_vert_nii_cleaned = perform_labeling_step(
                model=model_labeling,
                img_nii=img_nii,
                vert_nii=whole_vert_nii_cleaned,
                subreg_nii=seg_nii_cleaned,
            )

        logger.print("vert_nii", whole_vert_nii_cleaned.unique(), whole_vert_nii_cleaned.volumes())
        logger.print("seg_nii", seg_nii_cleaned.unique())

        whole_vert_nii_cleaned[seg_nii_cleaned.get_seg_array() == v_name2idx["S1"]] = v_name2idx["S1"]
        vert_arr_cleaned, seg_arr_cleaned = add_ivd_ep_vert_label(whole_vert_nii_cleaned, seg_nii_cleaned)
        #
        #
        cur_segarr = seg_nii_cleaned.get_seg_array()
        cur_segarr[cur_segarr == Location.Endplate.value] = seg_arr_cleaned[cur_segarr == Location.Endplate.value]
        seg_nii_cleaned.set_array_(cur_segarr)
        ###############
        # Uncrop

        vert_uncropped[crop_slices] = vert_arr_cleaned
        seg_uncropped[crop_slices] = seg_nii_cleaned.get_seg_array()

        logger.print("vert_uncropped", vert_uncropped.unique(), vert_uncropped.volumes())
        logger.print("seg_uncropped", seg_uncropped.unique())

        # subreg_nii_cleaned = vert_nii_cleaned.set_array(subreg_arr_cleaned, verbose=False)
        logger.print(
            "Vertebra whole_vert_nii_uncropped_backsampled",
            vert_uncropped.zoom,
            vert_uncropped.orientation,
            vert_uncropped.shape,
            verbose=verbose,
        )
        debug_data["vert_arr_return_final"] = vert_uncropped.copy()
    return seg_uncropped, vert_uncropped


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
            verbose=verbose,
        )
        subreg_vert_nii.set_array_(subreg_vert_arr)
        vert_arr_cleaned[subreg_vert_arr == 0] = 0
        subreg_arr[deletion_map == 1] = 0

    n_vert_pixels = np_count_nonzero(vert_arr_cleaned)
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

    label_rest = np_unique(subreg_arr_vert_rest)
    if len(label_rest) == 1 and label_rest[0] == 0:
        logger.print("No CC had to be assigned", Log_Type.OK, verbose=verbose)
        return target_arr, reference_arr, deletion_map
    # subreg_arr_vert_rest is not hit pixels bei vertebra prediction
    subreg_cc = np_connected_components_per_label(subreg_arr_vert_rest, connectivity=2)
    loop_counts = 0
    # for label, for each cc
    for label, subreg_cc_map in subreg_cc.items():
        if label == 0:
            continue
        cc_labels = np_unique_withoutzero(subreg_cc_map)
        loop_counts += len(cc_labels)
        # print(cc_labels)
        for cc_l in cc_labels:
            cc_map = subreg_cc_map.copy()
            cc_map[cc_map != cc_l] = 0
            cc_bbox = np_bbox_binary(cc_map, px_dist=2)
            vert_arr_c = target_arr.copy()[cc_bbox]
            cc_map_c = cc_map[cc_bbox]
            cc_map_c[cc_map_c != 0] = 1
            # print("cc_map_c\n", cc_map_c)
            # print("vert_arr_c\n", vert_arr_c)
            cc_map_dilated = np_dilate_msk(cc_map_c, 1, n_pixel=1, connectivity=2)
            # cc_map_dilated[vert_arr_c == 0] = 0
            # print("cc_map_dilated\n", cc_map_dilated)
            # majority voting
            mult = vert_arr_c * cc_map_dilated
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


def add_ivd_ep_vert_label(whole_vert_nii: NII, seg_nii: NII, verbose=True):
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
        try:
            coms_vert_dict[l] = np_center_of_mass(vert_l)[1][1]  # center_of_mass(vert_l)[1]
        except Exception:
            coms_vert_dict[l] = 0

    coms_vert_y = list(coms_vert_dict.values())
    coms_vert_labels = list(coms_vert_dict.keys())

    n_ivds = 0
    n_ivd_unique = 0
    if Location.Vertebra_Disc.value in seg_t.unique():
        # Map IVDS
        subreg_cc = seg_t.get_connected_components(labels=Location.Vertebra_Disc.value)
        subreg_cc_n = len(subreg_cc.unique())
        subreg_cc = subreg_cc.get_seg_array()
        cc_labelset = list(range(1, subreg_cc_n + 1))
        mapping_cc_to_vert_label = {}

        coms_ivd_dict = {}
        for c in cc_labelset:
            if c == 0:
                continue
            com_y = np_center_of_mass(subreg_cc == c)[1][1]  # center_of_mass(c_l)[1]

            if com_y < min(coms_vert_y):
                label = min(coms_vert_labels) - 1
            else:
                nearest_lower = find_nearest_lower(coms_vert_y, com_y)
                label = next(i for i in coms_vert_dict if coms_vert_dict[i] == nearest_lower)
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
        n_ivd_unique = len(np.unique(to_mapped_labels))
        subreg_ivd = np_map_labels(subreg_ivd, label_map=mapping_cc_to_vert_label)
        subreg_ivd += 100
        subreg_ivd[subreg_ivd == 100] = 0
        vert_arr[subreg_arr == Location.Vertebra_Disc.value] = subreg_ivd[subreg_arr == Location.Vertebra_Disc.value]
    n_eps = 0
    n_eps_unique = 0
    if Location.Endplate.value in seg_t.unique():
        # MAP Endplate
        ep_cc = seg_t.get_connected_components(labels=Location.Endplate.value)
        ep_cc_n = len(ep_cc.unique())
        ep_cc = ep_cc.get_seg_array()
        cc_ep_labelset = list(range(1, ep_cc_n + 1))
        mapping_ep_cc_to_vert_label = {}
        coms_ivd_dict = {}
        for c in cc_ep_labelset:
            if c == 0:
                continue
            com_y = np_center_of_mass(ep_cc == c)[1][1]  # center_of_mass(c_l)[1]
            nearest_lower = find_nearest_lower(coms_vert_y, com_y)
            label = next(i for i in coms_vert_dict if coms_vert_dict[i] == nearest_lower)
            mapping_ep_cc_to_vert_label[c] = label
            n_eps += 1

        subreg_ep = ep_cc.copy()
        n_eps_unique = len(np.unique(list(mapping_ep_cc_to_vert_label.values())))
        subreg_ep = np_map_labels(subreg_ep, label_map=mapping_ep_cc_to_vert_label)
        subreg_ep += 200
        subreg_ep[subreg_ep == 200] = 0
        vert_arr[subreg_arr == Location.Endplate.value] = subreg_ep[subreg_arr == Location.Endplate.value]
        vert_t.set_array_(vert_arr)

        # divide into upper and lower endplate
        out = seg_t * 0
        pref = 1
        old_vol = -1
        for dil in range(1, 15):
            curr = out.extract_label([Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value])
            new_vol = curr.sum()
            total = seg_t.extract_label(Location.Endplate.value).sum()
            logger.print(rf"{new_vol/total*100:.2f}% endplates detected", end="\r") if verbose else None
            if old_vol == new_vol and old_vol != 0:
                break
            old_vol = new_vol
            if total == new_vol:
                logger.print("Found all Endplates                                      ")
                break
            for i in vert_t.unique():
                if i >= 40:
                    break
                curr = out.extract_label([Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value])
                v = vert_t.extract_label(i).dilate_msk(dil, verbose=False)
                end = seg_t.extract_label(Location.Endplate.value) * v
                end *= -curr + 1  # type: ignore
                plates = vert_t * end
                plates.map_labels_(
                    {
                        i + 200: Location.Vertebral_Body_Endplate_Inferior.value,
                        pref + 200: Location.Vertebral_Body_Endplate_Superior.value,
                    },
                    verbose=False,
                )
                out += plates
                pref = i
        curr = out.extract_label([Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value])
        end = seg_t.extract_label(Location.Endplate.value)
        end *= -curr + 1
        # end += end.dilate_msk(3)
        out += end * Location.Endplate.value
        seg_t = out.extract_label(
            [Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value, Location.Endplate.value]
        )

    logger.print(f"Labeled {n_ivds} IVDs ({n_ivd_unique} unique), and {n_eps} Endplates ({n_eps_unique} unique)")
    return vert_t.set_array_(vert_arr).reorient_(orientation).get_seg_array(), seg_t.reorient_(orientation).get_seg_array()


def find_nearest_lower(seq, x):
    values_lower = [item for item in seq if item < x]
    if len(values_lower) == 0:
        return min(seq)
    return max(values_lower)


def label_instance_top_to_bottom(vert_nii: NII, labeling_offset: int = 0):
    ori = vert_nii.orientation
    vert_nii.reorient_()
    vert_arr = vert_nii.get_seg_array()
    com_i = np_center_of_mass(vert_arr)
    comb_l = list(zip(com_i.keys(), com_i.values(), strict=True))
    comb_l.sort(key=lambda a: a[1][1])  # PIR
    com_map = {comb_l[idx][0]: idx + 1 + labeling_offset for idx in range(len(comb_l))}

    vert_nii.map_labels_(com_map, verbose=False)
    vert_nii.reorient_(ori)
    return vert_nii, vert_nii.unique()


def assign_vertebra_inconsistency(seg_nii: NII, vert_nii: NII):
    seg_nii.assert_affine(shape=vert_nii.shape)
    seg_arr = seg_nii.get_seg_array()
    vert_arr = vert_nii.get_seg_array()

    # assign inconsistent substructures
    for loc in [
        Location.Superior_Articular_Left,
        Location.Superior_Articular_Right,
        Location.Inferior_Articular_Left,
        Location.Inferior_Articular_Right,
    ]:
        value = loc.value

        subreg_l = np_extract_label(seg_arr, value, inplace=False)  # type:ignore
        try:
            subreg_cc, _ = np_connected_components(subreg_l, label_ref=1)
        except AssertionError as e:
            print(f"Got error {e}, skip")
            break
        subreg_cc = subreg_cc[1]
        cc_labels = np_unique(subreg_cc)

        for ccl in cc_labels:
            if ccl == 0:
                continue
            cc_map = np_extract_label(subreg_cc, ccl, inplace=False)
            vert_arr_cc = vert_arr.copy()
            vert_arr_cc += 1
            vert_arr_cc[cc_map == 0] = 0
            gt_volume = np_volume(vert_arr_cc)
            k_keys_sorted = heapq.nlargest(2, gt_volume, key=gt_volume.__getitem__)

            if len(k_keys_sorted) == 1:
                continue
            biggest_volume = (k_keys_sorted[0], gt_volume[k_keys_sorted[0]])
            second_volume = (k_keys_sorted[1], gt_volume[k_keys_sorted[1]])

            # print(biggest_volume, second_volume)

            if biggest_volume[1] * 0.50 > second_volume[1]:
                to_label = biggest_volume[0] - 1  # int(list(gt_volume.keys())[argmax] - 1)

                vert_arr[cc_map == 1] = to_label
                logger.print(
                    f"set cc to {to_label}, with volume decision {gt_volume}, based on {biggest_volume}, {second_volume}", verbose=False
                )

        vert_nii.set_array_(vert_arr)


def detect_and_solve_merged_vertebra(seg_nii: NII, vert_nii: NII):
    seg_sem = seg_nii.map_labels({Location.Endplate.value: Location.Vertebra_Disc.value}, verbose=False)
    # get all ivd CCs from seg_sem

    stats = {}
    # Map IVDS
    subreg_cc: NII = seg_sem.get_connected_components(labels=Location.Vertebra_Disc.value)
    subreg_cc += 100

    coms = subreg_cc.center_of_masses()
    volumes = subreg_cc.volumes()
    stats = {i: (g[1], True, volumes[i]) for i, g in coms.items()}

    vert_coms = vert_nii.center_of_masses()
    vert_volumes = vert_nii.volumes()

    for i, g in vert_coms.items():
        stats[i] = (g[1], False, vert_volumes[i])

    stats_by_height = dict(sorted(stats.items(), key=lambda x: x[1][0]))
    stats_by_height_keys = list(stats_by_height.keys())

    # detect C2 split into two components
    first_key, second_key = stats_by_height_keys[0], stats_by_height_keys[1]
    first_stats, second_stats = stats_by_height[first_key], stats_by_height[second_key]
    if first_stats[1] is False and second_stats[1] is False:  # noqa: SIM102
        # both vertebra
        if first_stats[2] < 0.5 * second_stats[2]:
            # first is significantly smaller than second and they are close in height
            # how many pixels are touching
            vert_firsttwo_arr = vert_nii.extract_label(first_key).get_seg_array()
            vert_firsttwo_arr2 = vert_nii.extract_label(second_key).get_seg_array()
            vert_firsttwo_arr += vert_firsttwo_arr2 + 1
            contacts = np_contacts(vert_firsttwo_arr, connectivity=3)
            if contacts[(1, 2)] > 20:
                logger.print("Found first two instance weird, will merge", Log_Type.STRANGE)
                vert_nii.map_labels_({first_key: second_key}, verbose=False)

    return seg_nii, vert_nii
