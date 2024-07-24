# from utils.predictor import nnUNetPredictor
import numpy as np
from TPTBox import NII, Location, Log_Type
from TPTBox.core.np_utils import (
    np_calc_crop_around_centerpoint,
    np_center_of_mass,
    np_connected_components,
    np_count_nonzero,
    np_dice,
    np_dilate_msk,
    np_erode_msk,
    np_extract_label,
    np_get_largest_k_connected_components,
    np_unique,
    np_volume,
)
from tqdm import tqdm

from spineps.seg_enums import ErrCode, OutputType
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import logger
from spineps.utils.proc_functions import clean_cc_artifacts


def predict_instance_mask(
    seg_nii: NII,
    model: Segmentation_Model,
    debug_data: dict,
    pad_size: int = 0,
    proc_inst_fill_3d_holes: bool = True,
    proc_detect_and_solve_merged_corpi: bool = True,
    proc_corpus_clean: bool = True,
    proc_inst_clean_small_cc_artifacts: bool = True,
    proc_inst_largest_k_cc: int = 0,
    verbose: bool = False,
) -> tuple[NII | None, ErrCode]:
    """Based on subregion segmentation, feeds individual arcus coms to a network to get the vertebra body segmentations

    Args:
        seg_nii (NII): _description_
        cutout_size (tuple[int, int, int], optional): _description_. Defaults to (128, 88, 32).

    Returns:
        tuple[NII | None, ErrCode]: whole_vert_nii, errcode
    """
    logger.print("Predict instance mask", Log_Type.STAGE)
    with logger:
        # Fixed constants for this approach
        cutout_size = (248, 304, 64)  # (264, 304, 64)  # (248, 304, 64)  # (264, 304, 64)
        corpus_size_cleaning = 100 * 2.42  # voxel threshold * nako resolution
        corpus_border_threshold = 10
        vert_size_threshold = 250 * 2.42

        logger.print("Vertebra input", seg_nii.zoom, seg_nii.orientation, seg_nii.shape, verbose=verbose)
        # Save values for resample back later
        # orientation = seg_nii.orientation
        # shp = seg_nii.shape

        seg_nii_rdy = seg_nii.reorient(verbose=logger)
        debug_data["inst_uncropped_Subreg_nii_a_PIR"] = seg_nii_rdy.copy()

        # Padding?
        if pad_size > 0:
            # logger.print(seg_nii_rdy.shape)
            arr = seg_nii_rdy.get_array()
            arr = np.pad(arr, pad_size, mode="edge")
            seg_nii_rdy.set_array_(arr)
            # logger.print(seg_nii_rdy.shape)

        zms = seg_nii_rdy.zoom
        logger.print("zms", zms, verbose=verbose)
        expected_zms = model.calc_recommended_resampling_zoom(seg_nii_rdy.zoom)
        if not seg_nii_rdy.assert_affine(zoom=expected_zms, raise_error=False):
            seg_nii_rdy.rescale_(expected_zms, verbose=logger)  # in PIR
        seg_nii_uncropped = seg_nii_rdy.copy()
        logger.print(
            "Vertebra seg_nii_uncropped", seg_nii_uncropped.zoom, seg_nii_uncropped.orientation, seg_nii_uncropped.shape, verbose=verbose
        )
        debug_data["inst_uncropped_Subreg_nii_b_zms"] = seg_nii_uncropped.copy()
        uncropped_vert_mask = np.zeros(seg_nii_uncropped.shape, dtype=seg_nii_uncropped.dtype)
        logger.print("Vertebra uncropped_vert_mask empty", uncropped_vert_mask.shape, verbose=verbose)
        crop = seg_nii_rdy.compute_crop(dist=5)
        # logger.print("Crop", crop, verbose=verbose)
        seg_nii_rdy.apply_crop_(crop)
        logger.print(f"Crop down from {uncropped_vert_mask.shape} to {seg_nii_rdy.shape}", verbose=verbose)
        # arr[crop] = X, then set nifty to arr
        logger.print("Vertebra seg_nii_rdy", seg_nii_rdy.zoom, seg_nii_rdy.orientation, seg_nii_rdy.shape, verbose=verbose)
        debug_data["inst_cropped_Subreg_nii_b"] = seg_nii_rdy.copy()
        #
        # make threshold in actual mm
        corpus_border_threshold = int(corpus_border_threshold / expected_zms[1])
        corpus_size_cleaning = max(int(corpus_size_cleaning / (expected_zms[0] * expected_zms[1] * expected_zms[2])), 40)
        vert_size_threshold = max(int(vert_size_threshold / (expected_zms[0] * expected_zms[1] * expected_zms[2])), 40)

        seg_labels = seg_nii.unique()
        if 49 not in seg_labels:
            logger.print(f"no corpus ({Location.Vertebra_Corpus_border.value}) labels in this segmentation, cannot proceed", Log_Type.FAIL)
            return None, ErrCode.EMPTY

        # get all the 3vert predictions
        vert_predictions, hierarchical_existing_predictions, n_corpus_coms = collect_vertebra_predictions(
            seg_nii=seg_nii_rdy,
            model=model,
            corpus_size_cleaning=corpus_size_cleaning if proc_corpus_clean else 0,
            cutout_size=cutout_size,
            debug_data=debug_data,
            process_detect_and_solve_merged_corpi=proc_detect_and_solve_merged_corpi,
            proc_inst_largest_k_cc=proc_inst_largest_k_cc,
            proc_inst_fill_holes=False,
            verbose=verbose,
        )
        if vert_predictions is None:
            return None, ErrCode.UNKNOWN  # type:ignore
        logger.print("Vertebra Predictions done!", verbose=verbose)

        # debug_data["vert_predictions"] = vert_predictions
        whole_vert_nii, debug_data, errcode = from_vert3_predictions_make_vert_mask(
            seg_nii_rdy,
            vert_predictions,
            hierarchical_existing_predictions,
            vert_size_threshold,
            debug_data=debug_data,
            proc_inst_clean_small_cc_artifacts=proc_inst_clean_small_cc_artifacts,
        )
        del vert_predictions, hierarchical_existing_predictions
        if errcode != ErrCode.OK:
            return None, errcode
        logger.print("Merged predictions into vert mask")

        logger.print(
            "Vertebra whole_vert_nii_cropped", whole_vert_nii.zoom, whole_vert_nii.orientation, whole_vert_nii.shape, verbose=verbose
        )

        uniq_labels = whole_vert_nii.unique()

        if proc_inst_fill_3d_holes:
            whole_vert_nii.fill_holes_(verbose=logger)
        debug_data["inst_cropped_vert_arr_c_proc"] = whole_vert_nii.copy()
        n_vert_bodies = len(uniq_labels)
        logger.print(f"Predicted {n_vert_bodies} vertebrae")
        if n_vert_bodies < n_corpus_coms:
            logger.print(f"Number of vertebra {n_vert_bodies} smaller than number of corpus regions {n_corpus_coms}", Log_Type.WARNING)

        # label continously
        labelmap = {l: i + 1 for i, l in enumerate(uniq_labels)}
        whole_vert_nii.map_labels_(labelmap, verbose=False)

        # clean vertebra mask with subreg mask
        logger.print(
            "Vertebra whole_vert_nii_cropped2", whole_vert_nii.zoom, whole_vert_nii.orientation, whole_vert_nii.shape, verbose=verbose
        )
        vert_nii_cleaned = whole_vert_nii
        uncropped_vert_mask[crop] = vert_nii_cleaned.get_seg_array()
        logger.print(f"Uncrop back from {vert_nii_cleaned.shape} to {uncropped_vert_mask.shape}", verbose=verbose)
        whole_vert_nii_uncropped = seg_nii_uncropped.set_array(uncropped_vert_mask)
        debug_data["inst_uncropped_vert_arr_a"] = whole_vert_nii_uncropped.copy()

        # Uncrop again
        if pad_size > 0:
            # logger.print(whole_vert_nii_uncropped.shape)
            arr = whole_vert_nii_uncropped.get_array()
            arr = arr[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
            whole_vert_nii_uncropped.set_array_(arr)
            # logger.print(whole_vert_nii_uncropped.shape)

    return whole_vert_nii_uncropped, ErrCode.OK


def get_corpus_coms(
    seg_nii: NII,
    corpus_size_cleaning: int,
    process_detect_and_solve_merged_corpi: bool = True,
    verbose: bool = False,
) -> list:
    seg_nii.assert_affine(orientation=["P", "I", "R"])
    #
    # Extract Corpus region and try to find all coms naively (some skips shouldnt matter)
    corpus_nii = seg_nii.extract_label(Location.Vertebra_Corpus_border.value)
    corpus_nii.erode_msk_(mm=2, connectivity=2, verbose=False)
    if 1 in corpus_nii.unique() and corpus_size_cleaning > 0:
        corpus_nii.set_array_(
            clean_cc_artifacts(
                corpus_nii,
                labels=[1],
                cc_size_threshold=corpus_size_cleaning,
                only_delete=True,
                ignore_missing_labels=True,
                logger=logger,
                verbose=verbose,
            ),
            verbose=False,
        )

    if 1 not in corpus_nii.unique():
        logger.print("No 1 in corpus nifty, cannot make vertebra mask", Log_Type.FAIL)
        return None

    if not process_detect_and_solve_merged_corpi:
        corpus_coms = corpus_nii.get_segmentation_connected_components_center_of_mass(label=1, sort_by_axis=1)
        corpus_coms.reverse()  # from bottom to top
        return corpus_coms

    ############
    # Detect merged vertebra and use plane split algo on it
    ############

    # Get corpus CCs
    corpus_cc, corpus_cc_n = corpus_nii.get_segmentation_connected_components(labels=1)
    corpus_cc = corpus_cc[1]
    corpus_cc_n = corpus_cc_n[1]
    logger.print(f"Found {corpus_cc_n} Corpus ccs (naively)", verbose=verbose)

    # Check against ivd order
    seg_sem = seg_nii.map_labels({Location.Endplate.value: Location.Vertebra_Disc.value}, verbose=False)
    subreg_cc, subreg_cc_n = seg_sem.get_segmentation_connected_components(labels=Location.Vertebra_Disc.value)
    subreg_cc = subreg_cc[Location.Vertebra_Disc.value]
    subreg_cc[subreg_cc > 0] += 100
    subreg_cc_n = subreg_cc_n[Location.Vertebra_Disc.value]
    logger.print(f"Found {subreg_cc_n} IVD ccs (naively)", verbose=verbose)
    coms = np_center_of_mass(subreg_cc)
    volumes = np_volume(subreg_cc)
    stats = {i: (g[1], True, volumes[i]) for i, g in coms.items()}

    vert_coms = np_center_of_mass(corpus_cc)
    vert_volumes = np_volume(corpus_cc)

    for i, g in vert_coms.items():
        stats[i] = (g[1], False, vert_volumes[i])

    stats_by_height = dict(sorted(stats.items(), key=lambda x: x[1][0]))
    stats_by_height_keys = list(stats_by_height.keys())

    for vl in stats_by_height_keys:
        idx = stats_by_height_keys.index(vl)
        statsvl = stats_by_height[vl]

        is_ivd = statsvl[1]
        selfheight = statsvl[0]

        nidx = idx + 1
        if nidx < 0 or nidx >= len(stats_by_height_keys):
            continue

        nkey = stats_by_height_keys[nidx]
        if nkey in stats_by_height and stats_by_height[nkey][1] == is_ivd:
            neighbor = stats_by_height[nkey]
            neighborheight = neighbor[0]
            logger.print(
                f"Wrong ivd-vert alternation found in label {vl}, is_ivd = {is_ivd}, neighbor {nkey}",
                Log_Type.STRANGE,
                verbose=verbose,
            )
            # check if same heigh, then just merge ivd label
            if abs(neighborheight - selfheight) < 10:
                logger.print("Same height, just merge")
                stats_by_height.pop(vl)
                stats_by_height = dict(sorted(stats_by_height.items(), key=lambda x: x[1][0]))
                stats_by_height_keys = list(stats_by_height.keys())
                print(stats_by_height_keys)
                continue

            logger.print("Merged corpi, try to fix it", verbose=verbose)
            neighbor_verts = {
                stats_by_height_keys[idx + i]: stats_by_height[stats_by_height_keys[idx + i]]
                for i in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
                if (idx + i) in stats_by_height_keys and stats_by_height_keys[idx + i] < 99
            }
            #    stats_by_height_keys[idx + i]
            #    for i in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
            #    if (idx + i) in stats_by_height_keys and stats_by_height_keys[idx + i] < 99
            # ]  # (+-3)
            logger.print("neighbor_vert_labels", neighbor_verts, verbose=verbose)
            if len(neighbor_verts) == 0:
                logger.print("Got no neighbor vert labels to fix", Log_Type.FAIL)
                continue
            neighbor_volumes = [k[2] for nv, k in neighbor_verts.items()]
            logger.print("neighbor_volumes", neighbor_volumes, verbose=verbose)
            n_neighbors_without_target = len(neighbor_volumes) - 1
            if n_neighbors_without_target > 2:
                argmax = np.argmax(neighbor_volumes)
                n_avg_volume = sum([neighbor_volumes[i] for i in range(len(neighbor_volumes)) if i != argmax]) / n_neighbors_without_target

                diff_volume = neighbor_volumes[argmax] / n_avg_volume
                if diff_volume > 1.5:
                    logger.print(
                        f"Volume difference detected in label {vl}, diff = {diff_volume}, volume = {neighbor_volumes[argmax]}, neighbor_avg = {n_avg_volume}",
                        Log_Type.STRANGE,
                    )

                    target_vert_id = list(neighbor_verts.keys())[argmax]
                    segvert = np_extract_label(corpus_cc, target_vert_id, inplace=False)
                    try:
                        logger.print("get_separating_components to split vertebra", verbose=verbose)
                        (spart, tpart, spart_dil, tpart_dil, stpart) = get_separating_components(
                            segvert,
                            connectivity=3,
                        )

                        logger.print("Splitting by plane")
                        plane_split_nii = get_plane_split(segvert, corpus_nii, spart, tpart, spart_dil, tpart_dil)
                        split_vert = split_by_plane(segvert, plane_split_nii)
                        corpus_cc[split_vert == 2] = corpus_cc.max() + 1
                    except Exception as e:
                        logger.print(f"Separating Corpi failed with exception {e}")

    corpus_coms = list(np_center_of_mass(corpus_cc).values())
    corpus_coms.sort(key=lambda a: a[1])
    corpus_coms.reverse()  # from bottom to top
    logger.print(f"Found {len(corpus_coms)} final Corpus ccs", verbose=verbose)
    return corpus_coms


def get_separating_components(
    segvert: np.ndarray,
    max_iter: int = 10,
    connectivity: int = 3,
):
    check_connectivtiy = 3
    vol = segvert.copy()
    vol_old = vol.copy()
    iterations = 0
    while True:
        vol_erode = np_erode_msk(vol, mm=1, connectivity=connectivity)
        subreg_cc, subreg_cc_n = np_connected_components(vol_erode, connectivity=check_connectivtiy)
        if 1 in subreg_cc_n and subreg_cc_n[1] > 1:
            vol = subreg_cc[1]
            break
        elif 1 not in subreg_cc_n:
            vol_dilated = np_dilate_msk(vol, mm=1, connectivity=connectivity, mask=vol.copy())
            # use iteration before to get other CC
            vol[vol_old != 0] = 2  # all possible voxels are 2
            vol[vol_dilated == 1] = 1

            if 2 not in np_volume(vol):
                raise Exception(  # noqa: TRY002
                    f"cannot split volume into two parts after {iterations} iterations, all values are 0 after erosion."
                )
            volume = np_volume(vol)
            dil_iter = 0
            while volume[1] / (volume[1] + volume[2]) < 0.5:
                vol_dilated = np_dilate_msk(vol_dilated, mm=1, connectivity=connectivity, mask=vol.copy())
                # inst_nii.set_array(vol_dilated).save(files_out + f"subreg_cc_vol_dilated{dil_iter}.nii.gz")
                vol[vol_dilated == 1] = 1
                # inst_nii.set_array(vol).save(files_out + f"subreg_cc_dilation{dil_iter}.nii.gz")
                volume = np_volume(vol)
                if 1 not in volume or 2 not in volume:
                    raise Exception("Could not divide into two instance")  # noqa: TRY002
                dil_iter += 1

            vol_1 = np_get_largest_k_connected_components(vol == 1, k=1, connectivity=check_connectivtiy).astype(np.uint8)
            vol_2 = np_get_largest_k_connected_components(vol == 2, k=1, connectivity=check_connectivtiy).astype(np.uint8)
            vol_2 *= 2
            vol[vol == 1] = vol_1[vol == 1]
            vol[vol == 2] = vol_2[vol == 2]
            break
        vol_old = vol
        vol = vol_erode
        iterations += 1
        if iterations > max_iter:
            raise Exception(f"Could not divide into two instance after max iterations {max_iter}")  # noqa: TRY002
    if len(np_volume(vol)) != 2:
        logger.print("Get largest two components")
        subreg_cc_2k = np_get_largest_k_connected_components(vol, k=2, connectivity=check_connectivtiy, return_original_labels=False)
        spart = subreg_cc_2k == 1
        tpart = subreg_cc_2k == 2
    else:
        spart = vol == 1
        tpart = vol == 2

    if spart.sum() == 0 or tpart.sum() == 0:
        raise Exception("S or T are empty")  # noqa: TRY002

    spart_dil = np_dilate_msk(spart, mm=1, connectivity=connectivity)
    tpart_dil = np_dilate_msk(tpart, mm=1, connectivity=connectivity)
    stpart = (spart_dil + (tpart_dil * 2)).astype(np.uint8)
    while 3 not in np_volume(stpart):
        spart_dil = np_dilate_msk(spart_dil, mm=1, connectivity=connectivity)
        stpart = (spart_dil + (tpart_dil * 2)).astype(np.uint8)
        if 3 in np_volume(stpart):
            break
        tpart_dil = np_dilate_msk(tpart_dil, mm=1, connectivity=connectivity)
        stpart = (spart_dil + (tpart_dil * 2)).astype(np.uint8)
    stpart = spart_dil + (tpart_dil * 2)
    return spart, tpart, spart_dil, tpart_dil, stpart


def get_plane_split(
    segvert: np.ndarray,
    compare_nii: NII,
    spart: np.ndarray,
    tpart: np.ndarray,
    spart_dil: np.ndarray,
    tpart_dil: np.ndarray,
):
    s_dilint = spart_dil.astype(np.uint8)
    t_dilint = tpart_dil.astype(np.uint8)
    collision_arr = s_dilint + t_dilint
    if 2 not in collision_arr:
        logger.print("S and T dilated do not touch each other error")
        return compare_nii.set_array(np.zeros_like(segvert))
    collision_arr[collision_arr != 2] = 0
    collision_arr[collision_arr == 2] = 1
    collision_point = np_center_of_mass(collision_arr)[1]
    # TODO instead of COM, calc COM of S and T, make vector and use merge point along that vector as collision point? should be more accurate
    #
    normal_vector = np_center_of_mass(spart.astype(np.uint8))[1] - np_center_of_mass(tpart.astype(np.uint8))[1]
    normalized_normal = normal_vector / np.linalg.norm(normal_vector)
    axis = np.argmax(np.abs(normalized_normal))
    dims = [0, 1, 2]
    dims.remove(axis)
    dim1, dim2 = dims
    shift_total = -collision_point.dot(normal_vector)
    xx, yy = np.meshgrid(range(collision_arr.shape[dim1]), range(collision_arr.shape[dim2]))
    zz = (-normal_vector[dim1] * xx - normal_vector[dim2] * yy - shift_total) * 1.0 / normal_vector[axis]
    z_max = collision_arr.shape[axis] - 1
    zz[zz < 0] = 0
    zz[zz > z_max] = z_max
    # make cords to array again
    plane_coords = np.zeros([xx.shape[0], xx.shape[1], 3], dtype=int)
    plane_coords[:, :, axis] = zz
    plane_coords[:, :, dim1] = xx
    plane_coords[:, :, dim2] = yy

    plane = segvert * 0
    plane[plane_coords[:, :, 0], plane_coords[:, :, 1], plane_coords[:, :, 2]] = 1
    plane_nii = compare_nii.set_array(plane)

    plane_filled_nii = plane_nii.copy()
    orientation = plane_filled_nii.orientation
    plane_filled_nii.reorient_(("S", "A", "R"))
    plane_filled_arr = plane_filled_nii.get_array()
    x_slice = np.ones_like(plane_filled_arr[0]) * np.max(plane_filled_arr) + 1
    # plane_filled[:, 0, :] = 2
    # plane_filled[:, -1, :] = 1
    for i in range(plane_filled_arr.shape[0]):
        curr_slice = plane_filled_arr[i]
        cond = np.where(curr_slice != 0)
        x_slice[cond] = np.minimum(curr_slice[cond], x_slice[cond])
        plane_filled_arr[i] = x_slice

    plane_filled_nii.set_array_(plane_filled_arr).reorient_(orientation)
    return plane_filled_nii


def split_by_plane(
    segvert: np.ndarray,
    plane_filled_nii: NII,
):
    plane_filled_arr = plane_filled_nii.get_array()
    # 1 above, 2 below
    plane_filled_arr[segvert == 0] = 0
    segvert = plane_filled_arr
    # segvert[plane_filled_arr == 1] = 1
    # segvert[plane_filled_arr == 2] = 2
    return segvert


def collect_vertebra_predictions(
    seg_nii: NII,
    model: Segmentation_Model,
    corpus_size_cleaning: int,
    cutout_size,
    debug_data: dict,
    proc_inst_largest_k_cc: int = 0,
    process_detect_and_solve_merged_corpi: bool = True,
    proc_inst_fill_holes: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray | None, list[str], int]:
    corpus_coms = get_corpus_coms(
        seg_nii,
        corpus_size_cleaning=corpus_size_cleaning,
        process_detect_and_solve_merged_corpi=process_detect_and_solve_merged_corpi,
        verbose=verbose,
    )
    if corpus_coms is None:
        return None, [], 0
    n_corpus_coms = len(corpus_coms)

    if n_corpus_coms < 3:
        logger.print(f"Too few vertebra semantically segmented ({n_corpus_coms})", Log_Type.FAIL)
        return None, [], 0

    shp = (
        # n_corpus_coms,
        # 3
        seg_nii.shape[0],
        seg_nii.shape[1],
        seg_nii.shape[2],
    )
    hierarchical_existing_predictions = []
    hierarchical_predictions = np.zeros((n_corpus_coms, 3, *shp), dtype=seg_nii.dtype)
    # print("hierarchical_predictions", hierarchical_predictions.shape)
    vert_predict_template = np.zeros(shp, dtype=np.uint16)
    # print("vert_predict_template", vert_predict_template.shape)

    # relabel to the labels expected by the model
    seg_nii_for_cut: NII = seg_nii.copy().map_labels_(
        {
            41: 1,
            42: 2,
            43: 3,
            44: 4,
            45: 5,
            46: 6,
            47: 7,
            48: 8,
            49: 9,
            Location.Spinal_Cord.value: 0,
            Location.Spinal_Canal.value: 0,
            Location.Vertebra_Disc.value: 0,
            Location.Endplate.value: 0,
            26: 0,
        },
        verbose=False,
    )
    # print("seg_nii_for_cut", seg_nii_for_cut.shape)

    logger.print("Vertebra collect in", seg_nii.zoom, seg_nii.orientation, seg_nii.shape, verbose=verbose)

    # iterate over sorted coms and segment vertebra from subreg
    for com_idx, com in enumerate(tqdm(corpus_coms, desc=logger._get_logger_prefix() + " Vertebra Body predictions")):
        seg_arr_c = seg_nii_for_cut.get_seg_array()
        # Shift the com until there is a segmentation there (to account for mishaps in the com calculation)
        seg_at_com = seg_arr_c[int(com[0])][int(com[1])][int(com[2])] != 0
        orig_com = (com[0], com[1], com[2])
        while not seg_at_com:
            com = (com[0], com[1] + 5, com[2])  # noqa: PLW2901
            if com[1] >= shp[1]:
                logger.print("Collect Vertebra Predictions: One Cutout at weird position", Log_Type.FAIL)
                com = orig_com  # noqa: PLW2901
                break
            seg_at_com = seg_arr_c[int(com[0])][int(com[1])][int(com[2])] != 0

        cutout_size2 = cutout_size

        # Calc cutout
        arr_cut, cutout_coords, paddings = np_calc_crop_around_centerpoint(com, seg_arr_c, cutout_size2)
        cut_nii = seg_nii_for_cut.set_array(arr_cut, verbose=False).reorient_()
        debug_data[f"inst_cutout_vert_nii_{com_idx}_cut"] = cut_nii
        # print("cut_nii", cut_nii.shape)
        results = model.segment_scan(
            cut_nii,
            resample_to_recommended=False,
            pad_size=0,
            resample_output_to_input_space=False,
            verbose=False,
        )
        vert_cut_nii = results[OutputType.seg].reorient_()
        # print("vert_cut_nii", vert_cut_nii.shape)
        # logger.print(f"Done {com_idx}")
        debug_data[f"inst_cutout_vert_nii_{com_idx}_pred"] = vert_cut_nii.copy()
        vert_cut_nii = post_process_single_3vert_prediction(
            vert_cut_nii,
            None,
            fill_holes=proc_inst_fill_holes,
            largest_cc=proc_inst_largest_k_cc,  # type:ignore
        )
        vert_labels = vert_cut_nii.unique()  # 1,2,3
        debug_data[f"inst_cutout_vert_nii_{com_idx}_proc"] = vert_cut_nii.copy()

        cutout_sizes = tuple(cutout_coords[i].stop - cutout_coords[i].start for i in range(len(cutout_coords)))
        pad_cutout = tuple(slice(paddings[i][0], paddings[i][0] + cutout_sizes[i]) for i in range(len(paddings)))
        # print("cutout_sizes", cutout_sizes)
        # print("pad_cutout", pad_cutout)
        arr = vert_cut_nii.get_seg_array()
        vert_predict_map = vert_predict_template.copy()
        vert_predict_map[cutout_coords] = arr[pad_cutout]
        # vert_predict_map[com_idx][cutout_coords][vert_predict_map[com_idx][cutout_coords] == 0] = arr[pad_cutout][
        #    vert_predict_map[com_idx][cutout_coords] == 0
        # ]
        seg_at_com = vert_predict_map[int(com[0])][int(com[1])][int(com[2])]
        if seg_at_com == 0:
            logger.print("Zero at cutout center, mistake", Log_Type.WARNING)
        # if seg_at_com != 0:
        #    # before (1) is above
        #    shift = 2 - seg_at_com
        #    vert_predictions[com_idx][vert_predictions[com_idx] != 0] = vert_predictions[com_idx][vert_predictions[com_idx] != 0] + shift
        # debug_data[f"vert_nii_{com_idx}_proc2"] = seg_nii_for_cut.set_array(vert_predictions[com_idx][cutout_coords])
        for l in vert_labels:
            vert_l_map = vert_predict_map.copy()
            vert_l_map[vert_l_map != l] = 0
            vert_l_map[vert_l_map != 0] = 1
            labelindex = l - 1
            if vert_l_map.max() > 0:
                hierarchical_predictions[com_idx][labelindex] = vert_l_map
                hierarchical_existing_predictions.append(str_id_com_label(com_idx, labelindex))
    return hierarchical_predictions, hierarchical_existing_predictions, n_corpus_coms


def post_process_single_3vert_prediction(
    vert_nii: NII,
    labels: list[int] | None = None,
    largest_cc: int = 0,
    fill_holes: bool = False,
):
    if largest_cc > 0:  # 5 seems like a good number (if three, then at least center must be fully visible)
        vert_nii = vert_nii.get_largest_k_segmentation_connected_components(largest_cc, labels, return_original_labels=True)
    if fill_holes:
        labels = vert_nii.unique()  # type:ignore
        vert_nii.fill_holes_(labels=labels, verbose=False)
    return vert_nii


def str_id_com_label(com_idx: int, label: int):
    return str(com_idx) + "_" + str(label)


def from_vert3_predictions_make_vert_mask(
    seg_nii: NII,
    vert_predictions: np.ndarray,  # already hierarchical [com_idx, l, map]
    hierarchical_existing_predictions: list[str],  # list of actually used vert predictions
    vert_size_threshold: int,
    debug_data: dict,
    proc_inst_clean_small_cc_artifacts: bool = True,
    verbose: bool = False,
) -> tuple[NII, dict, ErrCode]:
    # instance approach: each 1/2/3 pred finds it most agreeing partner in surrounding predictions (com idx -2 to +2 all three pred)
    # Then sort by agreement, and segment each (this would be able to add more vertebra than input coms if one is skipped)
    # each one that has been used for fixing a segmentation cannot be used again (so object loose their partners if too weird)

    # idx is always in the order in predictions (so bottom2up corpus CC)
    # arcus_coms sorted bottom to top
    hierarchical_predictions = vert_predictions
    # search space: all neighboring predictions
    # all search for up to two other predictions with best agreement
    coupled_predictions = create_prediction_couples(hierarchical_predictions, hierarchical_existing_predictions)

    logger.print("Coupled predictions", coupled_predictions, verbose=verbose)
    return merge_coupled_predictions(
        seg_nii,
        coupled_predictions=coupled_predictions,
        hierarchical_predictions=hierarchical_predictions,
        debug_data=debug_data,
        proc_clean_small_cc_artifacts=proc_inst_clean_small_cc_artifacts,
        vert_size_threshold=vert_size_threshold,
        verbose=verbose,
    )


def create_prediction_couples(
    hierarchical_predictions: np.ndarray,
    hierarchical_existing_predictions,
    verbose: bool = False,
):
    n_predictions = hierarchical_predictions.shape[0]

    coupled_predictions = {}

    # TODO LANGSAMER!!! multiprocessing Pool verwenden?
    # task = []
    # for idx in range(0, n_predictions):
    #    for pred in range(3):
    #        task.append(
    #            delayed(find_prediction_couple)(
    #                idx, pred, hierarchical_predictions, hierarchical_existing_predictions, n_predictions, verbose
    #            )
    #        )
    # result = Parallel(n_jobs=5)(task)

    # TODO try to calculate list of candidates here, take the predictions and then parallelize the find_prediction_couple

    for idx in range(n_predictions):
        for pred in range(3):
            couple, agreement = find_prediction_couple(
                idx, pred, hierarchical_predictions, hierarchical_existing_predictions, n_predictions, verbose
            )
            if couple is None:
                continue
            if couple not in coupled_predictions:
                coupled_predictions[couple] = [agreement]
            else:
                coupled_predictions[couple].append(agreement)
    # with get_context("spawn").Pool() as pool:
    #    instance_pairs = [
    #        (idx, pred, hierarchical_predictions.copy(), hierarchical_existing_predictions, n_predictions)
    #        for idx in range(0, n_predictions)
    #        for pred in range(3)
    #    ]

    #    result = pool.starmap(find_prediction_couple, instance_pairs)

    # for r in result:
    #    couple = r[0]
    #    agreement = r[1]
    #    if couple is None:
    #        continue
    #    if couple not in coupled_predictions:
    #        coupled_predictions[couple] = [agreement]
    #    else:
    #        coupled_predictions[couple].append(agreement)
    coupled_predictions = {i: sum(v) / len(v) for i, v in coupled_predictions.items()}
    coupled_predictions = dict(
        sorted(
            coupled_predictions.items(),
            key=lambda item: (len(item[0]) + 1) * item[1],
            reverse=True,
        )
    )
    return coupled_predictions


def parallel_dice(anchor, pred, cand_loc):
    return float(np_dice(anchor, pred)), cand_loc


def find_prediction_couple(
    idx,
    pred,
    hierarchical_predictions: np.ndarray,
    hierarchical_existing_predictions,
    n_predictions,
    verbose: bool = False,
):
    if str_id_com_label(idx, pred) not in hierarchical_existing_predictions:
        logger.print(f"{str_id_com_label(idx, pred)} not in predictions {hierarchical_existing_predictions}", verbose=verbose)
        return None, 0
    anchor = hierarchical_predictions[idx][pred]
    dices = {}

    min_idx = max(0, idx - 2)
    max_idx = min(idx + 2, n_predictions)
    list_of_candidates = [
        (i, l)
        for i in range(min_idx, max_idx + 1)
        for l in [0, 1, 2]
        if i != idx and str_id_com_label(i, l) in hierarchical_existing_predictions
    ]

    # list_of_candidates = np.array(np.meshgrid(idx_candidates, [0, 1, 2])).T.reshape(-1, 2)
    for cand_loc in list_of_candidates:
        dices[tuple(cand_loc)] = float(np_dice(anchor, hierarchical_predictions[cand_loc[0]][cand_loc[1]]))

    # find k best partners
    dices = dict(sorted(dices.items(), key=lambda item: item[1], reverse=True))
    best_k = list(dices.keys())
    best_k = best_k[:2]

    couple = []
    dice_threshold = 0.3
    if dices[best_k[0]] > dice_threshold:
        couple.append(best_k[0])
    if dices[best_k[1]] > dice_threshold:
        couple.append(best_k[1])
    # if dices[best_k[2]] > dice_threshold:
    #    couple.append(best_k[2])
    if len(couple) == 2:
        # sort out if the other two do not overlap over threshold
        dice_partners = float(
            np_dice(
                hierarchical_predictions[best_k[0][0]][best_k[0][1]],
                hierarchical_predictions[best_k[1][0]][best_k[1][1]],
            )
        )
        if dice_partners < dice_threshold:
            logger.print(couple, " was skipped because the partners do not overlap", verbose=verbose)

    agreement = 0
    if len(couple) > 0:
        # print(couple)
        agreement = 0
        for c in couple:
            agreement += dices[c]
        agreement /= len(couple)
    couple.append((idx, pred))  # add anchor prediction into couple
    couple = tuple(sorted(couple, key=lambda x: x[0]))
    return couple, agreement


def merge_coupled_predictions(
    seg_nii: NII,
    coupled_predictions,
    hierarchical_predictions: np.ndarray,
    debug_data: dict,
    proc_clean_small_cc_artifacts: bool = True,
    vert_size_threshold: int = 0,
    verbose: bool = False,
) -> tuple[NII, dict, ErrCode]:
    whole_vert_nii = seg_nii.copy()
    whole_vert_arr = np.zeros(whole_vert_nii.shape, dtype=np.uint16)  # this is fixed segmentations from vert

    idx = 1
    for k, overall_agreement in coupled_predictions.items():
        # take_no_overlap = len(k) <= 2 or overall_agreement < 0.45
        take_no_overlap = len(k) <= 2
        if overall_agreement < 0.3 + 0.15 * (4 - len(k)):
            take_no_overlap = True
        combine = np.zeros(whole_vert_nii.shape, dtype=whole_vert_nii.dtype)
        for cid in k:
            combine += hierarchical_predictions[cid[0]][cid[1]]
        # print(combine.shape)
        m = 1 if take_no_overlap else 2
        # m = min(max(1, np.max(combine)), 2)  # type:ignore
        combine[combine < m] = 0
        combine[combine != 0] = idx

        count_new = np_count_nonzero(combine)
        if count_new == 0:
            logger.print("ZERO instance mask failure on vertebra instance creation", Log_Type.FAIL)
            return seg_nii, debug_data, ErrCode.EMPTY
        fixed_n = combine.copy()
        fixed_n[whole_vert_arr != 0] = 0
        count_cut = np_count_nonzero(fixed_n)
        relative_overlap = (count_new - count_cut) / count_new
        if relative_overlap > 0.6:
            logger.print(k, f" was skipped because it overlaps {round(relative_overlap, 4)} with established verts", verbose=verbose)
            continue
        whole_vert_arr[whole_vert_arr == 0] = combine[whole_vert_arr == 0]
        idx += 1

    debug_data["inst_crop_vert_arr_a_raw"] = seg_nii.set_array(whole_vert_arr)

    if len(np_unique(whole_vert_arr)) == 1:
        logger.print("Vert mask empty, will skip", Log_Type.FAIL)
        return whole_vert_nii.set_array_(whole_vert_arr, verbose=False), debug_data, ErrCode.EMPTY

    # Cleanup step
    if proc_clean_small_cc_artifacts:
        whole_vert_arr = clean_cc_artifacts(
            whole_vert_arr,
            labels=np_unique(whole_vert_arr)[1:],  # type:ignore
            cc_size_threshold=vert_size_threshold,
            only_delete=True,
            logger=logger,
            verbose=verbose,
        )
    # print("whole_vert_arr", whole_vert_arr.shape)
    # print("seg_nii", seg_nii.shape)
    whole_vert_nii_proc = seg_nii.set_array(whole_vert_arr)
    # print("whole_vert_arr_proc", whole_vert_arr_proc.shape)
    # debug_data["whole_vert_arr_proc"] = seg_nii.set_array(whole_vert_arr)
    # return seg_nii.set_array(whole_vert_arr, verbose=False).map_labels_(com_map, verbose=False), debug_data, ErrCode.OK
    return whole_vert_nii_proc, debug_data, ErrCode.OK
