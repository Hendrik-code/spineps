"""Post-processing phase: clean and reconcile the semantic and vertebra-instance masks and attach IVD/endplate instance labels."""

from __future__ import annotations

# from utils.predictor import nnUNetPredictor
import heapq

import numpy as np
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
from spineps.seg_pipeline import ENDPLATE_LABEL_OFFSET, IVD_LABEL_OFFSET, logger, vertebra_subreg_labels
from spineps.utils.compat import zip_strict
from spineps.utils.proc_functions import fix_wrong_posterior_instance_label
from spineps.utils.resolution import REFERENCE_VOXEL_VOLUME_MM3, REFERENCE_ZOOM, isotropic_area_to_voxels

# --- Label-id conventions for combined post-processing ---
# Intervertebral discs (IVDs) and vertebral endplates reuse their parent vertebra's instance label,
# shifted by IVD_LABEL_OFFSET / ENDPLATE_LABEL_OFFSET (imported from seg_pipeline, their canonical home).
# The dens (odontoid process) anatomically belongs to the C2 vertebra (instance label 2).
C2_INSTANCE_LABEL = 2
# Raw vertebra instance labels stay below this bound; anything above is an IVD/endplate/derived label.
INSTANCE_LABEL_LIMIT = 40

# --- Heuristic thresholds for combined post-processing ---
# Physical margin kept around the segmentation when cropping before processing.
POSTPROCESS_CROP_MARGIN_MM = 2 * min(REFERENCE_ZOOM)
# Warn when a vertebra's unmatched semantic volume exceeds this fraction of an average vertebra.
UNMATCHED_VOLUME_WARN_FRACTION = 0.5
# Endplate splitting dilates iteratively with radius 1 up to (but excluding) this value.
MAX_ENDPLATE_DILATION = 15
# Two stacked vertebrae are merged only if the smaller is below this fraction of the larger...
MERGED_VERTEBRA_SIZE_RATIO = 0.5
# ...and the two masks share at least this much contact area (orientation-agnostic).
MERGED_VERTEBRA_MIN_CONTACT_MM2 = 20 * REFERENCE_VOXEL_VOLUME_MM3 ** (2.0 / 3.0)
# An articular substructure CC is reassigned when its largest overlap dominates the second by this ratio.
ARTICULAR_DOMINANCE_RATIO = 0.5


def phase_postprocess_combined(
    img_nii: NII,
    seg_nii: NII,
    vert_nii: NII,
    model_labeling: VertLabelingClassifier | None,
    debug_data: dict | None,
    labeling_offset: int = 0,
    proc_lab_force_no_tl_anomaly: bool = False,
    proc_assign_missing_cc: bool = True,
    proc_assign_missing_cc_fast=False,
    proc_clean_inst_by_sem: bool = True,
    n_vert_bodies: int | None = None,
    process_merge_vertebra: bool = True,
    proc_vertebra_inconsistency: bool = True,
    proc_assign_posterior_instance_label: bool = True,
    verbose: bool = False,
    disable_c1=True,
    sacrum_ids=(v_name2idx["S1"],),
) -> tuple[NII, NII]:
    """Run the combined semantic/instance post-processing pipeline and return cleaned, anatomically labeled masks.

    Crops both masks to the segmentation, optionally fixes superior/inferior articular inconsistencies, reconciles the
    instance and semantic masks (reassigning or deleting unmatched connected components), splits accidentally merged vertebrae,
    fixes mislabeled posterior elements, labels instances top-to-bottom, optionally runs the anatomical labeling classifier,
    forces the sacrum and dens labels, attaches IVD and endplate instance labels (splitting endplates into superior/inferior),
    and finally un-crops back to the original field of view.

    Args:
        img_nii (NII): Input MRI image.
        seg_nii (NII): Subregion semantic segmentation mask.
        vert_nii (NII): Vertebra instance segmentation mask.
        model_labeling (VertLabelingClassifier | None): Anatomical labeling classifier; if None, instances keep their
            top-to-bottom labels.
        debug_data (dict | None): Optional dict that intermediate results are written into; created if None.
        labeling_offset (int): Offset added to the top-to-bottom instance labels.
        proc_lab_force_no_tl_anomaly (bool): If True, disallow T13 transitional-vertebra anomalies during labeling.
        proc_assign_missing_cc (bool): If True, reassign semantic connected components not covered by the instance mask.
        proc_assign_missing_cc_fast (bool): If True, use the faster infect-based missing-CC assignment.
        proc_clean_inst_by_sem (bool): If True, mask the instance mask by the semantic mask before processing.
        n_vert_bodies (int | None): Number of vertebra bodies; inferred from the instance mask if None.
        process_merge_vertebra (bool): If True, detect and merge accidentally split adjacent vertebrae.
        proc_vertebra_inconsistency (bool): If True, reassign inconsistent articular substructures by instance overlap.
        proc_assign_posterior_instance_label (bool): If True, fix wrongly labeled posterior instance elements.
        verbose (bool): If True, print verbose progress.
        disable_c1 (bool): If True (and ``labeling_offset >= 1``), do not predict a C1 label.
        sacrum_ids (tuple): Semantic label id(s) treated as sacrum and mapped to the S1 instance label.

    Returns:
        tuple[NII, NII]: The cleaned ``(seg_uncropped, vert_uncropped)`` semantic and vertebra-instance masks.
    """
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
        crop_slices = seg_nii.compute_crop(dist=POSTPROCESS_CROP_MARGIN_MM / min(seg_nii.zoom))

        # Save uncropped to uncrop later
        vert_uncropped = vert_nii.copy()
        seg_uncropped = seg_nii.copy()

        # Crop down
        img_nii = img_nii.apply_crop(crop_slices)
        vert_nii.apply_crop_(crop_slices)
        seg_nii.apply_crop_(crop_slices)

        # Post processing both
        ###################

        if proc_vertebra_inconsistency:
            # Assigns superior/inferior based on instance label overlap
            assign_vertebra_inconsistency(vert_nii, seg_nii)

        whole_vert_nii_cleaned, seg_nii_cleaned = mask_cleaning_other(
            whole_vert_nii=vert_nii,
            seg_nii=seg_nii,
            n_vert_bodies=n_vert_bodies,
            proc_assign_missing_cc=proc_assign_missing_cc,
            proc_assign_missing_cc_fast=proc_assign_missing_cc_fast,
            verbose=verbose,
        )

        if process_merge_vertebra and Location.Vertebra_Disc.value in seg_nii_cleaned.unique():
            detect_and_solve_merged_vertebra(seg_nii_cleaned, whole_vert_nii_cleaned)

        if proc_assign_posterior_instance_label:
            whole_vert_nii_cleaned = fix_wrong_posterior_instance_label(seg_nii_cleaned, seg_inst=whole_vert_nii_cleaned, logger=logger)

        # Label vertebra top -> down
        whole_vert_nii_cleaned, vert_labels = label_instance_top_to_bottom(whole_vert_nii_cleaned, labeling_offset=labeling_offset)
        logger.print(f"Labeled {len(vert_labels)} vertebra instances from top to bottom")

        if model_labeling is not None:
            whole_vert_nii_cleaned = perform_labeling_step(
                model=model_labeling,
                img_nii=img_nii,
                vert_nii=whole_vert_nii_cleaned,
                subreg_nii=seg_nii_cleaned,
                proc_lab_force_no_tl_anomaly=proc_lab_force_no_tl_anomaly,
                disable_c1=labeling_offset >= 1 and disable_c1,
            )

        logger.print("vert_nii volumes:", whole_vert_nii_cleaned.volumes())
        logger.print("seg_nii", seg_nii_cleaned.unique())

        whole_vert_nii_cleaned[seg_nii_cleaned.extract_label(sacrum_ids).get_seg_array() == 1] = v_name2idx["S1"]
        whole_vert_nii_cleaned[seg_nii_cleaned == Location.Dens_axis.value] = C2_INSTANCE_LABEL
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

        logger.print("vert_uncropped volumes", vert_uncropped.volumes())
        logger.print("seg_uncropped", seg_uncropped.unique())

        debug_data["vert_arr_return_final"] = vert_uncropped.copy()
    return seg_uncropped, vert_uncropped


def mask_cleaning_other(
    whole_vert_nii: NII,
    seg_nii: NII,
    n_vert_bodies: int,
    proc_assign_missing_cc: bool = False,
    proc_assign_missing_cc_fast=False,
    verbose: bool = False,
) -> tuple[NII, NII]:
    """Reconcile the vertebra instance mask with the vertebra portion of the semantic mask.

    Extracts the vertebra subregions from the semantic mask and (optionally) reassigns semantic connected components that the
    instance mask missed, either via a fast infect pass or via :func:`assign_missing_cc`; deleted components are removed from the
    semantic mask. Logs a warning when the unmatched vertebra volume between the two masks is anomalously large.

    Args:
        whole_vert_nii (NII): Vertebra instance segmentation mask.
        seg_nii (NII): Subregion semantic segmentation mask.
        n_vert_bodies (int): Number of vertebra bodies, used to scale the unmatched-volume warning.
        proc_assign_missing_cc (bool): If True, reassign missed semantic components via :func:`assign_missing_cc`.
        proc_assign_missing_cc_fast (bool): If True, additionally run a fast infect-based assignment first.
        verbose (bool): If True, print verbose progress.

    Returns:
        tuple[NII, NII]: The cleaned ``(whole_vert_nii, seg_nii)`` instance and semantic masks.

    Raises:
        AssertionError: If, with ``proc_assign_missing_cc`` enabled, the instance mask still has more vertebra voxels than the
            semantic mask (which should be impossible).
    """
    subreg_vert_nii = seg_nii.extract_label(vertebra_subreg_labels)

    if proc_assign_missing_cc_fast:
        logger.print("missing cc (fast)")
        missing = subreg_vert_nii.copy()
        missing[whole_vert_nii != 0] = 0
        infect = missing  # .filter_connected_components(max_volume=fast_assinge_missing_cc_size)
        whole_vert_nii = whole_vert_nii.infect(infect, verbose=verbose)
    # make copy where both masks clean each other
    vert_arr_cleaned = whole_vert_nii.get_seg_array()
    subreg_vert_arr = subreg_vert_nii.get_seg_array()
    # if dilation_fill:
    #    vert_arr_cleaned = np_dilate_msk(vert_arr_cleaned, label_ref=vert_labels, mm=5)  # , mask=subreg_vert_arr
    subreg_arr = seg_nii.get_seg_array()

    if proc_assign_missing_cc:
        vert_arr_cleaned, subreg_vert_arr, deletion_map = assign_missing_cc(vert_arr_cleaned, subreg_vert_arr, verbose=verbose)
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
    elif n_vert_pixels_rel_diff > UNMATCHED_VOLUME_WARN_FRACTION:
        logger.print(f"A volume of {n_vert_pixels_rel_diff} * avg_vertebra_volume in subreg not matched by vertebra mask", Log_Type.WARNING)

    return whole_vert_nii.set_array(vert_arr_cleaned), seg_nii.set_array(subreg_arr)


def assign_missing_cc(
    target_arr: np.ndarray,
    reference_arr: np.ndarray,
    verbose: bool = False,
    verbose_deletion: bool = False,
    proc_assign_missing_dilate_first: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign reference-mask connected components that the target mask does not cover to a neighboring target label.

    Finds connected components of ``reference_arr`` (e.g. the vertebra semantic mask) that have no overlap with ``target_arr``
    (the instance mask). Optionally dilates the target mask once first to absorb thin gaps. Each remaining component is dilated
    locally and assigned to the most common neighboring target label; components with no labeled neighbor are deleted from the
    reference mask and recorded in a deletion map.

    Args:
        target_arr (np.ndarray): Instance label array that components are assigned to.
        reference_arr (np.ndarray): Reference (semantic) label array whose uncovered components are processed.
        verbose (bool): If True, log each assignment.
        verbose_deletion (bool): If True, log each deletion even when ``verbose`` is False.
        proc_assign_missing_dilate_first (bool): If True, dilate the target mask once before searching for uncovered components.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(target_arr, reference_arr, deletion_map)`` with the updated instance and
            reference arrays and a binary map of voxels removed from the reference mask.

    Raises:
        AssertionError: If ``target_arr`` and ``reference_arr`` do not share the same shape.
    """
    assert target_arr.shape == reference_arr.shape

    deletion_map = np.zeros_like(reference_arr, dtype=np.uint8)

    # only reference pixels not already covered by target
    subreg_arr_vert_rest = np.where(target_arr == 0, reference_arr, 0)

    label_rest = np_unique(subreg_arr_vert_rest)
    if len(label_rest) == 1 and label_rest[0] == 0:
        logger.print("No CC had to be assigned", Log_Type.OK, verbose=verbose)
        return target_arr, reference_arr, deletion_map

    # dilate once first
    if proc_assign_missing_dilate_first:
        target_arr_ = np_dilate_msk(
            target_arr,
            None,
            n_pixel=2,
            connectivity=1,
            mask=reference_arr,
            use_crop=False,
        )
        subreg_arr_vert_rest = reference_arr.copy()
        subreg_arr_vert_rest[target_arr_ != 0] = 0
        deletion_map = np.zeros(reference_arr.shape)

        label_rest = np_unique(subreg_arr_vert_rest)
        if len(label_rest) == 1 and label_rest[0] == 0:
            logger.print("No CC had to be assigned", Log_Type.OK, verbose=verbose)
            return target_arr_, reference_arr, deletion_map

        target_arr = target_arr_
    # subreg_arr_vert_rest is not hit pixels bei vertebra prediction
    subreg_cc = np_connected_components_per_label(subreg_arr_vert_rest, connectivity=2)

    loop_counts = 0

    for label, subreg_cc_map in subreg_cc.items():
        if label == 0:
            continue

        cc_labels = np_unique_withoutzero(subreg_cc_map)
        loop_counts += len(cc_labels)

        for cc_l in cc_labels:
            cc_mask = subreg_cc_map == cc_l

            cc_bbox = np_bbox_binary(cc_mask, px_dist=2)

            cc_mask_c = cc_mask[cc_bbox]
            vert_arr_c = target_arr[cc_bbox]

            # dilate only local mask
            cc_map_dilated = np_dilate_msk(cc_mask_c.astype(np.uint8), 1, n_pixel=1, connectivity=2).astype(bool)

            # neighborhood labels
            neighbor_labels = vert_arr_c[cc_map_dilated]
            neighbor_labels = neighbor_labels[neighbor_labels != 0]

            if neighbor_labels.size > 0:
                new_label = np.bincount(neighbor_labels).argmax()
                if verbose:
                    logger.print(
                        f"Assign Missing CC {(label, cc_l)} to {new_label}, Location at {cc_bbox}, {cc_mask.sum()}", verbose=verbose
                    )

                vert_arr_c[cc_mask_c] = new_label
                target_arr[cc_bbox] = vert_arr_c

            else:
                logger.print(f"Assign {(label, cc_l)} to EMPTY, Location at {cc_bbox}", verbose=verbose or verbose_deletion)

                reference_arr[cc_bbox][cc_mask_c == 1] = 0
                deletion_map[cc_bbox][cc_mask_c == 1] = 1

    logger.print(f"Assign missing cc: Processed {loop_counts} missed ccs")

    return target_arr, reference_arr, deletion_map


def add_ivd_ep_vert_label(whole_vert_nii: NII, seg_nii: NII, verbose=True) -> tuple[np.ndarray, np.ndarray]:
    """Attach intervertebral-disc and endplate instance labels and split endplates into superior/inferior.

    Reorients both masks to PIR, computes each vertebra corpus center of mass along the inferior-superior axis, then assigns
    every IVD and endplate connected component to the nearest lower vertebra. IVD voxels are written into the instance array
    with ``IVD_LABEL_OFFSET`` added; endplate voxels with ``ENDPLATE_LABEL_OFFSET`` added. Endplates are further divided into
    inferior/superior plates by iteratively dilating each vertebra into the endplate region. Logs the number of assigned
    components and restores the original orientation before returning.

    Args:
        whole_vert_nii (NII): Vertebra instance segmentation mask.
        seg_nii (NII): Subregion semantic segmentation mask (must contain disc/endplate/corpus labels).
        verbose (bool): If True, print endplate-detection progress.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(vert_arr, seg_arr)`` arrays in the original orientation: the instance array with IVD
            and endplate instance labels added, and the semantic array with endplates split into superior/inferior plates.
    """
    # PIR
    orientation = whole_vert_nii.orientation
    vert_t = whole_vert_nii.reorient()
    seg_t = seg_nii.reorient()
    vert_labels = [t for t in vert_t.unique() if t <= 26 or t == 28]  # without zero
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
            if l not in to_mapped_labels and l != 1:
                logger.print(f"Vertebra {v_idx2name[l]} got no IVD component assigned", Log_Type.STRANGE)
            count = to_mapped_labels.count(l)
            if count > 1:
                logger.print(f"Vertebra {v_idx2name[l]} got {count} IVD components assigned", Log_Type.STRANGE)

        subreg_ivd = subreg_cc.copy()
        n_ivd_unique = len(np.unique(to_mapped_labels))
        subreg_ivd = np_map_labels(subreg_ivd, label_map=mapping_cc_to_vert_label)
        subreg_ivd += IVD_LABEL_OFFSET
        subreg_ivd[subreg_ivd == IVD_LABEL_OFFSET] = 0
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
        subreg_ep += ENDPLATE_LABEL_OFFSET
        subreg_ep[subreg_ep == ENDPLATE_LABEL_OFFSET] = 0
        vert_arr[subreg_arr == Location.Endplate.value] = subreg_ep[subreg_arr == Location.Endplate.value]
        vert_t.set_array_(vert_arr)

        # divide into upper and lower endplate
        out = seg_t * 0
        pref = 1
        old_vol = -1
        # seg_t and vert_t are not modified in this loop, so compute these invariants once.
        endplate_nii = seg_t.extract_label(Location.Endplate.value)
        total = endplate_nii.sum()
        vert_labels_to_split = vert_t.unique()
        for dil in range(1, MAX_ENDPLATE_DILATION):
            curr = out.extract_label([Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value])
            new_vol = curr.sum()
            logger.print(rf"{new_vol / total * 100:.2f}% endplates detected", end="\r") if verbose else None
            if old_vol == new_vol and old_vol != 0:
                break
            old_vol = new_vol
            if total == new_vol:
                logger.print("Found all Endplates                                      ")
                break
            for i in vert_labels_to_split:
                if i >= INSTANCE_LABEL_LIMIT:
                    break
                curr = out.extract_label([Location.Vertebral_Body_Endplate_Inferior.value, Location.Vertebral_Body_Endplate_Superior.value])
                v = vert_t.extract_label(i).dilate_msk(dil, verbose=False)
                end = endplate_nii * v
                end *= -curr + 1  # type: ignore
                plates = vert_t * end
                plates.map_labels_(
                    {
                        i + ENDPLATE_LABEL_OFFSET: Location.Vertebral_Body_Endplate_Inferior.value,
                        pref + ENDPLATE_LABEL_OFFSET: Location.Vertebral_Body_Endplate_Superior.value,
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


def find_nearest_lower(seq, x) -> float:
    """Return the largest element of ``seq`` strictly smaller than ``x``, or the minimum if none exists.

    Args:
        seq (Sequence[float]): Values to search.
        x (float): Reference value.

    Returns:
        float: The greatest element below ``x``, or ``min(seq)`` when no element is below ``x``.
    """
    values_lower = [item for item in seq if item < x]
    if len(values_lower) == 0:
        return min(seq)
    return max(values_lower)


def label_instance_top_to_bottom(vert_nii: NII, labeling_offset: int = 0) -> tuple[NII, np.ndarray]:
    """Relabel vertebra instances consecutively from top to bottom by center-of-mass height.

    Reorients to PIR, sorts the instances by their center of mass along the inferior-superior axis, and assigns consecutive
    labels (``1 + labeling_offset`` upward) from top to bottom, then restores the original orientation.

    Args:
        vert_nii (NII): Vertebra instance segmentation mask (modified in place).
        labeling_offset (int): Offset added to the consecutive labels.

    Returns:
        tuple[NII, np.ndarray]: The relabeled instance mask and its array of unique labels.
    """
    ori = vert_nii.orientation
    vert_nii.reorient_()
    vert_arr = vert_nii.get_seg_array()
    com_i = np_center_of_mass(vert_arr)
    comb_l = list(zip_strict(com_i.keys(), com_i.values()))
    comb_l.sort(key=lambda a: a[1][1])  # PIR
    com_map = {comb_l[idx][0]: idx + 1 + labeling_offset for idx in range(len(comb_l))}

    vert_nii.map_labels_(com_map, verbose=False)
    vert_nii.reorient_(ori)
    return vert_nii, vert_nii.unique()


def assign_vertebra_inconsistency(
    vert_nii: NII,
    seg_nii: NII,
    locations=(
        Location.Superior_Articular_Left,
        Location.Superior_Articular_Right,
        Location.Inferior_Articular_Left,
        Location.Inferior_Articular_Right,
    ),
) -> None:
    """Reassign articular-process components to the vertebra instance they most overlap with.

    For each given articular subregion location, finds its connected components in the semantic mask and, for each component,
    reassigns its instance label to the vertebra whose overlap volume dominates the second-largest by ``ARTICULAR_DOMINANCE_RATIO``.
    Updates ``vert_nii`` in place.

    Args:
        vert_nii (NII): Vertebra instance segmentation mask (modified in place).
        seg_nii (NII): Subregion semantic segmentation mask.
        locations (tuple[Location, ...]): Articular subregion locations to reconcile.

    Returns:
        None: ``vert_nii`` is modified in place.
    """
    seg_nii.assert_affine(shape=vert_nii.shape)
    seg_arr = seg_nii.get_seg_array()
    vert_arr = vert_nii.get_seg_array()

    # assign inconsistent substructures
    for loc in locations:
        value = loc.value

        subreg_l = np_extract_label(seg_arr, value, inplace=False)  # type:ignore
        try:
            subreg_cc, _ = np_connected_components(subreg_l, label_ref=1)
        except AssertionError as e:
            print(f"Got error {e}, skip")
            break
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

            if biggest_volume[1] * ARTICULAR_DOMINANCE_RATIO > second_volume[1]:
                to_label = biggest_volume[0] - 1  # int(list(gt_volume.keys())[argmax] - 1)

                vert_arr[cc_map == 1] = to_label
                logger.print(
                    f"set cc to {to_label}, with volume decision {gt_volume}, based on {biggest_volume}, {second_volume}", verbose=False
                )

        vert_nii.set_array_(vert_arr)


def detect_and_solve_merged_vertebra(seg_nii: NII, vert_nii: NII) -> tuple[NII, NII]:
    """Detect and merge a vertebra (typically C2) that was split into two stacked instances.

    Builds a height-sorted list of IVD components and vertebra instances. If the two topmost entries are both vertebrae, the
    upper one is significantly smaller (below ``MERGED_VERTEBRA_SIZE_RATIO`` of the other), and the two masks touch over more
    than ``MERGED_VERTEBRA_MIN_CONTACT_MM2`` of area, the smaller instance is merged into the larger one in ``vert_nii``.

    Args:
        seg_nii (NII): Subregion semantic segmentation mask.
        vert_nii (NII): Vertebra instance segmentation mask (modified in place when a merge occurs).

    Returns:
        tuple[NII, NII]: The ``(seg_nii, vert_nii)`` masks (``vert_nii`` possibly with two instances merged).
    """
    seg_sem = seg_nii.map_labels({Location.Endplate.value: Location.Vertebra_Disc.value}, verbose=False)
    # get all ivd CCs from seg_sem

    stats = {}
    # Map IVDS
    subreg_cc: NII = seg_sem.get_connected_components(labels=Location.Vertebra_Disc.value)
    subreg_cc += 100

    coms = subreg_cc.center_of_masses()
    volumes = subreg_cc.volumes()
    stats = {i: (g[1], True, volumes[i]) for i, g in coms.items()}

    corpus_nii = seg_sem.extract_label([Location.Vertebra_Corpus_border.value, Location.Arcus_Vertebrae.value]) * vert_nii
    vert_coms = corpus_nii.center_of_masses()
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
        if first_stats[2] < MERGED_VERTEBRA_SIZE_RATIO * second_stats[2]:
            # first is significantly smaller than second and they are close in height
            # how many pixels are touching
            vert_firsttwo_arr = vert_nii.extract_label(first_key).get_seg_array()
            vert_firsttwo_arr2 = vert_nii.extract_label(second_key).get_seg_array()
            vert_firsttwo_arr += vert_firsttwo_arr2 + 1
            contacts = np_contacts(vert_firsttwo_arr, connectivity=3)
            if contacts[(1, 2)] > isotropic_area_to_voxels(MERGED_VERTEBRA_MIN_CONTACT_MM2, vert_nii.zoom):
                logger.print("Found first two instance weird, will merge", Log_Type.STRANGE)
                vert_nii.map_labels_({first_key: second_key}, verbose=False)

    return seg_nii, vert_nii
