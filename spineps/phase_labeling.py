"""Vertebra-labeling phase: turns top-to-bottom vertebra instances into anatomical vertebra labels via a classifier and path search."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from TPTBox import NII, Location, No_Logger

from spineps.architectures.read_labels import (
    VertExact,
    VertExactClass,
    VertGroup,
    VertRegion,
    VertRel,
    VertT13,
    vert_group_idx_to_exact_idx_dict,
)
from spineps.get_models import get_actual_model
from spineps.lab_model import VertLabelingClassifier
from spineps.utils.find_min_cost_path import (
    DEFAULT_REGION_STARTS,
    L5_CLASS_IDX,
    T11_CLASS_IDX,
    T12_CLASS_IDX,
    find_most_probably_sequence,
)

logger = No_Logger(prefix="LabelingPhase")

VERT_CLASSES = 24
CERV = slice(None, 7)  # 0 to 7
THOR = slice(7, 19)  # 7 to 18
LUMB = slice(19, None)  # 19 to end (23)

DIVIDE_BY_ZERO_OFFSET = 1e-8

# Cost-matrix class indices (0-based, matching VertExact) of anatomically special vertebrae.
# T11/T12/L5 and the region starts (DEFAULT_REGION_STARTS) are imported from find_min_cost_path,
# their canonical home (the path solver that consumes them).
C1_CLASS_IDX = 0
C2_CLASS_IDX = 1
# Post-processing label for the (anomalous) T13 vertebra; it has no VertExact class.
T13_LABEL = 28
# Crop margin in millimeters kept around the vertebrae before labeling.
LABELING_CROP_MARGIN_MM = 128


def perform_labeling_step(
    model: VertLabelingClassifier,
    img_nii: NII,
    vert_nii: NII,
    subreg_nii: NII | None = None,
    proc_lab_force_no_tl_anomaly: bool = False,
    disable_c1: bool = True,
) -> NII:
    """Assign anatomical vertebra labels to a vertebra instance mask using the labeling classifier.

    Runs the labeling classifier on each vertebra instance, derives a globally consistent label sequence, and relabels the
    instance mask accordingly. If a subregion mask is given, the classifier only sees the vertebra corpus (not the whole vertebra).
    Optionally adds a missing C1 label and zeroes out any instances that could not be matched.

    Args:
        model (VertLabelingClassifier): Classifier used to predict per-instance vertebra labels.
        img_nii (NII): Input MRI image.
        vert_nii (NII): Vertebra instance segmentation mask to be relabeled.
        subreg_nii (NII | None): Subregion semantic mask; if given, vertebrae are masked to their corpus before classification.
        proc_lab_force_no_tl_anomaly (bool): If True, disallow thoracolumbar (T13) transitional-vertebra anomalies.
        disable_c1 (bool): If True, do not predict/add a C1 label.

    Returns:
        NII: The vertebra instance mask relabeled with anatomical vertebra labels (unmatched instances set to 0).
    """
    if model.predictor is None:
        model.load()

    if subreg_nii is not None:
        # crop for corpus instead of whole vertebra
        corpus_nii = subreg_nii.extract_label((Location.Vertebra_Corpus, Location.Vertebra_Corpus_border))
        vert_nii_c = vert_nii * corpus_nii
    else:
        vert_nii_c = vert_nii
    # run model
    labelmap = run_model_for_vert_labeling(
        model,
        img_nii,
        vert_nii_c,
        proc_lab_force_no_tl_anomaly=proc_lab_force_no_tl_anomaly,
        disable_c1=disable_c1,
    )[0]

    vert_nii_u = vert_nii.unique()
    # add C1 if it not set in labelmap, C2 exists, the minimal label (from sorting C1) is not mached anywhere.
    if not disable_c1 and min(vert_nii_u) not in labelmap and 1 not in labelmap.values() and 2 in labelmap.values():
        logger.on_debug("Add C1 after labeling")
        labelmap[min(vert_nii_u)] = 1
    # remove unmatched vertebras
    for i in vert_nii_u:
        if i not in labelmap:
            logger.on_debug(f"Set label to 0, because not found {i}")

            labelmap[i] = 0

    # relabel according to labelmap
    return vert_nii.map_labels_(labelmap)


def run_model_for_vert_labeling(
    model: VertLabelingClassifier,
    img_nii: NII,
    vert_nii: NII,
    verbose: bool = False,
    proc_lab_force_no_tl_anomaly: bool = False,
    disable_c1: bool = True,
) -> tuple[dict[int, int], float, list[int], list[int], list, list, dict]:
    """Run the labeling classifier over a whole image/instance pair and resolve a vertebra label sequence.

    Reorients, crops around the vertebrae, rescales to the model's recommended zoom, runs the classifier on every vertebra
    instance, and uses the cheapest-cost path search to turn per-instance predictions into a consistent anatomical sequence.

    Args:
        model (VertLabelingClassifier): Classifier used to predict per-instance vertebra labels.
        img_nii (NII): Input MRI image.
        vert_nii (NII): Vertebra instance segmentation mask.
        verbose (bool): If True, print intermediate weighting/path information.
        proc_lab_force_no_tl_anomaly (bool): If True, disallow thoracolumbar (T13) transitional-vertebra anomalies.
        disable_c1 (bool): If True, do not predict a C1 label.

    Returns:
        tuple: ``(labelmap, fcost, fpath, fpath_post, costlist, min_costs_path, predictions)`` where ``labelmap`` maps each
            original instance label to its assigned vertebra label, ``fcost`` is the total path cost, ``fpath``/``fpath_post``
            are the raw and post-processed label sequences, ``costlist`` is the cost matrix as a list, ``min_costs_path`` is the
            per-step minimum cost path, and ``predictions`` are the raw classifier outputs.

    Raises:
        AssertionError: If the number of original instances does not match the resolved path length.
    """
    # reorient
    img = img_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    vert = vert_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    zms_pir = img.zoom

    # crop
    crop = vert.compute_crop(dist=LABELING_CROP_MARGIN_MM / min(img.zoom))
    img.apply_crop_(crop)
    vert.apply_crop_(crop)

    # rescale
    img.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    vert.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    #
    img.assert_affine(other=vert)
    # extract vertebrae
    vert.extract_label_([i for i in range(1, 29) if i not in [26, 27]], keep_label=True)
    # counted label
    orig_label = vert.unique()
    # run model
    predictions = model.run_all_seg_instances(img, vert)

    fcost, fpath, fpath_post, costlist, min_costs_path, _args = find_vert_path_from_predictions(
        predictions=predictions,
        proc_lab_force_no_tl_anomaly=proc_lab_force_no_tl_anomaly,
        verbose=verbose,
        disable_c1=disable_c1,
    )
    assert len(orig_label) == len(fpath_post), f"{len(orig_label)} != {len(fpath_post)}"
    labelmap = {orig_label[idx]: fpath_post[idx] for idx in range(len(orig_label))}

    return labelmap, fcost, fpath, fpath_post, costlist, min_costs_path, predictions


def run_model_for_vert_labeling_cutouts(
    model: VertLabelingClassifier,
    img_arrays: dict[int, np.ndarray],
    disable_c1: bool = True,
    boost_c2: float = 3.0,
    allow_cervical_skip: bool = True,
    verbose: bool = True,
) -> tuple[dict[int, int], float, list[int], list[int], list, list, dict]:
    """Run the labeling classifier on precomputed per-instance image cutouts and resolve a vertebra label sequence.

    Like :func:`run_model_for_vert_labeling`, but skips reorienting/cropping/rescaling and instead consumes already-prepared
    image arrays keyed by instance label.

    Args:
        model (VertLabelingClassifier): Classifier used to predict per-instance vertebra labels.
        img_arrays (dict[int, np.ndarray]): Mapping of vertebra instance label to its cropped image array.
        disable_c1 (bool): If True, do not predict a C1 label.
        boost_c2 (float): Multiplicative boost applied to a prediction whose argmax is C2.
        allow_cervical_skip (bool): If True, allow the path search to skip a class within the cervical region.
        verbose (bool): If True, print intermediate weighting/path information.

    Returns:
        tuple: ``(labelmap, fcost, fpath, fpath_post, costlist, min_costs_path, predictions)`` (see
            :func:`run_model_for_vert_labeling`).

    Raises:
        AssertionError: If the number of input arrays does not match the resolved path length.
    """
    # reorient
    # img = img_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    # vert = vert_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    # zms_pir = img.zoom
    # rescale
    # img.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    # vert.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    #
    # img.assert_affine(other=vert)
    # extract vertebrae
    # vert.extract_label_(list(range(1, 26)), keep_label=True)
    # counted label
    orig_label = list(img_arrays.keys())
    # run model
    predictions = model.run_all_arrays(img_arrays)
    fcost, fpath, fpath_post, costlist, min_costs_path, _args = find_vert_path_from_predictions(
        predictions=predictions,
        verbose=verbose,
        disable_c1=disable_c1,
        boost_c2=boost_c2,
        allow_cervical_skip=allow_cervical_skip,
    )
    assert len(orig_label) == len(fpath_post), f"{len(orig_label)} != {len(fpath_post)}"
    labelmap = {orig_label[idx]: fpath_post[idx] for idx in range(len(orig_label))}

    return labelmap, fcost, fpath, fpath_post, costlist, min_costs_path, predictions


def region_to_vert(region_softmax_values: np.ndarray) -> np.ndarray:  # shape(1,3)
    """Broadcast a 3-region (cervical, thoracic, lumbar) softmax into a per-vertebra-class vector.

    Args:
        region_softmax_values (np.ndarray): Length-3 region softmax values ordered cervical, thoracic, lumbar.

    Returns:
        np.ndarray: Length-``VERT_CLASSES`` vector with each region's value broadcast across that region's vertebra classes.
    """
    vert_prediction_values = np.zeros(VERT_CLASSES)
    vert_prediction_values[CERV] = region_softmax_values[0]
    vert_prediction_values[THOR] = region_softmax_values[1]
    vert_prediction_values[LUMB] = region_softmax_values[2]
    return vert_prediction_values


def prepare_vert(
    vert_softmax_values: np.ndarray,
    gaussian_sigma: float = 0.85,
    gaussian_radius: int = 2,
    gaussian_regionwise: bool = True,
) -> np.ndarray:
    """Smooth and normalize a per-vertebra-class softmax vector.

    Optionally applies a 1-D Gaussian filter (either per spinal region or across all classes) and then normalizes to sum to 1.

    Args:
        vert_softmax_values (np.ndarray): Length-``VERT_CLASSES`` per-class softmax values.
        gaussian_sigma (float): Gaussian smoothing sigma; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.
        gaussian_regionwise (bool): If True, smooth each spinal region independently instead of across the whole vector.

    Returns:
        np.ndarray: The smoothed, sum-normalized per-class vector.
    """
    # gaussian region-wise
    softmax_values = vert_softmax_values.copy()
    if gaussian_sigma > 0.0:
        if gaussian_regionwise:
            for s in [CERV, THOR, LUMB]:
                softmax_values[s] = gaussian_filter1d(softmax_values[s], sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
        else:
            softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def prepare_vertgrp(
    vertgrp_softmax_values: np.ndarray,
    gaussian_sigma: float = 0.85,
    gaussian_radius: int = 2,
    gaussian_regionwise: bool = True,
) -> np.ndarray:
    """Expand a vertebra-group softmax to per-vertebra classes, then smooth and normalize it.

    Distributes each vertebra-group probability onto its member vertebra classes (via ``vert_group_idx_to_exact_idx_dict``),
    optionally applies a 1-D Gaussian filter (per region or globally), and normalizes to sum to 1.

    Args:
        vertgrp_softmax_values (np.ndarray): Per-vertebra-group softmax values.
        gaussian_sigma (float): Gaussian smoothing sigma; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.
        gaussian_regionwise (bool): If True, smooth each spinal region independently instead of across the whole vector.

    Returns:
        np.ndarray: The expanded, smoothed, sum-normalized per-class vector.
    """
    # gaussian region-wise
    softmax_values = np.zeros(VERT_CLASSES)
    for i, g in vert_group_idx_to_exact_idx_dict.items():
        softmax_values[g] = vertgrp_softmax_values[i]
    if gaussian_sigma > 0.0:
        if gaussian_regionwise:
            for s in [CERV, THOR, LUMB]:
                softmax_values[s] = gaussian_filter1d(softmax_values[s], sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
        else:
            softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def prepare_visible(predictions: dict, visible_w: float = 1.0, gaussian_sigma: float = 0.8, gaussian_radius: int = 2) -> np.ndarray:
    """Build a per-instance confidence-weighting chain from the classifier's "fully visible" head.

    For each instance, reads the probability of being fully visible (if the ``FULLYVISIBLE`` head is present, else assumes 1),
    optionally Gaussian-smooths it along the instance axis, and converts it into a multiplicative weight in ``[0, 1]`` that
    down-weights partially visible (cropped) vertebrae according to ``visible_w``.

    Args:
        predictions (dict): Per-instance classifier outputs, each holding a ``"soft"`` dict of head softmax arrays.
        visible_w (float): Strength of the visibility down-weighting; 0 disables it.
        gaussian_sigma (float): Gaussian smoothing sigma along the instance axis; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.

    Returns:
        np.ndarray: Per-instance multiplicative weights clipped to ``[0, 1]``.
    """
    # has soft and FULLYVISIBLE key
    predict_keys = list(predictions[list(predictions.keys())[0]]["soft"].keys())  # noqa: RUF015
    if "FULLYVISIBLE" in predict_keys:
        visible_chain = np.asarray([k["soft"]["FULLYVISIBLE"][1] for k in predictions.values()])
    else:
        visible_chain = np.ones(len(predictions))
    if gaussian_sigma > 0.0:
        visible_chain = gaussian_filter1d(visible_chain, sigma=gaussian_sigma, mode="constant", radius=gaussian_radius)
    visible_chain = np.round(visible_chain, 3)
    # weighting
    visible_chain = 1 - visible_chain
    visible_chain = np.multiply(visible_chain, visible_w)
    visible_chain = 1 - visible_chain
    visible_chain = np.clip(visible_chain, 0, 1)
    # visible_chain /= np.sum(visible_chain)
    return visible_chain


def prepare_region(region_softmax_values: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1) -> np.ndarray:
    """Broadcast a region softmax to per-vertebra classes, then smooth and normalize it.

    Args:
        region_softmax_values (np.ndarray): Length-3 region softmax values (cervical, thoracic, lumbar).
        gaussian_sigma (float): Gaussian smoothing sigma; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.

    Returns:
        np.ndarray: The broadcast, smoothed, sum-normalized per-class vector.
    """
    softmax_values = region_to_vert(region_softmax_values)
    if gaussian_sigma > 0.0 and np.sum(softmax_values) > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def prepare_vertrel_columns(vertrel_matrix: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1) -> np.ndarray:
    """Smooth and column-normalize the relative-position (VertRel) cost matrix.

    For each VertRel label (column, skipping the first), optionally Gaussian-smooths the values along the instance axis and
    normalizes the column so its values stay bounded (divides by the column sum when it exceeds 1, otherwise by ``1 + sum``).

    Args:
        vertrel_matrix (np.ndarray): Matrix of shape ``(n_instances, len(VertRel))`` of relative-position softmax values.
        gaussian_sigma (float): Gaussian smoothing sigma along the instance axis; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.

    Returns:
        np.ndarray: The smoothed, column-normalized relative-position matrix (modified in place and returned).
    """
    for i in range(1, min(len(VertRel), vertrel_matrix.shape[1])):
        if gaussian_sigma > 0.0 and np.sum(vertrel_matrix) > 0.0:
            vertrel_matrix[:, i] = gaussian_filter1d(vertrel_matrix[:, i], sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
        # normalize per column / label in this case
        vertrel_sum = np.sum(vertrel_matrix[:, i]) + DIVIDE_BY_ZERO_OFFSET
        if vertrel_sum > 1.0:
            vertrel_matrix[:, i] = vertrel_matrix[:, i] / vertrel_sum
        elif vertrel_sum < 1.0:
            vertrel_matrix[:, i] = vertrel_matrix[:, i] / (1.0 + vertrel_sum)
    return vertrel_matrix


def prepare_vertt13_columns(vertt13_matrix: np.ndarray) -> np.ndarray:
    """Column-normalize the T13-anomaly (VertT13) cost matrix.

    Normalizes each VertT13 label (column, skipping the first) so it sums to 1 along the instance axis.

    Args:
        vertt13_matrix (np.ndarray): Matrix of shape ``(n_instances, len(VertT13))`` of T13-anomaly softmax values.

    Returns:
        np.ndarray: The column-normalized matrix (modified in place and returned).
    """
    for i in range(1, min(len(VertT13), vertt13_matrix.shape[1])):
        # normalize per column / label in this case
        vertt13_matrix[:, i] = vertt13_matrix[:, i] / (np.sum(vertt13_matrix[:, i]) + DIVIDE_BY_ZERO_OFFSET)
    return vertt13_matrix


def prepare_vertrel(vertrel_softmax_values: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1) -> np.ndarray:
    """Optionally Gaussian-smooth a relative-position (VertRel) softmax vector.

    Args:
        vertrel_softmax_values (np.ndarray): Relative-position softmax values for a single instance.
        gaussian_sigma (float): Gaussian smoothing sigma; 0 disables smoothing.
        gaussian_radius (int): Half-width of the Gaussian kernel.

    Returns:
        np.ndarray: The (optionally smoothed) relative-position vector; not re-normalized.
    """
    softmax_values = vertrel_softmax_values.copy()
    if gaussian_sigma > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    # softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def find_vert_path_from_predictions(
    predictions,
    visible_w: float = 0.5,
    vert_w: float = 0.9,  # 0.9
    vertgrp_w: float = 0.8,
    region_w: float = 1.1,  # 1.1
    vertrel_w: float = 0.6,  # 0.3
    vertt13_w: float = 0.4,
    disable_c1: bool = True,
    boost_c2: float = 1.0,  # 3.0
    allow_cervical_skip: bool = False,
    allow_thoracic_skip: bool = False,
    allow_lumbar_skip: bool = False,
    #
    punish_multiple_sequence: float = 0.0,
    punish_skip_sequence: float = 0.0,
    punish_skip_at_region_sequence: float = 0.0,
    #
    region_gaussian_sigma: float = 0.0,  # 0 means no gaussian
    vert_gaussian_sigma: float = 0.8,  # 0.8 0 means no gaussian
    vert_gaussian_regionwise: bool = True,
    vertgrp_gaussian_sigma: float = 0.8,  # 0.8 0 means no gaussian
    vertgrp_gaussian_regionwise: bool = True,
    vertrel_column_norm: bool = True,
    vertrel_gaussian_sigma: float = 0.6,  # 0.6 # 0 means no gaussian
    #
    focus_tl_gap: bool = True,  # focus on T11/T13 gap (if T11/t13 case is detected, predict again using crops and then check again)
    argmax_combined_cost_matrix_instead_of_path_algorithm: bool = False,
    proc_lab_force_no_tl_anomaly: bool = False,
    #
    verbose: bool = False,
) -> tuple[float, list[int], list[int], list, list, dict]:
    """Combine the classifier's prediction heads into a cost matrix and solve for the most probable vertebra label sequence.

    Builds a per-instance / per-class cost matrix by weighting and summing the available prediction heads (VERT, VERTGRP,
    REGION), down-weighting by the "fully visible" chain, optionally boosting C2, and adding separate relative-position
    (VertRel) and T13-anomaly (VertT13) cost terms. The cheapest monotonically increasing label path is then found with
    :func:`find_most_probably_sequence` (unless ``argmax_combined_cost_matrix_instead_of_path_algorithm`` is set, which falls
    back to a plain per-instance argmax). Special transitional vertebrae (T11 skip, T12/L5 repeats) and per-region skips are
    permitted via the corresponding flags. Finally the path is post-processed (see :func:`fpath_post_processing`).

    Args:
        predictions (dict): Per-instance classifier outputs, each holding a ``"soft"`` dict of per-head softmax arrays.
        visible_w (float): Weight of the "fully visible" down-weighting (must be in ``[0, 1]``).
        vert_w (float): Weight of the per-vertebra (VERT) head.
        vertgrp_w (float): Weight of the vertebra-group (VERTGRP) head.
        region_w (float): Weight of the spinal-region (REGION) head.
        vertrel_w (float): Weight of the relative-position (VERTREL) cost term.
        vertt13_w (float): Weight of the T13-anomaly (VERTT13) cost term.
        disable_c1 (bool): If True, the path may not start at C1 (starts at C2 instead).
        boost_c2 (float): Multiplicative boost applied to a prediction whose argmax is C2; 0 disables it.
        allow_cervical_skip (bool): If True, allow skipping a class within the cervical region.
        allow_thoracic_skip (bool): If True, allow skipping a class within the thoracic region.
        allow_lumbar_skip (bool): If True, allow skipping a class within the lumbar region.
        punish_multiple_sequence (float): Extra cost for repeating an allowed-multiple class.
        punish_skip_sequence (float): Extra cost for skipping an allowed-skip class.
        punish_skip_at_region_sequence (float): Extra cost for skipping at a region boundary.
        region_gaussian_sigma (float): Gaussian sigma for the region head; 0 disables smoothing.
        vert_gaussian_sigma (float): Gaussian sigma for the vertebra head; 0 disables smoothing.
        vert_gaussian_regionwise (bool): If True, smooth the vertebra head per region.
        vertgrp_gaussian_sigma (float): Gaussian sigma for the vertebra-group head; 0 disables smoothing.
        vertgrp_gaussian_regionwise (bool): If True, smooth the vertebra-group head per region.
        vertrel_column_norm (bool): If True, column-normalize the relative-position matrix.
        vertrel_gaussian_sigma (float): Gaussian sigma used when column-normalizing the relative-position matrix.
        focus_tl_gap (bool): If True, focus on the T11/T13 thoracolumbar gap (reserved for the refinement pass).
        argmax_combined_cost_matrix_instead_of_path_algorithm (bool): If True, take a plain per-instance argmax instead of the
            path search.
        proc_lab_force_no_tl_anomaly (bool): If True, disallow T13 transitional-vertebra anomalies (no T11 skip / no T12 repeat).
        verbose (bool): If True, print the active head weights.

    Returns:
        tuple: ``(fcost, fpath, fpath_post, cost_matrix_list, min_costs_path, args)`` where ``fcost`` is the total path cost,
            ``fpath`` is the raw class path, ``fpath_post`` is the post-processed (1-based, T13-aware) label sequence,
            ``cost_matrix_list`` is the combined cost matrix as a nested list, ``min_costs_path`` is the per-step minimum cost
            path, and ``args`` is a snapshot of the call arguments.

    Raises:
        AssertionError: If a weight is negative, ``visible_w`` exceeds 1, or no vital classification head (VERT/VERTEXACT/
            VERTGRP) is present in the predictions.
    """
    args = locals()
    assert 0 <= visible_w, visible_w  # noqa: SIM300
    assert visible_w <= 1.0, f"visible_w must be <= 1.0, got {visible_w}"
    assert 0 <= vert_w, vert_w  # noqa: SIM300
    assert 0 <= region_w, region_w  # noqa: SIM300
    assert 0 <= vertrel_w, vertrel_w  # noqa: SIM300
    assert 0 <= boost_c2, boost_c2  # noqa: SIM300
    #
    n_vert = len(predictions)
    #
    cost_matrix = np.zeros((n_vert, VERT_CLASSES))
    relative_cost_matrix = np.zeros((n_vert, len(VertRel)))
    visible_chain = prepare_visible(predictions, visible_w)
    # print(visible_chain)

    predict_keys = list(predictions[list(predictions.keys())[0]]["soft"].keys())  # noqa: RUF015
    assert "VERT" in predict_keys or "VERTEXACT" in predict_keys or "VERTEX" in predict_keys or "VERTGRP" in predict_keys, (
        f"No vital classification head found, got {predict_keys}"
    )

    # VertRel normalize over labels
    if "VERTREL" in predict_keys:
        vertrel_matrix = np.asarray([k["soft"]["VERTREL"] for k in predictions.values()])
    else:
        vertrel_matrix = np.zeros((n_vert, len(VertRel)))
    if vertrel_column_norm:
        vertrel_matrix = prepare_vertrel_columns(vertrel_matrix, gaussian_sigma=vertrel_gaussian_sigma)

    if "VERTT13" in predict_keys:
        vertt13_softmax_output = np.asarray([k["soft"]["VERTT13"] for k in predictions.values()])
    else:
        vertt13_softmax_output = np.zeros((n_vert, len(VertT13)))
    vertt13_values = np.multiply(
        -prepare_vertt13_columns(vertt13_softmax_output),
        vertt13_w,
    )

    if verbose:
        print("visible_w", visible_w) if "FULLYVISIBLE" in predict_keys else None
        print("vert_w", vert_w) if "VERT" in predict_keys else None
        print("region_w", region_w) if "REGION" in predict_keys else None
        print("vertrel_w", vertrel_w) if "VERTREL" in predict_keys else None
        print("vertgrp_w", vertgrp_w) if "VERTGRP" in predict_keys else None
        print("vertt13_w", vertt13_w) if "VERTT13" in predict_keys else None
        print("disable_c1", disable_c1)
        print("boost_c2", boost_c2)
        print("allow_cervical_skip", allow_cervical_skip)
        print("region_gaussian_sigma", region_gaussian_sigma) if "VERTREGION" in predict_keys else None
        print("vert_gaussian_sigma", vert_gaussian_sigma) if "VERT" in predict_keys else None
        print("vert_gaussian_regionwise", vert_gaussian_regionwise) if "VERT" in predict_keys else None
        print("vertrel_gaussian_sigma", vertrel_gaussian_sigma) if "VERTREL" in predict_keys else None

    #
    for idx, (_, k) in enumerate(predictions.items()):
        vert_softmax_output = k["soft"]["VERT"] if "VERT" in predict_keys else np.zeros(len(VertExact))
        vert_values = np.multiply(
            prepare_vert(
                vert_softmax_output,
                gaussian_sigma=vert_gaussian_sigma,
                gaussian_regionwise=vert_gaussian_regionwise,
            ),
            vert_w,
        )

        vertgrp_softmax_output = k["soft"]["VERTGRP"] if "VERTGRP" in predict_keys else np.zeros(len(VertGroup))
        vertgrp_values = np.multiply(
            prepare_vertgrp(
                vertgrp_softmax_output,
                gaussian_sigma=vertgrp_gaussian_sigma,
                gaussian_regionwise=vertgrp_gaussian_regionwise,
            ),
            vertgrp_w,
        )

        # if "REGION" in k["soft"] else np.zeros((4, *vert_softmax_output.shape[1:]))
        region_softmax_output = k["soft"]["REGION"] if "REGION" in predict_keys else np.zeros(len(VertRegion))
        region_values = np.multiply(
            prepare_region(
                region_softmax_output,
                gaussian_sigma=region_gaussian_sigma,
            ),
            region_w,
        )

        #
        # add region and vert
        final_vert_pred = np.add(region_values, vert_values)
        final_vert_pred = np.add(final_vert_pred, vertgrp_values)
        # normalize
        final_vert_pred /= np.sum(final_vert_pred) + DIVIDE_BY_ZERO_OFFSET
        # boost c2 if enabled
        if boost_c2 > 0.0 and np.argmax(final_vert_pred) == C2_CLASS_IDX:
            final_vert_pred = np.multiply(final_vert_pred, boost_c2)
        # then multiply with visible factor
        final_vert_pred = np.multiply(final_vert_pred, visible_chain[idx])
        cost_matrix[idx] = final_vert_pred
        # relative gets own matrix

        # vertrel_softmax_output = k["soft"]["VERTREL"] if "VERTREL" in k["soft"] else np.zeros((6, *vert_softmax_output.shape[1:]))
        relative_cost_matrix[idx] = prepare_vertrel(
            vertrel_matrix[idx],
            gaussian_sigma=0.0,
        )
    cost_matrix = np.asarray(cost_matrix)
    # invert rel cost
    relative_cost_matrix = np.multiply(-relative_cost_matrix, vertrel_w)
    # for i in range(len(relative_cost_matrix)):
    #    print(relative_cost_matrix[i])
    #
    if argmax_combined_cost_matrix_instead_of_path_algorithm:
        fcost = 0
        min_costs_path = [[]]
        fpath = list(np.argmax(cost_matrix, axis=1))
    else:
        allow_multiple_at_class = [T12_CLASS_IDX, L5_CLASS_IDX] if not proc_lab_force_no_tl_anomaly else [L5_CLASS_IDX]
        allow_skip_at_class = [T11_CLASS_IDX] if not proc_lab_force_no_tl_anomaly else []
        allow_skip_at_region = []
        if allow_cervical_skip:
            allow_skip_at_region.append(0)
        if allow_thoracic_skip:
            allow_skip_at_region.append(1)
        if allow_lumbar_skip:
            allow_skip_at_region.append(2)
        fcost, fpath, min_costs_path = find_most_probably_sequence(
            # input
            cost_matrix,
            min_start_class=C1_CLASS_IDX if not disable_c1 else C2_CLASS_IDX,
            region_rel_cost=relative_cost_matrix,
            vertt13_cost=vertt13_values,
            invert_cost=True,
            # parameters
            punish_multiple_sequence=punish_multiple_sequence,
            punish_skip_sequence=punish_skip_sequence,
            # no touch
            regions=list(DEFAULT_REGION_STARTS),
            allow_multiple_at_class=allow_multiple_at_class,
            allow_skip_at_class=allow_skip_at_class,
            #
            allow_skip_at_region=allow_skip_at_region,
            punish_skip_at_region_sequence=punish_skip_at_region_sequence,
            verbose=False,
        )
    # post processing
    fpath_post = fpath_post_processing(fpath)
    return fcost, fpath, fpath_post, cost_matrix.tolist(), min_costs_path, args


def fpath_post_processing(fpath) -> list[int]:
    """Post-process a raw 0-based class path into the final 1-based vertebra label sequence.

    Resolves transitional-vertebra anomalies (two consecutive T12 become T12 + T13; a trailing double L5 becomes L5 + L6) and
    shifts every class index by 1 to the final label convention, leaving the special T13 label untouched.

    Args:
        fpath (list[int]): Raw 0-based class path from the cost/path search.

    Returns:
        list[int]: The post-processed 1-based vertebra label sequence (with T13/L6 anomalies applied).
    """
    fpath_post = fpath[:]

    # Two T12 -> T12 + T13
    if VertExact.T12.value in fpath_post:
        tidx = fpath_post.index(VertExact.T12.value)
        if tidx != 0 and fpath_post[tidx - 1] == VertExact.T12.value:
            fpath_post[tidx] = T13_LABEL
        elif tidx != len(fpath_post) - 1 and fpath_post[tidx + 1] == VertExact.T12.value:
            fpath_post[tidx + 1] = T13_LABEL
    # Two L5 -> L5, L6
    if (VertExact.L5.value in fpath_post and len(fpath_post) >= 2) and (
        fpath_post[-1] == VertExact.L5.value and fpath_post[-2] == VertExact.L5.value
    ):
        fpath_post[-1] += 1

    fpath_post = [f + 1 if f != T13_LABEL else T13_LABEL for f in fpath_post]
    return fpath_post


def is_valid_vertebra_sequence(sequence: list[VertExact] | list[int]) -> bool:
    """Check whether a vertebra label sequence is anatomically contiguous top-to-bottom.

    A sequence is valid if each label follows the previous one by exactly 1, or forms one of the allowed transitional jumps at
    the thoracolumbar junction (T13->L1, i.e. 28->20, and T12->L1, i.e. 18->20). ``VertExact`` inputs are first converted via
    :func:`fpath_post_processing`.

    Args:
        sequence (list[VertExact] | list[int]): The vertebra label sequence, either as ``VertExact`` enums or 1-based ints.

    Returns:
        bool: True if the sequence is a valid contiguous vertebra run, otherwise False.
    """
    sequence2: list[int] = fpath_post_processing([s.value for s in sequence]) if isinstance(sequence[0], VertExact) else sequence  # type: ignore
    # must be sequence of vertebrae
    for i in range(1, len(sequence2)):
        if (
            sequence2[i] - sequence2[i - 1] == 1
            or (sequence2[i] == 20 and sequence2[i - 1] == 28)
            or (sequence2[i] == 20 and sequence2[i - 1] == 18)
        ):
            continue
        else:
            return False
    return True
