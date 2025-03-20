import sys
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from TPTBox import NII, Location, Log_Type, No_Logger

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
from spineps.utils.find_min_cost_path import find_most_probably_sequence

logger = No_Logger(prefix="LabelingPhase")

VERT_CLASSES = 24
CERV = slice(None, 7)  # 0 to 7
THOR = slice(7, 19)  # 7 to 18
LUMB = slice(19, None)  # 19 to end (23)

DIVIDE_BY_ZERO_OFFSET = 1e-8


def perform_labeling_step(model: VertLabelingClassifier, img_nii: NII, vert_nii: NII, subreg_nii: NII | None = None):
    model.load()

    if subreg_nii is not None:
        # crop for corpus instead of whole vertebra
        corpus_nii = subreg_nii.extract_label((Location.Vertebra_Corpus, Location.Vertebra_Corpus_border))
        vert_nii_c = vert_nii * corpus_nii
    # run model
    labelmap = run_model_for_vert_labeling(model, img_nii, vert_nii_c)[0]
    # TODO make all vertebrae without visible corpus to visibility 0 but take into account for labeling
    for i in vert_nii.unique():
        if i not in labelmap:
            labelmap[i] = 0

    # relabel according to labelmap
    return vert_nii.map_labels_(labelmap)


def run_model_for_vert_labeling(
    model: VertLabelingClassifier,
    img_nii: NII,
    vert_nii: NII,
    verbose: bool = False,
):
    # reorient
    img = img_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    vert = vert_nii.reorient(model.inference_config.model_expected_orientation, verbose=False)
    # zms_pir = img.zoom
    # rescale
    # img.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    # vert.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=False)
    #
    img.assert_affine(other=vert)
    # extract vertebrae
    vert.extract_label_([i for i in range(1, 29) if i not in [26, 27]], keep_label=True)
    # counted label
    orig_label = vert.unique()
    # run model
    predictions = model.run_all_seg_instances(img, vert)

    fcost, fpath, fpath_post, costlist, min_costs_path, args = find_vert_path_from_predictions(
        predictions=predictions,
        verbose=verbose,
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
    verbose: bool = False,
):
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
    fcost, fpath, fpath_post, costlist, min_costs_path, args = find_vert_path_from_predictions(
        predictions=predictions,
        verbose=verbose,
        disable_c1=disable_c1,
        boost_c2=boost_c2,
        allow_cervical_skip=allow_cervical_skip,
    )
    assert len(orig_label) == len(fpath_post), f"{len(orig_label)} != {len(fpath_post)}"
    labelmap = {orig_label[idx]: fpath_post[idx] for idx in range(len(orig_label))}

    return labelmap, fcost, fpath, fpath_post, costlist, min_costs_path, predictions


def region_to_vert(region_softmax_values: np.ndarray):  # shape(1,3)
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
):
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
):
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


def prepare_visible(predictions: dict, visible_w: float = 1.0, gaussian_sigma: float = 0.8, gaussian_radius: int = 2):
    # has soft and FULLYVISIBLE key
    predict_keys = list(predictions[list(predictions.keys())[0]]["soft"].keys())  # noqa: RUF015
    if "FULLYVISIBLE" in predict_keys:
        visible_chain = np.asarray([k["soft"]["FULLYVISIBLE"][1] for v, k in predictions.items()])
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


def prepare_region(region_softmax_values: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1):
    softmax_values = region_to_vert(region_softmax_values)
    if gaussian_sigma > 0.0 and np.sum(softmax_values) > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def prepare_vertrel_columns(vertrel_matrix: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1):
    for i in range(1, min(len(VertRel), vertrel_matrix.shape[1])):
        if gaussian_sigma > 0.0 and np.sum(vertrel_matrix) > 0.0:
            vertrel_matrix[:, i] = gaussian_filter1d(vertrel_matrix[:, i], sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
        # normalize per column / label in this case
        vertrel_matrix[:, i] = vertrel_matrix[:, i] / (np.sum(vertrel_matrix[:, i]) + DIVIDE_BY_ZERO_OFFSET)
    return vertrel_matrix


def prepare_vertt13_columns(vertt13_matrix: np.ndarray):
    for i in range(1, min(len(VertT13), vertt13_matrix.shape[1])):
        # normalize per column / label in this case
        vertt13_matrix[:, i] = vertt13_matrix[:, i] / (np.sum(vertt13_matrix[:, i]) + DIVIDE_BY_ZERO_OFFSET)
    return vertt13_matrix


def prepare_vertrel(vertrel_softmax_values: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1):
    softmax_values = vertrel_softmax_values.copy()
    if gaussian_sigma > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    # softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def find_vert_path_from_predictions(
    predictions,
    visible_w: float = 1.0,
    vert_w: float = 0.9,  # 0.9
    vertgrp_w: float = 0.8,
    region_w: float = 1.1,  # 1.1
    vertrel_w: float = 0.3,  # 0.3
    vertt13_w: float = 0.4,
    disable_c1: bool = True,
    boost_c2: float = 1.0,  # 3.0
    allow_cervical_skip: bool = False,
    #
    punish_multiple_sequence: float = 0.0,
    punish_skip_sequence: float = 0.0,
    #
    region_gaussian_sigma: float = 0.0,  # 0 means no gaussian
    vert_gaussian_sigma: float = 0.8,  # 0.8 0 means no gaussian
    vert_gaussian_regionwise: bool = True,
    vertgrp_gaussian_sigma: float = 0.8,  # 0.8 0 means no gaussian
    vertgrp_gaussian_regionwise: bool = True,
    vertrel_column_norm: bool = True,
    vertrel_gaussian_sigma: float = 0.6,  # 0.6 # 0 means no gaussian
    #
    argmax_combined_cost_matrix_instead_of_path_algorithm: bool = False,
    #
    verbose: bool = False,
):
    args = locals()
    assert 0 <= visible_w, visible_w  # noqa: SIM300
    assert 0 <= vert_w, vert_w  # noqa: SIM300
    assert 0 <= region_w, region_w  # noqa: SIM300
    assert 0 <= vertrel_w, vertrel_w  # noqa: SIM300
    assert 0 <= boost_c2, boost_c2  # noqa: SIM300
    #
    n_vert = len(predictions)
    #
    cost_matrix = np.zeros((n_vert, 24))  # TODO 24 fix?
    relative_cost_matrix = np.zeros((n_vert, 6))  # TODO 6 fix?
    visible_chain = prepare_visible(predictions, visible_w)

    predict_keys = list(predictions[list(predictions.keys())[0]]["soft"].keys())  # noqa: RUF015
    assert (
        "VERT" in predict_keys or "VERTEXACT" in predict_keys or "VERTGRP" in predict_keys
    ), f"No vital classification head found, got {predict_keys}"

    # VertRel normalize over labels
    if "VERTREL" in predict_keys:
        vertrel_matrix = np.asarray([k["soft"]["VERTREL"] for v, k in predictions.items()])
    else:
        vertrel_matrix = np.zeros((n_vert, len(VertRel)))
    if vertrel_column_norm:
        vertrel_matrix = prepare_vertrel_columns(vertrel_matrix, gaussian_sigma=vertrel_gaussian_sigma)

    if "VERTT13" in predict_keys:
        vertt13_softmax_output = np.asarray([k["soft"]["VERTT13"] for v, k in predictions.items()])
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
        if boost_c2 > 0.0 and np.argmax(final_vert_pred) == 1:
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
        fcost, fpath, min_costs_path = find_most_probably_sequence(
            # input
            cost_matrix,
            min_start_class=0 if not disable_c1 else 1,
            region_rel_cost=relative_cost_matrix,
            vertt13_cost=vertt13_values,
            invert_cost=True,
            # parameters
            punish_multiple_sequence=punish_multiple_sequence,
            punish_skip_sequence=punish_skip_sequence,
            # no touch
            regions=[0, 7, 19],
            allow_multiple_at_class=[18, 23],  # T12 and L5
            allow_skip_at_class=[17],  # T11
            #
            allow_skip_at_region=[0] if allow_cervical_skip else [],
            punish_skip_at_region_sequence=0.2 if allow_cervical_skip else 0.0,
        )
    # post processing
    fpath_post = fpath_post_processing(fpath)
    return fcost, fpath, fpath_post, cost_matrix.tolist(), min_costs_path, args


def fpath_post_processing(fpath):
    fpath_post = fpath[:]

    # Two T12 -> T12 + T13
    if VertExact.T12.value in fpath_post:
        tidx = fpath_post.index(VertExact.T12.value)
        if tidx != 0 and fpath_post[tidx - 1] == VertExact.T12.value:
            fpath_post[tidx] = 28
        elif tidx != len(fpath_post) - 1 and fpath_post[tidx + 1] == VertExact.T12.value:
            fpath_post[tidx + 1] = 28
    # Two L5 -> L5, L6
    if (VertExact.L5.value in fpath_post and len(fpath_post) >= 2) and (
        fpath_post[-1] == VertExact.L5.value and fpath_post[-2] == VertExact.L5.value
    ):
        fpath_post[-1] += 1

    fpath_post = [f + 1 if f != 28 else 28 for f in fpath_post]
    return fpath_post


def is_valid_vertebra_sequence(sequence: list[VertExact] | list[int]) -> bool:
    if isinstance(sequence[0], VertExact):
        sequence = fpath_post_processing([s.value for s in sequence])
    # must be sequence of vertebrae
    for i in range(1, len(sequence)):
        if (
            sequence[i] - sequence[i - 1] == 1
            or (sequence[i] == 20 and sequence[i - 1] == 28)
            or (sequence[i] == 20 and sequence[i - 1] == 18)
        ):
            continue
        else:
            return False
    return True
