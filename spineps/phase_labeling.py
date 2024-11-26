import sys
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from TPTBox import NII, Log_Type, No_Logger

from spineps.architectures.read_labels import VertExact
from spineps.get_models import get_actual_model
from spineps.lab_model import VertLabelingClassifier
from spineps.utils.find_min_cost_path import find_most_probably_sequence

check_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/code/classifier/lightning_logs/"
model_p = "densenet169_v2_multilabel_img_a17_tr100-101-102-103-104_valtr_ad3_withFCN/version_0"
# "densenet169_v2_multilabel_img_a17_tr100-101-102-103-104_valtr_ad3_withFCN/version_0/"

logger = No_Logger(prefix="LabelingPhase")

VERT_CLASSES = 24
CERV = slice(None, 7)  # 0 to 7
THOR = slice(7, 19)  # 7 to 18
LUMB = slice(19, None)  # 19 to end (23)

DIVIDE_BY_ZERO_OFFSET = 1e-6


def perform_labeling_step(model: VertLabelingClassifier, img_nii: NII, vert_nii: NII):
    if model is None:
        model = get_actual_model(
            in_config=Path(check_dir + model_p),
        )
        model.load()
    # run model
    labelmap = run_model_for_vert_labeling(model, img_nii, vert_nii)[0]

    # relabel according to labelmap
    return vert_nii.map_labels_(labelmap)


def run_model_for_vert_labeling(
    model: VertLabelingClassifier,
    img_nii: NII,
    vert_nii: NII,
    verbose: bool = False,
):
    # reorient
    img = img_nii.reorient(model.inference_config.model_expected_orientation, verbose=logger)
    vert = vert_nii.reorient(model.inference_config.model_expected_orientation, verbose=logger)
    zms_pir = img.zoom
    # rescale
    img.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=logger)
    vert.rescale_(model.calc_recommended_resampling_zoom(zms_pir), verbose=logger)
    #
    img.assert_affine(other=vert)
    # extract vertebrae
    vert.extract_label_(list(range(1, 26)), keep_label=True)
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


def prepare_visible(predictions: dict, visible_w: float = 1.0, gaussian_sigma: float = 0.8, gaussian_radius: int = 2):
    visible_chain = np.asarray([k["soft"]["FULLYVISIBLE"][1] for v, k in predictions.items()])
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
    if gaussian_sigma > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def prepare_vertrel(vertrel_softmax_values: np.ndarray, gaussian_sigma: float = 0.75, gaussian_radius: int = 1):
    softmax_values = vertrel_softmax_values.copy()
    if gaussian_sigma > 0.0:
        softmax_values = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values /= np.sum(softmax_values) + DIVIDE_BY_ZERO_OFFSET
    return softmax_values


def find_vert_path_from_predictions(
    predictions,
    visible_w: float = 1.0,
    vert_w: float = 0.75,
    region_w: float = 0.75,
    vertrel_w: float = 0.5,
    disable_c1: bool = True,
    boost_c2: float = 3.0,
    allow_cervical_skip: bool = True,
    #
    punish_multiple_sequence: float = 0.0,
    punish_skip_sequence: float = 0.0,
    #
    region_gaussian_sigma: float = 0.0,  # 0 means no gaussian
    vert_gaussian_sigma: float = 1.0,  # 0 means no gaussian
    vert_gaussian_regionwise: bool = True,
    vertrel_gaussian_sigma: float = 0.75,  # 0 means no gaussian
    # TODO preprocess vertrel so that if gap between 3/4 is linear interpolated
    #
    verbose: bool = False,
):
    args = locals()
    assert 0 <= visible_w, visible_w  # noqa: SIM300
    assert 0 <= vert_w, vert_w  # noqa: SIM300
    assert 0 <= region_w, region_w  # noqa: SIM300
    assert 0 <= vertrel_w, vertrel_w  # noqa: SIM300
    assert 0 <= boost_c2, boost_c2  # noqa: SIM300
    cost_matrix = np.zeros((len(predictions), 24))  # TODO 24 fix?
    relative_cost_matrix = np.zeros((len(predictions), 6))  # TODO 6 fix?
    visible_chain = prepare_visible(predictions, visible_w)
    # print(visible_chain)
    if verbose:
        print("visible_w", visible_w)
        print("vert_w", vert_w)
        print("region_w", region_w)
        print("vertrel_w", vertrel_w)
        print("disable_c1", disable_c1)
        print("boost_c2", boost_c2)
        print("allow_cervical_skip", allow_cervical_skip)
        print("region_gaussian_sigma", region_gaussian_sigma)
        print("vert_gaussian_sigma", vert_gaussian_sigma)
        print("vert_gaussian_regionwise", vert_gaussian_regionwise)
        print("vertrel_gaussian_sigma", vertrel_gaussian_sigma)

    #
    for idx, (_, k) in enumerate(predictions.items()):
        region_values = np.multiply(
            prepare_region(
                k["soft"]["REGION"],
                gaussian_sigma=region_gaussian_sigma,
            ),
            region_w,
        )
        vert_values = np.multiply(
            prepare_vert(
                k["soft"]["VERT"],
                gaussian_sigma=vert_gaussian_sigma,
                gaussian_regionwise=vert_gaussian_regionwise,
            ),
            vert_w,
        )
        #
        # add region and vert
        final_vert_pred = np.add(region_values, vert_values)
        # normalize
        final_vert_pred /= np.sum(final_vert_pred) + DIVIDE_BY_ZERO_OFFSET
        # boost c2 if enabled
        if boost_c2 > 0.0 and np.argmax(final_vert_pred) == 1:
            final_vert_pred = np.multiply(final_vert_pred, boost_c2)
        # then multiply with visible factor
        final_vert_pred = np.multiply(final_vert_pred, visible_chain[idx])
        cost_matrix[idx] = final_vert_pred
        # relative gets own matrix
        relative_cost_matrix[idx] = prepare_vertrel(
            k["soft"]["VERTREL"],
            gaussian_sigma=vertrel_gaussian_sigma,
        )
    cost_matrix = np.asarray(cost_matrix)
    # invert rel cost
    relative_cost_matrix = np.multiply(-relative_cost_matrix, vertrel_w)
    #
    fcost, fpath, min_costs_path = find_most_probably_sequence(
        # input
        cost_matrix,
        min_start_class=0 if not disable_c1 else 1,
        region_rel_cost=relative_cost_matrix,
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
