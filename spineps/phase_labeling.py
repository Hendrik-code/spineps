import sys
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from TPTBox import NII, Log_Type, No_Logger

from spineps.architectures.read_labels import VertExact
from spineps.lab_model import VertLabelingClassifier
from spineps.utils.find_min_cost_path import find_most_probably_sequence

check_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/code/classifier/lightning_logs/"
model_p = "densenet169_v2_multilabel_img_a14_tr100-101-102_val103_ad3_withFCN/version_0/checkpoints/epoch=1-step=1542-val_f1=0.9791_valf1-weights.ckpt"

logger = No_Logger(prefix="LabelingPhase")

VERT_CLASSES = 24
CERV = slice(None, 7)  # 0 to 7
THOR = slice(7, 19)  # 7 to 18
LUMB = slice(19, None)  # 19 to end (23)


def perform_labeling_step(img_nii: NII, vert_nii: NII):
    model = VertLabelingClassifier.from_checkpoint_path(check_dir + model_p)
    img = img_nii.reorient()
    vert = vert_nii.reorient()
    img.assert_affine(other=vert)

    img.assert_affine(zoom=(0.8571, 0.8571, 3.3))
    # extract vertebrae
    vert.extract_label_(list(range(1, 26)), keep_label=True)
    # counted label
    orig_label = vert.unique()
    # run model
    predictions = model.run_all_seg_instances(img, vert)
    fcost, fpath, fpath_post, costlist = find_vert_path_from_predictions(predictions)  # TODO arguments
    # offset because C1 is 0, not 1 as in mask
    assert len(orig_label) == len(fpath_post), f"{len(orig_label)} != {len(fpath_post)}"
    labelmap = {orig_label[idx]: fpath_post[idx] for idx in range(len(orig_label))}
    # relabel according to fpath_post
    return vert_nii.map_labels_(labelmap)


def region_to_vert(region_softmax_values: np.ndarray):  # shape(1,3)
    vert_prediction_values = np.zeros(VERT_CLASSES)
    vert_prediction_values[CERV] = region_softmax_values[0]
    vert_prediction_values[THOR] = region_softmax_values[1]
    vert_prediction_values[LUMB] = region_softmax_values[2]
    return vert_prediction_values


def prepare_vert(vert_softmax_values: np.ndarray, gaussian_sigma: float = 0.85, gaussian_radius: int = 2):
    # region-wise gaussian?
    softmax_values2 = vert_softmax_values.copy()
    for s in [CERV, THOR, LUMB]:
        softmax_values2[s] = gaussian_filter1d(softmax_values2[s], sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values2 /= np.sum(softmax_values2)
    return softmax_values2


def prepare_visible(predictions: dict, visible_w: float = 1.0, gaussian_sigma: float = 0.8, gaussian_radius: int = 2):
    visible_chain = np.asarray([k["soft"]["FULLYVISIBLE"][1] for v, k in predictions.items()])
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
    softmax_values2 = gaussian_filter1d(softmax_values, sigma=gaussian_sigma, mode="nearest", radius=gaussian_radius)
    softmax_values2 /= np.sum(softmax_values2)
    return softmax_values2


def find_vert_path_from_predictions(
    predictions,
    visible_w: float = 1.0,
    vert_w: float = 1.0,
    region_w: float = 1.0,
    vertrel_w: float = 1.0,
    disable_c1: bool = True,
    boost_c2: bool = True,
    allow_cervical_skip: bool = True,
    #
    punish_multiple_sequence: float = 0.0,
    punish_skip_sequence: float = 0.0,
    # TODO all the parameters (gaussian sigmas and radius? region-wise vert or not)
    # TODO preprocess vertrel so that if gap between 3/4 is linear interpolated
):
    assert 0 <= visible_w, visible_w  # noqa: SIM300
    assert 0 <= vert_w, vert_w  # noqa: SIM300
    assert 0 <= region_w, region_w  # noqa: SIM300
    assert 0 <= vertrel_w, vertrel_w  # noqa: SIM300
    cost_matrix = np.zeros((len(predictions), 24))  # TODO 24 fix?
    relative_cost_matrix = np.zeros((len(predictions), 6))  # TODO 6 fix?
    visible_chain = prepare_visible(predictions, visible_w)
    # print(visible_chain)
    #
    for idx, (_, k) in enumerate(predictions.items()):
        region_values = np.multiply(prepare_region(k["soft"]["REGION"]), region_w)
        vert_values = np.multiply(prepare_vert(k["soft"]["VERT"]), vert_w)
        #
        # add region and vert
        final_vert_pred = np.add(region_values, vert_values)
        # normalize
        final_vert_pred /= np.sum(final_vert_pred)
        # boost c2 if enabled
        if boost_c2 and np.argmax(final_vert_pred) == 1:
            final_vert_pred = np.multiply(final_vert_pred, 3.0)
        # then multiply with visible factor
        final_vert_pred = np.multiply(final_vert_pred, visible_chain[idx])
        cost_matrix[idx] = final_vert_pred
        # relative gets own matrix
        relative_cost_matrix[idx] = k["soft"]["VERTREL"]
    cost_matrix = np.asarray(cost_matrix)
    # invert rel cost
    relative_cost_matrix = np.multiply(-relative_cost_matrix, vertrel_w)
    #
    fcost, fpath = find_most_probably_sequence(
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
    return fcost, fpath, fpath_post, cost_matrix.tolist()


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
