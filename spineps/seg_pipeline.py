"""Segmentation-pipeline helpers: shared logger, subregion label sets, centroid computation, and pipeline version reporting."""

from __future__ import annotations

# from utils.predictor import nnUNetPredictor
import subprocess
from typing import Any

from scipy.ndimage import center_of_mass
from TPTBox import NII, ZOOMS, Location, No_Logger, v_name2idx
from TPTBox.core import poi
from TPTBox.logger.log_file import format_time_short, get_time

from spineps.seg_model import SegmentationModel

logger = No_Logger(prefix="SPINEPS")

# IVD and endplate instances are stored as (vertebra label + offset). These offsets are the canonical
# home for the convention; other modules import them from here.
IVD_LABEL_OFFSET = 100
ENDPLATE_LABEL_OFFSET = 200
# Number of derived label ids reserved per type; the ranges below cover all IVD/endplate labels and are
# stripped before centroid computation.
_MAX_DERIVED_LABELS_PER_TYPE = 34
IVD_LABEL_RANGE = range(IVD_LABEL_OFFSET, IVD_LABEL_OFFSET + _MAX_DERIVED_LABELS_PER_TYPE)
ENDPLATE_LABEL_RANGE = range(ENDPLATE_LABEL_OFFSET, ENDPLATE_LABEL_OFFSET + _MAX_DERIVED_LABELS_PER_TYPE)

fill_holes_labels = [
    Location.Vertebra_Corpus_border.value,
    Location.Spinal_Canal.value,
    Location.Vertebra_Disc.value,
    Location.Spinal_Cord.value,
]

vertebra_subreg_labels = [
    Location.Vertebra_Full.value,
    Location.Arcus_Vertebrae.value,
    Location.Spinosus_Process.value,
    Location.Costal_Process_Left.value,
    Location.Costal_Process_Right.value,
    Location.Superior_Articular_Left.value,
    Location.Superior_Articular_Right.value,
    Location.Inferior_Articular_Left.value,
    Location.Inferior_Articular_Right.value,
    Location.Vertebra_Corpus_border.value,
    Location.Vertebra_Corpus.value,
    Location.Dens_axis.value,
]


def predict_centroids_from_both(
    vert_nii_cleaned: NII,
    seg_nii: NII,
    models: list[SegmentationModel | None],
    parameter: dict[str, Any],
) -> poi.POI:
    """Calculate the centroids of each vertebra corpus using both the semantic and instance masks.

    Strips the IVD and endplate derived instance labels from the instance mask, computes the per-vertebra centroids from the
    instance and semantic masks, adds an S1 corpus centroid when sacrum is present, and records pipeline metadata (model
    descriptions, version, revision, timestamp, and the given parameters) on the result.

    Args:
        vert_nii_cleaned (NII): Cleaned vertebra instance segmentation mask.
        seg_nii (NII): Subregion semantic segmentation mask.
        models (list[SegmentationModel | None]): Models used in the pipeline, recorded in the centroid metadata.
        parameter (dict[str, Any]): Pipeline parameters to record on the centroid metadata.

    Returns:
        POI: The computed point-of-interest / centroid object with pipeline metadata attached.
    """
    vert_nii_4_centroids = vert_nii_cleaned.copy()
    labelmap = dict.fromkeys([*IVD_LABEL_RANGE, *ENDPLATE_LABEL_RANGE], 0)
    vert_nii_4_centroids.map_labels_(labelmap, verbose=False)

    ctd = poi.calc_poi_from_subreg_vert(vert_nii_4_centroids, seg_nii, verbose=logger)

    if v_name2idx["S1"] in vert_nii_cleaned.unique():
        s1_nii = vert_nii_cleaned.extract_label(v_name2idx["S1"], inplace=False)
        ctd[v_name2idx["S1"], 50] = center_of_mass(s1_nii.get_seg_array())

    models_repr = {}
    for idx, m in enumerate(models):
        if m is not None:
            models_repr[idx] = m.dict_representation()
        else:
            models_repr[idx] = {"name": "No Model"}
    ctd.info["source"] = "MRI Segmentation Pipeline"
    ctd.info["version"] = pipeline_version()
    ctd.info["models"] = models_repr
    ctd.info["revision"] = pipeline_revision()
    ctd.info["timestamp"] = format_time_short(get_time())
    for pname, pvalue in parameter.items():
        ctd.info[pname] = str(pvalue)
    return ctd


def pipeline_version() -> str:
    """Return the pipeline version string derived from the git commit count on ``main``.

    Returns:
        str: A version like ``"v1.<commit-count>"``, or ``"Version not found"`` if git is unavailable.
    """
    try:
        label = subprocess.check_output(["git", "rev-list", "--count", "main"]).strip()
        label = str(label).replace("'", "")
        while not label[0].isdigit():
            label = label[1:]
    except Exception:
        return "Version not found"
    return "v1." + str(label)


def pipeline_revision() -> str:
    """Return the current git revision string for the pipeline.

    Returns:
        str: ``"<git-describe>::<full-commit-hash>"``; either part is empty if the corresponding git call fails.
    """
    label = ""
    rev = ""
    try:
        label = subprocess.check_output(["git", "describe", "--always"]).strip()
    except Exception:
        pass
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        pass
    return str(label) + "::" + str(rev)


if __name__ == "__main__":
    print(pipeline_version())
    print(pipeline_revision())
