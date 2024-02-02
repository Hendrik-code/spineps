# from utils.predictor import nnUNetPredictor
from TPTBox import NII, Location, No_Logger, Zooms
from TPTBox.core import poi
from TPTBox.logger.log_file import get_time, format_time_short
from spineps.seg_model import Segmentation_Model
import subprocess


logger = No_Logger()
logger.override_prefix = "SPINEPS"

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
]


def predict_centroids_from_both(
    vert_nii_cleaned: NII,
    seg_nii: NII,
    models: list[Segmentation_Model],
    input_zms_pir: Zooms | None = None,
):
    """Calculates the centroids of each vertebra corpus by using both semantic and instance mask

    Args:
        vert_nii_cleaned (NII): _description_
        seg_nii (NII): _description_
        models (list[Segmentation_Model]): _description_
        input_zms_pir (Zooms | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    vert_nii_4_centroids = vert_nii_cleaned.copy()
    labelmap = {i: 0 for i in range(100, 134)}
    for i in range(200, 234):
        labelmap[i] = 0
    vert_nii_4_centroids.map_labels_(labelmap, verbose=False)

    ctd = poi.calc_centroids_from_subreg_vert(vert_nii_4_centroids, seg_nii, verbose=logger)

    models_repr = {}
    for idx, m in enumerate(models):
        models_repr[idx] = m.dict_representation(input_zms_pir)
    ctd.info["source"] = "MRI Segmentation Pipeline"
    ctd.info["version"] = pipeline_version()
    ctd.info["models"] = models_repr
    ctd.info["revision"] = pipeline_revision()
    ctd.info["timestamp"] = format_time_short(get_time())
    return ctd


# TODO make automatic version of this repo (below is the repo the code is called from... -.-)


def pipeline_version():
    try:
        label = subprocess.check_output(["git", "rev-list", "--count", "main"]).strip()
        label = str(label).replace("'", "")
        while not label[0].isdigit():
            label = label[1:]
    except Exception as e:
        return "Version not found"
    return "v1." + str(label)


def pipeline_revision():
    label = ""
    rev = ""
    try:
        label = subprocess.check_output(["git", "describe", "--always"]).strip()
    except Exception as e:
        pass
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        pass
    return str(label) + "::" + str(rev)


if __name__ == "__main__":
    print(pipeline_version())
    print(pipeline_revision())
