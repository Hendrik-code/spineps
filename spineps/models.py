import os
from pathlib import Path

from TPTBox import Log_Type, No_Logger

from spineps.seg_enums import Modality
from spineps.seg_model import Segmentation_Model, modeltype2class
from spineps.seg_modelconfig import load_inference_config
from spineps.utils.filepaths import get_mri_segmentor_models_dir, search_path

logger = No_Logger()
logger.override_prefix = "Models"


def get_semantic_model(model_name: str) -> Segmentation_Model:
    """Finds and returns a semantic model by name

    Args:
        model_name (str): _description_

    Returns:
        Segmentation_Model: _description_
    """
    model_name = model_name.lower()
    _modelid2folder_subreg = modelid2folder_semantic()
    if model_name not in _modelid2folder_subreg.keys():
        logger.print(f"Model with name {model_name} does not exist, options are {_modelid2folder_subreg.keys()}")
    return get_segmentation_model(_modelid2folder_subreg[model_name])


def get_instance_model(model_name: str) -> Segmentation_Model:
    """Finds and returns an instance model by name

    Args:
        model_name (str): _description_

    Returns:
        Segmentation_Model: _description_
    """
    model_name = model_name.lower()
    _modelid2folder_vert = modelid2folder_instance()
    if model_name not in _modelid2folder_vert.keys():
        logger.print(f"Model with name {model_name} does not exist, options are {_modelid2folder_vert.keys()}")
    return get_segmentation_model(_modelid2folder_vert[model_name])


_modelid2folder_semantic: dict[str, Path] = None
_modelid2folder_instance: dict[str, Path] = None


def modelid2folder_semantic() -> dict[str, Path]:
    """Returns the dictionary mapping semantic model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    global _modelid2folder_semantic
    if _modelid2folder_semantic is not None:
        return _modelid2folder_semantic
    else:
        return check_available_models(get_mri_segmentor_models_dir())[0]


def modelid2folder_instance() -> dict[str, Path]:
    """Returns the dictionary mapping instance model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    global _modelid2folder_instance
    if _modelid2folder_instance is not None:
        return _modelid2folder_instance
    else:
        return check_available_models(get_mri_segmentor_models_dir())[1]


def check_available_models(models_folder: str | Path, verbose: bool = False) -> tuple[dict[str, Path], dict[str, Path]]:
    """Searches through the specified directories and finds models, sorting them into the dictionaries mapping to instance or semantic models

    Args:
        models_folder (str | Path): The folder to be analyzed for models
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[dict[str, Path], dict[str, Path]]: modelid2folder_semantic, modelid2folder_instance
    """
    logger.print("Check available models...")
    if isinstance(models_folder, str):
        models_folder = Path(models_folder)
    assert models_folder.exists(), f"models_folder {models_folder} does not exist"

    config_paths = search_path(models_folder, query="**/inference_config.json")
    global _modelid2folder_semantic, _modelid2folder_instance
    _modelid2folder_semantic = {}  # id to model_folder
    _modelid2folder_instance = {}  # id to model_folder
    for cp in config_paths:
        model_folder = cp.parent
        model_folder_name = model_folder.name.lower()
        try:
            inference_config = load_inference_config(str(cp))
            if Modality.SEG in inference_config.modalities:
                _modelid2folder_instance[model_folder_name] = model_folder
            else:
                _modelid2folder_semantic[model_folder_name] = model_folder
        except Exception as e:
            logger.print(f"Modelfolder '{model_folder_name}' ignored, caused by '{e}'", Log_Type.STRANGE, verbose=True)
            # raise e  #
    return _modelid2folder_semantic, _modelid2folder_instance


def get_segmentation_model(in_config: str | Path, *args, **kwargs) -> Segmentation_Model:
    """Creates the Model class from given path

    Args:
        in_config (str | Path): Path to the models inference config file

    Returns:
        Segmentation_Model: The returned model
    """
    # if isinstance(in_config, MODELS):
    #    in_dir = filepath_model(in_config.value, model_dir=None)
    # else:
    in_dir = in_config

    if os.path.isdir(str(in_dir)):
        # search for config
        path_search = search_path(in_dir, "**/*inference_config.json")
        assert (
            len(path_search) == 1
        ), f"get_segmentation_model: did not found a singular inference_config.json in {in_dir}/**/*inference_config.json"
        in_dir = path_search[0]
    # else:
    #    base = filepath_model(in_config, model_dir=None)
    #    in_dir = base

    inference_config = load_inference_config(str(in_dir))
    modeltype: type[Segmentation_Model] = modeltype2class(inference_config.modeltype)
    return modeltype(model_folder=in_config, inference_config=inference_config, *args, **kwargs)
