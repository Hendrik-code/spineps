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
    possible_keys = list(_modelid2folder_subreg.keys())
    if len(possible_keys) == 0:
        logger.print(
            "Found no available semantic models. Did you set one up by downloading modelweights and putting them into the folder specified by the env variable or did you want to specify with an absolute path instead?",
            Log_Type.FAIL,
        )
        raise KeyError(model_name)
    if model_name not in possible_keys:
        logger.print(f"Model with name {model_name} does not exist, options are {possible_keys}", Log_Type.FAIL)
        raise KeyError(model_name)
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
    possible_keys = list(_modelid2folder_vert.keys())
    if len(possible_keys) == 0:
        logger.print(
            "Found no available instance models. Did you set one up by downloading modelweights and putting them into the folder specified by the env variable or did you want to specify with an absolute path instead?",
            Log_Type.FAIL,
        )
        raise KeyError(model_name)
    if model_name not in possible_keys:
        logger.print(f"Model with name {model_name} does not exist, options are {possible_keys}", Log_Type.FAIL)
        raise KeyError(model_name)
    return get_segmentation_model(_modelid2folder_vert[model_name])


_modelid2folder_semantic: dict[str, Path] | None = None
_modelid2folder_instance: dict[str, Path] | None = None


def modelid2folder_semantic() -> dict[str, Path]:
    """Returns the dictionary mapping semantic model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    if _modelid2folder_semantic is not None:
        return _modelid2folder_semantic
    else:
        return check_available_models(get_mri_segmentor_models_dir())[0]


def modelid2folder_instance() -> dict[str, Path]:
    """Returns the dictionary mapping instance model ids to their corresponding path

    Returns:
        _type_: _description_
    """
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

    config_paths = search_path(models_folder, query="**/inference_config.json", suppress=True)
    global _modelid2folder_semantic, _modelid2folder_instance  # noqa: PLW0603
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
            logger.print(f"Modelfolder '{model_folder_name}' ignored, caused by '{e}'", Log_Type.STRANGE, verbose=verbose)
            # raise e  #
    if len(config_paths) == 0 or len(_modelid2folder_instance.keys()) == 0 or len(_modelid2folder_semantic.keys()) == 0:
        logger.print(
            "Automatic search for models did not find anything. Did you set the environment variable correctly? Did you download model weights and put them into the specified folder? Ignore this if you specified your model using an absolute path.",
            Log_Type.FAIL,
        )
    return _modelid2folder_semantic, _modelid2folder_instance


def get_segmentation_model(in_config: str | Path, **kwargs) -> Segmentation_Model:
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

    if os.path.isdir(str(in_dir)):  # noqa: PTH112
        # search for config
        path_search = search_path(in_dir, "**/*inference_config.json", suppress=True)
        if len(path_search) == 0:
            logger.print(
                f"get_segmentation_model: did not find a singular inference_config.json in {in_dir}/**/*inference_config.json. Is this the correct folder?",
                Log_Type.FAIL,
            )
            raise FileNotFoundError(f"{in_dir}/**/*inference_config.json")
        assert (
            len(path_search) == 1
        ), f"get_segmentation_model: found more than one inference_config.json in {in_dir}/**/*inference_config.json. Ambigous behavior, please manually correct this by removing one of these.\nFound {path_search}"
        in_dir = path_search[0]
    # else:
    #    base = filepath_model(in_config, model_dir=None)
    #    in_dir = base

    inference_config = load_inference_config(str(in_dir))
    modeltype: type[Segmentation_Model] = modeltype2class(inference_config.modeltype)
    return modeltype(model_folder=in_config, inference_config=inference_config, **kwargs)
