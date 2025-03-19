import os
from pathlib import Path

from TPTBox import Log_Type, No_Logger
from tqdm import tqdm

from spineps.lab_model import VertLabelingClassifier
from spineps.seg_enums import Modality, ModelType, SpinepsPhase
from spineps.seg_model import Segmentation_Model, Segmentation_Model_NNunet, Segmentation_Model_Unet3D
from spineps.utils.auto_download import download_if_missing, instances, labeling, semantic
from spineps.utils.filepaths import get_mri_segmentor_models_dir, search_path
from spineps.utils.seg_modelconfig import load_inference_config

logger = No_Logger()
logger.prefix = "Models"


def get_semantic_model(model_name: str, **kwargs) -> Segmentation_Model:
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
    config_path = _modelid2folder_subreg[model_name]
    if str(config_path).startswith("http"):
        # Resolve HTTP
        config_path = download_if_missing(model_name, config_path, phase=SpinepsPhase.SEMANTIC)
    return get_actual_model(config_path, **kwargs)


def get_instance_model(model_name: str, **kwargs) -> Segmentation_Model:
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
    config_path = _modelid2folder_vert[model_name]
    if str(config_path).startswith("http"):
        # Resolve HTTP
        config_path = download_if_missing(model_name, config_path, phase=SpinepsPhase.INSTANCE)

    return get_actual_model(config_path, **kwargs)


def get_labeling_model(model_name: str, **kwargs) -> VertLabelingClassifier:
    """Finds and returns an instance model by name

    Args:
        model_name (str): _description_

    Returns:
        Segmentation_Model: _description_
    """
    model_name = model_name.lower()
    _modelid2folder_labeling = modelid2folder_labeling()
    possible_keys = list(_modelid2folder_labeling.keys())
    if len(possible_keys) == 0:
        logger.print(
            "Found no available labeling models. Did you set one up by downloading modelweights and putting them into the folder specified by the env variable or did you want to specify with an absolute path instead?",
            Log_Type.FAIL,
        )
        raise KeyError(model_name)
    if model_name not in possible_keys:
        logger.print(f"Model with name {model_name} does not exist, options are {possible_keys}", Log_Type.FAIL)
        raise KeyError(model_name)
    config_path = _modelid2folder_labeling[model_name]
    if str(config_path).startswith("http"):
        # Resolve HTTP
        config_path = download_if_missing(model_name, config_path, phase=SpinepsPhase.LABELING)

    return get_actual_model(config_path, **kwargs)


_modelid2folder_semantic: dict[str, Path | str] | None = None
_modelid2folder_instance: dict[str, Path | str] | None = None
_modelid2folder_labeling: dict[str, Path | str] | None = None


def modelid2folder_semantic() -> dict[str, Path | str]:
    """Returns the dictionary mapping semantic model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    if _modelid2folder_semantic is not None:
        return _modelid2folder_semantic
    else:
        return check_available_models(get_mri_segmentor_models_dir())[0]


def modelid2folder_instance() -> dict[str, Path | str]:
    """Returns the dictionary mapping instance model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    if _modelid2folder_instance is not None:
        return _modelid2folder_instance
    else:
        return check_available_models(get_mri_segmentor_models_dir())[1]


def modelid2folder_labeling() -> dict[str, Path | str]:
    """Returns the dictionary mapping labeling model ids to their corresponding path

    Returns:
        _type_: _description_
    """
    if _modelid2folder_labeling is not None:
        return _modelid2folder_labeling
    else:
        return check_available_models(get_mri_segmentor_models_dir())[2]


def check_available_models(models_folder: str | Path, verbose: bool = False) -> tuple[dict[str, Path | int], dict[str, Path | int]]:
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
    global _modelid2folder_semantic, _modelid2folder_instance, _modelid2folder_labeling  # noqa: PLW0603
    _modelid2folder_semantic = semantic  # id to model_folder
    _modelid2folder_instance = instances  # id to model_folder
    _modelid2folder_labeling = labeling
    for cp in tqdm(config_paths, desc="Checking models"):
        model_folder = cp.parent
        model_folder_name = model_folder.name.lower()
        try:
            inference_config = load_inference_config(str(cp))
            if inference_config.modeltype == ModelType.classifier:
                _modelid2folder_labeling[model_folder_name] = model_folder
            elif Modality.SEG in inference_config.modalities:
                _modelid2folder_instance[model_folder_name] = model_folder
            else:
                _modelid2folder_semantic[model_folder_name] = model_folder
        except Exception as e:
            logger.print(f"Modelfolder '{model_folder_name}' ignored, caused by '{e}'", Log_Type.STRANGE, verbose=verbose)
            # raise e  #

    return _modelid2folder_semantic, _modelid2folder_instance, _modelid2folder_labeling


def modeltype2class(modeltype: ModelType):
    """Maps ModelType to actual Segmentation_Model Subclass

    Args:
        type (ModelType): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if modeltype == ModelType.nnunet:
        return Segmentation_Model_NNunet
    elif modeltype == ModelType.unet:
        return Segmentation_Model_Unet3D
    elif modeltype == ModelType.classifier:
        return VertLabelingClassifier
    else:
        raise NotImplementedError(modeltype)


def get_actual_model(
    in_config: str | Path,
    use_cpu: bool = False,
    **kwargs,
) -> Segmentation_Model | VertLabelingClassifier:
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
                f"get_actual_model: did not find a singular inference_config.json in {in_dir}/**/*inference_config.json. Is this the correct folder?",
                Log_Type.FAIL,
            )
            raise FileNotFoundError(f"{in_dir}/**/*inference_config.json")
        assert (
            len(path_search) == 1
        ), f"get_actual_model: found more than one inference_config.json in {in_dir}/**/*inference_config.json. Ambigous behavior, please manually correct this by removing one of these.\nFound {path_search}"
        in_dir = path_search[0]
    # else:
    #    base = filepath_model(in_config, model_dir=None)
    #    in_dir = base

    inference_config = load_inference_config(str(in_dir))
    modeltype: type[Segmentation_Model] = modeltype2class(inference_config.modeltype)
    return modeltype(model_folder=in_config, inference_config=inference_config, use_cpu=use_cpu, **kwargs)
