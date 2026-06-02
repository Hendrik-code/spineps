"""Discovery, lookup and instantiation of SPINEPS segmentation and labeling models from disk or remote URLs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

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

# Shown when no model of a given kind could be found in the configured models directory.
_NO_MODELS_AVAILABLE_MSG = (
    "Found no available {kind} models. Did you set one up by downloading the model weights and "
    "putting them into the folder specified by the env variable, or did you want to specify an "
    "absolute path instead?"
)


def _get_model_by_name(
    model_name: str,
    modelid2folder: dict[str, Path | str],
    phase: SpinepsPhase,
    kind: str,
    **kwargs,
) -> Segmentation_Model | VertLabelingClassifier:
    """Looks up a model by name in a model-id-to-folder map and instantiates it.

    Shared implementation behind get_semantic_model / get_instance_model / get_labeling_model.
    """
    model_name = model_name.lower()
    possible_keys = list(modelid2folder.keys())
    if len(possible_keys) == 0:
        logger.print(_NO_MODELS_AVAILABLE_MSG.format(kind=kind), Log_Type.FAIL)
        raise KeyError(model_name)
    if model_name not in possible_keys:
        logger.print(f"Model with name {model_name} does not exist, options are {possible_keys}", Log_Type.FAIL)
        raise KeyError(model_name)
    config_path = modelid2folder[model_name]
    if str(config_path).startswith("http"):
        # Resolve HTTP
        config_path = download_if_missing(model_name, config_path, phase=phase)
    return get_actual_model(config_path, **kwargs)


def get_semantic_model(model_name: str, **kwargs) -> Segmentation_Model:
    """Finds and returns a semantic (subregion) model by name.

    Args:
        model_name (str): Id of the semantic model to load (case-insensitive).
        **kwargs: Extra keyword arguments forwarded to the model constructor.

    Returns:
        Segmentation_Model: The instantiated semantic model.

    Raises:
        KeyError: If no model with the given name is available.
    """
    return _get_model_by_name(model_name, modelid2folder_semantic(), SpinepsPhase.SEMANTIC, "semantic", **kwargs)


def get_instance_model(model_name: str, **kwargs) -> Segmentation_Model:
    """Finds and returns an instance (vertebra) model by name.

    Args:
        model_name (str): Id of the instance model to load (case-insensitive).
        **kwargs: Extra keyword arguments forwarded to the model constructor.

    Returns:
        Segmentation_Model: The instantiated instance model.

    Raises:
        KeyError: If no model with the given name is available.
    """
    return _get_model_by_name(model_name, modelid2folder_instance(), SpinepsPhase.INSTANCE, "instance", **kwargs)


def get_labeling_model(model_name: str, **kwargs) -> VertLabelingClassifier:
    """Finds and returns a vertebra-labeling model by name.

    Args:
        model_name (str): Id of the labeling model to load (case-insensitive).
        **kwargs: Extra keyword arguments forwarded to the model constructor.

    Returns:
        VertLabelingClassifier: The instantiated labeling classifier.

    Raises:
        KeyError: If no model with the given name is available.
    """
    return _get_model_by_name(model_name, modelid2folder_labeling(), SpinepsPhase.LABELING, "labeling", **kwargs)


_modelid2folder_semantic: Optional[dict[str, Union[Path, str]]] = None
_modelid2folder_instance: Optional[dict[str, Union[Path, str]]] = None
_modelid2folder_labeling: Optional[dict[str, Union[Path, str]]] = None


def modelid2folder_semantic() -> dict[str, Path | str]:
    """Returns the dictionary mapping semantic model ids to their corresponding path.

    Uses the cached mapping if available, otherwise scans the configured models directory.

    Returns:
        dict[str, Path | str]: Mapping from semantic model id to its folder path or download URL.
    """
    if _modelid2folder_semantic is not None:
        return _modelid2folder_semantic
    else:
        return check_available_models(get_mri_segmentor_models_dir())[0]


def modelid2folder_instance() -> dict[str, Path | str]:
    """Returns the dictionary mapping instance model ids to their corresponding path.

    Uses the cached mapping if available, otherwise scans the configured models directory.

    Returns:
        dict[str, Path | str]: Mapping from instance model id to its folder path or download URL.
    """
    if _modelid2folder_instance is not None:
        return _modelid2folder_instance
    else:
        return check_available_models(get_mri_segmentor_models_dir())[1]


def modelid2folder_labeling() -> dict[str, Path | str]:
    """Returns the dictionary mapping labeling model ids to their corresponding path.

    Uses the cached mapping if available, otherwise scans the configured models directory.

    Returns:
        dict[str, Path | str]: Mapping from labeling model id to its folder path or download URL.
    """
    if _modelid2folder_labeling is not None:
        return _modelid2folder_labeling
    else:
        return check_available_models(get_mri_segmentor_models_dir())[2]


def check_available_models(
    models_folder: str | Path, verbose: bool = False
) -> tuple[dict[str, Path | str], dict[str, Path | str], dict[str, Path | str]]:
    """Searches the given directory for models and sorts them into semantic, instance and labeling id-to-folder maps.

    Recursively finds all inference_config.json files, loads each config and assigns the model to the labeling map
    (classifier), the instance map (segmentation input modality) or the semantic map (everything else). The results are
    cached in module-level globals. Models whose config fails to load are skipped.

    Args:
        models_folder (str | Path): The folder to be analyzed for models.
        verbose (bool, optional): If true, logs models that were skipped because their config could not be loaded.
            Defaults to False.

    Returns:
        tuple[dict[str, Path | str], dict[str, Path | str], dict[str, Path | str]]: The semantic, instance and labeling
            id-to-folder maps.

    Raises:
        AssertionError: If models_folder does not exist.
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
        model_folder_name = model_folder.parent.name.lower() if "nnUNetPlans" in model_folder.name else model_folder.name.lower()
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


def modeltype2class(modeltype: ModelType) -> type:
    """Maps a ModelType to the corresponding model class.

    Args:
        modeltype (ModelType): The model type from the inference config.

    Raises:
        NotImplementedError: If the model type is not supported.

    Returns:
        type: The class to instantiate (Segmentation_Model_NNunet, Segmentation_Model_Unet3D or VertLabelingClassifier).
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
    """Creates and returns the appropriate model from a given inference config path.

    Accepts either a path to an inference_config.json file or a folder containing exactly one such file (searched
    recursively). Loads the config, picks the matching model class and instantiates it.

    Args:
        in_config (str | Path): Path to the model's inference config file, or to a folder containing it.
        use_cpu (bool, optional): If true, runs inference on CPU instead of GPU. Defaults to False.
        **kwargs: Extra keyword arguments forwarded to the model constructor.

    Returns:
        Segmentation_Model | VertLabelingClassifier: The instantiated model.

    Raises:
        FileNotFoundError: If no inference_config.json is found in the given folder.
        AssertionError: If more than one inference_config.json is found in the given folder.
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
        assert len(path_search) == 1, (
            f"get_actual_model: found more than one inference_config.json in {in_dir}/**/*inference_config.json. Ambiguous behavior, please manually correct this by removing one of these.\nFound {path_search}"
        )
        in_dir = path_search[0]
    # else:
    #    base = filepath_model(in_config, model_dir=None)
    #    in_dir = base

    inference_config = load_inference_config(str(in_dir))
    modeltype: type[Segmentation_Model] = modeltype2class(inference_config.modeltype)
    return modeltype(model_folder=in_config, inference_config=inference_config, use_cpu=use_cpu, **kwargs)
