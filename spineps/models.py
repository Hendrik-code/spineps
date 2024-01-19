import os
from BIDS import No_Logger, Log_Type
from spineps.seg_enums import Enum_Compare, Modality
from spineps.utils.filepaths import search_path, get_mri_segmentor_models_dir, filepath_model

from pathlib import Path
from spineps.seg_model import Segmentation_Model, modeltype2class
from spineps.seg_modelconfig import load_inference_config


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
            model = get_segmentation_model(in_config=cp, default_verbose=False)
            if Modality.SEG in model.inference_config.modalities:
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


####################
# Saved models and their direct paths for development
####################


class MODELS(Enum_Compare):
    def __call__(self, *args, **kwargs) -> "Segmentation_Model":
        model_dir = "/DATA/NAS/ongoing_projects/hendrik/nako-segmentation/nnUNet/"
        return get_segmentation_model(in_config=filepath_model(self.value, model_dir=model_dir), *args, **kwargs)

    ###
    ### Base models (zero or first phase models)
    ###
    # Rostock segmentation
    # SHIP T1w 2 spine
    T1w_SEGMENTOR = "Dataset031_ship-t1-combined2/nnUNetTrainer__nnUNetPlans__3d_fullres_custom"
    # SHIP Vibe 2 spine
    VIBE_SEGMENTOR = "Dataset040_ship-t1dixon-subreg/nnUNetTrainer__nnUNetPlans__3d_fullres"
    # CT robust 2 spine
    CT_SEGMENTOR = "Dataset051_ct_spinegan_fiso/nnUNetTrainer__nnUNetPlans__3d_fullres_custom2"
    ###
    T2w_NAKOCUT_081 = "Dataset081_nako101_102chunk_cutclean_highres/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    T2w_NAKOCUT_110 = "Dataset110_nako102_cutclean_spiderarcus_hf_def/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    ###
    ### Spider
    ###
    # Spider untouched with untouched nnunet base (bad)
    SPIDER_NNBASE = "Dataset060_spideruntouched/nnUNetTrainer__nnUNetPlans__3d_fullres"
    # Spider train 2 spider structures
    SPIDER_FIRST_mer = "Dataset102_spider_subreg_aug_mer/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    SPIDER_FIRST_mer_highres = "Dataset107_spider_subreg_aug_mer_highres_normalized/nnUNetTrainer__nnUNetPlans__3d_fullres_custom/"

    # Paper For Test Sets:
    # Test Spider
    # SPIDER_NNBASE for spider
    # SPIDER_FIRST_mer for SPINEPS
    # Test GNC
    T2w_TEST_NAKO = "Dataset120_nako_cutclean_highres_wo_test/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    T2w_TEST_NAKOARCUS = "Dataset122_nako_wotest_spiderarcus/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    T2w_TEST_NAKOINFERENCE = "Dataset121_nako_inference_highres_wo_test/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    T2w_TEST_NAKOINFERENCEARCUS = "Dataset124_nako_inference_highres_wo_test/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    # T2w_TEST_NakoPlusSpiderPlus = "Dataset133_nako_plus_spider_annotation_plus/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    T2w_TEST_NakoPlusSpiderPlus2 = "Dataset137_nako_spider_annotation_plus_fix/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom/"
    #

    ###
    ### Second phase models:
    ###
    # Vert Unet in higher (third dim) resolution (still trained on nako)
    VERT_HIGHRES = "highres"
    VERT_SPIDER_MER2 = "spidernew_mer_shiftposi"
    VERT_HIGHRES2 = "nakospider_highres_shiftposi"
    ###
    ### Combination of datasets models (nako+spider, + dixctws, wsgan)
    ###
    # nako + spider 1st attempt in higher (more spider) resolution
    T2w_NAKOSPIDER_HIGHRES = (
        "Dataset073_nakochunk_spider_combined_highres_spider_fixed/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    )
    # high dimension with nakocut, BROKEN SPINAL CORD/CANAL!
    # T2w_NAKOSPIDER_HIGHRES_CUT = "Dataset076_highres_nakocut_spider_rhf/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom <- now called 75_LABELSWITCH
    # ^ fixed spinal cord/canal
    T2w_NAKOSPIDER_HIGHRES_CUT = "Dataset076_highres_nakocut_spider_rhf/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    T2w_NAKOSPIDER_HIGHRES_CUTCLEAN = "Dataset079_highres_nakocutclean_spider_rhf/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    # high dimension with nakocutmuch
    T2w_NAKOSPIDER_HIGHRES_CUTMUCH = "Dataset078_highres_nakocutmuch_spider_rhf/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    #
    T2w_NAKO102CUT_SPIDER_DEF = (
        "Dataset082_nako101_102chunk_cutclean_spider_hf_deform/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
    )
    #
    # Correction approach
    # Correct_Spider = "Dataset090_spider_inference_to_correct/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres_custom"
