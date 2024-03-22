# from utils.predictor import nnUNetPredictor
import nibabel as nib
from TPTBox import BIDS_FILE, NII, Log_Type, Zooms

from spineps.seg_enums import Acquisition, Modality
from spineps.seg_model import Segmentation_Model
from spineps.seg_pipeline import logger

Modality_Pair = tuple[list[Modality] | Modality, Acquisition]


class InputPackage:
    def __init__(self, mri_nii: NII, pad_size: int = 4) -> None:
        self._zms = mri_nii.zoom
        self._affine = mri_nii.affine
        self._header = mri_nii.header
        self._orientation = mri_nii.orientation
        self._shape = mri_nii.shape
        self.zms_pir = mri_nii.reorient().zoom
        self.pad_size = pad_size

    def sample_to_this(self, other_nii: NII, intermediate_nii: NII | None = None) -> NII:
        other_nii = other_nii.copy()
        other_nii.assert_affine(orientation=("P", "I", "R"))

        if intermediate_nii is not None:
            intermediate_nii.assert_affine(orientation=("P", "I", "R"))
            other_nii.rescale_(voxel_spacing=intermediate_nii.zoom, verbose=logger).reorient_(intermediate_nii.orientation, verbose=logger)
        other_nii.rescale_(voxel_spacing=self.zms_pir, verbose=logger).reorient_(self._orientation, verbose=logger)
        # other_nii.reorient_(self._orientation, verbose=logger).rescale_(voxel_spacing=self._zms, verbose=logger)
        if self.pad_size > 0:
            arr = other_nii.get_array()
            arr = arr[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]
            other_nii.set_array_(arr)
        other_nii.pad_to(self._shape, inplace=True)
        assert_true = other_nii.assert_affine(
            zoom=self._zms, orientation=self._orientation, shape=self._shape, raise_error=False, verbose=logger
        )
        assert assert_true, "sampled back to input did not meet affine criteria"
        return other_nii

    def make_nii_from_this(self, other_nii: NII) -> NII:
        other_nii.assert_affine(shape=self._shape, orientation=self._orientation, zoom=self._zms)
        other_nii.nii = nib.nifti1.Nifti1Image(other_nii.get_seg_array(), affine=self._affine, header=self._header)
        return other_nii

    def __str__(self) -> str:
        return f"Input image, {self._zms}, {self._orientation}, {self._shape}"


def find_best_matching_model(
    modality_pair: Modality_Pair,
    expected_resolution: Zooms | None,  # actual resolution here?
) -> Segmentation_Model:
    raise NotImplementedError("find_best_matching_model()")
    logger.print(expected_resolution)
    # TODO replace with automatic going through model configs to find best matching the resolution
    mapping: dict = {
        # (Modality.CT, Acquisition.sag): MODELS.CT_SEGMENTOR,
        # (Modality.T2w, Acquisition.sag): MODELS.T2w_NAKOSPIDER_HIGHRES,
        # (Modality.T1w, Acquisition.sag): MODELS.T1w_SEGMENTOR,
        # (Modality.Vibe, Acquisition.ax): MODELS.VIBE_SEGMENTOR,
        # (Modality.SEG, Acquisition.sag): MODELS.VERT_HIGHRES,
    }
    if isinstance(modality_pair[0], list) and len(modality_pair[0]) == 1:
        modality_pair = (modality_pair[0][0], modality_pair[1])
    if modality_pair not in mapping:
        raise NotImplementedError(str(modality_pair[0]), str(modality_pair[1]))
    else:
        return mapping[modality_pair]


def check_model_modality_acquisition(
    model: Segmentation_Model,
    mod_pair: Modality_Pair,
    verbose: bool = True,
):
    """Checks if a model is compatible with a specified Modality_Pair

    Args:
        model (Segmentation_Model): _description_
        mod_pair (Modality_Pair): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    compatible = True

    model_modalities = model.modalities()
    model_acquisition = model.acquisition()

    expected_modalities = mod_pair[0]
    expected_acquisition = mod_pair[1]

    if not isinstance(expected_modalities, list):
        expected_modalities = [expected_modalities]

    logger_texts = f"{mod_pair}: model incompatible"

    for m in expected_modalities:
        if m not in model_modalities:
            compatible = False
            logger_texts += f", model modalities {model_modalities}"

    if expected_acquisition != model_acquisition:
        compatible = False
        logger_texts += f", model acquisition {model_acquisition}"

    if not compatible:
        logger.print(logger_texts, Log_Type.WARNING, verbose=verbose)
    return compatible


def check_input_model_compatibility(
    img_ref: BIDS_FILE,
    model: Segmentation_Model,
    ignore_modality: bool = False,
    ignore_acquisition: bool = False,
    verbose: bool = True,
) -> bool:
    """Checks if a model is compatible with a specified input

    Args:
        img_ref (BIDS_FILE): _description_
        model (Segmentation_Model): _description_
        ignore_modality (bool, optional): _description_. Defaults to False.
        ignore_acquisition (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        bool: _description_
    """
    model_modalities = model.modalities()
    model_acquisition = model.acquisition()
    allowed_format = Modality.format_keys(model_modalities)
    allowed_acq = [*Acquisition.format_keys(model_acquisition), "iso"]

    file_dir = img_ref.file["nii.gz"]
    filename = file_dir.name

    compatible = True

    input_format = img_ref.format
    has_seg_key = "seg" in img_ref.info
    has_label_key = "label" in img_ref.info
    input_acquisition = img_ref.info.get("acq", None)
    is_debug = "debug" in file_dir.name or "debug" in file_dir.parent.name

    logger_texts = [f"{filename} input incompatible with model"]

    if input_format not in allowed_format:
        logger_texts.append(f"- Input format '{input_format}', model expected {allowed_format}")
        if not ignore_modality:
            compatible = False
    if has_seg_key and allowed_format not in Modality.format_keys(Modality.SEG):
        logger_texts.append("- Input acquisition not segmentation, but found a 'seg'-key")
        if not ignore_modality:
            compatible = False

    if input_acquisition is not None and input_acquisition not in allowed_acq:
        logger_texts.append(f"- Input acquisition '{input_acquisition}', model expected {allowed_acq}")
        if not ignore_acquisition:
            compatible = False

    if has_label_key:
        logger_texts.append("- Found 'label' key, which is not expected")
        compatible = False

    if is_debug:
        logger_texts.append("- probably a debug file (debug in name or parent)")
        compatible = False

    img_nii = img_ref.open_nii()
    if img_nii.get_plane() not in ["iso", *allowed_acq]:
        logger_texts.append(f"- get_plane() is not 'iso' or expected {allowed_acq}, skip")
        compatible = False

    if len(logger_texts) > 1:
        for l in logger_texts:
            logger.print(l, Log_Type.WARNING, verbose=verbose)
    return compatible
