"""Segmentation model abstractions: the abstract Segmentation_Model and its nnU-Net and Unet3D subclasses."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import from_numpy
from TPTBox import NII, ZOOMS, Image_Reference, Log_Type, No_Logger, to_nii
from typing_extensions import Self

from spineps.architectures.pl_unet import PLNet
from spineps.architectures_new.pl_unet import PLNet as PLNet_new
from spineps.seg_enums import Acquisition, InputType, Modality, OutputType
from spineps.utils.citation_reminder import citation_reminder
from spineps.utils.filepaths import search_path
from spineps.utils.inference_api import load_inf_model, run_inference
from spineps.utils.seg_modelconfig import Segmentation_Inference_Config, load_inference_config

threads_started = False

# Two zoom vectors are considered identical if every axis differs by less than this (mm).
ZOOM_MATCH_TOLERANCE = 1e-4
# Legacy single-channel Unet3D divided the input label ids by this value to scale them to ~[0, 1].
LEGACY_LABEL_NORMALIZATION = 9


class Segmentation_Model(ABC):
    """Abstract base class wrapping a segmentation network together with its inference configuration.

    Subclasses implement load() and run() for a concrete backend (e.g. nnU-Net or Unet3D). The class handles input
    preparation (reorientation, rescaling to the recommended zoom, padding), running the model and mapping the output back
    into the input space.

    Attributes:
        name (str): Optional human-readable model name.
        logger (No_Logger): Logger used for all model output.
        use_cpu (bool): If true, runs inference on CPU instead of GPU.
        inference_config (Segmentation_Inference_Config): Configuration describing expected inputs, resolution range and labels.
        default_verbose (bool): Default verbosity for printing.
        default_allow_tqdm (bool): Whether a progress bar is shown during segmentation by default.
        model_folder (str): Path to the model's folder on disk.
        predictor: The loaded backend predictor, or None until load() is called.
    """

    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,  # type:ignore
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        """Initializes the segmentation model, finding and loading the corresponding inference config for that model.

        Args:
            model_folder (str | Path): Path to that model's folder.
            inference_config (Segmentation_Inference_Config | None, optional): Inference config to use; if None, loads
                "inference_config.json" from the model folder. Defaults to None.
            use_cpu (bool, optional): If true, runs inference on CPU instead of GPU. Defaults to False.
            default_verbose (bool, optional): If true, prints more information when used. Defaults to False.
            default_allow_tqdm (bool, optional): If true, shows a progress bar while segmenting. Defaults to True.

        Raises:
            AssertionError: If model_folder does not exist.
        """
        self.name: str = ""
        assert Path(model_folder).exists(), f"model_folder does not exist, got {model_folder}"

        self.logger = No_Logger()
        self.use_cpu = use_cpu

        if inference_config is None:
            json_dir = Path(model_folder).joinpath("inference_config.json")
            self.inference_config = load_inference_config(json_dir, self.logger)
        else:
            self.inference_config = inference_config

        self.default_verbose = default_verbose
        self.logger.prefix = self.inference_config.log_name
        self.logger.default_verbose = self.default_verbose
        self.model_folder = str(model_folder)
        self.default_allow_tqdm = default_allow_tqdm
        self.predictor = None

        self.print("initialized with inference config", self.inference_config)

    @abstractmethod
    def load(self, folds: tuple[str, ...] | None = None) -> Self:
        """Loads the model weights from disk.

        Args:
            folds (tuple[str, ...] | None, optional): Which folds to load; if None, uses the folds from the inference config.
                Defaults to None.

        Returns:
            Self: This model with its predictor loaded.
        """
        return self

    def calc_recommended_resampling_zoom(self, input_zoom: ZOOMS) -> ZOOMS:
        """Calculates the resolution a corresponding input should be resampled to for this model.

        If the inference config defines a (min, max) resolution range, each axis of the input zoom is clamped into that
        range; otherwise the fixed configured resolution is returned.

        Args:
            input_zoom (ZOOMS): Voxel spacing (mm) of the input image, per axis.

        Returns:
            ZOOMS: Recommended voxel spacing (mm) to resample the input to before inference.
        """
        if len(self.inference_config.resolution_range) != 2:
            return self.inference_config.resolution_range
        output_zoom = tuple(
            max(
                min(input_zoom[idx], self.inference_config.resolution_range[1][idx]),  # type:ignore
                self.inference_config.resolution_range[0][idx],  # type:ignore
            )
            for idx in range(len(input_zoom))
        )
        return output_zoom

    def same_modelzoom_as_model(self, model: Self, input_zoom: ZOOMS) -> bool:
        """Checks whether another model would resample a given input to the same resolution as this model.

        Args:
            model (Self): The other segmentation model to compare against.
            input_zoom (ZOOMS): Voxel spacing (mm) of the input image, per axis.

        Returns:
            bool: True if both models' recommended resampling zooms agree on every axis within ZOOM_MATCH_TOLERANCE.
        """
        self_zms = self.calc_recommended_resampling_zoom(input_zoom=input_zoom)
        model_zms = model.calc_recommended_resampling_zoom(input_zoom=self_zms)
        match: bool = bool(np.all([abs(self_zms[i] - model_zms[i]) < ZOOM_MATCH_TOLERANCE for i in range(3)]))
        return match

    @citation_reminder
    def segment_scan(
        self,
        input_image: Image_Reference | dict[InputType, Image_Reference],
        pad_size: int = 0,
        step_size: float | None = None,
        resample_to_recommended: bool = True,
        resample_output_to_input_space: bool = True,
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        """Segments a given input with this model.

        Prepares each expected input (optional padding, reorientation to the model orientation and rescaling to the
        recommended zoom), runs the model and maps the outputs back into the input space.

        Args:
            input_image (Image_Reference | dict[InputType, Image_Reference]): A single image, or a mapping from InputType to
                image for multi-input models.
            pad_size (int, optional): Padding added in each dimension (this many extra voxels on each side per axis), removed
                again from the output. Defaults to 0.
            step_size (float | None, optional): Sliding-window tile step size; if None, uses the config default. Defaults to None.
            resample_to_recommended (bool, optional): If true, rescales each input to the model's recommended zoom. Defaults to True.
            resample_output_to_input_space (bool, optional): If true, resamples and pads the outputs back to the original input
                space. Defaults to True.
            verbose (bool, optional): If true, prints verbose information. Defaults to False.

        Returns:
            dict[OutputType, NII | None]: Mapping of output type to result NII (e.g. the segmentation mask, optionally softmax
                logits).
        """
        if self.predictor is None:
            self.load()
            assert self.predictor is not None, "self.predictor == None after load(). Error!"

        # Check if input matches expectation
        if not isinstance(input_image, dict):
            if len(self.inference_config.expected_inputs) >= 2:
                self.print(
                    "input is one Image_Reference but model expected more, if not already stacked correctly, this will fail!",
                    Log_Type.WARNING,
                )
            inputdict = {self.inference_config.expected_inputs[0]: input_image}
        else:
            inputdict: dict[InputType, Image_Reference] = input_image
        # Check if all required inputs are there
        if not set(inputdict.keys()).issuperset(self.inference_config.expected_inputs):
            self.print(f"expected {self.inference_config.expected_inputs}, but only got {list(inputdict.keys())}")
        orig_shape = None
        orientation = None
        zms = None
        input_niftys_in_order = []
        zms_pir: ZOOMS = None  # type: ignore
        for id in self.inference_config.expected_inputs:  # noqa: A001
            # Make nifty
            nii = to_nii(inputdict[id], seg=id == InputType.seg)
            # Padding
            if pad_size > 0:
                arr = nii.get_array()
                arr = np.pad(arr, pad_size, mode="edge")
                nii.set_array_(arr)
            input_niftys_in_order.append(nii)
            # Save first values for comparison
            if orig_shape is None:
                orig_shape = nii.shape
                orientation = nii.orientation
                zms = nii.zoom
            # Consistency check
            nii.assert_affine(shape=orig_shape, orientation=orientation, zoom=zms)
            # ), "All inputs need to be of same shape, orientation and zoom, got at least two different."
            # Reorient and rescale
            nii.reorient_(self.inference_config.model_expected_orientation, verbose=self.logger)
            zms_pir = nii.zoom
            if resample_to_recommended:
                nii.rescale_(self.calc_recommended_resampling_zoom(zms_pir), verbose=self.logger)

        assert orig_shape is not None
        if not resample_to_recommended:
            self.print("resample_to_recommended set to False, segmentation might not work. Proceed at own risk", Log_Type.WARNING)

        # set step_size
        if hasattr(self.predictor, "tile_step_size"):
            self.predictor.tile_step_size = step_size if step_size is not None else self.inference_config.default_step_size

        self.print("input", input_niftys_in_order[0], verbose=verbose)
        self.print("Run Segmentation")
        result = self.run(input_nii=input_niftys_in_order, verbose=verbose)
        assert OutputType.seg in result and isinstance(result[OutputType.seg], NII), "No seg output in segmentation result"
        for k, v in result.items():
            if isinstance(v, NII):  # and k != OutputType.seg_modelres:
                if resample_output_to_input_space:
                    v.resample_from_to_(inputdict[self.inference_config.expected_inputs[0]])
                    # v.rescale_(zms_pir, verbose=self.logger).reorient_(orientation, verbose=self.logger)
                    v.pad_to(orig_shape, inplace=True)
                if k == OutputType.seg:
                    v.map_labels_(self.inference_config.segmentation_labels, verbose=self.logger)
                if pad_size > 0:
                    arr = v.get_array()
                    arr = arr[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
                    v.set_array_(arr)

                self.print(f"out_seg {k}", v.zoom, v.orientation, v.shape, verbose=verbose)
        self.print("Segmenting done!")
        return result

    def modalities(self) -> list[Modality]:
        """Returns the modalities this model supports.

        Returns:
            list[Modality]: Modalities the model was trained for, as listed in its inference config.
        """
        return self.inference_config.modalities

    def acquisition(self) -> Acquisition:
        """Returns the acquisition this model supports.

        Returns:
            Acquisition: Acquisition plane/type the model expects, as listed in its inference config.
        """
        return self.inference_config.acquisition

    @abstractmethod
    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:
        """Runs the backend predictor on the prepared inputs.

        Args:
            input_nii (list[NII]): Inputs already reoriented and rescaled to the model's expectation, in the configured order.
            verbose (bool, optional): If true, prints verbose information. Defaults to False.

        Returns:
            dict[OutputType, NII | None]: Mapping of output type to result NII produced by the model.
        """

    def print(self, *text: object, verbose: bool | None = None):
        """Logs text via the model's logger.

        Args:
            *text: Items to print.
            verbose (bool | None, optional): Overrides the default verbosity; if None, uses default_verbose. Defaults to None.
        """
        if verbose is None:
            verbose = self.default_verbose
        self.logger.print(*text, verbose=verbose)

    def print_self(self):
        """Prints the model id and its inference config."""
        self.print(self.modelid(include_log_name=False), verbose=True)
        self.print("Config:", self.inference_config, verbose=True)

    def modelid(self, include_log_name: bool = False) -> str:
        """Returns an identifier string for this model.

        Args:
            include_log_name (bool, optional): If true and a name is set, appends the config log name. Defaults to False.

        Returns:
            str: The model name, or the inference config's log name if no name is set.
        """
        name: str = str(self.name)
        if name != "":
            if include_log_name:
                return name + " -- " + self.inference_config.log_name
            return name
        return self.inference_config.log_name

    def dict_representation(self) -> dict[str, str]:
        """Builds a summary dictionary describing this model.

        Returns:
            dict[str, str]: Model id, model path, modalities, acquisition and resolution range as strings.
        """
        info = {
            "name": self.modelid(),  # self.inference_config.__repr__()
            "model_path": str(self.model_folder),
            "modality": str(self.modalities()),
            "aquisition": str(self.acquisition()),
            "resolution_range": str(self.inference_config.resolution_range),
        }
        # if input_zms is not None:
        #    proc_zms = self.calc_recommended_resampling_zoom(input_zms)
        #    info["resolution_processed"] = str(proc_zms)
        return info

    def __str__(self) -> str:
        """Returns the model id together with its inference config representation.

        Returns:
            str: Human-readable description of the model.
        """
        return self.modelid(include_log_name=True) + "\nConfig: " + self.inference_config.__repr__()

    def __repr__(self) -> str:
        """Returns the same representation as __str__.

        Returns:
            str: Human-readable description of the model.
        """
        return str(self)


class Segmentation_Model_NNunet(Segmentation_Model):
    """Segmentation_Model backed by an nnU-Net predictor."""

    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        """Initializes an nnU-Net-backed segmentation model.

        Args:
            model_folder (str | Path): Path to the nnU-Net model folder.
            inference_config (Segmentation_Inference_Config | None, optional): Inference config; if None, loads it from the
                model folder. Defaults to None.
            use_cpu (bool, optional): If true, runs inference on CPU instead of GPU. Defaults to False.
            default_verbose (bool, optional): If true, prints more information when used. Defaults to False.
            default_allow_tqdm (bool, optional): If true, shows a progress bar while segmenting. Defaults to True.
        """
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:
        """Loads the nnU-Net predictor and its ensemble folds from the model folder.

        Args:
            folds (tuple[str, ...] | None, optional): Folds to load; if None, uses the folds from the inference config.
                Defaults to None.

        Returns:
            Self: This model with its nnU-Net predictor loaded.
        """
        global threads_started  # noqa: PLW0603
        if not os.path.exists(self.model_folder):  # noqa: PTH110
            self.print(f"Model weights not found in {self.model_folder}", Log_Type.FAIL)
        conf_folds = self.inference_config.available_folds
        if isinstance(conf_folds, int):
            conf_folds = tuple([str(i) for i in range(conf_folds)])
        elif isinstance(conf_folds, str):
            conf_folds = (conf_folds,)
        else:
            conf_folds = tuple([str(i) for i in conf_folds])
        self.predictor = load_inf_model(
            model_folder=self.model_folder,
            step_size=self.inference_config.default_step_size,
            use_folds=folds if folds is not None else conf_folds,
            inference_augmentation=self.inference_config.inference_augmentation,
            init_threads=not threads_started,
            allow_non_final=True,
            verbose=False,
            ddevice="cuda" if not self.use_cpu else "cpu",
        )
        threads_started = True
        self.predictor.allow_tqdm = self.default_allow_tqdm
        self.predictor.verbose = False
        self.print("Model loaded from", self.model_folder, Log_Type.OK, verbose=True)
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        """Runs nnU-Net inference on the prepared inputs.

        Args:
            input_nii (list[NII]): Inputs in the model's expected orientation and resolution, in the configured order.
            verbose (bool, optional): If true, prints verbose information. Defaults to False.

        Returns:
            dict[OutputType, NII | None]: The segmentation mask under OutputType.seg and the softmax logits under
                OutputType.softmax_logits.
        """
        self.print("Segmenting...")
        seg_nii, softmax_logits = run_inference(input_nii, self.predictor)
        self.print("Segmentation done!")
        self.print("out_inf", seg_nii.zoom, seg_nii.orientation, seg_nii.shape, verbose=verbose)
        return {OutputType.seg: seg_nii, OutputType.softmax_logits: softmax_logits}


class Segmentation_Model_Unet3D(Segmentation_Model):
    """Segmentation_Model backed by a single-input 3D U-Net (PyTorch Lightning PLNet).

    Used as the instance (vertebra) model: it takes a segmentation mask as input and refines it into the vertebra instance
    output. Supports both the current multi-channel network and a legacy single-channel network.
    """

    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        """Initializes a 3D U-Net-backed segmentation model.

        Args:
            model_folder (str | Path): Path to the model folder containing the checkpoint.
            inference_config (Segmentation_Inference_Config | None, optional): Inference config; if None, loads it from the
                model folder. Defaults to None.
            use_cpu (bool, optional): If true, runs inference on CPU instead of GPU. Defaults to False.
            default_verbose (bool, optional): If true, prints more information when used. Defaults to False.
            default_allow_tqdm (bool, optional): If true, shows a progress bar while segmenting. Defaults to True.

        Raises:
            AssertionError: If the inference config expects more than one input.
        """
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)
        assert len(self.inference_config.expected_inputs) == 1, "Unet3D cannot expect more than one input"

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        """Loads the 3D U-Net checkpoint, trying the current then the legacy PLNet implementation.

        Args:
            folds (tuple[str, ...] | None, optional): Unused; present for interface compatibility. Defaults to None.

        Returns:
            Self: This model with its 3D U-Net predictor loaded and moved to the selected device.

        Raises:
            AssertionError: If exactly one checkpoint file is not found in the model folder.
        """
        assert os.path.exists(self.model_folder)  # noqa: PTH110

        chktpath = search_path(self.model_folder, "**/*weights*.ckpt")
        assert len(chktpath) == 1, chktpath
        try:
            model = PLNet.load_from_checkpoint(checkpoint_path=chktpath[0], weights_only=False)
        except RuntimeError:
            model = PLNet_new.load_from_checkpoint(checkpoint_path=chktpath[0], weights_only=False)

        model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        model.to(self.device)
        self.predictor = model
        self.print("Model loaded from", self.model_folder, Log_Type.OK, verbose=True)
        return self

    def run(self, input_nii: list[NII], verbose: bool = False) -> dict[OutputType, NII | None]:
        """Runs the 3D U-Net on a single input segmentation mask.

        Converts the input mask to a network tensor (one-hot encoded for the multi-channel network, or intensity-normalized
        for the legacy single-channel network), runs the forward pass and returns the per-voxel argmax class as a mask.

        Args:
            input_nii (list[NII]): A single-element list containing the input segmentation mask.
            verbose (bool, optional): If true, prints verbose information. Defaults to False.

        Returns:
            dict[OutputType, NII | None]: The predicted segmentation mask under OutputType.seg.

        Raises:
            AssertionError: If more than one input is provided.
        """
        assert len(input_nii) == 1, "Unet3D does not support more than one input"
        input_nii_ = input_nii[0]

        arr = input_nii_.get_seg_array().astype(np.int16)

        target = from_numpy(arr)
        n_classes = self.predictor.network.channels

        target[target >= n_classes] = 0

        # channel-wise
        if n_classes != 1:
            targetc = target.to(torch.int64)
            targetc = F.one_hot(targetc, num_classes=n_classes)
            targetc = targetc.permute(3, 0, 1, 2)
            targetc = targetc.unsqueeze(0)
            targetc = targetc.to(torch.float32)
            logits = self.predictor.forward(targetc.to(self.device))
        else:
            # legacy version
            target = target.to(torch.float32)
            target /= LEGACY_LABEL_NORMALIZATION
            target = target.unsqueeze(0)
            target = target.unsqueeze(0)
            logits = self.predictor.forward(target.to(self.device))
        soft_max = torch.nn.Softmax(dim=1)
        pred_x = soft_max(logits)
        _, pred_cls = torch.max(pred_x, 1)
        del logits
        del pred_x
        pred_cls = pred_cls.detach().cpu().numpy()[0]
        seg_nii: NII = input_nii_.set_array(pred_cls)
        self.print("out", seg_nii.zoom, seg_nii.orientation, seg_nii.shape) if verbose else None
        return {OutputType.seg: seg_nii}
