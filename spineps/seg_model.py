import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import from_numpy
from TPTBox import NII, ZOOMS, Image_Reference, Log_Type, Logger, No_Logger, to_nii
from typing_extensions import Self

from spineps.seg_enums import Acquisition, InputType, Modality, ModelType, OutputType
from spineps.seg_modelconfig import Segmentation_Inference_Config, load_inference_config
from spineps.Unet3D.pl_unet import PLNet
from spineps.utils.citation_reminder import citation_reminder
from spineps.utils.filepaths import search_path
from spineps.utils.inference_api import load_inf_model, run_inference

threads_started = False


class Segmentation_Model(ABC):
    """Abstract Segmentation Model class

    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,  # type:ignore
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        """Initializes the segmentation model, finding and loading the corresponding inference config for that model

        Args:
            model_folder (str | Path): Path to that model's folder
            inference_config (Segmentation_Inference_Config | None, optional): Path to the inference config (if different from model folder). Defaults to None.
            default_verbose (bool): If true, will spam a lot more when using. Defaults to True.
            default_allow_tqdm (bool, optional): If true, will showcase a progress bar while segmenting. Defaults to True.
        """
        self.name: str = ""
        assert os.path.exists(str(model_folder)), f"model_folder doesnt exist, got {model_folder}"  # noqa: PTH110

        self.logger = No_Logger()
        self.use_cpu = use_cpu

        if inference_config is None:
            json_dir = Path(model_folder).joinpath("inference_config.json")
            self.inference_config = load_inference_config(json_dir, self.logger)
        else:
            self.inference_config = inference_config

        self.default_verbose = default_verbose
        self.logger.override_prefix = self.inference_config.log_name
        self.logger.default_verbose = self.default_verbose
        self.model_folder = str(model_folder)
        self.default_allow_tqdm = default_allow_tqdm
        self.predictor = None

        self.print("initialized with inference config", self.inference_config)

    @abstractmethod
    def load(
        self,
        folds: tuple[str, ...] | None = None,
    ) -> Self:
        """Loads the weights from disk

        Returns:
            Self: Segmentation_Model, but with loaded weights
        """
        return self

    def calc_recommended_resampling_zoom(self, input_zoom: ZOOMS) -> ZOOMS:
        """Calculates the resolution a corresponding input should be resampled to for this model

        Args:
            input_zoom (ZOOMS): _description_

        Returns:
            ZOOMS: _description_
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
        self_zms = self.calc_recommended_resampling_zoom(input_zoom=input_zoom)
        model_zms = model.calc_recommended_resampling_zoom(input_zoom=self_zms)
        match: bool = bool(np.all([self_zms[i] - model_zms[i] < 1e-4 for i in range(3)]))
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
    ) -> dict[OutputType, NII]:
        """Segments a given input with this model

        Args:
            input (Image_Reference | dict[InputType, Image_Reference]): input
            pad_size (int, optional): Padding in each dimension (times two more pixels in each dim). Defaults to 4.
            step_size (float | None, optional): _description_. Defaults to None.
            resample_to_recommended (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            dict[OutputType, NII]: _description_
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
        result = self.run(
            input_nii=input_niftys_in_order,
            verbose=verbose,
        )
        assert OutputType.seg in result and isinstance(result[OutputType.seg], NII), "No seg output in segmentation result"
        for k, v in result.items():
            if isinstance(v, NII):  # and k != OutputType.seg_modelres:
                if resample_output_to_input_space:
                    v.rescale_(zms_pir, verbose=self.logger).reorient_(orientation, verbose=self.logger)
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
        """Returns the modalities this model supports

        Returns:
            list[Modality]: _description_
        """
        return self.inference_config.modalities

    def acquisition(self) -> Acquisition:
        """Returns the acquisition this model supports

        Returns:
            Acquisition: _description_
        """
        return self.inference_config.acquisition

    @abstractmethod
    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        pass

    def print(self, *text, verbose: bool | None = None):
        if verbose is None:
            verbose = self.default_verbose
        self.logger.print(*text, verbose=verbose)

    def print_self(self):
        """Prints own model id"""
        self.print(self.modelid(include_log_name=False), verbose=True)
        self.print("Config:", self.inference_config, verbose=True)

    def modelid(self, include_log_name: bool = False):
        name: str = str(self.name)
        if name != "":
            if include_log_name:
                return name + " -- " + self.inference_config.log_name
            return name
        return self.inference_config.log_name

    def dict_representation(self, input_zms: ZOOMS | None):
        info = {
            "name": self.modelid(),  # self.inference_config.__repr__()
            "model_path": str(self.model_folder),
            "modality": str(self.modalities()),
            "aquisition": str(self.acquisition()),
            "resolution_range": str(self.inference_config.resolution_range),
        }
        if input_zms is not None:
            proc_zms = self.calc_recommended_resampling_zoom(input_zms)
            info["resolution_processed"] = str(proc_zms)
        return info

    def __str__(self):
        return self.modelid(include_log_name=True) + "\nConfig: " + self.inference_config.__repr__()

    def __repr__(self) -> str:
        return str(self)


class Segmentation_Model_NNunet(Segmentation_Model):
    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)

    def load(self, folds: tuple[str, ...] | None = None) -> Self:
        global threads_started  # noqa: PLW0603
        if not os.path.exists(self.model_folder):  # noqa: PTH110
            self.print(f"Model weights not found in {self.model_folder}", Log_Type.FAIL)
        self.predictor = load_inf_model(
            model_folder=self.model_folder,
            step_size=self.inference_config.default_step_size,
            use_folds=folds if folds is not None else tuple([str(i) for i in range(self.inference_config.available_folds)]),
            inference_augmentation=self.inference_config.inference_augmentation,
            init_threads=not threads_started,
            allow_non_final=True,
            verbose=False,
            ddevice="cuda" if not self.use_cpu else "cpu",
        )
        threads_started = True
        self.predictor.allow_tqdm = self.default_allow_tqdm
        self.predictor.verbose = False
        self.print("Model loaded from", self.model_folder, verbose=True)
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        self.print("Segmenting...")
        seg_nii, softmax_logits = run_inference(
            input_nii,
            self.predictor,
        )
        self.print("Segmentation done!")
        self.print("out_inf", seg_nii.zoom, seg_nii.orientation, seg_nii.shape, verbose=verbose)
        return {OutputType.seg: seg_nii, OutputType.softmax_logits: softmax_logits}


class Segmentation_Model_Unet3D(Segmentation_Model):
    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)
        assert len(self.inference_config.expected_inputs) == 1, "Unet3D cannot expect more than one input"

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        assert os.path.exists(self.model_folder)  # noqa: PTH110

        chktpath = search_path(self.model_folder, "**/*weights*.ckpt")
        assert len(chktpath) == 1
        model = PLNet.load_from_checkpoint(checkpoint_path=chktpath[0])
        model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        model.to(self.device)
        self.predictor = model
        self.print("Model loaded from", self.model_folder, verbose=True)
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        assert len(input_nii) == 1, "Unet3D does not support more than one input"
        input_nii = input_nii[0]

        arr = input_nii.get_seg_array().astype(np.int16)
        target = from_numpy(arr).to(torch.float32)
        target /= 9
        target = target.unsqueeze(0)
        target = target.unsqueeze(0)
        logits = self.predictor.forward(target.to(self.device))
        pred_x = self.predictor.softmax(logits)
        _, pred_cls = torch.max(pred_x, 1)
        del logits
        del pred_x
        pred_cls = pred_cls.detach().cpu().numpy()[0]
        seg_nii: NII = input_nii.set_array(pred_cls)
        self.print("out", seg_nii.zoom, seg_nii.orientation, seg_nii.shape) if verbose else None
        return {OutputType.seg: seg_nii}


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
    else:
        raise NotImplementedError(modeltype)
