"""Vertebra-labeling classifier: crops vertebra patches and predicts their anatomical labels."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from monai.transforms import CenterSpatialCropd, Compose, NormalizeIntensityd, ToTensor
from scipy.ndimage.interpolation import rotate
from TPTBox import NII, Log_Type, No_Logger, np_utils
from typing_extensions import Self

from spineps.architectures.pl_densenet import PLClassifier
from spineps.seg_enums import OutputType
from spineps.seg_model import Segmentation_Inference_Config, Segmentation_Model
from spineps.utils.filepaths import search_path

logger = No_Logger(prefix="VertLabelingClassifier")

# Default spatial size (voxels) of the cropped patch fed to the vertebra-labeling classifier.
DEFAULT_CLASSIFIER_INPUT_SIZE = (152, 168, 32)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, signed=True):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if signed:
        sign = np.array(np.sign(np.cross(v1, v2).dot((1, 1, 1))))
        # 0 means collinear: 0 or 180. Let's call that clockwise.
        sign[sign == 0] = 1
        angle = sign * angle
    return angle


def rotate_patch_sagitally(patch: np.ndarray, angle: float, msk: bool = False, cval: int = 0) -> np.ndarray:
    """Rotates a patch sagittally by a given angle (assuming the patch is in (I, P, L) orientation).

    Args:
        patch (np.ndarray): A numpy array in (I, P, L) orientation.
        angle (float): Angle of rotation in degrees.
        msk (bool, optional): If true, treats the patch as a mask and uses nearest-neighbour interpolation (order 0);
            otherwise uses cubic interpolation (order 3). Defaults to False.
        cval (int, optional): Constant value used to fill regions outside the rotated patch. Defaults to 0.

    Returns:
        np.ndarray: The rotated patch with the same shape as the input.
    """
    if msk:
        cval = 0
        order = 0
    else:
        order = 3
    rotated_patch = rotate(patch, angle=-angle, reshape=False, order=order, mode="constant", cval=cval)  # type: ignore
    return rotated_patch


class VertLabelingClassifier(Segmentation_Model):
    """Classifier that assigns anatomical labels to individual vertebrae.

    For each vertebra a patch is cropped around its center of mass, optionally rotated to align with the spine axis,
    normalized and center-cropped to a fixed size, then passed through a DenseNet (PLClassifier) that outputs per-head
    softmax predictions. Although it subclasses Segmentation_Model to reuse config loading, it does not perform voxel
    segmentation (run/segment_scan are not implemented).

    Attributes:
        device (torch.device): Device the classifier runs on.
        final_size (tuple[int, int, int]): Spatial size (voxels) the cropped patch is reduced to before inference.
        cutout_size (tuple[int, int, int]): Patch size used when cutting out a vertebra, set from the loaded model.
        totensor (ToTensor): Transform converting numpy arrays to tensors.
        transform (Compose): Intensity normalization and center-crop transform applied to each patch.
    """

    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,  # type:ignore
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        """Initializes the vertebra-labeling classifier and its preprocessing transforms.

        Args:
            model_folder (str | Path): Path to the classifier's model folder.
            inference_config (Segmentation_Inference_Config | None, optional): Inference config; if None, loads it from the
                model folder. Defaults to None.
            use_cpu (bool, optional): If true, runs inference on CPU instead of GPU. Defaults to False.
            default_verbose (bool, optional): If true, prints more information when used. Defaults to False.
            default_allow_tqdm (bool, optional): If true, shows a progress bar while predicting. Defaults to True.

        Raises:
            AssertionError: If the inference config expects more than one input.
        """
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)
        assert len(self.inference_config.expected_inputs) == 1, "Unet3D cannot expect more than one input"
        # self.model: PLClassifier = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.final_size: tuple[int, int, int] = DEFAULT_CLASSIFIER_INPUT_SIZE
        self.totensor = ToTensor()
        self.transform = Compose(
            [
                NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=False),
                CenterSpatialCropd(keys=["img", "seg"], roi_size=self.final_size),
            ]
        )

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        """Loads the classifier checkpoint and updates the preprocessing transform to the model's input size.

        Args:
            folds (tuple[str, ...] | None, optional): Unused; present for interface compatibility. Defaults to None.

        Returns:
            Self: This classifier with its predictor loaded and moved to the selected device.

        Raises:
            AssertionError: If no matching checkpoint file is found in the model folder.
        """
        assert os.path.exists(self.model_folder)  # noqa: PTH110

        chktpath = search_path(self.model_folder, "**/*val_f1=*valf1-weights.ckpt")
        assert len(chktpath) >= 1, chktpath
        model = PLClassifier.load_from_checkpoint(checkpoint_path=chktpath[-1], weights_only=False)
        if hasattr(model.opt, "final_size"):
            self.final_size = model.opt.final_size
            self.transform = Compose(
                [
                    NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=False),
                    CenterSpatialCropd(keys=["img", "seg"], roi_size=self.final_size),
                ]
            )
        model.eval()
        model.net.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu")
        model.to(self.device)
        self.predictor = model
        self.cutout_size = model.opt.final_size
        self.print("Model loaded from", self.model_folder, Log_Type.OK, verbose=True)
        return self

    def run(
        self,
        input_nii: list[NII],
        verbose: bool = False,
    ) -> dict[OutputType, NII | None]:
        """Not implemented: the classifier does not perform voxel segmentation.

        Args:
            input_nii (list[NII]): Unused.
            verbose (bool, optional): Unused. Defaults to False.

        Raises:
            NotImplementedError: Always, since running it as a segmentation model is not meaningful.
        """
        raise NotImplementedError("Doesnt make sense")

    def segment_scan(*args, **kwargs):
        """Not implemented: the classifier does not perform voxel segmentation.

        Raises:
            NotImplementedError: Always, since segmenting with this model is not meaningful.
        """
        raise NotImplementedError("Doesnt make sense")

    @classmethod
    def from_modelfolder(cls, model_folder: str | Path):
        """Not implemented: construction directly from a model folder.

        Args:
            model_folder (str | Path): Path to the model folder.

        Raises:
            NotImplementedError: Always; use from_checkpoint_path instead.
        """
        raise NotImplementedError()
        # find checkpoint yourself, then load from checkpoitn path

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str | Path):
        """Constructs a classifier from a checkpoint file path.

        Resolves the model folder as the grandparent of the checkpoint file and instantiates the classifier from it.

        Args:
            checkpoint_path (str | Path): Path to the checkpoint (.ckpt) file.

        Returns:
            VertLabelingClassifier: The constructed classifier.

        Raises:
            AssertionError: If the checkpoint path does not exist.
        """
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint path does not exist: {checkpoint_path}"
        # model = PLClassifier.load_from_checkpoint(
        #    str(checkpoint_path),
        # )
        d = cls(checkpoint_path.parent.parent)
        logger.print("Model loaded from", checkpoint_path, verbose=True)
        return d

    def run_all_position_instances(self, img: NII, com_list: list[tuple[int, int, int]]):
        """Runs the classifier on patches cropped around a list of center-of-mass positions.

        Args:
            img (NII): The intensity image (reoriented in place to the default orientation).
            com_list (list[tuple[int, int, int]]): Center-of-mass voxel positions, ordered top-to-bottom, one per vertebra.

        Returns:
            dict[int, dict[str, np.ndarray]]: Mapping from list index to a dict with "soft" (softmax outputs) and
                "pred" (argmax class) per classifier head.
        """
        img.reorient_()
        # assert coms are in PIR?
        # assert coms are in order top-to-bottom
        predictions = {}
        for idx, com in enumerate(com_list):
            logits_soft, pred_cls = self.run_given_center_pos(img, com)
            predictions[idx] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def run_all_seg_instances(self, img: NII, seg: NII) -> dict[int, dict[str, np.ndarray]]:
        """Runs the classifier on every vertebra instance present in a segmentation mask.

        For each label in the mask, computes the patch rotation angle from the neighbouring vertebra centers of mass (to
        align with the spine axis) and runs the classifier on the corresponding patch.

        Args:
            img (NII): The intensity image.
            seg (NII): The vertebra instance segmentation mask.

        Returns:
            dict[int, dict[str, np.ndarray]]: Mapping from vertebra label to a dict with "soft" (softmax outputs) and
                "pred" (argmax class) per classifier head.
        """
        img = img.reorient()
        seg = seg.reorient()
        # TODO assert order of seg labels are order from top to bottom
        predictions = {}

        coms = seg.reorient(("I", "P", "L")).center_of_masses()
        sorted_ctds = sorted([[a, *b] for a, b in coms.items()], key=lambda x: x[1])

        for v in seg.unique():
            # Find the index of the given vertebra in the sorted list
            idx = next(i for i, ct in enumerate(sorted_ctds) if ct[0] == v)

            # Get the centroids above and below
            ctd1 = sorted_ctds[idx - 1][1:] if idx > 0 else sorted_ctds[idx][1:]
            ctd2 = sorted_ctds[idx + 1][1:] if idx < len(sorted_ctds) - 1 else sorted_ctds[idx][1:]
            myradians = angle_between(np.asarray(ctd2) - np.asarray(ctd1), (1, 0, 0))  # type: ignore
            mydegrees = math.degrees(myradians)

            logits_soft, pred_cls = self.run_given_seg_pos(img, seg, vert_label=v, angle=mydegrees)
            predictions[v] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def run_given_seg_pos(self, img: NII, seg: NII, vert_label: int | None = None, angle: float | None = None):
        """Runs the classifier on the patch centered on a single vertebra defined by a segmentation.

        Selects the given vertebra label (or binarizes the mask if multiple labels are present), computes the center of its
        bounding box and runs the classifier there.

        Args:
            img (NII): The intensity image.
            seg (NII): The segmentation mask defining the vertebra location.
            vert_label (int | None, optional): Label of the vertebra to use; if None, the whole mask is used. Defaults to None.
            angle (float | None, optional): Rotation angle (degrees) to align the patch with the spine axis. Defaults to None.

        Returns:
            tuple[dict, dict]: The softmax outputs and argmax class predictions per classifier head.
        """
        if vert_label is not None:
            seg = seg.extract_label(vert_label)
        elif len(seg.unique()) > 1:
            logger.print("Found multiple labels in given seg for center of mass calculation, intended?", Log_Type.STRANGE)
            seg[seg != 0] = 1
        crop = seg.compute_crop()
        center_of_crop = []
        for i in range(len(crop)):
            size_t = crop[i].stop - crop[i].start
            center_of_crop.append(crop[i].start + (size_t // 2))
        return self.run_given_center_pos(img, seg, center_of_crop, angle=angle)  # type: ignore

    def run_given_center_pos(self, img: NII, seg: NII, center_pos: tuple[int, int, int], angle: float | None = None):
        """Crops image and segmentation patches around a center point, optionally rotates them, and runs the classifier.

        Cuts out a patch larger than the final size (with extra padding for rotation), reorients to (I, P, L), optionally
        rotates sagittally by the given angle, crops back to the cutout size and runs the classifier on the patch.

        Args:
            img (NII): The intensity image (or a raw array).
            seg (NII): The segmentation mask used as the second channel.
            center_pos (tuple[int, int, int]): Voxel position to center the patch on.
            angle (float | None, optional): Rotation angle (degrees) to align the patch with the spine axis; no rotation if
                None or 0. Defaults to None.

        Returns:
            tuple[dict, dict]: The softmax outputs and argmax class predictions per classifier head.
        """
        extra_rotation_padding = 64
        extra_rotation_padding_halfed = extra_rotation_padding // 2
        #
        # cut array then runs prediction
        arr = img.get_array() if isinstance(img, NII) else img
        arr_cut, cutout_coords_slices, padding = np_utils.np_calc_crop_around_centerpoint(
            center_pos,
            arr,
            (self.cutout_size[0] + extra_rotation_padding, self.cutout_size[1] + extra_rotation_padding, self.cutout_size[2]),
        )
        sem_cut = np.pad(seg[cutout_coords_slices], padding)
        # final cutout size (200, 160, 32)

        ori = img.orientation
        img_v = img.set_array(arr_cut).reorient_(("I", "P", "L"))
        seg_v = seg.set_array(sem_cut).reorient_(("I", "P", "L"))

        # angle = 0
        if angle is not None and angle != 0:
            arr_cut = rotate_patch_sagitally(img_v.get_array(), -angle, msk=False)
            sem_cut = rotate_patch_sagitally(seg_v.get_seg_array(), -angle, msk=True)

        # crop down to final cutout size (200, 160, 32)
        arr_cut = arr_cut[
            extra_rotation_padding_halfed:-extra_rotation_padding_halfed,
            extra_rotation_padding_halfed:-extra_rotation_padding_halfed,
            :,
        ]
        sem_cut = sem_cut[
            extra_rotation_padding_halfed:-extra_rotation_padding_halfed,
            extra_rotation_padding_halfed:-extra_rotation_padding_halfed,
            :,
        ]

        img_v.set_array_(arr_cut).reorient_(ori)
        seg_v.set_array_(sem_cut).reorient_(ori)
        # img_v.save("/DATA/NAS/ongoing_projects/hendrik/img_v.nii.gz")
        # seg_v.save("/DATA/NAS/ongoing_projects/hendrik/seg_v.nii.gz")
        return self._run_array(img_v.get_array(), seg_v.get_seg_array())  # sem_cut

    def _run_nii(self, img_nii: NII):
        """Runs the classifier on the raw array of an NII patch.

        Args:
            img_nii (NII): The patch image to classify.

        Returns:
            tuple[dict, dict]: The softmax outputs and argmax class predictions per classifier head.
        """
        # TODO check resolution
        # TODO check size
        return self._run_array(img_nii.get_array())

    def run_all_arrays(self, img_arrays: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
        """Runs the classifier on a set of pre-cut image patches.

        Args:
            img_arrays (dict[int, np.ndarray]): Mapping from vertebra id to its 3D image patch.

        Returns:
            dict[int, dict[str, np.ndarray]]: Mapping from vertebra id to a dict with "soft" (softmax outputs) and
                "pred" (argmax class) per classifier head.
        """
        # TODO assert order of seg labels are order from top to bottom
        predictions = {}
        for v, arr in img_arrays.items():
            logits_soft, pred_cls = self._run_array(arr)
            predictions[v] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def _run_array(self, img_arr: np.ndarray, seg_arr: np.ndarray | None | torch.Tensor = None):  # , seg_arr: np.ndarray):
        """Applies preprocessing and runs the classifier forward pass on a single image patch.

        Converts the patch (and optional segmentation) to tensors, applies intensity normalization and center cropping,
        adds the channel/batch dimensions and runs the network, returning per-head softmax probabilities and argmax classes.

        Args:
            img_arr (np.ndarray): The 3D image patch.
            seg_arr (np.ndarray | None | torch.Tensor, optional): Optional segmentation patch; if None, a copy of the image
                is used. Defaults to None.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Per-head softmax probabilities and per-head argmax classes.

        Raises:
            AssertionError: If img_arr is not 3-dimensional.
        """
        assert img_arr.ndim == 3, f"Dimension mismatch, {img_arr.shape}, expected 3 dimensions"
        #
        img_arr = self.totensor(img_arr)
        # add channel
        img_arr.unsqueeze_(0)

        if seg_arr is not None:
            seg_arr = self.totensor(seg_arr)
            seg_arr.unsqueeze_(0)
        else:
            seg_arr = img_arr.clone()

        d = self.transform({"img": img_arr, "seg": seg_arr})

        # TODO seg channelwise and stuff

        model_input = d["img"]
        # print(model_input.shape)
        model_input.unsqueeze_(0)
        # print(model_input.shape)
        model_input = model_input.to(torch.float32)
        model_input = model_input.to(self.device)

        self.predictor.eval()
        self.predictor.to(self.device)
        logits_dict = self.predictor.forward(model_input)
        logits_soft = {k: self.predictor.softmax(v)[0].detach().cpu().numpy() for k, v in logits_dict.items()}
        pred_cls = {k: np.argmax(v, 0) for k, v in logits_soft.items()}
        return logits_soft, pred_cls
