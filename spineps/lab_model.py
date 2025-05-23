import os
from pathlib import Path

import numpy as np
import torch
from monai.transforms import CenterSpatialCropd, Compose, NormalizeIntensityd, ToTensor
from TPTBox import NII, Log_Type, No_Logger, np_utils
from typing_extensions import Self

from spineps.architectures.pl_densenet import PLClassifier
from spineps.seg_enums import OutputType
from spineps.seg_model import Segmentation_Inference_Config, Segmentation_Model
from spineps.utils.filepaths import search_path

logger = No_Logger(prefix="VertLabelingClassifier")


class VertLabelingClassifier(Segmentation_Model):
    def __init__(
        self,
        model_folder: str | Path,
        inference_config: Segmentation_Inference_Config | None = None,  # type:ignore
        use_cpu: bool = False,
        default_verbose: bool = False,
        default_allow_tqdm: bool = True,
    ):
        super().__init__(model_folder, inference_config, use_cpu, default_verbose, default_allow_tqdm)
        assert len(self.inference_config.expected_inputs) == 1, "Unet3D cannot expect more than one input"
        # self.model: PLClassifier = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.final_size: tuple[int, int, int] = (152, 168, 32)
        self.totensor = ToTensor()
        self.transform = Compose(
            [
                NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=False),
                CenterSpatialCropd(keys=["img", "seg"], roi_size=self.final_size),
            ]
        )

    def load(self, folds: tuple[str, ...] | None = None) -> Self:  # noqa: ARG002
        assert os.path.exists(self.model_folder)  # noqa: PTH110

        chktpath = search_path(self.model_folder, "**/*val_f1=*valf1-weights.ckpt")
        assert len(chktpath) >= 1, chktpath
        model = PLClassifier.load_from_checkpoint(checkpoint_path=chktpath[-1])
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
        raise NotImplementedError("Doesnt make sense")

    def segment_scan(*args, **kwargs):
        raise NotImplementedError("Doesnt make sense")

    @classmethod
    def from_modelfolder(cls, model_folder: str | Path):
        raise NotImplementedError()
        # find checkpoint yourself, then load from checkpoitn path

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str | Path):
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
        img.reorient_()
        # assert coms are in PIR?
        # assert coms are in order top-to-bottom
        predictions = {}
        for idx, com in enumerate(com_list):
            logits_soft, pred_cls = self.run_given_center_pos(img, com)
            predictions[idx] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def run_all_seg_instances(self, img: NII, seg: NII) -> dict[int, dict[str, np.ndarray]]:
        img = img.reorient()
        seg = seg.reorient()
        # TODO assert order of seg labels are order from top to bottom
        predictions = {}
        for v in seg.unique():
            logits_soft, pred_cls = self.run_given_seg_pos(img, seg, vert_label=v)
            predictions[v] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def run_given_seg_pos(self, img: NII | np.ndarray, seg: NII, vert_label: int | None = None):
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
        return self.run_given_center_pos(img, center_of_crop)

    def run_given_center_pos(self, img: NII | np.ndarray, center_pos: tuple[int, int, int]):
        # cut array then runs prediction
        arr = img.get_array() if isinstance(img, NII) else img
        arr_cut, cutout_coords_slices, padding = np_utils.np_calc_crop_around_centerpoint(
            center_pos,
            arr,
            self.cutout_size,
        )
        # sem_cut = np.pad(vert_v.get_seg_array()[cutout_coords_slices], padding)
        return self._run_array(arr_cut)  # sem_cut

    def _run_nii(self, img_nii: NII):
        # TODO check resolution
        # TODO check size
        return self._run_array(img_nii.get_array())

    def run_all_arrays(self, img_arrays: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
        # TODO assert order of seg labels are order from top to bottom
        predictions = {}
        for v, arr in img_arrays.items():
            logits_soft, pred_cls = self._run_array(arr)
            predictions[v] = {"soft": logits_soft, "pred": pred_cls}
        return predictions

    def _run_array(self, img_arr: np.ndarray):  # , seg_arr: np.ndarray):
        assert img_arr.ndim == 3, f"Dimension mismatch, {img_arr.shape}, expected 3 dimensions"
        #
        img_arr = self.totensor(img_arr)
        # add channel
        img_arr.unsqueeze_(0)
        d = self.transform({"img": img_arr, "seg": img_arr})

        # TODO seg channelwise and stuff

        model_input = d["img"]
        model_input.unsqueeze_(0)
        model_input = model_input.to(torch.float32)
        model_input = model_input.to(self.device)

        self.predictor.eval()
        self.predictor.to(self.device)
        logits_dict = self.predictor.forward(model_input)
        logits_soft = {k: self.predictor.softmax(v)[0].detach().cpu().numpy() for k, v in logits_dict.items()}
        pred_cls = {k: np.argmax(v, 0) for k, v in logits_soft.items()}
        return logits_soft, pred_cls
