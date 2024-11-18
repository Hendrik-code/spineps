from pathlib import Path

import numpy as np
import torch
from monai.transforms import CenterSpatialCropd, Compose, NormalizeIntensityd, ToTensor
from TPTBox import NII, Log_Type, No_Logger, np_utils

from spineps.architectures.pl_densenet import PLClassifier

logger = No_Logger(prefix="VertLabelingClassifier")


class VertLabelingClassifier:
    def __init__(self, model: PLClassifier):
        self.model: PLClassifier = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        self.model.eval()
        self.model.net.eval()
        final_size: tuple[int, int, int] = (152, 168, 32)
        self.totensor = ToTensor()
        self.transform = Compose(
            [
                NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=False),
                CenterSpatialCropd(keys=["img", "seg"], roi_size=final_size),
            ]
        )

    @classmethod
    def from_modelfolder(cls, model_folder: str | Path):
        raise NotImplementedError()
        # pass  # find checkpoint yourself, then load from checkpoitn path

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str | Path):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint path does not exist: {checkpoint_path}"
        model = PLClassifier.load_from_checkpoint(
            str(checkpoint_path),
            # opt=ARGS_MODEL(),
            # objectives=Objectives(
            #    [
            #        Target.VERT,
            #        Target.REGION,
            #        Target.VERTREL,
            #        Target.FULLYVISIBLE,
            #    ]
            # ),
        )
        # print("weight", model.classification_heads["REGION"].weight[:5])
        d = cls(model)
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
            (152, 168, 32),
        )
        # sem_cut = np.pad(vert_v.get_seg_array()[cutout_coords_slices], padding)
        return self._run_array(arr_cut)  # sem_cut

    def _run_nii(self, img_nii: NII):
        # TODO check resolution
        # TODO check size
        return self._run_array(img_nii.get_array())

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

        self.model.eval()
        logits_dict = self.model.forward(model_input)
        logits_soft = {k: self.model.softmax(v)[0].detach().cpu().numpy() for k, v in logits_dict.items()}
        pred_cls = {k: np.argmax(v, 0) for k, v in logits_soft.items()}
        return logits_soft, pred_cls
