from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics.functional as mF  # noqa: N812
from torch import nn
from torch.optim import lr_scheduler

from spineps.architectures.unet3D import Unet3D


class PLNet(pl.LightningModule):
    def __init__(self, opt=None, do2D: bool = False, *args: Any, **kwargs: Any) -> None:  # noqa: N803, ARG002
        super().__init__()
        self.save_hyperparameters()

        nclass = Unet3D

        dim_mults = (1, 2, 4, 8)
        dim = 16  # 16

        # if opt.high_res:
        #    dim = 16
        #    dim_mults = (2, 4, 8, 8)

        self.network = nclass(
            dim=dim,
            dim_mults=dim_mults,
            out_dim=4,
            channels=10,  # 10,
        )

        self.opt = opt
        self.do2D = do2D
        self.n_epoch = opt.n_epoch if opt is not None else 0
        self.start_lr = 0.0001
        self.linear_end_factor = 0.01
        self.l2_reg_w = 0.0001
        self.n_classes = 4

        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.network(x)

    def _shared_step(self, target, gt, detach2cpu: bool = False):
        logits = self.forward(target)
        loss = self.loss(logits, gt)

        with torch.no_grad():
            # pred_cls = torch.max(logits, 1)
            pred_x = self.softmax(logits)  # , dim=1)
            _, pred_cls = torch.max(pred_x, 1)
            del pred_x
            if detach2cpu:
                # From here on CPU
                gt = gt.detach().cpu()
                logits = logits.detach().cpu()
                pred_cls = pred_cls.detach().cpu()
        return loss, logits, gt, pred_cls

    def _shared_metric_step(self, loss, _, gt, pred_cls):
        dice = mF.dice(pred_cls, gt, num_classes=self.n_classes)
        diceFG = mF.dice(pred_cls, gt, num_classes=self.n_classes, ignore_index=0)  # noqa: N806
        dice_p_cls = mF.dice(pred_cls, gt, average=None, num_classes=self.n_classes)
        return {"loss": loss.detach().cpu(), "dice": dice, "diceFG": diceFG, "dice_p_cls": dice_p_cls}

    def _shared_metric_append(self, metrics, outputs):
        for k, v in metrics.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)

    def _shared_cat_metrics(self, outputs):
        results = {}
        for m, v in outputs.items():
            stacked = torch.stack(v)
            results[m] = torch.mean(stacked) if m != "dice_p_cls" else torch.mean(stacked, dim=0)
        return results

    def __str__(self):
        text = "Unet"
        dim = "2D" if self.do2D else "3D"
        return text + "_" + dim


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)
