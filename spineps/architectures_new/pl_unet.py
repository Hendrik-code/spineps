from __future__ import annotations

from argparse import Namespace
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torchmetrics.functional as mF
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import Adam, lr_scheduler

from .dice import MemoryEfficientSoftDiceLoss
from .unet2D import Unet2D
from .unet3D import Unet3D


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def _tb_logger(module: pl.LightningModule) -> TensorBoardLogger:
    return cast(TensorBoardLogger, module.logger)


class PLNet(pl.LightningModule):
    def __init__(self, opt: Namespace, do2D: bool = False, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        super().__init__()
        self.save_hyperparameters()

        arch = Unet2D if do2D else Unet3D
        self.network = arch(
            dim=8,
            dim_mults=(1, 2, 4, 8),
            out_dim=4,
            channels=1 if not opt.channelwise else 11,
        )

        self.do2D = do2D
        self.n_epoch = opt.n_epoch
        self.start_lr = opt.lr
        self.linear_end_factor = opt.lr_end_factor
        self.l2_reg_w = opt.l2_reg_w
        self.dsc_loss_w = opt.dsc_loss_w
        self.n_classes = 4

        self.CEL = nn.CrossEntropyLoss()
        self.DC = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=False,
            do_bg=False,
            smooth=1e-5,
            ddp=False,
        )

        self.softmax = nn.Softmax(dim=1)
        self.train_step_outputs: dict[str, list] = {}
        self.val_step_outputs: dict[str, list] = {}

    def on_fit_start(self):
        tb = _tb_logger(self).experiment
        layout = {
            "loss_split": {
                "loss_train": [
                    "Multiline",
                    [
                        "loss_train/dice_loss",
                        "loss_train/l2_reg_loss",
                        "loss_train/ce_loss",
                    ],
                ],
                "loss_val": [
                    "Multiline",
                    ["loss_val/dice_loss", "loss_val/l2_reg_loss", "loss_val/ce_loss"],
                ],
            },
            "loss_merge": {
                "loss": ["Multiline", ["loss/train_loss", "loss/val_loss"]],
            },
            "diceFG_merge": {
                "diceFG": ["Multiline", ["diceFG/train_diceFG", "diceFG/val_diceFG"]],
            },
        }
        tb.add_custom_scalars(layout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch):
        target, gt = batch["target"], batch["class"]
        losses, gt, pred_cls = self._shared_step(target, gt)
        loss = self._merge_losses(losses)

        self.log(
            "loss/train_loss",
            loss.detach().cpu(),
            batch_size=target.shape[0],
            prog_bar=True,
        )
        for k, v in losses.items():
            self.log(f"loss_train/{k}", v.detach().cpu(), batch_size=target.shape[0])

        self._append_metrics(self._compute_metrics(loss, pred_cls, gt), self.train_step_outputs)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.train_step_outputs:
            metrics = self._aggregate_metrics(self.train_step_outputs)
            self.log("dice/train_dice", metrics["dice"], on_epoch=True)
            self.log("diceFG/train_diceFG", metrics["diceFG"], on_epoch=True)
            _tb_logger(self).experiment.add_text(
                "train_dice_p_cls",
                str(metrics["dice_p_cls"].tolist()),
                self.current_epoch,
            )
        self.train_step_outputs.clear()

    def validation_step(self, batch, _):
        target, gt = batch["target"], batch["class"]
        losses, gt, pred_cls = self._shared_step(target, gt)
        loss = self._merge_losses(losses).detach().cpu()

        metrics = self._compute_metrics(loss, pred_cls, gt)
        for k, v in losses.items():
            metrics[k] = v.detach().cpu()
        self._append_metrics(metrics, self.val_step_outputs)

    def on_validation_epoch_end(self):
        if self.val_step_outputs:
            metrics = self._aggregate_metrics(self.val_step_outputs)
            for k, v in metrics.items():
                if "loss" in k:
                    self.log(
                        "loss/val_loss" if k == "loss" else f"loss_val/{k}",
                        v,
                        on_epoch=True,
                    )
            self.log("dice/val_dice", metrics["dice"], on_epoch=True)
            self.log("diceFG/val_diceFG", metrics["diceFG"], on_epoch=True)
            _tb_logger(self).experiment.add_text(
                "val_dice_p_cls",
                str(metrics["dice_p_cls"].tolist()),
                self.current_epoch,
            )
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=self.linear_end_factor,
            total_iters=self.n_epoch,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_losses(self, logits: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        ce_loss = self.CEL(logits, gt.squeeze(1))
        dc_loss = (1 - self.DC(logits, gt)) * self.dsc_loss_w
        l2_reg = torch.stack([p.norm() for p in self.parameters()]).sum() * self.l2_reg_w
        return {"ce_loss": ce_loss, "dc_loss": dc_loss, "l2_reg_loss": l2_reg}

    def _merge_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        vals = list(losses.values())
        return vals[0] + vals[1] + vals[2]

    def _shared_step(self, target: torch.Tensor, gt: torch.Tensor):
        logits = self.forward(target)
        losses = self._compute_losses(logits, gt)

        with torch.inference_mode():
            pred_cls = torch.argmax(logits, dim=1).cpu()
            gt = gt.cpu()

        return losses, gt, pred_cls

    def _compute_metrics(self, loss: torch.Tensor, pred_cls: torch.Tensor, gt: torch.Tensor) -> dict:
        dice_p_cls = mF.dice(pred_cls, gt, average=None, num_classes=self.n_classes)
        return {
            "loss": loss,
            "dice": dice_p_cls.mean(),
            "diceFG": dice_p_cls[1:].mean(),
            "dice_p_cls": dice_p_cls,
        }

    def _append_metrics(self, metrics: dict, outputs: dict):
        for k, v in metrics.items():
            outputs.setdefault(k, []).append(v)

    def _aggregate_metrics(self, outputs: dict) -> dict:
        return {k: torch.mean(torch.stack(v)) if k != "dice_p_cls" else torch.mean(torch.stack(v), dim=0) for k, v in outputs.items()}

    def __str__(self):
        return f"Unet_{'2D' if self.do2D else '3D'}"
