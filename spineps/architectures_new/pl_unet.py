"""PyTorch Lightning wrapper training a 2D or 3D U-Net for spine segmentation."""

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
    """Apply softmax over dimension 1 (the channel/class dimension).

    Args:
        x (torch.Tensor): Logits tensor with the classes on dimension 1.

    Returns:
        torch.Tensor: Softmax probabilities of the same shape as ``x``.
    """
    return torch.softmax(x, 1)


def _tb_logger(module: pl.LightningModule) -> TensorBoardLogger:
    """Return the module's logger cast to :class:`TensorBoardLogger`.

    Args:
        module (pl.LightningModule): Lightning module whose ``logger`` is a TensorBoard logger.

    Returns:
        TensorBoardLogger: The module's logger typed as a TensorBoard logger.
    """
    return cast(TensorBoardLogger, module.logger)


class PLNet(pl.LightningModule):
    """LightningModule training a 2D or 3D U-Net with a combined cross-entropy, Dice and L2 loss.

    Wraps :class:`Unet2D` or :class:`Unet3D` and handles the training/validation loops, loss computation,
    Dice metric logging and optimizer configuration.
    """

    def __init__(self, opt: Namespace, do2D: bool = False, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Build the network and configure losses, metrics and training hyperparameters.

        Args:
            opt (Namespace): Configuration namespace providing ``channelwise``, ``n_epoch``, ``lr``,
                ``lr_end_factor``, ``l2_reg_w`` and ``dsc_loss_w``.
            do2D (bool): If ``True``, use the 2D U-Net; otherwise the 3D U-Net.
            *args (Any): Unused positional arguments.
            **kwargs (Any): Unused keyword arguments.
        """
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
        """Register custom TensorBoard scalar layouts grouping the train/val losses and Dice metrics."""
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
        """Run the wrapped U-Net on the input.

        Args:
            x (torch.Tensor): Input image/volume tensor.

        Returns:
            torch.Tensor: Per-class logits produced by the network.
        """
        return self.network(x)

    def training_step(self, batch):
        """Run a single training step: compute losses, log them and accumulate metrics.

        Args:
            batch (dict): Batch with the input image under ``"target"`` and the ground-truth labels under ``"class"``.

        Returns:
            torch.Tensor: The combined scalar training loss to back-propagate.
        """
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
        """Aggregate the accumulated training metrics, log mean/foreground Dice and clear the buffers."""
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
        """Run a single validation step, computing losses and metrics and accumulating them for the epoch end.

        Args:
            batch (dict): Batch with the input image under ``"target"`` and the ground-truth labels under ``"class"``.
            _ : Batch index (unused).
        """
        target, gt = batch["target"], batch["class"]
        losses, gt, pred_cls = self._shared_step(target, gt)
        loss = self._merge_losses(losses).detach().cpu()

        metrics = self._compute_metrics(loss, pred_cls, gt)
        for k, v in losses.items():
            metrics[k] = v.detach().cpu()
        self._append_metrics(metrics, self.val_step_outputs)

    def on_validation_epoch_end(self):
        """Aggregate the accumulated validation metrics, log losses and Dice scores and clear the buffers."""
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
        """Configure the Adam optimizer and a linear learning-rate decay schedule.

        Returns:
            dict: Mapping with the ``"optimizer"`` and its ``"lr_scheduler"``.
        """
        optimizer = Adam(self.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=self.linear_end_factor,
            total_iters=self.n_epoch,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_losses(self, logits: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the cross-entropy, (weighted) Dice and L2 regularization losses.

        Args:
            logits (torch.Tensor): Per-class logits of shape ``(b, n_classes, ...)``.
            gt (torch.Tensor): Ground-truth labels of shape ``(b, 1, ...)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys ``"ce_loss"``, ``"dc_loss"`` and ``"l2_reg_loss"``.
        """
        ce_loss = self.CEL(logits, gt.squeeze(1))
        dc_loss = (1 - self.DC(logits, gt)) * self.dsc_loss_w
        l2_reg = torch.stack([p.norm() for p in self.parameters()]).sum() * self.l2_reg_w
        return {"ce_loss": ce_loss, "dc_loss": dc_loss, "l2_reg_loss": l2_reg}

    def _merge_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum the three loss components into a single scalar loss.

        Args:
            losses (dict[str, torch.Tensor]): Dictionary with exactly three loss tensors.

        Returns:
            torch.Tensor: The sum of the three loss values.
        """
        vals = list(losses.values())
        return vals[0] + vals[1] + vals[2]

    def _shared_step(self, target: torch.Tensor, gt: torch.Tensor):
        """Forward the input, compute losses and produce CPU predictions and ground truth.

        Args:
            target (torch.Tensor): Input image/volume tensor.
            gt (torch.Tensor): Ground-truth labels.

        Returns:
            tuple: ``(losses, gt, pred_cls)`` where ``losses`` is the loss dict, ``gt`` is the ground truth moved
            to CPU and ``pred_cls`` is the per-voxel arg-max class prediction on CPU.
        """
        logits = self.forward(target)
        losses = self._compute_losses(logits, gt)

        with torch.inference_mode():
            pred_cls = torch.argmax(logits, dim=1).cpu()
            gt = gt.cpu()

        return losses, gt, pred_cls

    def _compute_metrics(self, loss: torch.Tensor, pred_cls: torch.Tensor, gt: torch.Tensor) -> dict:
        """Compute per-class, mean and foreground Dice scores alongside the given loss.

        Args:
            loss (torch.Tensor): Scalar loss value to carry through into the metrics dict.
            pred_cls (torch.Tensor): Predicted class indices.
            gt (torch.Tensor): Ground-truth class indices.

        Returns:
            dict: Dictionary with keys ``"loss"``, ``"dice"`` (mean over all classes), ``"diceFG"`` (mean over
            foreground classes) and ``"dice_p_cls"`` (per-class Dice).
        """
        dice_p_cls = mF.dice(pred_cls, gt, average=None, num_classes=self.n_classes)
        return {
            "loss": loss,
            "dice": dice_p_cls.mean(),
            "diceFG": dice_p_cls[1:].mean(),
            "dice_p_cls": dice_p_cls,
        }

    def _append_metrics(self, metrics: dict, outputs: dict):
        """Append each metric value to the matching list in the per-epoch output buffer.

        Args:
            metrics (dict): Metrics computed for a single step.
            outputs (dict): Accumulator mapping each metric name to a list of per-step values; mutated in place.
        """
        for k, v in metrics.items():
            outputs.setdefault(k, []).append(v)

    def _aggregate_metrics(self, outputs: dict) -> dict:
        """Average the accumulated per-step metrics over an epoch.

        Args:
            outputs (dict): Accumulator mapping each metric name to a list of per-step tensors.

        Returns:
            dict: Mapping from metric name to its mean; ``"dice_p_cls"`` is averaged per class (``dim=0``) while
            all other metrics are reduced to a scalar.
        """
        return {k: torch.mean(torch.stack(v)) if k != "dice_p_cls" else torch.mean(torch.stack(v), dim=0) for k, v in outputs.items()}

    def __str__(self):
        """Return a short name indicating whether the wrapped network is 2D or 3D.

        Returns:
            str: ``"Unet_2D"`` or ``"Unet_3D"``.
        """
        return f"Unet_{'2D' if self.do2D else '3D'}"
