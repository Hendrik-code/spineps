"""PyTorch Lightning wrapper around the 3D U-Net used for spine segmentation training and inference."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics.functional as mF
from torch import nn
from torch.optim import lr_scheduler

from spineps.architectures.unet3D import Unet3D


class PLNet(pl.LightningModule):
    """LightningModule wrapping a :class:`~spineps.architectures.unet3D.Unet3D` for multi-class segmentation.

    Configures a 4-class 3D U-Net with 10 input channels and provides shared loss/metric helpers (Dice scores) and softmax-based
    class prediction.
    """

    def __init__(self, opt=None, do2D: bool = False, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Build the wrapped U-Net and store training hyperparameters.

        Args:
            opt: Options object; ``opt.n_epoch`` sets the number of epochs when provided.
            do2D (bool): Whether the model operates in 2D mode (affects only the string representation).
            *args (Any): Ignored extra positional arguments.
            **kwargs (Any): Ignored extra keyword arguments.
        """
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
        """Run the wrapped U-Net on an input batch.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, 10, D, H, W)``.

        Returns:
            torch.Tensor: Raw class logits of shape ``(B, 4, D, H, W)``.
        """
        return self.network(x)

    def _shared_step(self, target, gt, detach2cpu: bool = False):
        """Run the forward pass and compute loss plus predicted class map for a batch.

        Args:
            target (torch.Tensor): Input batch fed to the network.
            gt (torch.Tensor): Ground-truth class labels.
            detach2cpu (bool): If True, detach ``gt``, ``logits`` and ``pred_cls`` and move them to CPU.

        Returns:
            tuple: ``(loss, logits, gt, pred_cls)`` where ``pred_cls`` is the argmax over the softmax of the logits.
        """
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
        """Compute segmentation metrics (overall, foreground and per-class Dice) for a batch.

        Args:
            loss (torch.Tensor): The batch loss to record.
            _: Unused logits placeholder.
            gt (torch.Tensor): Ground-truth class labels.
            pred_cls (torch.Tensor): Predicted class labels.

        Returns:
            dict: Metrics with keys ``loss``, ``dice``, ``diceFG`` (Dice ignoring the background class) and ``dice_p_cls``.
        """
        dice = mF.dice(pred_cls, gt, num_classes=self.n_classes)
        diceFG = mF.dice(pred_cls, gt, num_classes=self.n_classes, ignore_index=0)
        dice_p_cls = mF.dice(pred_cls, gt, average=None, num_classes=self.n_classes)
        return {"loss": loss.detach().cpu(), "dice": dice, "diceFG": diceFG, "dice_p_cls": dice_p_cls}

    def _shared_metric_append(self, metrics, outputs):
        """Append each metric value to the per-key list of accumulated outputs (in place).

        Args:
            metrics (dict): Metric name to value mapping for one step.
            outputs (dict): Accumulator mapping each metric name to a list of values.
        """
        for k, v in metrics.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)

    def _shared_cat_metrics(self, outputs):
        """Aggregate accumulated per-step metrics into mean values.

        Args:
            outputs (dict): Mapping of metric name to a list of per-step tensors.

        Returns:
            dict: Mean of each metric; ``dice_p_cls`` is averaged along the step dimension to keep per-class values.
        """
        results = {}
        for m, v in outputs.items():
            stacked = torch.stack(v)
            results[m] = torch.mean(stacked) if m != "dice_p_cls" else torch.mean(stacked, dim=0)
        return results

    def __str__(self):
        """Return a short model name including the spatial mode.

        Returns:
            str: ``"Unet_2D"`` or ``"Unet_3D"`` depending on ``do2D``.
        """
        text = "Unet"
        dim = "2D" if self.do2D else "3D"
        return text + "_" + dim


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    """Apply softmax along dimension 1 (the channel/class dimension).

    Args:
        x (torch.Tensor): Input tensor with classes on dimension 1.

    Returns:
        torch.Tensor: Tensor of the same shape with a softmax applied over dimension 1.
    """
    return torch.softmax(x, 1)
