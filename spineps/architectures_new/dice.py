"""Memory-efficient soft Dice loss for multi-class segmentation."""

from __future__ import annotations

import torch
from torch import nn


class MemoryEfficientSoftDiceLoss(nn.Module):
    """Soft Dice computed without materializing a full one-hot target when the prediction already matches its shape.

    Returns the mean soft Dice coefficient over classes (and optionally the batch). The caller typically uses
    ``1 - loss`` as the actual loss term.
    """

    def __init__(self, apply_nonlin=None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0, ddp: bool = True):
        """Configure the soft Dice computation.

        Args:
            apply_nonlin (Callable | None): Optional non-linearity (e.g. softmax) applied to ``x`` before the Dice
                computation.
            batch_dice (bool): If ``True``, accumulate the statistics over the whole batch before dividing; otherwise
                compute the Dice per sample.
            do_bg (bool): If ``True``, include the background class (channel 0); otherwise drop it.
            smooth (float): Smoothing constant added to the denominator for numerical stability.
            ddp (bool): Flag retained for distributed-data-parallel all-gather (currently unused in this implementation).
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        """Compute the mean soft Dice coefficient between predictions and targets.

        Args:
            x (torch.Tensor): Prediction tensor of shape ``(b, c, ...)``; the non-linearity is applied first if set.
            y (torch.Tensor): Target tensor, either class indices of shape ``(b, 1, ...)`` / ``(b, ...)`` or a one-hot
                encoding matching the shape of ``x``.
            loss_mask (torch.Tensor | None): Optional mask multiplied into the statistics to ignore certain voxels.

        Returns:
            torch.Tensor: Scalar mean soft Dice coefficient.
        """
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all(i == j for i, j in zip(shp_x, shp_y)):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        # if self.ddp and self.batch_dice:
        #    intersect = AllGatherGrad.apply(intersect)
        #    sum_pred = AllGatherGrad.apply(sum_pred)
        #    sum_gt = AllGatherGrad.apply(sum_gt)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))  # 2* intersect + self.smooth

        dc = dc.mean()
        return dc  # originally negative
