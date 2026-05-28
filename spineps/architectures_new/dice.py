import torch
from torch import nn


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0, ddp: bool = True):
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
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
