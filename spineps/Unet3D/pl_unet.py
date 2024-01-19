from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchmetrics.functional as mF

from spineps.Unet3D.unet3D import Unet3D


class PLNet(pl.LightningModule):
    def __init__(self, opt=None, do2D: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()

        nclass = Unet3D

        dim_mults = (1, 2, 4, 8)
        dim = 8

        # if opt.high_res:
        #    dim = 16
        #    dim_mults = (2, 4, 8, 8)

        self.network = nclass(
            dim=dim,
            dim_mults=dim_mults,
            out_dim=4,
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

    def training_step(self, batch):
        target = batch["target"]
        gt = batch["class"]
        loss, logits, gt, pred_cls = self._shared_step(target, gt, detach2cpu=True)
        metrics = self._shared_metric_step(loss, logits, gt, pred_cls)
        self.log("train_loss", loss.detach().cpu(), batch_size=target.shape[0], prog_bar=True)
        self._shared_metric_append(metrics, self.train_step_outputs)
        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_outputs) > 0:
            metrics = self._shared_cat_metrics(self.train_step_outputs)

            self.log("train_dice", metrics["dice"], on_epoch=True)
            self.log("train_diceFG", metrics["diceFG"], on_epoch=True)
            self.logger.experiment.add_text("train_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, logits, gt, pred_cls = self._shared_step(batch["target"], batch["class"], detach2cpu=True)
        loss = loss.detach().cpu()
        metrics = self._shared_metric_step(loss, logits, gt, pred_cls)
        self._shared_metric_append(metrics, self.val_step_outputs)

    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) > 0:
            metrics = self._shared_cat_metrics(self.val_step_outputs)

            self.log("val_loss", metrics["loss"], on_epoch=True)

            self.log("val_dice", metrics["dice"], on_epoch=True)
            self.log("val_diceFG", metrics["diceFG"], on_epoch=True)
            self.logger.experiment.add_text("val_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=self.linear_end_factor, total_iters=self.n_epoch
        )
        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def loss(self, logits, gt):
        return 0.0  # TODO don't use this for training

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

    def _shared_metric_step(self, loss, logits, gt, pred_cls):
        dice = mF.dice(pred_cls, gt, num_classes=self.n_classes)
        diceFG = mF.dice(pred_cls, gt, num_classes=self.n_classes, ignore_index=0)
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
            # v = np.asarray(v)
            # print(m, v.shape)
            stacked = torch.stack(v)
            results[m] = torch.mean(stacked) if m != "dice_p_cls" else torch.mean(stacked, dim=0)
        return results

    def __str__(self):
        text = "Unet"
        dim = "2D" if self.do2D else "3D"
        return text + "_" + dim


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)
