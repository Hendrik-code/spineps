from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.networks.nets import DenseNet121, DenseNet169
from monai.networks.nets.resnet import (
    ResNet,
    ResNetBlock,
    _resnet,
    get_inplanes,
    resnet10,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch import nn
from TypeSaveArgParse import Class_to_ArgParse


def resnet2(
    layers: list[int] | None = None,
    **kwargs,
):
    if layers is None:
        layers = [1, 1]
    return _resnet("resnet2", ResNetBlock, layers, get_inplanes(), False, False, **kwargs)


class MODEL(Enum):
    DENSENET169 = DenseNet169
    DENSENET121 = DenseNet121
    RESNET10 = 10  # resnet10
    RESNET18 = 18  # resnet18
    RESNET34 = 34  # resnet34
    RESNET50 = 50  # resnet50
    RESNET101 = 101  # resnet101
    RESNET152 = 152  # resnet152
    RESNET2 = 2  # resnet2

    def __call__(
        self,
        opt: ARGS_MODEL,
        remove_classification_head: bool = True,
    ):
        if "DENSENET" in self.name:
            return get_densenet_architecture(
                self.value,
                in_channel=opt.in_channel,
                out_channel=opt.num_classes,
                pretrained=not opt.not_pretrained,
                remove_classification_head=remove_classification_head,
            )
        elif "RESNET" in self.name:
            d = {
                10: resnet10,
                18: resnet18,
                34: resnet34,
                50: resnet50,
                101: resnet101,
                152: resnet152,
                2: resnet2,
            }
            return get_resnet_architecture(
                d[self.value],
                remove_classification_head=remove_classification_head,
            )
        else:
            raise ValueError(f"Model {self.name} not supported.")


@dataclass
class ARGS_MODEL(Class_to_ArgParse):
    backbone: MODEL = MODEL.DENSENET169.name
    classification_conv: bool = False
    classification_linear: bool = True
    #
    n_epoch: int = 100
    lr: float = 1e-4
    l2_regularization_w: float = 1e-6  # 1e-5 was ok
    scheduler_endfactor: float = 1e-3
    #
    in_channel: int = 1  # 1 for img, will be set elsewhere
    not_pretrained: bool = True
    #
    mse_weighting: float = 0.0
    dropout: float = 0.05
    weight_decay: float = 0  # 1e-4
    #
    num_classes: int | None = None  # Filled elsewhere
    n_channel_p_group: int | None = None  # Filled elsewhere


class PLClassifier(pl.LightningModule):
    def __init__(self, opt: ARGS_MODEL, group_2_n_channel: dict[str, int]):
        super().__init__()
        self.opt = opt
        assert isinstance(opt.num_classes, int), opt.num_classes
        self.num_classes: int = opt.num_classes
        self.group_2_n_channel = group_2_n_channel
        # save hyperparameter, everything below not visible
        self.save_hyperparameters()

        self.backbone = MODEL[opt.backbone]
        self.net, linear_in = self.backbone(opt, remove_classification_head=True)
        self.classification_heads = self.build_classification_heads(linear_in, opt.classification_conv, opt.classification_linear)
        self.classification_keys = list(self.classification_heads.keys())
        self.mse_weighting = opt.mse_weighting

        self.metrics_to_log = ["f1", "mcc", "acc", "auroc", "f1_avg"]
        self.metrics_to_log_overall = ["f1", "f1_avg"]

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.softmax = nn.Softmax(dim=1)  # use this group-wise?
        self.sigmoid = nn.Sigmoid()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction="none")
        self.l2_reg_w = opt.l2_regularization_w

    def forward(self, x):
        features = self.net(x)
        return {k: v(features) for k, v in self.classification_heads.items()}

    def build_classification_heads(self, linear_in: int, convolution_first: bool, fully_connected: bool):
        def construct_one_head(output_classes: int):
            modules = []
            n_channel = linear_in
            n_channel_next = linear_in
            if convolution_first:
                n_channel_next = n_channel // 2
                modules.append(nn.Conv3d(n_channel, n_channel_next, kernel_size=(3, 3, 3), device=self.device))
                n_channel = n_channel_next
            if fully_connected:
                n_channel_next = n_channel // 2
                modules.append(nn.Linear(n_channel, n_channel_next, device=self.device))
                modules.append(nn.ReLU())
                n_channel = n_channel_next
            modules.append(nn.Linear(n_channel, output_classes, device=self.device))

            return nn.Sequential(*modules)

        return nn.ModuleDict({k: construct_one_head(v) for k, v in self.group_2_n_channel.items()})

    def __str__(self) -> str:
        return "VertebraLabelingModel"


def get_densenet_architecture(
    model,
    in_channel: int = 1,
    out_channel: int = 1,
    pretrained: bool = True,
    remove_classification_head: bool = True,
):
    model = model(
        spatial_dims=3,
        in_channels=in_channel,
        out_channels=out_channel,
        pretrained=pretrained,
    )
    linear_infeatures = model.class_layers[-1].in_features
    if remove_classification_head:
        model.class_layers = model.class_layers[:-1]
    return model, linear_infeatures


def get_resnet_architecture(
    model,
    remove_classification_head: bool = True,
):
    model = model(
        spatial_dims=3,
        n_input_channels=1,
    )
    linear_infeatures = model.fc.in_features
    if remove_classification_head:
        model.fc = None
    return model, linear_infeatures
