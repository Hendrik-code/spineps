"""DenseNet/ResNet-based classifier (PLClassifier) and model configuration for vertebra labeling."""

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
) -> ResNet:
    """Build a very small 2-stage MONAI ResNet variant ("resnet2").

    Args:
        layers (list[int] | None): Number of blocks per stage; defaults to ``[1, 1]``.
        **kwargs: Additional keyword arguments forwarded to the MONAI ``_resnet`` factory.

    Returns:
        ResNet: The constructed ResNet model.
    """
    if layers is None:
        layers = [1, 1]
    return _resnet("resnet2", ResNetBlock, layers, get_inplanes(), False, False, **kwargs)


class MODEL(Enum):
    """Selectable backbone architectures (DenseNet and ResNet variants) for the vertebra classifier."""

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
    ) -> tuple[nn.Module, int]:
        """Instantiate the selected backbone network.

        Args:
            opt (ARGS_MODEL): Model configuration providing channels, class count and pretraining flag.
            remove_classification_head (bool): If True, strip the backbone's final classification layer so it acts as a
                feature extractor.

        Returns:
            tuple: ``(model, linear_in_features)`` where ``linear_in_features`` is the input feature size of the removed head.

        Raises:
            ValueError: If the enum member is neither a DenseNet nor a ResNet variant.
        """
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
    """Configuration (and argparse schema) for the vertebra labeling classifier, covering backbone, heads and training options."""

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
    """LightningModule that classifies vertebrae using a shared backbone with one classification head per target group.

    The configured backbone (DenseNet/ResNet) acts as a feature extractor, and a separate head is built for each entry in
    ``group_2_n_channel`` to produce that group's class logits.
    """

    def __init__(self, opt: ARGS_MODEL, group_2_n_channel: dict[str, int]):
        """Build the backbone, classification heads and loss/activation modules.

        Args:
            opt (ARGS_MODEL): Model configuration; ``opt.num_classes`` must be an int.
            group_2_n_channel (dict[str, int]): Mapping from each target group name to its number of output channels.

        Raises:
            AssertionError: If ``opt.num_classes`` is not an int.
        """
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

    def forward(self, x) -> dict[str, torch.Tensor]:
        """Extract features with the backbone and apply every classification head.

        Args:
            x (torch.Tensor): Input image batch fed to the backbone.

        Returns:
            dict[str, torch.Tensor]: Mapping from each group name to that head's output logits.
        """
        features = self.net(x)
        return {k: v(features) for k, v in self.classification_heads.items()}

    def build_classification_heads(self, linear_in: int, convolution_first: bool, fully_connected: bool) -> nn.ModuleDict:
        """Build one classification head per target group as a :class:`~torch.nn.ModuleDict`.

        Args:
            linear_in (int): Number of input features coming from the backbone.
            convolution_first (bool): If True, prepend a 3x3x3 convolution that halves the channels before the linear layers.
            fully_connected (bool): If True, insert a hidden linear+ReLU layer (halving channels) before the output layer.

        Returns:
            nn.ModuleDict: Mapping from each group name to its head, each ending in a linear layer with that group's class count.
        """

        def construct_one_head(output_classes: int):
            """Build a single classification head producing ``output_classes`` logits.

            Args:
                output_classes (int): Number of output classes for this head.

            Returns:
                nn.Sequential: The assembled head modules.
            """
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
        """Return the model name.

        Returns:
            str: The fixed name ``"VertebraLabelingModel"``.
        """
        return "VertebraLabelingModel"


def get_densenet_architecture(
    model: object,
    in_channel: int = 1,
    out_channel: int = 1,
    pretrained: bool = True,
    remove_classification_head: bool = True,
) -> tuple[nn.Module, int]:
    """Instantiate a 3D MONAI DenseNet and optionally remove its final classification layer.

    Args:
        model: A MONAI DenseNet constructor (e.g. ``DenseNet121`` or ``DenseNet169``).
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels for the original classification layer.
        pretrained (bool): Whether to load pretrained weights.
        remove_classification_head (bool): If True, drop the final classification layer to use the model as a feature extractor.

    Returns:
        tuple: ``(model, linear_infeatures)`` where ``linear_infeatures`` is the input feature size of the removed head.
    """
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
    model: object,
    remove_classification_head: bool = True,
) -> tuple[nn.Module, int]:
    """Instantiate a 3D MONAI ResNet and optionally remove its fully connected head.

    Args:
        model: A MONAI ResNet constructor (e.g. ``resnet18`` or ``resnet50``).
        remove_classification_head (bool): If True, set the final fully connected layer to None to use the model as a
            feature extractor.

    Returns:
        tuple: ``(model, linear_infeatures)`` where ``linear_infeatures`` is the input feature size of the removed head.
    """
    model = model(
        spatial_dims=3,
        n_input_channels=1,
    )
    linear_infeatures = model.fc.in_features
    if remove_classification_head:
        model.fc = None
    return model, linear_infeatures
