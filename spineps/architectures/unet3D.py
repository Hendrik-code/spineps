"""3D U-Net architecture with residual blocks used for volumetric spine segmentation."""

from __future__ import annotations

from functools import partial
from inspect import isfunction

import torch
from einops import rearrange
from torch import nn


class Unet3D(nn.Module):
    """A 3D U-Net with residual (ResNet) blocks, a symmetric encoder/decoder and skip connections.

    The encoder repeatedly applies two residual blocks followed by a strided convolution that halves each spatial dimension; a
    bottleneck of two residual blocks follows; the decoder mirrors the encoder with transposed convolutions and averages in the
    matching encoder skip features before a final residual block and 1x1x1 output convolution.
    """

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        conditional_dimensions=0,
        resnet_block_groups=8,
        learned_variance=False,
        conditional_label_size=0,
    ):
        """Build the 3D U-Net layers.

        Args:
            dim (int): Base feature dimension used to derive per-level channel counts.
            init_dim (int | None): Channels after the initial convolution; defaults to ``dim``.
            out_dim (int | None): Number of output channels; defaults to ``channels`` (doubled if ``learned_variance``).
            dim_mults (tuple[int, ...]): Per-resolution multipliers of ``dim`` defining encoder/decoder depth and widths.
            channels (int): Number of input image channels.
            conditional_dimensions (int): Extra conditioning channels concatenated to the input at the first convolution.
            resnet_block_groups (int): Number of groups for the GroupNorm inside each residual block.
            learned_variance (bool): If True, doubles the default output channels to also predict variance.
            conditional_label_size (int): Size of an optional conditional label vector (stored but unused in ``forward``).
        """
        super().__init__()

        self.learned_variance = learned_variance

        self.conditional_label_size = conditional_label_size
        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d((channels + conditional_dimensions), init_dim, 7, padding=3)

        dims = [init_dim, *(int(dim * m) for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))  # noqa: RUF007

        block_klass = partial(ResnetBlock3D, groups=resnet_block_groups)
        time_dim = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        nn.Conv3d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        nn.ConvTranspose3d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim) * 1

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)
        self.first_forward = False

    def forward(self, x, time=None, label=None, embedding=None) -> torch.Tensor:  # time  # noqa: ARG002
        """Run the U-Net forward pass on a 5D input volume.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, channels, D, H, W)``; each spatial dimension must be divisible by
                ``2 ** (num_downsampling_levels)``.
            time: Unused timestep input; replaced by a constant if None.
            label: Unused optional conditioning label.
            embedding: Unused optional conditioning embedding.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, out_dim, D, H, W)`` with the same spatial size as the input.

        Raises:
            AssertionError: If any spatial dimension of ``x`` is not divisible by the total downsampling factor.
        """
        down_factor = 2 ** (len(self.downs) - 1)
        shape = x.shape
        assert shape[-1] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-1]}"
        assert shape[-2] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-2]}"
        assert shape[-3] % down_factor == 0, f"dimensions are not dividable by {down_factor}, {shape}, {shape[-3]}"
        if self.first_forward:
            print("|", x.shape)
        if self.first_forward:
            print("|", x.shape)

        # time = None
        if time is None:
            time = torch.ones((1,), device=x.device)
        x = self.init_conv(x)
        r = x.clone()
        if self.first_forward:
            print("-", x.shape)

        t = None

        h = []
        o = "-"
        for block1, block2, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            if self.first_forward:
                o += "-"
                print(o, x.shape, "\t")

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        if self.first_forward:
            print(o, x.shape)
            o = o[:-1]

        for block1, block2, upsample in self.ups:  # type: ignore
            x = 0.5 * (x + h.pop())
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            if self.first_forward:
                print(o, x.shape, "\t")
                o = o[:-1]

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        if self.first_forward:
            print("|", x.shape)

        x = self.final_conv(x)

        if self.first_forward:
            print("|", x.shape)

        if self.first_forward:
            print("|", x.shape)

        self.first_forward = False

        return x


class Block3D(nn.Module):
    """Basic 3D conv block: 3x3x3 convolution, group normalization, optional FiLM-style scale/shift and LeakyReLU."""

    def __init__(self, dim, dim_out, groups=8):
        """Build the conv block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            groups (int): Number of groups for the GroupNorm.
        """
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x, scale_shift=None):
        """Apply convolution, normalization, optional scale/shift modulation and activation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, dim, D, H, W)``.
            scale_shift (tuple[torch.Tensor, torch.Tensor] | None): Optional ``(scale, shift)`` tensors applied as
                ``x * (scale + 1) + shift`` after normalization.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, dim_out, D, H, W)``.
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock3D(nn.Module):
    """Residual block of two :class:`Block3D` layers with a skip connection and optional time-embedding modulation."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        """Build the residual block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            time_emb_dim (int | None): If given, size of a time embedding mapped to per-channel scale and shift parameters.
            groups (int): Number of groups for the GroupNorm in each inner block.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block3D(dim, dim_out, groups=groups)
        self.block2 = Block3D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """Apply the two conv blocks plus residual connection, optionally modulated by a time embedding.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, dim, D, H, W)``.
            time_emb (torch.Tensor | None): Optional time embedding of shape ``(B, time_emb_dim)`` used to produce the
                scale/shift applied in the first inner block.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, dim_out, D, H, W)``.
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


def default(val, d):
    """Return ``val`` if it is not None, otherwise a default value.

    Args:
        val: The candidate value.
        d: The fallback value, or a zero-argument callable that produces it.

    Returns:
        ``val`` if not None; otherwise ``d()`` when ``d`` is callable, else ``d``.
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d
