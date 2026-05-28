from __future__ import annotations

import itertools
from functools import partial

import torch
from torch import nn


class Block3D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class ResnetBlock3D(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, groups: int = 8):
        super().__init__()
        self.block1 = Block3D(dim, dim_out, groups=groups)
        self.block2 = Block3D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.res_conv(x)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4, 8),
        channels: int = 1,
        conditional_dimensions: int = 0,
        resnet_block_groups: int = 8,
    ):
        super().__init__()
        self.channels = channels

        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv3d(channels + conditional_dimensions, init_dim, 7, padding=3)

        dims = [init_dim, *(int(dim * m) for m in dim_mults)]
        in_out = list(itertools.pairwise(dims))

        block = partial(ResnetBlock3D, groups=resnet_block_groups)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_out),
                        block(dim_out, dim_out),
                        nn.Conv3d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_block2 = block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out, dim_in),
                        block(dim_in, dim_in),
                        nn.ConvTranspose3d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.out_dim = out_dim if out_dim is not None else channels
        self.final_res_block = block(dim * 2, dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_factor = 2 ** (len(self.downs) - 1)
        for i in (-1, -2, -3):
            assert x.shape[i] % down_factor == 0, f"Spatial dim {x.shape[i]} not divisible by {down_factor}, input shape={x.shape}"

        x = self.init_conv(x)
        r = x.clone()

        skip_connections = []
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for block1, block2, upsample in self.ups:
            x = 0.5 * (x + skip_connections.pop())
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x)
        return self.final_conv(x)
