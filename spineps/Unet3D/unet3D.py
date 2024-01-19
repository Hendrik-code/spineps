import torch
import torch.nn as nn
from einops import rearrange
from functools import partial


class Unet3D(nn.Module):
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
        super().__init__()

        self.learned_variance = learned_variance

        self.conditional_label_size = conditional_label_size
        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d((channels + conditional_dimensions), init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

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

    def forward(self, x, time=None, label=None, embedding=None) -> torch.Tensor:  # time
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
        ö = "-"
        for block1, block2, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            if self.first_forward:
                ö += "-"
                print(ö, x.shape, "\t")

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        if self.first_forward:
            print(ö, x.shape)
            ö = ö[:-1]

        for block1, block2, upsample in self.ups:  # type: ignore
            x = 0.5 * (x + h.pop())
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            if self.first_forward:
                print(ö, x.shape, "\t")
                ö = ö[:-1]

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
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block3D(dim, dim_out, groups=groups)
        self.block2 = Block3D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


def default(val, d):
    from inspect import isfunction

    if val is not None:
        return val
    return d() if isfunction(d) else d
