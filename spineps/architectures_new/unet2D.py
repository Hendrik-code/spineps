"""2D U-Net architecture with time, label and embedding conditioning for diffusion-style models."""

from __future__ import annotations

import itertools
import math
from functools import partial

import torch
from einops import rearrange
from torch import nn


def default(val, d):
    """Return ``val`` if it is not ``None``, otherwise the default ``d``.

    Args:
        val: The value to use when it is not ``None``.
        d: The fallback value, or a zero-argument callable producing the fallback.

    Returns:
        ``val`` when it is not ``None``; otherwise ``d()`` if ``d`` is a function, else ``d``.
    """
    from inspect import isfunction

    if val is not None:
        return val
    return d() if isfunction(d) else d


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal positional embedding used to encode the diffusion time step."""

    def __init__(self, dim):
        """Initialize the embedding.

        Args:
            dim (int): Output embedding dimension. Half is used for the sine and half for the cosine components.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Compute the sinusoidal embedding for the given scalar values.

        Args:
            x (torch.Tensor): Tensor of shape ``(b,)`` with the values (e.g. time steps) to embed.

        Returns:
            torch.Tensor: Embedding of shape ``(b, dim)``.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        """Initialize the learned embedding.

        Args:
            dim (int): Output embedding dimension; must be even. ``dim // 2`` learnable frequencies are used.

        Raises:
            AssertionError: If ``dim`` is not even.
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))  # type: ignore

    def forward(self, x):
        """Compute the learned Fourier embedding for the given scalar values.

        Args:
            x (torch.Tensor): Tensor of shape ``(b,)`` with the values (e.g. time steps) to embed.

        Returns:
            torch.Tensor: Embedding of shape ``(b, dim + 1)`` concatenating the input with its sine and cosine features.
        """
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class PreNorm(nn.Module):
    """Apply layer normalization to the input before passing it through a wrapped module."""

    def __init__(self, dim, fn):
        """Initialize the pre-normalization wrapper.

        Args:
            dim (int): Number of channels to normalize over.
            fn (nn.Module): Module applied to the normalized input.
        """
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        """Normalize ``x`` and forward it (with any extra arguments) through the wrapped module.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, c, h, w)``.
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            torch.Tensor: Output of the wrapped module applied to the normalized input.
        """
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Residual(nn.Module):
    """Add a skip connection around a wrapped module."""

    def __init__(self, fn):
        """Initialize the residual wrapper.

        Args:
            fn (nn.Module): Module whose output is added to its input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """Forward ``x`` through the wrapped module and add the input as a residual.

        Args:
            x (torch.Tensor): Input tensor.
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            torch.Tensor: ``fn(x) + x``.
        """
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    """Channel-wise layer normalization for 4D ``(b, c, h, w)`` tensors with learnable scale and bias."""

    def __init__(self, dim, eps=1e-5):
        """Initialize the layer normalization.

        Args:
            dim (int): Number of channels to normalize over.
            eps (float): Small constant added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))  # type: ignore
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))  # type: ignore

    def forward(self, x):
        """Normalize ``x`` over the channel dimension and apply the learnable scale and bias.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, c, h, w)``.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as ``x``.
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class Block(nn.Module):
    """Convolution, group normalization and SiLU activation, with optional FiLM-style scale/shift modulation."""

    def __init__(self, dim, dim_out, groups=8):
        """Initialize the block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            groups (int): Number of groups for group normalization.
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """Apply convolution, normalization, optional modulation and activation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, dim, h, w)``.
            scale_shift (tuple[torch.Tensor, torch.Tensor] | None): Optional ``(scale, shift)`` tensors used to
                modulate the normalized features as ``x * (scale + 1) + shift``.

        Returns:
            torch.Tensor: Output tensor of shape ``(b, dim_out, h, w)``.
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Residual block of two convolutional blocks with optional time-embedding conditioning."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        """Initialize the residual block.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            time_emb_dim (int | None): Dimension of the time embedding. If given, an MLP produces per-channel
                scale and shift values; if ``None``, no time conditioning is applied.
            groups (int): Number of groups for the group normalization in each block.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """Apply the two blocks with optional time conditioning and a residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, dim, h, w)``.
            time_emb (torch.Tensor | None): Optional time embedding of shape ``(b, time_emb_dim)`` used to derive
                the scale/shift modulation of the first block.

        Returns:
            torch.Tensor: Output tensor of shape ``(b, dim_out, h, w)``.
        """
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """Multi-head linear attention over spatial positions with linear complexity in the number of pixels."""

    def __init__(self, dim, heads=4, dim_head=32):
        """Initialize the linear attention module.

        Args:
            dim (int): Number of input and output channels.
            heads (int): Number of attention heads.
            dim_head (int): Channel dimension per attention head.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        """Apply linear attention to the spatial feature map.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, dim, h, w)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(b, dim, h, w)``.
        """
        _b, _c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Standard multi-head softmax self-attention over spatial positions."""

    def __init__(self, dim, heads=4, dim_head=32):
        """Initialize the attention module.

        Args:
            dim (int): Number of input and output channels.
            heads (int): Number of attention heads.
            dim_head (int): Channel dimension per attention head.
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """Apply full self-attention to the spatial feature map.

        Args:
            x (torch.Tensor): Input tensor of shape ``(b, dim, h, w)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(b, dim, h, w)``.
        """
        _b, _c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet2D(nn.Module):
    """2D U-Net with residual blocks, (linear) attention and time/label/embedding conditioning.

    The network down-samples through a configurable number of resolution stages, applies a bottleneck with
    full attention and up-samples again using skip connections. The diffusion time step is encoded via a
    sinusoidal (or learned-sinusoidal) embedding, and the model can additionally be conditioned on a class
    label and/or an external embedding vector. Optional patching folds spatial patches into the channel
    dimension to reduce the spatial resolution processed by the network.
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
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=0,
        conditional_label_size=0,
        conditional_embedding_size=0,
        patch_size=1,  # Improving Diffusion Model Efficiency Through Patching https://arxiv.org/abs/2207.04316; 1 means deactivated (Note: Increases Training difficulty by a lot!)
    ):
        """Build the 2D U-Net layers.

        Args:
            dim (int): Base feature dimension used to derive the channel widths and the time embedding size.
            init_dim (int | None): Channels produced by the initial convolution. Defaults to ``dim``.
            out_dim (int | None): Number of output channels before patch unfolding. Defaults to ``channels``
                (doubled when ``learned_variance`` is set).
            dim_mults (tuple): Channel multipliers applied to ``dim`` for the successive resolution stages.
            channels (int): Number of image channels.
            conditional_dimensions (int): Number of additional conditioning channels concatenated to the input.
            resnet_block_groups (int): Number of groups for the group normalization inside the residual blocks.
            learned_variance (bool): If ``True``, double the default output channels to also predict a variance.
            learned_sinusoidal_cond (bool): If ``True``, use a learned sinusoidal time embedding instead of a fixed one.
            learned_sinusoidal_dim (int): Dimension of the learned sinusoidal embedding when enabled.
            conditional_label_size (int): Number of classes for label conditioning; 0 disables it.
            conditional_embedding_size (int): Size of an external embedding concatenated to the time embedding; 0 disables it.
            patch_size (int): Spatial patch size folded into channels; 1 disables patching.
        """
        super().__init__()
        self.patch_size = patch_size
        self.learned_variance = learned_variance
        self.conditional_label_size = conditional_label_size
        self.conditional_dimensions = conditional_dimensions
        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        # print(init_dim, channels)
        self.init_conv = nn.Conv2d((channels + conditional_dimensions) * patch_size * patch_size, init_dim, 7, padding=3)

        dims = [init_dim, *(int(dim * m) for m in dim_mults)]
        in_out = list(itertools.pairwise(dims))

        res_block = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinus_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinus_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinus_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        if conditional_label_size != 0:
            self.label_emb = nn.Embedding(conditional_label_size, time_dim)

        self.conditional_embedding_size = conditional_embedding_size
        if conditional_embedding_size:
            time_dim += conditional_embedding_size
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        res_block(dim_in, dim_out, time_emb_dim=time_dim),
                        res_block(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = res_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = res_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        res_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        res_block(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim) * patch_size * patch_size

        self.final_res_block = res_block(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    # Improving Diffusion Model Efficiency Through Patching https://arxiv.org/abs/2207.04316 (Note: Increases Training difficulty by a lot!)
    def to_patches(self, x):
        """Fold ``patch_size x patch_size`` spatial patches into the channel dimension.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C, H, W)`` with ``H`` and ``W`` divisible by ``patch_size``.

        Returns:
            torch.Tensor: Tensor of shape ``(B, C * patch_size**2, H // patch_size, W // patch_size)``.
        """
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H, W // p, C * p)
        x = x.permute(0, 2, 1, 3).reshape(B, W // p, H // p, C * p * p)
        return x.permute(0, 3, 2, 1)

    def from_patches(self, x):
        """Invert :meth:`to_patches`, unfolding the channel dimension back into spatial patches.

        Args:
            x (torch.Tensor): Patched tensor of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Tensor of shape ``(B, C // patch_size**2, H * patch_size, W * patch_size)``.
        """
        p = self.patch_size
        B, C, H, W = x.shape

        x = x.permute(0, 3, 2, 1).reshape(B, W, H * p, C // p)
        x = x.permute(0, 2, 1, 3).reshape(B, H * p, W * p, C // (p * p))
        return x.permute(0, 3, 1, 2)

    def forward(self, x, time=None, label=None, embedding=None) -> torch.Tensor:
        """Run the U-Net forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape ``(b, channels (+ conditional_dimensions), h, w)``.
            time (torch.Tensor | None): Diffusion time steps of shape ``(b,)``. Defaults to a tensor of ones.
            label (torch.Tensor | None): Class labels of shape ``(b,)``; required if the model was built with
                ``conditional_label_size != 0``.
            embedding (torch.Tensor | None): External conditioning embedding; required if the model was built
                with ``conditional_embedding_size != 0``.

        Returns:
            torch.Tensor: Output tensor of shape ``(b, out_dim, h, w)``.

        Raises:
            AssertionError: If a required ``label`` or ``embedding`` is not provided.
        """
        if self.patch_size != 1:
            x = self.to_patches(x)

        x = self.init_conv(x)
        r = x.clone()

        if time is None:
            time = torch.ones((1,), device=x.device)

        t = self.time_mlp(time)
        if hasattr(self, "label_emb"):
            assert label is not None, "This UNet requires a class label"
            t = t + self.label_emb(label)

        if self.conditional_embedding_size != 0:
            assert embedding is not None, "This UNet requires a embedding"
            # This is a general implementation, you my specialize this to your needs. The + operator instead of cat is possible.
            t = torch.cat([embedding, t], dim=-1)

        h = []

        for block1, block2, attn, downsample in self.downs:  # type: ignore
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        if self.patch_size != 1:
            x = self.from_patches(x)
        return x
