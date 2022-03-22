"""
This code is adapted from: https://github.com/lucidrains/perceiver-pytorch.
"""

from math import pi
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(
        1.0, max_freq / 2, num_bands, device=device, dtype=dtype
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# main class


class LatentCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands: int,
        depth: int,
        max_freq: int,
        input_channels: int,
        input_axis: int,
        latent_dim: int,
        cross_heads: int,
        cross_dim_head: int,
        num_classes: int,
        attn_dropout: float,
        ff_dropout: float,
        weight_tie_layers: bool,
        fourier_encode_data: bool,
        final_classifier_head: bool,
    ):
        """
        :param num_freq_bands: number of freq bands original value (2 * K + 1).
        :param depth: depth of net.
        :param max_freq: hyperparameter depending on how fine the data is.
        :param freq_base: base for the frequency
        :param input_channels: number of channels for each token of the input.
        :param input_axis: number of input axes (2 for images, 3 for video).
        :param num_latents: number of latents.
        :param latent_dim: latent dimension.
        :param cross_heads: number of heads for cross attention.
        :param cross_dim_head: number of dimensions per cross attention head.
        :param num_classes: output number of classes.
        :param attn_dropout: attention dropout
        :param ff_dropout: feedforward dropout
        :param weight_tie_layers: whether to weight tie layers (optional).
        :param fourier_encode_data: whether to auto-fourier encode the data,
            using the input_axis given.
        :param final_classifier_head: mean pool and project embeddings to
            number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (
            (input_axis * ((num_freq_bands * 2) + 1))
            if fourier_encode_data
            else 0
        )
        input_dim = fourier_channels + input_channels

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                input_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=input_dim,
        )
        get_cross_ff = lambda: PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout)
        )

        get_cross_attn, get_cross_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                    ]
                )
            )
        self.to_logits = (
            nn.Sequential(
                Reduce("b n d -> b d", "mean"),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes),
                nn.Sigmoid(),
            )
            if final_classifier_head
            else nn.Identity()
        )

    def _encode_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Calculate fourier encoded positions in the range of [-1, 1], for
        all axis.
        """
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        axis_pos = list(
            map(
                lambda size: torch.linspace(
                    -1.0, 1.0, steps=size, device=device, dtype=dtype
                ),
                axis,
            )
        )
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
        enc_pos = repeat(enc_pos, "... -> b ...", b=b)

        data = torch.cat((data, enc_pos), dim=-1)

        return data

    def forward(
        self,
        latents: torch.Tensor,
        data: torch.Tensor,
        mask: torch.Tensor = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        # Encode or not data
        data = self._encode_data(data) if self.fourier_encode_data else data

        # Concat to channels of data and flatten axis
        data = rearrange(data, "b ... d -> b (...) d")
        x = repeat(latents, "n d -> b n d", b=data.shape[0])

        # Layers
        for cross_attn, cross_ff in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

        # Allow for fetching embeddings
        if return_embeddings:
            return x

        # To logits
        return self.to_logits(x)


def make_latent_ca() -> nn.Module:
    """Load a latent cross attention model."""
    model = LatentCrossAttention(
        input_channels=2,
        input_axis=2,
        num_freq_bands=6,
        max_freq=10.0,
        depth=1,
        latent_dim=1024,
        cross_heads=1,
        cross_dim_head=32,
        num_classes=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        fourier_encode_data=True,
        final_classifier_head=True,
    )
    return model
