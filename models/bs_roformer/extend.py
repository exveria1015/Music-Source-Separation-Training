# coding: utf-8
# Copyright 2026 Exveria
# SPDX-License-Identifier: Apache-2.0
#
# Extension-only utilities for experimental modules that need BS-RoFormer
# internals without modifying the native BS-RoFormer implementation.

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

__all__ = ["RMSNorm", "TimeScreeningSelector", "l2norm", "tanh_norm"]


def exists(val):
    return val is not None


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1, p=2)


def tanh_norm(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    return t * (torch.tanh(norm) / norm.clamp(min=eps))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class TimeScreeningSelector(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
        rotary_embed=None,
        norm_values: bool = False,
        use_tanh_norm: bool = True,
        init_window: float = 64.0,
        init_relevance_width: float = 4.0,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.norm = RMSNorm(dim)
        self.rotary_embed = rotary_embed
        self.norm_values = norm_values
        self.use_tanh_norm = use_tanh_norm

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)
        self.to_gates = nn.Linear(dim, heads)

        init_window = max(float(init_window), 1.0001)
        init_relevance_width = max(float(init_relevance_width), 1.0001)

        self.log_window = nn.Parameter(torch.log(torch.full((heads,), init_window - 1.0)))
        self.log_relevance_width = nn.Parameter(torch.log(torch.full((heads,), init_relevance_width - 1.0)))
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        q = l2norm(q)
        k = l2norm(k)

        if self.norm_values:
            v = l2norm(v)

        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k).clamp(min=-1.0, max=1.0)

        relevance_width = (self.log_relevance_width.exp() + 1.0).to(
            dtype=similarity.dtype,
            device=similarity.device,
        )
        relevance_width = rearrange(relevance_width, "h -> 1 h 1 1")
        relevance = torch.clamp(1.0 - relevance_width * (1.0 - similarity), min=0.0).square()

        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device, dtype=similarity.dtype)
        offsets = (positions.view(1, -1) - positions.view(-1, 1)).abs().unsqueeze(0)

        window = (self.log_window.exp() + 1.0).to(dtype=similarity.dtype, device=similarity.device)
        window = rearrange(window, "h -> h 1 1")
        softmask = torch.where(
            offsets < window,
            0.5 * (torch.cos(torch.pi * offsets / window.clamp(min=1.0)) + 1.0),
            torch.zeros_like(offsets),
        )

        relevance = relevance * softmask.unsqueeze(0)
        screened = torch.einsum("b h i j, b h j d -> b h i d", relevance, v)

        if self.use_tanh_norm:
            screened = tanh_norm(screened)

        gates = rearrange(self.to_gates(x).sigmoid(), "b n h -> b h n 1")
        screened = screened * gates

        out = rearrange(screened, "b h n d -> b n (h d)")
        return self.to_out(out) * torch.tanh(self.residual_scale)
