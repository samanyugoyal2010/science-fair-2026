import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kan import KANConfig, KANLinear
from src.models.ssm_block import SSMConfig, SimpleSelectiveSSM


@dataclass
class HybridConfig:
    vocab_size: int = 257
    d_model: int = 256
    n_heads: int = 8
    n_local_layers: int = 3
    n_ssm_layers: int = 3
    ssm_state_size: int = 64
    dropout: float = 0.1
    max_seq_len: int = 1024
    local_window: int = 128
    fusion_mode: str = "learned"  # learned|sum|kan
    kan_basis: int = 8
    disable_local_stream: bool = False
    disable_ssm_stream: bool = False


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, window_size: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _mask(self, t: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(t, device=device)
        causal = idx[None, :] <= idx[:, None]
        local = (idx[:, None] - idx[None, :]) < self.window_size
        return ~(causal & local)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._mask(t, x.device)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.proj(y)
        return self.dropout(y)


class LocalBlock(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SlidingWindowAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.local_window)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SplitStreamHybridLM(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.local_blocks = nn.ModuleList([LocalBlock(cfg) for _ in range(cfg.n_local_layers)])
        self.ssm_blocks = nn.ModuleList(
            [SimpleSelectiveSSM(SSMConfig(cfg.d_model, cfg.ssm_state_size, cfg.dropout)) for _ in range(cfg.n_ssm_layers)]
        )

        if cfg.fusion_mode == "learned":
            self.fuse = nn.Linear(2 * cfg.d_model, cfg.d_model)
        elif cfg.fusion_mode == "kan":
            self.fuse = KANLinear(
                KANConfig(in_features=2 * cfg.d_model, out_features=cfg.d_model, n_basis=cfg.kan_basis)
            )
        else:
            self.fuse = None
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")
        pos = torch.arange(0, t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])

        local_out = x
        if not self.cfg.disable_local_stream:
            for block in self.local_blocks:
                local_out = block(local_out)

        ssm_out = x
        if not self.cfg.disable_ssm_stream:
            for block in self.ssm_blocks:
                ssm_out = ssm_out + block(ssm_out)

        if self.cfg.disable_local_stream:
            fused = ssm_out
        elif self.cfg.disable_ssm_stream:
            fused = local_out
        elif self.cfg.fusion_mode == "sum":
            fused = local_out + ssm_out
        else:
            cat = torch.cat([local_out, ssm_out], dim=-1)
            if self.cfg.fusion_mode == "kan":
                fused = self.fuse(cat.view(b * t, -1)).view(b, t, -1)
            else:
                fused = self.fuse(cat)

        y = self.ln_f(fused + x)
        logits = self.lm_head(y)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
