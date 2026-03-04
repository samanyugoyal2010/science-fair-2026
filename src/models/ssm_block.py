from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SSMConfig:
    d_model: int = 256
    state_size: int = 64
    dropout: float = 0.1


class SimpleSelectiveSSM(nn.Module):
    """Minimal selective SSM-style recurrent block using stable gated updates."""

    def __init__(self, cfg: SSMConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, 3 * cfg.state_size)
        self.out_proj = nn.Linear(cfg.state_size, cfg.d_model)
        self.skip = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        u = self.norm(x)
        gates = self.in_proj(u)
        delta, b_proj, c_proj = gates.chunk(3, dim=-1)

        h = torch.zeros(b, self.cfg.state_size, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(t):
            alpha = torch.sigmoid(delta[:, i, :])
            h = (1.0 - alpha) * h + alpha * b_proj[:, i, :]
            y_i = h * torch.tanh(c_proj[:, i, :])
            ys.append(y_i.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        y = self.out_proj(y)
        y = y + self.skip(x)
        return self.dropout(y)
