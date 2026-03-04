from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KANConfig:
    in_features: int
    out_features: int
    n_basis: int = 8
    grid_min: float = -2.0
    grid_max: float = 2.0


class KANLinear(nn.Module):
    """Lightweight KAN-style projection via per-feature radial basis expansion."""

    def __init__(self, cfg: KANConfig):
        super().__init__()
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        self.n_basis = cfg.n_basis
        centers = torch.linspace(cfg.grid_min, cfg.grid_max, cfg.n_basis)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.randn(cfg.out_features, cfg.in_features, cfg.n_basis) * 0.02)
        self.bias = nn.Parameter(torch.zeros(cfg.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_features]
        if x.size(-1) != self.in_features:
            raise ValueError(f"Expected input dim {self.in_features}, got {x.size(-1)}")
        width = torch.exp(self.log_width).clamp(min=1e-4)
        # [N, in_features, n_basis]
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) / (2.0 * (width**2)))
        out = torch.einsum("nib,oib->no", basis, self.weight) + self.bias
        return out
