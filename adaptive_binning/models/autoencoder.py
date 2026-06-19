from __future__ import annotations

import torch
from torch import nn


def _make_mlp(d_in: int, d_out: int, width: int, depth: int, dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(d_in)
    for _ in range(int(depth)):
        layers.append(nn.Linear(prev, int(width)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = int(width)
    layers.append(nn.Linear(prev, int(d_out)))
    return nn.Sequential(*layers)


class AdaptiveBinningModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_features: list[int],
        num_features: list[int],
        cat_cardinalities: list[int],
        width: int,
        depth: int,
        max_bins: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.cat_cardinalities = [int(x) for x in cat_cardinalities]
        self.width = int(width)
        self.max_bins = int(max_bins)

        self.encoder = _make_mlp(self.input_dim, self.width, self.width, depth, dropout)
        self.decoder = _make_mlp(self.width, self.width, self.width, 1, dropout)
        self.cat_heads = nn.ModuleList([nn.Linear(self.width, c) for c in self.cat_cardinalities])
        self.num_heads = nn.ModuleList([nn.Linear(self.width, self.max_bins) for _ in self.num_features])

    def _select_input(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        if self.num_features:
            pieces.append(x[:, self.num_features])
        if self.cat_features:
            pieces.append(x[:, self.cat_features])
        if not pieces:
            raise ValueError("Model received no input features.")
        return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._select_input(x))

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        h = self.decoder(z)
        return {
            "embedding": z,
            "cat_logits": [head(h) for head in self.cat_heads],
            "num_logits": [head(h) for head in self.num_heads],
        }


class DownstreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, input_selector, width: int, ydim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_selector = input_selector
        self.head = nn.Linear(int(width), int(ydim))
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.input_selector(x))
        return self.head(z)
