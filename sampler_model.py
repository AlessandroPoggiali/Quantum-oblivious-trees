import torch
import torch.nn as nn


class SamplerThresholds(nn.Module):
    """Direct threshold optimization via learnable parameters.

    Each threshold is a single scalar parameter initialized from U(0,1),
    passed through sigmoid to stay in [0,1]. This is the simplest possible
    threshold generator: d free parameters, no network overhead.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.raw_thresholds = nn.Parameter(torch.rand(d))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_thresholds)
