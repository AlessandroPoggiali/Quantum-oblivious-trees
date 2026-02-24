import torch
import torch.nn as nn


# ------------------------ Classical Module ------------------------

class ClassicalThresholds(nn.Module):
    def __init__(self, d: int, hidden_layers: int = 1, hidden_size: int = 32, use_bias: bool = True):
        super().__init__()
        self.d = d
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.use_bias = use_bias
        if hidden_layers < 1:
            self.net = nn.Sequential(nn.Linear(1, d, bias=self.use_bias), nn.Sigmoid())
        else:
            layers = [nn.Linear(1, hidden_size, bias=self.use_bias), nn.ReLU()]
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=self.use_bias))
                layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
            self.net.append(nn.Linear(hidden_size, d, bias=self.use_bias))
            self.net.append(nn.Sigmoid())  # Output thresholds in [0,1]
            # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                

    def forward(self) -> torch.Tensor:
        # Dummy input
        dummy_input = torch.ones(1, 1)
        thresholds = self.net(dummy_input).squeeze(0)
        return thresholds

