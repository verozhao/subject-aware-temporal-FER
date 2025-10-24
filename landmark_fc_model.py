
import torch
from torch import nn

class LandmarkFCModel(nn.Module):
    """
    A simple fully-connected network for tabular facial landmark features.

    Args:
        input_dim (int): Number of input features (e.g., 10 for 5 (x,y) landmarks).
        num_classes (int): Number of emotion classes.
        hidden_dims (tuple[int,...]): Sizes of hidden layers.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        dims = (input_dim, *hidden_dims, num_classes)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # logits

        self.net = nn.Sequential(*layers)

        # Kaiming init for ReLU MLP
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
