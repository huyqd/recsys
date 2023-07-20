from torch import nn as nn


class MLP(nn.Module):
    def __init__(self, mlp_dims):
        super().__init__()

        model = []
        for in_dim, out_dim in zip(mlp_dims[:-1], mlp_dims[1:]):
            model.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                ]
            )
        model.pop()
        self.mlp = nn.Sequential(*model)

    def forward(self, x):
        return self.mlp(x)
