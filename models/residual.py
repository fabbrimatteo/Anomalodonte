import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    """
    A single residual block.
    """

    def __init__(self, in_channels, out_channels, mid_channels, padding_mode):
        # type: (int, int, int) -> None
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.SiLU(),

            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
                padding_mode=padding_mode
            ),
            nn.SiLU(),

            nn.Conv2d(
                mid_channels, out_channels,
                kernel_size=1, stride=1, bias=False
            )
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers.
    """


    def __init__(self, in_channels, out_channels, mid_channels, n_res_layers, padding_mode='zeros'):
        # type: (int, int, int, int) -> None
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers

        layers = [ResidualLayer(in_channels, out_channels, mid_channels, padding_mode)] * n_res_layers
        self.stack = nn.ModuleList(layers)
        self.last_activation = nn.SiLU()


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        for layer in self.stack:
            x = layer(x)
        return self.last_activation(x)
