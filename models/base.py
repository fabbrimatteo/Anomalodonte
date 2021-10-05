# -*- coding: utf-8 -*-
# ---------------------

from abc import ABCMeta
from abc import abstractmethod
from typing import Union

import torch
from path import Path
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()


    def kaiming_init(self, activation):
        # type: (str) -> ()
        """
        Apply "Kaiming-Normal" initialization to all Conv2D(s) of the model.
        :param activation: activation function after conv; values in {'relu', 'leaky_relu'}
        :return:
        """
        assert activation in ['ReLU', 'LeakyReLU', 'leaky_relu'], \
            '`activation` must be \'ReLU\' or \'LeakyReLU\''

        if activation == 'LeakyReLU':
            activation = 'leaky_relu'
        activation = activation.lower()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        ...


    @property
    def n_param(self):
        # type: (BaseModel) -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def current_device(self):
        # type: () -> str
        """
        :return: string that represents the device on which the model is currently located
            >> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return 'cuda' in self.current_device


    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        save model weights to the specified path
        """
        torch.save(self.state_dict(), path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        load model weights from the specified path
        """
        self.load_state_dict(torch.load(path))


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        :param flag: True if the model requires gradient, False otherwise
        """
        for p in self.parameters():
            p.requires_grad = flag


# ---------------------

class BasicConv2D(nn.Module):
    """
    Basic 2D Convolution with optional batch normalization and activation function
    """


    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 with_batch_norm=False, activation='LeakyReLU'):
        # type: (int, int, int, int, float, bool, str) -> None
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size of the 2D convolution
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param stride: stride of the convolution (default = 1)
            * NOTE: if `stride` is < 1, a trasnpsposed 2D convolution with stride=1/`stride`
            is used instead of a normal 2D convolution
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolution
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        self.with_batch_norm = with_batch_norm

        # 2D convolution
        if stride >= 1:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=int(stride),
                padding=padding,
                bias=(not self.with_batch_norm)
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=int(1 / stride),
                padding=padding,
                output_padding=padding,
                bias=(not self.with_batch_norm)
            )

        # batch normalization
        if with_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        # activation function
        assert activation in ['ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh']
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'Linear':
            self.activation = lambda x: x
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class SeparableConv2D(nn.Module):
    """
    Separable 2D Convolution Layer (by Google)
    >> paper: https://arxiv.org/pdf/1610.02357.pdf
    """


    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # type: (int, int, int, int, int) -> None
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=False
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=1
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
