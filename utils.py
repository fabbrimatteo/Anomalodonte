# -*- coding: utf-8 -*-
# ---------------------

import os

import matplotlib


__DISPLAY = os.environ.get('DISPLAY', None)
if __DISPLAY is None or __DISPLAY == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

from matplotlib import figure
from matplotlib import cm
import numpy as np
import PIL
from PIL.Image import Image
from path import Path
from torchvision.transforms import ToTensor
from torch import Tensor
from typing import Union
from typing import Optional
from typing import Tuple


def imread(path):
    # type: (Union[Path, str]) -> Image
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(pyplot_figure):
    # type: (figure.Figure) -> np.ndarray
    """
    Converts a PyPlot figure into a NumPy array
    :param pyplot_figure: figure you want to convert
    :return: converted NumPy array
    """
    pyplot_figure.canvas.draw()
    x = np.fromstring(pyplot_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(pyplot_figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(pyplot_figure):
    # type: (figure.Figure) -> Tensor
    """
    Converts a PyPlot figure into a PyTorch tensor
    :param pyplot_figure: figure you want to convert
    :return: converted PyTorch tensor
    """
    x = pyplot_to_numpy(pyplot_figure=pyplot_figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', value_range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param value_range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=value_range[0], vmax=value_range[1])
    x = x.detach().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)


def main():
    pass


if __name__ == '__main__':
    main()
