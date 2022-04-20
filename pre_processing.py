# -*- coding: utf-8 -*-
# ---------------------

from typing import Union

import numpy as np
import torch
import torchvision


# type shortcut(s):
ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def bgr2rgb(img):
    # type: (ArrayOrTensor) -> ArrayOrTensor
    """
    Colorspace conversion: BGR -> RGB.
    :param img: image (tensor or array) in BGR format
        >> if `img` is a tensor, the order must be (C,H,W)
        >> if `img` is an array, the order must be (H,W,C)
    :return: image (tensor or array) in RGB format
    """
    if type(img) is np.ndarray:
        return img[:, :, [2, 1, 0]]
    elif type(img) is torch.Tensor:
        return img[[2, 1, 0]]
    else:
        raise TypeError(
            f'image type must be {np.ndarray} '
            f'or {torch.Tensor}, not `{type(img)}`'
        )


class PreProcessingTr(object):

    def __init__(self, to_tensor=False):
        # type: (bool) -> None
        """
        (1) optional conversion from `np.ndarray` to `torch.Tensor`
        (2) conversion from BGR colorspace to RGB colorspace

        :param to_tensor: if True, the image is converted into torch.Tensor
            with shape (C,H,W) and values in [0,1], otherwise it remains a
            numpy array with shape (H,W,C) and values in [0,255]
        """

        trs_list = []

        if to_tensor:
            array2tensor = torchvision.transforms.ToTensor()
            trs_list.append(array2tensor)

        trs_list.append(bgr2rgb)

        self.trs = torchvision.transforms.Compose(trs_list)


    def apply(self, img):
        # type: (np.ndarray) -> ArrayOrTensor
        return self.trs(img)


    def __call__(self, img):
        # type: (np.ndarray) -> ArrayOrTensor
        return self.apply(img)
