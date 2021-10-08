# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
import torch


def tensor2img(tensor_img):
    # type: (torch.Tensor) -> np.ndarray
    """
    :param tensor_img: tensor image with (C,H,W) order and values in range [0,1] (float)
    :return: numpy_img -> array image with (H,W,C) order and values in range [0,255] (uint8)
    """
    numpy_img = tensor_img.detach().cpu().numpy()  # from tensor to array
    if len(numpy_img.shape) == 4 and numpy_img.shape[0] == 1:
        numpy_img = numpy_img[0]
    numpy_img = numpy_img.transpose((1, 2, 0))  # from CHW to HWC
    numpy_img = (numpy_img * 255).astype(np.uint8)  # from [0,1] to [0,255]
    return numpy_img


def __debug():
    import cv2

    tensor_img = torch.rand((3, 256, 256))
    img = tensor2img(tensor_img)
    cv2.imshow('debug :)', img)
    cv2.waitKey()


if __name__ == '__main__':
    __debug()
