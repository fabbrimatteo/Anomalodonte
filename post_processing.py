# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
import torch


def tensor2img(tensor_img):
    # type: (torch.Tensor) -> np.ndarray
    """
    :param tensor_img: tensor image with (C,H,W) order
        and values in range [0,1] (float);
        >> NOTE: shape can be (C,H,W) or (1,C,H,W)
    :return: numpy_img -> array image with (H,W,C) order
        and values in range [0,255] (uint8)
    """
    numpy_img = tensor_img.detach().cpu().numpy()  # from tensor to array

    # remove batch dimension (if it is a singleton dimension)
    if len(numpy_img.shape) == 4 and numpy_img.shape[0] == 1:
        numpy_img = numpy_img[0]

    # from (C,H,W) order to (H,W,C) order
    if len(numpy_img.shape) == 3:
        numpy_img = numpy_img.transpose((1, 2, 0))

    # from float in [0,1] to uint8 in [0, 255]
    numpy_img = (numpy_img * 255).astype(np.uint8)

    return numpy_img


def __debug():
    import cv2

    tensor_img = torch.rand((3, 256, 256))
    img = tensor2img(tensor_img)
    cv2.imshow('debug :)', img)
    cv2.waitKey()


if __name__ == '__main__':
    __debug()
