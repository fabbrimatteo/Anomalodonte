# -*- coding: utf-8 -*-
# ---------------------

from typing import Union

import numpy as np
import torch
from torch import nn


# type shortcut(s):
ArrayOrTensor = Union[np.ndarray, torch.Tensor]


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


class ADProcessor(object):

    def __init__(self, model):
        # type: (nn.Module) -> None
        self.mode = model


    def get_anomaly_score(self, img):
        # type: (ArrayOrTensor) -> float
        raise NotImplementedError


    def get_anomaly_perc(self, img, max_val=100):
        # type: (ArrayOrTensor, float) -> float
        """
        Returns the anomaly percentage related to the input image;
        `anomaly_perc` is such that a value >= 50 corresponds to an
        anomalous example.

        :param img:
        :param max_val:
        :return:
        """

        anomaly_score = self.get_anomaly_score(img)
        anomaly_th = self.get_anomaly_th()

        anomaly_prob = anomaly_score / anomaly_th
        anomaly_prec = (anomaly_prob * 100) - 50

        if max_val is None:
            return max(0., anomaly_prec)
        else:
            return float(np.clip(anomaly_prec, 0, max_val))


    def get_anomaly_th(self):
        # type: () -> float
        """
        Returns the anomaly threshold; a sample with an `anomaly_score`
        greather than this treshold is considered as "anomalous".

        >> NOTE: a sample with an anomaly score euqal to the threshold should
           have an `anomaly_perc` equal to 50.

        :return: anomaly threshold
        """
        raise NotImplementedError


def __debug():
    import cv2

    tensor_img = torch.rand((3, 256, 256))
    img = tensor2img(tensor_img)
    cv2.imshow('debug :)', img)
    cv2.waitKey()


if __name__ == '__main__':
    __debug()
