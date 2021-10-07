from typing import Union

import cv2
import numpy as np
import torch
import torchvision


ArrayOrTensor = Union[np.ndarray, torch.Tensor]


class ReseizeThenCrop(object):

    def __init__(self, resized_h, resized_w, crop_x_min, crop_y_min, crop_side):
        # type: (int, int, int, int, int) -> None
        """
        :param resized_h:
        :param resized_w:
        :param crop_x_min:
        :param crop_y_min:
        :param crop_side:
        """
        self.h = resized_h
        self.w = resized_w
        self.x_min = crop_x_min
        self.y_min = crop_y_min
        self.x_max = self.x_min + crop_side
        self.y_max = self.y_min + crop_side


    def apply(self, img):
        # type: (ArrayOrTensor) -> ArrayOrTensor
        if type(img) is np.ndarray:
            img = cv2.resize(img, (self.w, self.h))
            img = img[self.y_min:self.y_max, self.x_min:self.x_max, :]
        else:
            img = torchvision.transforms.Resize((self.h, self.w))(img)
            img = img[:, self.y_min:self.y_max, self.x_min:self.x_max]

        return img


    def __call__(self, img):
        return self.apply(img)


def apply_pre_processing(img):
    # type: (ArrayOrTensor) -> ArrayOrTensor

    h, w = 628, 751
    start_x, start_y = 147, 213

    if type(img) is np.ndarray:
        img = cv2.resize(img, (w, h))
        img = img[start_y:start_y + 256, start_x:start_x + 256, :]
    else:
        img = torchvision.transforms.Resize((h, w))(img)
        img = img[:, start_y:start_y + 256, start_x:start_x + 256]

    return img


def debug():
    img = cv2.imread('/home/matteo/PycharmProjects/Anomalodonte/dataset/spal_fake/train/good_0105.png')
    img = ReseizeThenCrop(resized_h=628, resized_w=751, crop_x_min=147, crop_y_min=213, crop_side=256)(img)
    cv2.imshow('', img)
    cv2.waitKey()


if __name__ == '__main__':
    debug()
