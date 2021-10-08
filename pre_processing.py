from typing import Union

import cv2
import numpy as np
import torch
import torchvision


ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def bgr2rgb(img):
    # type: (ArrayOrTensor) -> ArrayOrTensor
    return img[[2, 1, 0]]


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


class PreProcessingTr(object):

    def __init__(self, resized_h, resized_w, crop_x_min, crop_y_min, crop_side):
        # type: (int, int, int, int, int) -> None
        self.trs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), bgr2rgb,
            ReseizeThenCrop(resized_h, resized_w, crop_x_min, crop_y_min, crop_side)
        ])


    def apply(self, img):
        # type: (np.ndarray) -> torch.Tensor
        return self.trs(img)


    def __call__(self, img):
        # type: (np.ndarray) -> torch.Tensor
        return self.apply(img)


def debug():
    img = cv2.imread('/home/fabio/output.jpg')
    img = PreProcessingTr(resized_h=628, resized_w=751, crop_x_min=147, crop_y_min=213, crop_side=256)(img)
    print(type(img), img.shape)


if __name__ == '__main__':
    debug()
