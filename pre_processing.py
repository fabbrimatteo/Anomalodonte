# -*- coding: utf-8 -*-
# ---------------------

from typing import Union

import cv2
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
    :return: image (tensor or array) in RGB format
    """
    return img[[2, 1, 0]]


class CropThenResize(object):

    def __init__(self, resized_h, resized_w, crop_x_min, crop_y_min, crop_side):
        # type: (int, int, int, int, int) -> None
        """
        :param resized_h: image height after resizing, but before cutting
        :param resized_w: image width after resizing, but before cutting
        :param crop_x_min: x coordinate of the top left corner of the crop square
        :param crop_y_min: y coordinate of the top left corner of the crop square
        :param crop_side: side (in pixels) of the crop square
        """
        self.h = resized_h
        self.w = resized_w
        self.x_min = crop_x_min
        self.y_min = crop_y_min
        self.x_max = self.x_min + crop_side
        self.y_max = self.y_min + crop_side


    def apply(self, img):
        # type: (ArrayOrTensor) -> ArrayOrTensor
        """
        :param img: image you want to resize and crop
        :return: resized and corpped image
        """
        if type(img) is np.ndarray:
            img = img[self.y_min:self.y_max, self.x_min:self.x_max, :]
            img = cv2.resize(img, (self.w, self.h))
        elif type(img) is torch.Tensor:
            img = img[:, self.y_min:self.y_max, self.x_min:self.x_max]
            img = torchvision.transforms.Resize((self.h, self.w))(img)
        else:
            raise TypeError(f'image type must be {np.ndarray} or {torch.Tensor}, not `{type(img)}`')

        return img


    def __call__(self, img):
        """
        :param img: image you want to resize and crop
        :return: resized and corpped image
        """
        return self.apply(img)


class PreProcessingTr(object):

    def __init__(self, resized_h, resized_w, crop_x_min, crop_y_min, crop_side):
        # type: (int, int, int, int, int) -> None
        """
        :param resized_h: image height after resizing, but before cutting
        :param resized_w: image width after resizing, but before cutting
        :param crop_x_min: x coordinate of the top left corner of the crop square
        :param crop_y_min: y coordinate of the top left corner of the crop square
        :param crop_side: side (in pixels) of the crop square
        """

        __trs = [torchvision.transforms.ToTensor(), bgr2rgb]

        # if `c1` and `c2`, you don't need to crop and resize the image
        # otherwise, add the `CropThenResize` transformation to the list
        c1 = crop_x_min == 0 and crop_y_min == 0
        c2 = crop_side == resized_h and crop_side == resized_w
        need_to_crop_and_resize = not (c1 and c2)
        if need_to_crop_and_resize:
            cr = CropThenResize(
                resized_h, resized_w,
                crop_x_min, crop_y_min,
                crop_side
            )
            __trs.append(cr)

        self.trs = torchvision.transforms.Compose(__trs)


    def apply(self, img):
        # type: (np.ndarray) -> torch.Tensor
        return self.trs(img)


    def __call__(self, img):
        # type: (np.ndarray) -> torch.Tensor
        return self.apply(img)


def __debug():
    from path import Path
    import post_processing

    img = cv2.imread(Path(__file__).parent / 'debug' / 'debug_img_00.png')
    print(f'$> before -> type: {type(img)}, shape: {tuple(img.shape)}')
    img = PreProcessingTr(resized_h=256, resized_w=256, crop_x_min=812, crop_y_min=660, crop_side=315)(img)
    print(f'$> after -> type: {type(img)}, shape: {tuple(img.shape)}')
    img = post_processing.tensor2img(img)
    cv2.imshow('', img)
    cv2.waitKey()


if __name__ == '__main__':
    __debug()
