# -*- coding: utf-8 -*-
# ---------------------

import random
import time
from typing import Tuple

import cv2
import imgaug
import numpy as np
import torch
import torchvision
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

import pre_processing
from conf import Conf


class SpalDS(Dataset):

    def __init__(self, cnf, mode):
        # type: (Conf, str) -> None
        """
        :param cnf: configuration object
        :param mode: working mode
            -> must be one of {"train", "test"}
        """
        self.cnf = cnf

        assert mode in ['train', 'test'], \
            'mode must be one of {"train", "test"}'

        self.mode = mode

        self.imgs = []
        self.labels = []
        self.avg_img = None

        self.to_tensor = torchvision.transforms.ToTensor()

        t0 = time.time()
        if mode == 'train':
            print(f'$> loading images into memory: please wait...')

        all_paths = (self.cnf.ds_path / mode / self.cnf.cam_id).files()
        all_paths.sort(key=lambda p: p.basename())

        for i, img_path in enumerate(all_paths):
            if mode == 'train':
                print(f'\r\t$> {i + 1} of {len(all_paths)}', end='')

            x = cv2.imread(img_path)
            x = pre_processing.bgr2rgb(x)
            self.imgs.append(x)

            self.data_aug = iaa.SomeOf(
                n=(0, 3),
                children=[
                    iaa.AddToHue((-16, 16)),
                    iaa.AddToBrightness((-16, 16), to_colorspace='HSV'),
                    iaa.AddToSaturation((-16, 16))
                ],
                random_order=True
            )

            # (1) in `training` mode, all images have label "good"
            # (2) in `test` mode the label of an image can be
            #     inferred from its filename;
            #     >> an image with label "good"
            #        has a filename that starts with "good_"
            #     >> an image with label "bad"
            #        has a filename that starts with "bad_"
            name = img_path.basename()
            if mode == 'train':
                label = 'good'
            else:
                label = name.split('_')[0]
            self.labels.append(label)

        if mode == 'train':
            print(f'\r\t$> done in {time.time() - t0:.0f} seconds')


    def __len__(self):
        # type: () -> int
        return len(self.imgs)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, str]

        # `img` -> shape (H,W,C) and values in [0,255] (uint8)
        img = self.imgs[i]

        # string ("good" or "bad")
        label = self.labels[i]

        # `x` & `y` -> shape (C,H,W) and values in [0,1] (float)
        # if self.mode == 'train':
        #     img = self.data_aug.augment_image(img)

        x = self.to_tensor(img)
        y = self.to_tensor(img)

        return x, y, label


    @staticmethod
    def wif(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = (int(round(time.time() * 1000)) + worker_id) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


    @staticmethod
    def wif_test(worker_id):
        # type: (int) -> None
        """
        Worker initialization function: set random seeds
        :param worker_id: worker int ID
        """
        seed = worker_id * 0 + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


def main():
    cnf = Conf(exp_name='mar2022')
    cnf.data_aug = True
    ds = SpalDS(cnf=cnf, mode='train')
    for i in range(len(ds)):
        x, y, label = ds[i]
        print(x.shape, y.shape, label)


if __name__ == '__main__':
    main()
