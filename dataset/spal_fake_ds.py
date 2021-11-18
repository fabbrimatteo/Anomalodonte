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
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

from conf import Conf
from pre_processing import PreProcessingTr


class SpalDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: RGB image of size 128x128 representing a light blue circle (radius = 16 px)
         on a dark background (random circle position, randomly colored dark background)
    * y: copy of x, with the light blue circle surrounded with a red line (4 px internale stroke)
    """


    def __init__(self, cnf, mode):
        # type: (Conf, str) -> None
        """
        :param cnf: configuration object
        :param mode: working mode
            -> must be one of {"train", "test", "ev-test"}
            TODO: add a comment about the meaning of each mode
        """
        self.cnf = cnf

        assert mode in ['train', 'test', 'ev-test'], \
            'mode must be one of {"train", "test", "ev-test"}'
        self.mode = mode

        self.imgs = []
        self.labels = []
        self.avg_img = None

        # ---- pre processing transformations
        # (1) BGR to RGB conversion
        # (2) cut (optional)
        # (3) resize (optional)
        self.trs = PreProcessingTr(
            resized_h=cnf.resized_h, resized_w=cnf.resized_w,
            crop_x_min=cnf.crop_x_min, crop_y_min=cnf.crop_y_min,
            crop_side=cnf.crop_side, to_tensor=False
        )
        # (4) from array (H,W,C) in [0,255] to tensor (C,H,W) in [0,1]
        self.to_tensor = torchvision.transforms.ToTensor()

        self.cdrop = iaa.SomeOf((1, 3), [
            iaa.CoarseSalt(0.04, size_percent=(0.1, 0.2)),
            iaa.CoarseSalt(0.04, size_percent=(0.1, 0.2), per_channel=True),
            iaa.PerspectiveTransform((0.03, 0.1)),
            iaa.SaltAndPepper(p=0.2, per_channel=True),
        ], random_order=True)

        t0 = time.time()
        if mode == 'train':
            print(f'$> loading images into memory: please wait...')

        if mode == 'ev-test':
            test_paths = (self.cnf.ds_path / 'test').files()
            ev_test_paths = (self.cnf.ds_path / 'ev-test').files()
            all_paths = test_paths + ev_test_paths
        else:
            all_paths = (self.cnf.ds_path / mode).files()

        for i, img_path in enumerate(all_paths):
            if mode == 'train':
                print(f'\r\t$> {i + 1} of {len(all_paths)}', end='')

            x = cv2.imread(img_path)
            x = self.trs(x)
            self.imgs.append(x)

            # the label of an image can be inferred from its filename;
            # >> (1) for "train" or "test" mode
            #    >> an image with label "good"
            #       has a filename that starts with "good_"
            #    >> an image with label "bad"
            #       has a filename that starts with "bad_"
            # >> (2) for "ev-test" mode
            #    >> an image with label "bad"
            #       has a filename that starts with "invalid_"
            #    >> an image with label "good"
            #       has a filename that does NOT starts with "invalid_"
            name = img_path.basename()
            if self.mode in ['train', 'test']:
                label = name.split('_')[0]
            else:
                label = 'bad' if name.startswith('invalid_') else 'good'
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
        if self.mode == 'train' and self.cnf.data_aug:
            x = self.cdrop.augment_image(img)
            # cv2.imshow('', x)
            # cv2.waitKey()
            x = self.to_tensor(x)
        else:
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
    cnf = Conf(exp_name='a5_noise_new')
    cnf.data_aug = True
    ds = SpalDS(cnf=cnf, mode='train')
    for i in range(len(ds)):
        x, y, label = ds[i]
        print(x.shape, y.shape, label)


if __name__ == '__main__':
    main()
