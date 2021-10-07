# -*- coding: utf-8 -*-
# ---------------------

import random
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from conf import Conf
from pre_processing import ReseizeThenCrop


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
        :param ds_len: dataset length
        """
        self.cnf = cnf
        self.mode = mode

        self.imgs = []
        self.avg_img = None

        self.trs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ReseizeThenCrop(resized_h=628, resized_w=751, crop_x_min=147, crop_y_min=213, crop_side=256),
        ])

        t0 = time.time()
        print(f'$> loading images into memory: please wait...')

        all_paths = (self.cnf.ds_path / mode).files()
        for i, img_path in enumerate(all_paths):
            print(f'\r\t$> {i + 1} of {len(all_paths)}', end='')
            x = cv2.imread(img_path)
            x = self.trs(x)
            self.imgs.append(x)
        print(f'\r\t$> done in {time.time() - t0:.0f} seconds')


    def __len__(self):
        # type: () -> int
        return len(self.imgs)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        x = self.imgs[i]
        y = x
        return x, y


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


def main():
    cnf = Conf(exp_name='default')
    ds = SpalDS(cnf=cnf, mode='train')


if __name__ == '__main__':
    main()
