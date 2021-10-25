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
from pre_processing import PreProcessingTr
from pre_processing import bgr2rgb


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
        :param mode: working mode -> must be one of {"train", "test"}
        """
        self.cnf = cnf

        assert mode in ['train', 'test'], \
            'mode must be one of {"train", "test"}'
        self.mode = mode

        self.imgs = []
        self.labels = []
        self.avg_img = None

        # pre processing traqnsformations
        self.trs = PreProcessingTr(
            resized_h=cnf.resized_h, resized_w=cnf.resized_w,
            crop_x_min=cnf.crop_x_min, crop_y_min=cnf.crop_y_min,
            crop_side=cnf.crop_side
        )

        t0 = time.time()
        print(f'$> loading images into memory: please wait...')

        all_paths = (self.cnf.ds_path / mode).files()

        # TODO: remove this
        if mode == 'test':
            all_paths.sort()
            b = [b for b in all_paths if b.basename().startswith('bad')]
            g = [g for g in all_paths if g.basename().startswith('good')]
            all_paths = b + g[:len(b)]

        for i, img_path in enumerate(all_paths):
            print(f'\r\t$> {i + 1} of {len(all_paths)}', end='')

            x = cv2.imread(img_path)
            x = self.trs(x)
            self.imgs.append(x)

            # the label of an image can be inferred from its filename:
            # >> an image with label "good"
            #    has a filename that starts with "good_"
            # >> an image with label "bad"
            #    has a filename that starts with "bad_"
            label = img_path.basename().split('_')[0]
            self.labels.append(label)

        print(f'\r\t$> done in {time.time() - t0:.0f} seconds')


    def __len__(self):
        # type: () -> int
        return len(self.imgs)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, str]
        x = self.imgs[i]
        label = self.labels[i]
        return x, x, label


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
    cnf = Conf(exp_name='p1')
    ds = SpalDS(cnf=cnf, mode='train')
    for i in range(10):
        x, y, label = ds[i]
        print(x.shape, y.shape, label)


if __name__ == '__main__':
    main()
