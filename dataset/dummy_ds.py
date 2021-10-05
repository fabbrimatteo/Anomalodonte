# -*- coding: utf-8 -*-
# ---------------------

import random
import time
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import utils
from conf import Conf


class DummyDS(Dataset):
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

        self.avg_img_path = cnf.ds_path / 'average_train_img.png'
        if self.avg_img_path.exists():
            self.avg_img = utils.imread(self.avg_img_path)
            self.avg_img = transforms.ToTensor()(self.avg_img)

        else:
            print(f'$> WARNING: missing average image @ dataset root ({cnf.ds_path})')
            self.avg_img = None

        name_filter = '*' if mode == 'train' else 'good_*'
        for img_path in (self.cnf.ds_path / mode).files(name_filter):
            x = utils.imread(img_path)
            x = transforms.Resize(256)(x)
            x = transforms.ToTensor()(x)
            self.imgs.append(x)


    def __len__(self):
        # type: () -> int
        return len(self.imgs) if self.mode == 'test' else len(self.imgs) * 100


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        x = self.imgs[i % len(self.imgs)] - self.avg_img
        y = self.imgs[i % len(self.imgs)]
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
        seed = worker_id + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    cnf = Conf(exp_name='default')
    ds = DummyDS(cnf=cnf, mode='train')

    x_mean = None
    ds_len = len(ds)

    print(f'$> generating average image: ')
    for i in range(ds_len):
        x, y = ds[i]
        if x_mean is None:
            x_mean = (1 / ds_len) * x
        else:
            x_mean += (1 / ds_len) * x
        print(f'\r\t$> {i + 1} of {ds_len}', end='')

    out_path = cnf.ds_path / 'average_train_img.png'
    torchvision.utils.save_image(x_mean, out_path)
    print(f'\n$> image saved @ `{out_path.abspath()}`')


if __name__ == '__main__':
    main()
