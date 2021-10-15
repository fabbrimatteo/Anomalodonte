# -*- coding: utf-8 -*-
# ---------------------

import random
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from conf import Conf
from pre_processing import PreProcessingTr
import torchvision
from pre_processing import bgr2rgb
from path import Path

def get_anomaly_type(file_name):
    file_name = Path(file_name).basename()
    if 'good' in file_name:
        return '000'
    else:
        x = file_name.split('_')[1]
        a1, a2, a3 = int(x[1]), int(x[2]), int(x[3])
        return f'{a1}{a2}{a3}'


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
        self.labels = []
        self.avg_img = None

        # pre processing traqnsformations
        if '_rect' in self.cnf.ds_path:
            self.trs = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), bgr2rgb,
            ])
        else:
            self.trs = PreProcessingTr(
                resized_h=256, resized_w=256,
                crop_x_min=812, crop_y_min=660, crop_side=315
            )

        t0 = time.time()
        print(f'$> loading images into memory: please wait...')

        all_paths = (self.cnf.ds_path / mode).files()
        for i, img_path in enumerate(all_paths):
            anomaly_type = get_anomaly_type(img_path)
            if anomaly_type.startswith('0') and '1' in anomaly_type:
                continue
            if not '_rect' in str(self.cnf.ds_path.basename()):
                print(f'\r\t$> {i + 1} of {len(all_paths)}', end='')
            x = cv2.imread(img_path)
            x = self.trs(x)
            self.imgs.append(x)
            self.labels.append('good' if get_anomaly_type(img_path).startswith('0') else 'bad')
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
    cnf = Conf(exp_name='capra')
    ds = SpalDS(cnf=cnf, mode='test')
    print(len(ds), len([l for l in ds.labels if l=='good']), len([l for l in ds.labels if l=='bad']))


if __name__ == '__main__':
    main()
