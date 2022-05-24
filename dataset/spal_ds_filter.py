# -*- coding: utf-8 -*-
# ---------------------

from typing import Tuple

import torch
from torch.utils.data import Dataset

from conf import Conf
from path import Path


class SpalFilter(Dataset):

    def __init__(self, source_dir, buffer_size):
        # type: (str, int) -> None

        # get all paths in the source directory
        self.source_dir = Path(source_dir)
        self.all_paths = self.source_dir.files()

        # sort paths by date (ascending)
        # >> self.all_paths[0] is the path of the oldest image
        # >> self.all_paths[-1] is the path of the youngest image
        self.sort()

        # keep the `buffer_size` oldest image
        self.all_paths = self.all_paths[0:buffer_size]


    def sort(self):
        self.all_paths.sort(key=lambda p: p.basename())


    def add_img(self, img_path, sort_afeter_add=False):
        self.all_paths.pop(0)
        self.all_paths.append(img_path)
        if sort_afeter_add:
            self.sort()


    def __len__(self):
        # type: () -> int
        return len(self.all_paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, str]
        return self.all_paths[i]


def main():
    cnf = Conf(exp_name='lof1')
    cnf.data_aug = True
    ds = SpalFilter(
        source_dir=cnf.ds_path / 'train' / 'cam_1',
        buffer_size=500
    )
    for i in range(10):
        x = ds[i]
        print(x)


if __name__ == '__main__':
    in_dir = Path('/goat-nas/Datasets/spal/spal_cuts/train/cam_1')
    out_dir = Path('/goat-nas/Datasets/spal/spal_cuts/train/cam_1')
    all_paths = list(in_dir.files())
    all_paths.sort(key=lambda p: p.basename())
    all_paths = all_paths[:5000]
    for p in all_paths:
        print(p)
