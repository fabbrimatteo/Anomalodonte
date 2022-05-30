# -*- coding: utf-8 -*-
# ---------------------

from typing import Tuple

import torch
from path import Path
from torch.utils.data import Dataset


class DSCleaner(Dataset):

    def __init__(self, src_dir, dst_dir, buffer_size):
        # type: (str, str, int) -> None
        """
        :param src_dir: source directory containing cuts
        :param dst_dir: destination directory
            ->> old cuts will be moved here
        :param buffer_size: number of cuts you want to keep;
            ->> only the `buffer_size` newest elements will be kept
                in the source directory, while the others will be moved
                to the destination directory
        """

        # get all paths in the source directory
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.all_paths = self.src_dir.files()

        # sort paths by date (ascending)
        # >> self.all_paths[0] is the path of the oldest image
        # >> self.all_paths[-1] is the path of the youngest image
        self.sort()

        print(f'$> there are {len(self.all_paths)} cuts '
              f'in the source directory and buffer size is {buffer_size}')

        # keep the `buffer_size` oldest image
        if len(self.all_paths) < buffer_size:
            self.paths2keep = self.all_paths
            print(f'───$> we need to remove 0 cuts')
        else:
            _n = len(self.all_paths) - buffer_size
            print(f'───$> we need to remove the {_n} oldest cut(s)')
            self.paths2keep = self.all_paths[-buffer_size:]
        print('')


    def sort(self):
        """
        Sort paths by date (ascending) so that:
        ->> self.all_paths[0] is the path of the oldest image
        ->> self.all_paths[-1] is the path of the youngest image
        """
        self.all_paths.sort(key=lambda p: p.basename())


    def remove_old_paths(self):
        for p in self.all_paths:
            if p not in self.paths2keep:
                new_path = self.dst_dir / p.basename()
                cmd = f'mv "{p.abspath()}" "{new_path}'
                print(f'───$> {cmd}')


    def __len__(self):
        # type: () -> int
        return len(self.all_paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, str]
        return self.all_paths[i]


def main():
    ds = DSCleaner(
        src_dir='/goat-nas/Datasets/spal/progression_demo',
        dst_dir='/goat-nas/Datasets/spal/progression_trash',
        buffer_size=4992
    )
    ds.remove_old_paths()


if __name__ == '__main__':
    main()
