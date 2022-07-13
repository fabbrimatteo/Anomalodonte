import os
from datetime import datetime, timedelta
from typing import Set
from typing import Tuple

import numpy as np
import torch
from path import Path
import cv2
import random


DRElement = Tuple[str, int]


class DayDB(object):

    def __init__(self, root_dir, train_buffer_size=5000,
                 test_buffer_size=1000, debug=False):
        # type: (str, int, int) -> None
        """
        :param root_dir: path of the dataset root directory
            >> root_dir must contain a subdirectory "daily_cuts"
        :param train_buffer_size: number of cuts you want to keep for training;
            ->> only the `train_buffer_size` newest elements will be kept
                in the source directory, while the others will be moved
                to the destination directory
        :param test_buffer_size: number of cuts you want to keep for testing;
            ->> only the `test_buffer_size` newest elements will be kept
                in the source directory, while the others will be moved
                to the destination directory
        """

        self.root_dir = Path(root_dir)

        assert self.root_dir.exists(), \
            f'`root_dir` must be an existing directory'

        self.daily_cuts_dir = self.root_dir / 'daily_cuts'
        self.daily_cuts_dir.makedirs_p()

        self.train_dir = self.root_dir / 'train'
        self.train_dir.makedirs_p()

        self.test_dir = self.root_dir / 'test'
        self.test_dir.makedirs_p()

        self.trash_dir = self.root_dir / 'trash'
        self.trash_dir.makedirs_p()

        # TODO: handle the file as a checkpoint
        self.info_file_path = self.root_dir / 'today_scores.pth'
        self.info = {}

        self.train_buffer_size = train_buffer_size
        self.test_buffer_size = test_buffer_size

        self.debug = debug
        if self.debug:
            self.start_datetime = datetime.now()
            self.counter = 0


    def save_info(self):
        # type: () -> None
        torch.save(self.info, self.info_file_path)


    def get_subset(self, set_type, u_range):
        # type: (str, Tuple[float, float]) -> Set[DRElement]

        assert set_type.upper() in ['G', 'B', 'U'], \
            f'`set_type` must be one of {{"G", "B", "U"}}, ' \
            f'not "{set_type}"'

        s_min, s_max = u_range

        # 'G' set -> images that are clearly "GOOD"
        if set_type.upper() == 'G':
            subset = [
                (k, self.info[k]) for k in self.info
                if self.info[k] < s_min
            ]

        # 'B' set -> images that are clearly "BAD"
        elif set_type.upper() == 'B':
            subset = [
                (k, self.info[k]) for k in self.info
                if self.info[k] > s_max
            ]

        # 'U' set -> images whose nature is unclear
        else:
            subset = [
                (k, self.info[k]) for k in self.info
                if s_min <= self.info[k] <= s_max
            ]

        return set(subset)


    # TODO: maybe percentage is not the right way to pick elements
    def integrate_subsets(self, subsets, expl_perc):
        # type: (dict, float) -> None

        print(f'$> DAILY PROCEDURE')
        print(f'----')

        print(f'$> initial sets cardinality:')
        print(f'───$> #G={len(subsets["G"])}, '
              f'#B={len(subsets["B"])}, '
              f'#U={len(subsets["U"])}')

        print(f'----')
        print(f'$> moving some elements from G and B to U')
        for set_type in ['G', 'B']:
            if len(subsets[set_type]) < 1:
                continue
            to_move = self.__choice_perc(subsets[set_type], perc=expl_perc)
            subsets[set_type] = subsets[set_type].difference(to_move)
            subsets['U'] = subsets['U'].union(to_move)
            print(f'───$> moving {expl_perc}% of {set_type} '
                  f'({len(to_move)}) elements to U')

        print(f'$> current sets cardinality:')
        print(f'───$> #G={len(subsets["G"])}, '
              f'#B={len(subsets["B"])}, '
              f'#U={len(subsets["U"])}')

        print(f'----')
        print(f'$> updating training set with 90% of G')
        if len(subsets['G']) >= 1:
            g_train = self.__choice_perc(subsets['G'], perc=90)
            for element in g_train:
                date_str, anomaly_score = element
                src_path = self.daily_cuts_dir / date_str + '.jpg'
                dst_path = self.train_dir / date_str + '.jpg'

                cmd = f'mv "{src_path.abspath()}" "{dst_path.abspath()}"'
                os.system(cmd)
                print(f'───$> {cmd} '
                      f'(anomaly_score={anomaly_score:03d}) to training set')

        print(f'----')
        print(f'$> updating test set with the remaining 10% of G')
        if len(subsets['G']) >= 1:
            g_test = subsets['G'] - g_train
            if len(g_test) >= 1:
                for element in g_test:
                    date_str, anomaly_score = element
                    src_path = self.daily_cuts_dir / date_str + '.jpg'
                    dst_path = self.test_dir / f'good_{date_str}.jpg'

                    cmd = f'mv "{src_path.abspath()}" "{dst_path.abspath()}"'
                    os.system(cmd)
                    print(f'───$> {cmd} '
                          f'(anomaly_score={anomaly_score:03d}) to test set')

        # TODO: all the elements?
        print(f'----')
        print(f'$> updating test set with all the elements of B')
        b_test = subsets['B']
        if len(b_test) >= 1:
            for element in b_test:
                date_str, anomaly_score = element
                src_path = self.daily_cuts_dir / date_str + '.jpg'
                dst_path = self.test_dir / f'bad_{date_str}.jpg'

                cmd = f'mv "{src_path.abspath()}" "{dst_path.abspath()}"'
                os.system(cmd)
                print(f'───$> {cmd} '
                      f'(anomaly_score={anomaly_score:03d}) to test set')

        # TODO: update the dataset with U


    def clean_dataset(self, source_dir_path, buffer_size):
        # type: () -> None

        # get all paths in the source directory
        all_paths = source_dir_path.files()

        # sort paths by date (ascending)
        # >> all_paths[0] is the path of the oldest image
        # >> all_paths[-1] is the path of the youngest image
        all_paths.sort(key=lambda p: p.basename())

        print(f'----')
        print(f'$> there are {len(all_paths)} cuts '
              f'in {source_dir_path} and buffer size is {buffer_size}')

        # keep the `buffer_size` oldest image
        if len(all_paths) < buffer_size:
            paths2keep = all_paths
            print(f'───$> we need to remove 0 cuts')
        else:
            _n = len(all_paths) - buffer_size
            print(f'───$> we need to remove the {_n} oldest cut(s)')
            paths2keep = all_paths[-buffer_size:]
        print('')

        # remove old files
        for p in all_paths:
            if p not in paths2keep:
                new_path = self.trash_dir / p.basename()
                cmd = f'mv "{p.abspath()}" "{new_path}"'
                os.system(cmd)
                print(f'───$> {cmd}')


    def update_dataset(self, u_range, expl_perc):
        # type: (Tuple[float, float], float) -> None

        subsets = {}
        for set_type in ['G', 'B', 'U']:
            subsets[set_type] = self.get_subset(
                set_type=set_type, u_range=u_range
            )

        self.integrate_subsets(subsets, expl_perc)
        self.clean_dataset(self.train_dir, self.train_buffer_size)
        self.clean_dataset(self.test_dir, self.test_buffer_size)


    def update_dataset_all(self):
        all_cuts = set([(k, self.info[k]) for k in self.info])

        for element in all_cuts:
            date_str, anomaly_score = element
            src_path = self.daily_cuts_dir / date_str + '.jpg'
            dst_path = self.train_dir / date_str + '.jpg'

            cmd = f'mv "{src_path.abspath()}" "{dst_path.abspath()}"'
            os.system(cmd)
            print(f'───$> {cmd} '
                  f'(anomaly_score={anomaly_score:03d}) to training set')


    def add(self, img_cut, anomaly_score):
        # type: (np.ndarray, int) -> None

        # obtain current date string
        if self.debug:
            now = self.start_datetime + timedelta(seconds=self.counter)
            self.counter += 1
        else:
            now = datetime.now()
        date_str = f'{now.year}_{now.month:02d}_{now.day:02d}'
        date_str = f'{date_str}_{now.hour:02d}_{now.minute:02d}'
        date_str = f'{date_str}_{now.second:02d}'

        self.info[date_str] = anomaly_score

        # save image cut
        out_img_path = self.root_dir / 'daily_cuts' / f'{date_str}.jpg'
        cv2.imwrite(out_img_path, img_cut)

        self.save_info()


    def __choice_perc(self, x, perc):
        # type: (Set[DRElement], float) -> Set[DRElement]
        """
        :param x: set of DRElement(s)
        :param perc: percentage of items to select
            >> value in range [0, 100]
        :return: a randomly drawn subset of `x`,
            with size equal to `perc`% of `x`;
            >> NOTE: minimum size is 1
        """
        size = int(round(len(x) * (perc / 100)))
        size = max(size, 1)
        random_subset = random.sample(list(x), size)
        return set(random_subset)
