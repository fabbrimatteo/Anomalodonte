from datetime import datetime
from typing import Set
from typing import Tuple

import numpy as np
import torch
from path import Path


DRElement = Tuple[str, int]


class DayRecorder(object):

    def __init__(self, root_dir):
        # type: (str) -> None
        """
        :param root_dir: path of the dataset root directory
            >> root_dir must contain a subdirectory "daily_cuts"
        """

        self.root_dir = Path(root_dir)

        assert self.root_dir.exists(), \
            f'`root_dir` must be an existing directory'

        self.daily_cuts_dir = self.root_dir / 'daily_cuts'
        if not self.daily_cuts_dir.exists():
            self.daily_cuts_dir.makedirs()

        self.info_file_path = self.root_dir / 'today_scores.pth'
        if self.info_file_path.exists():
            self.info = torch.load(self.info_file_path)
        else:
            self.info = {}


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


    def add(self, img_cut, anomaly_score):
        # type: (np.ndarray, int) -> None

        # obtain current date string
        now = datetime.now()
        date_str = f'{now.year}_{now.month:02d}_{now.day:02d}'
        date_str = f'{date_str}_{now.hour:02d}_{now.minute:02d}'
        date_str = f'{date_str}_{now.second:02d}'

        self.info[date_str] = anomaly_score

        # save image cut
        # out_img_path = self.root_dir / 'daily_cuts' / f'{date_str}.jpg'
        # cv2.imwrite(out_img_path, img_cut)

        self.save_info()
