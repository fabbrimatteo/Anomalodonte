from time import sleep

import cv2
import mmu
from path import Path

from ds_utils import mpath2info, cpath2info


class Emitter(object):

    @staticmethod
    def __sorter(p):
        # type: (str) -> str
        info = mpath2info(p)
        return info['datestr']


    def __init__(self, maugeri_root, cuts_root, cam_id, start_idx):
        """
        :param maugeri_root: path of the "maugeri_ds" directory
        :param maugeri_root: path of the "spal_cuts" directory
        :param cam_id: int identifier of the camera to use
            ->> values in {1, 2, 3}
        :param start_idx: skip the first `start_index` images
            according to an ascending order by date
        """
        self.root = Path(maugeri_root) / 'goods'
        self.cuts_root = Path(cuts_root)
        self.all_paths = []
        self.index = 0
        for sub_dir in self.root.dirs():
            for img_path in sub_dir.files(f'*.bmp'):
                info = mpath2info(img_path)
                if info['camera-id'] != cam_id:
                    continue
                self.all_paths.append(img_path.abspath())

        assert start_idx < len(self.all_paths), \
            f'`start_index` must be less than {len(self.all_paths)}'

        test_dir = self.cuts_root / 'test' / f'cam_{cam_id}'
        skip_datestrs = [cpath2info(x)['datestr'] for x in test_dir.files()]

        self.all_paths = [
            p for p in self.all_paths
            if mpath2info(p)['datestr'] not in skip_datestrs
        ]

        self.all_paths.sort(key=self.__sorter)
        self.all_paths = self.all_paths[start_idx:]


    def read(self):
        """
        :return: tuple of 2 elements:
            ->> read_ok: boolean value that is `False` if no image has
                been read because of an error or because there are
                no other images to read; it is `True` otherwise;
            ->> frame: current frame (BGR);
                ->> numpy array with shape (H, W, 3)
                ->> NOTE: `frame` is `None` when `read_ok` is False
        """
        if self.index < len(self.all_paths):
            img_path = self.all_paths[self.index]
            img = cv2.imread(img_path)
            self.index += 1
            sleep(1.25)
            return True, img
        else:
            return False, None


def demo():
    from ds_utils import read_and_cut
    emitter = Emitter(
        maugeri_root='/goat-nas/Datasets/spal/maugeri_ds',
        cuts_root='/goat-nas/Datasets/spal/spal_cuts',
        cam_id=1, start_idx=0
    )

    for i in range(5000):
        print(i)
        p = emitter.all_paths[i]
        img = read_and_cut(p, cam_name='cam_1')
        out_name = mpath2info(p)['datestr'] + '.jpg'
        cv2.imwrite(
            f'/goat-nas/Datasets/spal/progression_demo/{out_name}',
            img
        )
    # for day in range(5):
    #     print(f'\nDAY #{day}')
    #     for i in range(10):
    #         read_ok, frame = emitter.read()
    #         if not read_ok:
    #             break
    #         else:
    #             print(f'───$> sample #{i} of day #{day}')


if __name__ == '__main__':
    demo()
