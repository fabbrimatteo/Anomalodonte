import random
from datetime import datetime
from typing import Any
from typing import Dict

import cv2
import numpy as np
from path import Path


BOX_DICT = {
    'cam_1': [480, 100, 1140, 760],
    'cam_2': [359, 148, 1053, 842],
    'cam_3': [493, 212, 1169, 888],
}


def mpath2info(mpath):
    # type: (str) -> Dict[str, Any]
    """
    :param mpath: file path of an image in "maugeri" format
    :return: information dictionary with the following keys:
        'datetime': image creation datetime (datetime object)
        'datestr': image creation datetime as string in the format:
            >> %Y_%m_%d_%H_%M_%S
        'camera-id': id that identifies the camera that has taken the image;
            >> values in {1, 2, 3}
        'label': annotation given by the operator:
            >> values in {'good', 'bad', 'unknown'}
        'original_path': original file path
    """
    if type(mpath) is str:
        mpath = Path(mpath)
    date, uid, cam_and_label = mpath.basename().split('_')

    # parse date
    date = datetime(
        year=int(date[0:4]),
        month=int(date[4:6]),
        day=int(date[6:8]),
        hour=int(date[8:10]),
        minute=int(date[10:12]),
        second=int(date[12:14])
    )

    # camera ID
    cam = cam_and_label[0]

    # GT label
    if 'ok' in mpath.lower():
        label = 'good'
    elif 'ko' in mpath.lower():
        label = 'bad'
    else:
        label = 'unknown'

    info = {
        'datetime': date,
        'datestr': date.strftime('%Y_%m_%d_%H_%M_%S'),
        'camera-id': int(cam),
        'label': label,
        'original_name': str(mpath),
        'uid': uid
    }

    return info


def cpath2info(cpath):
    # type: (str) -> Dict[str, Any]
    """
    :param cpath: file path of an image in "cut-dataset" format
    :return: information dictionary with the following keys:
        'datetime': image creation datetime (datetime object)
        'datestr': image creation datetime as string in the format:
            >> %Y_%m_%d_%H_%M_%S
        'camera-id': id that identifies the camera that has taken the image;
            >> values in {1, 2, 3}
        'label': annotation given by the operator:
            >> values in {'good', 'bad', 'unknown'}
        'original_path': original file path
        'location': is it part of training set or test set?
            >> values in {'train', 'test'}
    """
    if type(cpath) is str:
        cpath = Path(cpath)

    cam_name = cpath.parent.basename()
    cam_id = int(cam_name.replace('cam_', ''))

    datestr = clean_cpath(cpath)

    # parse date
    date = datestr.split('_')
    date = datetime(
        year=int(date[0]),
        month=int(date[1]),
        day=int(date[2]),
        hour=int(date[3]),
        minute=int(date[4]),
        second=int(date[5])
    )

    location = cpath.parent.parent.basename()

    if location == 'train':
        label = 'good'
    else:
        label = cpath.basename().split('_')[0]

    info = {
        'datetime': date,
        'datestr': date.strftime('%Y_%m_%d_%H_%M_%S'),
        'camera-id': cam_id,
        'label': label,
        'original_name': str(cpath),
        'location': location
    }

    return info


def clean_cpath(cpath):
    # type: (str) -> str
    """
    :param cpath: cut-style path you want to clean
    :return: filename without GT label and file extension
    """
    cpath = Path(cpath).basename().split('.')[0]
    cpath = cpath.replace('bad_', '').replace('_bad', '')
    cpath = cpath.replace('good_', '').replace('_good', '')
    cpath = cpath.replace('nc_', '').replace('_nc', '')
    cpath = cpath.replace('nc_', '').replace('_nc', '')
    return cpath


def cut_full_img(img, cam_name, side):
    # type: (str, str, int) -> np.ndarray
    """
    :param img: full size image you want to cut
    :param cam_name: name of the camera that has taken the image
    :param side: side [px] of the square cut
    :return: square cut of the image with shape (side, side, 3)
    """
    x_min, y_min, x_max, y_max = BOX_DICT[cam_name]
    cut = img[y_min:y_max, x_min:x_max]
    return cv2.resize(cut, (side, side), interpolation=cv2.INTER_AREA)


def read_and_cut(img_mpath, cam_name, side=256):
    # type: (str, str, int) -> np.ndarray
    """
    :param img_mpath: path of the image you want to read and cut
        ->> full image in the "maugeri" dataset
    :param cam_name: name of the camera that has taken the image
    :param side: side [px] of the square cut
    :return: square cut of the image with shape (side, side, 3)
    """
    img = cv2.imread(img_mpath)
    return cut_full_img(img, cam_name, side)


class Checker(object):

    def __init__(self, cut_ds_root):
        # type: (str) -> None
        """
        :param cut_ds_root: root directory of the cut-style dataset
        """
        self.ds_root = Path(cut_ds_root)
        self.all = {
            'cam_1': [],
            'cam_2': [],
            'cam_3': []
        }
        for mode in ['train', 'test']:
            for cam in ['cam_1', 'cam_2', 'cam_3']:
                d = self.ds_root / mode / cam
                self.all[cam] += [clean_cpath(f) for f in d.files()]


    def check(self, mpath):
        # type: (str) -> bool
        """
        :param mpath: check if image at `mpath` is inside the "cuts" dataset
        :return: True if image is inside the "cuts" dataset, False otherwise
        """
        info = mpath2info(mpath)
        name = info['datestr']
        cam = f'cam_{info["camera-id"]}'
        return name in self.all[cam]


def create_cut_ds(m_root, c_root):
    # type: (str, str) -> None
    m_root = Path(m_root)
    c_root = Path(c_root)

    ck = Checker(cut_ds_root=c_root)

    for day_dir in m_root.dirs():
        for mpath in day_dir.files():
            if ck.check(mpath) is False:
                info = mpath2info(mpath=mpath)
                cam_name = f'cam_{info["camera-id"]}'
                cut = read_and_cut(img_mpath=mpath, cam_name=cam_name)

                cname = info['datestr'] + '.jpg'

                if random.random() <= 0.01:
                    cname = 'good_' + cname
                    cpath = c_root / 'test' / cam_name / cname
                else:
                    cpath = c_root / 'train' / cam_name / cname

                cv2.imwrite(cpath, cut)
                print(f'$> "{cpath}": saved')
