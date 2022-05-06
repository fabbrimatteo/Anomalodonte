import random
from datetime import datetime
from typing import Any
from typing import Dict

import cv2
from path import Path


BOX_DICT = {
    'cam_1': [480, 100, 1140, 760],
    'cam_2': [359, 148, 1053, 842],
    'cam_3': [493, 212, 1169, 888],
}


def mpath2info(fpath):
    # type: (str) -> Dict[str, Any]
    """
    :param fpath: file path of an image in "maugeri" format
    :return: information dictionary with the following keys:
        'datetime': image creation datetime (datetime object)
        'datestr': image creation datetime as string in the format:
            >> %Y_%m_%d_%H_%M_%S
        'camera-id': id that identifies the camera that has taken the image;
            >> values in {1, 2, 3}
        'label': annotation given by the operator:
            >> values in {'good', 'bad', 'unknown'}
        'original_name': original file name
    """
    if type(fpath) is str:
        fpath = Path(fpath)
    date, tag, cam_and_label = fpath.basename().split('_')

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
    if 'ok' in fpath.lower():
        label = 'good'
    elif 'ko' in fpath.lower():
        label = 'bad'
    else:
        label = 'unknown'

    info = {
        'datetime': date,
        'datestr': date.strftime('%Y_%m_%d_%H_%M_%S'),
        'camera-id': int(cam),
        'label': label,
        'original_name': str(fpath.basename()),
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
        'original_name': original file name
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
        'original_name': str(cpath.basename()),
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


def read_and_cut(img_path, cam_name, side=256):
    # type: (str, str, int) -> np.ndarray
    """
    :param img_path: path of the image you want to read and cut
    :param cam_name: name of the camera that has taken the image
    :return: square cut of the image with shape (side, side, 3)
    """
    x_min, y_min, x_max, y_max = BOX_DICT[cam_name]
    img = cv2.imread(img_path)
    cut = img[y_min:y_max, x_min:x_max]
    return cv2.resize(cut, (side, side), interpolation=cv2.INTER_AREA)


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
        :param mpath:
        :return:
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
                info = mpath2info(fpath=mpath)
                cam_name = f'cam_{info["camera-id"]}'
                cut = read_and_cut(img_path=mpath, cam_name=cam_name)

                cname = info['datestr'] + '.jpg'

                if random.random() <= 0.01:
                    cname = 'good_' + cname
                    cpath = c_root / 'test' / cam_name / cname
                else:
                    cpath = c_root / 'train' / cam_name / cname

                cv2.imwrite(cpath, cut)
                print(f'$> "{cpath}": saved')


if __name__ == '__main__':
    create_cut_ds(
        m_root='/goat-nas/Datasets/spal/maugeri_ds/goods',
        c_root='/goat-nas/Datasets/spal/spal_cuts'
    )
