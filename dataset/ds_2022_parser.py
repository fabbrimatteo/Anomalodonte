import os
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

SPAL_ROOT = Path('/goat-nas/Datasets/spal')

MAUGERI_DS = {
    'train': SPAL_ROOT / '2022_03_09' / 'goods',
    'test': SPAL_ROOT / '2022_03_09' / 'bads'
}

CUTS_ROOT = Path('/goat-nas/Datasets/spal/spal_cuts')
CUTS_ROOT.mkdir_p()


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
    cpath = Path(cpath).basename().split('.')[0]
    cpath = cpath.replace('bad_', '').replace('_bad', '')
    cpath = cpath.replace('good_', '').replace('_good', '')
    cpath = cpath.replace('nc_', '').replace('_nc', '')
    cpath = cpath.replace('nc_', '').replace('_nc', '')
    return cpath


def get_resized_cut(img_path):
    info = mpath2info(img_path)
    cam_id = 'cam_' + str(info['camera-id'])
    x_min, y_min, x_max, y_max = BOX_DICT[cam_id]
    img = cv2.imread(img_path)
    cut = img[y_min:y_max, x_min:x_max]
    return cv2.resize(cut, (256, 256))


def refactor(mode):
    if mode == 'train':
        all_img_paths = []
        for d in MAUGERI_DS[mode].dirs():
            for img_path in d.files('*.bmp'):
                all_img_paths.append(img_path)
    else:
        all_img_paths = MAUGERI_DS[mode].files('*.bmp')

    for img_path in all_img_paths:

        info = mpath2info(img_path)
        cam_id = 'cam_' + str(info['camera-id'])

        if info['label'] == 'good' or mode == 'test':
            x_min, y_min, x_max, y_max = BOX_DICT[cam_id]
            img = cv2.imread(img_path)
            cut = img[y_min:y_max, x_min:x_max]
            resized_cut = cv2.resize(cut, (256, 256))

            out_name = info['new_name']
            if mode == 'test':
                out_name = 'bad_' + out_name

            out_path = CUTS_ROOT / mode / cam_id / out_name
            out_path.parent.mkdir_p()

            cv2.imwrite(out_path, resized_cut)
            print(f'$> {out_path}')


def mv_some_goods_to_test(n_imgs_to_move):
    for cam_dir in (CUTS_ROOT / 'train').dirs():
        cam_id = cam_dir.basename()
        print(f'\n$> CAMERA: {cam_dir.basename()}')
        all_img_paths = []
        for img_path in cam_dir.files('*.jpg'):
            all_img_paths.append(img_path)
        selection = random.choices(all_img_paths, k=n_imgs_to_move)
        for img_path in selection:
            new_name = 'good_' + img_path.basename()
            print(f'mv "{img_path}" '
                  f'"{CUTS_ROOT / "test" / cam_id / new_name}"')
            os.system(f'mv "{img_path}" '
                      f'"{CUTS_ROOT / "test" / cam_id / new_name}"')


class Checker(object):

    def __init__(self, ds_root):
        self.ds_root = Path(ds_root)
        self.all = {
            'cam_1': [],
            'cam_2': [],
            'cam_3': []
        }
        for mode in ['train', 'test']:
            for cam in ['cam_1', 'cam_2', 'cam_3']:
                d = self.ds_root / mode / cam
                self.all[cam] += [clean_cpath(f) for f in d.files()]


    def check(self, maugeri_path):
        info = mpath2info(maugeri_path)
        name = info['datestr']
        cam = f'cam_{info["camera-id"]}'
        return name in self.all[cam]


def demo():
    mroot = Path('/goat-nas/Datasets/spal/maugeri_ds/goods')
    croot = Path('/goat-nas/Datasets/spal/spal_cuts')
    ck = Checker('/goat-nas/Datasets/spal/spal_cuts')

    counter = {
        1: 0,
        2: 0,
        3: 0
    }

    for day_dir in mroot.dirs():
        for mpath in day_dir.files():
            info = mpath2info(fpath=mpath)
            #
            if ck.check(mpath) is False:
                cut = get_resized_cut(mpath)
                cam_name = f'cam_{info["camera-id"]}'
                counter[info['camera-id']] += 1
                cname = info['datestr'] + '.jpg'

                if random.random() <= 0.01:
                    cname = 'good_' + cname
                    cpath = croot / 'test' / cam_name / cname

                else:
                    cpath = croot / 'train' / cam_name / cname

                print(f'cv2.imwrite("{cpath}", cut)')
                cv2.imwrite(cpath, cut)
                # print((croot / out_mode / cam_name).exists(), '\n')


if __name__ == '__main__':
    cpath2info('/goat-nas/Datasets/spal/spal_cuts/train/cam_2/2022_02_02_08_47_24.jpg')
