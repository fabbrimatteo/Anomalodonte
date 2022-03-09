import random
from datetime import datetime

import cv2
from path import Path
import os

BOX_DICT = {
    'cam_1': [480, 100, 1140, 760],
    'cam_2': [455, 148, 958, 842],
    'cam_3': [540, 212, 1122, 888],
}

SPAL_ROOT = Path('/goat-nas/Datasets/spal')

MAUGERI_DS = {
    'train': SPAL_ROOT / '2022_03_09' / 'goods',
    'test': SPAL_ROOT / '2022_03_09' / 'bads'
}

CUTS_ROOT = Path('/goat-nas/Datasets/spal/spal_cuts')
CUTS_ROOT.mkdir_p()


def fpath2info(fpath):
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
        'tag': tag,
        'datetime': date,
        'camera-id': int(cam),
        'label': label,
        'original_name': str(fpath.basename()),
        'new_name': date.strftime('%Y_%m_%d_%H_%M_%S.jpg'),
    }

    return info


def get_resized_cut(img_path):
    info = fpath2info(img_path)
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

        info = fpath2info(img_path)
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


if __name__ == '__main__':
    mv_some_goods_to_test(n_imgs_to_move=64)
