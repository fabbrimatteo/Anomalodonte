import cv2
from path import Path


CUTS = {
    "cam_1": [480, 100, 1140, 760],
    "cam_2": [359, 148, 1053, 842],
    "cam_3": [493, 212, 1169, 888]
}

MG_ROOT = Path('/goat-nas/Datasets/spal')
MG_ROOT = MG_ROOT / 'output_maugeri_aprile_2022/anomaly_test'

DS_ROOT = Path('/goat-nas/Datasets/spal/spal_cuts')

OLD_ROOT = Path('/goat-nas/Datasets/spal/2022_03_09')


def new():

    counter = {
        'cam_1': 0,
        'cam_2': 0,
        'cam_3': 0
    }
    for img_path in MG_ROOT.files():

        img = cv2.imread(img_path)
        if 'tutto_ok' in img_path:
            continue
        cam_id = img_path.basename()[-5]
        cam_tag = f'cam_{cam_id}'
        data_str = img_path.basename().split('_')[0]

        yyyy = data_str[0:4]
        mm = data_str[4:6]
        dd = data_str[6:8]
        hh = data_str[8:10]
        min = data_str[10:12]
        ss = data_str[12:14]

        x_min, y_min, x_max, y_max = CUTS[cam_tag]
        cut = img[y_min:y_max, x_min:x_max]
        cut = cv2.resize(cut, (256, 256))

        counter[cam_tag] += 1
        new_name = f'nc_{yyyy}_{mm}_{dd}_{hh}_{min}_{ss}.jpg'

        cv2.imwrite(DS_ROOT / 'test' / cam_tag / new_name, cut)
        print(img_path)
        print(DS_ROOT / 'test' / cam_tag / new_name)
        print((DS_ROOT / 'test' / cam_tag).exists())
    print(counter)


if __name__ == '__main__':
    new()
