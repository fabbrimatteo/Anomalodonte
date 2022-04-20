import cv2
from path import Path
import mmu

CUTS = {
    "cam_1": [480, 100, 1140, 760],
    "cam_2": [359, 148, 1053, 842],
    "cam_3": [493, 212, 1169, 888]
}

MG_ROOT = Path('/goat-nas/Datasets/spal')
MG_ROOT = MG_ROOT / 'output_maugeri_aprile_2022/anomaly_test'

DS_ROOT = Path('/goat-nas/Datasets/spal/spal_cuts')

OLD_ROOT = Path('/goat-nas/Datasets/spal/2022_03_09')


def fix():
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
        print(cut.shape)
        cv2.imshow('', cut)
        cv2.waitKey()

        new_name = f'nc_{yyyy}_{mm}_{dd}_{hh}_{min}_{ss}.jpg'
        print(img_path)
        print(DS_ROOT / 'test' / cam_tag / new_name)
        print((DS_ROOT / 'test' / cam_tag).exists())
        print()


def get_resized_cut(img, cam_id):
    x_min, y_min, x_max, y_max = CUTS[cam_id]
    cut = img[y_min:y_max, x_min:x_max]
    return cv2.resize(cut, (256, 256))


class ReverseFinder(object):

    def __init__(self):
        o1 = list((OLD_ROOT / 'bads').files())
        o2 = []
        for sub_dir in (OLD_ROOT / 'goods').dirs():
            o2 += list(sub_dir.files())
        all_old_paths = o1 + o2
        all_old_names = []
        for f in all_old_paths:
            f = f.basename().replace('OK.bmp', '')
            f = f.replace('KO.bmp', '')
            f = f.replace('.bmp', '')
            f = f.split('_')[0] + '_' + f.split('_')[-1]
            all_old_names.append(f)

        self.all_old_paths = all_old_paths
        self.all_old_names = all_old_names


    def find(self, new_path):
        new_path = Path(new_path)
        new_name = new_path.basename()
        new_name = new_name.replace('bad_', '')
        new_name = new_name.replace('good_', '')
        new_name = new_name.replace('nc_', '')
        new_name = new_name.replace('_', '').replace('.jpg', '')
        cam_id = new_path.parent.basename().split('_')[-1]
        new_name = new_name + f'_{cam_id}'
        try:
            idx = self.all_old_names.index(new_name)
            return self.all_old_paths[idx]
        except ValueError:
            return None


def find(new_path):
    rf = ReverseFinder()
    for mode in ['test', 'train']:
        for cam_dir in (DS_ROOT / mode).dirs():
            if 'cam_1' in cam_dir:
                continue

            for img_path in cam_dir.files():
                old_img_path = rf.find(img_path)
                assert old_img_path is not None

                new_img = cv2.imread(img_path)
                old_img = cv2.imread(old_img_path)

                old_img = get_resized_cut(old_img, cam_dir.basename())
                # cv2.imwrite(img_path, old_img)
                cv2.imshow('', old_img)
                cv2.waitKey()

                cv2.imshow('', new_img)
                cv2.waitKey()


if __name__ == '__main__':
    find(new_path='/goat-nas/Datasets/spal/spal_cuts/test/cam_3/bad_2022_02_09_17_21_12.jpg')
