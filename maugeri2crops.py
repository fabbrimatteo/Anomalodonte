

from sift_aligner import SiftAligner
from path import Path
from dataset.ds_utils import mpath2info, BOX_DICT
import cv2


def crop():
    source_img_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS/04/12/20220412085004_601102626722200000327101_1.bmp')
    source_img = cv2.imread(source_img_path)

    x_min, y_min, x_max, y_max = BOX_DICT[f'cam_1']
    source_img = cv2.rectangle(source_img, (x_min, y_min), (x_max, y_max), (0,0,0), -1)
    source_img = cv2.resize(source_img, (0,0), fx=0.5, fy=0.5)
    sift_aligner = SiftAligner(source_img)


    data_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS')
    out_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/spal_cuts_aligned')

    for month_dir in data_path.dirs():
        for day_dir in month_dir.dirs():
            for file_name in day_dir.files():
                info = mpath2info(file_name)
                if info["camera-id"] == 1:
                    img = cv2.imread(file_name)

                    x_min, y_min, x_max, y_max = BOX_DICT[f'cam_{info["camera-id"]}']
                    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,0), -1)
                    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                    _, img = sift_aligner.align(img)
                    cv2.imshow('1', img)
                    cv2.imshow('2', source_img)
                    cv2.waitKey()






if __name__ == '__main__':
    crop()