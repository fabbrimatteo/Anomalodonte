from time import sleep

import cv2
import mmu
from path import Path

from dataset import Emitter
from dataset.ds_utils import cut_full_img
from models.autoencoder_plus import AutoencoderPlus
from conf import Conf
from eval.lof import Loffer
from visual_utils import draw_anomaly_ui


def demo():
    cnf = Conf(exp_name='progression')

    emitter = Emitter(
        maugeri_root='/goat-nas/Datasets/spal/maugeri_ds',
        cuts_root='/goat-nas/Datasets/spal/spal_cuts',
        cam_id=1, start_idx=0
    )

    # init autoencoder
    model = AutoencoderPlus.init_from_pth(
        cnf.exp_log_path / 'last.pth',
        device=cnf.device, mode='eval'
    )

    loffer = Loffer(
        train_dir=cnf.ds_path / 'train',
        model=model, n_neighbors=20
    )

    for day in range(5):
        print(f'\nDAY #{day}')
        for i in range(10):
            read_ok, frame = emitter.read()
            if not read_ok:
                break
            else:
                cut = cut_full_img(img=frame, cam_name='cam_1', side=256)

                # cut = cv2.imread('/goat-nas/Datasets/spal/progression_demo/test/bad_2022_02_10_10_16_11.jpg')
                # cut = cv2.imread('/goat-nas/Datasets/spal/progression_demo/test/good_2022_02_21_20_05_14.jpg')

                anomaly_score = loffer.get_anomaly_score(cut)
                cut = draw_anomaly_ui(cut, anomaly_score)

                print(f'───$> sample #{i} of day #{day}')
                cv2.imshow('', cut)
                cv2.waitKey(0)


if __name__ == '__main__':
    demo()
