from time import sleep

import cv2
from path import Path
import os

from dataset import Emitter
from dataset.ds_utils import cut_full_img
from models.autoencoder_plus import AutoencoderPlus
from conf import Conf
from eval.lof import Loffer
from visual_utils import draw_anomaly_ui
from update_proc.day_db import DayDB


def demo():
    exp_name = 'progression'
    proj_log_path = '/goat-nas/Datasets/spal/progression_demo/experiments'
    cnf = Conf(exp_name=exp_name, proj_log_path=proj_log_path)
    n_days = 10
    n_sample_per_day = 10
    n_neighbors = 20

    if cnf.exp_log_path.exists():
        print(f'{cnf.exp_log_path} already exists!')
        exit()

    # copying train and test sets
    print('Copying train and test sets...')
    cnf.exp_log_path.makedirs_p()
    cmd = f'cp -r "{cnf.ds_path.abspath()}/train" "{cnf.exp_log_path.abspath()}"'
    os.system(cmd)
    cmd = f'cp -r "{cnf.ds_path.abspath()}/test" "{cnf.exp_log_path.abspath()}"'
    os.system(cmd)

    emitter = Emitter(
        maugeri_root='/goat-nas/Datasets/spal/maugeri_ds',
        cuts_root='/goat-nas/Datasets/spal/spal_cuts',
        cam_id=1, start_idx=5000
    )

    # init autoencoder
    model = AutoencoderPlus.init_from_pth(
        '/home/matteo/PycharmProjects/Anomalodonte/log/progression/last.pth',
        device=cnf.device, mode='eval'
    )

    print('Initializing Loffer...')
    loffer = Loffer(
        train_dir=cnf.exp_log_path / 'train',
        model=model, n_neighbors=n_neighbors
    )

    for day in range(n_days):
        print(f'\nDAY #{day}')
        day_db = DayDB(root_dir=cnf.exp_log_path)

        for i in range(n_sample_per_day):
            read_ok, frame = emitter.read()
            if not read_ok:
                break
            else:
                cut = cut_full_img(img=frame, cam_name='cam_1', side=256)
                anomaly_score = int(round(loffer.get_anomaly_score(cut)))
                day_db.add(img_cut=cut, anomaly_score=anomaly_score)

                print(f'───$> sample #{i} of day #{day}: anomaly score {anomaly_score}')

                # cut = draw_anomaly_ui(cut, anomaly_score)
                # cv2.imshow('', cut)
                # cv2.waitKey(1)

        day_db.update_dataset(u_range=(45, 150), expl_perc=5)


if __name__ == '__main__':
    demo()
