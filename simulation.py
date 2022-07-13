from trainer import Trainer
from time import sleep

import cv2
from path import Path
import os
import numpy as np

from dataset import Emitter
from dataset.ds_utils import cut_full_img
from models.autoencoder_plus import AutoencoderPlus
from conf import Conf
from eval.lof import Loffer
from visual_utils import draw_anomaly_ui
from update_proc.day_db import DayDB



def demo():
    exp_name = Path('progression')
    proj_log_path = Path('/goat-nas/Datasets/spal/progression_demo/experiments')
    pth_file_path = Path('/home/matteo/PycharmProjects/Anomalodonte/log/progression/last.pth')
    yaml_file_path = Path('/home/matteo/PycharmProjects/Anomalodonte/conf/progression.yaml')
    log_dir_path = proj_log_path / exp_name / 'log'
    daily_res_path = proj_log_path / exp_name / 'daily_res'

    cnf = Conf(exp_name=exp_name, proj_log_path=proj_log_path)
    n_days = 10
    n_sample_per_day = 500
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

    for day in range(n_days):
        print(f'\nDAY #{day}')

        cnf = Conf(exp_name=exp_name, proj_log_path=proj_log_path)
        exp_name_day_pretrain = exp_name + f'_{day}_pretrain'
        exp_name_day = exp_name + f'_{day}'
        today_res_path = daily_res_path / f'{day}'
        today_res_path.makedirs_p()

        # init autoencoder
        model = AutoencoderPlus.init_from_pth(
            pth_file_path,
            device=cnf.device, mode='eval'
        )

        print('Initializing Loffer...')
        loffer = Loffer(
            train_dir=cnf.exp_log_path / 'train',
            model=model, n_neighbors=n_neighbors
        )

        day_db = DayDB(root_dir=cnf.exp_log_path, debug=True)

        for i in range(n_sample_per_day):
            read_ok, frame = emitter.read()
            if not read_ok:
                break
            else:
                cut = cut_full_img(img=frame, cam_name='cam_1', side=256)
                anomaly_score = int(round(loffer.get_anomaly_score(cut)))
                day_db.add(img_cut=cut, anomaly_score=anomaly_score)

                img_ui = draw_anomaly_ui(cut, anomaly_score)
                cv2.imwrite(today_res_path / f'{i:03d}.jpg', img_ui)

                print(f'───$> sample #{i} of day #{day}: anomaly score {anomaly_score}')

                # cut = draw_anomaly_ui(cut, anomaly_score)
                # cv2.imshow('', cut)
                # cv2.waitKey(1)

        # day_db.update_dataset(u_range=(45, 150), expl_perc=5)
        day_db.update_dataset_all()

        # pretraining
        cnf = Conf(exp_name=exp_name_day_pretrain,
                   proj_log_path=log_dir_path,
                   yaml_file_path=yaml_file_path)
        cnf.ds_path = proj_log_path / exp_name
        cnf.epochs = 2
        trainer = Trainer(cnf=cnf)
        trainer.run()

        # init autoencoder
        pth_file_path_pretrain = log_dir_path / exp_name_day_pretrain / 'last.pth'
        model = AutoencoderPlus.init_from_pth(
            pth_file_path_pretrain,
            device=cnf.device, mode='eval'
        )

        print('Initializing Loffer after Pretraining...')
        loffer = Loffer(
            train_dir=proj_log_path / exp_name / 'train',
            model=model, n_neighbors=n_neighbors
        )
        train_set = loffer.get_train_labels()

        out_split_dir_good = log_dir_path / exp_name_day_pretrain / 'good'
        out_split_dir_bad = log_dir_path / exp_name_day_pretrain / 'bad'
        out_split_dir_good.makedirs_p()
        out_split_dir_bad.makedirs_p()
        for sample in train_set:
            old_path = sample[0]
            name, label = old_path.basename().split('.')[0], sample[1]
            if name in list(day_db.info.keys()):
                if label == -1:
                    new_path = out_split_dir_bad / name + '.jpg'
                elif label == 1:
                    new_path = out_split_dir_good / name + '.jpg'

                cmd = f'cp "{old_path}" "{new_path}"'
                os.system(cmd)
                print(f'───$> {cmd}')

                # TODO: move into test set
                if label == -1:
                    cmd = f'rm "{old_path}"'
                    os.system(cmd)
                    print(f'───$> {cmd}')

        # training
        cnf = Conf(exp_name=exp_name_day,
                   proj_log_path=log_dir_path,
                   yaml_file_path=yaml_file_path)
        cnf.ds_path = proj_log_path / exp_name
        trainer = Trainer(cnf=cnf)
        trainer.run()

        pth_file_path = Path(cnf.exp_log_path / 'best.pth') # TODO: or last?


if __name__ == '__main__':
    demo()
