
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


def clean():
    n_neighbors = 20
    trash_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/spal_cuts/trash/cam_1')

    cnf = Conf(exp_name='clean')
    cnf.ds_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/spal_cuts')
    # cnf.ds_path = Path('/goat-nas/Datasets/spal/spal_cuts')
    cnf.epochs = 2
    cnf.cam_id = 'cam_1'
    # trainer = Trainer(cnf=cnf)
    # trainer.run()

    # init autoencoder
    pth_file_path_pretrain = 'log/clean/last.pth'
    model = AutoencoderPlus.init_from_pth(
        pth_file_path_pretrain,
        device=cnf.device, mode='eval'
    )

    print('Initializing Loffer after Pretraining...')
    loffer = Loffer(
        train_dir= cnf.ds_path / 'train/cam_1',
        model=model, n_neighbors=n_neighbors
    )

    train_set = loffer.get_train_scores()
    train_set.sort(key=lambda x: x[1], reverse=True)

    for sample in train_set:
        old_path = sample[0]
        name, score = old_path.basename().split('.')[0], sample[1]
        print(score)
        img = cv2.imread(old_path)


        cv2.imshow('', img)
        cv2.waitKey()

        # if score > 50:
        #     new_path = trash_path / name + '.jpg'
        #     Path(old_path).move(new_path)




if __name__ == '__main__':
    clean()