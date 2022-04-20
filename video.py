import os

import cv2
import numpy as np
from path import Path

import ad_drawer
from conf import Conf
from eval.lof import Loffer
from models.autoencoder_plus import AutoencoderPlus


SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def video2(mode, exp_name):
    cnf = Conf(exp_name=exp_name)

    model = AutoencoderPlus.init_from_pth(
        cnf.exp_log_path / 'best.pth',
        device=cnf.device, mode='eval'
    )

    os.system(f'rm -r "{cnf.exp_log_path}/demo_{mode}"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_{mode}"')

    loffer = Loffer(
        train_dir=SPAL_PATH / 'train' / cnf.cam_id,
        model=model, n_neighbors=12
    )

    test_dir = SPAL_PATH / 'test' / cnf.cam_id

    for img_path in test_dir.files():
        img = cv2.imread(img_path)
        anomaly_perc = loffer.get_anomaly_perc(img, max_val=999)
        anomaly_perc = int(round(anomaly_perc))

        label_true = img_path.basename().split('_')[0]

        if mode == 'train':
            label_true = 'OK'
        else:
            if label_true == 'good':
                label_true = 'OK'
            elif label_true == 'bad':
                label_true = 'KO'
            elif label_true == 'nc':
                label_true = 'NC'

        if anomaly_perc < 50:
            label_pred = 'OK'
        else:
            label_pred = 'KO'

        anomaly_prob = np.clip(anomaly_perc / 100, 0, 1)

        header = f'prediction: {label_pred}  (GT: {label_true})'
        out_img = ad_drawer.show_anomaly(
            img, anomaly_prob, header=header, ret=True
        )

        name = img_path.basename().split('@')[-1]
        name = f'{int(round(anomaly_perc)):03d}_{name}'
        name = name.replace('good_', '')
        name = name.replace('bad_', '')

        out_path = cnf.exp_log_path / f'demo_{mode}' / name
        cv2.imwrite(out_path, out_img)
    print(f'$> metrics: {loffer.evaluate(test_dir)}')


if __name__ == '__main__':
    video2(mode='test', exp_name='lof1')
