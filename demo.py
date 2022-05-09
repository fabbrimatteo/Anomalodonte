import os

import cv2
import numpy as np
from path import Path

import ad_drawer
from conf import Conf
from eval.lof import Loffer
from models.autoencoder_plus import AutoencoderPlus


SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')

LABEL_MAP = {
    'good': 'OK',
    'bad': 'KO',
    'nc': 'NC'
}


def demo(exp_name):
    # type: (str) -> None
    cnf = Conf(exp_name=exp_name)

    # init autoencoder
    model = AutoencoderPlus.init_from_pth(
        cnf.exp_log_path / 'best.pth',
        device=cnf.device, mode='eval'
    )

    # init "Loffer" object
    loffer = Loffer(
        train_dir=SPAL_PATH / 'train' / cnf.cam_id,
        model=model, n_neighbors=12
    )

    # create/clear output directories
    os.system(f'rm -r "{cnf.exp_log_path}/demo_test"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_test"')
    os.system(f'rm -r "{cnf.exp_log_path}/demo_train"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_train"')

    test_dir = SPAL_PATH / 'test' / cnf.cam_id
    loffer.evaluate(test_dir)

    for key in loffer.train_outliers:
        an_perc = loffer.train_outliers[key]
        old_name = key.basename()
        new_name = f'{int(round(an_perc)):03d}_{old_name}'
        os.system(f'cp "{key}" "{cnf.exp_log_path}/demo_train/{new_name}"')

    for img_path in test_dir.files():
        img = cv2.imread(img_path)
        anomaly_perc = loffer.get_anomaly_perc(img, max_val=999)
        anomaly_perc = int(round(anomaly_perc))

        label_true = img_path.basename().split('_')[0]
        label_true = LABEL_MAP[label_true]

        label_pred = 'OK' if anomaly_perc < 50 else 'KO'

        e1 = label_true == 'OK' and label_pred == 'KO'
        e2 = label_true == 'KO' and label_pred == 'OK'
        e3 = label_true == 'NC' and label_pred == 'OK'
        err = 'err_' if (e1 or e2 or e3) else ''

        anomaly_prob = np.clip(anomaly_perc / 100, 0, 1)

        header = f'prediction: {label_pred}  (GT: {label_true})'
        out_img = ad_drawer.show_anomaly(
            img, anomaly_prob, header=header, ret=True
        )

        name = img_path.basename().split('@')[-1]
        name = f'{int(round(anomaly_perc)):03d}_{err}{name}'
        name = name.replace('good_', '')
        name = name.replace('bad_', '')
        name = name.replace('nc_', '')

        out_path = cnf.exp_log_path / f'demo_test' / name
        cv2.imwrite(out_path, out_img)
    print(f'$> metrics: {loffer.evaluate(test_dir)}')


if __name__ == '__main__':
    demo(exp_name='lof22')
