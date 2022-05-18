import os

import cv2
import numpy as np
from path import Path

from visual_utils import draw_anomaly_ui
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
        model=model, n_neighbors=20
    )

    # "demo_test" contains all test images with their predicted
    # anomaly percentage and class; the file name starts with the
    # predicted anomaly score
    os.system(f'rm -r "{cnf.exp_log_path}/demo_test"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_test"')

    # "demo_train" contains all the detected outliers in the training
    # set with theirs anomaly percentage and class; the file name
    # starts with the predicted anomaly score
    os.system(f'rm -r "{cnf.exp_log_path}/demo_train"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_train"')

    test_dir = SPAL_PATH / 'test' / cnf.cam_id
    metrics, _ = loffer.evaluate(test_dir)

    # save training outliers in "demo_train"
    for key in loffer.train_outliers:
        an_perc = loffer.train_outliers[key]
        old_name = key.basename()
        new_name = f'{int(round(an_perc)):03d}_{old_name}'
        os.system(f'cp "{key}" "{cnf.exp_log_path}/demo_train/{new_name}"')

    # save test results in "demo_test"
    for img_path in test_dir.files():
        img = cv2.imread(img_path)
        anomaly_score = loffer.get_anomaly_score(img)
        anomaly_score = int(round(anomaly_score))

        label_true = img_path.basename().split('_')[0]
        label_true = LABEL_MAP[label_true]

        # anomaly threshold is 50 by default
        label_pred = 'OK' if anomaly_score < 50 else 'KO'

        # since NC samples are poorly annotated, we do not
        # consider an error if our model predicts "KO" for
        # a sample that is labelled as "NC"
        e1 = label_true == 'OK' and label_pred == 'KO'
        e2 = label_true == 'KO' and label_pred == 'OK'
        e3 = label_true == 'NC' and label_pred == 'OK'
        err = 'err_' if (e1 or e2 or e3) else ''

        # draw output image with GT and prediction
        header = f'prediction: {label_pred}  (GT: {label_true})'
        out_img = draw_anomaly_ui(
            img, anomaly_score, header=header
        )

        # save output image
        name = img_path.basename().split('@')[-1]
        name = f'{int(round(anomaly_score)):03d}_{err}{name}'
        name = name.replace('good_', '')
        name = name.replace('bad_', '')
        name = name.replace('nc_', '')
        out_path = cnf.exp_log_path / f'demo_test' / name
        cv2.imwrite(out_path, out_img)

    print(f'$> metrics: {metrics}')


if __name__ == '__main__':
    demo(exp_name='lof3_d')
