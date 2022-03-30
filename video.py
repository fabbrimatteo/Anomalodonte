import os
import string

import cv2
import numpy as np
import torch
from path import Path

import ad_drawer
from conf import Conf
from evaluator import Evaluator
from lof import Loffer


ST = string.digits + string.ascii_letters

SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def video(mode, exp_name, show_if_gt=0.):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _cpath = cnf.exp_log_path / 'clusters' / 'clustering_dict.pth'
    clust = torch.load(_cpath)

    cfilter = np.array(clust['cardinalities']) < 16
    centroids = clust['centroids']
    centroids[cfilter] = np.nan

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    os.system(f'rm -r "{cnf.exp_log_path}/demo_{mode}"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_{mode}"')

    for img_path in in_dir_path.files():
        img = cv2.imread(img_path)

        flat_code = evaluator.model.get_flat_code(img)
        dists = np.linalg.norm(flat_code - centroids, axis=1)
        clust_idx = np.nanargmin(dists)
        # dist = dists[clust_idx]

        cs = clust['clusters'][clust_idx]
        dists = np.linalg.norm(flat_code - cs, axis=1)
        dists = np.sort(dists)
        dist = dists[:16].mean()

        anomaly_perc = evaluator.model.get_code_anomaly_perc(
            img, anomaly_th, max_val=999
        )

        prob = np.clip(anomaly_perc, 0, 100) / 100.

        print(f'$> "{img_path.basename()}": {anomaly_perc:.2f}%')

        label_true = img_path.basename().split('_')[0]

        if mode == 'train':
            label_true = 'OK'
        else:
            label_true = 'OK' if label_true == 'good' else 'KO'

        label_pred = 'OK' if prob < 0.5 else 'KO'
        out_img = ad_drawer.show_anomaly(
            img, prob, header=f'GT: {label_true}, pred: {label_pred}', ret=True
        )

        card = clust['cardinalities'][clust_idx]
        name = img_path.basename().split('@')[-1]
        name = f'{int(round(anomaly_perc)):03d}_{name}'
        name = name.replace('good_', '')
        name = name.replace('bad_', '')
        # name = f'{int(round(anomaly_perc)):03d}' \
        #        f'@c{clust_idx:02d}w{dist:.2f}_{card}--{name}'

        if mode == 'train' and prob < 0.85:
            continue
        out_path = cnf.exp_log_path / f'demo_{mode}' / name
        cv2.imwrite(out_path, out_img)


def video2(mode, exp_name):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    os.system(f'rm -r "{cnf.exp_log_path}/demo_{mode}"')
    os.system(f'mkdir "{cnf.exp_log_path}/demo_{mode}"')

    loffer = Loffer(
        train_dir=SPAL_PATH / 'train' / cnf.cam_id,
        model=evaluator.model, n_neighbors=16
    )

    test_dir = SPAL_PATH / 'test' / cnf.cam_id

    errors = 0
    for img_path in test_dir.files():
        img = cv2.imread(img_path)
        anomaly_perc = loffer.get_anomaly_perc(img, max_val=999)
        anomaly_perc = int(round(anomaly_perc))

        label_true = img_path.basename().split('_')[0]

        if mode == 'train':
            label_true = 'OK'
        else:
            label_true = 'OK' if label_true == 'good' else 'KO'

        label_pred = 'OK' if anomaly_perc < 50 else 'KO'
        if label_pred != label_true:
            errors += 1

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
    # demo_anomaly_perc(exp_name='cam3', mode='train', show_if_gt=0.95)
    # demo_clustering(exp_name='cam1', mode='train', n_clusters=8)
    # demo_clustering(exp_name='cam2', mode='train', n_clusters=8)
    # demo_clustering(exp_name='cam3_big', mode='train', n_clusters=8)
    video2(mode='test', exp_name='cam1')
    video2(mode='test', exp_name='cam2')
    video2(mode='test', exp_name='cam3')
