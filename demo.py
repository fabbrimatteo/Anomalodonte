import string

import cv2
import numpy as np
import torch
from path import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

import ad_drawer
import visual
from conf import Conf
from evaluator import Evaluator


ST = string.digits + string.ascii_letters

SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def demo_anomaly_perc(mode, exp_name, show_if_gt=0.):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    for img_path in in_dir_path.files():
        img = cv2.imread(img_path)
        anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)

        prob = anomaly_perc / 100
        if prob > show_if_gt:
            print(f'$> "{img_path.basename()}": {anomaly_perc:.2f}%')
            ad_drawer.show_anomaly(img, prob, label=img_path.basename())


def demo_clustering(mode, exp_name, n_clusters=8):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    flat_codes = []
    for img_path in in_dir_path.files():
        print(img_path)
        img = cv2.imread(img_path)
        flat_code = evaluator.model.get_flat_code(img)
        flat_codes.append(flat_code)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    kout = kmeans.fit(np.array(flat_codes))
    labels = kout.labels_
    centroids = kout.cluster_centers_

    clusters = []
    for i in range(n_clusters):
        centroid = centroids[i]

        tmp = (((centroid + 1) / 2) * len(ST)).astype(int)
        cname = ''
        for ci in tmp:
            cname += ST[ci]
        print('@@@', cname)

        centroid = centroid.reshape((1, -1, 4, 4))
        centroid = torch.tensor(centroid).to(cnf.device)
        y_pred = evaluator.model.decode(centroid)
        y_pred = y_pred[0].cpu().numpy().transpose((1, 2, 0))
        y_pred = (255 * y_pred).astype(np.uint8)
        clusters.append(centroid.cpu())
        cv2.imshow(f'C{i}: {len(labels[labels == i])}', y_pred[:, :, ::-1])
        cv2.imshow(f'C{i}-code', visual.code2img(centroid))

    cv2.waitKey()

    torch.save(clusters, cnf.exp_log_path / 'kmeans_centroids.pth')


def demo_lof(mode, exp_name, n_neighbors=12):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    flat_codes = []

    train_paths = list(in_dir_path.files())

    for img_path in train_paths:
        print(img_path)
        img = cv2.imread(img_path)
        flat_code = evaluator.model.get_flat_code(img)
        flat_codes.append(flat_code)

    novelty = True if mode == 'test' else False
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=novelty)

    import os
    os.system(f'rm -r "{cnf.exp_log_path}/train_sort"')
    os.system(f'mkdir "{cnf.exp_log_path}/train_sort"')

    if mode == 'train':
        labels = clf.fit_predict(np.array(flat_codes))
        scores = clf.negative_outlier_factor_
        argsort = np.argsort(scores)
        for idx in argsort:
            img_path = train_paths[idx]
            img = cv2.imread(train_paths[idx])
            label = labels[idx]
            score = scores[idx]
            anomaly_score = max(0, (score / (-1.5)) - 0.5)

            plus = ''
            if anomaly_score > 1.5:
                plus = '+'
            if anomaly_score > 2.0:
                plus = '++'
            if anomaly_score > 2.5:
                plus = '+++'

            name = img_path.basename().split('@')[-1]
            anomaly_score_100i = int(round(anomaly_score * 100))
            anomaly_score_100i = min(anomaly_score_100i, 999)
            name = f'{anomaly_score_100i:03d}@{name}'
            anomaly_score = min(anomaly_score, 1)

            if anomaly_score > 0.:
                # key = ad_drawer.show_anomaly(
                #     img, anomaly_score, img_path.basename(), plus
                # )
                import os

                os.system(f'ln -s '
                          f'"{img_path}" '
                          f'"{cnf.exp_log_path}/train_sort/{name}"')

                # cv2.imshow(f'{scores[idx]:.2f}', img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


if __name__ == '__main__':
    # demo_anomaly_perc(exp_name='cam2', mode='train', show_if_gt=0.95)
    # demo_clustering(exp_name='cam1_fc', mode='train', n_clusters=16)
    demo_lof(exp_name='cam3', mode='train')
