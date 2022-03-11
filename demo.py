import cv2
import numpy as np
import torch
from path import Path
from sklearn.cluster import KMeans

import ad_drawer
from conf import Conf
from evaluator import Evaluator


SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def demo_anomaly_perc(in_dir_path, exp_name, show_if_gt=0.):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    for img_path in in_dir_path.files():
        img = cv2.imread(img_path)
        anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)

        print(f'$> "{img_path.basename()}": {anomaly_perc:.2f}%')

        prob = anomaly_perc / 100
        if prob > show_if_gt:
            ad_drawer.show_anomaly(img, prob, label=img_path.basename())


def demo_clustering(in_dir_path, exp_name, n_clusters=8):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

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

    for i in range(n_clusters):
        centroid = centroids[i]
        centroid = centroid.reshape((1, 3, 4, 4))
        centroid = torch.tensor(centroid).to(cnf.device)
        y_pred = evaluator.model.decode(centroid)
        y_pred = y_pred[0].cpu().numpy().transpose((1, 2, 0))
        y_pred = (255 * y_pred).astype(np.uint8)
        cv2.imshow(f'C{i}: {len(labels[labels == i])}', y_pred[:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    demo_clustering(
        exp_name='reg_up2',
        in_dir_path=SPAL_PATH / 'train' / 'cam_1',
    )
