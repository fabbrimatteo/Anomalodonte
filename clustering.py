import cv2
import numpy as np
import torch
from path import Path
from sklearn.cluster import KMeans

import visual
from conf import Conf
from models import Autoencoder


def flat_code_to_tensor(flat_code, device):
    code = flat_code.reshape((1, -1, 4, 4))
    return torch.tensor(code).to(device)


def clustering(exp_name, in_dir, n_clusters=8):
    cnf = Conf(exp_name=exp_name)
    in_dir = Path(in_dir)

    model = Autoencoder.init_from_pth(
        pth_file_path=(cnf.exp_log_path / 'best.pth'),
        device=cnf.device
    )

    flat_codes = []
    for img_path in in_dir.files():
        print(f'\r$> processing image "{img_path.basename()}"', end='')
        img = cv2.imread(img_path)
        flat_code = model.get_flat_code(img)
        flat_codes.append(flat_code)
    print()

    kmeans = KMeans(n_clusters=n_clusters)

    kout = kmeans.fit(np.array(flat_codes))
    labels = kout.labels_
    centroids = kout.cluster_centers_

    clusters = []
    for i in range(n_clusters):
        centroid = centroids[i]
        code = flat_code_to_tensor(centroid, device=cnf.device)
        y_pred = model.decode(code)
        y_pred = y_pred[0].cpu().numpy().transpose((1, 2, 0))
        y_pred = (255 * y_pred).astype(np.uint8)
        clusters.append(centroid.cpu())
        cv2.imshow(f'C{i}: {len(labels[labels == i])}', y_pred[:, :, ::-1])
        cv2.imshow(f'C{i}-code', visual.code2img(centroid))

    cv2.waitKey()

    torch.save(clusters, cnf.exp_log_path / 'kmeans_centroids.pth')


clustering(exp_name='cam1', in_dir='/goat-nas/Datasets/spal/spal_cuts/train/cam_1')
