from typing import Tuple

import cv2
import numpy as np
import torch
from path import Path
from sklearn.cluster import KMeans

from conf import Conf
from models import Autoencoder


def flat_code_to_tensor(flat_code, device):
    code = flat_code.reshape((1, -1, 4, 4))
    return torch.tensor(code).to(device)


def tensor_to_img(x):
    if x.shape[0] == 1:
        x = x[0]
    x = x.cpu().numpy().transpose((1, 2, 0))
    return (255 * x).astype(np.uint8)


def clustering(exp_name, in_dir, n_clusters=8):
    # type: (str, str, int) -> Tuple[np.ndarray, np.ndarray]
    cnf = Conf(exp_name=exp_name)

    in_dir = Path(in_dir)

    model = Autoencoder.init_from_pth(
        pth_file_path=(cnf.exp_log_path / 'best.pth'),
        device=cnf.device
    )

    flat_codes = []
    paths = list(in_dir.files())
    for img_path in paths:
        print(f'\r$> processing image "{img_path.basename()}"', end='')
        img = cv2.imread(img_path)
        flat_code = model.get_flat_code(img)
        flat_codes.append(flat_code)
    print()

    kmeans = KMeans(n_clusters=n_clusters, n_init=128)

    kmeans_out = kmeans.fit(np.array(flat_codes))
    centroids = kmeans_out.cluster_centers_
    labels = kmeans_out.labels_
    cardinalities = [(labels == i).sum() for i in range(n_clusters)]

    decoded_centroids = []
    flat_codes = np.array(flat_codes)
    clusters = []
    for i in range(n_clusters):
        clusters.append(flat_codes[labels == i])
        centroid = centroids[i]
        code = flat_code_to_tensor(centroid, device=cnf.device)
        y_pred = model.decode(code)
        decoded_centroid = tensor_to_img(y_pred)
        decoded_centroids.append(decoded_centroid)

    out_dict = {
        'centroids': centroids,
        'decoded_centroids': decoded_centroids,
        'clusters': clusters,
        'labels': labels,
        'cardinalities': cardinalities,
        'paths': [str(p) for p in paths],
    }

    return out_dict
