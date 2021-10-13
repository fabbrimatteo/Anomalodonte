import matplotlib
import piq


matplotlib.use('TkAgg')
import cv2
import numpy as np
from conf import Conf
from models.autoencoder import SimpleAutoencoder
from pre_processing import PreProcessingTr
import torch


def get_anomaly_map(x_pred, x_true, n_scales=3):
    h, w, _ = x_true.shape
    maps = [((x_pred - x_true) ** 2).mean(-1)]
    for i in range(1, n_scales):
        h, w = h // 2, w // 2
        resized_x_pred = cv2.resize(x_pred, (h, w))
        resized_x_true = cv2.resize(x_true, (h, w))
        m = ((resized_x_pred - resized_x_true) ** 2).mean(-1)
        maps.append(cv2.resize(m, (x_true.shape[0], x_true.shape[1])))
    return np.sum(maps, 0)


def get_ideal_pred(model, x, p_list, device):
    code = model.encode(x)[0]
    all_diffs = [torch.nn.MSELoss()(p, code).item() for p in p_list]
    selected_prototype = p_list[np.argmin(all_diffs)]
    return model.decode(selected_prototype.unsqueeze(0).to(device))


def get_stats(array):

    if len(array) == 0:
        return None

    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    stats = {
        'mean': np.mean(array),
        'std': np.std(array),
        'min': np.min(array),
        'max': np.max(array),
        'q0.05': np.quantile(array, 0.05),
        'q0.25': q1,
        'q0.50': np.quantile(array, 0.50),
        'q0.75': q3,
        'q0.95': np.quantile(array, 0.95),
        'w0': q1 - 1.5 * (q3 - q1),
        'w1': q3 + 1.5 * (q3 - q1),
        'wth2': q3 + 2 * (q3 - q1),
    }

    return stats


def get_anomaly_type(file_name):
    if 'good' in file_name:
        return '000'
    else:
        x = file_name.split('_')[1]
        a1, a2, a3 = int(x[1]), int(x[2]), int(x[3])
        return f'{a1}{a2}{a3}'


class StatsExtractor(object):

    def __init__(self, exp_name, mode):
        self.cnf = Conf(exp_name=exp_name)
        self.mode = mode

        pth_file_path = self.cnf.exp_log_path / 'best.pth'
        self.model = SimpleAutoencoder.init_from_pth(pth_file_path=pth_file_path, device=self.cnf.device)

        self.trs = PreProcessingTr(
            resized_h=self.cnf.resized_h, resized_w=self.cnf.resized_w,
            crop_x_min=self.cnf.crop_x_min, crop_y_min=self.cnf.crop_y_min, crop_side=self.cnf.crop_side
        )

        self.ds_path = self.cnf.ds_path / mode
        self.good_img_scores = []
        self.bad_img_scores = []


    def get_stats(self):

        ds_len = len(self.ds_path.files())
        for i, img_path in enumerate(self.ds_path.files()):

            print(f'\r$> analyzing img {i + 1} of {ds_len}', end='')

            img = cv2.imread(img_path)

            # apply pre-processing
            x = self.trs(img).unsqueeze(0).to(self.cnf.device)

            # target and prediction: type=torch.Tensor
            # with shape (C, H, W) and values in [0, 1]
            y_true = x.clone()
            y_pred = self.model.forward(x)

            anomaly_score = piq.MultiScaleSSIMLoss()(y_pred, y_true).item()

            # get GT class (good or bad) from image name
            is_good = get_anomaly_type(img_path.basename()).startswith('0')
            if is_good:
                self.good_img_scores.append(anomaly_score)
            else:
                self.bad_img_scores.append(anomaly_score)

        stats = {
            'good': get_stats(self.good_img_scores),
            'bad': get_stats(self.bad_img_scores)
        }
