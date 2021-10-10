import matplotlib
import piq

matplotlib.use('TkAgg')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from conf import Conf
from models.autoencoder import SimpleAutoencoder
from pre_processing import PreProcessingTr
from post_processing import tensor2img
from piq import MultiScaleSSIMLoss
from torch import nn


def get_anomaly_map(x_pred, x_true, n_scales=3):
    h, w, _ = x_true.shape
    maps = [((x_pred - x_true)**2).mean(-1)]
    for i in range(1, n_scales):
        h, w = h // 2, w // 2
        resized_x_pred = cv2.resize(x_pred, (h, w))
        resized_x_true = cv2.resize(x_true, (h, w))
        m = ((resized_x_pred - resized_x_true)**2).mean(-1)
        maps.append(cv2.resize(m, (x_true.shape[0], x_true.shape[1])))
    return np.sum(maps, 0)


def print_stats(array):
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    print(f'\t$> mean: {np.mean(array):.2f}')
    print(f'\t$> std: {np.std(array):.2f}')
    print(f'\t$> min: {np.min(array):.2f}')
    print(f'\t$> max: {np.max(array):.2f}')
    print(f'\t$> q0.05: {np.quantile(array, 0.05):.2f}')
    print(f'\t$> q0.25: {q1:.2f}')
    print(f'\t$> q0.50: {np.quantile(array, 0.50):.2f}')
    print(f'\t$> q0.75: {q3:.2f}')
    print(f'\t$> q0.95: {np.quantile(array, 0.95):.2f}')
    print(f'\t$> whiskers: [{q1 - 1.5 * (q3 - q1):.2f}, {q3 + 1.5 * (q3 - q1):.2f}]')


def get_anomaly_type(file_name):
    if 'good' in file_name:
        return '000'
    else:
        x = file_name.split('_')[1]
        a1, a2, a3 = int(x[1]), int(x[2]), int(x[3])
        return f'{a1}{a2}{a3}'


def results(exp_name, show_visual_res=False):
    cnf = Conf(exp_name=exp_name)
    cnf.device = 'cpu'

    model = SimpleAutoencoder.init_from_pth(pth_file_path=cnf.exp_log_path / 'best.pth', device=cnf.device)
    trs = PreProcessingTr(
        resized_h=256, resized_w=256,
        crop_x_min=812, crop_y_min=660, crop_side=315
    )

    all_paths = list((cnf.ds_path / 'test').files())
    random.shuffle(all_paths)

    good_img_scores = []
    bad_img_scores = []
    for i, img_path in enumerate(all_paths):

        print(f'\r$> analyzing img {i + 1} of {len(all_paths)}', end='')

        img = cv2.imread(img_path)

        # apply pre-processing
        x = trs(img).unsqueeze(0).to(cnf.device)

        # target and prediction: type=torch.Tensor
        # with shape (C, H, W) and values in [0, 1]
        y_true = x.clone()
        y_pred = model.forward(x)
        mse_map = ((y_pred[0] - y_true[0])**2).mean(0)

        # target and prediction: type=np.ndarray
        # with shape (H, W, C) and values in [0, 255]
        img_true = tensor2img(y_true)
        img_pred = tensor2img(y_pred)
        mse_map = tensor2img(mse_map)

        anomaly_score = piq.MultiScaleSSIMLoss()(y_pred, y_true)
        anomaly_score *= 255

        if show_visual_res:
            fig, axes = plt.subplots(1, 3, figsize=(40, 30), dpi=100)
            axes[0].imshow(img_true)
            axes[0].set_title(f'true ({img_path.basename()})')
            axes[1].imshow(img_pred)
            axes[1].set_title(f'pred')
            axes[2].imshow(mse_map, cmap='jet', vmin=0, vmax=255)
            axes[2].set_title(f'{100*(anomaly_score/165.62):.2f}%')
            plt.show()
            plt.close(fig)

        # get GT class (good or bad) from image name
        is_good = get_anomaly_type(img_path.basename()).startswith('0')

        if is_good:
            good_img_scores.append(anomaly_score)
        else:
            bad_img_scores.append(anomaly_score)

    if len(good_img_scores) > 4:
        print(f'$> GOOD:')
        print_stats(good_img_scores)
    if len(bad_img_scores) > 4:
        print(f'$> BAD:')
        print_stats(bad_img_scores)

    plt.figure(1)
    if len(bad_img_scores) > 0:
        plt.hlines(np.quantile(good_img_scores, 0.50), 0, 3, linestyles='--', colors='#e74c3c')
        plt.hlines(np.quantile(good_img_scores, 0.25), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(good_img_scores, 0.75), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(bad_img_scores, 0.25), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(bad_img_scores, 0.75), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.boxplot(
            [good_img_scores, bad_img_scores], labels=['good', 'bad'],
            showfliers=True, medianprops={'color': '#e74c3c'}
        )
    else:
        plt.boxplot(good_img_scores, labels=['good'], showfliers=True, medianprops={'color': '#e74c3c'})
    plt.show()

    return 0


if __name__ == '__main__':
    results(exp_name='ultra_small', show_visual_res=False)
