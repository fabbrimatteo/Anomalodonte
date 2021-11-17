from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn import metrics
import utils

def draw_boxplot(good_scores, bad_scores):
    plt.figure(figsize=(8, 4), dpi=128)
    flierprops = dict(marker='o', markerfacecolor='green', markeredgecolor='none')
    # draw the actual boxplot with the 2 boxes: "good" and "bad"
    data = [bad_scores.tolist(), good_scores.tolist()]
    plt.boxplot(
        data,
        showfliers=True, medianprops={'color': '#9b59b6'}, vert=False,
        flierprops=flierprops
    )
    plt.xlim(0, bad_scores.max() * 1.1)
    for i in [1, 2]:
        plt.hlines(
            i, 0, 10, linestyles='--', linewidth=0.5,
            colors='#95a5a6'
        )
    plt.ylim(0.5, 2.5)


def get_ad_rates(anomaly_scores, anomaly_th, labels_true):
    # type: (np.ndarray, float, np.ndarray) -> Dict[str, float]
    """
    :param anomaly_scores: sequence of anomaly scores;
        >> 1 value (float) for each sample
        >> anomaly_scores[i] is the anomaly score of the i-anomaly_th sample
    :param anomaly_th: anomaly threshold
        >> a sample with an anomaly score <= then `anomaly_th`
           is classified as "good" (not anomalous)
        >> a sample with an anomaly score > then `anomaly_th`
           is classified as "bad" (anomalous)
    :param labels_true: sequence of GT labels;
        >> 1 value (str: "good" or "bad") for each sample
        >> labels_true[i] is the label of the i-anomaly_th sample
    :return: rates dictionary, whith the following keys:
        >> "TPR": true positive rate
        >> "FPR": false positive rate
        >> "TNR": true negative rate
        >> "FNR": false negative rate
    """

    # compute predicted labels by thresholding
    # anomaly scores with the given threshold
    labels_pred = np.ones_like(labels_true)
    labels_pred[anomaly_scores > anomaly_th] = 'bad'
    labels_pred[anomaly_scores <= anomaly_th] = 'good'

    # number of GT samples with label "good" (non anomalous)
    n_good = (labels_true == 'good').sum()

    # number of GT samples with label "bad" (anomalous)
    n_bad = (labels_true == 'bad').sum()

    # boolean array of correct predictions, i.e.:
    # >> matches[i] is True <=> labels_pred[i] == labes_true[i]
    # >> matches[i] is False <=> labels_pred[i] != labes_true[i]
    matches = np.array(labels_pred == labels_true, dtype=bool)

    # boolean array of wrong predictions, i.e.
    # >> errors[i] is True <=> labels_pred[i] != labes_true[i]
    # >> errors[i] is False <=> labels_pred[i] == labes_true[i]
    errors = ~matches

    # compute true-positive-rate, i.e.:
    # number of correctly detected anomalies
    tp = matches[labels_true == 'bad'].sum()  # type: int
    tpr = tp / n_bad

    # compute false-positive-rate, i.e.:
    # number of samples erroneously classified as anomalous
    # (label_pred="bad") even though they are not (label_true="good")
    fp = errors[labels_true == 'good'].sum()  # type: int
    fpr = fp / n_good

    rates_dict = {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': 1 - fpr,
        'fnr': 1 - tpr,
        'bal_acc': (tpr + (1 - fpr)) / 2
    }

    return rates_dict


def get_roc_dict(anomaly_scores, labels_true):
    th_min = np.min(anomaly_scores)
    th_max = np.max(anomaly_scores)
    th_range = th_max - th_min

    fpr_list = []
    tpr_list = []
    best_ba = (0, th_min)
    for th in np.arange(th_min, th_max, th_range / 32):
        rates_dict = get_ad_rates(anomaly_scores, th, labels_true)

        tpr = rates_dict['tpr']
        fpr = rates_dict['fpr']

        if rates_dict['bal_acc'] > best_ba[0]:
            best_ba = (rates_dict['bal_acc'], th)

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    auroc = sklearn.metrics.auc(fpr_list, tpr_list)

    roc_dict = {
        'tpr_list': tpr_list,
        'fpr_list': fpr_list,
        'auroc': auroc
    }

    return roc_dict


def plt_rocplot(roc_dict):

    fpr_list = roc_dict['fpr_list']
    tpr_list = roc_dict['tpr_list']

    fig = plt.figure(figsize=(5, 5))
    plt.grid(True, linestyle='dashed', color='#bdc3c7', linewidth=0.75)
    plt.plot((-0.1, 1.1), (-0.1, 1.1), '--', color='#34495e', linewidth=1)
    plt.plot(fpr_list, tpr_list, 'o-', color='#16a085')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    img = utils.pyplot_to_numpy(fig)
    plt.close(fig)

    return img


def debug():
    N_GOOD = 256
    N_BAD = 128

    labels_true = ['good' for _ in range(N_GOOD)]
    labels_true += ['bad' for _ in range(N_BAD)]
    labels_true = np.array(labels_true)

    goods_scores = np.random.normal(2, 0.4, (N_GOOD,)).astype(float)
    bads_scores = np.random.normal(4, 1.1, (N_BAD,)).astype(float)
    anomaly_scores = np.concatenate([goods_scores, bads_scores], 0)

    g_min = goods_scores.min()
    g_max = goods_scores.max()
    b_min = bads_scores.min()
    b_max = bads_scores.max()

    points = [g_min, g_max, b_min, b_max]
    points.sort()
    r_min = points[1]
    r_max = points[2]
    print(points)
    #draw_boxplot(goods_scores, bads_scores)
    #plot_roc(anomaly_scores, labels_true)


if __name__ == '__main__':
    debug()