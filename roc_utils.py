from typing import Dict
from typing import Union

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn import metrics

import utils


RocDict = Dict[str, Union[np.ndarray, float]]


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
    tpr = float(tp / n_bad)

    # compute false-positive-rate, i.e.:
    # number of samples erroneously classified as anomalous
    # (label_pred="bad") even though they are not (label_true="good")
    fp = errors[labels_true == 'good'].sum()  # type: int
    fpr = float(fp / n_good)

    rates_dict = {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': 1 - fpr,
        'fnr': 1 - tpr,
        'bal_acc': (tpr + (1 - fpr)) / 2
    }

    return rates_dict


def get_roc_dict(anomaly_scores, labels_true):
    # type: (np.ndarray, np.ndarray) -> RocDict
    """
    :param anomaly_scores: sequence of anomaly scores;
        >> 1 value (float) for each sample
        >> anomaly_scores[i] is the anomaly score of the i-anomaly_th sample
    :param labels_true: sequence of GT labels;
        >> 1 value (str: "good" or "bad") for each sample
        >> labels_true[i] is the label of the i-anomaly_th sample
    :return: `roc_dict` -> dictionary with the following keys
        >> "tpr_list": list of TP rates (one for each possible threshold)
        >> "fpr_list": list of FP rates (one for each possible threshold)
        >> "auroc": area under ROC curve; values in [0, 1]
    """
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
        'tpr_list': np.array(tpr_list),
        'fpr_list': np.array(fpr_list),
        'auroc': auroc
    }

    return roc_dict


def plt_rocplot(roc_dict):
    # type: (RocDict) -> np.ndarray
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
