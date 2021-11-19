from typing import Dict
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

import utils


BoxplotDict = Dict[str, Dict[str, Union[float, np.ndarray]]]


def get_boxplot_dict(scores, labels_true):
    # type: (np.ndarray, np.ndarray) -> BoxplotDict

    gb_scores = {
        'good': scores[labels_true == 'good'],
        'bad': scores[labels_true == 'bad'],
    }

    # build `boxplot_dict`, that contains the statistics
    # relating to the distribution of the anomaly scores
    # of the "good" and "bad" test samples
    boxplot_dict = {}
    for key in ['good', 'bad']:
        mean = np.mean(gb_scores[key])
        std = np.std(gb_scores[key])
        min_value = np.min(gb_scores[key])
        max_value = np.max(gb_scores[key])
        q1 = np.quantile(gb_scores[key], 0.25)
        q2 = np.quantile(gb_scores[key], 0.50)
        q3 = np.quantile(gb_scores[key], 0.75)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        boxplot_dict[key] = {
            'lower_whisker': float(lower_whisker),
            'upper_whisker': float(upper_whisker),
            'min': float(min_value), 'max': float(max_value),
            'mean': float(mean), 'std': float(std),
            'q1': float(q1), 'q2': float(q2), 'q3': float(q3),
            'scores': np.array(gb_scores[key])
        }

    return boxplot_dict


def get_anomaly_th(boxplot_dict):
    # type: (BoxplotDict) -> float
    return boxplot_dict['good']['upper_whisker']


def plt_boxplot(boxplot_dict):
    # type: (BoxplotDict) -> np.ndarray
    fig = plt.figure(figsize=(8, 4), dpi=128)

    # draw a solid red line for the anomaly threshold,
    # i.e. the upper whisker of the "good" boxplot
    anomaly_th = boxplot_dict['good']['upper_whisker']
    plt.vlines(anomaly_th, 0, 3, colors='#1abc9c', linewidth=2)

    # draw the actual boxplot with the 2 boxes: "good" and "bad"
    plt.boxplot(
        [boxplot_dict['bad']['scores'], boxplot_dict['good']['scores']],
        labels=['bad', 'good'], showfliers=True,
        medianprops={'color': '#9b59b6'}, vert=False
    )
    plt.xlim(0, boxplot_dict['bad']['upper_whisker'] * 1.1)
    for i in [1, 2]:
        plt.hlines(
            i, 0, 10, linestyles='--', linewidth=0.5,
            colors='#95a5a6'
        )
    plt.ylim(0.5, 2.5)

    img = utils.pyplot_to_numpy(fig)
    plt.close(fig)

    return img
