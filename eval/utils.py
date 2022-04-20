from typing import Dict

import numpy as np


def bin_class_metrics(labels_pred, labels_true, mode='anomaly-detection'):
    # type: (np.ndarray, np.ndarray, str) -> Dict[str, float]
    """
    :param labels_pred: array of all predicted labels;
        >> anomaly-detection mode ==> values in ['good', 'bad']
        >> entry-validation mode ==> values in ['c', 'nc']
    :param labels_true: array of all GT labels;
        >> values in ['good', 'bad', 'nc']
    :param mode: working mode; values in
        >> {"anomaly-detection", "entry-validation"}
    :return: dictionary with classification metrics;
        it contains the following keys:
        >> 'tpr': true positive rate
        >> 'fpr': false positive rate
        >> 'tnr': true negative rate
        >> 'fnr': false negative rate
        >> 'bal_acc': balanced accuracy
    """

    assert mode in {'anomaly-detection', 'entry-validation'}, \
        f'mode must be "anomaly-detection" or "entry-validation",' \
        f'not "{mode}"'

    # for the anomaly detection task, "non-compliant" ("nc") samples
    # are considered "bad" samples
    if mode == 'anomaly-detection':
        good_class = 'good'
        bad_class = 'bad'
        labels_true[labels_true == 'nc'] = 'bad'
    # for the entry validation task, both "good" and "bad" samples
    # are considered "compliant" ("c")
    else:
        good_class = 'c'
        bad_class = 'nc'
        labels_true[labels_true == 'good'] = 'c'
        labels_true[labels_true == 'bad'] = 'c'

    # number of GT samples belonging to the good class
    n_good = (labels_true == good_class).sum()

    # number of GT samples belonging to the bad class
    n_bad = (labels_true == bad_class).sum()

    # boolean array of correct predictions, i.e.:
    # >> matches[i] is True <=> labels_pred[i] == labels_true[i]
    # >> matches[i] is False <=> labels_pred[i] != labels_true[i]
    matches = np.array(labels_pred == labels_true, dtype=bool)

    # boolean array of wrong predictions, i.e.
    # >> errors[i] is True <=> labels_pred[i] != labels_true[i]
    # >> errors[i] is False <=> labels_pred[i] == labels_true[i]
    errors = ~matches

    # compute true-positive-rate, i.e.:
    # number of samples correctly classified as bad/non-compliant
    tp = matches[labels_true == bad_class].sum()  # type: int
    tpr = float(tp / n_bad)

    # compute false-positive-rate, i.e.:
    # number of samples erroneously classified as anomalous/non-compliant
    fp = errors[labels_true == good_class].sum()  # type: int
    fpr = float(fp / n_good)

    rates_dict = {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': 1 - fpr,
        'fnr': 1 - tpr,
        'bal_acc': (tpr + (1 - fpr)) / 2
    }

    return rates_dict
