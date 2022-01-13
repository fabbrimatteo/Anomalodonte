import matplotlib


matplotlib.use('TkAgg')
import numpy as np
from models.autoencoder import SimpleAutoencoder
from torch.utils.data import DataLoader
from typing import Dict
from conf import Conf
from dataset.spal_fake_ds import SpalDS
from typing import Tuple
import roc_utils
import boxplot_utils as bp_utils


class Evaluator(object):

    def __init__(self, cnf, model=None, test_loader=None, mode='test'):
        # type: (Conf, SimpleAutoencoder, DataLoader, str) -> None

        self.test_loader = test_loader
        self.model = model
        self.cnf = cnf

        if self.model is None:
            self.model = SimpleAutoencoder.init_from_pth(
                pth_file_path=(cnf.exp_log_path / 'best.pth'),
                device=cnf.device
            )

        if self.test_loader is None:
            test_set = SpalDS(cnf=self.cnf, mode=mode)
            self.test_loader = DataLoader(
                dataset=test_set, batch_size=8, num_workers=0,
                worker_init_fn=test_set.wif_test, shuffle=False,
                pin_memory=False,
            )


    def get_test_scores(self):
        # type: () -> Tuple[np.ndarray, np.ndarray]
        """
        :return: (all_scores, labels)
            >> `all_scores` -> array of anomaly scores; one element
                for each sample of the test set
            >> `labels` -> array of GT labels; one label for each sample
                of the test set; labels[i] is the label of the i-th
                sample of the test set.
        """
        all_scores = []
        labels_true = []
        for i, sample in enumerate(self.test_loader):
            x, _, labels = sample

            anomaly_scores = self.model.get_anomaly_score(
                x, score_function=self.cnf.score_fn
            )

            batch_size = anomaly_scores.shape[0]
            for i in range(batch_size):
                all_scores.append(anomaly_scores[i].item())
                labels_true.append(labels[i])

        all_scores = np.array(all_scores)
        labels_true = np.array(labels_true)

        return all_scores, labels_true


    def get_stats(self):
        # type: () -> Tuple[Dict[str, ...], Dict[str, ...], Dict[str, ...], Dict[str, ...]]
        """
        :return: (boxplot_dict, boxplot) ->
        # >> boxplot_dict: a dictionary that contains the statistics
        #    relating to the distribution of the anomaly scores
        #    of the "good" and "bad" test samples
        # >> boxplot: boxplot with 2 boxes representing
        #    the distribution of the anomaly scores
        #    of the "good" and "bad" test samples
        """

        all_scores, labels_true = self.get_test_scores()

        # build `boxplot_dict`, that contains the statistics
        # relating to the distribution of the anomaly scores
        # of the "good" and "bad" test samples
        boxplot_dict = bp_utils.get_boxplot_dict(all_scores, labels_true)

        # build `roc_dict`, that contains the TPR and FPR lists
        # of the model with one element for each possible threshold
        # on the anomaly score; it also contains the AUROC
        roc_dict = roc_utils.get_roc_dict(all_scores, labels_true)
        scores_dict = {'scores': all_scores, 'labels_true': labels_true}

        # build `sol_dict`, that contains the statistics
        # relating to the chosen solution
        anomaly_th = bp_utils.get_anomaly_th(boxplot_dict)
        sol_dict = roc_utils.get_ad_rates(all_scores, anomaly_th, labels_true)

        return scores_dict, sol_dict, boxplot_dict, roc_dict


def main(exp_name, mode):
    import cv2

    cnf = Conf(exp_name=exp_name)

    evaluator = Evaluator(cnf=cnf, mode=mode)
    scores_dict, sol_dict, boxplot_dict, roc_dict = evaluator.get_stats()

    # "test" mode -> anomaly detection test (prefix: "ad")
    # "ev-test" mode -> entry validation test (prefix: "ev")
    assert mode in ['test', 'ev-test']
    pref = 'ad' if mode == 'test' else 'ev'

    boxplot = bp_utils.plt_boxplot(boxplot_dict)
    out_path = cnf.exp_log_path / f'{pref}_boxplot.png'
    anomaly_th = bp_utils.get_anomaly_th(boxplot_dict)
    cv2.imwrite(out_path, boxplot[:, :, ::-1])

    rocplot = roc_utils.plt_rocplot(roc_dict)
    out_path = cnf.exp_log_path / f'{pref}_rocplot.png'
    cv2.imwrite(out_path, rocplot[:, :, ::-1])

    print(f'\nEXP: `{exp_name}` ({pref.upper()})')
    print(f'anomaly threshold: {anomaly_th:.4f} with `{cnf.score_fn}`')
    print(f'------------------------------')
    print(f'>> TPR..: {sol_dict["tpr"] * 100:.2f}% (acc bad)')
    print(f'>> TNR..: {sol_dict["tnr"] * 100:.2f}% (acc good)')
    print(f'>> BA...: {sol_dict["bal_acc"] * 100:.2f}%')
    print(f'>> auroc: {100 * roc_dict["auroc"]:.2f}%')
    print(f'------------------------------')


if __name__ == '__main__':
    main(exp_name='exp-02', mode='test')
