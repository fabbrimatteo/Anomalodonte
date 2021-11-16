import matplotlib


matplotlib.use('TkAgg')
import numpy as np
from models.autoencoder5 import SimpleAutoencoder
from torch import nn
import torch
from torch.utils.data import DataLoader
from typing import Dict
from conf import Conf
from dataset.spal_fake_ds import SpalDS
import piq
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


    def __get_anomaly_score(self, code_pred, code_true, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Returns the anomaly score based on the error function
        specified in the configuration file.

        The error function can be:
        >> 'MSE_LOSS': MSE between target `x` and reconstruction `d(e(x))`
        >> 'MS_SSIM_LOSS':  multi-scale structural similarity loss
            between target `x` and reconstruction `d(e(x))`
        >> 'CODE_MSE_LOSS': MSE between target code `e(x)` and
            reconstruction code`e(d(e(x)))`

        :param code_pred: reconstruction code -> e(d(e(x)))
        :param code_true: target code -> e(x)
        :param y_pred: reconstructed image -> d(e(x))
        :param y_true: input image / target image -> x
        :return: anomaly score -> torch.Tensor with shape (batch_size,)
        """
        anomaly_score = 0

        if self.cnf.rec_error_fn == 'CODE_MSE_LOSS':
            x = nn.MSELoss(reduction='none')(code_pred, code_true)
            anomaly_score = x.mean((1, 2, 3))
            # anomaly_score = nn.MSELoss()(code_pred, code_true).item()

        elif self.cnf.rec_error_fn == 'MSE_LOSS':
            x = nn.MSELoss(reduction='none')(y_pred, y_true)
            anomaly_score = x.mean((1, 2, 3))

        elif self.cnf.rec_error_fn == 'MS_SSIM_LOSS':
            anomaly_score = piq.MultiScaleSSIMLoss(reduction='none')(y_pred, y_true)

        return anomaly_score


    def __get_scores_of_batch(self, sample_batch):
        x, y_true, label = sample_batch
        x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

        code_true = self.model.encode(x)
        y_pred = self.model.decode(code_true)
        code_pred = self.model.encode(y_pred)

        return self.__get_anomaly_score(
            code_pred, code_true, y_pred, y_true
        )


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

        # bild `score_dict`, that is a disctionary with 2 keys:
        # >> "good": list of anomaly scores -> one for each "good" sample
        # >> "bad": list of anomaly scores -> one for each "bad" sample
        # >> "all": list of anomaly scores -> one for each sample
        # >> "labels_true": list of GT labels -> one for each sample

        scores = []
        labels_true = []
        for i, sample in enumerate(self.test_loader):
            labels = sample[-1]
            anomaly_scores = self.__get_scores_of_batch(sample)
            batch_size = anomaly_scores.shape[0]
            for i in range(batch_size):
                scores.append(anomaly_scores[i].item())
                labels_true.append(labels[i])
        scores = np.array(scores)
        labels_true = np.array(labels_true)

        # build `boxplot_dict`, that contains the statistics
        # relating to the distribution of the anomaly scores
        # of the "good" and "bad" test samples
        boxplot_dict = bp_utils.get_boxplot_dict(scores, labels_true)
        roc_dict = roc_utils.get_roc_dict(scores, labels_true)
        scores_dict = {'scores': scores, 'labels_true': labels_true}

        anomaly_th = boxplot_dict['good']['upper_whisker']
        sol_dict = roc_utils.get_ad_rates(scores, anomaly_th, labels_true)
        # for i in range(len(scores)):
        #     print(f'{labels_true[i]}{i:03d} -> {max(0, 100*(scores[i]/anomaly_th) - 50):.0f}%')

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
    cv2.imwrite(out_path, boxplot[:, :, ::-1])

    rocplot = roc_utils.plt_rocplot(roc_dict)
    out_path = cnf.exp_log_path / f'{pref}_rocplot.png'
    cv2.imwrite(out_path, rocplot[:, :, ::-1])

    print(f'\nEXP: `{exp_name}` ({pref.upper()})')
    print(f'------------------------------')
    print(f'>> TPR..: {sol_dict["tpr"] * 100:.2f}% (acc bad)')
    print(f'>> TNR..: {sol_dict["tnr"] * 100:.2f}% (acc good)')
    print(f'>> BA...: {sol_dict["bal_acc"] * 100:.2f}%')
    print(f'>> auroc: {100 * roc_dict["auroc"]:.2f}%')
    print(f'------------------------------')


if __name__ == '__main__':
    main(exp_name='a5_noise', mode='ev-test')
    main(exp_name='a5_noise', mode='test')
