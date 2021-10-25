import matplotlib


matplotlib.use('TkAgg')
import numpy as np
from models.autoencoder import SimpleAutoencoder
from torch import nn
import torch
from torch.utils.data import DataLoader
from typing import Dict
from conf import Conf
from dataset.spal_fake_ds import SpalDS
from matplotlib import pyplot as plt
import utils
import piq
from typing import Tuple


class Evaluator(object):

    def __init__(self, cnf, model=None, test_loader=None):
        # type: (Conf, SimpleAutoencoder, DataLoader) -> None

        self.test_loader = test_loader
        self.model = model
        self.cnf = cnf

        if self.model is None:
            self.model = SimpleAutoencoder.init_from_pth(
                pth_file_path=(cnf.exp_log_path / 'best.pth'),
                device=cnf.device
            )

        if self.test_loader is None:
            test_set = SpalDS(cnf=self.cnf, mode='test')
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
            anomaly_score = nn.MSELoss()(y_pred, y_true).item()

        elif self.cnf.rec_error_fn == 'MS_SSIM_LOSS':
            anomaly_score = piq.MultiScaleSSIMLoss()(y_pred, y_true).item()

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
        # type: () -> Tuple[Dict[str, Dict[str, float]], torch.Tensor]

        # bild `score_dict`, that is a disctionary with 2 keys:
        # >> "good": list of anomaly scores -> one for each "good" sample
        # >> "bad": list of anomaly scores -> one for each "bad" sample
        score_dict = {'good': [], 'bad': []}
        for i, sample in enumerate(self.test_loader):
            labels = sample[-1]
            anomaly_scores = self.__get_scores_of_batch(sample)
            batch_size = anomaly_scores.shape[0]
            for i in range(batch_size):
                score_dict[labels[i]].append(anomaly_scores[i].item())

        # build `stats_dict`, that contains the statistics
        # relating to the distribution of the anomaly scores
        # of the "good" and "bad" test samples
        stats_dict = {}
        for key in ['good', 'bad']:
            mean = np.mean(score_dict[key])
            std = np.std(score_dict[key])
            min_value = np.min(score_dict[key])
            max_value = np.max(score_dict[key])
            q1 = np.quantile(score_dict[key], 0.25)
            q2 = np.quantile(score_dict[key], 0.50)
            q3 = np.quantile(score_dict[key], 0.75)
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr

            stats_dict[key] = {
                'rec_error_fn': self.cnf.rec_error_fn,
                'lower_whisker': lower_whisker,
                'upper_whisker': upper_whisker,
                'min': min_value, 'max': max_value,
                'mean': mean, 'std': std,
                'q1': q1, 'q2': q2, 'q3': q3,
            }

        fig = plt.figure()

        # draw dashed grey (#ecf0f1) lines for each quarile
        # (for both the "good" and the "bad" boxplot)
        for k in ['good', 'bad']:
            for q in ['q1', 'q2', 'q3']:
                x = stats_dict[k][q]
                plt.hlines(x, 0, 3, linestyles='--', colors='#ecf0f1')

        # draw a solid red line for the anomaly threshold,
        # i.e. the upper whisker of the "good" boxplot
        anomaly_th = stats_dict['good']['upper_whisker']
        plt.hlines(anomaly_th, 0, 3, colors='#1abc9c')

        # draw the actual boxplot with the 2 boxes: "good" and "bad"
        plt.boxplot(
            [score_dict['good'], score_dict['bad']], labels=['good', 'bad'],
            showfliers=True, medianprops={'color': '#e74c3c'}
        )
        # plt.ylim(0, 0.035)

        boxplot = utils.pyplot_to_tensor(fig)
        plt.close(fig)
        return stats_dict, boxplot


    def get_accuracy(self, stats_dict=None):

        if stats_dict is None:
            stats_dict, _ = self.get_stats()
        self.model.stats = stats_dict

        anomaly_th = stats_dict['good']['upper_whisker']

        score = {'all': 0, 'good': 0, 'bad': 0}
        counter = {'all': 0, 'good': 0, 'bad': 0}
        for i, sample in enumerate(self.test_loader):
            labels_true = sample[-1]
            anomaly_scores = self.__get_scores_of_batch(sample)

            batch_size = anomaly_scores.shape[0]
            for i in range(batch_size):
                label_true = labels_true[i]
                anomaly_score = anomaly_scores[i].item()
                label_pred = 'good' if anomaly_score < anomaly_th else 'bad'

                counter['all'] += 1
                counter[label_true] += 1
                if label_pred == label_true:
                    score['all'] += 1
                    score[label_true] += 1

        acc_dict = {
            'all': score['all'] / counter['all'],
            'good': score['good'] / counter['good'],
            'bad': score['bad'] / counter['bad']
        }
        return acc_dict


def main(exp_name):
    import torchvision

    cnf = Conf(exp_name=exp_name)

    evaluator = Evaluator(cnf=cnf)
    stats_dict, boxplot = evaluator.get_stats()
    accuracy_dict = evaluator.get_accuracy(stats_dict=stats_dict)

    out_path = cnf.exp_log_path / 'boxplot.png'
    torchvision.utils.save_image(boxplot, out_path)

    print(f'EXP: `{exp_name}`')
    print(f'------------------------------')
    print(f'>> accuracy (good): {100 * accuracy_dict["good"]:.2f}%')
    print(f'>> accuracy (bad) : {100 * accuracy_dict["bad"]:.2f}%')
    print(f'------------------------------')
    print(f'>> accuracy (all) : {100 * accuracy_dict["all"]:.2f}%')


if __name__ == '__main__':
    main(exp_name='p1_rect_nonoise')
