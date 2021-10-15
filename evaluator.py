import matplotlib


matplotlib.use('TkAgg')
import numpy as np
from models.autoencoder import SimpleAutoencoder
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict
from conf import Conf
from dataset.spal_fake_ds import SpalDS
from matplotlib import pyplot as plt
import utils
import piq
import torchvision


class Evaluator(object):

    def __init__(self, cnf, model=None, test_loader=None):
        # type: (Conf, SimpleAutoencoder, DataLoader) -> None

        self.test_loader = test_loader
        self.model = model
        self.cnf = cnf

        if self.model is None:
            pth_file_path = cnf.exp_log_path / 'best.pth'
            self.model = SimpleAutoencoder.init_from_pth(pth_file_path=pth_file_path, device=cnf.device)

        if self.test_loader is None:
            test_set = SpalDS(cnf=self.cnf, mode='test')
            self.test_loader = DataLoader(
                dataset=test_set, batch_size=1, num_workers=0,
                worker_init_fn=test_set.wif_test, shuffle=False, pin_memory=False,
            )

        self.loss_fn = nn.MSELoss()


    def __get_anomaly_score(self, code_pred, code_true, y_pred, y_true):
        anomaly_score = 0
        if self.cnf.rec_error_fn == 'CODE_MSE_LOSS':
            anomaly_score = nn.MSELoss()(code_pred, code_true).item()
        elif self.cnf.rec_error_fn == 'MSE_LOSS':
            anomaly_score = nn.MSELoss()(y_pred, y_true).item()
        elif self.cnf.rec_error_fn == 'MS_SSIM_LOSS':
            anomaly_score = piq.MultiScaleSSIMLoss()(y_pred, y_true).item()
        return anomaly_score


    def get_stats(self):
        # type: () -> Dict[str, Dict[str, float]]

        score_dict = {}
        stats_dict = {}
        for i, sample in enumerate(self.test_loader):
            x, y_true, label = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            label = label[0]

            code_true = self.model.encode(x)
            y_pred = self.model.decode(code_true)
            code_pred = self.model.encode(y_pred)

            anomaly_score = self.__get_anomaly_score(code_pred, code_true, y_pred, y_true)

            if not label in score_dict:
                score_dict[label] = []
            score_dict[label].append(anomaly_score)

        for key in score_dict:
            q1 = np.quantile(score_dict[key], 0.25)
            q3 = np.quantile(score_dict[key], 0.75)
            stats_dict[key] = {
                'rec_error_fn': self.cnf.rec_error_fn,
                'mean': np.mean(score_dict[key]),
                'std': np.std(score_dict[key]),
                'min': np.min(score_dict[key]),
                'max': np.max(score_dict[key]),
                'q0.05': np.quantile(score_dict[key], 0.05),
                'q0.25': q1,
                'q0.50': np.quantile(score_dict[key], 0.50),
                'q0.75': q3,
                'q0.95': np.quantile(score_dict[key], 0.95),
                'w0': q1 - 1.5 * (q3 - q1),
                'w1': q3 + 1.5 * (q3 - q1),
                'wth2': q3 + 2 * (q3 - q1),
            }

        fig = plt.figure()
        plt.hlines(np.quantile(score_dict['good'], 0.50), 0, 3, linestyles='--', colors='#e74c3c')
        plt.hlines(np.quantile(score_dict['good'], 0.25), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(score_dict['good'], 0.75), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(score_dict['bad'], 0.25), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(np.quantile(score_dict['bad'], 0.75), 0, 3, linestyles='--', colors='#ecf0f1')
        plt.hlines(stats_dict['good']['wth2'], 0, 3, colors='#1abc9c')
        plt.boxplot(
            [score_dict['good'], score_dict['bad']], labels=['good', 'bad'],
            showfliers=True, medianprops={'color': '#e74c3c'}
        )
        plt.ylim(0, 0.035)

        boxplot = utils.pyplot_to_tensor(fig)
        plt.close(fig)
        return stats_dict, boxplot


    def get_accuracy(self, stats_dict=None):

        if stats_dict is None:
            stats_dict, _ = self.get_stats()

        anomaly_th = stats_dict['good']['wth2']

        score = 0
        for i, sample in enumerate(self.test_loader):
            x, y_true, label_true = sample

            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            label_true = label_true[0]

            code_true = self.model.encode(x)
            y_pred = self.model.decode(code_true)
            code_pred = self.model.encode(y_pred)
            anomaly_score = self.__get_anomaly_score(code_pred, code_true, y_pred, y_true)

            label_pred = 'good' if anomaly_score < anomaly_th else 'bad'
            if label_pred == label_true:
                score += 1
            print(f'P: {label_pred}, T: {label_true}, A: {self.model.get_code_anomaly_perc(x)*100}%')
            # else:
            #     torchvision.utils.save_image(y_true, f'error_gt_{label_pred}.{i}.png')

        accuracy = score / len(self.test_loader)
        return accuracy


if __name__ == '__main__':
    eval = Evaluator(cnf=Conf(exp_name='magalli'))
    a = eval.get_accuracy()
    print(a)
