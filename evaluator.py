import matplotlib


matplotlib.use('TkAgg')
import numpy as np
from models.autoencoder import SimpleAutoencoder
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict
from conf import Conf
from dataset.spal_fake_ds import SpalDS


class Evaluator(object):

    def __init__(self, cnf=None, model=None, test_loader=None):
        # type: (Conf, SimpleAutoencoder, DataLoader) -> None

        assert (model is not None) or (cnf is not None), \
            '[!] `model` and `cnf` can not be both `None`; ' \
            'please specify a value for (at least) one of them'

        assert (test_loader is not None) or (cnf is not None), \
            '[!] `test_loader` and `cnf` can not be both `None`; ' \
            'please specify a value for (at least) one of them'

        self.test_loader = test_loader
        self.model = model
        self.cnf = cnf

        if self.model is None:
            pth_file_path = cnf.exp_log_path / 'best.pth'
            self.model = SimpleAutoencoder.init_from_pth(pth_file_path=pth_file_path, device=cnf.device)

        if self.test_loader is None:
            test_set = SpalDS(cnf=self.cnf, mode='test')
            self.test_loader = DataLoader(
                dataset=test_set, batch_size=1, num_workers=1,
                worker_init_fn=test_set.wif_test, shuffle=False, pin_memory=False,
            )

        self.loss_fn = nn.MSELoss()


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

            anomaly_score = self.loss_fn(y_pred, y_true).item()
            if not label in score_dict:
                score_dict[label] = []
            score_dict[label].append(anomaly_score)

        for key in score_dict:
            q1 = np.quantile(score_dict[key], 0.25)
            q3 = np.quantile(score_dict[key], 0.75)
            stats_dict[key] = {
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

        return stats_dict


    def get_accuracy(self):
        stats_dict = self.get_stats()

        anomaly_th = stats_dict['good']['wth2']

        score = 0
        for i, sample in enumerate(self.test_loader):
            x, y_true, label_true = sample

            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            label_true = label_true[0]

            code_true = self.model.encode(x)
            y_pred = self.model.decode(code_true)
            code_pred = self.model.encode(y_pred)
            anomaly_level = self.loss_fn(y_pred, y_true).item()

            label_pred = 'good' if anomaly_level < anomaly_th else 'bad'
            if label_pred == label_true:
                score += 1

        accuracy = score / len(self.test_loader)
        return accuracy


if __name__ == '__main__':
    eval =Evaluator(cnf=Conf(exp_name='ultra_small_less_noise'))
    a = eval.get_accuracy()
    print(a)
