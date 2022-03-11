# -*- coding: utf-8 -*-
# ---------------------

from time import time

import numpy as np
import torch
import torchvision as tv
from torch import optim
from torch.utils.data import DataLoader
# this MUST be imported before `import torch`
from torch.utils.tensorboard import SummaryWriter

import boxplot_utils
import roc_utils
from conf import Conf
from dataset.spal_ds import SpalDS
from evaluator import Evaluator
from models.ano_loss import AnoLoss
from models.autoencoder import SimpleAutoencoder
from models.dd_loss import DDLoss
from progress_bar import ProgressBar
from regularization import interpol_loss


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        training_set = SpalDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size, num_workers=4,
            worker_init_fn=training_set.wif, shuffle=True, pin_memory=True,
            drop_last=True,
        )

        # init test loader
        test_set = SpalDS(cnf, mode='test')
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=cnf.batch_size, num_workers=1,
            worker_init_fn=test_set.wif_test, shuffle=False, pin_memory=False,
        )

        # init model
        self.model = SimpleAutoencoder(
            code_channels=cnf.code_channels,
            code_h=cnf.code_h, code_w=cnf.code_w
        )
        self.model = self.model.to(cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values
        self.epoch = 0
        self.best_test_acc = None
        self.patience = self.cnf.max_patience

        # init progress bar
        self.progress_bar = ProgressBar(
            max_step=self.log_freq, max_epoch=self.cnf.epochs
        )

        # possibly load checkpoint
        self.load_ck()

        if self.cnf.loss_fn == 'L1+MS_SSIM+VGG':
            self.loss_fn = DDLoss(
                mse_w=10, ssim_w=3, vgg_w=0.00001,
                device=cnf.device
            )
        elif self.cnf.loss_fn == 'L1+MS_SSIM':
            self.loss_fn = AnoLoss(l1_w=10, ms_ssim_w=3)
        else:
            self.loss_fn = lambda x, y: 100 * torch.nn.MSELoss()(x, y)


    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.patience = ck['patience']
            self.best_test_acc = ck['best_test_accuracy']


    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_accuracy': self.best_test_acc,
            'patience': self.patience
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        start_time = time()
        self.model.train()

        times = []
        train_losses = []
        cc_losses = []
        int_losses = []
        for step, sample in enumerate(self.train_loader):
            t = time()

            self.optimizer.zero_grad()

            x, y_true, _ = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            code_true = self.model.encode(y_true, self.cnf.code_noise)
            y_pred = self.model.decode(code_true)
            code_pred = self.model.encode(y_pred, self.cnf.code_noise)

            # code commitment loss
            cc_loss = 10 * torch.nn.MSELoss()(code_pred, code_true)
            cc_losses.append(cc_loss.item())

            # interpolation loss
            int_loss = 100 * interpol_loss(model=self.model, x=x)
            int_losses.append(int_loss.item())

            loss = self.loss_fn(y_pred, y_true) + cc_loss + int_loss

            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            times.append(time() - t)
            done = (not self.cnf.log_each_step) \
                   and self.progress_bar.progress == 1
            if self.cnf.log_each_step or done:
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(train_losses):.5f} '
                      f'│ (reg: {np.mean(int_losses):.5f}) '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_loss = np.mean(train_losses)
        self.sw.add_scalar(
            tag='train_loss', global_step=self.epoch,
            scalar_value=mean_loss
        )

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()

        t = time()
        evaluator = Evaluator(
            model=self.model, cnf=self.cnf,
            test_loader=self.test_loader
        )
        scores_dict, sol_dict, boxplot_dict, roc_dict = evaluator.get_stats()

        boxplot = boxplot_utils.plt_boxplot(boxplot_dict)
        rocplot = roc_utils.plt_rocplot(roc_dict)

        # --- TENSORBOARD: boxplot
        self.sw.add_image(
            tag=f'boxplot', img_tensor=boxplot,
            global_step=self.epoch, dataformats='HWC'
        )

        # --- TENSORBOARD: rocplot
        self.sw.add_image(
            tag=f'rocplot', img_tensor=rocplot,
            global_step=self.epoch, dataformats='HWC'
        )

        accuracy = sol_dict['bal_acc']
        auroc = roc_dict['auroc']

        test_losses = []
        for step, sample in enumerate(self.test_loader):
            x, y_true, _ = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            y_pred = self.model.forward(x)

            loss = self.loss_fn(y_pred, y_true)
            test_losses.append(loss.item())

            # draw results for this step in a 2 rows grid:
            # row #1: predicted_output (y_pred)
            # row #2: target (y_true)
            if step % 8 == 0:
                grid = torch.cat([y_pred, y_true], dim=0)
                grid = tv.utils.make_grid(
                    grid, normalize=True,
                    value_range=(0, 1), nrow=x.shape[0]
                )
                self.sw.add_image(
                    tag=f'results_{step}', img_tensor=grid,
                    global_step=self.epoch
                )

        # save best model
        if self.best_test_acc is None or accuracy > self.best_test_acc:
            self.best_test_acc = accuracy
            self.patience = self.cnf.max_patience
            anomaly_th = boxplot_dict['good']['upper_whisker']
            self.model.save_w(
                self.log_path / 'best.pth',
                cnf_dict=self.cnf.dict_view,
                anomaly_th=anomaly_th
            )
            torch.save(None, self.log_path / 'anomaly_th.pth')
        else:
            self.patience = self.patience - 1

        # log test results
        print(f'\t● Bal. Acc: {100 * accuracy:.2f}%'
              f' │ AUROC: {100 * auroc:.2f}%'
              f' │ patience: {self.patience}'
              f' │ T: {time() - t:.2f} s')

        self.sw.add_scalar(
            tag='test_loss', scalar_value=np.mean(test_losses),
            global_step=self.epoch
        )

        self.sw.add_scalar(
            tag='accuracy', scalar_value=100 * accuracy,
            global_step=self.epoch
        )

        self.sw.add_scalar(
            tag='auroc', scalar_value=100 * auroc,
            global_step=self.epoch
        )

        self.sw.add_scalar(
            tag='patience', scalar_value=self.patience,
            global_step=self.epoch
        )

        if self.patience == 0:
            print('\n--------')
            print(f'[■] Done! -> `patience` reached 0.')
            exit(0)


    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()
