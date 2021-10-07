# -*- coding: utf-8 -*-
# ---------------------

from time import time

import numpy as np
import torch
import torchvision as tv
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import Conf
from dataset.spal_fake_ds import SpalDS
from models.autoencoder import SimpleAutoencoder
from models.dd_loss import DDLoss
from progress_bar import ProgressBar


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf

        # init train loader
        training_set = SpalDS(cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size, num_workers=cnf.n_workers,
            worker_init_fn=training_set.wif, shuffle=True, pin_memory=True,
        )

        # init test loader
        test_set = SpalDS(cnf, mode='test')
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=cnf.batch_size, num_workers=1,
            worker_init_fn=test_set.wif_test, shuffle=False, pin_memory=False,
        )

        # init model
        self.model = SimpleAutoencoder(code_channels=cnf.code_channels)
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
        self.best_test_loss = None
        self.patience = self.cnf.max_patience

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        self.loss_fn = DDLoss(mse_w=10, ssim_w=3, vgg_w=0.00001, device=cnf.device)


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
            self.best_test_loss = self.best_test_loss


    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_loss
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
        for step, sample in enumerate(self.train_loader):
            t = time()

            self.optimizer.zero_grad()

            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            y_pred = self.model.forward(x)
            loss = self.loss_fn(y_pred, y_true)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()

        t = time()
        test_losses = []
        for step, sample in enumerate(self.test_loader):
            x, y_true = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            y_pred = self.model.forward(x)

            loss = self.loss_fn(y_pred, y_true)
            test_losses.append(loss.item())

            # draw results for this step in a 2 rows grid:
            # row #1: predicted_output (y_pred)
            # row #2: target (y_true)
            if step % 16 == 0:
                grid = torch.cat([y_pred, y_true], dim=0)
                grid = tv.utils.make_grid(grid, normalize=True, value_range=(0, 1), nrow=x.shape[0])
                self.sw.add_image(tag=f'results_{step}', img_tensor=grid, global_step=self.epoch)

        # save best model
        mean_test_loss = np.mean(test_losses)
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            self.patience = self.cnf.max_patience
            self.model.save_w(self.log_path / 'best.pth')
        else:
            self.patience = self.patience - 1

        # log test results
        print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f}'
              f' │ patience: {self.patience}'
              f' │ T: {time() - t:.2f} s')
        self.sw.add_scalar(tag='test_loss', scalar_value=mean_test_loss, global_step=self.epoch)
        self.sw.add_scalar(tag='patience', scalar_value=self.patience, global_step=self.epoch)

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
