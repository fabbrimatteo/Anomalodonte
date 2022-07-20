# -*- coding: utf-8 -*-
# ---------------------

from time import time

import numpy as np

# this MUST be imported before `import torch`
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision as tv
from torch import optim
from torch.utils.data import DataLoader

from conf import Conf
from dataset.spal_ds import SpalDS
from eval.lof import Loffer
from models import AutoencoderPlus
from models.inter_loss import interpol_loss
from models.rec_loss import RecLoss
from progress_bar import ProgressBar
from visual_utils import draw_anomaly_ui
from post_processing import tensor2img


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf
        self.res_out_path = self.cnf.proj_log_path / self.cnf.exp_name / 'res'

        # init train loader
        training_set = SpalDS(self.cnf, mode='train')
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=self.cnf.batch_size, num_workers=4,
            worker_init_fn=training_set.wif, shuffle=True, pin_memory=True,
            drop_last=True,
        )

        # init test loader
        test_set = SpalDS(self.cnf, mode='test')
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=self.cnf.batch_size // 4, num_workers=1,
            worker_init_fn=test_set.wif_test, shuffle=False, pin_memory=False,
        )

        # init model
        if self.cnf.pretrained_weights_path:
            self.model = AutoencoderPlus.init_from_pth(
                self.cnf.pretrained_weights_path, mode='train', device=cnf.device)
        else:
            self.model = AutoencoderPlus(
                code_channels=self.cnf.code_channels,
                code_h=self.cnf.code_h, code_w=self.cnf.code_w
            )
            self.model = self.model.to(self.cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.cnf.lr)

        # init logging stuffs
        self.log_path = self.cnf.exp_log_path
        print(f'tensorboard --logdir={self.cnf.proj_log_path.abspath()}\n')
        self.log_path.makedirs_p()
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

        if self.cnf.loss_fn == 'L1+MS_SSIM':
            self.rec_loss_fn = RecLoss(l1_w=10, ms_ssim_w=3)
        elif self.cnf.loss_fn == 'MSE':
            self.rec_loss_fn = lambda x, y: 100 * torch.nn.MSELoss()(x, y)
        else:
            raise ValueError(
                f'unsupported loss function '
                f'"{self.rec_loss_fn}"'
            )


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
        int_losses = []
        rec_losses = []
        for step, sample in enumerate(self.train_loader):
            t = time()

            self.optimizer.zero_grad()

            x, y_true, _ = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

            code_pred = self.model.encode(x)
            y_pred = self.model.decode(code_pred)

            # interpolation loss
            if self.cnf.int_loss_w > 0:
                int_loss = interpol_loss(model=self.model, x=x)
            else:
                int_loss = torch.tensor([0]).to(self.cnf.device)
            int_losses.append(int_loss.item())

            # reconstruction loss
            rec_loss = self.rec_loss_fn(y_pred, y_true)
            rec_losses.append(rec_loss.item())

            # global loss: weighted sum of reconstruction
            # and interpolation losses
            rec_loss = self.cnf.rec_loss_w * rec_loss
            int_loss = self.cnf.int_loss_w * int_loss
            loss = rec_loss + int_loss
            train_losses.append(loss.item())

            loss.backward()
            self.optimizer.step(None)

            # print progress bar
            times.append(time() - t)
            done = (not self.cnf.log_each_step) \
                   and self.progress_bar.progress == 1
            if self.cnf.log_each_step or done:
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):6.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        self.sw.add_scalar(
            tag='rec_loss', global_step=self.epoch,
            scalar_value=np.mean(rec_losses)
        )
        self.sw.add_scalar(
            tag='int_loss', global_step=self.epoch,
            scalar_value=np.mean(int_losses)
        )
        self.sw.add_scalar(
            tag='train_loss', global_step=self.epoch,
            scalar_value=np.mean(train_losses)
        )

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()

        t = time()

        train_dir = self.cnf.ds_path / 'train'
        test_dir = self.cnf.ds_path / 'test'
        if self.cnf.cam_id is not None:
            train_dir = train_dir / self.cnf.cam_id
            test_dir = test_dir / self.cnf.cam_id

        loffer = Loffer(
            train_dir=train_dir, model=self.model,
            n_neighbors=20,
        )
        ad_rates, top16_errs = loffer.evaluate(test_dir=test_dir,
                                               out_dir=self.res_out_path)
        lof_ba = ad_rates['bal_acc']
        accuracy = lof_ba

        test_losses = []
        for step, sample in enumerate(self.test_loader):
            x, y_true, _ = sample
            x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
            code = self.model.encode(x)
            y_pred = self.model.decode(code)

            loss = self.rec_loss_fn(y_pred, y_true)
            test_losses.append(loss.item())

            # draw results for this step in a 2 rows grid:
            # ->> row #1: predicted code (same size as `y_pred`)
            # ->> row #2: predicted output (y_pred)
            # ->> row #3: target (y_true)
            bs = x.shape[0] // 2
            if step % 2 == 0 and bs > 0:
                y_pred = y_pred[:bs, ...]
                y_true = y_true[:bs, ...]
                code = code[:bs, ...]
                code = (0.5 * (code + 1))
                code = torch.nn.Upsample(size=(256, 256))(code)
                grid = torch.cat([code, y_pred, y_true], dim=0)
                grid = tv.utils.make_grid(
                    grid, normalize=True,
                    value_range=(0, 1), nrow=bs
                )
                self.sw.add_image(
                    tag=f'results_{step}', img_tensor=grid,
                    global_step=self.epoch
                )

        err_img = np.array([e[1] for e in top16_errs])
        err_img = torch.tensor(err_img.transpose((0, 3, 1, 2))) / 255.
        err_img = tv.utils.make_grid(
            err_img, normalize=True,
            value_range=(0, 1), nrow=2,
        )
        self.sw.add_image(
            tag=f'top16_errors', img_tensor=err_img,
            global_step=self.epoch
        )

        # save best model
        if self.best_test_acc is None or accuracy > self.best_test_acc:
            self.best_test_acc = accuracy
            self.patience = self.cnf.max_patience
            self.model.save_w(
                self.log_path / 'best.pth',
                cnf_dict=self.cnf.dict_view,
            )
        else:
            self.patience = self.patience - 1

        # save last model
        self.model.save_w(
            self.log_path / 'last.pth',
            cnf_dict=self.cnf.dict_view,
        )

        # log test results
        print(f'\t● Bal. Acc: {100 * accuracy:.2f}%'
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
            tag='patience', scalar_value=self.patience,
            global_step=self.epoch
        )


    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """

        while True:
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()

            stop_condition_1 = self.epoch >= self.cnf.epochs
            stop_condition_2 = self.patience <= 0
            if stop_condition_1 or stop_condition_2:
                break

        print('\n--------')
        if stop_condition_1:
            print(f'[■] Done! -> max epoch has been reached.')
        elif stop_condition_2:
            print(f'[■] Done! -> `patience` reached 0.')
