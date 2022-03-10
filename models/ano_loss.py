# -*- coding: utf-8 -*-
# ---------------------

import piq
import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class AnoLoss(_Loss):

    def __init__(self, l1_w, ms_ssim_w):
        # type: (float, float) -> None
        super().__init__()

        self.l1 = nn.L1Loss()
        self.ms_ssim = piq.MultiScaleSSIMLoss(kernel_size=7)

        self.w1 = l1_w
        self.w2 = ms_ssim_w


    def forward(self, y_pred, y_true):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        return self.w1 * self.l1(y_pred, y_true) + \
               self.w2 * self.ms_ssim(y_pred, y_true)
