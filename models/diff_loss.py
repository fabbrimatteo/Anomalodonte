# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch.nn.modules.loss import _Loss


T = torch.Tensor


class CodeDiffLoss(_Loss):

    def __init__(self, batch_size):
        # type: (int) -> None
        super().__init__()
        self.batch_size = batch_size


    def forward(self, x_batch, code_batch):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor

        p = list(range(self.batch_size))
        p.reverse()
        p = torch.LongTensor(p)

        x1 = x_batch
        x2 = x_batch[p]

        code_1 = code_batch
        code_2 = code_batch[p]

        x_mse = ((x1 - x2) ** 2).mean([1, 2, 3])
        x_mse2 = x_mse ** 2
        c_mse = ((code_1 - code_2) ** 2).mean([1, 2, 3])
        return torch.abs((100 * x_mse2) - c_mse).mean()


    def f2(self, code_batch, eps=0.001):
        # type: (torch.Tensor, float) -> torch.Tensor
        return 1 / (code_batch.std(0).mean() + eps)


if __name__ == '__main__':
    l = CodeDiffLoss(batch_size=7)
    x = torch.rand((7, 3, 256, 256))
    c = torch.rand((7, 3, 4, 4))
    print(l.f2(c))
