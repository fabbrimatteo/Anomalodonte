# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch.nn.modules.loss import _Loss


T = torch.Tensor


class CodeDiffLoss(_Loss):

    def __init__(self, batch_size, margin=0.2):
        # type: (int, float) -> None
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.relu = torch.nn.ReLU()


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
        return torch.abs((10 * x_mse2) - c_mse).mean()


    def forward2(self, x_batch, code_batch):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor

        p = list(range(self.batch_size))
        p.reverse()
        p = torch.LongTensor(p)

        x1 = x_batch
        x2 = x_batch[p]

        code_1 = code_batch
        code_2 = code_batch[p]

        x_diff = self.rec_loss(x1, x2)
        x_diff2 = x_diff ** 2
        c_mse = ((code_1 - code_2) ** 2).mean([1, 2, 3])
        return torch.abs((0.1 * x_diff2) - c_mse).mean()


    def forward3(self, code_batch):
        # type: (torch.Tensor) -> torch.Tensor

        p = list(range(self.batch_size))
        p.reverse()
        p = torch.LongTensor(p)

        code_1 = code_batch
        code_2 = code_batch[p]

        c_mse = self.mse_loss(code_1, code_2)
        return self.relu(self.margin - c_mse).mean()


if __name__ == '__main__':
    l = CodeDiffLoss(batch_size=7)
    x = torch.rand((7, 3, 256, 256))
    c = torch.rand((7, 3, 4, 4))
    print(l.forward3(x, c))
