from typing import Optional

import numpy as np
import piq
import torch
import torch.nn as nn

import pre_processing
from models.autoencoder_core import AutoencoderCore


class AutoencoderPlus(AutoencoderCore):

    def __init__(self, mid_channels=128, n_res_layers=2, code_channels=3, code_h=4, code_w=4):
        # type: (int, int, int, int, int) -> None
        """
        Simple Autoencoder with residual blocks.

        :param mid_channels: intermediate channels of the downscale/upscale part
        :param n_res_layers: number of residual layers (for both the encoder and the decoder)
        :param code_channels: number of code channels
        """

        super().__init__(
            mid_channels=mid_channels, n_res_layers=n_res_layers,
            code_channels=code_channels, code_h=code_h, code_w=code_w
        )


    @classmethod
    def init_from_pth(cls, pth_file_path, mode='eval', device='cuda'):
        # type: (str, str, str) -> AutoencoderPlus

        if not torch.cuda.is_available():
            pth_dict = torch.load(pth_file_path, map_location='cpu')
        else:
            pth_dict = torch.load(pth_file_path)

        autoencoder = cls(
            mid_channels=pth_dict['mid_channels'],
            n_res_layers=pth_dict['n_res_layers'],
            code_channels=pth_dict['code_channels'],
            code_h=pth_dict['code_h'],
            code_w=pth_dict['code_w'],
        )
        autoencoder.load_state_dict(pth_dict['state_dict'])
        autoencoder.cnf_dict = pth_dict['cnf_dict']
        autoencoder.anomaly_th = pth_dict['anomaly_th']

        if mode == 'eval':
            autoencoder.requires_grad(False)
            autoencoder.eval()

        return autoencoder.to(device)


    def get_anomaly_score(self, x, score_function):
        # type: (torch.Tensor, str) -> torch.Tensor
        """
        Returns the anomaly scores of the input batch `x`
        w.r.t. the specified score function.

        :param x: batch of input images with shape (B, C, H, W)
        :param score_function: The score function can be:
            >> 'MSE_LOSS': MSE between `x` and reconstruction `d(e(x))`
            >> 'MS_SSIM_LOSS':  multi-scale structural similarity loss
                between target `x` and reconstruction `d(e(x))`
            >> 'CODE_MSE_LOSS': MSE between target code `e(x)` and
                reconstruction code`e(d(e(x)))`

        :return: anomaly scores -> tensor with shape (B,)
            >> (one score for each batch element)
        """

        # backup function to restore the working mode after method call
        backup_function = self.train if self.training else self.eval

        x = x.to(self.device)
        code_true = self.encode(x)
        x_pred = self.decode(code_true)

        self.eval()
        with torch.no_grad():

            if score_function == 'CODE_MSE_LOSS':
                code_pred = self.encode(x_pred)
                score = ((code_pred - code_true) ** 2).mean([1, 2, 3])

            elif score_function == 'MSE_LOSS':
                x = nn.MSELoss(reduction='none')(x_pred, x)
                score = x.mean((1, 2, 3))

            elif score_function == 'MS_SSIM_LOSS':
                ssim_f = piq.MultiScaleSSIMLoss(reduction='none')
                score = ssim_f(x_pred, x)

            else:
                raise ValueError(
                    f'[!] unknown score function "{score_function}"'
                )

        # restore previous working mode (train/eval)
        backup_function()

        return score


    def get_code_anomaly_perc(self, img, anomaly_th=None, max_val=100):
        # type: (np.ndarray, float, Optional[int]) -> float
        """
        :param img:
        :param anomaly_th:
        :return:
        """

        if anomaly_th is None:
            assert self.anomaly_th is not None, \
                f'one of `anomaly_th` and `self.anomaly_th` ' \
                f'must not be `None`'

        # ---- pre-processing transformations
        # (1) crop if needed
        # (2) resize if needed
        # (3) convert from BGR to RGB
        # (4) convert from `np.ndarray` to `torch.Tensor`
        # TODO: change this
        pre_proc_tr = pre_processing.PreProcessingTr(
            resized_h=256,  # self.cnf_dict['resized_h'],
            resized_w=256,  # self.cnf_dict['resized_w'],
            crop_x_min=0,  # self.cnf_dict['crop_x_min'],
            crop_y_min=0,  # self.cnf_dict['crop_y_min'],
            crop_side=256,  # self.cnf_dict['crop_side'],
            to_tensor=True
        )

        x = pre_proc_tr(img)
        x = x.unsqueeze(0).to(self.device)

        anomaly_score = self.get_anomaly_score(x, 'CODE_MSE_LOSS').item()

        # we want an image with `anomaly_score == anomaly_th`
        # to have an anomaly percentage of 50%
        anomaly_prob = anomaly_score / anomaly_th
        anomaly_prec = (anomaly_prob * 100) - 50

        if max_val is None:
            return max(0, anomaly_prec)
        else:
            return float(np.clip(anomaly_prec, 0, max_val))


    def get_flat_code(self, img):
        # type: (np.ndarray) -> torch.Tensor
        """
        ...
        """

        pre_proc_tr = pre_processing.PreProcessingTr(
            resized_h=256,  # self.cnf_dict['resized_h'],
            resized_w=256,  # self.cnf_dict['resized_w'],
            crop_x_min=0,  # self.cnf_dict['crop_x_min'],
            crop_y_min=0,  # self.cnf_dict['crop_y_min'],
            crop_side=256,  # self.cnf_dict['crop_side'],
            to_tensor=True
        )

        x = pre_proc_tr(img)
        x = x.unsqueeze(0).to(self.device)
        code = self.encode(x)
        return code.cpu().numpy().reshape(-1)
