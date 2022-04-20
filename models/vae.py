from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import piq
import torch
import torch.nn as nn
from torchvision import transforms

import pre_processing
from models.base_model import BaseModel
from models.residual import ResidualStack


TConv2D = nn.ConvTranspose2d  # shorcut


class SimpleAutoencoder(BaseModel):

    def __init__(self, mid_channels=128, n_res_layers=2, code_channels=8, code_h=None, code_w=None):
        # type: (int, int, int) -> None
        """
        Simple Autoencoder with residual blocks.

        :param mid_channels: intermediate channels of the downscale/upscale part
        :param n_res_layers: number of residual layers (for both the encoder and the decoder)
        :param code_channels: number of code channels
        """

        super(SimpleAutoencoder, self).__init__()

        self.mid_channels = mid_channels
        m = mid_channels  # shortcut

        self.n_res_layer = n_res_layers
        self.code_channels = code_channels
        self.code_h = code_h
        self.code_w = code_w
        self.code_shape_chw = (code_channels, code_h, code_w)

        self.encoder = nn.Sequential(
            # --- downscale part: (3, H, W) -> (m, H/8, W/8)
            nn.Conv2d(3, m // 2, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(m // 2, m // 2, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(m // 2, m, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(m, m, kernel_size=3, stride=1, padding=1),
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            # --- last conv: (m, H/8, W/8) -> (code_channels, H/8, W/8)
            nn.Conv2d(mid_channels, code_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(in_features=3072, out_features=3 * 4 * 4),
            nn.Tanh(),
        )

        self.fc_logvar = nn.Linear(
            in_features=3072,
            out_features=3 * 4 * 4
        )

        self.decoder = nn.Sequential(
            # --- first conv: (code_channels, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(code_channels, mid_channels, kernel_size=(3, 3), stride=1, padding=1),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            nn.Conv2d(m, m, kernel_size=3, stride=1, padding=1),
            # --- upscale part: (m, H/8, W/8) -> (3, H, W)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m, m // 2, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m // 2, m // 2, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m // 2, 3, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

        self.normal = torch.distributions.Normal(0, 1)

        self.pre_processing_tr = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        self.cache = {}
        self.anomaly_th = None
        self.cnf_dict = None

        self.kl_loss = 0

        self.kaiming_init(activation='LeakyReLU')


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        hidden = self.encoder(x)
        self.cache['h'] = hidden.shape[2]
        self.cache['w'] = hidden.shape[3]
        flat_h = hidden.view(hidden.shape[0], -1)

        mu = self.fc_mu(flat_h)
        logvar = self.fc_logvar(flat_h)
        std = torch.exp(0.5 * logvar)

        if self.training:
            code = mu + std * self.normal.sample(mu.shape)
            code = code.view((-1, self.code_channels, self.code_h, self.code_w))
            self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            code = mu.view((-1, self.code_channels, self.code_h, self.code_w))

        return code


    def decode(self, code):
        # type: (torch.Tensor) -> torch.Tensor
        code = nn.AdaptiveAvgPool2d((self.cache['h'], self.cache['w']))(code)
        y = self.decoder(code)
        return y


    def forward(self, x, code_noise=None):
        # type: (torch.Tensor, float) -> torch.Tensor
        code = self.encode(x)
        x = self.decode(code)
        return x


    def to(self, device):
        # type: (Union[str, torch.device]) -> SimpleAutoencoder
        super(SimpleAutoencoder, self).to(device)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)
        return self


    def save_w(self, path, anomaly_th=None, cnf_dict=None):
        # type: (str, Optional[float], Optional[Dict]) -> None
        """
        save model weights at the specified path
        """
        __state = {
            'state_dict': self.state_dict(),
            'mid_channels': self.mid_channels,
            'n_res_layers': self.n_res_layer,
            'code_channels': self.code_channels,
            'code_h': self.code_h,
            'code_w': self.code_w,
            'anomaly_th': anomaly_th,
            'cnf_dict': cnf_dict
        }
        torch.save(__state, path)


    def load_w(self, path):
        # type: (str) -> None
        """
        load model weights from the specified path
        """
        if not torch.cuda.is_available():
            __state = torch.load(path, map_location='cpu')
        else:
            __state = torch.load(path)
        self.load_state_dict(__state['state_dict'])


    @classmethod
    def init_from_pth(cls, pth_file_path, device='cuda', mode='eval'):
        # type: (str, Union[str, torch.device], str) -> SimpleAutoencoder
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

        try:
            autoencoder.anomaly_th = pth_dict['anomaly_th']
        except KeyError:
            print('WARNING: you are using an old PTH file!')
            autoencoder.anomaly_th = None

        if mode == 'eval':
            autoencoder.requires_grad(False)
            autoencoder.eval()

        return autoencoder.to(device)


    def get_anomaly_score(self, x, score_function):
        # type: (torch.Tensor, str) -> torch.Tensor
        """
        Returns the anomaly scores of the input batch `x`
        w.r.t. the specified score function.

        The score function can be:
        >> 'MSE_LOSS': MSE between target `x` and reconstruction `d(e(x))`
        >> 'MS_SSIM_LOSS':  multi-scale structural similarity loss
        between target `x` and reconstruction `d(e(x))`
        >> 'CODE_MSE_LOSS': MSE between target code `e(x)` and
        reconstruction code`e(d(e(x)))`

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
        code_true = self.encode(x, code_noise=None)
        x_pred = self.decode(code_true)

        self.eval()
        with torch.no_grad():

            if score_function == 'CODE_MSE_LOSS':
                code_pred = self.encode(x_pred, code_noise=None)
                score = ((code_pred - code_true) ** 2).mean([1, 2, 3])

            elif score_function == 'MSE_LOSS':
                x = nn.MSELoss(reduction='none')(x_pred, x)
                score = x.mean((1, 2, 3))

            elif score_function == 'MS_SSIM_LOSS':
                ssim_f = piq.MultiScaleSSIMLoss(reduction='none')
                score = ssim_f(x_pred, x)

            else:
                raise ValueError(f'[!] unknown score function '
                                 f'{score_function}')

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


# ---------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = SimpleAutoencoder(n_res_layers=2, code_channels=3, code_h=4, code_w=4).to(device)

    y = model.forward(x)
    code = model.encode(x)

    print(f'$> input shape: {tuple(x.shape)}')
    print(f'$> output shape: {tuple(y.shape)}')
    print(f'$> code shape: {tuple(code.shape)}')

    in_size = x.shape[1] * x.shape[2] * x.shape[3]
    code_size = code.shape[1] * code.shape[2] * code.shape[3]

    print(f'$> code is {in_size / code_size:.02f} times smaller than input')


if __name__ == '__main__':
    main()
