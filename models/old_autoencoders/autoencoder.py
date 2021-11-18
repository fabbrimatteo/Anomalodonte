from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

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
            nn.Conv2d(3, m // 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(m // 2, m // 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(m // 2, m, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(m, m, kernel_size=3, stride=1, padding=1),
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            # --- last conv: (m, H/8, W/8) -> (code_channels, H/8, W/8)
            nn.Conv2d(mid_channels, code_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            # --- first conv: (code_channels, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(code_channels, mid_channels, kernel_size=(3, 3), stride=1, padding=1),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            TConv2D(m, m, kernel_size=3, stride=1, padding=1),
            # --- upscale part: (m, H/8, W/8) -> (3, H, W)
            TConv2D(m, m // 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            TConv2D(m // 2, m // 2, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            TConv2D(m // 2, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

        self.normal = torch.distributions.Normal(0, 1)

        self.pre_processing_tr = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        self.cache = {}
        self.stats = None
        self.cnf_dict = None


    def encode(self, x, code_noise=None):
        # type: (torch.Tensor, Optional[float]) -> torch.Tensor
        code = self.encoder(x)

        if self.training and code_noise is not None:
            code = code + code_noise * self.normal.sample(code.shape)

        self.cache['h'] = code.shape[2]
        self.cache['w'] = code.shape[3]
        h = code.shape[2] if self.code_h is None else self.code_h
        w = code.shape[3] if self.code_w is None else self.code_w
        code = nn.AdaptiveAvgPool2d((h, w))(code)

        return code


    def decode(self, code):
        # type: (torch.Tensor) -> torch.Tensor
        code = nn.AdaptiveAvgPool2d((self.cache['h'], self.cache['w']))(code)
        y = self.decoder(code)
        return y


    def forward(self, x, code_noise=None):
        # type: (torch.Tensor, float) -> torch.Tensor
        code = self.encode(x, code_noise=code_noise)
        x = self.decode(code)
        return x


    def to(self, device):
        # type: (Union[str, torch.device]) -> SimpleAutoencoder
        super(SimpleAutoencoder, self).to(device)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)
        return self


    def get_code(self, rgb_img):
        # type: (np.ndarray) -> torch.Tensor
        """
        :param rgb_img: numpy image (RGB) with shape (H, W, 3)
        :return: encoded version of the input image -> `code`
            >> `code` is a torch.Tensor with shape (self.code_channels, H/8, W/8)
        """
        x = self.pre_processing_tr(rgb_img)
        with torch.no_grad():
            x = x.unsqueeze(0).to(self.device)
            code = self.encode(x)[0]
        return code


    def get_numpy_code(self, rgb_img):
        # type: (np.ndarray) -> np.ndarray
        """
        :param rgb_img: numpy image (RGB) with shape (H, W, 3)
        :return: encoded version of the input image -> `code`
            >> `code` is a np.array with shape (self.code_channels, H/8, W/8)
        """
        code = self.get_code(rgb_img)
        return code.cpu().numpy()


    def save_w(self, path, test_stats=None, cnf_dict=None):
        # type: (str, Optional[Dict], Optional[Dict]) -> None
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
            'test_stats': test_stats,
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
        autoencoder.stats = pth_dict['test_stats']

        if mode == 'eval':
            autoencoder.requires_grad(False)
            autoencoder.eval()

        return autoencoder.to(device)


    def get_code_anomaly_score(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        # backup function to restore the working mode after method call
        backup_function = self.train if self.training else self.eval

        self.eval()
        with torch.no_grad():
            code_true = self.encode(x, code_noise=None)
            x_pred = self.decode(code_true)
            code_pred = self.encode(x_pred, code_noise=None)

            # MSE with aggregation (mean) on (C, H, W,)
            # >> code_error.shape = (<batch_size>,)
            code_error = ((code_pred - code_true) ** 2).mean([1, 2, 3])

        # restore previous working mode (train/eval)
        backup_function()

        return code_error


    def get_code_anomaly_perc(self, x, stats=None):
        # type: (torch.Tensor, Optional[Dict]) -> float

        if stats is not None:
            self.stats = stats
        assert self.stats is not None, \
            'you need to build the anomaly_th dictionary for this model ' \
            'before using the `get_code_anomaly_perc` method, ' \
            'or you can pass the anomaly_th dictionary as input'

        anomaly_score = self.get_code_anomaly_score(x).item()
        if anomaly_score < self.stats['good']['q0.75']:
            anomaly_perc = 0
        else:
            anomaly_perc = anomaly_score / (2 * self.stats['good']['w1'])
        return min(anomaly_perc, 1)


# ---------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = SimpleAutoencoder(n_res_layers=2, code_channels=8, code_h=8, code_w=8).to(device)

    model.get_code_anomaly_score(x)
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
