from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel


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
            # --
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # --
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # --
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # --
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # --
            nn.Conv2d(128, code_channels, kernel_size=1),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            # --
            nn.Conv2d(code_channels, 128, kernel_size=1),
            # --
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # --
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # --
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # --
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.cache = {}


    def encode(self, x, *args, **kwargs):
        # type: (torch.Tensor, ..., ...) -> torch.Tensor
        code = self.encoder(x)

        self.cache['h'] = code.shape[2]
        self.cache['w'] = code.shape[3]
        h = code.shape[2] if self.code_h is None else self.code_h
        w = code.shape[3] if self.code_w is None else self.code_w
        code = nn.AdaptiveAvgPool2d((h, w))(code)

        return code


    def decode(self, code):
        # type: (torch.Tensor) -> torch.Tensor
        code = nn.AdaptiveAvgPool2d((self.cache['h'], self.cache['w']))(code)
        return self.decoder(code)


    def forward(self, x, *args, **kwargs):
        # type: (torch.Tensor, ..., ...) -> torch.Tensor
        code = self.encode(x)
        x = self.decode(code)
        return x


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


# ---------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = SimpleAutoencoder(n_res_layers=2, code_channels=8, code_h=8, code_w=8).to(device)

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
