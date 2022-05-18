from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.residual import ResidualStack


class AutoencoderCore(BaseModel):

    def __init__(self, mid_channels=128, n_res_layers=2, code_channels=3, code_h=4, code_w=4):
        # type: (int, int, int, int, int) -> None
        """
        Simple Autoencoder with residual blocks.

        :param mid_channels: intermediate channels of the downscale/upscale part
        :param n_res_layers: number of residual layers (for both the encoder and the decoder)
        :param code_channels: number of code channels
        """

        super().__init__()

        self.n_res_layer = n_res_layers
        self.mid_channels = mid_channels

        self.code_channels = code_channels
        self.code_h = code_h
        self.code_w = code_w

        m = mid_channels  # shortcut

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
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            # --- first conv: (code_channels, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(code_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            nn.Conv2d(m, m, kernel_size=3, stride=1, padding=1),
            # --- upscale part: (m, H/8, W/8) -> (3, H, W)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m, m // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m // 2, m // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(m // 2, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.cache = {}
        self.cnf_dict = None

        self.kaiming_init(activation='LeakyReLU')


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        code = self.encoder(x)

        # save code shape (H, W) before adaptive average pooling
        # this information will be used by the decoder
        self.cache['h'] = code.shape[2]
        self.cache['w'] = code.shape[3]
        h = code.shape[2] if self.code_h is None else self.code_h
        w = code.shape[3] if self.code_w is None else self.code_w

        code = nn.AdaptiveAvgPool2d((h, w))(code)

        return code


    def decode(self, code):
        # type: (torch.Tensor) -> torch.Tensor
        __h, __w = self.cache['h'], self.cache['w']
        code = nn.AdaptiveAvgPool2d((__h, __w))(code)
        y = self.decoder(code)
        return y


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        code = self.encode(x)
        x = self.decode(code)
        return x


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


# ---------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = SimpleAutoencoder(
        n_res_layers=2, code_channels=3, code_h=4, code_w=4
    ).to(device)

    code = model.encode(x)
    y = model.forward(x)

    print(f'$> input shape: {tuple(x.shape)}')
    print(f'$> output shape: {tuple(y.shape)}')
    print(f'$> code shape: {tuple(code.shape)}')

    in_size = x.shape[1] * x.shape[2] * x.shape[3]
    code_size = code.shape[1] * code.shape[2] * code.shape[3]
    print(f'$> code is {in_size / code_size:.02f} times smaller than input')


if __name__ == '__main__':
    main()
