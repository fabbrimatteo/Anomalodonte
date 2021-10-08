import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from models.base_model import BaseModel
from models.residual import ResidualStack
from typing import Union

TConv2D = nn.ConvTranspose2d  # shorcut


class SimpleAutoencoder(BaseModel):

    def __init__(self, mid_channels=128, n_res_layers=2, code_channels=8):
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


    def encode(self, x, code_noise=None):
        # type: (torch.Tensor, float) -> torch.Tensor
        code = self.encoder(x)
        if self.training and code_noise is not None:
            code = code + code_noise * self.normal.sample(code.shape)
        return code


    def decode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        y = self.decoder(x)
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


    def save_w(self, path):
        # type: (str) -> None
        """
        save model weights at the specified path
        """
        __state = {
            'state_dict': self.state_dict(),
            'mid_channels': self.mid_channels,
            'n_res_layers': self.n_res_layer,
            'code_channels': self.code_channels,
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = SimpleAutoencoder(n_res_layers=2, code_channels=8).to(device)

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
