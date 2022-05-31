from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.residual import ResidualStack
import pre_processing


class Variational(BaseModel):

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

        code_size = self.code_channels * self.code_h * self.code_w

        self.fc_mu = nn.Linear(
            in_features=3072,
            out_features=code_size
        )

        self.fc_logvar = nn.Linear(
            in_features=3072,
            out_features=code_size
        )

        self.fc_up = nn.Sequential(
            nn.Linear(
                in_features=code_size,
                out_features=3072
            ),
            nn.SiLU()
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

        self.cache = {}
        self.cnf_dict = None

        self.kaiming_init(activation='LeakyReLU')

        self.kl_loss = 0
        self.normal = torch.distributions.Normal(0, 1)


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        h = self.encoder(x)
        self.cache['shape'] = h.shape
        flat_h = h.view(h.shape[0], -1)

        mu = self.fc_mu(flat_h)
        log_var = self.fc_logvar(flat_h)
        std = torch.exp(0.5 * log_var)

        if self.training:
            code = mu + std * self.normal.sample(mu.shape)
            __k = 1 + log_var - mu.pow(2) - log_var.exp()
            self.kl_loss = -0.5 * torch.mean(__k)
        else:
            code = mu

        return code


    def decode(self, code):
        # type: (torch.Tensor) -> torch.Tensor
        code = self.fc_up(code)
        code = code.view(self.cache['shape'])
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


    def to(self, *args, **kwargs):
        device = args[0]
        super(Variational, self).to(device)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)
        return self


    def get_flat_code(self, img):
        # type: (np.ndarray) -> torch.Tensor
        """
        ...
        """

        pre_proc_tr = pre_processing.PreProcessingTr(to_tensor=True)

        x = pre_proc_tr(img)
        x = x.unsqueeze(0).to(self.device)
        code = self.encode(x)
        return code.cpu().numpy().reshape(-1)


# ---------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    model = Variational(
        n_res_layers=2, code_channels=3, code_h=4, code_w=4
    ).to(device)

    code = model.encode(x)
    y = model.forward(x)

    print(f'$> input shape: {tuple(x.shape)}')
    print(f'$> output shape: {tuple(y.shape)}')
    print(f'$> code shape: {tuple(code.shape)}')

    in_size = x.shape[1] * x.shape[2] * x.shape[3]
    code_size = code.shape[-1]
    print(f'$> code is {in_size / code_size:.02f} times smaller than input')


if __name__ == '__main__':
    main()
