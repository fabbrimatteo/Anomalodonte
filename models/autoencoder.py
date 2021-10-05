import torch
import torch.nn as nn
from torchvision import transforms

from models.base_model import BaseModel
from models.residual import ResidualStack


TConv2D = nn.ConvTranspose2d  # shorcut


class DDAutoencoder(BaseModel):

    def __init__(self, mid_channels=128, n_res_layers=2, code_channels=8, avg_img=None):
        # type: (int, int, int, torch.Tensor) -> None
        """
        Digital-Design Autoencoder.

        :param mid_channels: intermediate channels of the downscale/upscale part
        :param n_res_layers: number of residual layers (for both the encoder and the decoder)
        :param code_channels: number of code channels
        :param noise_type: type of code noise
        """

        super(DDAutoencoder, self).__init__()

        self.mid_channels = mid_channels
        self.n_res_layer = n_res_layers
        self.code_channels = code_channels

        m = mid_channels
        kernel_size = 4

        self.avg_img = avg_img
        if self.avg_img is not None:
            self.avg_img = avg_img.unsqueeze(0)

        self.encoder = nn.Sequential(
            # --- downscale part: (3, H, W) -> (m, H/8, W/8)
            nn.Conv2d(3, m // 2, kernel_size, stride=2, padding=1, padding_mode='zeros'), nn.SiLU(),
            nn.Conv2d(m // 2, m // 2, kernel_size, stride=2, padding=1, padding_mode='zeros'), nn.SiLU(),
            nn.Conv2d(m // 2, m, kernel_size, stride=2, padding=1, padding_mode='zeros'), nn.SiLU(),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(m, m, kernel_size - 1, stride=1, padding=1, padding_mode='zeros'),
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers, padding_mode='zeros'),
            # --- last conv: (m, H/8, W/8) -> (code_channels, H/8, W/8)
            nn.Conv2d(mid_channels, code_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            # --- first conv: (code_channels, H/8, W/8) -> (m, H/8, W/8)
            nn.Conv2d(code_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            # --- residual part: (m, H/8, W/8) -> (m, H/8, W/8)
            ResidualStack(m, m, mid_channels=m // 4, n_res_layers=n_res_layers),
            TConv2D(m, m, kernel_size - 1, stride=1, padding=1),
            # --- upscale part: (m, H/8, W/8) -> (3, H, W)
            TConv2D(m, m // 2, kernel_size, stride=2, padding=1), nn.SiLU(),
            TConv2D(m // 2, m // 2, kernel_size, stride=2, padding=1), nn.SiLU(),
            TConv2D(m // 2, 3, kernel_size, stride=2, padding=1), nn.Tanh()
        )

        self.normal = torch.distributions.Normal(0, 1)

        self.pre_processing_tr = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])


    @staticmethod
    def normalize_code(code):
        # type: (torch.Tensor) -> torch.Tensor
        m = code.mean([1, 2, 3]).view(-1, 1, 1, 1)
        s = code.std([1, 2, 3]).view(-1, 1, 1, 1)
        return (code - m) / (s + 0.00001)


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        code = self.encoder(x)
        if self.training:
            code = code + 0.25 * self.normal.sample(code.shape)
        return code


    def decode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        y = self.decoder(x)
        if self.avg_img is not None:
            y = torch.clip(y + self.avg_img, 0, 1)
        return y


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        code = self.encode(x)
        x = self.decode(code)
        return x


    def to(self, *args, **kwargs):
        device = args[0]
        super(DDAutoencoder, self).to(device)
        self.normal.loc = self.normal.loc.to(device)
        self.normal.scale = self.normal.scale.to(device)
        if self.avg_img is not None:
            self.avg_img = self.avg_img.to(device)
        return self


    def get_code(self, rgb_img):
        x = self.pre_processing_tr(rgb_img)
        x = x.unsqueeze(0).to(self.device)
        if self.avg_img is not None:
            x = x - self.avg_img
        code = self.encode(x)
        return code[0]


    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        save model weights in the specified path
        """
        __state = {
            'state_dict': self.state_dict(),
            'mid_channels': self.mid_channels,
            'n_res_layers': self.n_res_layer,
            'code_channels': self.code_channels,
            'avg_img': self.avg_img
        }
        torch.save(__state, path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        load model weights from the specified path
        """
        if not torch.cuda.is_available():
            __state = torch.load(path, map_location='cpu')
        else:
            __state = torch.load(path)
        self.load_state_dict(__state['state_dict'])
        self.avg_img = __state['avg_img']


# ---------

def main():
    import torchsummary

    batch_size = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DDAutoencoder(n_res_layers=2, code_channels=8).to(device)

    x = torch.rand((batch_size, 3, 256, 256)).to(device)
    torchsummary.summary(model=model, input_size=x.shape[1:], device=str(device))

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
