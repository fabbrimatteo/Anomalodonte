import numpy as np
import torch

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

        if mode == 'eval':
            autoencoder.requires_grad(False)
            autoencoder.eval()

        return autoencoder.to(device)


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
