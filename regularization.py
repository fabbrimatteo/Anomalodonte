import torch
from models.autoencoder_core import AutoencoderCore


def interpol_loss(x, model):
    # type: (torch.Tensor, AutoencoderCore) -> torch.Tensor

    # half batch size
    b = x.shape[0] // 2

    x1 = x[:b, ...]
    x2 = x[b:, ...]

    # interpolation factor
    alpha = torch.rand((b, 1, 1, 1)).to(model.device)

    # we want interpolation-A to be the same as interpolation-B
    inter_a = alpha * model.encode(x1) + (1 - alpha) * model.encode(x2)
    inter_b = model.encode(alpha * x1 + (1 - alpha) * x2)
    inter_loss = torch.nn.MSELoss()(inter_a, inter_b)

    return inter_loss
