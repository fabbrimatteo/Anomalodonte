import torch


def interpol_loss(x, model, device):

    # half batch size
    b = x.shape[0] // 2

    x1 = x[:b, ...]
    x2 = x[b:, ...]

    # interpolation factor
    alpha = torch.rand((b, 1, 1, 1)).to(device)

    inter_a = alpha * model.encode(x1) + (1 - alpha) * model.encode(x2)
    inter_b = model.encode(alpha * x1 + (1 - alpha) * x2)

    return torch.nn.MSELoss()(inter_a, inter_b)
