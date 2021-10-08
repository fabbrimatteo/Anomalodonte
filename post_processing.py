import numpy as np
import torch


def tensor2img(x):
    # type: (torch.Tensor) -> np.ndarray
    x = x.detach().cpu().numpy()  # from tensor to array
    x = x.transpose((1, 2, 0))  # from CHW to HWC
    x = (x * 255).astype(np.uint8)  # from [0,1] to [0,255]
    return x


if __name__ == '__main__':
    debug()
