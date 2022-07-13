# -*- coding: utf-8 -*-
# ---------------------

import numpy as np


def tensor2img(img):
    img = img.detach().cpu().numpy().squeeze()
    img = (255*img).astype(np.uint8)
    img = img.transpose((1,2,0))
    return img

