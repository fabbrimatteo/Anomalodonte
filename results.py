import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from path import Path
from skimage import filters
from torchvision import transforms

import utils
from conf import Conf
from models.autoencoder import SimpleAutoencoder
from prototypes import load_prototypes


def ssim(img1, img2, scale=1):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    h, w, _ = img1.shape
    if scale != 1:
        img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1x2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1x2

    ssim_map = ((2 * mu1x2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.mean(-1)
    ssim_map = cv2.resize(ssim_map, (w, h))
    return ssim_map


def get_anomaly_map(img_pred, img_true):
    ssim_map = 0
    for scale, weight in [(1, 0.25), (0.5, 0.5), (0.25, 0.25)]:
        ssim_map = weight * ssim(img_pred * 255, img_true * 255, scale=scale) + ssim_map
    ssim_map = (1 - ssim_map) / 2

    l1_map = np.abs(img_true - img_pred).mean(-1)

    return 0.5 * l1_map + 0.5 * ssim_map


def results(exp_name, avg_map=None):
    cnf = Conf(exp_name=exp_name)

    avg_img_path = cnf.ds_path / 'average_train_img.png'
    if avg_img_path.exists():
        avg_img = utils.imread(avg_img_path)
        avg_img = transforms.ToTensor()(avg_img)
    else:
        avg_img = None

    model = SimpleAutoencoder(code_channels=cnf.code_channels, avg_img=avg_img)
    model = model.to(cnf.device)
    model.requires_grad(False)
    model.load_w(cnf.exp_log_path / 'best.pth')
    model.eval()

    mse = torch.nn.MSELoss()
    p_list = load_prototypes(ds_path=cnf.exp_log_path)

    avg_final_map = 0
    n_imgs = len(Path('dataset/transistors/test').files('*.png'))
    for f in Path('dataset/transistors/test').files('*.png'):
        img = utils.imread(f)
        code = model.get_code(rgb_img=img)

        all_diffs = [mse(p, code).item() for p in p_list]
        selected_prototype = p_list[np.argmin(all_diffs)]

        x_true = model.pre_processing_tr(img).cpu().numpy().transpose((1, 2, 0))
        x_pred = model.decode(selected_prototype.unsqueeze(0).to(cnf.device)).cpu().numpy()[0].transpose((1, 2, 0))

        final_map = get_anomaly_map(x_pred, x_true)
        print(f.basename())
        if avg_map is not None:
            final_map = np.clip(final_map - avg_map, 0, 1)

        avg_final_map = (1 / n_imgs) * final_map + avg_final_map

        fig, axes = plt.subplots(1, 3)
        x_true = filters.gaussian(x_true, sigma=1.2, multichannel=True)
        axes[0].imshow(x_true)
        axes[1].imshow(x_pred)
        axes[2].imshow(final_map, cmap='inferno', vmin=0, vmax=1)
        axes[2].set_title(f'{final_map.mean() * 100:.2f}')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        print()
        plt.show()

    return avg_final_map


if __name__ == '__main__':
    f = results(exp_name='default')
    results(exp_name='default', avg_map=f)
