import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from skimage import filters
import random
from conf import Conf
from models.autoencoder import SimpleAutoencoder
from pre_processing import ReseizeThenCrop
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


def results(exp_name):
    cnf = Conf(exp_name=exp_name)

    model = SimpleAutoencoder.init_from_pth(pth_file_path=cnf.exp_log_path / 'best.pth')

    mask = cv2.imread('mask.png', 0)
    mask[mask < 128] = 0
    mask[mask >= 128] = 1

    trs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ReseizeThenCrop(resized_h=628, resized_w=751, crop_x_min=147, crop_y_min=213, crop_side=256),
    ])

    mse = torch.nn.MSELoss()
    p_list = load_prototypes(ds_path=cnf.exp_log_path)

    flist = list((cnf.ds_path / 'test').files())
    random.shuffle(flist)

    score = 0

    goods = []
    bads = []
    for i, f in enumerate(flist):
        img = cv2.imread(f)
        x = trs(img).unsqueeze(0).to(cnf.device)
        x_true = x[0].clone()
        code = model.encode(x)[0]

        all_diffs = [mse(p, code).item() for p in p_list]
        selected_prototype = p_list[np.argmin(all_diffs)]

        x_true = x_true.cpu().numpy().transpose((1, 2, 0))
        x_pred = model.decode(selected_prototype.unsqueeze(0).to(cnf.device)).cpu().numpy()[0].transpose((1, 2, 0))

        final_map = get_anomaly_map(x_pred, x_true)
        print(f.basename())

        # anomaly_score = final_map.mean() * 100
        anomaly_score = ((final_map * mask).sum() / (mask == 1).sum()) * 100

        label_true = 0 if 'good' in f.basename() else 1
        if label_true == 0:
            goods.append(anomaly_score)
        else:
            bads.append(anomaly_score)
        label_pred = 0 if anomaly_score < 7.5 else 1
        score += int(label_true == label_pred)

        print(
            f'G: {np.mean(goods):.2f} (+- {np.std(goods):.2f})  |  '
            f'B: {np.mean(bads):.2f} (+- {np.std(bads):.2f}) ')

        fig, axes = plt.subplots(1, 3, figsize=(80, 60), dpi=100)
        axes[0].imshow(x_true[:, :, ::-1])
        axes[0].set_title(f'{f.basename()}')
        axes[1].imshow(x_pred[:, :, ::-1])
        axes[2].imshow(final_map, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title(f'{final_map.mean() * 100:.2f}')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        print()
        plt.show()

    return 0


if __name__ == '__main__':
    results(exp_name='sandramilo')
