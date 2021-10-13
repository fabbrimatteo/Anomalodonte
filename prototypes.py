import cv2
import torch
import torchvision

from conf import Conf
from models.autoencoder import SimpleAutoencoder
from pre_processing import PreProcessingTr


def generate_prototypes(exp_name):
    cnf = Conf(exp_name=exp_name)

    model = SimpleAutoencoder.init_from_pth(cnf.exp_log_path / 'best.pth')

    trs = PreProcessingTr(
        resized_h=256, resized_w=256,
        crop_x_min=812, crop_y_min=660, crop_side=315
    )

    out_dir = cnf.exp_log_path / 'prototypes'
    out_dir.makedirs_p()
    for f in (cnf.ds_path / 'train').files():
        img = cv2.imread(f)
        x = trs(img).unsqueeze(0).to(cnf.device)
        code = model.encode(x)

        print(f'$> generating code for image `{f}`')
        out_path = out_dir / f.basename().replace('.png', '.prot')
        torch.save(code[0], out_path)


def load_prototypes(ds_path):
    in_path = ds_path / 'prototypes'
    p_list = [torch.load(f) for f in in_path.files('*.prot')]
    return p_list


if __name__ == '__main__':
    generate_prototypes(exp_name='ultra_small')
