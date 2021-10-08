import cv2
import torch
import torchvision

from conf import Conf
from models.autoencoder import SimpleAutoencoder
from pre_processing import CropThenResize


def generate_prototypes(exp_name):
    cnf = Conf(exp_name=exp_name)

    model = SimpleAutoencoder(code_channels=cnf.code_channels)
    model = model.to(cnf.device)
    model.requires_grad(False)
    model.load_w(cnf.exp_log_path / 'best.pth')
    model.eval()

    trs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        CropThenResize(resized_h=628, resized_w=751, crop_x_min=147, crop_y_min=213, crop_side=256),
    ])

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
    generate_prototypes(exp_name='sandramilo')
