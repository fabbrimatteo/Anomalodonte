import torch
from torchvision import transforms

import utils
from conf import Conf
from models.autoencoder import DDAutoencoder
from path import Path

def generate_prototypes(exp_name):
    cnf = Conf(exp_name=exp_name)

    avg_img_path = cnf.ds_path / 'average_train_img.png'
    if avg_img_path.exists():
        avg_img = utils.imread(avg_img_path)
        avg_img = transforms.ToTensor()(avg_img)
    else:
        avg_img = None

    model = DDAutoencoder(code_channels=cnf.code_channels, avg_img=avg_img)
    model = model.to(cnf.device)
    model.requires_grad(False)
    model.load_w(cnf.exp_log_path / 'best.pth')
    model.eval()

    out_dir = cnf.exp_log_path / 'prototypes'
    out_dir.makedirs_p()
    for f in (cnf.ds_path / 'train').files('*.png'):
        img = utils.imread(f)
        code = model.get_code(rgb_img=img)

        print(f'$> generating code for image `{f}`')
        out_path = out_dir / f.basename().replace('.png', '.prot')
        torch.save(code, out_path)


def load_prototypes(ds_path):
    in_path = ds_path / 'prototypes'
    p_list = [torch.load(f) for f in in_path.files('*.prot')]
    return p_list


if __name__ == '__main__':
    generate_prototypes(exp_name='default')
