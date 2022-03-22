# -*- coding: utf-8 -*-
# ---------------------


import random

from path import Path


DS_ROOT = Path('/nas/softechict-nas-3/rgasparini/datasets/spal/data_cavi')


def main(in_dir_path, out_dir_path):
    in_dir_path = Path(in_dir_path)
    assert in_dir_path.exists(), \
        f'directory `{in_dir_path}` does not exist'

    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir_p()

    train_path = out_dir_path / 'train'
    train_path.mkdir_p()
    test_path = out_dir_path / 'test'
    test_path.mkdir_p()

    # analyze bad images and create a dictionary
    # containing file path and error(s) type
    bad_dict = {}
    in_dir_6mm = in_dir_path / 'SCARTI' / '6mm'
    for d in in_dir_6mm.dirs():
        error_type = d.basename()
        for file in (d / 'CAM1').files():
            k = str(file.basename())
            if not k in bad_dict:
                bad_dict[k] = {'path': file, 'errors': []}
            bad_dict[k]['errors'].append(str(error_type))

    # create sym-train_sort of bad images
    for i, k in enumerate(bad_dict):
        error_list = bad_dict[k]['errors']
        error_type = ''
        for error in ['P1', 'S', 'P2']:
            error_type += '1' if error in error_list else '0'

        new_name = test_path / f'bad_{error_type}_{i:04d}.png'
        print(f'ln -s \'{bad_dict[k]["path"]}\' \'{new_name.abspath()}\'')

    # number of bad images
    n_bad = len(bad_dict)

    # split good images in 2 sets -> train and test
    in_dir_6mm = DS_ROOT / 'BUONI' / '6mm' / 'CAM1_OK'
    allfiles = list(in_dir_6mm.files())
    random.Random(42).shuffle(allfiles)
    test_files = allfiles[:n_bad]
    train_files = allfiles[n_bad:]

    # create sim-train_sort of good images
    for i, file in enumerate(test_files):
        new_name = test_path / f'good_{i:04d}.png'
        print(f'ln -s \'{file.abspath()}\' \'{new_name.abspath()}\'')
    for i, file in enumerate(train_files):
        new_name = train_path / f'good_{len(test_files) + i:04d}.png'
        print(f'ln -s \'{file.abspath()}\' \'{new_name.abspath()}\'')


if __name__ == '__main__':
    main(in_dir_path=DS_ROOT, out_dir_path='cavallo')
