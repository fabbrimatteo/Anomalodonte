# -*- coding: utf-8 -*-
# ---------------------

import os

from path import Path


DS_ROOT = Path('/nas/softechict-nas-3/rgasparini/datasets/spal/data_cavi')


def main():
    allfiles = []
    in_dir = DS_ROOT / 'BUONI' / '6mm' / 'CAM1_OK'
    for i, file in enumerate(in_dir.files()):
        print(f'ln -s \'{file.abspath()}\' spal_fake/good_{i:04d}.png')
        allfiles.append(file.basename())

    bad_dict = {}
    in_dir = DS_ROOT / 'SCARTI' / '6mm'
    for d in in_dir.dirs():
        for file in (d / 'CAM1').files():
            k = str(file.basename())
            if k in bad_dict:
                bad_dict[k].append(file.split('/')[-3])
            else:
                bad_dict[k] = [file.split('/')[-3]]

    counter = 0
    for d in in_dir.dirs():
        for file in (d / 'CAM1').files():
            k = str(file.basename())

            if k in bad_dict:
                bad_str = ''
                for c in str(bad_dict[k]):
                    bad_str += c.lower() if c in ['1', '2', 'S'] else ''
            else:
                bad_str = file.split('/')[-3]

            bad_str = f'{1 if "1" in bad_str else 0}{1 if "s" in bad_str else 0}{1 if "2" in bad_str else 0}'
            print(f'ln -s \'{file.abspath()}\' spal_fake/bad_t{bad_str}_{counter:04d}.png')
            counter += 1


if __name__ == '__main__':
    main()
