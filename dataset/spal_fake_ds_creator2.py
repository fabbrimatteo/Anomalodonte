# -*- coding: utf-8 -*-
# ---------------------

import os

from path import Path


DS_ROOT = Path('/home/matteo/PycharmProjects/Anomalodonte/dataset/spal_fake_rect/test_real')


def get_anomaly_type(file_name):
    file_name = Path(file_name).basename()
    if 'good' in file_name:
        return '000'
    else:
        x = file_name.split('_')[1]
        a1, a2, a3 = int(x[1]), int(x[2]), int(x[3])
        return f'{a1}{a2}{a3}'

def main():

    goods = []
    bads = []
    for f in DS_ROOT.files():
        ant = get_anomaly_type(f.basename())

        if ant == '000':
            goods.append(f)
        elif ant[0] == '1':
            bads.append(f)
        else:
            print(f)

    print(len(goods))
    print(len(bads))


if __name__ == '__main__':
    main()
