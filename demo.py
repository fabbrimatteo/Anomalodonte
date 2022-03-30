import os
import string

import cv2
import numpy as np
import torch
import torchvision.utils
from path import Path
from sklearn.neighbors import LocalOutlierFactor

import ad_drawer
import clustering
import visual
from conf import Conf
from evaluator import Evaluator


ST = string.digits + string.ascii_letters

SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def demo_anomaly_perc(mode, exp_name, show_if_gt=0.):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    for img_path in in_dir_path.files():
        img = cv2.imread(img_path)
        anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)

        prob = anomaly_perc / 100
        if prob > show_if_gt:
            print(f'$> "{img_path.basename()}": {anomaly_perc:.2f}%')
            ad_drawer.show_anomaly(img, prob, header=img_path.basename())


def demo_clustering(mode, exp_name, n_clusters=8):
    cnf = Conf(exp_name=exp_name)
    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    out_dict = clustering.clustering(
        exp_name=exp_name, in_dir=in_dir_path,
        n_clusters=n_clusters
    )

    out_path = cnf.exp_log_path / 'clusters'
    if out_path.exists():
        os.system(f'rm -r "{out_path.abspath()}"')
    out_path.mkdir_p()

    torch.save(out_dict, out_path / 'clustering_dict.pth', )

    for i, dc in enumerate(out_dict['decoded_centroids']):
        dc = cv2.cvtColor(dc, cv2.COLOR_RGB2BGR)
        card = out_dict['cardinalities'][i]
        c = out_dict['centroids'][i]
        c = clustering.flat_code_to_tensor(c, device='cpu')
        c = visual.code2img(c)
        c = cv2.resize(c, (0, 0), fx=32, fy=32, interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imwrite(out_path / f'c{i:02d}_{card}.png', c)
        cv2.imwrite(out_path / f'c{i:02d}_{card}_decoded.png', dc)


def demo_lof(mode, exp_name, n_neighbors=12):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)
    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    in_dir_path = SPAL_PATH / mode / cnf.cam_id

    flat_codes = []

    train_paths = list(in_dir_path.files())

    for img_path in train_paths:
        print(img_path)
        img = cv2.imread(img_path)
        flat_code = evaluator.model.get_flat_code(img)
        flat_codes.append(flat_code)

    novelty = True if mode == 'test' else False
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=novelty)

    import os

    os.system(f'rm -r "{cnf.exp_log_path}/train_sort"')
    os.system(f'mkdir "{cnf.exp_log_path}/train_sort"')

    os.system(f'rm -r "{cnf.exp_log_path}/strange_oks_lof"')
    os.system(f'mkdir "{cnf.exp_log_path}/strange_oks_lof"')

    os.system(f'rm -r "{cnf.exp_log_path}/strange_oks"')
    os.system(f'mkdir "{cnf.exp_log_path}/strange_oks"')

    if mode == 'train':
        labels = lof.fit_predict(np.array(flat_codes))
        scores = lof.negative_outlier_factor_
        argsort = np.argsort(scores)
        for idx in argsort:
            img_path = train_paths[idx]
            img = cv2.imread(train_paths[idx])
            label = labels[idx]
            score = scores[idx]
            anomaly_score = max(0, (score / (-1.5)) - 0.5)

            plus = ''
            if anomaly_score > 1.5:
                plus = '+'
            if anomaly_score > 2.0:
                plus = '++'
            if anomaly_score > 2.5:
                plus = '+++'

            name = img_path.basename().split('@')[-1]
            anomaly_score_100i = int(round(anomaly_score * 100))
            anomaly_score_100i = min(anomaly_score_100i, 999)
            name = f'{anomaly_score_100i:03d}@{name}'
            anomaly_score = min(anomaly_score, 1)

            if anomaly_score >= 0.5:
                cv2.imwrite(f'{cnf.exp_log_path}/strange_oks_lof/{img_path}', img)

            anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)

            if anomaly_perc >= 50:
                cv2.imwrite(f'{cnf.exp_log_path}/strange_oks/{img_path}', img)

            if anomaly_score > 0.:
                # key = ad_drawer.show_anomaly(
                #     img, anomaly_score, img_path.basename(), plus
                # )
                import os

                os.system(f'ln -s '
                          f'"{img_path}" '
                          f'"{cnf.exp_log_path}/train_sort/{name}"')

                # cv2.imshow(f'{scores[idx]:.2f}', img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


if __name__ == '__main__':
    print()
    # demo_anomaly_perc(exp_name='cam3', mode='train', show_if_gt=0.95)
    demo_clustering(exp_name='cam1', mode='train', n_clusters=16)
    demo_clustering(exp_name='cam2', mode='train', n_clusters=16)
    demo_clustering(exp_name='cam3', mode='train', n_clusters=16)
    demo_clustering(exp_name='cam3_big', mode='train', n_clusters=16)
    # demo_lof(exp_name='cam1', mode='train')
