import cv2
from path import Path

import ad_drawer
from conf import Conf
from evaluator import Evaluator


SPAL_PATH = Path('/goat-nas/Datasets/spal/spal_cuts')


def demo(in_dir_path, exp_name):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    for f in in_dir_path.files():
        img = cv2.imread(f)
        anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)

        print(f, anomaly_perc)

        prob = anomaly_perc / 100
        if prob > 0.95:
            ad_drawer.show_anomaly(img, prob, label=f.basename())


if __name__ == '__main__':
    demo(
        exp_name='mar2022',
        in_dir_path=SPAL_PATH / 'train' / 'cam_1'
    )
