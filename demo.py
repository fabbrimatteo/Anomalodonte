import cv2
from path import Path

import ad_drawer
from conf import Conf
from evaluator import Evaluator


SPAL_PATH = Path('/nas/softechict-nas-3/matteo/Datasets/Spal')


def demo(img_path, exp_name):
    cnf = Conf(exp_name=exp_name)
    evaluator = Evaluator(cnf=cnf)

    _, _, boxplot_dict, _ = evaluator.get_stats()
    anomaly_th = boxplot_dict['good']['upper_whisker']

    for f in (SPAL_PATH / 'cables_6mm_p1_rect' / 'ev-test').files():
        img = cv2.imread(f)
        anomaly_perc = evaluator.model.get_code_anomaly_perc(img, anomaly_th)
        print(f, anomaly_perc)

        prob = anomaly_perc / 100
        ad_drawer.show_anomaly(img, prob, label=f.basename())


if __name__ == '__main__':
    demo(exp_name='a5_noise', img_path=None)
