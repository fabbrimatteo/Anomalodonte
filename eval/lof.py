from typing import Tuple

import cv2
import numpy as np
from path import Path
from sklearn.neighbors import LocalOutlierFactor

from eval.sorted_buffer import SortedBuffer
from eval.utils import bin_class_metrics
from models.autoencoder_plus import AutoencoderPlus
from visual_utils import draw_anomaly_ui


class Loffer(object):

    def __init__(self, train_dir, model, n_neighbors=12, train_dir_master=None):
        # type: (str, AutoencoderPlus, int) -> None
        """
        :param train_dir: root directory of the training set
        :param model: model you want to evaluate
        :param n_neighbors: neighborhood size for LOF algorithm
        """

        self.model = model
        self.master_ratio = 4
        train_dir = Path(train_dir)

        # build list of sorted training images
        # >> images are in ascending alphabetical order
        self.train_paths = list(train_dir.files())
        if train_dir_master is not None:
            train_master_paths = list(train_dir_master.files())
            train_master_paths.sort(key=lambda p: p.basename())
            train_master_paths = train_master_paths[::self.master_ratio]
            self.train_paths = self.train_paths + train_master_paths
        self.train_paths = sorted(self.train_paths)

        # build list of code (one code for each training image)
        train_codes = []
        for j, img_path in enumerate(self.train_paths):
            img = cv2.imread(img_path)
            flat_code = model.get_flat_code(img)
            train_codes.append(flat_code)

        # find outliers in training codes
        # using a LOF model as outlier-detector (i.e. novelty=False)
        self.lof_train = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=False
        )
        self.train_labels = self.lof_train.fit_predict(train_codes)

        # score training samples
        self.train_scores = self.lof_train.negative_outlier_factor_
        self.train_scores = 100 * ((self.train_scores / (-1.5)) - 0.5)
        self.train_scores[self.train_scores < 0] = 0
        idxs = np.argwhere(self.train_scores > 50)[:, 0]

        self.train_outliers = {}
        for i in idxs:
            key = self.train_paths[i]
            self.train_outliers[key] = self.train_scores[i]

        # remove outliers from training codes
        n0 = len(train_codes)
        train_codes = np.array(train_codes)[self.train_labels == 1]
        n1 = len(train_codes)
        print(f'$> {n0 - n1} outlier(s) removed')

        # fit a LOF model for novelty-detection
        # using the filtered training codes
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
        )
        self.lof.fit(train_codes)


    def get_train_labels(self):
        return list(zip(self.train_paths, self.train_labels.tolist()))

    def get_train_scores(self):
        return list(zip(self.train_paths, self.train_scores.tolist()))

    def get_anomaly_perc(self, img):
        # type: (np.ndarray) -> float
        """
        :param img: input BGR image;
            >> shape: (H, W, 3) and values in [0, 255]
        :return: anomaly percentage in range [0, 100]
            >> NOTE: it's just a clipped version of the anomaly score!
        """
        anomaly_score = self.get_anomaly_score(img)
        anomaly_perc = min(anomaly_score, 100)
        return anomaly_perc


    def get_anomaly_score(self, img):
        # type: (np.ndarray) -> float
        """
        :param img: input BGR image;
            >> shape: (H, W, 3) and values in [0, 255]
        :return: anomaly score
        """

        flat_code = self.model.get_flat_code(img)
        flat_code = flat_code.reshape(1, -1)
        lof_score = self.lof.score_samples(flat_code)[0]
        anomaly_score = 100 * max(0, (lof_score / (-1.5)) - 0.5)
        return anomaly_score


    def evaluate(self, test_dir, out_dir=None):

        if out_dir:
            out_dir.makedirs_p()

        labels_true = []
        labels_pred = []
        top16_errors = SortedBuffer[Tuple[float, np.ndarray, float]](
            buffer_size=16, sort_key=lambda x: x[0]
        )
        for img_path in test_dir.files():
            img = cv2.imread(img_path)
            label_true = img_path.basename().split('_')[0]
            ap_pred = self.get_anomaly_perc(img)
            ap_true = 0 if label_true == 'good' else 100

            ap_error = abs(ap_true - ap_pred)
            top16_errors.append((ap_error, img[:, :, ::-1], ap_pred))

            labels_true.append(label_true)
            label_pred = 'good' if ap_pred < 50 else 'bad'
            labels_pred.append(label_pred)

            if out_dir:
                img_ui = draw_anomaly_ui(img, ap_pred, label_true)
                cv2.imwrite(out_dir / img_path.basename(), img_ui)


        rates_dict = bin_class_metrics(
            labels_pred=np.array(labels_pred),
            labels_true=np.array(labels_true)
        )

        return rates_dict, top16_errors
