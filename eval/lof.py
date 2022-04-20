import cv2
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from eval.utils import bin_class_metrics
from models.autoencoder_plus import AutoencoderPlus


class Loffer(object):

    def __init__(self, train_dir, model, n_neighbors=12):
        # type: (str, AutoencoderPlus, int) -> None
        """
        :param train_dir: root directory of the training set
        :param model: model you want to evaluate
        :param n_neighbors: neighborhood size for LOF algorithm
        """

        self.model = model

        # build list of sorted training images
        # >> images are in ascending alphabetical order
        train_paths = list(train_dir.files())
        train_paths = sorted(train_paths)

        # build list of code (one code for each training image)
        train_codes = []
        for j, img_path in enumerate(train_paths):
            img = cv2.imread(img_path)
            flat_code = model.get_flat_code(img)
            train_codes.append(flat_code)

        # find outliers in training codes
        # using a LOF model as outlier-detector (i.e. novelty=False)
        labels = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=False,
        ).fit_predict(train_codes)

        # remove outliers from training codes
        # n0 = len(train_codes)
        train_codes = np.array(train_codes)[labels == 1]
        # n1 = len(train_codes)
        # print(f'$> {n0 - n1} outlier(s) removed')

        # fit a LOF model for novelty-detection
        # using the filtered training codes
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
        )
        self.lof.fit(train_codes)


    def get_anomaly_perc(self, img, max_val=100.):
        # type: (np.ndarray, float) -> float
        """
        :param img: input BGR image;
            >> shape: (H, W, 3) and values in [0, 255]
        :param max_val: clip anomaly perc using `max_val` as upper bound
            >> use `max_val=100` for a value in [0, 100]
        :return: anomaly percentage
        """

        flat_code = self.model.get_flat_code(img)
        flat_code = flat_code.reshape(1, -1)
        lof_score = self.lof.score_samples(flat_code)[0]
        anomaly_perc = 100 * max(0, (lof_score / (-1.5)) - 0.5)
        return min(anomaly_perc, max_val)


    def evaluate(self, test_dir):

        labels_true = []
        labels_pred = []
        for img_path in test_dir.files():
            img = cv2.imread(img_path)
            anomaly_perc = self.get_anomaly_perc(img)
            label_true = img_path.basename().split('_')[0]
            labels_true.append(label_true)
            label_pred = 'good' if anomaly_perc < 50 else 'bad'
            labels_pred.append(label_pred)

        rates_dict = bin_class_metrics(
            labels_pred=np.array(labels_pred),
            labels_true=np.array(labels_true)
        )

        return rates_dict
