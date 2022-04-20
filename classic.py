import cv2
import numpy as np


def classic(img):
    h, w = img.shape[:2]
    dh = h // 3
    dw = w // 3

    cv2.imshow('', img)
    cv2.waitKey()
    img = cv2.resize(img, (3, 3), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imshow('', img)
    cv2.waitKey()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) / 255.
    for i in range(0, h, dh):
        for j in range(0, w, dw):
            x_min, y_min = j, i
            x_max, y_max = x_min + dw, y_min + dh
            cut = img[y_min:y_max, x_min:x_max]
            m = np.mean(cut, axis=(0, 1))
            s = np.std(cut, axis=(0, 1))
            print([x_min, y_min, x_max, y_max], m, s)


if __name__ == '__main__':
    r = classic(img=cv2.imread('/goat-nas/Datasets/spal/spal_cuts/test/cam_2/bad_2022_02_09_17_28_48.jpg'))
