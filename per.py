import cv2
import numpy as np

import mmu


C0_HSV = np.array([[[127, int(round(255 * (94 / 100))), int(round(255 * (88 / 100)))]]], dtype=np.uint8)
C1_HSV = np.array([[[0, int(round(255 * (94 / 100))), int(round(255 * (88 / 100)))]]], dtype=np.uint8)


def bar(img, x_min, y_min, h, w, color):
    img = cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color=color, thickness=-1)
    img = cv2.circle(img, center=(x_min, y_min + h // 2), radius=(h // 2), color=color, thickness=-1)
    img = cv2.circle(img, center=(x_min + w, y_min + h // 2), radius=(h // 2), color=color, thickness=-1)
    return img


def main():
    pad = 32
    barh = 64
    perc = 0.75
    img_path = mmu.resources.DEMO_RGB_JPG
    img = cv2.imread(img_path)

    h, w, _ = img.shape
    scale_factor = 960 / w
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    h, w, _ = img.shape

    bck = np.zeros((h + pad * 3 + barh, w + pad * 2, 3), dtype=np.uint8) + np.array([43, 34, 29], dtype=np.uint8)
    bck[pad:pad + h, pad:pad + w, :] = img

    bck = bar(bck, x_min=pad + barh // 2, y_min=2 * pad + h, h=barh, w=w - barh, color=(26, 20, 17))

    cx, cy = bck.shape[1] // 2, bck.shape[0] // 2
    barlen = int(round((w - barh) * perc))

    c = int(round(127 * 0.5 - perc * 127 * 0.5))
    c = np.array([[[c, int(round(255 * (90 / 100))), int(round(255 * (64 / 100)))]]], dtype=np.uint8)
    c = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
    c = tuple([int(i) for i in c[0, 0]])

    barpad = int(round(barh * 0.12))
    bck = bar(bck, x_min=(cx - barlen // 2), y_min=2 * pad + h + barpad, h=barh - 2 * barpad, w=barlen, color=c)

    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{perc * 100:.0f}%'

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # add text centered on image
    textX = round((cx - textsize[0]) / 2)
    textY = round((cy + textsize[1]) / 2)

    print((textX, textY))
    cv2.putText(bck, text, (cx - (textsize[0] // 2), 2 * pad + barh // 2 + h + (textsize[1] // 2)), font, 1,
                (255, 255, 255), 2)

    cv2.imshow('', bck)
    cv2.waitKey()
    print(img.shape)


if __name__ == '__main__':
    main()