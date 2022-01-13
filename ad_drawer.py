import cv2
import numpy as np


def bar(img, x_min, y_min, h, w, color):
    img = cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color=color, thickness=-1)
    img = cv2.circle(img, center=(x_min, y_min + h // 2), radius=(h // 2), color=color, thickness=-1)
    img = cv2.circle(img, center=(x_min + w, y_min + h // 2), radius=(h // 2), color=color, thickness=-1)
    return img


def show_anomaly(img, perc, label=''):
    pad = 32
    barh = 64

    h, w, _ = img.shape
    scale_factor = 540 / w
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    h, w, _ = img.shape

    bck = np.zeros((h + pad * 3 + barh, w + pad * 2, 3), dtype=np.uint8) + np.array([43, 34, 29], dtype=np.uint8)
    bck[pad:pad + h, pad:pad + w, :] = img

    bck = bar(bck, x_min=pad + barh // 2, y_min=2 * pad + h, h=barh, w=w - barh, color=(26, 20, 17))

    cx, cy = bck.shape[1] // 2, bck.shape[0] // 2
    barlen = max(int(round((w - barh) * perc)), (w - barh) // 8)

    c = get_color(perc * 100)

    barpad = int(round(barh * 0.12))
    bck = bar(bck, x_min=(cx - barlen // 2), y_min=2 * pad + h + barpad, h=barh - 2 * barpad, w=barlen, color=c)

    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{perc * 100:.0f}%'

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # add text centered on image
    dx = textsize[0] // 2
    dy = textsize[1] // 2
    cv2.putText(
        img=bck, text=text,
        org=(cx - dx, 2 * pad + barh // 2 + h + dy),
        fontFace=font, fontScale=1,
        color=(255, 255, 255), thickness=2
    )

    cv2.putText(
        img=bck, text=label,
        org=(int(round(1.25 * pad)), 2 * pad),
        fontFace=font, fontScale=0.75,
        color=(0, 0, 0), thickness=6
    )

    cv2.putText(
        img=bck, text=label,
        org=(int(round(1.25 * pad)), 2 * pad),
        fontFace=font, fontScale=0.75,
        color=(255, 255, 255), thickness=2
    )

    cv2.imshow(label, bck)
    cv2.waitKey()
    cv2.destroyWindow(label)


def get_color(perc, min_red=80, max_green=20):
    space = (min_red - max_green)

    if perc < max_green:
        mul = 0.
    elif perc > min_red:
        mul = 1.
    else:
        mul = (perc - max_green) / space

    h_value = int(round(62 - mul * 62))
    hsv_color = np.array([[[h_value, 199, 214]]], dtype=np.uint8)

    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    bgr_color = tuple([int(i) for i in bgr_color[0, 0]])

    return bgr_color


if __name__ == '__main__':

    for i in range(0, 101):

        c = get_color(i)

        img = np.ones((256, 256, 3), dtype=np.uint8)
        for j in range(3):
            img[:, :, j] *= c[j]

        print(i)
        cv2.imshow('', img)
        cv2.waitKey()
