from typing import Tuple

import cv2
import numpy as np
import torch


Color = Tuple[int, int, int]


def __draw_anomaly_bar(img, x_min, y_min, h, w, color):
    # type: (np.ndarray, int, int, int, int, Color) -> np.ndarray

    # draw actual bar (rectangle)
    img = cv2.rectangle(
        img, (x_min, y_min), (x_min + w, y_min + h),
        color=color, thickness=-1, lineType=cv2.LINE_AA
    )

    # draw left end (circle) of the bar
    img = cv2.circle(
        img, center=(x_min, y_min + h // 2), radius=(h // 2),
        color=color, thickness=-1, lineType=cv2.LINE_AA
    )

    # draw right end (circle) of the bar
    img = cv2.circle(
        img, center=(x_min + w, y_min + h // 2), radius=(h // 2),
        color=color, thickness=-1, lineType=cv2.LINE_AA
    )

    return img


def __get_color(anomaly_score, min_red=80, max_green=20):
    # type: (float, int, int) -> Color
    space = (min_red - max_green)

    if anomaly_score < max_green:
        mul = 0.
    elif anomaly_score > min_red:
        mul = 1.
    else:
        mul = (anomaly_score - max_green) / space

    h_value = int(round(62 - mul * 62))
    hsv_color = np.array([[[h_value, 199, 214]]], dtype=np.uint8)

    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    bgr_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

    return bgr_color


def draw_anomaly_ui(img, anomaly_score, header=''):
    # type: (np.ndarray, float, str) -> np.ndarray
    """
    :param img: image on which you want to draw the UI
    :param anomaly_score: anomaly score associated with `img`
    :param header: text that goes in the top left corner of the image
    :return: image with UI:
        >> header text in the top left corner
        >> anomaly bar (with anomaly perc) at the bottom
    """
    anomaly_prob = min(anomaly_score, 100) / 100
    pad = 32
    bar_h = 64

    h, w, _ = img.shape
    scale_factor = 540 / w
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    h, w, _ = img.shape

    # create background
    bck = np.zeros((h + pad * 3 + bar_h, w + pad * 2, 3), dtype=np.uint8)
    bck = bck + np.array([43, 34, 29], dtype=np.uint8)

    # put the image on the background
    bck[pad:pad + h, pad:pad + w, :] = img

    # draw the background of the anomaly bar
    bck = __draw_anomaly_bar(
        bck, x_min=pad + bar_h // 2, y_min=2 * pad + h,
        h=bar_h, w=w - bar_h, color=(26, 20, 17)
    )

    # define bar length and color based on the given anomaly perc
    bar_len = max(int(round((w - bar_h) * anomaly_prob)), (w - bar_h) // 8)
    bar_col = __get_color(anomaly_prob * 100)

    # draw the actual anomaly bar
    bar_pad = int(round(bar_h * 0.12))
    cx, cy = bck.shape[1] // 2, bck.shape[0] // 2
    bck = __draw_anomaly_bar(
        bck, x_min=(cx - bar_len // 2), y_min=2 * pad + h + bar_pad,
        h=bar_h - 2 * bar_pad, w=bar_len, color=bar_col
    )

    # define anomaly perc label
    font = cv2.FONT_HERSHEY_SIMPLEX
    perc_label = f'{anomaly_prob * 100:.0f}%'
    text_size_wh = cv2.getTextSize(perc_label, font, 1, 2)[0]

    # write perc label in the center of the anomaly bar
    dx = text_size_wh[0] // 2
    dy = text_size_wh[1] // 2
    cv2.putText(
        img=bck, text=perc_label,
        org=(cx - dx, 2 * pad + bar_h // 2 + h + dy),
        fontFace=font, fontScale=1,
        color=(255, 255, 255), thickness=2,
        lineType=cv2.LINE_AA
    )

    # write header text
    # --- (1) dark outline
    cv2.putText(
        img=bck, text=header,
        org=(int(round(1.25 * pad)), 2 * pad),
        fontFace=font, fontScale=0.75,
        color=(0, 0, 0), thickness=6,
        lineType=cv2.LINE_AA
    )
    # --- (2) white text
    cv2.putText(
        img=bck, text=header,
        org=(int(round(1.25 * pad)), 2 * pad),
        fontFace=font, fontScale=0.75,
        color=(255, 255, 255), thickness=2,
        lineType=cv2.LINE_AA
    )

    return bck


def code2img(code, side=None):
    # type: (torch.Tensor, int) -> np.ndarray

    if len(code.shape) == 4 and code.shape[0] == 1:
        code = code[0]

    # shape=(H, W, C) and values in [-1, 1]
    code = code.cpu().numpy().transpose((1, 2, 0))

    # values in [0, 255]
    code = (0.5 * (code + 1)) * 255
    code = code.astype(np.uint8)

    h, w, nc = code.shape
    codes = []
    out = None
    for a in range(0, nc, 3):
        code_chunk = code[:, :, a:a + 3]
        cc = code_chunk.shape[-1]
        if cc == 1:
            code_chunk = np.concatenate([
                code_chunk, code_chunk, code_chunk
            ], -1)
        elif cc == 2:
            z = np.zeros((code_chunk.shape[0], code_chunk.shape[1], 1))
            code_chunk = np.concatenate([code_chunk, z], -1)

        out = code_chunk if out is None else np.hstack((out, code_chunk))

        codes.append(code_chunk)

    if side is not None:
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_NEAREST)

    return out
