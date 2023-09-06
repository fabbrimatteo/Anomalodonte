from typing import List
from typing import Tuple

import cv2
import numpy as np


# type alias
ArucoOut = Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]

# aruco corner refinement modes
CM0 = cv2.aruco.CORNER_REFINE_SUBPIX
CM1 = cv2.aruco.CORNER_REFINE_APRILTAG
CM2 = cv2.aruco.CORNER_REFINE_CONTOUR


def ir(x):
    # type: (float) -> int
    return int(round(x))


def detect_aruco4(img):
    # type: (np.ndarray) -> ArucoOut
    """
    :param img: image in with at least the following 4 aruco
        markers of the DICT_4X4_50 dictionary:
        - marker with ID=0
        - marker with ID=1
        - marker with ID=2
        - marker with ID=3
    :return: cv2.aruco.detectMarkers, but raises a runtime error
        if it does not find the 4 markers above
    """

    for scale in [1, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]:
        for cm in [CM0, CM1, CM2]:
            for atc in range(7, -20, -1):
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                aruco_params = cv2.aruco.DetectorParameters_create()
                aruco_params.adaptiveThreshConstant = atc
                aruco_params.cornerRefinementMethod = cm
                aruco_params.cornerRefinementMaxIterations = 128
                aruco_params.adaptiveThreshWinSizeMin = 3
                aruco_params.adaptiveThreshWinSizeMax = ir(73 * scale)
                aruco_params.adaptiveThreshWinSizeStep = ir(10 * scale)
                aruco_params.maxErroneousBitsInBorderRate = 0.35
                aruco_params.errorCorrectionRate = 1.0
                aruco_params.minMarkerPerimeterRate = 0.05
                aruco_params.maxMarkerPerimeterRate = 4
                aruco_params.polygonalApproxAccuracyRate = 0.1
                aruco_params.minCornerDistanceRate = 0.05

                img_grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                img_grey = cv2.resize(img_grey, (0, 0), fx=scale, fy=scale)

                corners, ids, rejected = cv2.aruco.detectMarkers(
                    img_grey, aruco_dict,
                    parameters=aruco_params
                )

                ok1 = len(corners) == 4
                ok2 = (0 in ids) and (1 in ids) and (2 in ids) and (3 in ids)

                # out = cv2.aruco.drawDetectedMarkers(img.copy(), rejected, np.zeros((len(rejected), 1)), borderColor=(255,255,0))
                # out = cv2.aruco.drawDetectedMarkers(out.copy(), corners, ids)
                # cv2.imshow('', out)
                # cv2.waitKey()

                corners = [c * (1 / scale) for c in corners]
                if ok1 and ok2:
                    return corners, ids, rejected

    raise RuntimeError('[!] unable to detect aruco markers!')


def read_board(img, n_rows=3, n_cols=3):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    :param img: image with color correction board (CCB)
    :param n_rows: number of rows of the CCB matrix
    :param n_cols: number of columns of the CCB matrix
    :return: array of detected colors (one for each CCB
        matrix element); shape: (n_rows*n_cols, 3)
    """

    square_side = 100

    # detect aruco markers
    corners, ids, rejected = detect_aruco4(img)

    # sort aruco markers based on their ID
    sort_idxs = np.argsort(ids[:, 0].tolist())
    corners = np.concatenate(corners, 0)
    corners = corners[sort_idxs]

    # define source points for the perspective transformation
    # ->> marker centers
    src_points = []
    for c in corners:
        c = np.round(c.mean(0))
        src_points.append(c)
    src_points = np.array(src_points, dtype=np.float32)

    # define destination points for the perspective transformation
    k = square_side // 2
    dst_points = np.array([
        [0 + k, 0 + k],
        [n_cols * square_side - k, 0 + k],
        [n_cols * square_side - k, n_rows * square_side - k],
        [0 + k, n_rows * square_side - k]
    ], dtype=np.float32)

    # apply perspective transformation
    m = cv2.getPerspectiveTransform(src_points, dst_points)
    out = cv2.warpPerspective(
        img, m, (n_cols * square_side, n_rows * square_side),
        flags=cv2.INTER_CUBIC
    )

    # read board colors (skipping aruco squares)
    aruco_indexes = [
        (0, 0), (n_rows - 1, 0),
        (n_rows - 1, n_cols - 1),
        (0, n_cols - 1)
    ]
    radius = 32
    colors = []
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in aruco_indexes:
                continue
            cx = k + col * square_side
            cy = k + row * square_side
            y0 = cy - radius
            y1 = cy + radius
            x0 = cx - radius
            x1 = cx + radius
            cut = out[y0:y1, x0:x1]
            color = cut.reshape((-1, 3)).mean(0)
            colors.append(color)

    return np.array(colors)


def bw_points_correction(img):
    # type: (np.ndarray) -> np.ndarray
    """
    :param img: image on which you want to apply the black
        and white points color correction; it must contain
        a fully-visible color calibration board (CCB)
    :return: color corrected image using black and white points
    """
    colors = read_board(img)
    w = colors[2]  # white
    b = colors[3]  # black
    img = img.copy().astype(float)
    for i in range(3):
        img[:, :, i] = ((img[:, :, i] - b[i]) * 255) / (w[i] - b[i])
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return img


def apply_color_correction(src_img, dst_img, with_bw_points=False):
    # type: (np.ndarray, np.ndarray, bool) -> np.ndarray
    """
    :param src_img: image on which you want to apply the
        color correction; it must contain a fully-visible
        color calibration board (CCB)
    :param dst_img: target image with respect to which you want
        to correct the source image `src_image`;  it must contain a
        fully-visible CCB
    :param with_bw_points: if `True`, applies white point and black
        point correction before performing full color correction
    :return: color corrected version of the source image `src_img`
    """
    if with_bw_points:
        src_img = bw_points_correction(src_img)

    # find source and destination colors
    src_colors = read_board(src_img)
    dst_colors = read_board(dst_img)

    # compute color correction matrix (CCM)
    h, w = src_img.shape[:2]
    ccm = np.linalg.lstsq(
        src_colors, dst_colors,
        rcond=None
    )[0]
    ccm = np.round(ccm, 9)

    # apply color correction to source image
    res = src_img.copy().astype(float)
    res = np.matmul(res.reshape((-1, 3)), ccm)
    res = res.reshape((h, w, 3))
    res = np.clip(res, 0, 255).astype(np.uint8)

    return res


def main():
    img = cv2.imread('C:\\Test\\20220905131631_601102626722100000868401_3OK.bmp')
    detect_aruco4(img)


if __name__ == '__main__':
    main()