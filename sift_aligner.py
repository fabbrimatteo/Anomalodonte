import random
from time import time
from typing import Tuple

import cv2
import numpy as np
from path import Path

from pre_processing import CropThenResize


class SiftAligner(object):

    def __init__(self, target_img, min_match_count=10):
        # type: (np.ndarray, int) -> None
        """
        :param target_img: objective image with respect to which
            you want to make the alignment
        :param min_match_count: minimum number of keypoint matching;
            if not enough matches are found, the alignment is not carried out
        """

        self.min_match_count = min_match_count

        # init sift detector and matcher
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()

        # pre-calculates the keypoints of the target image
        k, d = self.detector.detectAndCompute(target_img, None)
        self.target_kpts = k
        self.target_des = d


    def align(self, img):
        # type: (np.ndarray) -> Tuple[bool, np.ndarray]

        # find the keypoints and descriptors with SIFT
        img_kpts, img_des = self.detector.detectAndCompute(img, None)

        # $> each keypoint of the first image is matched
        #    with a number of keypoints from the second image.
        # $> `knnMatch` keeps the best (`k=2`) best matches for each keypoint
        #    (best matches = the ones with the smallest distance measurement).
        # $>  `matches` if a list of keypoint matches,
        #     where each element is a tuple containing the best match
        #     and the second best match
        matches = self.bf_matcher.knnMatch(img_des, self.target_des, k=2)

        # filter the `matches` list, keeping only those where the `best_match`
        # is significantly better than the` second_best_match`.
        # ==> keep only the strongest matches
        good_matches = []
        for best_match, second_best_match in matches:
            if best_match.distance < 0.75 * second_best_match.distance:
                good_matches.append(best_match)

        # if you have found enough matches, calculate the homography matrix
        # and apply to the input image (success), otherwise keep the input
        # image unchanged (failure)
        if len(good_matches) >= self.min_match_count:
            success = True

            # >> `src_pts` -> points on input image
            src_pts = [img_kpts[m.queryIdx].pt for m in good_matches]
            src_pts = np.array(src_pts, dtype=float).reshape((-1, 1, 2))
            # >> `dst_pts` -> points on target image
            dst_pts = [self.target_kpts[m.trainIdx].pt for m in good_matches]
            dst_pts = np.array(dst_pts, dtype=float).reshape((-1, 1, 2))

            h_matrix, _ = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0
            )

            aligned_img = cv2.warpPerspective(
                img, h_matrix, (img.shape[1], img.shape[0])
            )

        else:
            success = False
            aligned_img = img

        return success, aligned_img


def realign_and_cut(in_dir, out_dir):
    in_dir = Path(in_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir_p()
    (out_dir / 'train').mkdir_p()
    (out_dir / 'test').mkdir_p()

    target = cv2.imread('debug/debug_img_00.png')

    sift_aligner = SiftAligner(target_img=target)

    tr = CropThenResize(
        resized_h=256, resized_w=256, crop_x_min=812,
        crop_y_min=660, crop_side=315
    )

    all_files = list((in_dir / 'test').files())
    all_files += list((in_dir / 'train').files())
    random.shuffle(all_files)

    for i, img_path in enumerate(all_files):
        if len(out_dir.files(str(img_path.basename()))) == 0:
            t0 = time()
            img = cv2.imread(img_path)
            # _, aligned_img = sift_aligner.align(img)

            mode = img_path.parent.basename()
            name = img_path.basename()
            print('WRITE', out_dir / mode / name, f'{time() - t0:.2f}s')
            cv2.imwrite(out_dir / mode / name, img)
        else:
            mode = img_path.parent.basename()
            name = img_path.basename()
            print('SKIP', out_dir / mode / name)


def cut(in_dir, out_dir):
    in_dir = Path(in_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir_p()
    (out_dir / 'train').mkdir_p()
    (out_dir / 'test').mkdir_p()

    target = cv2.imread('debug/debug_img_00.png')

    sift_aligner = SiftAligner(target_img=target)

    tr = CropThenResize(
        resized_h=256, resized_w=256, crop_x_min=812,
        crop_y_min=660, crop_side=315
    )

    all_files = list((in_dir / 'test').files())
    all_files += list((in_dir / 'train').files())
    random.shuffle(all_files)

    for i, img_path in enumerate(all_files):
        if len(out_dir.files(str(img_path.basename()))) == 0:
            t0 = time()
            img = cv2.imread(img_path)
            mode = img_path.parent.basename()
            name = img_path.basename()
            print('WRITE', out_dir / mode / name, f'{time() - t0:.2f}s')
            cv2.imwrite(out_dir / mode / name, tr(img))
        else:
            mode = img_path.parent.basename()
            name = img_path.basename()
            print('SKIP', out_dir / mode / name)


if __name__ == '__main__':
    DS_ROOT = Path('/nas/softechict-nas-3/rgasparini/datasets/spal/data_cavi')
    cut(
        in_dir='/nas/softechict-nas-3/matteo/Datasets/Spal/cables_6mm',
        out_dir='/nas/softechict-nas-3/matteo/Datasets/Spal/cables_6mm_p1'
    )
