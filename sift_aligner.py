from typing import Tuple

import cv2
import numpy as np


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
