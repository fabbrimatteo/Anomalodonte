import numpy as np
import cv2
from path import Path
from pre_processing import CropThenResize
import random
from time import time

OUT_DIR = Path('dataset/spal_fake_rect2')


class SiftAligner(object):

    def __init__(self, target_img, min_match_count=10):

        self.min_match_count = min_match_count

        # Initiate SIFT detector
        self.detector = cv2.xfeatures2d.SIFT_create()

        # create BFMatcher object
        self.bf = cv2.BFMatcher()

        self.target_kpts, self.target_des = self.detector.detectAndCompute(target_img, None)


    def align(self, img):
        good_end = False

        # find the keypoints and descriptors with SIFT
        img_kpts, img_des = self.detector.detectAndCompute(img, None)

        # Match descriptors.
        matches = self.bf.knnMatch(img_des, self.target_des, k=2)

        # Apply ratio test
        good = []
        good_list = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_list.append([m, n])
                good.append(m)

        if len(good) >= self.min_match_count:
            src_pts = np.float32([img_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.target_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            img_out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
            good_end = True
        else:
            img_out = img

        return good_end, img_out


def realign_and_cut(mode='train'):
    target = cv2.imread('debug/debug_img_00.png')


    sift_aligner = SiftAligner(target_img=target)

    tr = CropThenResize(resized_h=256, resized_w=256, crop_x_min=812, crop_y_min=660, crop_side=315)

    t2 = target.copy()
    t2 = cv2.rectangle(t2, (812, 660), (816+315, 660+315), color=(52//2, 235//2, 128//2), thickness=9)
    t2 = cv2.rectangle(t2, (812, 660), (816+315, 660+315), color=(52, 235, 128), thickness=7)

    cv2.imwrite(f'report/reference_guide.jpg', t2)

    cv2.imwrite(f'report/reference.jpg', target)
    cv2.imwrite(f'report/reference_cut.jpg', tr.apply(target))



    in_dir = Path(f'dataset/spal_fake/{mode}')
    out_dir = OUT_DIR / mode

    all_files = list(in_dir.files())
    random.shuffle(all_files)

    for i, img_path in enumerate(all_files):
        if len(out_dir.files(str(img_path.basename()))) == 0:
            t0 = time()
            img = cv2.imread(img_path)
            cv2.imwrite(f'report/{i:03d}_original.jpg', img)
            _, aligned_img = sift_aligner.align(img)
            cv2.imwrite(f'report/{i:03d}_aligned.jpg', img)
            print('W', out_dir / img_path.basename(), f'{time()-t0:.2f}s')
            cv2.imwrite(f'report/{i:03d}_original_cut.jpg', tr.apply(img))
            cv2.imwrite(f'report/{i:03d}_aligned.jpg', tr.apply(aligned_img))
            #cv2.imwrite(out_dir / img_path.basename(), tr.apply(aligned_img))
        else:
            print('S', out_dir / img_path.basename())


if __name__ == '__main__':
    realign_and_cut(mode='test')
