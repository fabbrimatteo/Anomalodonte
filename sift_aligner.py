import numpy as np
import cv2


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


if __name__ == '__main__':
    # img1 = cv2.imread('1/20210920173740_601102626521200000037704_1.bmp')  # queryImage
    img1 = cv2.imread('2/good_0100.png')  # queryImage
    # img2 = cv2.imread('1/20210920173902_601102626521200000037701_1.bmp')  # trainImage
    img2 = cv2.imread('2/good_0156.png')  # trainImage

    sift_aligner = SiftAligner(target_img=img2)
    res, out = sift_aligner.align(img1)

    if res:
        cv2.imshow("out", out)
        cv2.waitKey()
        cv2.imwrite('2/out.png', out)
    else:
        print("error no match found")
