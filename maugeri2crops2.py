
# import the necessary packages
import numpy as np
import cv2
from path import Path
from dataset.ds_utils import mpath2info

BOX_DICT = {
    'cam_1': [590//2, 142//2, 940//2, 668//2],
    'cam_2': [359, 148, 1053, 842],
    'cam_3': [493, 212, 1169, 888],
}



def align_images(image, template, maxFeatures=500, keepPercent=0.2,
                 debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    # resA = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # resB = orb.detectAndCompute(templateGray, None)

    kpsA_filtered = []
    descsA_filtered = []
    # filter out points
    for i in range(len(kpsA)):
        k = kpsA[i]
        d = descsA[i]
        if not (k.pt[0] > BOX_DICT['cam_1'][0] - 10 and  k.pt[0] < BOX_DICT['cam_1'][2] + 10 and k.pt[1] > BOX_DICT['cam_1'][1] -10 and  k.pt[1] < BOX_DICT['cam_1'][3] + 10):
            kpsA_filtered.append(k)
            descsA_filtered.append(d)

    kpsA = kpsA_filtered
    descsA = np.array(descsA_filtered)

    kpsB_filtered = []
    descsB_filtered = []
    # filter out points
    for i in range(len(kpsB)):
        k = kpsB[i]
        d = descsB[i]
        if not (k.pt[0] > BOX_DICT['cam_1'][0] - 10 and  k.pt[0] < BOX_DICT['cam_1'][2] + 10 and k.pt[1] > BOX_DICT['cam_1'][1] -10 and  k.pt[1] < BOX_DICT['cam_1'][3] + 10):
            kpsB_filtered.append(k)
            descsB_filtered.append(d)

    kpsB = kpsB_filtered
    descsB = np.array(descsB_filtered)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                                     matches, None)
        matchedVis = cv2.resize(matchedVis, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt


    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


BOX_DICT = {
    'cam_1': [590, 142, 940, 668],
    'cam_2': [359, 148, 1053, 842],
    'cam_3': [493, 212, 1169, 888],
}

source_img_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS/04/12/20220412085004_601102626722200000327101_1.bmp')
source_img = cv2.imread(source_img_path)

x_min, y_min, x_max, y_max = BOX_DICT[f'cam_1']
source_img = cv2.rectangle(source_img, (x_min, y_min), (x_max, y_max), (0,0,0), -1)
source_img = cv2.resize(source_img, (0,0), fx=0.5, fy=0.5)

data_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/DS')
out_path = Path('/goat-nas/Datasets/spal/raw_dataset_aprile_maggio_giugno/spal_cuts_aligned')

for month_dir in data_path.dirs():
    for day_dir in month_dir.dirs():
        for file_name in day_dir.files():
            info = mpath2info(file_name)
            if info["camera-id"] == 1:
                img = cv2.imread(file_name)

                x_min, y_min, x_max, y_max = BOX_DICT[f'cam_{info["camera-id"]}']
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,0), -1)
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                img = align_images(img, source_img, debug=True)
                cv2.imshow('1', img)
                cv2.imshow('2', source_img)
                cv2.waitKey()