from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from datetime import datetime
from shapely.geometry import MultiPoint
import shapely.geometry

f1 = open('Data/loсalization2_orb.txt', 'w')
f2 = open('Data/time2_orb.txt', 'w')
f3 = open('Data/features2_orb.txt', 'w')

parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='Photos1/photo0.jpg')
args = parser.parse_args()
img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
akaze = cv.ORB_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)

for i in range( 1, 100):

    path = 'Photos1/photo'+str(i)+'.jpg'
    start = datetime.now()
    img2 = cv.imread(path, cv.IMREAD_GRAYSCALE)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    matched1 = []
    matched2 = []
    good_matches = []
    nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            good_matches.append(cv.DMatch(len(matched1), len(matched2), m.distance))
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])


    f2.write((str(datetime.now() - start).split(sep=':')[2])+'\t'+str(img2.shape[0])+'\t'+str(img2.shape[0])+'\n')


    matched1_сonv = cv.KeyPoint_convert(matched1)
    matched2_conv = cv.KeyPoint_convert(matched2)
    localization = np.zeros(len(matched1))
    for i in range(0, len(matched1_сonv)):
        dist = np.zeros(len(matched1_сonv))
        for j in range(0, len(matched1_сonv)):
            dist[j] = np.linalg.norm(matched1_сonv[i] - matched1_сonv[j])
        indexes = dist.argsort()[:11]
        indexes = np.delete(indexes, [0])
        indicators = list()
        for k in indexes:
            dist2 = np.linalg.norm(matched2_conv[i] - matched2_conv[k])
            if dist[k] < dist2 * 0.7:
                indicators.append(1)
            else:
                indicators.append(0)
        S = sum(indicators)
        if S > 7:
            localization[i] = 1
            # плохие обозначила 1, хорошие нулями


    matched1_1 = []
    matched2_1 = []
    better_matches = []
    indexes = list()
    for i in range(0, len(matched1)):
        if localization[i] == 0:
            indexes.append(i)
    for i in indexes:
        better_matches.append(cv.DMatch(len(matched1_1), len(matched2_1), good_matches[i].distance))
        matched1_1.append(matched1[i])
        matched2_1.append(matched2[i])

    MIN_MATCH_COUNT = 8

    if len(better_matches) >= MIN_MATCH_COUNT:
        # src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        src_pts = np.float32([x.pt for x in matched1_1]).reshape(-1, 1, 2)
        dst_pts = np.float32([x.pt for x in matched2_1]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 6.0)
        if M is None:
            f1.write('findHomography error\n')
            f3.write('findHomography error\n')
        else:
            matchesMask = mask.ravel().tolist()

            pts = np.float32([[167, 290], [167, 1016], [719, 1016], [719, 290]]).reshape(-1, 1, 2)
            img1 = cv.polylines(img1, [np.int32(pts)], True, 255, 3, cv.LINE_AA)
            dst = cv.perspectiveTransform(pts, M)

            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)




            metrics1 = 0
            for i in range(0, len(better_matches)):
                metrics1 = metrics1 + better_matches[i].distance
            metrics1 = metrics1 / len(better_matches)
            f1.write(str(metrics1)+'\n')

            points = list()
            for i in range(0, 4):
                l = list()
                l.append(dst[i][0][0])
                l.append(dst[i][0][1])
                points.append(l)
            polygon = MultiPoint(points).convex_hull
            error1 = list()
            x2_1 = np.float32([x.pt for x in matched2_1])
            for i in range(0, len(matched2_1)):
                p = shapely.geometry.Point(x2_1[i])

                k = polygon.intersects(p)
                error1.append(k)
            metrics2 = sum(error1) / len(error1)

            f3.write(str(metrics2) + '\n')
    else:
        src_pts = np.float32([x.pt for x in matched1]).reshape(-1, 1, 2)
        dst_pts = np.float32([x.pt for x in matched2]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 6.0)
        if M is None:
            f1.write('findHomography error\n')
            f3.write('findHomography error\n')
        else:
            matchesMask = mask.ravel().tolist()

            pts = np.float32([[167, 290], [167, 1016], [719, 1016], [719, 290]]).reshape(-1, 1, 2)
            img1 = cv.polylines(img1, [np.int32(pts)], True, 255, 3, cv.LINE_AA)
            dst = cv.perspectiveTransform(pts, M)

            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

            metrics1 = 0
            for i in range(0, len(good_matches)):
                metrics1 = metrics1 + good_matches[i].distance
            metrics1 = metrics1 / len(good_matches)
            f1.write(str(metrics1) + '\n')

            points = list()
            for i in range(0, 4):
                l = list()
                l.append(dst[i][0][0])
                l.append(dst[i][0][1])
                points.append(l)
            polygon = MultiPoint(points).convex_hull
            error1 = list()
            x2_1 = np.float32([x.pt for x in matched2])
            for i in range(0, len(matched2)):
                p = shapely.geometry.Point(x2_1[i])

                k = polygon.intersects(p)
                error1.append(k)

            metrics2 = sum(error1) / len(error1)

            f3.write(str(metrics2) + '\n')


f1.close()
f2.close()
f3.close()
