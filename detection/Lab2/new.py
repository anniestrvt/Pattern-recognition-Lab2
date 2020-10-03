from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from shapely.geometry import MultiPoint
import shapely.geometry
import scipy
from datetime import datetime
from matplotlib import pyplot as plt

import math
#parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
#parser.add_argument('--input1', help='Path to input image 1.', default='Photos/photo0.jpg')
#parser.add_argument('--input2', help='Path to input image 2.', default='Photos/photo1.jpg')
#args = parser.parse_args()

img1 = cv.imread('Photos/photo0.jpg',0)          # queryImage
img2 = cv.imread('Photos/photo1.jpg',0)
akaze = cv.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)
matched1 = []
matched2 = []
good_matches = []
nn_match_ratio = 0.8 # Nearest neighbor matching ratio
for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        good_matches.append(cv.DMatch(len(matched1), len(matched2), m.distance))
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])


matched1_сonv = cv.KeyPoint_convert(matched1)
matched2_conv = cv.KeyPoint_convert(matched2)
localization = np.zeros(len(matched1))
for i in range(0, len(matched1_сonv)):
    dist = np.zeros(len(matched1_сonv))
    for j in range(0, len(matched1_сonv)):
        dist[j] = np.linalg.norm(matched1_сonv[i]-matched1_сonv[j])
    indexes = dist.argsort()[:4]
    indexes = np.delete(indexes, [0])
    indicators =list()
    for k in indexes:
        dist2 = np.linalg.norm(matched2_conv[i]-matched2_conv[k])
        if dist[k]<dist2*0.7:
            indicators.append(1)
        else:
            indicators.append(0)
    S = sum(indicators)
    if S>2:
        localization[i] = 1
        #плохие обозначила 1, хорошие нулями

#S = sum(localization)
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


#res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
#cv.drawMatches(img1, matched1_1, img2, matched2_1, better_matches, res)
#res = cv.resize(res, (1200,800))
#cv.imshow('result', res)
#cv.waitKey()

# до сюда все нормально, убрали лишнее



MIN_MATCH_COUNT = 8

if len(better_matches)>=MIN_MATCH_COUNT:
    #src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    #dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    src_pts = np.float32([x.pt for x in matched1_1]).reshape(-1, 1, 2)
    dst_pts = np.float32([x.pt for  x in matched2_1]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts,cv.RANSAC ,5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([ [167,290],[167,1016],[719,1016],[719,290] ]).reshape(-1,1,2)
    img1 = cv.polylines(img1, [np.int32(pts)], True, 255, 3, cv.LINE_AA)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    points = list()

    for i in range(0, 4):
        l = list()
        l.append(dst[i][0][0])
        l.append(dst[i][0][1])
        points.append(l)

    polygon = MultiPoint(points).convex_hull
    error = list()
    x2 = np.float32([x.pt for x in matched2])
    for i in range(0, len(matched2)):
        p = shapely.geometry.Point(x2[i])

        k = polygon.intersects(p)
        error.append(k)
    metrics1 = sum(error) / len(error)
    metrics2 = 0
    for i in range(0, len(better_matches)):
        metrics2 = metrics2 + better_matches[i].distance
    metrics2 = metrics2/len(better_matches)
    print(metrics2)


else:
    src_pts = np.float32([x.pt for x in matched1]).reshape(-1, 1, 2)
    dst_pts = np.float32([x.pt for x in matched2]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 6.0)

    matchesMask = mask.ravel().tolist()

    pts = np.float32([[167, 290], [167, 1016], [719, 1016], [719, 290]]).reshape(-1, 1, 2)
    img1 = cv.polylines(img1, [np.int32(pts)], True, 255, 3, cv.LINE_AA)
    dst = cv.perspectiveTransform(pts, M)

    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    points = list()

    for i in range(0, 4):
        l = list()
        l.append(dst[i][0][0])
        l.append(dst[i][0][1])
        points.append(l)

    polygon = MultiPoint(points).convex_hull

    results = list()
    error = list()
    x2 = np.float32([x.pt for x in matched2])
    for i in range(0, len(matched2)):
        p = shapely.geometry.Point(x2[i])
        print(p)
        k = polygon.intersects(p)
        error.append(k)

    metrics1 = sum(error) / len(error)

    print(metrics1)
    metrics2 = 0
    for i in range(0, len(good_matches)):
        metrics2 = metrics2 + good_matches[i].distance
    metrics2 = metrics2 / len(good_matches)
    print(metrics2)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask =None,#matchesMask, # draw only inliers
                   flags = 2)

res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, matched1_1, img2, matched2_1, better_matches, res,  **draw_params)
res = cv.resize(res, (1200,800))
cv.imwrite("sample2.jpg", res)
cv.imshow('result', res)
cv.waitKey()





