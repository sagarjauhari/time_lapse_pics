# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:34:30 2013

@author: sagar
"""
import cv2
import numpy as np
import itertools
import sys
from os.path import join

MIN_MATCH_COUNT = 10

class TimeLapse:
    def find_keypoints(self, img1, img2):
        sift = cv2.SIFT()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        return kp1, kp2, good
    
    def homography(self, img1, img2):
        kp1, kp2, good = self.find_keypoints(img1, img2)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        
            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            
            
            img3 =cv2.warpPerspective(img2, M, (w,h))
            cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
            
            imgs = np.hstack((img2,img3,img1))

            cv2.imshow('imgc',imgs)
            
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

img1 = cv2.imread(join('images','rohit_1.jpg'))
img2 = cv2.imread(join('images','rohit_2.jpg'))
"""
im = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
surfDetector = cv2.FeatureDetector_create("SURF")
surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
keypoints = surfDetector.detect(im)
(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)
"""
tl = TimeLapse()
tl.homography(img1, img2)