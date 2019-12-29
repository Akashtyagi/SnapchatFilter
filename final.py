#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:29:35 2019

@author: Akash Tyagi
"""

import cv2
import dlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

glass_img = cv2.imread('filters/glass1.png', -1)
mustache_img = cv2.imread('filters/mustache1.png', -1)
blunt_img = cv2.imread('filters/blunt.png', -1)


def apply_filter(x1,x2,y1,y2,frame,frame_height,frame_width,filterimg,orig_filter_mask,orig_filter_mask_inv):
    ''' Add filter to the frame.'''
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > frame_width:
        x2 = frame_width
    if y2 > frame_height:
        y2 = frame_height
    if x1==x2:
        if y1>y2:
            x2=0
        else:
            x1=0
    if y1==y2:
        if x1>x2:
            y2=0
        else:
            y1=0
        
    flag=True
    glasses_height = y2-y1
    glasses_width = x2-x1
    if glasses_height<0 or glasses_width<0:
        flag = False
    roi1 = frame[y1:y2, x1:x2]
#    print("y1:y2,x1:x2",y1,y2,x1,x2)
#    cv2.imshow('x',roi1)
    
    if flag:
        glass = cv2.resize(filterimg, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(orig_filter_mask, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_filter_mask_inv, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
        roi_bg = cv2.bitwise_and(roi1,roi1,mask = mask_inv)
        roi_fg = cv2.bitwise_and(glass,glass,mask = mask)
        frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)
    return frame


camera = cv2.VideoCapture(0)
#frame = cv2.imread('/home/qainfotech/Downloads/IMG_3284.JPG')

while True:
    grab_trueorfalse, frame = camera.read()       # Read data from the webcam
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    img = np.copy(frame)
    frame_height,frame_width = frame.shape[0],frame.shape[0]

    
    for face in faces:
        landmarks = predictor(frame, face)
        
        # Visualize all predicted frames on frame
#        for i in range(1,68):
#            cv2.circle(frame,(landmarks.part(i-1).x, landmarks.part(i).y),3,(255,0,0),-1)
    
# =============================================================================
#         # Specs Filter
# =============================================================================
            
        # Mask of Glasses
        orig_mask_g = glass_img[:,:,3]
        orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
        imgGlass = glass_img[:,:,0:3]
        
        glasses_width = landmarks.part(46).x - landmarks.part(37).x     # Right eye and left eye corner
        glasses_height = int(landmarks.part(30).y-(landmarks.part(22).y+landmarks.part(23).y)/2) # (actual glasses height/width=0.45)
        
        
        y1 = int((landmarks.part(22).y+landmarks.part(23).y)/2) # Centre of left and right brow  
        y2 = int(landmarks.part(30).y) # lower nose
        x1 = int(landmarks.part(37).x - glasses_width*0.30)
        x2 = int(landmarks.part(46).x + glasses_width*0.30)
        
        frame = apply_filter(x1,x2,y1,y2,frame,frame_height,frame_width,imgGlass,orig_mask_g,orig_mask_inv_g)        
        
# =============================================================================
#         # mustache Filter
# =============================================================================
        orig_mustache_mask = mustache_img[:,:,3]
        orig_mustache_mask_inv = cv2.bitwise_not(orig_mustache_mask)
        imgMustache = mustache_img[:,:,0:3]
        
        origMustacheHeight = mustache_img.shape[0]
        origMustacheWidth = mustache_img.shape[1]
        
        mustacheWidth = abs(3 * (landmarks.part(31).x - landmarks.part(35).x))
        mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
        
        y1 = int(landmarks.part(33).y - (mustacheHeight/2)) + 10
        y2 = int(y1 + mustacheHeight)
        x1 = int(landmarks.part(51).x - (mustacheWidth/2))
        x2 = int(x1 + mustacheWidth)

#        frame = apply_filter(x1,x2,y1,y2,frame,frame_height,frame_width,imgMustache,orig_mustache_mask,orig_mustache_mask_inv)        
        
# =============================================================================
#         # Blunt Filter
# =============================================================================
        orig_blunt_mask = blunt_img[:,:,3]
        orig_blunt_mask_inv = cv2.bitwise_not(orig_blunt_mask)
        imgBlunt = blunt_img[:,:,0:3]
        
        origBluntHeight = blunt_img.shape[0]
        origBluntWidth = blunt_img.shape[1]
        
        bluntWidth = abs(int(landmarks.part(66).x - landmarks.part(65).x)*3)
        bluntHeight = abs(int((landmarks.part(66).x-landmarks.part(11).x)))
        
#        y1 = int((landmarks.part(66).y+landmarks.part(64).y)/2)
        y1 = int(landmarks.part(53).y)
        y2 = int(y1+bluntHeight)
        x1 = int(landmarks.part(65).x)
        x2 = int(x1+bluntWidth)
        
        frame = apply_filter(x1,x2,y1,y2,frame,frame_height,frame_width,imgBlunt,orig_blunt_mask,orig_blunt_mask_inv)        
#        if x2-x1==0:
#            pass
            
    cv2.imshow('Webcam',frame)
#    cv2.imshow("GlassesArea",orig_blunt_mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()

#import random
#import matplotlib.pyplot as plt
#for _ in range(1):
#  n = 0
#  plt.imshow(frame)
#  
#  for i in range(1,68):
##      print(keypoints[i-1], keypoints[i])
#      plt.plot(landmarks.part(i-1).x, landmarks.part(i).y, 'ro')
#    # plt.plot(y_train[n][i-1], y_train[n][i], 'x', color='green')
#
#  plt.show()    