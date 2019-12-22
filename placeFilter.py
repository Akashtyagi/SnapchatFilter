#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:03:39 2019

@author: Akash Tyagi
"""


import numpy as np
from keras import models
import cv2
import matplotlib.pyplot as plt
# Load the trained model
model = models.load_model('akash.h5')
# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



camera = cv2.VideoCapture(0)
while True:
    grab_trueorfalse, img = camera.read()       # Read data from the webcam
    
    # Preprocess input fram webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Convert RGB data to Grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # Identify faces in the webcam
    
    # For each detected face using tha Haar cascade
    for (x,y,w,h) in faces:
        face = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        img_copy_1 = np.copy(img)
        roi_color = img_copy_1[y:y+h, x:x+w]
        
        img_gray = cv2.resize(roi_gray, (96, 96))       # Resize image to size 96x96
        img_gray2 = np.copy(img_gray)
        img_model = np.reshape(img_gray, (1,96,96,1))   # Model takes input of shape = [batch_size, height, width, no. of channels]
        
        keypoints = model.predict(img_model)[0]
        
        # Plot model predicted points over the face
        for i in range(1,31,2):
              print(i,keypoints[i-1], keypoints[i])
              cv2.circle(img_gray2,(keypoints[i-1], keypoints[i]),3,(255,0,0),-1)
        
        left_lip_coords = (int(keypoints[24]), int(keypoints[25])) # left from our view
        right_lip_coords = (int(keypoints[22]), int(keypoints[23]))
        top_lip_coords = (int(keypoints[26]), int(keypoints[27]))
        bottom_lip_coords = (int(keypoints[28]), int(keypoints[29]))
        left_eye_coords = (int(keypoints[10]), int(keypoints[11]))
        right_eye_coords = (int(keypoints[6]), int(keypoints[7]))
        left_brow_coords = (int(keypoints[18]), int(keypoints[19]))
        right_brow_coords = (int(keypoints[14]), int(keypoints[15]))
        
# =============================================================================
#         Filter
# =============================================================================
        glasses = cv2.imread('filters/glass1.png', -1)
        # Create the mask for the glasses
        
        glasses_width = right_eye_coords[0] - left_eye_coords[0]
        glasses_height = int(glasses_width*0.45) # (actual image height/width=0.45)
        glasses = cv2.resize(glasses,(glasses_width,glasses_height))
                
        cv2.rectangle(img_gray,(left_brow_coords[0],left_eye_coords[0]+int(glasses_height/2)),(right_brow_coords[0],left_eye_coords[1]+int(glasses_height/2)),(255,0,0),2)
        
    cv2.imshow('ML face predictions',img_gray2)
    cv2.imshow('Glasses positioning',img_gray)
    cv2.imshow('Face',roi_color)
    cv2.imshow('Webcam',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()





#import random
#import matplotlib.pyplot as plt
#for _ in range(1):
#  n = 0
#  xv = img_model.reshape((96,96))
#  plt.imshow(xv)
#  
#  for i in range(1,31,2):
#      print(i,keypoints[i-1], keypoints[i])
#      plt.plot(keypoints[i-1], keypoints[i], 'x',color='white')
##     plt.plot(y_train[n][i-1], y_train[n][i], 'x', color='green')
#  plt.show()

            



