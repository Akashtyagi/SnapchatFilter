#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:03:45 2019

@author: Akash Tyagi
"""


import cv2
import dlib
import numpy as np
from keras import models

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('/home/qainfotech/AkashTyagi/Pythonn/FaceDetection/haarcascade_frontalface_default.xml')
model = models.load_model('/home/qainfotech/AkashTyagi/Pythonn/FaceDetection/akash.h5')

camera = cv2.VideoCapture(0)

while True:
    grab_trueorfalse, img = camera.read()       # Read data from the webcam
    
    # Preprocess input fram webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Convert RGB data to Grayscale
#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # Identify faces in the webcam
    dlibfaces = detector(gray)
    facesHaarcascade = face_cascade.detectMultiScale(gray, 1.3, 5)     # Identify faces in the webcam
    img_copy = np.copy(img)
    # For each detected face using DLIB
    for face in dlibfaces:
        x = face.left()
        y = face.top() 
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        roi_gray = gray[y:y+h, x:x+w]
        img_copy = np.copy(img)
        
        dlibface = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # Marking rectangle around face.
        
        img_gray = cv2.resize(roi_gray, (96, 96))      
        img_model = np.reshape(img_gray, (1,96,96,1))   # Model takes input of shape = [batch_size, height, width, no. of channels]
        keypoints = model.predict(img_model)[0] # Predicting face point on Dlib cropped face.
        
        # Marking predicted points on Dlib recognised face.
        for i in range(1,31,2):
            cv2.circle(img_gray,(keypoints[i-1], keypoints[i]),3,(255,0,0),-1)
        
        # Marking Dlib recognised face points on Dlib recognised face.
        landmarks = predictor(gray, face)
        for i in range(1,68):
            cv2.circle(img,(landmarks.part(i-1).x, landmarks.part(i).y),3,(255,0,0),-1)
    
    # Finding faces using HaarCasscade
    for (x,y,w,h) in facesHaarcascade:
        face = cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray2 = gray[y:y+h, x:x+w]
        img_gray2 = cv2.resize(roi_gray2, (96, 96))      
        img_model2 = np.reshape(img_gray2, (1,96,96,1))   # Model takes input of shape = [batch_size, height, width, no. of channels]
        keypoints2 = model.predict(img_model2)[0]
    
        for i in range(1,31,2):
    #            plt.plot(keyp[i-1], keyp[i], 'ro')
            cv2.circle(img_gray2,(keypoints2[i-1], keypoints2[i]),3,(255,0,0),-1)
                
    cv2.imshow('Model Prediction on Dlib Face',img_gray)
    cv2.imshow("Model Prediction on HarCasscade Face",img_gray2)
    cv2.imshow('DlibCamera',img)    
    cv2.imshow('Harcascade Camera',img_copy)    
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()