#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier( '/Users/kpickrell/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml' )
img = cv2.imread( '/Users/kpickrell/research/image/test_images/items/brightsvsneutrals/brights14.jpg' )
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
faces = face_cascade.detectMultiScale( gray, 1.3, 5 )
faces
for (x,y,w,h) in faces:
    cv2.rectangle( img, (x,y), (x+w,y+h), (255,0,0), 2 )
    
print( 'press space bar to close...' )
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
