# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:41:12 2019

@author: napra
"""

import numpy as np
import cv2
cap = cv2.VideoCapture('C:/Users/napra/imagpro/tofu/tofu/dumpro/Grayscale.avi')

back = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    movie = back.apply(frame)
    
    cv2.imshow('frame',movie)
    #cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
