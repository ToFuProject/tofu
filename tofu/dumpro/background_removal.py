# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:41:12 2019

@author: napra
"""

import numpy as np
import cv2

def Background_Removal(Video_file):
    
    cap = cv2.VideoCapture(Video_file)
    
    back = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv2.VideoWriter('Foreground.avi',fourcc, 25 ,(frame_width,frame_height),0) 
    

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if not ret:
            break
        
        movie = back.apply(frame)
        
        out.write(movie)
        
        cv2.imshow('frame',movie)
        #cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    return 'Foreground.avi'