# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal

input: video_file (The original video of the shot capturedby the camera)
output: gray(The gray scale version of the shot on which a threshold function
             will be applied in the next step)
"""

# to convert video into grayscale for setting threshold
# this is the first step

import numpy as np
import cv2

def ConvertGray(video_file):
    
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('video',gray)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return gray