# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:56:18 2019

@author: napra
"""

import numpy as np 
import cv2

def VIDEO_DURATION(video_file):
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    
    fps= cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    duration = frame_count/fps
    
    