#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:14:49 2019

@author: AK258850
"""

import numpy as np
import cv2
import os

def slice_video(video_file,fw = None,fh = None):
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    pfe = 'E:/NERD/Python/Video_test.avi'
    out = cv2.VideoWriter(pfe, fourcc, fps, (frame_width,frame_height), 0)
    
    out = cv2.VideoWriter
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret: break
        print(type(frame))
#       
        cv2.imshow('frame',frame)
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
