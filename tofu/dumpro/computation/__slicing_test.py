#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:14:49 2019

@author: AK258850
"""

import numpy as np
import cv2
import os

def slice_video(video_file,tlim):
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start = str(tlim[0])
    start_time = start.split(':'or '.')
    
    
    start_frame_count = fps*60*tlim[0]
    stop_frame_count = fps*60*tlim[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
#    pfe = os.path.join(path, output_name + output_type)
#    out = cv2.VideoWriter('sliced.avi', fourcc, fps, (frame_width, frame_height),0)
    curr_frame = cap.set(2,start_frame_count) 
    
    
    
    while (curr_frame <= stop_frame_count):
        ret,frame = cap.read()
        
        if not ret: break
        
        cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
 #       out.write(frame
        
    cap.release()
#    out.release()
    cv2.destroyAllWindows()
    