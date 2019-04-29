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
    
    stop = str(tlim[1])
    stop_time = start.split(':' or '.')
    
    start_time = start_time[0] + start_time[1]
    stop_time = stop_time[0] + stop_time[1]
        
    start_frame_count = fps*60*start_time
    stop_frame_count = fps*60*stop_time
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
#    pfe = os.path.join(path, output_name + output_type)
#    out = cv2.VideoWriter('sliced.avi', fourcc, fps, (frame_width, frame_height),0)
    curr_frame = cap.set(2,start_frame_count) 
    
    cap.release()
#    out.release()
    cv2.destroyAllWindows()
    