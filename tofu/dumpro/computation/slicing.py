#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:15:20 2019

@author: AK258850
"""

import numpy as np
import cv2

def slice_video(video_file,path = './', output_name,output_name,tlim):
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame_count = fps*60*tlim[0]
    stop_frame_count = fps*60*tlim[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps, (frame_width, frame_height),0)
    
    curr_frame = 0
    
    while (curr_frame >= start_frame_count and curr_frame <= stop_frame_count):
        ret,frame = cap.read()
        
        if not ret: break
        
        out.write(frame)
        
        curr_frame += 1
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    