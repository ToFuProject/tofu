#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:15:20 2019

@author: AK258850
"""
#built in
import os

#standard
import numpy as np
#special
try:
    import cv2
except ImportError:
    print("Cannot find opencv package. Try pip intall opencv-contrib-python")
    

def slice_video(video_file,path = None, output_name = None,output_type = None,
                tlim = None, fh = None, fw = None):
    """
    This Subroutine crops the video in time and the size of frame. The User 
    inputs the output video location and the name of the output file and the 
    time limits for which he wants the video and the region of interest in the 
    frame
    
    Parameters
    -----------------------
    video_file:      mp4,avi,mpg
     input video passed in as argument
    path:            string
     Path where the user wants to save the video
    output_name:     string
     Name of the output file
    output_type      string
     the video output format
    tlim:            tuple
     the time window of the video in which the user is interested
    fh:              tuple
     the height of the region of interest
    fw:              tuple
     the width of the region of interest
     
    Return
    -----------------------
    pfe:            String
     The path along with the name of the video file
    metadata:       dictionary
     A dictionary containin gthe metadata of the video i.e., the frame height
     the frame width and the fps
    reshape:        dictionary
     A dictionary containing the information on the metadata of the sliced video
     i.e., the time stamp and the frame size
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file    
    if path is None:
        path = os.path.join(drive,path_of_file)
    #checking for the name of the output file
    if output_name is None:
        output_name = file[0]+'_slice'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file,0)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
        
    ret, frame = cap.read()
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps, ((fh[1]-fh[0]),(fw[1]-fw[0])),0)
    
    curr_frame = 0
    
    while (cap.isOpened()):
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret: break
        if (curr_frame >= tlim[0] and curr_frame <= tlim[1]):
            print(curr_frame)
            print(frame.shape)
            frame = frame[fh[0]:fh[1],fw[0]:fw[1]]
            print(frame.shape)
            out.write(frame)
            cv2.imshow('frame',frame)
        
        curr_frame += 1
    metadata = None
    reshape = None    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return pfe, metadata, reshape