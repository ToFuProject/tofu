# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:34:38 2019

@author: napra
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
    

def crop_video(video_file, tlim, path = None, output_name = None, output_type = None):
    """
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
    
    ret,frame = cap.read()
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps, (frame_height, frame_width),0)
    frame_counter = 0
    while cap.isOpened():
        
        ret,frame = cap.read()
        if not ret: break
        print(ret)
        print(frame.shape)
        if (frame_counter>=tlim[0] and frame_counter<=tlim[1]):
            out.write(frame)
            cv2.imshow('frame',frame)
            print(frame.shape)
            print('writting frame :', frame_counter)
        frame_counter += 1

    metadata = None
    reshape = None    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return pfe, metadata, reshape