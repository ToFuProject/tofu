# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:41:12 2019

@author: napra

This removes the background of the video and return the foreground as a video 
file

The user must have opencv 3 or greater to use this subroutine 
"""

#Built-ins
import numpy as np
import cv2

def Background_Removal(Video_file):
    """ Removes the background from video and returns it as Foreground.avi
    
    Parameters
    -----------------------
    video_file:      supported formats - mp4,avi
     input video passed in as argument
    
    Return
    -----------------------
    File:            Foreground.avi
     video file which is in .avi format    
     
    """
    
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
    
    cap.release()
    cv2.destroyAllWindows()

    return 'Foreground.avi'