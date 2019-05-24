#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:24:40 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#nuilt in
import os

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def play_video(video_file):
    """
    This subroutine plays any video passed in as input 
    Parameters
    -----------------------
    video_file:      supported formats - mp4,avi,mpg
     input video passed in as argument
    
    Return
    -----------------------
     Only displays the video
    """
    # Create a VideoCapture object and read from input file 
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
    
    # Check if camera opened successfully 
    if (cap.isOpened()== False):  
        print("Error opening video  file") 
        
    # Read until video is completed 
    while(cap.isOpened()): 

        # Capture frame-by-frame 
        ret, frame = cap.read() 
        
        if ret == True: 
            # Display the resulting frame 
            cv2.imshow('Frame', frame)         
            # Press Q on keyboard to  exit or the program automatically exits 
            #when the video has been played completely
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

        # Break the loop 
        else:  
            break
    
    # When everything done, release  
    # the video capture object 
    cap.release() 
                
    # Closes all the frames 
    cv2.destroyAllWindows() 