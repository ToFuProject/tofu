# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

Input the video file on which the Dumpro wants the operation to be performed
and the output is the grayscale video file that will be used for futher 
preprocessing 

You need to have Opencv 3 or greater installed, for using this subroutine
"""

import numpy as np
import cv2

def ConvertGray(video_file):
    """Converts imput video file to grayscale and saves it as Grayscale.avi
    
    Parameters
    -----------------------
    video_file:      supported formats - mp4,avi
     input video passed in as argument
    
    Return
    -----------------------
    File:            Grayscale.avi
     video file which is in .avi format    
     
    """

    # Didier: all comments indentation should be aligned with code identation
    # I correct some of them below for example
    
    # reading the input file   
    cap = cv2.VideoCapture(video_file)
    #read the first frame    
    ret,frame = cap.read()

    #describing the four character code fourcc  
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #getting frame height and width
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    #videowriter writes the new video with the frame height and width and fps   
    #videowriter(videoname, format, fps, dimensions_of_frame,)
    out = cv2.VideoWriter('Grayscale.avi',fourcc, 25 ,(frame_width,frame_height),0)    
    
    #loops over the entire video frame by frame and convert each to grayscale
    #then writting it to output file     
    while(cap.isOpened()):
        ret, frame = cap.read()
        #to check whether cap read the file successfully         
        if not ret: break
        #conversion from RGB TO GRAY frame by frame        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = gray
        #writing the gray frames to out        
        out.write(frame)
    
    #closing everything       
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 'Grayscale.avi'
