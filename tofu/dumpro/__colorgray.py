# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal

input: video_file (The original video of the shot capturedby the camera)
output: gray(The gray scale version of the shot on which a threshold function
             will be applied in the next step)
"""

import numpy as np
import cv2

def ConvertGray(video_file):
    """Converts imput video file to grayscale and saves it as Grayscale.avi
    
    ==========================================================================
    Parameters
    ==========================================================================
    input:
    --------------------------------------------------------------------------
     video file(mp4,avi)
    --------------------------------------------------------------------------
    output:
     video file name which is in .avi format
    """

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