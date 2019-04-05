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
    """

   
    cap = cv2.VideoCapture(video_file)
    ret,frame = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
# videowriter(videoname, format, fps, dimensions_of_frame,)
    out = cv2.VideoWriter('Grayscale.avi',fourcc, 25 ,(frame_width,frame_height),0)    
    
     
    while(cap.isOpened()):
        ret, frame = cap.read()
#to check whether cap read the file successfully         
        if not ret: break
#conversion from RGB TO GRAY frame by frame        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = gray
#writing the gray frames to out        
        out.write(frame)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 'Grayscale.avi'