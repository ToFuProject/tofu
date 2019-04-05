# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal

input: video_file (The original video of the shot capturedby the camera)
output: gray(The gray scale version of the shot on which a threshold function
             will be applied in the next step)
"""

# to convert video into grayscale for setting threshold
# this is the first step
# then it stores the converted video as gray.avi for further processing

import numpy as np
import cv2

def ConvertGray(video_file):
    """Converts imput video file to grayscale and saves it as Grayscale.avi
    
    Initially it reads the video file and gets the values of its parameters,
    such as frame height, width etc. According to this the output file is decided.
    The output file will have the same size as the input video but its fps is set 
    to 25 fps
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
    
    return none