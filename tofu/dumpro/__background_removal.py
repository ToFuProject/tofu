# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:41:12 2019

@author: napra

This removes the background of the video and return the foreground as a video 
file

The user must have opencv 3 or greater to use this subroutine 
"""

#Built-ins
import numpy as np
try:
    import cv2
except ImportError:
    print("Cannot find opencv package. Try pip intall opencv-contrib-python")


def Background_Removal(Video_file, path = './', output_name = "Foreground", output_type = ".avi"):
    """ Removes the background from video and returns it as Foreground.avi
    
    Parameters
    -----------------------
    video_file:      supported formats - mp4,avi
     input video passed in as argument
    path:             string
     Path where the user wants to save the video
    output_name:      String
     Name of the Background subtracted video
    output_type:      String
     Format of output defined by user. By default .avi 
    
    Return
    -----------------------
    File:             String
     Path of the video along with it's name and format    
     
    """
    #trying to open the video file
    try:
        #Reading the video file
        cap = cv2.VideoCapture(Video_file)
     #incase of error in file name or path raising exception    
    except IOError:
        print("Path/Filename incorrect or File/path does not exits")
    
    #creating the background subtraction method for applying to the video
    back = cv2.bgsegm.createBackgroundSubtractorMOG()
    #describing the four character code fourcc
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    #get frame width and height of the original video
    #the result video has the same number of pixels
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    #describing the output file
    out = cv2.VideoWriter(path+output_name+output_type,fourcc, 25 ,(frame_width,frame_height),0) 
    
    #looping over the video applying the background subtaction method to each frame
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        #to break out of the loop after exhausting all frames
        if not ret:
            break
        #Applying the background subtraction method
        movie = back.apply(frame)
        #publishing the video
        out.write(movie)
    
    #realeasing outputfile and closing any open windows
    cap.release()
    cv2.destroyAllWindows()

    return path+output_name+output_type