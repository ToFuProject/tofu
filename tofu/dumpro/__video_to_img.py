# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:03 2019

@author: napra

input: video_file (The Grayscale converted video)
output: all the frames in video are converted to images
"""


#standard
import os
import warnings

#built-ins
import numpy as np
try:
    import cv2
except ImportError:
    print("Cannot find opencv package. Try pip intall opencv-contrib-python")


# Playing video from file:
def video2imgconvertor(video_file, path = './data'):
    """Breaks up an input video file into it's constituent frames and 
    saves them as jpg image
    
    Parameters
    -----------------------
    video_file:      mp4/avi
     input video passed in as argument
    path:            string
     Path where the user wants to save the images
    Return
    -----------------------
    File:            String
     Path where the images are stored    
     
    """
    
    print("Converting Video to Images ...... Please Wait")
    
    #trying to open the video file
    try:
        #Reading the video file
        cap = cv2.VideoCapture(video_file)
    #incase of error in file name or path raising exception    
    except IOError:
        print("Path ot file name incorrect or file does not exist")
          
    #Creating Directory
    try:
        if not os.path.exists(path):
            msg = "The provided path does not exist:\n"
            msg += "\t-path: %s"%path
            msg += "\t=> Please create the repository and try again" 
            raise Exception(msg)
    
    #Checking for permission error
    except OSError:
        print ('Error: Creating directory of data')
    
    #Loop Variable    
    currentFrame = 0
    
    #Looping over the entire video    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Saves image of the current frame in jpg file
        #frame number starts from 0
        name = path + str(currentFrame) + '.jpg'
        cv2.imwrite(name, frame)
        
        # To stop duplicate images
        currentFrame += 1
        #To break out of loop when conversion is done
        #ret reads false after we have exhausted through our frames 
        if not ret:
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return path
