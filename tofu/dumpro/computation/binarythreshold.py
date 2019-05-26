# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:05:18 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#Built-ins
import os
#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Cannot find opencv package. Try pip intall opencv-contrib-python")
    

def binary_threshold(video_file, path = None, output_name = None, output_type = None):
    """This Subroutine converts a video into binary.  
    
    For more informatio look into the foloowinf resource
    1. https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    
    Parameters
    -----------------------
    video_file:       mp4,avi,mpg
     input video along with its path passed in as argument
    path:             string
     Path where the user wants to save the video. By default it take the path 
     from where the raw video file was loaded
    output_name:      String
     Name of the binary converted video. By default it appends to the 
     name of the original file '_bi'
    output_type:      String
     Format of output defined by user. By default it uses the format of the 
     input video
    
    Return
    -----------------------
    pfe:              String
     The path along with the videofile name
    metadata:         Dictionary
     A dictionary containing all the information on the metadata of the video
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
        output_name = file[0]+'_binary'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
        
    cap = cv2.VideoCapture(video_file,0)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    #get frame width and height of the original video
    #the result video has the same number of pixels
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_height' : frame_height, 'frame_width' : frame_width}
    #describing the output file
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe,fourcc, fps ,(frame_width,frame_height),0) 
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        print(type(frame))
        #to break out of the loop after exhausting all frames
        if not ret:
            break
        #Applying the binary threshold method
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nex, movie = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
        print(movie.shape)
        #publishing the video
        out.write(movie)
    
    #realeasing outputfile and closing any open windows
    cap.release()
    cv2.destroyAllWindows()
    
    return pfe, meta_data
    
    