# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:05:18 2019

@author: napra
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
    """    """
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
        output_name = file[0]+'_foreground'
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
        nex, movie = cv2.threshold(frame,100,255,cv2.THRESH_BINARY)
        print(movie.shape)
        #publishing the video
        out.write(movie)
    
    #realeasing outputfile and closing any open windows
    cap.release()
    cv2.destroyAllWindows()
    
    return pfe, meta_data
    
    