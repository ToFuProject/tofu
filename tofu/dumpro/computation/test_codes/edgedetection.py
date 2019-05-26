# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:17:50 2019

@author: napra
"""

import os
import numpy as np
import cv2

def detect_edge(video_file, path = None, output_name = None, output_type = None):
    
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
        output_name = file[0]+'_edge'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    
    # reading the input file 
    try:
        #checking if the path provided is correct or not
        if os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
    except IOError:
        print("Path or file name incorrect or file does not exist")
    #read the first frame    
    ret,frame = cap.read()

    #describing the four character code fourcc  
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #getting frame height and width and fps
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_height' : frame_height, 'frame_width' : frame_width}
    #videowriter writes the new video with the frame height and width and fps   
    #videowriter(videoname, format, fps, dimensions_of_frame,)
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps,
                          (frame_width,frame_height),0)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        #to check whether cap read the file successfully         
        if not ret: break
        
        edge = cv2.Canny(frame,127,255)    
        
        out.write(edge)
        
        #closing everything       
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #returning the output file and metadata
    return pfe, meta_data