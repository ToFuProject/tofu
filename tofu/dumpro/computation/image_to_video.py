# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:43:55 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

#built-ins
import os

#standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def image2video(image_path,path = None,video_name = None,video_type = None ,fps = None):
    """This subroutine takes a folder containing all the images and stich them
    up to make a video. The path, name, format and the spped of the video is 
    upto the user to decide 
    
    Parameters
    -----------------------
    image_path:       string
     path of the folder containing the images
    video_path:       string
     Path where the user wants to save the video. By default it take the path 
     from where the images were loaded
    fps:              integer
     Number of frames per second of the video
    
    Return
    -----------------------
    pfe:              String
     Path along with the name and type of video    
    meta_data:        dictionary
     A dictionary containing the meta data of the video.
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(image_path)
    
    #default arguments
    if (path == None):
        path = os.path.join(drive,path_file,'')
    if (video_name == None):
        video_name = 'Stiched'
    if (video_type == None):
        video_type = '.avi'
    if fps == None:
        fps = 20
        
    #the output path and file
    pfe = path + video_name + video_type
    
    frame_array = []
    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path,f))]    
    
    #sorting files according to names
    files.sort(key = lambda  x: int(x[5:-4]))
    
    for i in range(len(files)):
        filename = image_path + files[i]
        
        img = cv2.imread(filename)
        height,width,layer = img.shape
        size = (height, width)
        print(filename)
        
        frame_array.append(img)
        
    out = cv2.VideoWriter(pfe,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    meta_data = {'fps' : fps, 'frame_height' : height, 
                 'frame_width' : width, 'N_frames' : (len(files))}
    
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
#        cv2.imshow('frame',frame_array)
    out.release()
    
    return pfe, meta_data
