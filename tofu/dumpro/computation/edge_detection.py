# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:17:50 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

#nuilt in
import os

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")


def detect_edge(video_file, path = None, output_name = None, output_type = None):
    """This subroutine detects edges from the the proivided video. The video provided
    must consists of binary images. This is the next step after performing the 
    binary conversion step.
    
    for more information look into 
    1. https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    2. https://en.wikipedia.org/wiki/Canny_edge_detector
    
    Parameters
    -----------------------
    video_file:       mp4,avi,mpg
     input video along with its path passed in as argument
    path:             string
     Path where the user wants to save the video. By default it take the path 
     from where the raw video file was loaded
    output_name:      String
     Name of the Grayscale converted video. By default it appends to the 
     name of the original file '_grayscale'
    output_type:      String
     Format of output defined by user. By default it uses the format of the 
     input video
    
    Return
    -----------------------
    pfe:              String
     Path along with the name and type of video    
    meta_data:        dictionary
     A dictionary containing the meta data of the video.
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
        output_name = file[0]+'_edge'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    
    # reading the input file 
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
        
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
    
        #applying the Canny edge detection algorithm.
        #check docstring for further information
        edge = cv2.Canny(frame,127,255)    
        out.write(edge)
    
        #closing everything       
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #returning the output file and metadata
    return pfe, meta_data