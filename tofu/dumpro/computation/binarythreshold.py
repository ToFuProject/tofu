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
    

def binary_threshold(video_file, meta_data = None, path = None, output_name = None, output_type = None):
    """This Subroutine converts a video into binary.  
    
    For more informatio look into the following resource
    1. https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    
    Parameters
    -----------------------
    video_file:       mp4,avi,mpg
     input video along with its path passed in as argument
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
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
    
    if meta_data == None:
        #defining the four character code
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        #defining the frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #defining the fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        #defining the meta_data dictionary
        meta_data = {'fps' : fps, 'frame_height' : frame_height, 
                     'frame_width' : frame_width, 'fourcc' : fourcc}
        
    else:
        #describing the four character code      
        fourcc = meta_data.get('fourcc', int(cap.get(cv2.CAP_PROP_FOURCC)))
        if 'fourcc' not in meta_data:
            meta_data['fourcc'] = fourcc
        
        #describing the frame width
        frame_width = meta_data.get('frame_width', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if 'frame_width' not in meta_data:
            meta_data['frame_width'] = frame_width
        
        #describing the frame height
        frame_height = meta_data.get('frame_height', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if 'frame_height' not in meta_data:
            meta_data['frame_height'] = frame_height
            
        #describing the speed of the video in frames per second 
        fps = meta_data.get('fps', int(cap.get(cv2.CAP_PROP_FPS)))
        if 'fps' not in meta_data:
            meta_data['fps'] = fps

        #describing the total number of frames in the video
        N_frames = meta_data.get('N_frames', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if 'N_frames' not in meta_data:
            meta_data['N_frames'] = N_frames
    
    #describing the output file
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe,fourcc, fps ,(frame_width,frame_height),0) 
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        #to break out of the loop after exhausting all frames
        if not ret:
            break
        #Applying the binary threshold method
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nex, movie = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
        
        #publishing the video
        out.write(movie)
    
    #realeasing outputfile and closing any open windows
    cap.release()
    cv2.destroyAllWindows()
    
    return pfe, meta_data
    
    