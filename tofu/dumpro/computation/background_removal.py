# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:41:12 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

This removes the background of the video and return the foreground as a video 
file

The user must have opencv 3 or greater to use this subroutine 
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
    
#dumpro specific
#import framebyframe_sub as rm
#import video_to_array as vta


def remove_background(video_file, meta_data = None, path = None, output_name = None, output_type = None):
    """ Removes the background from video and returns it as Foreground.avi
    
    For further information consult the following resources
    1. https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html
       
    Parameters
    -----------------------
    video_file:      supported formats - mp4,avi
     input video passed in as argument
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
    path:             string
     Path where the user wants to save the video
    output_name:      String
     Name of the Background subtracted video
    output_type:      String
     Format of output defined by user. By default .avi 
    
    Return
    -----------------------
    pfe:             String
     Path of the video along with it's name and format    
    metadata          dictionary
     A dictionary containing the metadata of the video
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
        output_name = file[0]+'_foreground'
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
    
    #creating the background subtraction method for applying to the video
    back = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    #describing the output file
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe,fourcc, fps ,(frame_width,frame_height),0) 
    
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
    
    #(videodata, fps, rows,columns) = vta.video_to_pixel(file)
    #rm.removebackground(videodata, fps, rows, columns, path, output_name, output_type)
    
    return pfe, meta_data