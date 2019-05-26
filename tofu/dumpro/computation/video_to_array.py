# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:20:59 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

This Subroutine converts a video file into a numpy array with dimensions time, frame _height and frame_width
The user must have Opencv 3 or greater and numpy installed to run this subroutine
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
    

def video_to_pixel(video_file):
    """Converts imput video file to a numpy array
    The video file is converted to Grayscale and hence the array is of 
    3 dimension
    
    Parameters
    -----------------------
    video_file:       mp4,avi,mpg
     input video along with its path passed in as argument
        
    Return
    -----------------------
    data:            Numpy array
     The numpy array containing the pixel internsity information of every
     frame of the video
    """
    #reading the input file'
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
    
    #reading the first frame to get video metadata    
    ret,frame = cap.read()
    
    #reading the video metadata
    rows = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    columns = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #declaration of the empty array
    pixel = np.ndarray(( total_frame, rows, columns), dtype = int)
    
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_height' : rows, 'frame_width' : columns}
    
    #initialization of the frame variable 
    frame_counter = 0
    #looping over the entire video
    while (cap.isOpened()):
        
        ret,frame = cap.read()
        #breaking out of loop when the frames are exhausted
        if not ret: break
        
        #conversion of each input frame to grayscale single channel image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #assigning each frame to the video array
        pixel [frame_counter] = frame
        #changing the frame variable 
        frame_counter += 1 
    
    print('total number of frames converted :', total_frame)
    #releasing the output file
    cap.release()
    cv2.destroyAllWindows()
    
    return (pixel, meta_data)

