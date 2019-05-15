# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:20:59 2019

@author: napra
"""

try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
import numpy as np
from PIL import Image

def video_to_pixel(videofile):
    """Converts imput video file to a numpy array
    The video file is converted to Grayscale and hence the array is of 
    3 dimension
    
    Parameters
    -----------------------
    video_file:       mp4,avi
     input video along with its path passed in as argument
        
    Return
    -----------------------
    data:            Numpy array
     The numpy array containing the pixel internsity information of every
     frame of the video
    """
    #reading the input file'
    try:
        cap = cv2.VideoCapture(videofile)
    except IOError:
        print("Path or file name incorrect or file does not exist")
    
    #reading the first frame to get video metadata    
    ret,frame = cap.read()
    
    #reading the video metadata
    rows = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    columns = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(rows, columns, total_frame)
    #declaration of the empty array
    pixel = np.ndarray(( total_frame, rows, columns), dtype = int)
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_width' : rows, 'frame_height' : columns}
    #initialization of the frame variable 
    frame_counter = 0
    
    #looping over the entire video
    while (cap.isOpened()):
        
        ret,frame = cap.read()
        if not ret: break
        
        #conversion of each input frame to grayscale single channel image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print('frame number',frame_counter)
        
        #assigning each frame to the video array
        pixel [frame_counter] = frame
        #changing the frame variable 
        frame_counter += 1 
    
    cap.release()
    cv2.destroyAllWindows()
    
    return (pixel, fps, rows, columns)

