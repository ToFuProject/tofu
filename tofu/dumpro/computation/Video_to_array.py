# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:45:45 2019

@author: napra
email: napraarpan@gmail.com

"""
#from readpixel import read_pixel
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
    #reading the input file
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
    
    #declaration of the empty array
    pixel = np.ndarray((rows , columns, total_frame), dtype = int)
    
    #initialization of the frame variable 
    frame_counter = 0
    
    #looping over the entire video
    while (cap.isOpened()):
        
        ret,frame = cap.read()
        if not ret: break
        
        #conversion of each input frame to grayscale single channel image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print('frame number',frame_counter)
        
        #looping through all pixels of the image
        for i in range(0,rows):
            for j in range(0,columns):
                pixel [i][j][frame_counter] = frame.item(i, j)
        #changing the frame variable 
        frame_counter += 1 
    
    cap.release()
    cv2.destroyAllWindows()
    
    return pixel