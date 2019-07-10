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
    

def video_to_pixel(video_file, meta_data = None, verb = True):
    """Converts imput video file to a numpy array
    The video file is converted to Grayscale and hence the array is of 
    3 dimension
    
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
    if verb == True:
        print('video reading has been successful ...\n')
        print('reading meta_data ...\n')
    if meta_data == None:
        #defining the four character code
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        #defining the frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #defining the fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        #defining the total number of frames
        N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #defining the meta_data dictionary
        meta_data = {'fps' : fps, 'frame_height' : frame_height, 
                     'frame_width' : frame_width, 'fourcc' : fourcc,
                     'N_frames' : N_frames}
        
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


    #declaration of the empty array
    pixel = np.ndarray(( N_frames, frame_width, frame_height), dtype = int)
    
    if verb == True:
        print('converting to numpy arrray ...\n')
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
    
    if verb == True:
        print('total number of frames converted :', N_frames,'\n')
        print('array creation successful ...')
    #releasing the output file
    cap.release()
    cv2.destroyAllWindows()
    
    return pixel, meta_data

