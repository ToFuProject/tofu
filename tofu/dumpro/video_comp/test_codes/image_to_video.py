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
    
def image2video(image_path, meta_data = None, path = None, video_name = None, video_type = None, fps = None, verb = True):
    """This subroutine takes a folder containing all the images and stich them
    up to make a video. The path, name, format and the spped of the video is 
    upto the user to decide 
    
    Parameters
    -----------------------
    image_path:       string
     path of the folder containing the images
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
    video_path:       string
     Path where the user wants to save the video. By default it take the path 
     from where the images were loaded
    fps:              integer
     Number of frames per second of the video. If no value is provided, checks
     for value from meta_data. If no value present in meta_data, then uses a 
     default value of 25 frames per seconds.
    
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
        
    #the output path and file
    pfe = path + video_name + video_type
    
    if verb == True:
        print('Reading path has been successfull ... \n')
        print('Reading the image files and sorting them ...\n')
    
    #describing an empty list that will later contain all the frames
    frame_array = []
    #creating a list of all the files
    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path,f))]    
    
    #sorting files according to names using lambda function
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('The folowing files have been read :\n')
    for i in range(len(files)):
        #converting to path
        filename = image_path + files[i]
        #reading each file to extract its meta_data
        img = cv2.imread(filename)
        height,width,layer = img.shape
        size = (height, width)
        #providing information to user
        if verb == True:
            print(filename)
        
        frame_array.append(img)
        
    if verb == True:
        print('Reading meta_data...\n')
        
    if meta_data == None:
        #defining the four character code
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #defining the frame dimensions
        frame_width = width
        frame_height = height
        #defining the fps
        fps = 25
        #defining the total number of frames
        N_frames = len(files)
        #defining the meta_data dictionary
        meta_data = {'fps' : fps, 'frame_height' : frame_height, 
                     'frame_width' : frame_width, 'fourcc' : fourcc,
                     'N_frames' : N_frames}
    else:
        #describing the four character code      

        fourcc = meta_data.get('fourcc', cv2.VideoWriter_fourcc(*'MJPG'))
        if 'fourcc' not in meta_data:
            meta_data['fourcc'] = fourcc
        
        #describing the frame width
        frame_width = meta_data.get('frame_width', width)
        if 'frame_width' not in meta_data:
            meta_data['frame_width'] = frame_width
        
        #describing the frame height
        frame_height = meta_data.get('frame_height', height)
        if 'frame_height' not in meta_data:
            meta_data['frame_height'] = frame_height
            
        #describing the speed of the video in frames per second 
        fps = meta_data.get('fps', 25)
        if 'fps' not in meta_data:
            meta_data['fps'] = fps

        #describing the total number of frames in the video
        N_frames = meta_data.get('N_frames', len(files))
        if 'N_frames' not in meta_data:
            meta_data['N_frames'] = N_frames
    
    #the output file     
    out = cv2.VideoWriter(pfe,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    
    if verb == True:
        print('converting to video ...\n')
    #looping through all the files
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    
    if verb == True:
        print('Video conversion successfull ...\n')
        print('Creating output file ...\n')
    #releasing output
    out.release()
    
    return pfe, meta_data
