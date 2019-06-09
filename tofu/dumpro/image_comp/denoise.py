# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:45:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

# Built-in
import os

# Standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def denoise(im_path, w_dir, shot_name, im_out = None, meta_data = None, verb = True):
    """
    This subroutine applies grayscale conversion to a collection of images
    The images are read in native form i.e., without any modification.
    
    Parameters
    -----------------------
    video_file:       mp4,avi
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
    
    
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_denoise'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            os.mkdir(im_out)
    
    if verb == True:
        print('output directory is : ', im_out,'\n')
    
    #describing an empty list that will later contain all the frames
    frame_array = []
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    
    #sorting files according to names using lambda function
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('The following files have been read ...')
        
    f_count = 1
    for i in range(len(files)):
        #converting to path
        filename = im_path + files[i]
        #reading each file to extract its meta_data
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        #grayscale conversion
        dst = cv2.fastNlMeansDenoising(img,None,5,21,7)
        #generic name of each image
        name = im_out + 'frame' + str(f_count) + '.jpg'
        #writting the output file
        cv2.imwrite(name, dst)
        height,width = img.shape
        size = (height, width)
        #providing information to user
        f_count += 1
    
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
            
        
    
    return im_out, meta_data