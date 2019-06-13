# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:18:45 2019

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

def crop_video(im_path, w_dir, shot_name, im_out = None, 
               meta_data = None, verb = True):
    """This subroutine crops a video and also slices it to return us a 
    collection of all the frames from our desired time window in our desired region
    of interest
    The images are read in original form i.e., without any modifications
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name and meta_data are provided by the image processing 
    class in the core file.
    The verb paramenter is used when this subroutine is used independently.
    Otherwise it is suppressed by the core class.
    
    The region of interest and the time window of interest is later 
    gathered from the user during runtime.
    
    Parameters
    -----------------------
    im_path:          string
     input path where the images are stored
    w_dir:            string
     A working directory where the proccesed images are stored
    shot_name:        String
     The name of the tokomak machine and the shot number. Generally
     follows the nomenclature followed by the lab
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
    disp              boolean
     to display the frames set equal to True. By default is set to True
    
    Return
    -----------------------
    im_out:              String
     Path along where the proccessed images are stored  
    meta_data:        dictionary
     A dictionary containing the meta data of the video.
    """
    
    #the output directory based on w_dir and shot_name
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_slice'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            os.mkdir(im_out)
    #the output directory shown to user
    if verb == True:
        print('output directory is : ', im_out,'\n')
        
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    
    #sorting files according to names using lambda function
    #-4 is to remove the extension of the images i.e., .jpg
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('The following files have been read ...')

    user_input = input('Enter the values for cropping :')
    print(type(user_input))
    user_input = user_input.split(',')
#    fw = user_input[0]
#    fh = user_input[1]
    lfw = int(user_input[0])
    ufw = int(user_input[1])
    lfh = int(user_input[2])
    ufh = int(user_input[3])
    
    frame_width = ufw - lfw
    frame_height = ufh - lfh
    
    print('The total number of frames are :', len(files),'\n')
    tlim = input('Enter the frames of interest :')
    tlim = tlim.split(',')
    start = int(tlim[0])
    end = int(tlim[1])

    
    curr_frame = 1
    print('creating temp file...\n')
    for i in range(len(files)):
        filename = im_path + files[i]
        if curr_frame >= start and curr_frame <= end:
            if verb == True:
                print(filename)
            #reading each file to extract its meta_data
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            height,width = img.shape[0],img.shape[1]
            img = img[lfw:ufw, lfh:ufh]
            name =im_out + 'frame' + str(curr_frame) + '.jpg'
            cv2.imwrite(name,img)
        curr_frame += 1
        
    #meta_data of the collection of images
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
            
    if verb == True:
        print('meta_data read successfully ...\n')
        
    print('Releasing output...\n')
    
    cv2.destroyAllWindows()
    return im_out, meta_data
    