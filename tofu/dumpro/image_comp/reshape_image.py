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

def reshape_image(im_path, w_dir, shot_name, 
               tlim = None, hlim = None, wlim = None, 
               im_out = None, meta_data = None, verb = True):
    """This subroutine crops a video and also slices it to return us a 
    collection of all the frames from our desired time window in our desired region
    of interest
    The images are read in original form i.e., without any modifications
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name and meta_data are provided by the image processing 
    class in the core file.
    The verb paramenter is used when this subroutine is used independently.
    Otherwise it is suppressed by the core class.
    
    It is better to let the code use the default value of im_out. The default 
    method of setting the output path by the code is to create an output
    folder using the shotname in the working directory
    
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
    tlim:             tuple
     The time limits for the image files, i.e the frames of interest
    hlim, wlim:       tuple
     The height and width limits of the frame to select the region of interest
    im_out:           string
     The output path for the images after processing.
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
  
    Return
    -----------------------
    im_out:           String
     Path along where the proccessed images are stored  
    meta_data:        dictionary
     A dictionary containing the meta data of the video.
    reshape:          dictionary
     A dictionary containing information of the selected and cropped frames
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
    
    #cropping frames based on time frame
    if tlim == None:
        start = 1
        end = len(files)
    else:
        start = int(tlim[0])
        end = int(tlim[1])

    #looping through video file
    curr_frame = 1
    print('creating temp file...\n')
    for i in range(len(files)):
        filename = im_path + files[i]
        if curr_frame >= start and curr_frame <= end:
            if verb == True:
                print(filename)
            #reading each file to extract its meta_data
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            height, width = img.shape[0], img.shape[1]
            
            #cropping the frame
            if hlim == None and wlim == None:
                img = img
            elif hlim == None and wlim != None:
                img = img[wlim[0]:wlim[1],:]
            elif hlim != None and wlim == None:
                img = img[:,hlim[0]:hlim[1]]
            elif hlim != None and wlim != None:
                img = img[wlim[0]:wlim[1],hlim[0]:hlim[1]]
                
            #output name of image
            name =im_out + 'frame' + str(curr_frame) + '.jpg'
            #writting the output file
            cv2.imwrite(name,img)
        #incrementing frame counter
        curr_frame += 1
        
    #meta_data of the collection of images
    if meta_data == None:
        #defining the four character code
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        frame_height = height
        frame_width = width
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
            meta_data['frame_width'] = width
        
        #describing the frame height
        frame_height = meta_data.get('frame_height', height)
        if 'frame_height' not in meta_data:
            meta_data['frame_height'] = height
            
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
    
    if verb == True:
        print('creating reshape dictionary')
    #creating reshape dictionary
    reshape = {'height' : hlim, 'width' : wlim, 'tlim' : tlim}
    
    cv2.destroyAllWindows()
    
    return im_out, meta_data, reshape
    