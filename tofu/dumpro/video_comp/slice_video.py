# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:18:45 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
# Built-in
import os
import tempfile
# Standard
import numpy as np

#dumpro specific
import image_to_video as imv

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

def crop_video(video_file, meta_data = None, path = None, output_name = None, output_type = None, verb = True):
    
    temp = 'images'
    
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    if path is None:
        path = os.path.join(drive,path_of_file,'')
    #checking for the name of the output file
    if output_name is None:
        output_name = file[0]+'_crop'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    
    im_path = os.path.join(drive, path_of_file,'')
    im_path += temp
    im_out = os.path.join(im_path,'')
    if not os.path.exists(im_out):
        os.mkdir(im_out)
    
    # reading the input file 
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
     
    #read the first frame    
    ret,frame = cap.read()
    if verb == True:
        print('File successfully loaded for cropping...\n')
    
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
            
    if verb == True:
        print('meta_data reading successfull ...\n')
    print('The size of the frame is :',frame_width,',',frame_height,'\n')
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
    
    print('The total number of frames are :', N_frames,'\n')
    tlim = input('Enter the frames of interest :')
    tlim = tlim.split(',')
    start = int(tlim[0])
    end = int(tlim[1])
    
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps,
                          (frame_width,frame_height),0)
    
    curr_frame = 1
    print('creating temp file...\n')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if ret == True:
            if curr_frame >= start and curr_frame <= end:
                frame = frame[lfw:ufw, lfh:ufh]
                
                name =im_out + 'image' + str(curr_frame) + '.jpg'
                print(name)
                cv2.imwrite(name,frame)
        curr_frame += 1
        
    print('Releasing output...\n')
    
    cap.release()
    cv2.destroyAllWindows()
    return im_out, meta_data
    