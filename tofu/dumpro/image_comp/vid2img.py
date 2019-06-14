# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:03 2019

@author: Arpan Khandelwal
email: napraarpan@gail.com

input: video_file 
output: all the frames in video are converted to images
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


def video2img(video_file, w_dir, shot_name = None, meta_data = None, verb = True):
    """Breaks up an input video file into it's constituent frames and 
    saves them as jpg image
    
    Parameters
    -----------------------
    video_file:      mp4,avi,mpg
     input video passed in as argument
    w_dir;           string
     The working directory to provide the output
    shot_name:       string
     the name of the tokomak machine followed by the shot number.
     By default it is none, this code automatically extracts the shot information
     from the video name.
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video.
     
     Return
    -----------------------
    path:            String
     Path where the images are stored    
    meta_data:       dictionary
     disctionary containing the meta_data information
    shot_name:          string
     Tokomak shot name and number
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    folder = file[0]
    
    if shot_name == None:
        shot_name = folder
    
    #the directory path is defined.  
    #The last argument is for adding a trailing slash
    path = os.path.join(w_dir,shot_name,'')
    #defining the folder inside whch the images will be stored
    path += folder
    path = os.path.join(path,'')
    #checking whether path already exists or not
    #if not, creates the path
    if not os.path.exists(path):
        os.mkdir(path)
    
    #trying to open the video file
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
        
    if verb == True:
        print('Video reading has been successful ...\n')
        print('reading destination directory ...\n')
    
    #Creating Directory
    try:
        if not os.path.exists(path):
            msg = "The provided path does not exist:\n"
            msg += "\t-path: %s"%path
            msg += "\t=> Please create the repository and try again" 
            raise Exception(msg)
    
    #Checking for permission error
    except OSError:
        print ('Error: Creating directory of data')
        
    if verb == True:
        print('destiantion directory reading successfull ...\n')
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
        folder
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
        print('reading meta_data successfull ...\n')
    
    #Loop Variable    
    currentFrame = 1
    
    if verb == True:
        print("Converting Video to Images ...... Please Wait")
    
    #Looping over the entire video    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #To break out of loop when conversion is done
        #ret reads false after we have exhausted through our frames 
        if ret == True:
            # Saves image of the current frame in user defined format 
            #or by default jpg file
            #frame number starts from 0
            name = path + 'frame' + str(currentFrame) + '.jpg'
            print('Converting frame :', currentFrame)
            cv2.imwrite(name, frame)
        else:
            break
        # To stop duplicate images
        currentFrame += 1
    print('Total number of frames converted :', currentFrame)    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return path, meta_data, shot_name
