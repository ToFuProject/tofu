# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

Input the video file on which the user wants the operation to be performed
and the output is the grayscale denoised video file that will be used for futher 
preprocessing 

You need to have Opencv 3 or greater installed, for using this subroutine
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

def convertgray(video_file,meta_data = None , path = None, output_name = None, output_type = None, verb = True):
    """Converts input video file to grayscale, denoises it and saves it as 
    Grayscale.avi
    
    The denoising process is very processor intensive and takes time, 
    but it give very good results. 
    
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
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    if path is None:
        path = os.path.join(drive,path_of_file)
    #checking for the name of the output file
    if output_name is None:
        output_name = file[0]+'_grayscale'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    
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
        print('File successfully loaded for grayscale conversion...\n')
    
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
            
    #videowriter writes the new video with the frame height and width and fps   
    #videowriter(videoname, format, fps, dimensions_of_frame,)
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps,
                          (frame_width,frame_height),0)    
    print(frame_height, frame_width)
    
    frame_count = 0
    if verb == True:
        print('initiating grayscale conversion and denoising ... \n')
    
    #loops over the entire video frame by frame and convert each to grayscale
    #then writting it to output file     
    while(cap.isOpened()):
        ret, frame = cap.read()
        #to check whether cap read the file successfully         
        if not ret: break
        #conversion from RGB TO GRAY frame by frame        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #denoising the image 
        #consult opencv documentation on fastnlMeansDenoising for 
        #further information on the parameters used
        dst = cv2.fastNlMeansDenoising(gray,None,5,21,7)
        if verb == True:
            frame_count += 1
            frames_left = N_frames - frame_count
            print('Frames left to process : ', frames_left)
            
        #writing the gray frames to out        
        out.write(dst)
        
    print('All frames processed successfully and output file has been written ... \n')
    #closing everything       
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #returning the output file and metadata
    return pfe, meta_data