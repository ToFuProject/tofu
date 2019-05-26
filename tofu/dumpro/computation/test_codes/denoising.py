# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:56:47 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

This subroutine denoises video files by gausian average.
"""

#built in
import os
import time

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def denoising(video_file, colored, path = None, output_name = None, output_type = None):
    """
    """
    start = time.time()
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
        output_name = file[0]+'_denoise'
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

    #describing the four character code fourcc  
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #getting frame height and width and fps
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_height' : frame_height, 'frame_width' : frame_width}
    #videowriter writes the new video with the frame height and width and fps   
    #videowriter(videoname, format, fps, dimensions_of_frame,)
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe, fourcc, fps,
                          (frame_width,frame_height),0)
    no = 0
    if (colored == True):
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            dst = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
            
            print('writing frame :',no)
            no+=1
            out.write(dst)
            cv2.imshow('frame',dst)
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        end = time.time()
        print(end - start)
        
        return pfe, meta_data
    else:
        while cap.isOpened():
            ret,frame = cap.read()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dst = cv2.fastNlMeansDenoising(frame,None,5,21,7)
            
            out.write(dst)
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return pfe, meta_data