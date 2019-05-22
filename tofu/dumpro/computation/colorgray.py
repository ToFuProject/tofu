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

def ConvertGray(video_file, path = None, output_name = None, output_type = None):
    """Converts input video file to grayscale, denoises it and saves it as Grayscale.avi
    
    Parameters
    -----------------------
    video_file:       mp4,avi
     input video along with its path passed in as argument
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
        #checking if the path provided is correct or not
        if os.path.isfile(video_file):
            cap = cv2.VideoCapture(video_file)
    except IOError:
        print("Path or file name incorrect or file does not exist")
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
        #writing the gray frames to out        
        out.write(dst)
    
    #closing everything       
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #returning the output file and metadata
    return pfe, meta_data