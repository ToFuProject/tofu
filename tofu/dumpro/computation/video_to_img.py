# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:03 2019

@author: Arpan Khandelwal
email: napraarpan@gail.com

input: video_file (The Grayscale converted video)
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


def video2img(video_file, path = None, image_name = None, image_type = None):
    """Breaks up an input video file into it's constituent frames and 
    saves them as jpg image
    
    Parameters
    -----------------------
    video_file:      mp4,avi,mpg
     input video passed in as argument
    path:            string
     Path where the user wants to save the images. By it will try to make a 
     folder in the same directory as the video, by the name data. If it fails 
     in creating the directory due t permissions, then it will give out an 
     error.
    image_name:      string
     Name of the image file. The frame number will be appended to the name when
     writting the image file. By default it is 'frame'
    image_type:      string
     format of the image. By default it is .jpg
     Return
    -----------------------
    File:            String
     Path where the images are stored    
     
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    folder = 'data'
    if path is None:
        #the directory path is defined.  
        #The last argument is for adding a trailing slash
        path = os.path.join(drive,path_of_file,'')
        #defining the folder inside whch the images will be stored
        path += folder
        path = os.path.join(path,'')
        #checking whether path already exists or not
        #if not, creates the path
        if not os.path.exists(path):
            os.mkdir(path)
        
    #checking for the name of the output file
    if image_name is None:
        image_name = 'frame'
    #checking for the putput format of the video
    if image_type is None:
        image_type = '.jpg'
    
    print("Converting Video to Images ...... Please Wait")
    
    #trying to open the video file
    try:
        if not os.path.isfile(video_file):
            raise Exception
        cap = cv2.VideoCapture(video_file)
        
    except Exception:
        msg = 'the path or filename is incorrect.'
        msg += 'PLease verify the path or file name and try again'
        raise Exception(msg)
    
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
    
    #Loop Variable    
    currentFrame = 0
    
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
            name = path + image_name + str(currentFrame) + image_type
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
    
    return path
