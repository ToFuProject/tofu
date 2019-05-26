# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

# class definitions (object-oriented codes)
#class video is an object containing information about the videofile:
#resolution, size, no of frames

# Built-ins
import os
import warnings

# Standard
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import datetime


#more special packages
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
# tofu-specific
#import tf.__version__ as __version__

# dumpro-specific
import computation as _comp

##########################################################
##########################################################
#               Title
##########################################################

class Video(object):
    """ A Video class that will pe preocessed using the various methods
    defined below.

    Input:
    --------------------------------------------
    filename = video file along with the path
    
    Attributes:
    --------------------------------------------
    __frame_width = width of the video
    __frame_height = height of the video
    __video_time = time length of the video
    __N_frames =  total number of frames in the video
    __fps = number of frames per second 
    meta_data = dictionary containing total frames, fps and frame size of video
    reshape = dictionary containing the croping and time slicing of the video

    Methods:
    --------------------------------------------
    
    """

    def __init__(self, filename):

        #checking if the file provided exists or not
        if not os.path.isfile(filename): 
            msg = "The provided path does not exist:\n"
            msg += "\t- path: %s"%filename
            msg += "\t=> Please provide the correct path and try again"
            raise Exception(msg)
        #reading the video file
        self.cap = cv2.VideoCapture(filename)
        #getting the meta data of the video
        self.__frame_width = int(self.cap.get(3))
        self.__frame_height = int(self.cap.get(4))
        self.__video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.__N_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        #calculating the duration of the video        
        self.__duration = self.__N_frames/self.__fps
#        self.__reshape = {'time': (t1,t2), 'size':((x1,x2),(y1,y2))}
    
    #defining a getter for the video resolution
    def resolution(self):
        """Returns the size of frame of the video"""
        
        #getting width and height
        width = self.__frame_width
        height = self.__frame_height
        return (width,height)
    
     #defining  a getter for the video duration
    def video_time(self):
        """Returns the  """
        return self.__video_time
    
    #defining a getter for the total number of frames in the video
    def frame_count(self):
        """ Returns the total number of frames in the video"""
        
        return self.__N_frames
    
    #defining a getter for the frames per second
    def fps(self):
        """Returns the playspeed of the video in frames per second"""
        
        return self.__fps
    
    #defining a getter for the video duration
    def duration(self):
        
        return self.__duration
    
#    def slice_video(self,(t1,t2),((x1,x2),(y1,y2)):
#        
#        self.__reshape = {'time': (t1,t2), 'size' :((x1,x2),(y1,y2))}

    def grayscale(self,path = None,output_name = None,output_type = None):
        """Converts input video file to grayscale, denoises it and saves it as 
        Grayscale.avi
    
        The denoising process is very processor intensive and takes time, 
        but it give very good results. 
        
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
        
        #gray will contain the video path and meta_data will contain the 
        #size of frames, total number of frames and the fps of the video
        gray, meta_data = _comp.colorgray.convertgray(self, path,output_name, output_type)
        #returning the grayscale converted video as a new instance 
        return self.__class__(gray)

    def removebackground(self,path = None,output_name = None,output_type = None):
        """ Removes the background of the video and returns the foreground
        as a new instance"""
        
        #applying the background removal operation
        foreground, meta_data = _comp.background_removal.remove_background(self,path,output_name,output_type)
        return self.__class__(foreground)
    
    def applybinary(self,path = None,output_name = None,output_type = None):
        """ Applies binary threshold to the video and then creates a new instance
        with the binary video"""
        
        #applying the method of binary conversion
        binary,meta_data = _comp.binarythreshold.binary_threshold(self,path,output_name,output_type)
        #returning the binary converted video as a new instance
        return self.__class__(binary)
    
    def play(self):
        """ Displays the video"""
        #playing the video
        _comp.playvideo.play_video(self)
        
    def detectedge(self, path = None, output_name = None, output_type = None):
        """ detects edges in the video and returns the video as a new instance
        """
        
        #applying the edge detection method
        edge, meta_data = _comp.edge_detection.detect_edge(self, path,output_name,output_type)
        #returns the edge detected video as a new instance
        return self.__class__(edge)
    
    def convert2image(self, path = None , image_name = None, image_type = None):
        """Breaks the video up into its frames and saves them"""
        
        #applying the video to image conversion method
        directory = _comp.video_to_img.video2img(self, path, image_name, image_type)
        #returning the directory in which the video is stored
        return directory
        
    def convert2pixel(self):
        """Converts the video into numpy array and returns the array"""
        
        #applying the video to array conversion method
        pixel, meta_data = _comp.video_to_array.video_to_pixel(self)
        
        return pixel, meta_data
    
    def dumpro(self, path = None, output_name = None, output_type = None):
        
        print('Performing Preprocessing on the Video')
        print('Performing Grayscale conversion and Noise Removal')
        
        gray, meta_data = _comp.colorgray.convertgray(self, path, output_name, output_type)
        print('grayscale conversion complete: video stored at: '+str(gray))
        
        print('Performing background removal')
        foreground,meta_data = _comp.background_removal.remove_background(gray, path, output_name, output_type)
        print('background removal complete: video stored at: '+str(foreground))
        
        
