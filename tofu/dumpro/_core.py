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
    """ A Dust movie processing algorithm designed to track dust and plot their
    trajectories

    Input:
    --------------------------------------------
    filename = video file along with the path
    time_windows = the time interval to apply the DUMPRO code
    
    Attributes:
    --------------------------------------------
    __frame_width = width of the video
    __frame_height = height of the video
    __video_time = time length of the video
    __N_frames =  total number of frames in the video
    __fps = number of frames per second 

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
        self._time = 0. + 
        self.__duration = self.__N_frames/self.__fps
    
    #defining a getter for the video resolution
    def resolution(self):

        width = self.__frame_width
        height = self.__frame_height
        return (width,height)

     #defining  a getter for the video duration
    def video_time(self):

        return self.__video_time
    
    #defining a getter for the total number of frames in the video
    def frame_count(self):

        return self.__N_frames
    
    #defining a getter for the frames per second
    def fps(self):

        return self.__fps
    
    #defining a getter for the video duration
    def duration(self):
        
        return self.__duration

    def grayscale(self,path = None,output_name = None,output_type = None):
        """ Create a new instance with the grayscale-converted video """

        gray, meta_data = _comp.colorgray.ConvertGray(self, path,output_name, output_type)

        return self.__class__(gray)

    def removebackground(self,path = None,output_name = None,output_type = None):

        foreground = _comp.background_removal.Background_Removal(self,path,output_name,output_type)
        final = 
        return self.__class__(foreground)
    
    def dumpro(self, path = None, output_name = None, output_type = None):
        
        print('Performing Preprocessing on the Video')
        print('Performing Grayscale conversion and Noise Removal')
        
        gray = _comp.colorgray.ConvertGray(self, path, output_name, output_type)
        print('grayscale conversion complete: video stored at: '+str(gray))
        
        print('Performing background removal')
        foreground = _comp.background_removal.Background_Removal(gray, path, output_name, output_type)
        print('background removal complete: video stored at: '+str(foreground))
        
        
