
# class definitions (object-oriented codes)
#class video is an object containing information about the videofile:
#resolution, size, no of frames

# Built-ins
import os
import warnings

# Standard
import numpy as np
import scipy as scp
import cv2
import matplotlib.pyplot as plt
import datetime
# tofu-specific
#import tf.__version__ as __version__

# dumpro-specific
import _comp
# !!!! never use 'from ... import *'
# Always remain explicit !!! (e.g.: np.cos() and not cos())


##########################################################
##########################################################
#               Title
##########################################################

class Video(object):
    """ A Dust movie processing algorithm designed to track dust and plot their
    trajectories
   
    Provide a filename
   
    """
    
    def __init__(self, filename):
        
        # join is for joining together several path parts
        # os.path.abspath(path) returns the absolute path of your path
        
        # you can split the char str into path + file
        # path, filename = os.path.split(filename)
        
        path = os.path.abspath(path)
        try:
            if not os.path.exists(path): # os.path.isfile
                msg = "The provided path does not exist:\n"
                msg += "\t-path: %s"%path
                msg += "\t=> Please provide the correct path and try again" 
                raise Exception(msg)
            else:
                self.cap = cv2.VideoCapture(path)
                    
                self.__frame_width = int(self.cap.get(3))
                self.__frame_height = int(self.cap.get(4))
                self.__video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.__N_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def resolution(self):
        
        width = self.__frame_width
        height = self.__frame_height
        
        return (width,height)

    @property
    def video_time(self):
        return self.__video_time
    
    def Frame_count(self):
        
        return self.N_frames
    
    def fps(self):
        
        return self.__fps
    
#    def grayscale(self):
#        
#        
#    
    def dumpro(self):
        """ Create a new instance with the grayscale-converted video """
        
        gray = __colorgray.ConvertGray(video_file)
            
        return self.__class__(gray)


    
