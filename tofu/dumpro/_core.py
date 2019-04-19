
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

#more special packages
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
# tofu-specific
#import tf.__version__ as __version__

# dumpro-specific
import computation as _comp
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
    
    def __init__(self, filename, time_window):
        
        # join is for joining together several path parts
        # os.path.abspath(path) returns the absolute path of your path
        
        # you can split the char str into path + file
        # path, filename = os.path.split(filename)
        
#        path = os.path.abspath(path)
#        try:
#            if not os.path.isfile(path): # os.path.isfile(path)
#                msg = "The provided path does not exist:\n"
#                msg += "\t-path: %s"%path
#                msg += "\t=> Please provide the correct path and try again" 
#                raise Exception(msg)
#                
            self.cap = cv2.VideoCapture(filename)
                    
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
    def grayscale(self,path,output_name,output_type):
        """ Create a new instance with the grayscale-converted video """
        
        gray = _comp.__colorgray.ConvertGray(self,path,output_name,output_type)
            
        return self.__class__(gray)
    
    def removebackground(self):
        
        foreground = _comp.__background_removal.Background_Removal(self)
        
        return self.__class__(foreground)
    