
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
    
    """
    
    def __init__(self,path = "./",):
        
        
        pfe = os.path.join(path)
        try:
            if not os.path.exists(pfe):
                msg = "The provided path does not exist:\n"
                msg += "\t-path: %s"%path
                msg += "\t=> Please provide the correct path and try again" 
                raise Exception(msg)
            else:
                self.cap = cv2.VideoCapture(pfe)
                    
                self.frame_width = int(self.cap.get(3))
                self.frame_height = int(self.cap.get(4))
                self.video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.N_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def Resolution(self):
        
        width = self.frame_width
        height = self.frame_height
        
        return (width,height)

    def Duration(self):
        
        return self.video_time
    
    def Frame_count(self):
        
        return self.N_frames
    
    def FPS(self):
        
        return self.fps
    
#    def grayscale(self):
#        
#        
#    
    def Dumpro(self):
        
        gray = __colorgray.ConvertGray(video_file)
            
        return 'conversion successful'


    