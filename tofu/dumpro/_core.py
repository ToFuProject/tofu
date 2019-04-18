
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
import __colorgray
import __video_to_imgver2
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
        
        self.cap = cv2.VideoCapture(path)
        
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.video_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.N_frames = self.cap.get(cv2.CAP_POP_POS_FRAMES)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
    def Resolution(self):
        
        width = self.frame_width
        height = self.frame_height
        
        return (width,height)

    def 
        
#class Frame(object):
#    """ An Object containing information on a single frame like it's height,
#    width, number of pixels and the values of each of those pixels.
#    Using a numpy array to store the values of the pixels
#    
#    """
#    
#    def __init__(self, path ="./", :):
#        
def Dumpro(object):
    try:
        cap = cv2.VideoCapture(video_file)
    except IOError:
        print("file not found, PLease check path ot file name")
        
    gray = __colorgray.ConvertGray(video_file)
    __video_to_imgver2.video2imgconvertor(gray)
    
    return 'conversion successful'


    