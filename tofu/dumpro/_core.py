
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

def Dumpro(video_file):
    cap = cv2.VideoCapture(video_file)
    print(cap)
    
    gray = __colorgray.ConvertGray(video_file)
    __video_to_imgver2.video2imgconvertor(gray)
    
    return 'conversion successful'

        