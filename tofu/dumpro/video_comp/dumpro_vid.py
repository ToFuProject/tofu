# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:05:49 2019

@author: Arpan Khandelwal
@email: napraarpan@gmail.com
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

#dumpro-specific
from . import cluster_det
from . import average_area
from . import get_distance

def dumpro_vid(filename, disp_choice = 'n', tlim = None, hlim = None, wlim = None,
               disp_option = None, meta_data = None, verb = True):
    
    

