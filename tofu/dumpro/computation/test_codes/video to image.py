# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:43:55 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

#built-ins
import os
import glob

#standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def image2video(path):
    
    
