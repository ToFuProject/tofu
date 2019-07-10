# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:39:31 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

Input a grayscale image(image of only one channel) and get the corresponding 
intensity of each pixel

User must have Opencv3 or higher installed, for using this subroutine
"""
#Built-in
import os

#Standard
import numpy as np

#more special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

def read_pixel(Image):
    """ Converts input Grayscale Image into Numpy Array and then returns the 
    array
    
    Parameters
    -----------------
    Image:          .jpg/png/jpeg
     Input image along with its path passed in as argument
     
    Return
    -----------------
    pixel:          Numpy array 
     Numpy array having the dimention of the Image frame and the corresponding
     value of the intensity(since it is a grayscale image at the corresponding)
     position
    """
    
    #reading the image, O ndicates that the image is in single channel which
    #means it is a grayscale image
    try:
        pixel = cv2.imread(Image,0)
    except IOError:
        msg = "The provided file does not exist:\n"
        msg += "\t-path: %s"%path
        msg += "\t=> Please check the path or the filename" 
        Print(msg)
    
    #returning the array
    return pixel
            