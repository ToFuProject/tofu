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
    
import PIL

def read_pixel(Image):
    """ Converts input Grayscale Image into Numpy Array and then returns the 
    array
    
    Parameters
    -----------------
    Image:          .jpg/png/jpeg
     Input image along with its path passed in as argument
     
    Return
    -----------------
    Pixel:          Numpy array 
     Numpy array having the dimention of the Image frame and the corresponding
     value of the intensity(since it is a grayscale image at the corresponding)
     position
    """
    
    #reading the image, O ndicates that the image is in single channel which
    #means it is a grayscale image
    try:
        image = cv2.imread(Image,0)
    except IOError:
        msg = "The provided file does not exist:\n"
        msg += "\t-path: %s"%path
        msg += "\t=> Please check the path or the filename" 
        Print(msg)
    
    #Finding the dimensions of the frame 
    height, width = image.shape
    
    #Creating array with the dimensions of the image where each element 
    #represents the corresponding pixel
    pixel = np.ndarray((image.shape[0],image.shape[1]), dtype = int)
    
    #Looping through the image assigning values to the array
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            pixel[i][j] = image.item(i, j)

    #returning the array
    return pixel
            