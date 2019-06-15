# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:24:36 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
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
    
# dumpro specific
import conv_gray
import denoise
import denoise_col
import to_binary
import rm_background
import reshape_image

def dumpro_img(im_path, w_dir, shot_name, tlim = None, 
               hlim = None, wlim = None, im_out = None, 
               meta_data = None, verb = True):
    """This is the dust movie processing computattion subroutine
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name and meta_data are provided by the image processing 
    class in the core file.
    The verb paramenter is used when this subroutine is used independently.
    Otherwise it is suppressed by the core class.
    
    Parameters
    ----------------------------------
    im_path:          string
     input path where the images are stored
    w_dir:            string
     A working directory where the proccesed images are stored
    shot_name:        String
     The name of the tokomak machine and the shot number. Generally
     follows the nomenclature followed by the lab
    tlim:             tuple
     The time limits for the image files, i.e the frames of interest
    hlim, wlim:       tuple
     The height and width limits of the frame to select the region of interest
    im_out:           string
     The output path for the images after processing.
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
    
    """
    #reshaping images
    print('Reshaping...\n')
    cropped, meta_data, reshape = reshape_image.reshape_image(im_path, w_dir, 
                                                              shot_name, tlim,
                                                              hlim, wlim,
                                                              im_out, 
                                                              meta_data, verb)
    
    #conversion to Grayscale
    print('Grayscale conversion...\n')
    gray, meta_data = conv_gray.conv_gray(cropped, w_dir, shot_name, im_out,
                                          meta_data, verb)
    
    #denoising images
    print('denoising...\n ')
    den_gray, meta_data = denoise.denoise(gray, w_dir, shot_name, im_out,
                                          meta_data, verb)
    
    #removing background
    print('Removing Background...\n')
    back, meta_data = rm_background.rm_back(den_gray, w_dir,shot_name, im_out,
                                            meta_data, verb)
    
    #conversion to binary form
    print('converting to binary')
    bina, meta_data = to_binary.bin_thresh(back, w_dir, shot_name, im_out,
                                           meta_data, verb)
    
    
    return None

