# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:45:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
This subroutine requires opencv3 or higher
"""

# Built-in
import os
from sys import stdout
from time import sleep

# Standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def denoise_col(im_path, w_dir, shot_name, im_out = None, verb = True):
    """
    This subroutine applies denoising to colored images
    The images are read in original form i.e., without any modifications
    The denoising algorithm follows non local means of denoising. For more 
    information look at the following resources:
    
    1. https://docs.opencv.org/trunk/d5/d69/tutorial_py_non_local_means.html
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir and shot_name are provided by the image processing 
    class in the core file.
    
    Parameters
    -----------------------
    im_path:          string
     input path where the images are stored
    w_dir:            string
     A working directory where the proccesed images are stored
    shot_name:        String
     The name of the tokomak machine and the shot number. Generally
     follows the nomenclature followed by the lab
    disp              boolean
     to display the frames set equal to True. By default is set to True
    
    Return
    -----------------------
    im_out:              String
     Path along where the proccessed images are stored  
    """
    if verb == True:
        print('###########################################')
        print('Denoising Images')
        print('###########################################\n')
    #reading output path
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_denoise_col'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            #creating the output directory using w_dir and shot_name
            os.mkdir(im_out)
    
    if verb == True:
        print('output directory is : ', im_out,'\n')
    
    #describing an empty list that will later contain all the frames
    #frame_array = []
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    #total number of frames to process
    duration = len(files)
    #sorting files according to names using lambda function
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('denoising images...')
    
    #looping through files and applying denoising to them
    for time in range(0,duration):
        #converting to path
        filename = im_path + files[time]
        if verb == True:
            stdout.write("\r[%s/%s]" % (time, duration))
            stdout.flush()   
        #reading each file to extract its meta_data
        img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        #grayscale conversion
        dst = cv2.fastNlMeansDenoisingColored(img,None,5,21,7)
        #generic name of each image
        name = im_out + 'frame' + str(time) + '.jpg'
        #writting the output file
        cv2.imwrite(name, dst)
    
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    #frame_array.append(img)
    
    if verb == True:
        print('images denoised...\n')

    return im_out