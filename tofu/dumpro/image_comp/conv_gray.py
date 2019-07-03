# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:45:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com

This subroutine requires Opencv3 or higher version
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
    
def conv_gray(im_path, w_dir, shot_name, im_out = None, verb = True):
    """
    This subroutine applies grayscale conversion to a collection of images
    The images are read in native form i.e., without any modification.
    
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
    
    Return
    -----------------------
    im_out:              String
     Path along where the proccessed images are stored  
    """
    if verb == True:
        print('###########################################')
        print('Grayscale Conversion')
        print('###########################################\n')
    #the output directory based on w_dir and shot_name
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_grayscale'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            os.mkdir(im_out)
    #the output directory shown to user
    if verb == True:
        print('output directory is : ', im_out,'\n')
        
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    #total number of frames to process
    duration = len(files)
    #sorting files according to names using lambda function
    #-4 is to remove the extension of the images i.e., .jpg
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('starting grayscale conversion ...')
    
    # loop to read through all the frames and
    # apply grayscale conversion to them
    for time in range(0, duration):
        #converting to path
        filename = im_path + files[time]
        if verb == True:
            stdout.write("\r[%s/%s]" % (time, duration))
            stdout.flush()    
        #reading each file to extract its meta_data
        img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        #grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #generic name of each image
        name = im_out + 'frame' + str(time) + '.jpg'
        #writting the output file
        cv2.imwrite(name, gray)
        
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    if verb == True:
        print('conversion successfull...\n')

    return im_out