# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:45:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
This subroutine requires opencv3 and higher
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
    
def bin_thresh(im_path, w_dir, shot_name, im_out = None, verb = True):
    """
    This subroutine converts a collection of images to binary
    The images are read in grayscale form i.e., in single channel mode
    For more information consult:
    
    1. https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html
    2. https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    
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
        print('Binary Conversion')
        print('###########################################\n')
    
    #reading the output directory
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_binary'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            #creating output directory using w_dir and shotname
            os.mkdir(im_out)
    
    if verb == True:
        print('output directory is : ', im_out,'\n')
    
    #describing an empty list that will later contain all the frames
    #frame_array = []
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    #duration in terms of number of frames
    duration = len(files)
    #sorting files according to names using lambda function
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('performing binary conversion...')
    
    #looping through the files
    for time in range(0, duration):
        #converting to path
        filename = im_path + files[time]
        #dynamic printing
        if verb == True:
            stdout.write("\r[%s/%s]" % (time, duration))
            stdout.flush()
        #reading each file to extract its meta_data
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        #grayscale conversion
        ret, out = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        #generic name of each image
        name = im_out + 'frame' + str(time) + '.jpg'
        #writting the output file
        cv2.imwrite(name, out)
        
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    if verb == True:
        print('binary conversion successful.../n')
    
    return im_out