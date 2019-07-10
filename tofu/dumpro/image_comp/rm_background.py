# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:45:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
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
    
def rm_back(im_path, w_dir, shot_name, rate = None,
            im_out = None, verb = True):
    """
    This subroutine removes background from a collection of images
    It follows frame by frame subtraction where the previous frame is 
    subtracted from the successive frame
    The images are read in original form i.e., without any modifications
    For more information consult:
    
    1. https://docs.opencv.org/3.4/dd/d4d/tutorial_js_image_arithmetics.html
    2. https://docs.opencv.org/3.1.0/d1/dc5/tutorial_background_subtraction.html
    3. https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name and rate are provided by the image processing 
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
    rate:             integer
     if rate = 0 then a guassian mixture based background/foreground 
     segmentation algorithm is used and if rate = 1 then it is a slow camera 
     and the method of background removal will be frame by frame subtraction  
     By default rate is set at zero. Some information loss happens in the frame
     by frame subtraction is used but we get more false positives for 
     the guassian based method.
    
    Return
    -----------------------
    im_out:              String
     Path along where the proccessed images are stored
    """

    if verb == True:
        print('###########################################')
        print('Background Removal')
        print('###########################################\n')
    #reading the output directory
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_frground'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            #creating output directory using w_dir and shot_name
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
    
    if rate == 0:
        if verb == True:
            print('Using background subtractor MOG')
    if rate == 1:
        if verb == True:
            print('Using frame by frame subtraction method')
    
    if rate == None:
        rate = 0
        if verb == True:
            print('Using background subtractor MOG')
    
    if verb == True:
        print('subtracting background...')
        print('Reading the image files ...')
        print('Files processing...')
    
    #background subtractor method
    back = cv2.bgsegm.createBackgroundSubtractorMOG()
    #looping through the video
    for time in range(0, duration-1):
        #converting to path
        f_name1 = im_path + files[time]
        #dynamic printing
        if verb == True:
            stdout.write("\r[%s/%s]" % (time, duration))
            stdout.flush()   
        if rate == 0:
            #reading image file
            img1 = cv2.imread(f_name1,cv2.IMREAD_UNCHANGED)
            #removing backgroud
            dst = back.apply(img1)
        elif rate == 1:
            img1 = cv2.imread(f_name1,cv2.IMREAD_UNCHANGED)
            f_name2 = im_path + files[time+1]
            img2 = cv2.imread(f_name2,cv2.IMREAD_UNCHANGED)
            #performing frame by frame subtraction
            dst = cv2.subtract(img1, img2)

        #generic name of each image
        name = im_out + 'frame' + str(time) + '.jpg'
        #writting the output file
        cv2.imwrite(name, dst)
    
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    if verb == True:
        print('background subtraction successfull...\n')
    
    return im_out