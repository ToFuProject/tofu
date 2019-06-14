#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:50:11 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#nuilt in
import os

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

def det_cluster(im_path, w_dir, shot_name, im_out = None, meta_data = None, verb = True):
    
    #the output directory based on w_dir and shot_name
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_clu'
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
    
    #sorting files according to names using lambda function
    #-4 is to remove the extension of the images i.e., .jpg
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('starting grayscale conversion ...\n')
        print('The following files have been read ...')
    
    # loop to read through all the images and
    # apply grayscale conversion to them
    f_count = 1
    for i in range(len(files)):
        #converting to path
        filename = im_path + files[i]
        if verb == True:
            print(filename)
        #reading each file to extract its meta_data
        img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center = cx,cy
                
            else:
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = int(x), int(y)
                radius = int(radius)

        #generic name of each image
        name = im_out + 'frame' + str(f_count) + '.jpg'
        #writting the output file
        cv2.imwrite(name, )
        height,width = img.shape[0],img.shape[1]
        size = (height, width)
        #providing information to user
        f_count += 1
    
    #frame_array.append(img)

