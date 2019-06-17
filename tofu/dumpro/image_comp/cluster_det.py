#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:50:11 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#nuilt in
import os
from sys import stdout
from time import sleep

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

def det_cluster(im_path, w_dir, shot_name, im_out = None, meta_data = None, verb = True):
    """
    This subroutine detects clusters in a collection binary images
    The images are read in native form i.e., without any modification.
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name and meta_data are provided by the image processing 
    class in the core file.
    The verb paramenter is used when thsi subroutine is used independently.
    Otherwise it is suppressed by the core class.
    
    Parameters
    -----------------------
    im_path:          string
     input path where the images are stored
    w_dir:            string
     A working directory where the proccesed images are stored
    shot_name:        String
     The name of the tokomak machine and the shot number. Generally
     follows the nomenclature followed by the lab
    meta_data:        dictionary
     A dictionary containing all the video meta_data. By default it is None
     But if the user inputs some keys into the dictionary, the code will use 
     the information from the dictionary and fill in the missing gaps if
     required
     meta_data has information on total number of frames, demension, fps and 
     the four character code of the video
    
    Return
    -----------------------
    im_out:              String
     Path along where the proccessed images are stored  
    meta_data:        dictionary
     A dictionary containing the meta data of the video.
    """
    
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
        print('detecting clusters...\n')
    #to store the barycenter of each cluster
    cen_clus = []
    #to store the size of each cluster
    area_clus = []
    #to store the contour infomation of each frame
    contour = []
    # loop to read through all the images and
    # apply grayscale conversion to them
    f_count = 1
    for i in range(len(files)):
        #converting to path
        filename = im_path + files[i]
        #dynamic printing
        if verb == True:
            stdout.write("\r[%s/%s]" % (f_count, len(files)))
            stdout.flush() 
        #reading each binary image to extract its meta_data
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #detecting contours
        ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        area_frame = []
        cen_frame = []
        #looping over contours
        for c in contours:
            
            x, y, w, h = cv2.boundingRect(c)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            #draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (255, 255, 255))
            
            #getting moments
            M = cv2.moments(c)
            #centroid calculation
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center = cx,cy
                area = cv2.contourArea(c)
                
            else:
                #center estimation if m00 == 0
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = int(x), int(y)
                radius = int(radius)
                area = 3.14*(radius**2)
                
            cen_frame.append(center)
            area_frame.append(area)
        cen_clus.append(cen_frame)
        area_clus.append(area_frame)
        #drawing contours
        #cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        #generic name of each image
        name = im_out + 'frame' + str(f_count) + '.jpg'
        #writting the output file
        cv2.imwrite(name, img)
        height,width = img.shape[0],img.shape[1]
        size = (height, width)
        #providing information to user
        f_count += 1
    
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    if verb == True:
        print('Reading meta_data...')
        
    if meta_data == None:
        #defining the four character code
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #defining the frame dimensions
        frame_width = width
        frame_height = height
        #defining the fps
        fps = 25
        #defining the total number of frames
        N_frames = len(files)
        #defining the meta_data dictionary
        meta_data = {'fps' : fps, 'frame_height' : frame_height, 
                     'frame_width' : frame_width, 'fourcc' : fourcc,
                     'N_frames' : N_frames}
    else:
        #describing the four character code      

        fourcc = meta_data.get('fourcc', cv2.VideoWriter_fourcc(*'MJPG'))
        if 'fourcc' not in meta_data:
            meta_data['fourcc'] = fourcc
        
        #describing the frame width
        frame_width = meta_data.get('frame_width', width)
        if 'frame_width' not in meta_data:
            meta_data['frame_width'] = frame_width
        
        #describing the frame height
        frame_height = meta_data.get('frame_height', height)
        if 'frame_height' not in meta_data:
            meta_data['frame_height'] = frame_height
            
        #describing the speed of the video in frames per second 
        fps = meta_data.get('fps', 25)
        if 'fps' not in meta_data:
            meta_data['fps'] = fps

        #describing the total number of frames in the video
        N_frames = meta_data.get('N_frames', len(files))
        if 'N_frames' not in meta_data:
            meta_data['N_frames'] = N_frames
            
    if verb == True:
        print('meta_data read successfully ...\n')
    
    #frame_array.append(img)
    cv2.destroyAllWindows
    
    return im_out, meta_data, cen_clus, area_clus

