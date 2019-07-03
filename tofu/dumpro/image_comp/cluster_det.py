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

def det_cluster(im_path, w_dir, shot_name, im_out = None, verb = True):
    """
    This subroutine detects clusters in a collection binary images
    The images are read in native form i.e., without any modification.
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir and shot_name are provided by the image processing 
    class in the core file.
    The verb paramenter is used when thsi subroutine is used independently.
    Otherwise it is suppressed by the core class.
    
    for more information:
    1. Opencv Contour Features(Rotated Rectrangle)
    
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
    im_out:           String
     Path along where the proccessed images are stored  
    cen_clus:         list
     A list of lists containing all the centers of clusters in each frame
    area_clus:        list
     A list of lists containing the area of all the clusters in each frame
    t_clusters:       array
     An array contaning the totaL number of clusters in each frame
    ang_cluster:      list
     A list containing the angular orientation of each cluster
    indt:             array
    """
    if verb == True:
        print('###########################################')
        print('Detecting Clusters')
        print('###########################################\n')
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
    #duration    
    nt = len(files)
    
    if verb == True:
        print('detecting clusters...')
    #to store the centroid of each cluster
    cen_clus = [None for _ in range(0,nt)]
    #to store the size of each cluster
    area_clus = [None for _ in range(0,nt)]
    #to store the contour infomation of each frame
    t_clusters = np.zeros((nt,),dtype = int)
    #to store angle
    ang_cluster = [None for _ in range(0,nt)]
    #indices array(True if cluster present or else False)
    indt = np.ones((nt,), dtype = bool)
    #looping throuah all the file names in the list
    #converting them to image path
    #and applying contour detection algorithm to it
    for tt in range(0,nt):
        #converting to path
        filename = im_path + files[tt]
        #dynamic printing
        if verb == True:
            stdout.write("\r[%s/%s]" % (tt, nt))
            stdout.flush() 
        #reading each binary image to extract its meta_data
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #detecting contours
        ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        #total number of clusters in each frame
        t_clusters[tt] = len(contours)
        if t_clusters[tt] == 0:
            indt[tt] = False
            continue
        #array for area for each cluster for each frame
        area_frame = np.zeros((t_clusters[tt],),dtype = float)
        #aray for center for each each cluster for each frame
        center_frame = np.zeros((t_clusters[tt],2),dtype = int)
        #array for angular orientation for each cluster for each frame
        angle_frame = np.zeros((t_clusters[tt],),dtype = float)
        
        #looping over each contours
        for ii in range(0,t_clusters[tt]):
            c = contours[ii]
            x, y, w, h = cv2.boundingRect(c)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            angle = rect[2]
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
                area = cv2.contourArea(c)
            area_frame[ii] = area
            center_frame[ii,:] = center
            angle_frame[ii] = angle
            
        area_clus[tt] = area_frame
        cen_clus[tt]  = center_frame
        ang_cluster[tt] = angle_frame
        #drawing contours
        #cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        #generic name of each image
        name = im_out + 'frame' + str(tt) + '.jpg'
        #writting the output file
        cv2.imwrite(name, img)
    
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    cv2.destroyAllWindows
    
    return im_out, cen_clus, area_clus, t_clusters, ang_cluster, indt

