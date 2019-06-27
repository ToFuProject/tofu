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
import cluster_det
import average_area
import average_distance
#import plotting as _plot

def dumpro_img(im_path, w_dir, shot_name, rate = None, tlim = None, 
               hlim = None, wlim = None, im_out = None, 
               meta_data = None, cen_clus = None, t_clus = None,
               area_clus = None, ang_clus = None, verb = True):
    """This is the dust movie processing computattion subroutine
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name, meta_data, t_clus, area_clus, cen_clus are provided by 
    the image processing class in the core file.
    The verb paramenter can be used for additional information. It runtime
    information on processing, intended only to keep the user informed.
    
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
    cen_clus:         list
     Centers of all the clusters in each frame
    area_clus:        list
     Area of all the clusters in each frame
    t_clus:           list
     Total number of clusters in each frame
    
    """
    if rate == None:
        rate = 1
    #reshaping images
    cropped, meta_data, reshape = reshape_image.reshape_image(im_path, w_dir, 
                                                              shot_name, tlim,
                                                              hlim, wlim,
                                                              im_out, 
                                                              meta_data, verb)
    
    #conversion to Grayscale
    gray, meta_data = conv_gray.conv_gray(cropped, w_dir, shot_name, im_out,
                                          meta_data, verb)
    
    #denoising images
    den_gray, meta_data = denoise.denoise(gray, w_dir, shot_name, im_out,
                                          meta_data, verb)
    
    #removing background
    back, meta_data = rm_background.rm_back(den_gray, w_dir,shot_name, rate, im_out,
                                            meta_data, verb)
    
    #detecting clusters
    clus, meta_data, cen_clus, area_clus, t_clus, ang_clus = cluster_det.det_cluster(back, w_dir, 
                                                                                     shot_name, 
                                                                                     im_out, 
                                                                                     meta_data, 
                                                                                     verb)
    #getting average area
    area, avg_area = average_area.get_area(area_clus, t_clus)
    
    #getting average distance
    avg_dist, avg_dist_big = average_distance.get_distance(cen_clus, area_clus, t_clus)
    
    
    return None

