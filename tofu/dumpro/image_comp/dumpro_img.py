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
from . import conv_gray
from . import denoise
from . import denoise_col
from . import to_binary
from . import rm_background
from . import reshape_image
from . import cluster_det
from . import average_area
from . import get_distance
#import plotting as _plot

def dumpro_img(im_path, w_dir, shot_name, infocluster, rate = None, tlim = None, 
               hlim = None, wlim = None, im_out = None, meta_data = None, 
               verb = True):
    """This is the dust movie processing computattion subroutine
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name, t_clus, area_clus, cen_clus are provided by the image 
    processing class in the core file.
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
    cen_clus:         list
     Centers of all the clusters in each frame
    area_clus:        list
     Area of all the clusters in each frame
    t_clus:           list
     Total number of clusters in each frame
    
    """    
    if rate == None:
        rate = 0
    #reshaping images
    cropped, reshape = reshape_image.reshape_image(im_path, w_dir, 
                                                   shot_name, tlim,
                                                   hlim, wlim,
                                                   im_out, 
                                                   verb)
    
    #conversion to Grayscale
    gray = conv_gray.conv_gray(cropped, w_dir, shot_name, im_out, verb)
    
    #denoising images
    den_gray = denoise.denoise(gray, w_dir, shot_name, im_out, verb)
    
    #removing background
    back = rm_background.rm_back(den_gray, w_dir,shot_name, rate, im_out, verb)
    
    #detecting clusters
    clus, center, area, total, angle, indt = cluster_det.det_cluster(back, w_dir,
                                                                     shot_name, 
                                                                     im_out,
                                                                     verb)
    
    infocluster['center'] = center
    infocluster['area'] = area
    infocluster['total'] = total
    infocluster['angle'] = angle
    infocluster['indt'] = indt
    
    #getting average area
    #differentiates cluster into two parts small and big
    #return two different averages (one for small clusters and one for big)
    avg_small, avg_big, t_clus_small, t_clus_big = average_area.get_area(area, total, indt)
    infocluster['avg_area_small'] = avg_small
    infocluster['avg_area_big'] = avg_big
    
    #getting distances between clusters in the current frame to clusters in 
    #the next frame
    clus_dist = get_distance.get_distance(center, area, total, indt)
    infocluster['distances'] = clus_dist

    return infocluster

