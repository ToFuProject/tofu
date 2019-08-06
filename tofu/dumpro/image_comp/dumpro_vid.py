# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:20:48 2019

@author: Arpan Khandelwal
@email: napraarpan@gmail.com
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
from . import guassian_blur
from . import vid2img

def dumpro_vid(filename, w_dir, shot_name, rate = None, tlim = None,
               hlim = None, wlim = None, blur = True, im_out = None, 
               meta_data = None, verb = True):
    """This is the dust movie processing computattion subroutine
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir, shot_name, infocluster are provided by the video 
    processing class in the core file.
    The verb paramenter can be used for additional information. It runtime
    information on processing, intended only to keep the user informed.
    
    Parameters
    ----------------------------------
    filename:         string
     input path of the video
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
    blur:             boolean
     if True, will use the smoothening subroutine
    meta_data:        dictionary
     A dictionary containing meta_data of the film
    
    Returns
    ----------------------------------
    infocluster:     dictionary
     A dictionary containing all the information available on clusters present
     in a frame
    reshape:         dictionary
     A dictionary containing information on the reshaping of the frames
    imdir:           dictioanry
     A dictionary containing path of all the different images created during
     runtime  
    """
    #dictionary to store information on cluster
    infocluster = {}
    #dictionary to store image path
    imdir = {}
    
    if rate == None:
        fps = meta_data.get('fps')
        if fps < 10:
            rate = 1
        else:
            rate = 0
        
    #converting video to image
    im_path, m_data, s_name = vid2img.video2img(filename, w_dir, shot_name,
                                                meta_data, verb)
    
    imdir['original'] = im_path
    
    if (hlim == None and tlim == None and wlim == None):
        cropped = im_path
        reshape = {}
    else:
        #reshaping images
        cropped, reshape = reshape_image.reshape_image(im_path, w_dir, 
                                                       shot_name, tlim,
                                                       hlim, wlim,
                                                       im_out, 
                                                       verb)
    imdir['reshaped'] = cropped
    #conversion to Grayscale
    gray = conv_gray.conv_gray(cropped, w_dir, shot_name, im_out, verb)
    imdir['grayscale'] = gray
    
    #denoising images
    den_gray = denoise.denoise(gray, w_dir, shot_name, im_out, verb)
    imdir['denoised'] = den_gray
    
    #removing background
    back = rm_background.rm_back(den_gray, w_dir,shot_name, rate, im_out, verb)
    imdir['foreground'] = back
    
    if blur == True:
        #Smoothing images
        blur = guassian_blur.blur_img(back, w_dir, shot_name, im_out, verb)
        imdir['blur'] = blur
    else:
        blur = back
    
    #detecting clusters
    clus, center, area, total, angle, indt = cluster_det.det_cluster(blur, w_dir,
                                                                     shot_name, 
                                                                     im_out,
                                                                     verb)
    imdir['clusters'] = clus
    
    #adding information to infocluster
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

    return infocluster, reshape, imdir