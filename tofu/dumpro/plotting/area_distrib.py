# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:27:01 2019

@author: napra
"""
#nuilt in
import os

#standard
import numpy as np
import matplotlib.pyplot as plt


def get_frame_distrib(area, indt, w_dir, shot_name):
    """This subroutine plots the density distribution of the cluster for each
    frame and their sizes
    
    Parameters:
    ----------------------------------
    area              lists
     A list containing arrays representing each frame that contains the area in
     terms of number of pixels in each clusters
    indt              array
     An index array representing whether there are clusters in each arry or not
    w_dir             string
     The working directory to store the plot
    shot_name         string
     The shot nomenclature from the machine
    """
    #total number of frames in the video 
    duration = len(indt)
    f = plt.figure()     
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    for i in range(0, duration):
        if indt[i] != False:
            ax.plot(i*np.ones((area[i].size,)),area[i],'.')
    ax.set_title('dist')
    print('Plotting density distribution per frame ...')
    plt.savefig(w_dir + shot_name + 'f_dist.jpg')
    return None

def get_distrib(area, indt, total, w_dir, shot_name):
    """Plots the size distribution of the dust particles
    """
    
    duration = len(indt)
    area_array = 0
    for i in range(0, duration):
        if indt[i] != False:
            area_array = np.append(area_array, area[i])
    #plt.plot(area_array,'.')
    bins = np.arange(1, int(area_array.max())+10, 10 )
    plt.yscale('log')
    plt.xscale('log')
    plt.hist(area_array, bins)
    print('Plotting size distribution of dust particles ...')
    plt.savefig(w_dir + shot_name + 'size_dist.png')
    
    return area_array
            