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
    """This subroutine plots the size distribution of the cluster for each
    frame
    """
    duration = len(indt)
    f = plt.figure()     
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    for i in range(0, duration):
        if indt[i] != False:
            ax.plot(i*np.ones((area[i].size,)),area[i],'.')
    ax.set_title('dist')
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
    plt.savefig(w_dir + shot_name + 'size_dist.png')
    
    return area_array
            