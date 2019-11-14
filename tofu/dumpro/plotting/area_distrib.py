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


def get_frame_distrib(infocluster, w_dir, shot_name):
    """This subroutine plots the density distribution of the cluster for each
    frame and their sizes
    
    Parameters:
    ----------------------------------
    infocluster       dictionary
     A dictionary containing all the information of the clusters
    w_dir             string
     The working directory to store the plot
    shot_name         string
     The shot nomenclature from the machine
    """
    #gathering information from infocluster
    area = infocluster.get('area')
    indt = infocluster.get('indt')
    #total number of frames in the video 
    duration = len(indt)
    #declaring figure
    f = plt.figure()
    #defining axes
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    #setting up labels
    ax.set_xlabel('frame number')
    ax.set_ylabel('cluster size')
    #plotting information    
    for i in range(0, duration):
        if indt[i] != False:
            ax.plot(i*np.ones((area[i].size,)),area[i],'r.')
    #setting plot title
    ax.set_title('Framewise cluster size distribution')
    print('Plotting frame-wise size distribution ... \n')
    #saving plot as an image file
    plt.savefig(w_dir + shot_name + 'f_dist.jpg')
    return None


def get_distrib(infocluster, w_dir, shot_name):
    """Plots the size distribution of the dust particles
    
    Parameters:
    ----------------------------------
    infocluster       dictionary
     A dictionary containing all the information of the clusters
    w_dir             string
     The working directory to store the plot
    shot_name         string
     The shot nomenclature from the machine
    """
    #extracting information from infocluster
    total = infocluster.get('total')
    area = infocluster.get('area')
    indt = infocluster.get('indt')
    #total number of frames present
    duration = len(indt)
    #an empty area array
    area_array = 0
    #add all areas to the array
    for i in range(0, duration):
        if indt[i] != False:
            area_array = np.append(area_array, area[i])
    #describe bins for histogram
    bins = np.arange(1, int(area_array.max())+10, 10 )
    #x and y scales
    plt.yscale('log')
    plt.xscale('log')
    #x and y labels
    plt.xlabel('Size of cluster')
    plt.ylabel('number of clusters present')
    #plotting figure
    plt.hist(area_array, bins)
    print('Plotting size distribution of dust particles ... \n')
    #saving plot as an image file
    plt.savefig(w_dir + shot_name + 'size_dist.png')

    return area_array
            