# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:55:37 2019

@author: napra
"""
import numpy as np
import matplotlib.pyplot as plt

def num_dist(infocluster, w_dir, shot_name):
    """This subroutine plots the number density distribution of the cluster for each
    frame
    
    Parameters:
    ----------------------------------
    infocluster       dictionary
     A dictionary containing all the information of the clusters
    w_dir             string
     The working directory to store the plot
    shot_name         string
     The shot nomenclature from the machine
    """
    #total clusters from infocluster
    total = infocluster.get('total')
    #indices array from infocluster
    indt = infocluster.get('indt')
    #total number of frames in the video 
    duration = len(indt)
    #creating the figure
    f = plt.figure()
    #declaring the axes
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    #axes labels
    ax.set_xlabel('frame number')
    ax.set_ylabel('number of clusters present')
    #looping over all frames
    for ii in range(0, duration):
            ax.plot(ii, total[ii], 'r.')
    #setting title to plot
    ax.set_title('Framewise cluster distribution')
    #providing information to users
    print('Plotting frame-wise cluster density distribution ...\n')
    #saving the plot file as a jpg image
    plt.savefig(w_dir + shot_name + 'f_density.jpg')
    
    return None
