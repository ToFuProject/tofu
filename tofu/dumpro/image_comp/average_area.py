# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:12:47 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#built-in
from sys import stdout
from time import sleep

import matplotlib.pyplot as plt

#standard
import numpy as np

def get_area(clus_area, t_clusters, verb = True):
    """This subroutine calculates the average area of all the clusters
    present
    
    Parameters:
    --------------------------
    clus_area          list
     A list containing the area of all the clusters
    t_clusters         list
     A list contaning the total clusters in each frame
     
    Return:
    --------------------------
    area               float
     The total area of all the clusters
    avg_area           float
     The average area of the clusters
    """
    if verb == True:
        print('Calculating average area...')
    #total area 
    area = 0
    area_big = 0
    t_clus_big = 0
    t_clus_small = 0
    #converting list to array
    clus_area = np.array(clus_area)
    #looping through array
    for c in clus_area:
        #if c is empty list, go to next frame
        if c != []:
            #converting list to array
            c = np.array(c)
            #applying a size threshold 
            #area > 60 big cluster, #area < 60 small cluster
            d = c[c>=60]
            #counting the number of big cluster
            t_clus_big += d.shape[0]
            c = c[c<60]
            #counting the number of small clusters
            t_clus_small += c.shape[0]
            #adding up all the elements 
            area += c.sum()
            area_big += d.sum()
    
    #converting total cluster list to array
    t_clusters = np.array(t_clusters)
    #total number of cluster present in shot
    n_clus = t_clusters.sum()
    #calculating average area
    avg_area = area/t_clus_small
    avg_area_big = area_big/t_clus_big
    
    return avg_area, avg_area_big, t_clus_small, t_clus_big