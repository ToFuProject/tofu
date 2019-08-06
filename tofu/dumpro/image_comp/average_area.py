# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:12:47 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#built-in
import matplotlib.pyplot as plt

#standard
import numpy as np

def get_area(clus_area, t_cluster, indt, verb = True):
    """This subroutine calculates the average area of all the clusters
    present. This subroutine calls function get_total_area to calculate the 
    area of the clusters. The clusters are divided into two parts. Small and
    big clusters

    Parameters:
    --------------------------
    clus_area          list
     A list containing the area of all the clusters
    t_cluster          array
     An array containing the total number of cluster in each frame
    indt               array
     An array containing True if cluster in present in frame or else False
     
    Return:
    --------------------------
    avg_area           float
     The average area of all the small clusters
    avg_area_big       float
     The average area of the big clusters
    t_clus_small       int
     The total number of small clusters
    t_clus_big         int
     The total number of big clusters
    """
    if verb == True:
        print('Calculating average area...\n')
    #differentiating clusters into big and small
    #getting the total number of small and big clusters
    #getting the total area of big and  small clusters
    area_small,area_big,t_clus_small,t_clus_big = get_total_area(clus_area,indt,t_cluster)
    
    #calculating average area
    if t_clus_small != 0:
        avg_area = area_small/t_clus_small
    else: 
        avg_area = 0
        
    if t_clus_big != 0:
        avg_area_big = area_big/t_clus_big
    else:
        avg_area_big = 0
    return avg_area, avg_area_big, t_clus_small, t_clus_big

def get_total_area(area_array, indt, t_cluster):
    a_small = 0
    a_big = 0
    t_big = 0
    t_small = 0
    for tt in range(len(t_cluster)):
        if indt[tt] == False:
            continue
        else:
            c = area_array[tt]
            #convertinng list to array
            #applying a size threshold 
            #area > 60 big cluster, #area < 60 small cluster
            d = c[(c > 60) & (c < 1000)]
            #counting the number of big cluster
            t_big += d.shape[0]
            c = c[(c > 0) & (c <= 60)]
            #counting the number of small clusters
            t_small += c.shape[0]
            #adding up all the elements 
            a_small += c.sum()
            a_big += d.sum()
            
    return  a_small, a_big, t_small, t_big