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

def get_area(clus_area, verb = True):
    """This subroutine calculates the average area of all the clusters
    present. This subroutine calls function get_total_area to calculate the 
    area of the clusters. The clusters are divided into two parts. Small and
    big clusters
    
    Parameters:
    --------------------------
    clus_area          list
     A list containing the area of all the clusters
     
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
        print('Calculating average area...')
    #total area 
    area = 0
    area_big = 0
    t_clus_big = 0
    t_clus_small = 0
    #converting list to array
    clus_area = np.array(clus_area)
    #differentiating clusters into big and small
    #getting the total number of small and big clusters
    #getting the total area of big and  small clusters
    area_small, area_big, t_clus_small, t_clus_big = get_total_area(clus_area)
    
    #calculating average area
    avg_area = area_small/t_clus_small
    avg_area_big = area_big/t_clus_big
    
    return avg_area, avg_area_big, t_clus_small, t_clus_big

def get_total_area(area_array):
    for c in area_array:
        if c != []:
            #convertinng list to array
            c = np.asarray(c)
            #applying a size threshold 
            #area > 60 big cluster, #area < 60 small cluster
            d = c[c>=60]
            d = d[d<1000]
            #counting the number of big cluster
            t_clus_big += d.shape[0]
            c = c[c<60]
            c = c[c>0]
            #counting the number of small clusters
            t_clus_small += c.shape[0]
            #adding up all the elements 
            area_small += c.sum()
            area_big += d.sum()
            
    return  area_small, area_big, t_clus_big, t_clus_small