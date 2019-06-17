# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:12:47 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

#standard
import numpy as np

def get_area(clus_area, t_clusters):
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
    area = 0
    clus_area = np.array(clus_area)
    for c in clus_area:
        if c != []:
            c = np.array(c)
            area += c.sum()
    
    t_clusters = np.array(t_clusters)
    
    n_clus = t_clusters.sum()
    print(n_clus)
    avg_area = area/n_clus
    return area, avg_area