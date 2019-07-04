# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:23:29 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#built-in
from sys import stdout
from time import sleep

#standard
import numpy as np

def get_distance(clus_center, clus_area, t_clusters, indt, verb = True):
    """This subroutine calculate the average distance between clusters of one
    frame and the clusters of the next frame.
    
    Parameters:
    --------------------------
    clus_center:          list
     A list containing the center points of the clusters present in the frame
    clus_area:            list
     A list contaning the area of the clusters on each frame
    t_clusters            list
     A list contaning the total number of clusters in each frame
    indt                  array
     An array contaning boolean values representing whether there is a cluster
     present in the frame or not
     
    Returns:
    ---------------------------
    clus_dist             list
     The distance between clusters in current frame to clusters in next frame
     which is stored as a 2D array.
    """
    if verb == True:
        print('Calculating average distance...')
    #calculating the number of frames
    nt = len(clus_center)
    #creating clust_dist list and filling it with default values
    clust_dist = [np.full((t_clusters[ii], t_clusters[ii+1]), np.nan)
                  for ii in range(0,nt-1)]
    #looping through all the frames
    for tt in range(0, nt-1):
        if verb == True:
            stdout.write("\r[%s/%s]" % (tt, nt-2))
            stdout.flush()    
        #checking for frames without any clusters
        if indt[tt] == False or indt[tt+1] == False:
            continue
        
        #getting cluster in the current frame
        for ii in range(0,t_clusters[tt]):
            # calulating distance for each cluster in the next frame
            clust_dist[tt][ii,:] = np.hypot(clus_center[tt+1][:,0]-clus_center[tt][ii,0],
                                            clus_center[tt+1][:,1]-clus_center[tt][ii,1])
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    return clust_dist
