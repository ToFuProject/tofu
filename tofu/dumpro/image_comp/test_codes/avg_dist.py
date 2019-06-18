# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:23:29 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#standard
import numpy as np

def get_distance(clus_center, clus_area, t_clusters):
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
     
    Returns:
    ---------------------------
    avg_dist              float
     The average distance between two clusters in adjascent frames
    avg_dist_big          float
     The average distance between two clusters in adjascent frames, taking into
     account the standard deviation
    """
    #list of all cluster distance
    clus_dist = []
    #looping through all the cluster centers
    for t in range(len(clus_center)):
        #checking for frames without any clusters
        if clus_center[t] != []:
            #getting cluster in frame t
            for i in range(len(clus_center[t])):
                center1 = clus_center[t][i]
                #getting cluster in frame t+1
                for j in range(len(clus_center[t+1])):
                    if clus_center[t+1] != []:
                        center2 = clus_center[t+1][j]
                        area1 = (clus_area[t][i])
                        area2 = (clus_area[t+1][j])
                        diff = abs(area1 - area2)
                        if (diff< 0.5 and area1 != 0 and area2 != 0):
                            #calulating distance
                            x2=((center2[0]-center1[0])**2)
                            y2=((center2[1]-center1[1])**2)
                            dist = (x2 + y2)**0.5
                            #adding to cluster distance list
                            clus_dist.append(dist)
                            
    #converting clus_dist to array                    
    clus_dist = np.array(clus_dist)
    #adding up all distances
    tot_dist = clus_dist.sum()
    #calculating average distance
    avg_dist = clus_dist.mean()
    #getting the maximum distance between two clusters
    maxi = clus_dist.max()
    #getting the minimum distance between two clusters
    mini = clus_dist.min()
    #calculating standard deviation
    std_dev = clus_dist.std()
    avg_dist_big = avg_dist + 2*std_dev
                    
    return avg_dist, avg_dist_big

