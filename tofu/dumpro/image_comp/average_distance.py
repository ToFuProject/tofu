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
    clus_dist = []
    for i in range(len(clus_center)):
        print(i)
        #checking for frames without any clusters
        if clus_center[i] != []:
            for c in clus_center[i]:
                center1 = c
                print(center1)
                for c in clus_center[i+1]:
                    if clus_center[i+1] != []:
                        center2 = c
                        print(center2)
                        x2=((center2[0]-center1[0])**2)
                        y2=((center2[1]-center1[1])**2)
         
                        dist = (x2 + y2)**0.5
                        print(dist)
                        clus_dist.append(dist)
                        a = input('press c :')
                        if a == 'c':
                            continue
                        
    clus_dist = np.array(clus_dist)
    tot_dist = clus_dist.sum()
    print(tot_dist)
    avg_dist = clus_dist.mean()
    maxi = clus_dist.max()
    print(maxi)
    mini = clus_dist.min()
    print(mini)
    std_dev = maxi - mini
    print(std_dev)
    print(len(clus_dist))
    avg_dist_big = avg_dist + 2*std_dev
                    
    return avg_dist, avg_dist_big
