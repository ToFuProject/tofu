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

def get_distance(clus_center, clus_area, t_clusters, verb = True):
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
    if verb == True:
        print('Calculating average distance...')
    #list of all cluster distance
    clus_dist = []
    #looping through all the cluster centers
    for t in range(len(clus_center)-1):
        if verb == True:
            stdout.write("\r[%s/%s]" % (t, len(clus_center)-2))
            stdout.flush()    
        #checking for frames without any clusters
        if clus_center[t] == []:
            continue
        frame_dist = []
        #getting cluster in frame t
        for i in range(len(clus_center[t])):
            #getting the center of ith cluster
            center1 = clus_center[t][i]
            #getting all the clusters in frame t+1
            center2 = np.asarray(clus_center[t+1])

            # calulating distance for each cluster in 
            dist = np.hypot(center2[:,0]-center1[0],center2[:,1]-center1[1])
            print(dist)
            # adding to cluster distance list
            frame_dist[i].append(dist)
        clus_dist[t].append(frame_dist)
            
#            for j in range(len(clus_center[t+1])):
#                if clus_center[t+1] != []
#                    center2 = clus_center[t+1][j]
#                    #comparing area betweem two clusters
#                    area2 = (clus_area[t+1][j])
#                    #area is zero ignore 
#                    if area1 != 0 and area2 != 0:
#                        diff = abs(area1 - area2)
#                        #difference in area threshold 
#                        if (diff< 0.5):
#                            dist = np.hypot(center2[0]-center1[0],center2[1]-center1[1])
#                            #calulating distance
#                            #adding to cluster distance list
#                            clus_dist.append(dist)
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
#    #converting clus_dist to array                    
#    clus_dist = np.array(clus_dist)
#    #calculating average distance
#    avg_dist = dist.mean()
#    #getting the maximum distance between two clusters
#    maxi = clus_dist.max()
#    #getting the minimum distance between two clusters
#    mini = clus_dist.min()
#    #calculating standard deviation
#    std_dev = clus_dist.std()
#    avg_dist_big = avg_dist + 2*std_dev
#    if verb == True:
#        print('Average distance calculated...')
                    
#    return avg_dist, avg_dist_big
    return clus_dist
