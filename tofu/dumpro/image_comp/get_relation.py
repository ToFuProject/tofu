# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 00:07:08 2019

@author: napra
"""

#nuilt in
import os
from sys import stdout
from time import sleep

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def get_relation(c_id, infocluster):
    """This subroutine assigns parent child relation to all the clusters
    
    Parameters
    --------------------
    c_id:             list
     A list of cluster objects
    infocluster:      dicitonary
     A dictionary contaning the information of all the clusters
     
    Return
    --------------------
    traj              dictionary
     A dictionary containing cluster objects with relation assigned to them
    """
    
    total = infocluster.get('total')
    indt = infocluster.get('indt')
    area = infocluster.get('area')
    angle = infocluster.get('angle')
    center = infocluster.get('center')
    distance = infocluster.get('distances')
    #empty dictionary for all processed clusters
    traj={}
    #total number of frames present
    nt = total.shape[0]
    #looping over each frame
    for tt in range(0,nt):
        #if cluster not present in frame then go to next one
        if indt[tt] == False:
            continue
        #if clusters present in current frame and previous one
        if indt[tt] == True and indt[tt-1] == True:
            #total clusters in current frame
            curr = total[tt]
            #total clusters in previous frame
            prev = total[tt-1]
            #looping over clusters in current frame
            for ii in range(0,curr):
                #assigning cluster object to a variable
                clus_ob1 = c_id[tt][ii]
                #getting the diatnce array rekated to the cluster in current frame
                d_array = distance[tt][:,ii+1]
                #getting max probability of the cluster with cluster of the previous frame
                max_prob = get_prob(clus_ob1, c_id[tt-1], d_array, c_id, distance)
                
                    

def get_prob(cluster1, prev_frame, d_array, c_id, distance):
    """It calculates the probabilty of the cluster in the current frame
    with all the cluster in the next frame
    
    Parameters
    ----------------------
    cluster1:         Cluster
     An object of type cluster. It represents a cluster from the current frame
    prev_frame        list
     A list of all the cluster objects in the previous frame
    d_array:          array
     An array of the distance relation of the cluster in the current frame with
     all the clusters in the previous frame
    
    """
    #total number of cluster in the previous frame
    nt = len(prev_frame)
    #looping over clusters in the previous frame
    for tt in range(0, nt):
        #assigning cluster object from previous frame to a variable
        cluster2 = prev_frame[tt]
        #gettign details of cluster from current frame
        ang1 = cluster1.angle
        cen1 = cluster1.center
        id1 = cluster1.get_id
        #getting details of cluster from previous frame
        ang2 = cluster2.angle
        cen2 = cluster2.center
        id2 = cluster2.get_id
        #getting distance related to the two clusters
        d = d_array[tt]
        #1st term of probability function
        P1 = 5/d
        P2, P3 = get_history(cluster2, c_id, distance)
        
        
def get_history(cluster, c_id, distance):
    """To get history of parent and calculate probability based on it
    """
    #to keep track of trajectory
    center = []
    #to keep a track of all the id so as to get the distances later on
    list_id = []
    #getting id of first cluster
    clus_id = cluster.get_id
    #adding the id to list id
    list_id.append(clus_id)
    #add the center of first cluster 
    center.append(cluster.center)
    #get parent of the cluster
    parent_id = cluster.parent
    #loop through to access parents and add their centers
    while parent_id != 0:
        #using parent id access parent object
        obj = c_id[parent_id[0]][parent_id[1]]
        #add center of parent object to the list
        center.append(obj.center)
        #add id to list id
        list_id.append(obj.get_id)
        #get next parent
        parent_id = obj.parent
    #Now that we have the list of all parents and all ids
    #to approximate the direction of the parent
    if len(center) == 1:
        P2 = 0
        P3 = 0
    elif len(center) == 2:
        #getting the trajectory in right order
        center.reverse()        
        
        #reversing center list to get the actual direction
    #we will get the distances
    for ii in range(0, len(list_id)-1):
        dist = distance[list_id[ii][0],list_id[ii][1],list_id[ii+1][1]]
    dist = np.asarray(dist)
    mean_dist = dist.mean()
                