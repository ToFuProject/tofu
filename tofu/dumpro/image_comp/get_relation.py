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
    distance = infocluster.get('distances')
    #empty dictionary for all processed clusters
    traj={}
    #total number of frames present
    nt = total.shape[0]
    #looping over each frame
    for tt in range(0,nt):
        #if cluster not present in frame then go to next one
        if indt[tt] == False:
            print('no cluster \n')
            continue
        #if clusters present in current frame and previous one
        if indt[tt] != False and indt[tt-1] != False:
            #total clusters in current frame
            curr = total[tt]
            print('total clusters in current frame:',curr)
            #total clusters in previous frame
            prev = total[tt-1]
            print('total cluster in previous frame',prev)
            print('cluster in both current and previous frame ...')
            #looping over clusters in current frame
            for ii in range(0,curr):
                #assigning cluster object to a variable
                clus_ob1 = c_id[tt][ii]
                print('cluster in current frame :', clus_ob1.get_id)
                #getting the diatnce array related to the cluster in current frame
                d_array = distance[tt-1][:,ii]
                #getting max probability of the cluster with cluster of the previous frame
                print('clusters being considered are',c_id[tt-1])
                max_prob, indc = get_prob(clus_ob1, c_id[tt-1], d_array, c_id)
                print(max_prob, indc)
                clus_ob2 = c_id[tt-1][indc]
                print('cluster in current frame :', clus_ob2.get_id)
                #getting id of the two clusters
                id1 = clus_ob1.get_id
                id2 = clus_ob2.get_id
                #to check if they are present in the trajectory dictioanary
                listofkeys = list(traj.keys())
                print('before if', clus_ob1.get_id)
                if id1 in listofkeys:
                    #if object is present replace the default object with the 
                    #object that has more information
                    clus_ob1 = traj.get('id1')
                print('after if',clus_ob1.get_id)
                print('before_if', clus_ob2.get_id)
                if id2 in listofkeys:
                    #if object is present replace the default object with the 
                    #object that has more information
                    clus_ob2 = traj.get('id2')
                print('after_if', clus_ob2.get_id)
                #set child values
                clus_ob2.set_child(clus_ob1.get_id)
                #setting up parent for object
                clus_ob1.set_parent(clus_ob2.get_id)
                #assigning values to the trajectory dictionary using id as key
                traj[id1] = clus_ob1
                traj[id2] = clus_ob2

    return traj                
                    

def get_prob(cluster1, prev_frame, d_array, c_id):
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
    print('calculating probability\n')
    #total number of cluster in the previous frame
    nt = len(prev_frame)
    prob = np.zeros((nt,), dtype = float)
    #looping over clusters in the previous frame
    for tt in range(0, nt):
        #assigning cluster object from previous frame to a variable
        cluster2 = prev_frame[tt]
        #gettign details of cluster from current frame
        ang1 = cluster1.angle
        cen1 = cluster1.center
        id1 = cluster1.get_id
        area1 = cluster1.area
        #getting details of cluster from previous frame
        ang2 = cluster2.angle
        cen2 = cluster2.center
        id2 = cluster2.get_id
        area2 = cluster2.area
        #getting distance related to the two clusters
        d = d_array[tt]
        #1st term of probability function
        P1 = 5/d
        print('p1',P1)
        P2, P3 = get_history(cluster2, c_id, cluster1)
        print('p2,p3',P2,P3)
        #calculating the raw difference in area between the two clusters
        diff_area = abs(area2  - area1)
        P4 = 1/diff_area
        print('p4',P4)
        P = P1*P2*P3*P4
        prob[tt] = P
    max_prob = prob.max()
    indc = np.argmax(prob)
    return max_prob,indc
        
def get_history(cluster, c_id, cluster_1):
    """To get history of parent and calculate probability based on it
    
    Parameters:
    --------------------------
    cluster              object
     A object of type cluster. This is the parent and its history will be 
     evaluated
    c_id                 list
     A list of all the cluster objects
    cen1                 array
     The position of the current cluster
    """
    print('calculating history')
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
    list_area = [c_id[list_id[ii][0]][list_id[ii][1]].area for ii in range(0,len(list_id))]
    #to approximate the direction of the parent
    if len(center) == 1:
        P2 = 1
        diff = abs(cluster_1.area - list_area[0])
        if diff == 0:
            P3 = 1
        else:
            P3 = 1/diff
    else:
        #getting the trajectory in right order
        center = np.array(center)[::-1,:]
        #flipping the list and converting it to an array
        list_area = np.array(list_area)[::-1]
        #get the total time step
        nt = center.shape[0]
        #convert the time step into an iterable
        t_step = np.arange(0,nt)
        #define the degree of the polynomial that will be used to fit the data
        deg = 1 if nt == 2 else 2
        #if total time step greater than 3 then use 3 as the limit
        if nt >= 3:
            #use only the last 3 points
            center = center[-3:,:]
            list_area = list_area[-3:]
            #set total time step = 3 for calculation
            nt = 3
        #calculate the area polynomial
        area_fit = np.polyfit(t_step, list_area, deg)
        #calculate the position polynomial
        xy_fit = np.polyfit(t_step, center, deg)
        #calculate to get probable position
        point = np.polyval(xy_fit, [nt])
        #calculate to get probable shape
        area = np.polyval(area_fit,[nt])
        #getting the distance between probable position of cluster and actual
        #position of cluster
        x1 = int(round(point[0]))
        y1 = int(round(point[1]))
        x2 = cluster_1.center[0]
        y2 = cluster_1.center[1]
        distance = (((x1 - x2)**2)+((y1 - y2))**2)**0.5
        #computing probability
        P2 = 1/distance
        #calculating the area difference
        diff_area = abs(cluster_1.area - area)
        if diff_area == 0:
            P3 = 1
        else:
            P3 = 1/diff_area

    return P2, P3
        #reversing center list to get the actual direction
    #we will get the distances
#    for ii in range(0, len(list_id)-1):
#        dist = distance[list_id[ii][0],list_id[ii][1],list_id[ii+1][1]]
#    dist = np.asarray(dist)
#    mean_dist = dist.mean()
                