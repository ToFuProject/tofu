# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:37:30 2019

@author: napra
"""

#standard
import numpy as np

def trace_traj(traj):
    """ This subroutine converts every trajectory into a list of cluster
    objects
       
    
    """
    #get the list of keys from the trajectory dictionary
    listofkeys = list(traj.keys())
    #empty list of all the end points
    end_pt = []
    #total number of points in the traj dictionary
    n_points = len(listofkeys)
    print('total number of points in the traj dictionary ', n_points,'\n')
    traj_obs = {}
    #looping over all the points to detect end points
    for ii in range(0, n_points):
        #get the object based on the key
        obj = traj.get(listofkeys[ii])
        #check if child is zero
        if obj.child == 0:
            #if yes then it is a end point
            end_pt.append(listofkeys[ii])
    
    #total  number of trajectories detected is equal to the number of endpoints
    n_traj = len(end_pt)
    print('end points detected :',n_traj,'\n')
    #looping over the start points and tracing the trajectory
    for ii in range(0, n_traj):
        #empty trace list for each trajectory
        trace = []
        #get the first object
        obj = traj.get(end_pt[ii])
        parent = obj.parent
        #looping over until parent = 0
        while parent != 0:
            #get object if parent not zero
            obj = traj.get(parent)
            #add object to trajectory
            trace.append(obj)
            #get parent
            parent = obj.parent
        trace.reverse()
        if len(trace) <2:
            continue
        traj_obs[ii] = trace

    return traj_obs