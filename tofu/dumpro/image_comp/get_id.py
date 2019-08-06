# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:42:55 2019

@author: napra
"""
#nuilt in
import os
from sys import stdout
from time import sleep

#standard
import numpy as np

def get_id(infocluster, Cluster):
    """This subroutine takes each cluster and converts it into an object.
    It assigns a unique id to ecah cluster present on each frame and returns 
    a list containing all the cluster objects
    
    Parameters
    ----------------------
    infocluster:        dictionary
     A dictionary containing all the information about each cluster
    Cluster:            object
     The main cluster class
     
    Return
    ----------------------
    clus_list:          list
     A list of all the cluster objects
    """
    #getting different parameters from infocluster dictionary
    area = infocluster.get('area')
    center = infocluster.get('center')
    angle = infocluster.get('angle')
    indt = infocluster.get('indt')
    total = infocluster.get('total')
    #total length of film
    duration = indt.shape[0]
    
    #creating a map for the entire length of film
    clus_list = [None for _ in range(0,duration)]
    #looping over all thg frames
    for tt in range(0, duration):
        #if no cluster present go to next frame
        if indt[tt] == False:
            continue
        #total number of clusters in the frame
        nt = total[tt]
        #empty list with size equal to the no. of cluster present in the frame
        frame_clus = [None for _ in range(0,nt)]
        #looping over each cluster to create object
        for ii in range(0, nt):
            #populating the list with objects
            frame_clus[ii] = Cluster(ii,tt,center[tt][ii], angle[tt][ii], area[tt][ii])
        #populating list with lists that represent each frame
        clus_list[tt] = frame_clus
    
    return clus_list
