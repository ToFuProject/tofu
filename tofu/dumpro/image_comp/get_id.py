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

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

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
        if indt[tt] == False:
            continue
        nt = total[tt]
        frame_clus = [None for _ in range(0,nt)]
        for ii in range(0, nt):
            frame_clus[ii] = Cluster(ii,tt,center[tt][ii], angle[tt][ii], area[tt][ii])
        clus_list[tt] = frame_clus
    
    return clus_list
