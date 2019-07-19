# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:03:32 2019

@author: napra
"""

import numpy as np

def info_cluster(infocluster):
    """
    """
    #get ll values from infocluster dictionary
    center = infocluster.get('center')
    area = infocluster.get('area')
    t_cluster = infocluster.get('total')
    angles = infocluster.get('angle')
    indt = infocluster.get('indt')
    avg_small = infocluster.get('avg_area_small')
    avg_big = infocluster.get('avg_area_big')
    clust_dist = infocluster.get('distances')
    #dictionary for storing trajectory information
    traj = {}
    
    #Total number of frames in the cluster
    duration = len(t_cluster)
    #looping over each frame
    for frame in range(0, duration):
        #if there are clusters in the current frame
        if indt[frame] != False:
            #if there are clusters in the previous frame and it is not frame 0
            if indt[frame-1] != False and frame != 0:
                #get total number of clusters in previous frame
                tot_prev = t_cluster[frame-1]
                tot_curr = t_cluster[frame]
                #looping over each cluster
                for c1 in range(0, tot_prev):
                    cen = center[frame-1][c1]
                    for c2 in range(0, tot_curr):
                        cen = center[frame][c2]
                        d = clust_dist[frame-1][c1][c2]
                        ifsteq,
   cen
                        