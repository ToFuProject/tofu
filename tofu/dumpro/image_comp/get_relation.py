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
    """
    """
    total = infocluster.get('total')
    indt = infocluster.get('indt')
    area = infocluster.get('area')
    angle = infocluster.get('angle')
    center = infocluster.get('center')
    
    traj={}
    
    nt = total.shape[0]
    
    for tt in range(0,nt):
        if indt[tt] == False:
            continue
        if indt[tt] == True and indt[tt-1] == True:
            curr = total[tt]
            prev = total[tt-1]
            
            for ii in range(0,curr):
                clus_ob1 = c_id[tt][ii]
                max_prob = get_prob(clus_ob1, c_id[tt-1])
                
                for jj in range(0, prev):
                    clus_ob2 = c_id[]
                    

def get_prob(cluster1, prev_frame):
    
    nt = len(prev_frame)
    for tt in range(0, nt):
        cluster2 = prev_frame[tt]
        ang1 = cluster1.angle
        cen1 = cluster1.center
        id1 = cluster1.get_id
        
        ang2 = cluster2.angle
        cen2 = cluster2.center
        id2 = cluster2.get_id