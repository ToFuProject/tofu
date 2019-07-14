# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:03:32 2019

@author: napra
"""

import numpy as np

def info_cluster(infocluster):
    center = infocluster.get('center')
    area = infocluster.get('area')
    t_cluster = infocluster.get('total')
    angles = infocluster.get('angle')
    indt = infocluster.get('indt')
    avg_small = infocluster.get('avg_area_small')
    avg_big = infocluster.get('avg_area_big')
    clust_dist = infocluster.get('distances')
    
    frame = 3
    
    duration = len(t_cluster)
    for frame in range(0, duration):
        if indt[frame] == False:
            if frame <3:
                for tt in range(frame-3, frame+3):
                    if indt[tt] == False:
                        continue
                    else:
                        
        
        
        
        
        
        