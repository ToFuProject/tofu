# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:09:46 2019

@author: napra
"""
import numpy as np

def info_cluster(t_clus, area_clus, cen_clus, ang_clus, avg_area, avg_dist):
    
    cluster = []
    frame = 2
    #checking each frame
    for i in range(len(t_clus)):
        #if cluster is present in frame
        if t_clus[i] != 0:
            #checking each cluster in frame i
            for j in range(len(cen_clus[i])):
                max_prob = 0
                #getting co_ordinates of center
                center1 = cen_clus[i][j]
                #if next frames has clusters
                if cen_clus[i+1] != []:
                    #iterating over clusters in next frame
                    for k in range(len(cen_clus[i+1])):
                        #getting center
                        center2 = cen_clus[i+1][k]
                        #probability based on distance
                        dist = get_dist(center1, center2)
                        if dist < avg_dist:
                            prob = (avg_dist - dist)/avg_dist
                        else:
                            prob = (dist - avg_dist)/avg_dist
                        
                        
                        ang1 = abs(ang_clus[i][j])
                        ang2 = abs(ang_clus[i+1][k])
                        
                        ang3 = 180 - ang1
                        
                        if ((abs(ang3 - ang2))<30) or ((abs(ang1 - ang2))<30):
                            prob *= 0.5
                        else:
                            prob *= 0.2
                        
                        area1 = area_clus[i][j]
                        area2 = area_clus[i+1][k]

                        area = abs(area1 - area2)
                        
                        if area < avg_area:
                            prob *= 0.5
                        else:
                            prob *= 0.2
                        
                        if prob > max_prob:
                            max_prob = prob
                        
                        
                        
                        
                        
        
    return None

def get_dist(cen1, cen2):
    x2 = cen2[0]
    x1 = cen1[0]
    y2 = cen2[1]
    y1 = cen2[1]
    
    x = (x2 - x1)**2
    y = (y2 - y1)**2
    
    dist = (x + y)**0.5
    
    return dist