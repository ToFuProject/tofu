# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:25:38 2019

@author: napra
"""
import os
import numpy as np 
import cv2

def removebackground(pixel, fps, height, width, path = '/', output_name = 'final',output_type = '.avi'):
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('final.avi',fourcc, fps, (height, width),False)
    
    new_pixel = np.diff(pixel, 1, 0)
    len(new_pixel)
    for i in range (len(new_pixel)):
        frame = new_pixel[i]
        out.write(frame)
    
    meta_data = {'fps' : fps, 'rows': height, 'columns': width, file : path + output_name + output_type}
    return new_pixel, meta_data