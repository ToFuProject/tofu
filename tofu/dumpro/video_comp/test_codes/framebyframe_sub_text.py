# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:30:42 2019

@author: napra
"""
from video_to_array import video_to_pixel
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
def removebackground(pixel, meta_data = None):
    
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('final.avi',fourcc, fps, (height, width),False)
    
   # new_pixel = np.diff(pixel, 1, 1)
    
    for i in range(len(pixel)):
        frame = pixel[i]
        plt.figure()
        j = plt.imshow(frame)
    
    return pixel, meta_data


#video = video_to_pixel('E:/NERD/Python/Foreground.avi')
file = removebackground(out)