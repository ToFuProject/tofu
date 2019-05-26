# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:43:55 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

#built-ins
import os
import glob

#standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def image2video(image_path,video_path):
    """
    """
    
    img_array=[]
    for count in (len(os.listdir(image_path))):
        filename = './image_path/mask_frame_'+str(count)+'jpg'
        print(filename)
        img = cv2.imread(filename)
        print('reading image' )
        height,width,layer = img.shape
        print(img.shape)
        size = height,width
        img_array.append(img)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')    
    out = cv2.VideoWriter(video_path,fourcc,20,size)
    
    for i in range(len(img_array)):
        print('writting frame ',i)
        out.write(img_array[i])       
    out.release
    
