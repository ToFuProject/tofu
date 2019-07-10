# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:05:56 2019

@author: napra
"""

import os
import numpy as np
import cv2

def display_array(array):
    
    shape = array.shape
    print(shape)
    print(array[0].shape)
    num = 0 
    while (num <= array,shape):
        frame = array[num]
        num+=1
        cv2.imshow('frame', frame)
    
    cv2.destroyAllWindows()
    