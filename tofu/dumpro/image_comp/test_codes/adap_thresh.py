# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:11:19 2019

@author: napra
"""

import numpy as np
import cv2

def get_avg(frame):
    
    img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
    mean = np.mean(img)
    return mean
