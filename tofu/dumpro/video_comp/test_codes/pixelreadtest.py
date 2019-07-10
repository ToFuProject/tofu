# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:00:28 2019

@author: napra
"""

import cv2
import numpy as np

img = cv2.imread('E:/NERD/Python/data/29.jpg',0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows,columns = img.shape
array = np.ndarray((rows, columns),dtype = int)

k = []
for i in range(rows):
    for j in range(cols):
        array[i][j] = img[i][j]
        
print(array)