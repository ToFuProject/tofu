# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:02:51 2019

@author: napra
"""

import numpy as np
import cv2
 
im = cv2.imread('E:/NERD/Python/data/frame928.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)