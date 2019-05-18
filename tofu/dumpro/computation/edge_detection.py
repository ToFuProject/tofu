# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:21:27 2019

@author: napra
"""
import os
import numpy as np
import cv2

def edge_detection(image_file):
    img = cv2.imread(image_file)
    type(img)
    print(img)
    edge = cv2.Canny(img,200,255)
    
    cv2.imshow("edge detected image",edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return None