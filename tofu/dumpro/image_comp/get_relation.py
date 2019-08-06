# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 00:07:08 2019

@author: napra
"""

#nuilt in
import os
from sys import stdout
from time import sleep

#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def get_relation(c_id, infocluster):
    """
    """
    total = infocluster.get('total')
    indt = infocluster.get('indt')
    