# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:16:01 2019

@author: napra
"""

import numpy as np
import os
import cv2

def removebackground():
#    image1 = cv2.imread("/data/frame28")
#    image2 = cv2.imread("/data/frame29")
#    image3 = image1 - image2
    image1= np.int32('/data/frame28')

    image2= np.int32('/data/frame29')

    image3 = image1 - image2
    cv2.imwrite('subframe',image3)

    return'done'