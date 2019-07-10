# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:49:36 2019

@author: napra
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('E:/NERD/Python/data/928.jpg',0)
ret,thresh1 = cv.threshold(img,100,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

cv.imshow('binary',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#for i in range(6):
#    plt.plot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#
#plt.show()
