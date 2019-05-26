# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:58:32 2019

@author: napra
"""
import numpy 
import cv2

img = numpy.zeros([5,5,3])

img[:,:,0] = numpy.ones([5,5])*64/255.0
img[:,:,1] = numpy.ones([5,5])*128/255.0
img[:,:,2] = numpy.ones([5,5])*192/255.0

cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img);
cv2.waitKey();