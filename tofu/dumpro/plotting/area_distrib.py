# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:27:01 2019

@author: napra
"""
#nuilt in
import os

#standard
import numpy as np
import matplotlib.pyplot as plt


def get_distrib(area, indt):
    """This subroutine plots the area distribution
    """
    duration = len(indt)
    f = plt.figure()     
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    for i in range(0, duration):
        if indt[i] != False:
            ax.plot(i*np.ones((area[i].size,)),area[i],'.')
    ax.set_title('dist')
    
    return None