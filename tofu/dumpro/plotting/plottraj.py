# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:51:26 2019

@author: napra
"""
import numpy as np

import matplotlib.pyplot as plt


def plot_traj(traj_obs):
    """
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    for ii in range(0, len(traj_obs)):
        obj = traj_obs.get(ii)
        ax.plot(obj.points[0], obj.points[1])
    plt.show()
    
    return None