# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:51:26 2019

@author: Arpan Khandelwal
@email: napraarpan@gmail.com
"""
#standard
import numpy as np
import matplotlib.pyplot as plt

def plot_traj(traj_obs, reshape, w_dir, shot_name):
    """This subroutine plots all the trajectories
    
    Parameters
    ----------------------
    traj_obs           dictionary
     A dictionary containing all the trajectory objects
    
    Returns
    ----------------------
    None
    """
    #getting frame dimensions from reshape dictionary
    height = reshape.get('height')
    width = reshape.get('width') 
    print('Plotting trajectories ... \n')
    #creating new figure
    fig = plt.figure()
    #addign axes
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #axes lables
    ax.set_xlabel('Frame width')
    ax.set_ylabel('Frame height')
    #setting x and y limits
    ax.set_ylim(height)
    ax.set_xlim(width)
    #plotting all trajectories
    for ii in range(0, len(traj_obs)):
        obj = traj_obs.get(ii)
        ax.plot(obj.points[:,0], obj.points[:,1],
                c = 'r', ls = '-', lw =1, marker = '|')
    ax.invert_yaxis()
    plt.savefig(w_dir + shot_name + 'Trajectory.png')
    #displaying plot
    plt.show()
    
    return None