"""Plotting is a sub-package for displaying all information processed and 
gathered by DUMPRO.
=============================================================================

Provides:
1. Plots distributions of number and size
2. Plots trajectories of dust events

Subroutines:
1. playvideo: displays a videofile
2. playimages: plays a collection of image files
3. area_distrib: displays information related to the pixel size of dust 
                 clusters
4. f_density_dist: displays number density per frame.
5. plottraj: Plots trajectories of dust events

=============================================================================

Created on Tue June 11 2019

@author: Arpan Khandelwal
@author_email: napraarpan@gmail.com
"""


#Modules related to maily plotting and display of videos
#######################################################
##    Play image or video files                      ##
#######################################################
#play videofile
from . import playvideo
#play image collection
from . import playimages

#######################################################
## Plot distributions                                ##
#######################################################
#plot size distributions
from . import area_distrib
#plot number distributions
from . import f_density_dist

#######################################################
## Plot trajectories                                 ##
#######################################################
#plot trajectory
from . import plottraj

