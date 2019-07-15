""" DUMPRO is a pyhton library for performing dust movie processing on tokamak
 shots and then gather statistics on dust distribution. It is a subpackage of
 Tofu.

DUMPRO consists of 3 subpackages.
-----------------------------
video_comp  : Performs dust tracking computations on videos
image_comp  : Performs dust tracking computation on Images
plotting    : Plots trajectories and statistics on dust distribuitions

Created on February 25, 2019

@author: Arpan Khandelwal
@email: napraarpan@gmail.com
"""

#importing core of dumpro
from ._core import *
