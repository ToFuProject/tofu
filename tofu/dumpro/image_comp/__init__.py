""" Image_comp is a sub-package for all computations related to DUMPRO
======================================================================

Provides:
1. Performs preprocessing on video_file or collection of images
2. Identifies dust events and converts them into object
3. Computes relation between dust events of two frames to estimate trajectories
4. Computes statistics related to dust events

Subroutines:

1. vid2img: Converts video to image
2. reshape_image: Performs reshaping to remove unnecessary information
3. conv_gray: Converts image to grayscale
4. denoise: Denoising for grascale images
5. denoise_col: Denoising for color images
6. rm_background: Background Removal
7. to_binary: Converts images to binary
8. guassian_blur: Image Smoothing 
9. cluster_det: Cluster detection 
10. get_distance: Distance calculation for clusters in adjascent frames
11. average_area: Calculates average area of clusters
12. get_id: converts each cluster to object
13. get_relation: computes trajectories
14. verify_traj: verifies trajectories
15. trace_traj: converts each trajectory to object
16. dumpro_img: performs dumpro on images
17. dumpro_vid: performs dumpro on videos

=======================================================================

Created on Mon March 04 2019

@author: Arpan Khandelwal
@author_email: napraarpan@gmail.com

"""
#######################################################
##    Pre-processing                                 ##
#######################################################

#video to image conversion subroutine
from . import vid2img
#reshaping images subroutine
from . import reshape_image
#grayscale conversion subroutine
from . import conv_gray
#grayscale images denoising subroutine
from . import denoise
#colored images denoising subroutine
from . import denoise_col
#background removal subroutine
from . import rm_background
#conversion to binary image subroutine
from . import to_binary
#guassian blur subroutine
from . import guassian_blur

#######################################################
##    Cluster detection                              ##
#######################################################

#cluster detection subroutine
from . import cluster_det
#distance calculation subroutine
from . import get_distance
#average area calculation subroutine
from . import average_area

#######################################################
##    Computing Trajectories                         ##
#######################################################

#assign id to each cluster
from . import get_id
#calculating trajectories
from . import get_relation
#verifying trajectories
from . import verify_traj
#tracing trajectories
from . import trace_traj

#######################################################
##    Performing Dumpro                              ##
#######################################################

#dumpro images subroutine
from . import dumpro_img
#dumpro video subroutine
from . import dumpro_vid