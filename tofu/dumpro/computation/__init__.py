#A master module containing all the subroutines required for computation
#Most subroutines require Opencv 3 or greater version


#grascale conversion subroutine
from . import colorgray
#background removal subroutine
from . import background_removal
#video to image conversion subroutine
from . import video_to_img 
#Image to numpy array conversion routine
from . import video_to_array
#subroutine to play a video
from . import playvideo
#edge detection subroutine
from . import edge_detection

from . import binarythreshold
