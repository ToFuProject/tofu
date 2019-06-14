# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:48:58 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""

# class definitions (object-oriented codes)
#class video is an object containing information about the videofile:
#resolution, size, no of frames

# Built-ins
import os
import warnings
import inspect

# Standard
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import datetime


#more special packages
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
# tofu-specific
#import tf.__version__ as __version__

# dumpro-specific
import computation as _comp
import image_comp as _i_comp
import plotting as _plot
                  

##########################################################
##########################################################
#               Video Class
##########################################################

class Video(object):
    """ A Video class that will pe preocessed using the various methods
    defined below.

    Input:
    --------------------------------------------
    filename = video file along with the path
    
    Attributes:
    --------------------------------------------
    __filename = Path of the file along with its name and extension
    __w_dir = A path to a working directory for storage during processing
    __frame_width = width of the video
    __frame_height = height of the video
    __video_time = time length of the video
    __N_frames =  total number of frames in the video
    __fps = number of frames per second 
    meta_data = dictionary containing total frames, fps and frame size of video
    reshape = dictionary containing the croping and time slicing of the video

    Setters:
    --------------------------------------------
    set_w_dir
    
    Getters:
    --------------------------------------------
    filename
    resolution
    N_frames
    fps
    fourcc
    w_dir
    meta_data
    
    Methods:
    --------------------------------------------
    grayscale
    
    """

    def __init__(self, filename, verb=True):

        #checking if the file provided exists or not
        if not os.path.isfile(filename): 
            msg = "The provided path does not exist:\n"
            msg += "\t- path: %s"%filename
            msg += "\t=> Please provide the correct path and try again"
            raise Exception(msg)
        #reading the video file
        self.cap = cv2.VideoCapture(filename)
        if verb:
            msg = "The following file was successfully loaded:\n"
            msg += "    %s"%filename
            msg += "    {}".format(filename)
            print(msg)
        self.__filename = filename
        self.__w_dir = ''
        #getting the meta data of the video
        self.__frame_width = int(self.cap.get(3))
        self.__frame_height = int(self.cap.get(4))
        self.__N_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.__fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        #calculating the duration of the video        
        self.__duration = self.__N_frames/self.__fps
        self.__meta_data = {'fps' : self.__fps, 
                          'fourcc' : self.__fourcc, 
                          'N_frames' : self.__N_frames,
                          'frame_width' : self.__frame_width,
                          'frame_height' : self.__frame_height}
    
##############################################################################
#   setter for Working Directory
##############################################################################
 
    def set_w_dir(self):
        """getter function for Working Directory"""
        message = 'Please provide a working directory where files can be stored '
        message += 'while image processing is being done...\n'
        message += 'Note:- Please create a separate directory for each video :'
        #getting user input
        self.__w_dir = input(message)
    

#############################################################################
#     Getters for the class attributes
#############################################################################
        
    @property
    def filename(self):
        """Returns the path containing the video"""
        return self.__filename
    
    
    #defining a getter for the video resolution
    def resolution(self):
        """Returns the size of frame of the video"""
        #getting width and height
        width = self.__frame_width
        height = self.__frame_height
        return (width,height)
    
    #defining a getter for the total number of frames in the video
    @property
    def N_frames(self):
        """ Returns the total number of frames in the video"""
        return self.__N_frames
    
    #defining a getter for the frames per second
    @property
    def fps(self):
        """Returns the playspeed of the video in frames per second"""
        return self.__fps
    
    #defining a gettter for the four character code of the video
    @property
    def fourcc(self):
        """ Returns the four character code of the video"""
        return self.__fourcc
    
    @property
    def w_dir(self):
        """Returns the working directory"""
        return self.__w_dir
    
    @property
    def meta_data(self):
        """Returns the metadata of the videofile"""
        return self.__meta_data
    
#############################################################################
#   Grayscale conversion method
#############################################################################
        
    #defining a method for grayscale conversion
    def grayscale(self, output_name = None, output_type = None, verb = True):

        #gray will contain the video path and meta_data will contain the 
        #size of frames, total number of frames and the fps of the video
        gray, meta_data = _comp.colorgray.convertgray(self.__filename,
                                                      self.__meta_data,
                                                      self.__w_dir,
                                                      output_name,
                                                      output_type,
                                                      verb)
        #returning the grayscale converted video as a new instance 
        return self.__class__(gray)
    
#############################################################################
#   background removal method
#############################################################################

    def removebackground(self, output_name = None, output_type = None, verb = True):
               
        #applying the background removal operation
        foreground, meta_data = _comp.background_removal.remove_background(self.__filename, 
                                                                           self.__meta_data, 
                                                                           self.__w_dir, 
                                                                           output_name, 
                                                                           output_type,
                                                                           verb)
        return self.__class__(foreground)
    
#############################################################################
#   binary conversion method
#############################################################################
    
    def applybinary(self, output_name = None,output_type = None, verb = True):
        
        #applying the method of binary conversion
        out = _comp.binarythreshold.binary_threshold(self.__filename,
                                                     self.__meta_data,
                                                     self.__w_dir,
                                                     output_name,
                                                     output_type,
                                                     verb)
        #returning the binary converted video as a new instance
        return self.__class__(out[0])
    
#############################################################################
#   edge detection method
#############################################################################

    def detectedge(self, output_name = None, output_type = None, verb = True):
        
        #applying the edge detection method
        edge, meta_data = _comp.edge_detection.detect_edge(self.__filename,
                                                           self.__meta_data,
                                                           self.__w_dir,
                                                           output_name,
                                                           output_type,
                                                           verb)
        #returns the edge detected video as a new instance
        return self.__class__(edge)
    
#############################################################################
#   video to image conversion method
#############################################################################
    
    def convert2image(self, image_name = None, image_type = None, verb = True):
        
        #applying the video to image conversion method
        directory, meta_data = _comp.video_to_img.video2img(self.__filename,
                                                 self.__meta_data,
                                                 self.__w_dir,
                                                 image_name,
                                                 image_type,
                                                 verb)
        #returning the directory in which the video is stored
        return directory, meta_data
    
#############################################################################
#   video to numpy arraay conversion method
#############################################################################

    def convert2pixel(self, verb = True):
        
        #applying the video to array conversion method
        pixel, meta_data = _comp.video_to_array.video_to_pixel(self.__filename,
                                                               self.__meta_data,
                                                               verb)
        
        return pixel, meta_data

#############################################################################
#   displaying a video
#############################################################################
    def playvideo(self):

        #calling play video function from plotting library
        _plot.playvideo.play_video(self.__filename)


    
    def dumpro(self, output_name = None, output_type = None, verb = True):
        
        return None
        
        
#sig = inspect.signature(_comp.colorgray.convertgray)
#lp = [p for p in sig.parameters.values() if p.name != 'video_file']
#Video.grayscale.__signature__ = sig.replace(parameters = lp)
#############################################################################
#   Docstrings for video class methods
#############################################################################
Video.grayscale.__doc__ = _comp.colorgray.convertgray.__doc__
Video.removebackground.__doc__ = _comp.background_removal.remove_background.__doc__
Video.applybinary.__doc__ = _comp.binarythreshold.binary_threshold.__doc__
Video.detectedge.__doc__ = _comp.edge_detection.detect_edge.__doc__
Video.convert2pixel.__doc__ = _comp.video_to_array.video_to_pixel.__doc__
Video.convert2image.__doc__ = _comp.video_to_img.video2img.__doc__
Video.playvideo.__doc__ = _plot.playvideo.play_video.__doc__



###################################################################
###################################################################
#    For a collection of images
###################################################################

class img_dir(object):
    """A class for handeling image processing on a collection of images
    The input to create class is to pass the path containing all the images
    inside it.
    
    Input:
    --------------------------------------------
    filename = path where the images are located
    
    Attributes:
    --------------------------------------------
    __filename = Path where the images are present
    __w_dir = Working directory where images can be stored during computation
    __shot_name = Name of tokomak and the shot number as a single string
    __meta_data = dictionary containing total frames, fps and frame size of video
    __reshape = dictionary containing the croping and time slicing of the video

    Methods:
    --------------------------------------------
    
    """
    
    def __init__(self, filename):
        if not os.path.exists(filename):
            msg = 'The path provided is wrong'
            msg += 'Please provide the correct path'
            raise Exception(msg)

        self.__im_dir = filename
        files = [f for f in os.listdir(filename) if os.path.isfile(os.path.join(filename,f))]
        img = cv2.imread(filename+files[0])
        self.__w_dir = None
        self.__shot_name = None
        self.__meta_data = {'N_frames' : len(files),
                            'frame_height': img.shape[0],
                            'frame_width' : img.shape[1]}
        self.__reshape = {}

####################################################################
#   setters for attributes
####################################################################
    
    def set_w_dir(self, w_dir):
        msg = 'The path provided is not correct'
        msg += 'Please provide the correct path for the working directory'
        if not os.path.exits(w_dir):
            raise Exception(msg)
        self.__w_dir = w_dir
        
    def set_shot_name(self,shot_name):
        self.__shot_name = shot_name
        
    def set_meta_data(self, meta_data):
        self.__meta_data = meta_data
        
####################################################################
#   getters for attribiutes
####################################################################

    @property
    def im_dir(self):
        return self.__im_dir
    
    @property
    def w_dir(self):
        return self.__w_dir
    
    @property
    def shot_name(self):
        return self.__shot_name
    
    @property
    def meta_data(self):
        return self.__meta_data

###################################################################
#   image slicing method
###################################################################    
    
    def crop_im(self, im_out = None, meta_data = None, verb = True):
        
        return None
        
    
###################################################################
#   grayscale conversion method
###################################################################
    
    def to_gray(self, im_out = None, verb = True):
        
        out_path, meta_data = _i_comp.conv_gray.conv_gray(self.__im_dir, 
                                                        self.__w_dir, 
                                                        self.__shot_name, 
                                                        im_out, 
                                                        self.__meta_data, 
                                                        verb)
        return self.__class__(out_path)
    
####################################################################
#   Background removal method
####################################################################
    
    def remove_backgrd(self, im_out = None, verb = True):
            
        out_path, meta_data = _i_comp.rm_background.rm_back(self.__im_dir,
                                                            self.__w_dir,
                                                            self.__shot_name,
                                                            im_out,
                                                            self.__meta_data,
                                                            verb)
        return self.__class__(out_path)
    
####################################################################
#   denoising method for grayscale images
####################################################################
        
    def denoise_gray(self, im_out = None, verb = True):
            
        out_path, meta_data = _i_comp.denoise.denoise(self.__im_dir,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
        return self.__class__(out_path)
    
#####################################################################
#  denoising method for color images
#####################################################################
        
    def denoise_col(self, im_out = None, verb = True):

        out_path, meta_data = _i_comp.denoise_col.denoise_col(self.__im_dir,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
        return self.__class__(out_path)

#####################################################################
#   binary conversion method
#####################################################################
        
    def to_bin(self, im_out = None, verb = True):

        out_path, meta_data = _i_comp.to_binary.bin_thresh(self.__im_dir,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
        return self.__class__(out_path)

#####################################################################
#   playing of the images
#####################################################################

    def play(self):
        
        _plot.playimages.play_img(self.__im_dir)
        
#####################################################################
#   cluster detection
#####################################################################
        
    def det_cluster(self):
            
        return None    
        
#####################################################################
#   dumpro
#####################################################################        
        
    def dumpro(self, im_out = None, verb = True):
#
#       denoise, meta_data = _i_comp.denoise_col.denoise_col(self.__im_dir,
#                                                                 self.__w_dir,
#                                                                 self.__shot_name,
#                                                                 im_out,
#                                                                 meta_data,
#                                                                 verb)
#       

        gray, meta_data = _i_comp.conv_gray.conv_gray(self.__im_dir,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
            
        den_gray, meta_data = _i_comp.denoise.denoise(gray,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
            
        rmback, meta_data = _i_comp.rm_background.rm_back(den_gray,
                                                          self.__w_dir,
                                                          self.__shot_name,
                                                          im_out,
                                                          self.__meta_data,
                                                          verb)
        
        binary, meta_data = _i_comp.to_binary.bin_thresh(rmback,
                                                         self.__w_dir,
                                                         self.__shot_name,
                                                         im_out,
                                                         self.__meta_data,
                                                         verb)
            
            
            
        return None
        

img_dir.to_gray.__doc__ = _i_comp.conv_gray.conv_gray.__doc__
img_dir.denoise_col.__doc__ = _i_comp.denoise_col.denoise_col.__doc__
img_dir.denoise_gray.__doc__ = _i_comp.denoise.denoise.__doc__
img_dir.remove_backgrd.__doc__ = _i_comp.rm_background.rm_back.__doc__
img_dir.to_bin.__doc__ = _i_comp.to_binary.bin_thresh.__doc__
img_dir.play.__doc__ = _plot.playimages.play_img.__doc__

#####################################################################
#####################################################################
#   A class for handling videos and images 
#####################################################################

class vid_img(Video, img_dir):
    """This is a derived class from both video class and img_dir class.
    This is an intermediate approach, between working completely with videos
    and completely with images.
    This class was maily created beacause it follows a more computationally 
    robust way of detecting dust particles.
    
    Input:
    --------------------------------------------
    filename = video file along with the path
    
    Attributes:
    --------------------------------------------
    __filename = Path of the file along with its name and extension
    __w_dir = A path to a working directory for storage during processing
    __frame_width = width of the video
    __frame_height = height of the video
    __video_time = time length of the video
    __N_frames =  total number of frames in the video
    __fps = number of frames per second 
    meta_data = dictionary containing total frames, fps and frame size of video
    reshape = dictionary containing the croping and time slicing of the video

    Setters:
    --------------------------------------------
    set_w_dir_shotname
    set_w_dir
    
    Getters:
    --------------------------------------------
    filename
    resolution
    N_frames
    fps
    fourcc
    w_dir
    meta_data
    
    Methods:
    --------------------------------------------
    
    """
    def __init__(self,filename):
         Video.__init__(self, filename, verb = True)
         self.__shot_name = None
         self.__im_dir = None
         
         
    
    
    def set_im_dir_shotname(self):
        """Setter for image directory and shotname
        This function calls the video to image convertor function in image
        computation package and
        """
        self.__im_dir, mt_data ,self.__shot_name = _i_comp.vid2img.video2img(self.__filename,
                                                                             self.__w_dir, 
                                                                             None, 
                                                                             self.__meta_data)
        
        return self.__im_dir,self.__shot_name
    
    def dumpro(self,tlim, crop, w_dir = None, im_out= None, verb = True):
        """This method performs dust movie processing on the video_file
        It converts the video to image and extracts the shotname and the
        image directory. Then it 
        """
        drive, path_of_file = os.path.splitdrive(self.__filename)
        path, file = os.path.split(path_of_file)
        folder = 'dumpro'
        path = os.path.join(path,folder,'')
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.__w_dir == None and w_dir == None:
            msg = 'Working directory not provided:'
            msg += 'Creating :'
            warnings.warn(msg)
            print(path)
            vid_img.set_w_dir(path)
        
        self.__im_dir, self.__shot_name = vid_img.set_im_dir_shotname()
        gray, meta_data = _i_comp.conv_gray.conv_gray(self.__im_dir,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
            
        den_gray, meta_data = _i_comp.denoise.denoise(gray,
                                                      self.__w_dir,
                                                      self.__shot_name,
                                                      im_out,
                                                      self.__meta_data,
                                                      verb)
            
        rmback, meta_data = _i_comp.rm_background.rm_back(den_gray,
                                                          self.__w_dir,
                                                          self.__shot_name,
                                                          im_out,
                                                          self.__meta_data,
                                                          verb)
        
        binary, meta_data = _i_comp.to_binary.bin_thresh(rmback,
                                                         self.__w_dir,
                                                         self.__shot_name,
                                                         im_out,
                                                         self.__meta_data,
                                                         verb)
        
        
         
    
    
        
  