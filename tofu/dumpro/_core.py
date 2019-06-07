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
import plotting as _plot


##########################################################
# Working directory
##########################################################


                  

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
    __path = A path to a working directory for storage during processing
    __frame_width = width of the video
    __frame_height = height of the video
    __video_time = time length of the video
    __N_frames =  total number of frames in the video
    __fps = number of frames per second 
    meta_data = dictionary containing total frames, fps and frame size of video
    reshape = dictionary containing the croping and time slicing of the video

    Methods:
    --------------------------------------------
    
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
        self.__path = user_path
        #getting the meta data of the video
        self.__frame_width = int(self.cap.get(3))
        self.__frame_height = int(self.cap.get(4))
        self.__N_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.__fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        #calculating the duration of the video        
        self.__duration = self.__N_frames/self.__fps
        self.meta_data = {'fps' : self.__fps, 
                          'fourcc' : self.__fourcc, 
                          'N_frames' : self.__N_frames,
                          'frame_width' : self.__frame_width,
                          'frame_height' : self.__frame_height}
    
##############################################################################
#   setter for Working Directory
##############################################################################
 
    def w_dir(self):
        message = 'Please provide a working directory where files can be stored '
        message += 'while image processing is being done...\n'
        message += 'Note:- Please create a separate directory for each video :'
        #getting user input
        self.__path = input(message)
    

#############################################################################
#     Getters for the class attributes
#############################################################################
        
    @property
    def filename(self):
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
    def path(self):
        """Returns the working directory"""
        return self.__path
    
#############################################################################
#   Grayscale conversion method
#############################################################################
        
    #defining a method for grayscale conversion
    def grayscale(self, meta_data = None, path = None,
                  output_name = None, output_type = None, verb = True):

        #gray will contain the video path and meta_data will contain the 
        #size of frames, total number of frames and the fps of the video
        if meta_data == None:
            meta_data = self.meta_data
        gray, meta_data = _comp.colorgray.convertgray(self.__filename,
                                                      meta_data,
                                                      path,
                                                      output_name,
                                                      output_type,
                                                      Verb)
        #returning the grayscale converted video as a new instance 
        return self.__class__(gray)
    
#############################################################################
#   background removal method
#############################################################################

    def removebackground(self, meta_data = None, 
                         path = None,output_name = None, output_type = None, verb = True):
               
        #applying the background removal operation
        if meta_data == None:
            meta_data = self.meta_data
        foreground, meta_data = _comp.background_removal.remove_background(self.__filename, 
                                                                           meta_data, 
                                                                           path, 
                                                                           output_name, 
                                                                           output_type,
                                                                           verb)
        return self.__class__(foreground)
    
#############################################################################
#   binary conversion method
#############################################################################
    
    def applybinary(self, meta_data = None, path = None,
                    output_name = None,output_type = None, verb = True):
        
        #applying the method of binary conversion
        if meta_data == None:
            meta_data = self.meta_data
        out = _comp.binarythreshold.binary_threshold(self.__filename,
                                                     meta_data,
                                                     path,
                                                     output_name,
                                                     output_type,
                                                     verb)
        #returning the binary converted video as a new instance
        return self.__class__(out[0])
    
#############################################################################
#   edge detection method
#############################################################################

    def detectedge(self, meta_data = None, path = None, 
                   output_name = None, output_type = None, verb = True):
        
        #applying the edge detection method
        if meta_data == None:
            meta_data = self.meta_data
        edge, meta_data = _comp.edge_detection.detect_edge(self.__filename,
                                                           meta_data,
                                                           path,
                                                           output_name,
                                                           output_type,
                                                           verb)
        #returns the edge detected video as a new instance
        return self.__class__(edge)
    
#############################################################################
#   video to image conversion method
#############################################################################
    
    def convert2image(self, meta_date = None, path = None , image_name = None, image_type = None, verb = True):
        
        #applying the video to image conversion method
        if meta_data == None:
            meta_data = self.meta_data
        directory, meta_data = _comp.video_to_img.video2img(self.__filename,
                                                 meta_data,
                                                 path,
                                                 image_name,
                                                 image_type,
                                                 verb)
        #returning the directory in which the video is stored
        return directory, meta_data
    
#############################################################################
#   video to numpy arraay conversion method
#############################################################################

    def convert2pixel(self, meta_data = None, verb = True):
        
        #applying the video to array conversion method
        if meta_data == None:
            meta_data = self.meta_data
        pixel, meta_data = _comp.video_to_array.video_to_pixel(self.__filename,
                                                               meta_data,
                                                               verb)
        
        return pixel, meta_data

#############################################################################
#   displaying a video
#############################################################################
    def playvideo(self):
        """Subroutine for playing a video"""
        #calling play video function from plotting library
        _plot.playvideo.play_video(self.__filename)


    
    def dumpro(self, meta_data = None, path = None, output_name = None, output_type = None, verb = True):
        
        return None
        
        
#sig = inspect.signature(_comp.colorgray.convertgray)
#lp = [p for p in sig.parameters.values() if p.name != 'video_file']
#Video.grayscale.__signature__ = sig.replace(parameters = lp)
        
#Applying the docstring of functions to class methods
Video.grayscale.__doc__ = _comp.colorgray.convertgray.__doc__
Video.removebackground.__doc__ = _comp.background_removal.remove_background.__doc__
Video.applybinary.__doc__ = _comp.binarythreshold.binary_threshold.__doc__
Video.detectedge.__doc__ = _comp.edge_detection.detect_edge.__doc__
Video.convert2pixel.__doc__ = _comp.video_to_array.video_to_pixel.__doc__
Video.convert2image.__doc__ = _comp.video_to_img.video2img.__doc__
