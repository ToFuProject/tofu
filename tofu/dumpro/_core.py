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
import datetime as dt
import time


#more special packages
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
# tofu-specific
#import tf.__version__ as __version__

# dumpro-specific
#video computation
from . import video_comp as _comp
#image computation
from . import image_comp as _i_comp
#ploting file
from . import plotting as _plot


__all__ = ['Img_dir', 'Vid_img', 'Cluster','Trajectory']


###################################################################
###################################################################
#    Class for cluster
###################################################################

class Cluster(object):
    """This class stores information of each cluster and assigns each cluster
    a unique id. This Id is provided when a cluster is assigned as a parent or
    child
    Input:
    --------------------------------------------
    num           integer
     cluster number in the current frame
    frame         integer
     frame in which the cluster is present
    center        tuple
     position of the cluster in the current frame
    angle         float
     the orientation of the cluster
    area          float
     the pixel size of the cluster
    parent        tuple
     id number of the parent of the cluster. If value is zero, the this is the
     start of the trajectory.
    child         tuple
     id number of the child. If value is zeroi then this is the end of the 
     trajectory
    
    Attributes:
    --------------------------------------------
    __id     = id of the cluster
    __frame  = frame number of the cluster
    __center = position of the cluster in the current frame
    __angle  = orientation of the cluster in the current frame
    __area   = pixel area of the cluster
    __parent = id of the parent of the cluster
    __child  = id of the child of the cluster

    Getters:
    --------------------------------------------
    get_id = Returns the id of the cluster
    frame  = frame number of the cluster
    center = position of the cluster in the current frame
    angle  = orientation of the cluster in the current frame
    area   = pixel area of the cluster
    parent = id of the parent of the cluster
    child  = id of the child of the cluster
    
    Methods:
    --------------------------------------------
    """
    
    def __init__(self, num, frame, center, angle, area, parent = 0, child = 0):
        self.__id = frame,num
        self.__frame = frame
        self.__center = center
        self.__angle = angle
        self.__area = area
        self.__parent = parent
        self.__child = child
    
    def set_child(self,child):
        if self.__child == 0:
            self.__child = [child]
        else:
            self.__child.append(child)
    
    def set_parent(self,parent):
        self.__parent = parent
    
    @property
    def get_id(self):
        """Returns the of the cluster"""
        return self.__id
       
    @property
    def frame(self):
        """Returns the frame number of the cluster"""
        return self.__frame
    
    @property
    def center(self):
        """Returns the position of the cluster"""
        return self.__center
    
    @property
    def angle(self):
        """Returns the angle of orientation of the cluster"""
        return self.__angle
    
    @property
    def area(self):
        """Returns the pixel area of the cluster"""
        return self.__area
    
    @property
    def parent(self):
        """Returns the parent id of the cluster"""
        return self.__parent
    
    @property
    def child(self):
        """Returns the child id of the cluster"""
        return self.__child

###################################################################
###################################################################
#    Class for Trajectory
###################################################################
    
    
class Trajectory(object):
    """A class for all trajectories. This creates a trajectory object that 
    provides all the available relevant information for each trajectory.
    
    Input:
    --------------------------------------------
    traj             list
     A list of all the cluster objects that make up the trajectory
     
    Attributes:
    --------------------------------------------
    __np             integer
     Number of points in the trajectory
    __trajob         list
     A list of all the cluster objects that make up the trajectory
    __points         2D Numpy Array
     A 2D array with all the points that make up the trajectory
    __avg_vel        float
     The average velocity of the trajectory in terms of pixels per frame
    __areaevo        array
     An array containing information on the size evolution of the cluster
     in terms number of pixels.
    
    Getters:
    --------------------------------------------
    n_points = Returns the number of points in the trajectory
    points   = Returns all the points in the trajectory
    avg_vel  = Returns the average velocity of the trajectory
    areaevo  = Returns the area evolution of the trajectory
     """
    def __init__(self, traj):
        self.__np = len(traj)
        self.__trajob = traj
        self.__points = np.array([traj[ii].center for ii in range(0, self.__np)])
        dist = 0
        for ii in range(0,self.__np-1):
            x1 = self.__points[ii][0]
            y1 = self.__points[ii][1]
            x2 = self.__points[ii+1][0]
            y2 = self.__points[ii+1][1]
            d = (((x1-x2)**2)+((y1-y2)**2))**0.5
            d = abs(d)
            dist += d
        self.__avg_vel = dist/self.__np
        self.__areaevo = (traj[ii].area for ii in range(0, self.__np))

####################################################################
#   Getters for attributes
####################################################################
        
    @property
    def n_points(self):
        """Returns the number of points in the trajectory"""
        return self.__np
    
    @property
    def points(self):
        """Returns the points in the trajectory"""
        return self.__points
    
    @property
    def avg_vel(self):
        """Returns the average velocity of the cluster"""
        return self.__avg_vel
    
    @property
    def areaevo(self):
        """Returns the area evolution of the trajectory"""
        return self.__areaevo
            
###################################################################
###################################################################
#    For a collection of images
###################################################################

class Img_dir(object):
    """A class for handeling image processing on a collection of images
    The input to create class is to pass the path containing all the images
    inside it.
    
    Input:
    --------------------------------------------
    filename = path where the images are located
    
    Attributes:
    --------------------------------------------
    __filename     = Path where the images are present
    __w_dir        = Working directory where images can be stored during 
                     computation
    __shot_name    = Name of tokomak and the shot number as a single string
    __meta_data    = dictionary containing total frames, fps and frame size.
    __reshape      = dictionary containing the croping and time slicing of the
                     frames
    __infoclusters = dictionary containing cluster information.
    
    Setters:
    --------------------------------------------
    set_shot_name = Setter for shotname
    set_meta_data = Setter for meta_data dictioanry
    set_reshape   = Setter for reshape dictionary
    
    Getters:
    --------------------------------------------
    im_dir      = Getter for image path
    w_dir       = Getter for working directory
    shot_name   = Getter for shotname
    resolution  = Getter for frame resolution
    meta_data   = Getter for meta data dictionary
    rehsape     = Getter for reshape dictiionary
    infocluster = Getter for infocluster dicitonary  
    
    Methods:
    --------------------------------------------
    crop_im        = Reshape image method
    to_gray        = Grayscale conversion method
    remove_backgrd = Bcakground removal method
    denoise_gray   = Grayscale image denoising method
    denoise_col    = Coloured image denoising method
    to_bin         = Binary image conversion method
    play           = Play collaction of images as a Video
    det_cluster    = Cluster detection method
    dumpro         = Dust Movie processing method
    
    """
    
    def __init__(self, filename, w_dir = None):
        if not os.path.exists(filename):
            msg = 'The path provided is wrong'
            msg += 'Please provide the correct path'
            raise Exception(msg)
        
        self.__im_dir = filename
        files = [f for f in os.listdir(filename) if os.path.isfile(os.path.join(filename,f))]
        img = cv2.imread(filename+files[0])
        
        #working directory
        if w_dir == None:
            #default directory value
            msg = 'The working directory has not been provided !'
            msg += 'Falling to default values...\n'
            #warning message
            warnings.warn(msg)
            folder = '_dumpro'
            if os.path.split(filename)[1] == '':
                temp_dir, temp_name = os.path.split(filename[:-1])
            path = os.path.join(temp_dir, temp_name+folder, '')
            if not os.path.exists(path):
                #creating working directory using default values
                os.makedirs(path)
            self.__w_dir = path
            print('The working directory is :\n')
            print(self.__w_dir,'\n')
        else:
            #in case of wrong path
            msg = 'The path provided is not correct!'
            msg += '\n Please provide the correct path for the working directory'
            if not os.path.exists(w_dir):
                raise Exception(msg)
            #if path is correct
            self.__w_dir = w_dir
            print('The working directory is :\n')
            print(self.__w_dir)
        
        #shotname    
        self.__shot_name = temp_name
        
        #meta_data
        self.__meta_data = {'N_frames' : len(files),
                            'frame_height': img.shape[0],
                            'frame_width' : img.shape[1]}
        #reshape dictionary
        self.__reshape = {}
        #information on the total number of clusters in each frame, their area,
        #centers, distances between clusters in two adjascent frames, and index
        self.__infocluster = {}
        self.__c_id = None

####################################################################
#   setters for attributes
####################################################################
    #defining setter for shotname
    def set_shot_name(self,shot_name):
        """Setter for shotname"""
        self.__shot_name = shot_name
        
    #defining setter for meta_data
    def set_meta_data(self, meta_data):
        """Setter for meta_data"""
        self.__meta_data = meta_data
    
    #defining setter for infocluster
    def set_infocluster(self, infoclusters):
        """Setter for infocluster dictionary"""
        self.__infocluster = infoclusters
    
    #defining setter for reshape
    def set_reshape(self, reshape):
        """Setter for reshape dictionary"""
        self.__reshape = reshape
        
    def set_c_id(self, c_id):
        """Setter for list of cluster objects"""
        self.__c_id = c_id
        
####################################################################
#   getters for attribiutes
####################################################################
    #defining getter for im_dir
    @property
    def im_dir(self):
        return self.__im_dir
    
    #defining getter for w_dir
    @property
    def w_dir(self):
        return self.__w_dir
    
    #defining getter for shotname
    @property
    def shot_name(self):
        return self.__shot_name
    
    #define getter for meta_data
    @property
    def meta_data(self):
        return self.__meta_data
    
    #define getter for resolution
    @property
    def resolution(self):
        height = self.__meta_data.get('frame_height')
        width = self.__meta_data.get('frame_width')
        return height,width
    
    #define getter for reshape dictionary
    @property
    def reshape(self):
        return self.__reshape
    
    #define getter for infocluster
    @property
    def infoclusters(self):
        return self.__infocluster
    
    @property
    def c_id(self):
        return self.__c_id

###################################################################
#   image slicing method
###################################################################    
    
    def crop_im(self, tlim, height, width, im_out = None, verb = True):
        #cropping the images for faster computation
        out_path, reshape = _i_comp.reshape_image.reshape_image(self.im_dir,
                                                                self.w_dir,
                                                                self.shot_name,
                                                                tlim, height,
                                                                width, im_out, 
                                                                verb)
        new = self.__class__(out_path)
        new.set_meta_data = self.meta_data
        new.__reshape = reshape
        return new
        
    
###################################################################
#   grayscale conversion method
###################################################################
    
    def to_gray(self, im_out = None, verb = True):
        #grayscale function
        out_path = _i_comp.conv_gray.conv_gray(self.im_dir, 
                                               self.w_dir, 
                                               self.shot_name, im_out, verb)
        return self.__class__(out_path)
    
####################################################################
#   Background removal method
####################################################################
    
    def remove_backgrd(self, rate = None, im_out = None, verb = True):
        #background removal
        out_path = _i_comp.rm_background.rm_back(self.im_dir,
                                                 self.w_dir,
                                                 self.shot_name,
                                                 rate, im_out,
                                                 verb)
        return self.__class__(out_path)
    
####################################################################
#   denoising method for grayscale images
####################################################################
        
    def denoise_gray(self, im_out = None, verb = True):
        #denoising grayscale images
        out_path = _i_comp.denoise.denoise(self.im_dir,
                                           self.w_dir,
                                           self.shot_name,
                                           im_out, verb)
        return self.__class__(out_path)
    
#####################################################################
#  denoising method for color images
#####################################################################
        
    def denoise_col(self, im_out = None, verb = True):
        #denoising coloured images
        out_path = _i_comp.denoise_col.denoise_col(self.im_dir,
                                                   self.w_dir,
                                                   self.shot_name,
                                                   im_out, verb)
        return self.__class__(out_path)

#####################################################################
#   binary conversion method
#####################################################################
        
    def to_bin(self, im_out = None, verb = True):
        #converting to binary images
        out_path = _i_comp.to_binary.bin_thresh(self.im_dir,
                                                self.w_dir,
                                                self.shot_name,
                                                im_out, verb)
        return self.__class__(out_path)

#####################################################################
#   playing of the images
#####################################################################

    def play(self):
        #play images as a video
        _plot.playimages.play_img(self.__im_dir)
        
#####################################################################
#   cluster detection
#####################################################################
        
    def det_cluster(self, im_out = None, verb = True):
        #cluster detection subroutine        
        out_path, centers, area, total, angle, indt = _i_comp.cluster_det.det_cluster(self.im_dir,
                                                                                      self.w_dir,
                                                                                      self.shot_name,
                                                                                      im_out, verb)
        infocluster = {}
        infocluster['center'] =  centers
        infocluster['area'] =  area
        infocluster['total'] =  total
        infocluster['angle'] =  angle
        infocluster['indt'] =  indt
        #setting infocluster dictionary
        self.set_infocluster(infocluster)
        return out_path
        
#####################################################################
#   dumpro
#####################################################################        
        
    def dumpro(self, rate = None, tlim = None, hlim = None, wlim = None, blur = True,
               im_out = None, verb = True):
        #performing DUMPRO 
        infocluster, reshape, im_col = _i_comp.dumpro_img.dumpro_img(self.im_dir, 
                                                                     self.w_dir,
                                                                     self.shot_name, 
                                                                     rate, tlim, 
                                                                     hlim, wlim, blur,
                                                                     im_out, verb)
        self.set_infocluster(infocluster)
        self.set_reshape(reshape)
        
        
        
        
        return None
       

#############################################################################
#   Docstrings for image class methods
#############################################################################
Img_dir.to_gray.__doc__ = _i_comp.conv_gray.conv_gray.__doc__
Img_dir.denoise_col.__doc__ = _i_comp.denoise_col.denoise_col.__doc__
Img_dir.denoise_gray.__doc__ = _i_comp.denoise.denoise.__doc__
Img_dir.remove_backgrd.__doc__ = _i_comp.rm_background.rm_back.__doc__
Img_dir.to_bin.__doc__ = _i_comp.to_binary.bin_thresh.__doc__
Img_dir.play.__doc__ = _plot.playimages.play_img.__doc__
Img_dir.dumpro.__doc__ = _i_comp.dumpro_img.dumpro_img.__doc__                  

#####################################################################
#####################################################################
#   A class for handling videos and images 
#####################################################################

class Vid_img(object):
    """This is a derived class from Video class. This is an intermediate 
    approach, between working completely with videos and completely with
    images. This class was maily created beacause it follows a more 
    computationally robust way of detecting dust particles.
    
    Input:
    --------------------------------------------
    filename = video file along with the path
    
    Attributes:
    --------------------------------------------
    __filename     = Path of the file along with its name and extension
    __w_dir        = A path to a working directory for storage during
                     processing
    __shot_name    = Information on the shot being processed
    __im_dir       = Dictionary containing path of images created during 
                     preprocessing 
    __frame_width  = Width of the video
    __frame_height = Height of the video
    __N_frames     =  Total number of frames in the video
    __fps          = Number of frames per second 
    __fourcc       = The four character code of the video
    __meta_data    = Dictionary containing total frames, fps and frame size 
                     of video
    __reshape      = Dictionary containing the croping and time slicing 
                     of the video
    __infocluster  = Dictionary containing all the information regarding 
                     clusters
    
    Setters:
    --------------------------------------------
    set_w_dir       = Sets value for w_dir attribute
    set_reshape     = Sets value for reshape dictionary attribute
    set_infocluster = Sets value for infocluster dictionary attribute
    set_im_dir      = Setter for Image directory dictionary
    
    Getters:
    --------------------------------------------
    filename    = Returns the path of the video
    shotname    = Returns the shotname of the video
    resolution  = Returns the resolution of the video
    N_frames    = Returns the total number of frames of the video 
    fps         = Returns the frames per second of the video
    fourcc      = Returns the four character code of the video
    w_dir       = Returns the working directory of the video
    meta_data   = Returns the meta data of the video
    infocluster = Returns the infocluster dictionary of the video
    reshape     = Returns the rehsape dictionary of the video
    
    Methods:
    --------------------------------------------
    """
        
    def __init__(self, filename, w_dir = None, verb=True):

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
        
        #working directory
        if w_dir == None:
            msg = 'The working directory has not been provided !'
            msg += 'Falling to default values...\n'
            warnings.warn(msg)
            folder = '_dumpro'
            path_of_file, file = os.path.split(filename)
            file,ext = file.split('.')
            path = os.path.join(path_of_file, file + folder, '')
            if not os.path.exists(path):
                os.makedirs(path)
            self.__w_dir = path
            print('The working directory is :\n')
            print(self.__w_dir)
        else:
            msg = 'The path provided is not correct'
            msg += 'Please provide the correct path for the working directory'
            if not os.path.exits(w_dir):
                raise Exception(msg)
            self.__w_dir = w_dir
            print('The working directory is :\n')
            print(self.__w_dir)
        
        self.__shot_name = file
            
        #getting the meta data of the video
        self.__frame_width = int(self.cap.get(3))
        self.__frame_height = int(self.cap.get(4))
        self.__N_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.__fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        
        #calculating the duration of the video        
        self.__duration = self.__N_frames/self.__fps
        #meta_data dictionary
        self.__meta_data = {'fps' : self.__fps, 
                          'fourcc' : self.__fourcc, 
                          'N_frames' : self.__N_frames,
                          'frame_width' : self.__frame_width,
                          'frame_height' : self.__frame_height}
        
        self.__infocluster = {}
        self.__reshape = {}
        self.__c_id = None
        self.__traj = {}
        self.__im_dir = {}
        
#############################################################################
#     Setters for the class attributes
#############################################################################
    
    #setter for im_dir dictionary
    def set_im_dir(self, imdir):
        """Setter for im_dir dictionary"""
        self.__im_dir = imdir
    
    def set_w_dir(self, w_dir):
        """Setter function for Working Directory"""
        #getting user input
        self.__w_dir = w_dir
    
    def set_reshape(self, reshape):
        """Setters for reshape dictionary"""
        self.__reshape = reshape
        
    def set_infocluster(self, infocluster):
        """Setter for infocluster dictionary"""
        self.__infocluster = infocluster
 
    def set_c_id(self, c_id):
        """Setter for list of cluster objects"""
        self.__c_id = c_id
        
    def set_traj(self, traj):
        """Setter for trajectory objects"""
        self.__traj = traj
#############################################################################
#     Getters for the class attributes
#############################################################################
    @property
    def imdir(self):
        """Returns im_dir dictionary"""
        return self.__im_dir

    #defining a getter for the path of the video file
    @property
    def filename(self):
        """Returns the path containing the video"""
        return self.__filename
    
    #defining a getter for the shot nomenclature information
    @property
    def shotname(self):
        """Returns the shotname of the video"""
        return self.__shot_name
    
    #defining a getter for the video resolution
    @property
    def resolution(self):
        """Returns the size of frame of the video"""
        #getting width and height
        width = self.__frame_width
        height = self.__frame_height
        return (height, width)
    
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
    
    #defining a getter for working directory
    @property
    def w_dir(self):
        """Returns the working directory"""
        return self.__w_dir
    
    #defining a getter for meta_data dictionary
    @property
    def meta_data(self):
        """Returns the metadata of the videofile"""
        return self.__meta_data
    
    #defning a getter for infocluster dictioanry
    @property
    def infocluster(self):
        """Returns the infocluster dictionary"""
        return self.__infocluster
    
    #defining a getter for reshape dictionary
    @property
    def reshape(self):
        """Returns the reshape dictionary"""
        return self.__reshape
    
    @property
    def c_id(self):
        """Returns the cluster id list"""
        return self.__c_id
    
    @property
    def traj(self):
        """Returns the Trajctory dictionary"""
        return self.__traj

#############################################################################
#     Video to image converter method
#############################################################################
    
    #dumpro
    def dumpro(self, rate = None, tlim = None, hlim = None, wlim = None, 
               blur = True, im_out = None, verb = True):
        #starting time counter
        start_time = time.perf_counter()
        #performing preprocessing and cluster detection on the video
        infoclus, reshp, im_dir = _i_comp.dumpro_vid.dumpro_vid(self.filename,
                                                                self.w_dir,
                                                                self.shotname,
                                                                rate, tlim,
                                                                hlim, wlim, 
                                                                blur, im_out, 
                                                                self.meta_data,
                                                                verb)
        #setting infocluster dictionary
        self.set_infocluster(infoclus)
        #setting reshape dictionary
        self.set_reshape(reshp)
        #setting image directories dictionary
        self.set_im_dir(im_dir)
        
        ######################################################################
        #### Processing information on clusters                           ####
        ######################################################################
        
        #converting all clusters to objects
        c_id = _i_comp.get_id.get_id(self.infocluster, Cluster)
        #setting c_id 
        self.set_c_id(c_id)
        #calculating trajectories and assigning parent child value
        traj = _i_comp.get_relation.get_relation(self.c_id, self.infocluster)
        #using clusters with updated value to set c_id
        self.set_c_id(traj)
        
        ######################################################################
        #### Calculating trajectories                                     ####
        ######################################################################
        
        #getting a dictionary of trajectory objects
        traj_obs = _i_comp.trace_traj.trace_traj(traj)
        #l=getting all the keys as a list
        listofkeys = list(traj_obs.keys())
        n_traj = len(traj_obs)
        print(n_traj)
        trajects = {}
        for ii in range(0,n_traj):
            trajects[ii] = Trajectory(traj_obs.get(listofkeys[ii]))
        self.set_traj(trajects)

        end_time = time.perf_counter()
        print('---',end_time - start_time,' seconds ---')
        
        ######################################################################
        #### Plotting trajctories and distribution                        ####
        ######################################################################
        
        #plotting dust size distribution
        _plot.area_distrib.get_distrib(self.infocluster, self.w_dir, 
                                       self.shotname)
        #plotting framewise dust size distribution
        _plot.area_distrib.get_frame_distrib(self.infocluster, self.w_dir, 
                                             self.shotname)
        #plotting framewise dust distribution
        _plot.f_density_dist.num_dist(self.infocluster, self.w_dir, self.shotname)
        #plotting trajectories
        _plot.plottraj.plot_traj(self.traj, self.w_dir, self.shotname)
        
        return None
    
    def play_im(self):
        #play images as a video
        _plot.playimages.play_img(self.__im_dir)

Vid_img.dumpro.__doc__ = _i_comp.dumpro_vid.dumpro_vid.__doc__
Vid_img.play_im.__doc__ = _plot.playimages.play_img.__doc__
        


    
    
         
    
    
        
  