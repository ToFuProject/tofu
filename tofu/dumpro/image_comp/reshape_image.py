# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:18:45 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
# Built-in
import os
from sys import stdout
from time import sleep

# Standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")

def reshape_image(im_path, w_dir, shot_name, 
                  tlim = None, hlim = None, wlim = None, 
                  im_out = None, verb = True):
    """This subroutine crops a video and also slices it to return us a 
    collection of all the frames from our desired time window in our desired region
    of interest
    The images are read in original form i.e., without any modifications
    
    Among the parameters present, if used as a part of dumpro, 
    w_dir and shot_name are provided by the image processing 
    class in the core file.
    
    It is better to let the code use the default value of im_out. The default 
    method of setting the output path by the code is to create an output
    folder using the shotname in the working directory
    
    The region of interest and the time window of interest is later 
    gathered from the user during runtime.
    
    Parameters
    -----------------------
    im_path:          string
     input path where the images are stored
    w_dir:            string
     A working directory where the proccesed images are stored
    shot_name:        String
     The name of the tokomak machine and the shot number. Generally
     follows the nomenclature followed by the lab
    tlim:             tuple
     The time limits for the image files, i.e the frames of interest
    hlim, wlim:       tuple
     The height and width limits of the frame to select the region of interest
    im_out:           string
     The output path for the images after processing.
  
    Return
    -----------------------
    im_out:           String
     Path along where the proccessed images are stored 
    reshape:          dictionary
     A dictionary containing information of the selected and cropped frames
    """
    if verb == True:
        print('###########################################')
        print('Reshaping Images')
        print('###########################################\n')
    
    #the output directory based on w_dir and shot_name
    if verb == True:
        print('Creating output directory ...')
    #default output folder name
    folder = shot_name + '_slice'
    #creating the output directory
    if im_out == None:
        im_out = os.path.join(w_dir, folder, '')
        if not os.path.exists(im_out):
            os.mkdir(im_out)
    #the output directory shown to user
    if verb == True:
        print('output directory is : ', im_out,'\n')
        
    #creating a list of all the files
    files = [f for f in os.listdir(im_path) if os.path.isfile(os.path.join(im_path,f))]    
    #duration in terms of numberof frames
    duration = len(files)
    #sorting files according to names using lambda function
    #-4 is to remove the extension of the images i.e., .jpg
    files.sort(key = lambda x: int(x[5:-4]))
    #looping throuah all the file names in the list and converting them to image path
    
    if verb == True:
        print('Processing frames ...')
    
    #cropping frames based on time frame
    if tlim == None:
        start = 1
        end = len(files)
    else:
        start = int(tlim[0])
        end = int(tlim[1])

    #dynamic printing variable
    f_count = 1
    joblen = (end-start)
    #looping through the video
    print('creating temp file...\n')
    for time in range(0, duration):
        filename = im_path + files[time]
        #slicing video according to the interested frames
        if time >= start and time <= end:
            #dynamic printing
            if verb == True:
                stdout.write("\r[%s/%s]" % (f_count, joblen))
                stdout.flush()
            f_count += 1
            #reading each file to extract its meta_data
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            
            #cropping the frame
            if hlim == None and wlim == None:
                img = img
                hlim = (0,img.shape[0])
                wlim = (0,img.shape[1])
            elif wlim == None and hlim != None:
                img = img[hlim[0]:hlim[1],:]
                wlim = (0,img.shape[1])
            elif wlim != None and hlim == None:
                img = img[:,wlim[0]:wlim[1]]
                hlim = (0,img.shape[0])
            elif hlim != None and wlim != None:
                img = img[hlim[0]:hlim[1],wlim[0]:wlim[1]]
                
            #output name of image
            name =im_out + 'frame' + str(time) + '.jpg'
            #writting the output file
            cv2.imwrite(name,img)
        
    #dynamic printing
    stdout.write("\n")
    stdout.flush()
    
    if verb == True:
        print('rehsaping done...')
        
    print('Releasing output...')
    
    if verb == True:
        print('creating reshape dictionary...\n')
    #creating reshape dictionary
    reshape = {'height' : hlim, 'width' : wlim, 'tlim' : tlim}
    
    cv2.destroyAllWindows()
    
    return im_out, reshape
    