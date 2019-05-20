#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:06:39 2019

@author: AK258850
"""

# Built-in
import os

# Standard
import numpy as np

# More special
try:
    import cv2
except ImportError:
    print("Could not find opencv package. Try pip intall opencv-contrib-python")
    
def getrectrangles(video_file, path = None, output_name = None, output_type = None):
    """
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    if path is None:
        path = os.path.join(drive,path_of_file)
    #checking for the name of the output file
    if output_name is None:
        output_name = file[0]+'_rect'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
        
    cap = cv.VideoCapture(video_file)
    
    ret, frame = cap.read()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame. cv2.COLOR_BGR2GRAY)
        image, contours, hier = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            
            x,y,w,h =cv2.boundingRect(c)
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
    
            box = np.int0(box)
    
            cv2.drawContours(frame, [box], 0, (0, 0, 255))
            
            (x,y,),radius = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            
            frame = cv2.circle(frame, center, radius,(0,255,0),2)
        print(len(contours))
        cv2.drawContours(frame, contours, -1,(255,255,0),1)
        if ret == True: 
            # Display the resulting frame 
            cv2.imshow('Frame', frame)         
            # Press Q on keyboard to  exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    cv2.destroyAllWindows()
    
    return None
        