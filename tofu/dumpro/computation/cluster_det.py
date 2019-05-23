#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:56:31 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
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
    
def cluster_det(video_file):
    """
    """
    try:
        if os.path.isfile(video_file):
            cap = cv2.VideoCapture(video_file)
    except IOError:
        print('the path or filename is incorrect.')
        print('PLease verify the path or file name and try again')
            
    ret, frame = cap.read()
    
    back = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret: break
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movie = back.apply(frame)
        ret, threshed_img = cv2.threshold(movie,
                127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshed_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            #get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            #draw a red 'nghien' rectangle
            cv2.drawContours(frame, [box], 0, (0, 0, 255))
 
            #finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            #convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            #and draw the circle in blue
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            print(center)
        print(len(contours))
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        if ret == True: 
            # Display the resulting frame 
            cv2.imshow('Frame', frame)         
            # Press Q on keyboard to  exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    # When everything done, release  
    # the video capture object 
    cap.release() 
                
    # Closes all the frames 
    cv2.destroyAllWindows() 