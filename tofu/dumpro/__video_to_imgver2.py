# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:03 2019

@author: napra

input: video_file (The Grayscale converted video)
output: all the frames in video are converted to images
"""



import cv2
import numpy as np
import os
import __colorgray
# Playing video from file:
def video2imgconvertor(video_file):
    print(video_file)
    print("inside convertor right now")
    cap = cv2.VideoCapture(video_file)

    try:
        if not os.path.exists('/video/data'):
            os.makedirs('/video/data')

    except OSError:
        print ('Error: Creating directory of data')
        
    currentFrame = 0
        
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        
        # To stop duplicate images
        currentFrame += 1
        if not ret:
            break
        # When everything done, release the capture
            
    cap.release()
    cv2.destroyAllWindows()
    
    return 'successful'