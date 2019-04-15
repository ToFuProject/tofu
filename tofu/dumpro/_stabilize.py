# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:13:45 2019

@author: napra
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('E:/NERD/Python/KSTAR_003723_tv01.avi')&nbsp;
 
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('Stab_video.avi', fourcc, 25, (width,height))

ret,frame = cap.read()

transforms = np.zeros((n_frames-1,3), np.float32)

for i in range(n_frames-2):
    prev_pts = cv2.goodFeaturesToTrack(frame, maxCorners = 200,qualityLevel = 0.01,minDistance = 30,blockSize = 3)
    
    success, curr = cap.read()
    if not success:
        break
    
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(frame,curr,prev_pts,None)
    
    assert prev_pts.shape == curr_pts.shape
    
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)
    
    dx = m[0,2]
    dy = m[1,2]
    
    da = np.arctan2(m[1,0], m[0,0])
    
    transforms[i] =[dx,dy,da]
    
    frame = curr
    
    print("frame:" +str(i) + "/" + str(n_frames) + "- Tracked points : " + str(len(prev_pts)))
    
    trajectory = np.cumsum(transforms,axis=0)

def movingAverage(curve,radius):
    window_size = 2* radius + 1
    
    f = np.ones(window_size)/window_size
    curve_pad= np.lib.pad(curve,(radius,radius),'edge')
    curve_smoothed = np.convolve(curve_pad,f,mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    
    return curve_smoothed

