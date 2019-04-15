# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:13:45 2019

@author: napra
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('/AK258850/imgpro/tofu/tofu/dumpro/KSTAR_003723_tv01.avi')&nbsp;
 
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

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory

trajectory = np.cumsum(transforms, axis=0) 

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
  
# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break
 
  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]
 
  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
 
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
 
  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 
 
  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])
 
  # If the image is too big, resize it.
  if(frame_out.shape[1] and gt; 1920): 
    frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
   
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(10)
  out.write(frame_out)
  
  def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame