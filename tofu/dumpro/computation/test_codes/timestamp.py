# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:30:31 2019

@author: napra
"""

import cv2
import numpy as np

cap = cv2.VideoCapture('E:/NERD/Python/KSTAR_013101_tv02_inj01.avi')
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
print("", timestamps)
calc_timestamps = [0.0]

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break

cap.release()

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print('Frame %d difference:'%i, abs(ts - cts))