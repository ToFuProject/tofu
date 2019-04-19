#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:29:57 2019

@author: AK258850
"""

from PIL import Image
import numpy as np

def load_image(file_name):
    img = Image.open(file_name)
    try:
        data = np.asarray(img, dtype='uint8')
    except SystemError:
        data = np.asarray(img.getdata(),dtype='uint8')
    return data
    

def save_image( npdata, out_filename ) :
    img = Image.fromarray( np.ndarray( np.clip(npdata,0,255), 
                                      dtype="unit8"), "L" )
    img.save( out_filename )