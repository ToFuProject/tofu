# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 07:30:35 2019

@author: napra
"""

import os  
path="C:/Users/napra/Downloads/AVI_files/AVI_files/"  
if os.path.isdir(path):  
    print("\nIt is a directory")  
elif os.path.isfile(path):  
    print("\nIt is a normal file")  
else:  
    print("It is a special file (socket, FIFO, device file)" )
print()
