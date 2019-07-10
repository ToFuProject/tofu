# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:59:31 2019

@author: napra
"""

def seconds_to_minutes(time):
    
    temp = time/60
    minutes = int(temp)
    
    temp -= minutes
    
    seconds = temp*60
    seconds = round(seconds,2)
    
    return str(minutes)+':'+str(seconds)