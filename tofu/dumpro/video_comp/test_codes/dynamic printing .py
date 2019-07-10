# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:52:04 2019

@author: napra
"""

from sys import stdout
from time import sleep

jobLen = 100
progressReport = 0

while progressReport < jobLen:
    progressReport += 1
    stdout.write("\r[%s/%s]" % (progressReport, jobLen))
    stdout.flush()
    sleep(0.1)

stdout.write("\n")
stdout.flush()