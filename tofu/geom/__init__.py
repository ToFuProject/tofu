# -*- coding: utf-8 -*-
#! /usr/bin/python


"""
Load all core packages and modules which are all machine-independent, diagnostic-independent and code-independent
"""


from ._core import *
from . import General_Geom_cy as _GG

del _core, General_Geom_cy

__author__ = "Didier Vezinet"
__all__ = ['Ves','Struct','LOS','GLOS','Lens','Apert','Detect','GDetect']


