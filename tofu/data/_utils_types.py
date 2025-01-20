# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:49:55 2025

@author: dvezinet
"""

import numpy as np

__all__ = [
    '_NINT', '_INT',
    '_NFLOAT', '_FLOAT',
    '_NUMB',
    '_BOOL',
]


# Useful scalar types
_NINT = (np.int32, np.int64)
_INT = (int,) + _NINT
_NFLOAT = (np.float32, np.float64)
_FLOAT = (float,) + _NFLOAT
_NUMB = _INT + _FLOAT
_BOOL = (bool, np.bool_)