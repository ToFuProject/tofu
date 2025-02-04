# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:49:55 2025

@author: dvezinet
"""

import numpy as np


# Useful scalar types
_DTYPES = {
    'nint': (np.int32, np.int64),
    'nfloat': (np.float32, np.float64),
    'bool': (bool, np.bool_),
    'str': (str, np.str_),
}

_DTYPES['int'] = _DTYPES['nint'] + (int,)
_DTYPES['float'] = _DTYPES['nfloat'] + (float,)
_DTYPES['scalar'] = _DTYPES['int'] + _DTYPES['float']