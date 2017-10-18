# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys
from _core import Ves, Struct, _GG, _comp, _plot, _tfd

#__all__ = ['Ves', 'Struct',
#           '_GG','_comp','_plot','_tfd']




try:
    del _defaults, _core02, _plot02
except:
    try:
        del tofu.geom._defaults, tofu.geom._core02, tofu.geom._plot02
    except:
        pass

if sys.version[0]=='2':
    try:
        del _GG02
    except:
        try:
            del tofu.geom._GG02
        except:
            pass
else:
    try:
        del _GG03
    except:
        try:
            del tofu.geom._GG03
        except:
            pass

del sys
