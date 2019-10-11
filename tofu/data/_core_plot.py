
# -*- coding: utf-8 -*-

# Built-in
import sys
import os
# import itertools as itt
import copy
import warnings
if sys.version[0] == '3':
    import inspect
else:
    # Python 2 back-porting
    import funcsigs as inspect

# Common
import numpy as np

# tofu
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.data._core_new as _core_new
except Exception:
    from . import _core_new as _core_new


__all__ = ['DataCollectPlot']
_SAVEPATH = os.path.abspath('./')


#############################################
#############################################
#       Matplotlib
#############################################
#############################################


class DataCollectionPlot(_core_new.DataCollection):

    _dax = {}
    _dobj = {}
    _dx = {}
    _dy = {}

    # ---------------------
    # Method for getting corresponding nearest index in != ref
    # ---------------------

    def _get_current_ind_for_data(self, key, obj=None):
        if obj is None:
            ind = [self._dref['dict'][kr]['ind']
                   for kr in self._ddata['dict'][key]['refs']]
        else:
            ind = [self._dref['dict'][kr]['ind'][self._dobj['dict'][obj]['indi'][kr]]
                   for kr in self._ddata['dict'][key]['refs']]
        return ind

    def _get_current_data_along_axis(self, key, axis=None, obj=None):
        dim = self._ddata['dict'][key]['dim']
        if dim == 1:
            return self._ddata['dict'][key]['data']
        else:
            ind = self._get_current_ind_for_data(key)
            if dim == 2:
                if axis == 0:
                    return self._ddata['dict'][key]['data'][:,ind[1]]
                else:
                    return self._ddata['dict'][key]['data'][ind[0]]
            elif dim == 3:
                if axis == 0:
                    return self._ddata['dict'][key]['data'][:, ind[1], ind[2]]
                elif axis == 1:
                    return self._ddata['dict'][key]['data'][ind[0]][:, ind[2]]
                else:
                    return self._ddata['dict'][key]['data'][ind[0]][ind[1]]

    def _get_data_ind_from_value(self, value=None, key=None,
                                 ref=None, axis=None):
        if axis is None:
            axis = self._ddata['dict'][key]['refs'].index(ref)
            size = self._dref['dict'][key]['size']
        else:
            pass

        if self._ddata['dict'][key]['bins'] is not None:
            # bins only exist for 1d sorted data
            ind = np.searchsorted([value], self._ddata['dict'][key]['bins'])[0]
        else:
            x = self._get_current_data_along_axis(key, axis)
            ind = np.nanargmin(np.abs(x-value))
        return ind % size

    def _get_current_value(self, key):
        ind = self._get_current_ind_for_data(key)
        dim = self._ddata['dict'][key]['dim']
        if dim == 1:
            return self._ddata['dict'][key]['data'][ind[0]]
        elif dim == 2:
            return self._ddata['dict'][key]['data'][ind[0]][ind[1]]
        elif dim == 3:
            return self._ddata['dict'][key]['data'][ind[0]][ind[1]][ind[2]]
