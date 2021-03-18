


import numpy as np


from ._core_new import DataCollection
from . import _comp_spectrallines
from . import _plot_spectrallines


__all__ = ['Plasma2D']


#############################################
#############################################
#       Plasma2D
#############################################



class Plasma2D(DataCollection):

    _ddef = {'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
             'params': {'origin': (str, 'unknown'),
                        'dim':    (str, 'unknown'),
                        'quant':  (str, 'unknown'),
                        'name':   (str, 'unknown'),
                        'units':  (str, 'a.u.')}}
    _forced_group = ['time', 'radius', 'mesh2d']
    _data_none = False

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _max_ndim = 2













