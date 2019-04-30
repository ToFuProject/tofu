



# Built-in
import sys
import os
import warnings
import itertools as itt
import operator
if sys.version[0] == '2':
    import funcsigs as inspect
else:
    import inspect

# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib as mpl

# tofu-specific
try:
    import tofu.imas2tofu._utils as _utils
    import tofu.geom as tfg
except Exception:
    from . import _utils
    from .. import geom as tfg

# imas-specific
import imas



__all__ = ['ConfigLoader','load_Config']

#######################################
#######################################
#       class
######################################

class DiagLoader(object):
    """ A generic class for handling diag data and geometry

    Provides:
        - DataCamND, optionally with CamLOSND and config

    """

    #----------------
    # Class attributes
    #----------------


    #----------------
    # Class creation and instanciation
    #----------------

    def __init__(self, dids=None, verb=True):

        # Preformat

        # Check inputs
        dids = self._checkformat_dins(dids=dids)

        # Get quantities
        lk0 = list(dids.keys())
        for k0 in lk0:
            # Fall back to idsref if None
            dids[k0]['ids'] = self._openids(dids[k0]['dict'])
            if k0 == lk0[0]:
                msg = ''.rjust(20)
                msg += '  '.join([kk.rjust(vv)
                                  for kk,vv in self._didsk.items()])
                print(msg)
            msg = ("Getting %s..."%k0).rjust(20)
            if dids[k0]['dict'] is not None:
                msg += '  '.join([str(dids[k0]['dict'][kk]).rjust(vv)
                                    for kk,vv in self._didsk.items()])
            print(msg)
            idsnode = self._get_idsnode(dids[k0]['ids'], k0)

        # Get dStruct
        dStruct = self._get_dStruct(dids, idsnode)

        # Close ids
        for k0 in lk0:
            dids[k0]['ids'].close()
        self._dids = dids
