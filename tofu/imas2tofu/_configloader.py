


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

class ConfigLoader(object):
    """ A generic class for handling tokamak descriptions

    Provides:
        - config

    """

    #----------------
    # Class attributes
    #----------------

    _didsk = {'tokamak':15, 'user':15, 'version':7,
              'shot':6, 'run':3, 'occ':3, 'shotr':6, 'runr':3}


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
        self._get_dStruct(dids, idsnode)

        # Close ids
        for k0 in lk0:
            dids[k0]['ids'].close()
        self._dids = dids




    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @classmethod
    def _checkformat_dins(cls, dids=None):
        assert dids is not None

        # Check dids
        lk0 = ['wall']
        dids = {'wall':{'dict':dids, 'ids':None}}
        lc = [vv['dict'] is None for vv in dids.values()]

        idsref = [k for k in lk0 if dids[k]['dict'] is not None][0]
        dids[idsref]['dict'] = _utils._get_defaults( dids=dids[idsref]['dict'] )
        for k in lk0:
            if dids[k]['dict'] is not None:
                dids[k]['dict'] = _utils._get_defaults( dids=dids[k]['dict'] )
        return dids

    @staticmethod
    def _get_idsnode(ids, idskey='wall'):
        if idskey == 'wall':
            idsnode = ids.wall
        idsnode.get()
        return idsnode

    @staticmethod
    def _openids(dids):
        try:
            ids = imas.ids(s=dids['shot'], r=dids['run'])
        except Exception:
            msg = "Error running:\n"
            msg += "    ids = imas.ids(s=%s, r=%s)"%(str(dids['shot']),
                                                     str(dids['run']))
            raise Exception(msg)
        try:
            ids.open_env(dids['user'], dids['tokamak'], dids['version'])
        except Exception:
            ids.close()
            msg = "Error running:\n"
            msg += "    ids.open_env(%s, %s, %s)"%(dids['user'],
                                                   dids['tokamak'],
                                                   dids['version'])
            raise Exception(msg)
        return ids

    def _get_dStruct(self, dids, idsnode):

        # Initiate common dict
        Exp = dids['wall']['dict']['tokamak'].replace('_','')
        dcom = dict(Exp=Exp,
                    shot=dids['wall']['dict']['shot'],
                    Type='Tor')

        # Get limiters
        lpfc = idsnode.description_2d[0].limiter.unit
        if len(lpfc) == 1:
            pfc = lpfc[0]
            name = 'dummy' if pfc.name == '' else pfc.name
            dout = {'PlasmaDomain': [{'Name':name,
                                      'Poly':np.array([pfc.outline.r,
                                                       pfc.outline.z])}]}
        elif len(lpfc) > 1:
            # Add limits
            lout = []
            for ii in range(0,len(lpfc)):
                if pfc.name == '':
                    name = 'pfc%s'%str(ii)
                else:
                    name = lpfc[ii].name
                lout.append({'Name':name,
                             'Poly':np.array([lpfc[ii].outline.r,
                                             lpfc[ii].outline.z])})
            dout = {'PFC': lout}


        # get vessel


        # Add dcom to all
        for cls,ll in dout.items():
            for ii in range(0,len(ll)):
                dout[cls][ii].update(dcom)

        self._dStruct = dout
        self._nStruct = np.sum([len(vv) for vv in dout.values()])


    def to_object(self, Name=None):

        lS = [None for ii in range(0,self._nStruct)]
        ii = 0
        for cls, ll in self._dStruct.items():
            cc = eval('tfg.%s'%cls)
            for dd in ll:
                lS[ii] = cc(**dd)
            ii += 1

        config = tfg.Config(Name=Name, lStruct=lS, Exp=lS[0].Id.Exp)
        return config


#######################################
#######################################
#       function
######################################

def load_Config(Name=None, **kwdargs):

    config = ConfigLoader(**kwdargs)
    return config.to_object(Name=Name)

sig = inspect.signature(ConfigLoader)
kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
params = list(sig.parameters.values())
params = params + [params[0].replace(name='Name', default=None, kind=kind)]
load_Config.__signature__ = sig.replace(parameters=params)

del sig, params, kind
