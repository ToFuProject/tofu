



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

    _dref = {'ece':{'ids':'ece', 'geom':False, 'signal':'t_e'},
             'sxr':{'ids':'soft_x_rays', 'geom':'CamLOS1D',
                    'signal':'power'},
             'bolo':{'ids':'bolometer', 'geom':'CamLOS1D'},
             'interfero':{'ids':'interferometer', 'geom':'CamLOS1D'},
             'brem':{'ids':'bremsstrahlung_visible', 'geom':'CamLOS1D'},
             'spectrovis':{'ids':'spectrometer_visible', 'geom':'CamLOS1D'}}



    #----------------
    # Class creation and instanciation
    #----------------

    def __init__(self, dids=None, diag=None, signal=None, tlim=None,
                 geom=True, data=True, verb=True):

        # Preformat

        # Check inputs
        dids, diag = self._checkformat_dins(dids=dids, diag=diag)

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

        # Get data
        self._get_ddiag(dids, idsnode, idskey=k0, geom=geom, data=data)

        # Close ids
        for k0 in lk0:
            dids[k0]['ids'].close()
        self._dids = dids




    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @classmethod
    def _checkformat_dins(cls, dids=None, diag=None):
        assert dids is not None
        assert diag is not None

        # diag
        assert diag in self._dref.keys()

        # Check dids
        lk0 = [diag]
        dids = {diag:{'dict':dids, 'ids':None}}
        lc = [vv['dict'] is None for vv in dids.values()]

        idsref = [k for k in lk0 if dids[k]['dict'] is not None][0]
        dids[idsref]['dict'] = _utils._get_defaults( dids=dids[idsref]['dict'] )
        for k in lk0:
            if dids[k]['dict'] is not None:
                dids[k]['dict'] = _utils._get_defaults( dids=dids[k]['dict'] )
        return dids, diag

    @staticmethod
    def _get_idsnode(ids, idskey=None):
        idsnode = eval('ids.%s'%self._dref[idkey]['ids'])
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


    def _get_ddiag(self, dids, idsnode, idskey=None,
                   signal=None, geom=True, data=True):

        # Initiate common dict
        Exp = dids['wall']['dict']['tokamak'].replace('_','')
        dcom = dict(Exp=Exp,
                    shot=dids['wall']['dict']['shot'])

        # Get geom
        cam = None
        if geom and self._dref[idskey] is not False:
            conf = ConfigLoader(dids=dids[idskey]['dict'])
            conf = conf.to_object(Name=)
            Ds = np.array([[cc.los_of_sight.first_point.r,
                            cc.los_of_sight.first_point.z,
                            cc.los_of_sight.first_point.phi]
                           for cc in idsnode.channel])
            ends = np.array([[cc.los_of_sight.second_point.r,
                            cc.los_of_sight.second_point.z,
                            cc.los_of_sight.second_point.phi]
                           for cc in idsnode.channel])
            Ds = np.array([Ds[:,0]*np.cos(Ds[:,2]),
                           Ds[:,0]*np.sin(Ds[:,2]),
                           Ds[:,1]])
            ends = np.array([ends[:,0]*np.cos(ends[:,2]),
                             ends[:,0]*np.sin(ends[:,2]),
                             ends[:,1]])
            us = ends-Ds
            us = us / np.sqrt(np.sum(us**2, axis=0))[None,:]

            if hasattr(idsnode.channel[0], 'etendue'):
                etend = np.array([cc.etendue for cc in idsnode.channel])
                etend[etend<0.] = np.nan
            else:
                etend = None
            c0 = hasattr(idsnode.channel[0], 'detector')
            c0 = c0 and hasattr(idsnode.channel[0].detector, 'surface')
            if c0:
                surf = np.array([cc.detector.surface
                                 for cc in idsnode.channel])
                surf[surf<0.] = np.nan
            else:
                surf = None

            cls = eval('tfg.%s'%self._dref[idskey]['geom'])
            cam = cls(Diag=idskey, Name='ids', dgeom=(Ds,us),
                      Etendues=etend, Surfaces=surf,
                      config=conf, **dcom)

        # Get data
        if data:
            if signal is None:
                signal = self._dref[idskey]['signal']
            data = np.array([eval('cc.%s.data'%signal)
                             for cc in idsnode])



        self._ddiag = dout
