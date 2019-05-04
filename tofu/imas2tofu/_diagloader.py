



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
    import tofu.geom as tfg
    import tofu.data as tfd
    import tofu.imas2tofu._utils as _utils
    from tofu.imas2tofu._configloader import ConfigLoader
except Exception:
    from .. import geom as tfg
    from .. import data as tfd
    from . import _utils
    from ._configloader import ConfigLoader

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

    _dref = {'ece':{'ids':'ece',
                    'geom':False,
                    'data':'DataCam1D',
                    'signal':'t_e',
                    'X':'position.r',
                    'units':r'$eV$'},
             'sxr':{'ids':'soft_x_rays',
                    'geom':'CamLOS1D',
                    'data':'DataCam1D',
                    'signal':'power',
                    'units':r'$W$'},
             'bolo':{'ids':'bolometer',
                     'geom':'CamLOS1D',
                     'data':'DataCam1D',
                     'signal':'power',
                     'units':r'$W$'},
             'interfero':{'ids':'interferometer',
                          'geom':'CamLOS1D',
                          'data':'DataCam1D',
                          'signal':'n_e_line',
                          'units':r'$/m^2$'},
             'brem':{'ids':'bremsstrahlung_visible',
                     'geom':'CamLOS1D',
                     'data':'DataCam1D',
                     'signal':'radiance_spectral',
                     'units':r'$photons/(m^2.sr.s.m)$'},
             'spectrovis':{'ids':'spectrometer_visible',
                           'geom':'CamLOS1D',
                           'data':'DataCam1DSpectral',
                           'signal':'grating_spectrometer.radiance_spectral',
                           'lamb':'grating_spectrometer.wavelengths',
                           'units':r'$photons/(m^2.sr.s.m)$'}}

    _didsk = {'tokamak':15, 'user':15, 'version':7,
              'shot':6, 'run':3, 'occ':3, 'shotr':6, 'runr':3}

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
        assert diag in cls._dref.keys()

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

    @classmethod
    def _get_idsnode(cls, ids, idskey=None):
        idsnode = getattr(ids, cls._dref[idskey]['ids'])
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

    def _get_shapes(self, idsnode, idskey, datastr, lambstr=None):
        (s0, s1) = datastr.split('.') if '.' in datastr else (datastr, None)
        nch = len(idsnode.channel)
        ls = None
        if s1 is None:
            ts = np.squeeze([getattr(cc,s0).time.shape
                             for cc in idsnode.channel])
            ds = np.squeeze([getattr(cc,s0).data.shape
                             for cc in idsnode.channel])
        else:
            ts = np.squeeze([getattr(getattr(cc,s0),s1).time.shape
                             for cc in idsnode.channel])
            ds = np.squeeze([getattr(getattr(cc,s0),s1).data.shape
                             for cc in idsnode.channel])
        if lambstr is not None:
            (ls0, ls1) = (lambstr.split('.') if '.' in lambstr
                          else (lambstr, None))
            if ls1 is None:
               ls = np.squeeze([getattr(cc,ls0).shape
                                for cc in idsnode.channel])
            else:
               ls = np.squeeze([getattr(getattr(cc,ls0),ls1).shape
                                for cc in idsnode.channel])

            ts = np.argmax(np.bincount(ts[ts>0]))
            if ls is not None:
                assert ds.ndim == 2
                ls = np.bincount(ls[ls>0]).argmax()
                indch0 = (ds[:,0] == ts) & (ds[:,1] == ls)
                indch1 = (ds[:,1] == ts) & (ds[:,0] == ls)
                lc = [indch0.sum() == 0, indch1.sum() == 0]
                assert np.sum(lc) == 1
                indch = indch1 if lc[0] else indch0
            else:
                assert ds.ndim == 1
                indch = ds == ts
        return indch, ts, ls

    def _get_ddiag(self, dids, idsnode, idskey=None,
                   signal=None, units=None, geom=True, data=True):

        dout = {}

        # Initiate common dict
        Exp = dids[idskey]['dict']['tokamak'].replace('_','')
        shot = dids[idskey]['dict']['shot']

        # --------------
        # Get data
        # --------------
        if signal is None:
            signal = self._dref[idskey]['signal']
            units = self._dref[idskey]['units']
        lambstr = None
        if 'lamb' in self._dref[idskey].keys():
            lambstr = self._dref[idskey]['lamb']

        lamb = None
        if data:
            (s0, s1) = signal.split('.') if '.' in signal else (signal, None)
            if not idsnode.ids_properties.homogeneous_time == 1:
                mm = "    - ids.{0}.ids_properties.homogeneous_time = {1}"
                msg = "This version of tofu only handles homogenous time\n"
                msg += "The following ids does not fulfill this criteron:\n"
                msg += mm.format(idskey,
                                 idsnode.ids_properties.homogeneous_time)
                warnings.warn(msg)
                indch, ts, ls = self._get_shapes(idsnode, idskey,
                                                 signal, lambstr=lambstr)
                chans = [idsnode.channel[ii] for ii in indch.nonzero()[0]]
                if s1 is None:
                    t = getattr(chans[0],s0).time
                else:
                    t = getattr(getattr(chans[0],s0),s1).time
                if lambstr is not None:
                    (sl0, sl1) = (lambstr.split('.') if '.' in lambstr
                                else (lambstr, None))
                    if sl1 is None:
                        lamb = getattr(chans[0],sl0)
                    else:
                        lamb = getattr(getattr(chans[0],sl0),sl1)
            else:
                t = idsnode.time
                chans = idsnode.channel

            if s1 is None:
                data = np.squeeze([getattr(cc,s0).data for cc in chans]).T
            else:
                data = np.squeeze([getattr(getattr(cc,s0),s1).data
                                   for cc in chans]).T
            if idskey == 'spectrovis':
                data = np.swapaxes(data, 1,2)

            # Get X if any
            X = None
            if 'X' in self._dref[idskey].keys():
                Xstr = self._dref[idskey]['X']
                (x0, x1) = Xstr.split('.') if '.' in Xstr else (Xstr,None)
                if x1 is None:
                    X = np.array([getattr(cc,x0).data for cc in chans]).T
                else:
                    X = np.array([getattr(getattr(cc,x0),x1).data for cc in chans]).T

        # --------------
        # Get geom
        # --------------
        lCam = None
        if geom and self._dref[idskey]['geom'] is not False:
            conf = ConfigLoader(dids=dids[idskey]['dict'])
            conf = conf.to_object(Name='ids')
            Ds = np.array([[cc.line_of_sight.first_point.r,
                            cc.line_of_sight.first_point.z,
                            cc.line_of_sight.first_point.phi]
                           for cc in chans])
            ends = np.array([[cc.line_of_sight.second_point.r,
                            cc.line_of_sight.second_point.z,
                            cc.line_of_sight.second_point.phi]
                           for cc in chans])
            Ds = np.array([Ds[:,0]*np.cos(Ds[:,2]),
                           Ds[:,0]*np.sin(Ds[:,2]),
                           Ds[:,1]])
            ends = np.array([ends[:,0]*np.cos(ends[:,2]),
                             ends[:,0]*np.sin(ends[:,2]),
                             ends[:,1]])
            us = ends-Ds
            us = us / np.sqrt(np.sum(us**2, axis=0))[None,:]
            if np.any(np.abs(Ds)>1000.):
                # mm -> m
                Ds = Ds / 1000.

            if hasattr(chans[0], 'etendue'):
                etend = np.squeeze([cc.etendue for cc in chans])
                etend[etend<0.] = np.nan
            else:
                etend = None
            c0 = hasattr(chans[0], 'detector')
            c0 = c0 and hasattr(chans[0].detector, 'surface')
            if c0:
                surf = np.squeeze([cc.detector.surface for cc in chans])
                surf[surf<0.] = np.nan
            else:
                surf = None

            # Get names, try deducing cameras
            cls = eval('tfg.%s'%self._dref[idskey]['geom'])
            names = [cc.name for cc in chans]
            if all([nn != '' for nn in names]) and idskey == 'spectrovis':
                lcam, nb = zip(*[(nn[:-2], nn[-2:]) for nn in names])
                lcamu = list(set(lcam))
                lind = [np.asarray(lcam) == cam for cam in lcamu]
                lCam = [cls(Diag=idskey, Name=lcamu[ii],
                            dgeom=(Ds[:,lind[ii]], us[:,lind[ii]]),
                            Etendues=etend[lind[ii]],
                            Surfaces=surf[lind[ii]],
                            dchans={'names':np.asarray(names)[lind[ii]]},
                            config=conf, Exp=Exp, shot=shot)
                        for ii in range(0,len(lcamu))]
            else:
                lCam = [cls(Diag=idskey, Name='ids', dgeom=(Ds,us),
                            Etendues=etend, Surfaces=surf,
                            config=conf, Exp=Exp, shot=shot)]

        dout[self._dref[idskey]['data']] = {'data':data, 't':t, 'X':X, 'lamb':lamb,
                                            'Diag':idskey, 'lCam':lCam, 'Exp':Exp,
                                            'shot':shot}
        self._ddiag = dout

    def to_object(self, Name=None):
        if Name is None:
            Name = 'ids'
        for k, v in self._ddiag.items():
            cls = getattr(tfd,k)
            obj = cls(Name=Name, **v)
        return obj



#######################################
#######################################
#       function
######################################

def load_Diag(Name=None, **kwdargs):
    diag = DiagLoader(**kwdargs)
    try:
        diag = diag.to_object(Name=Name)
    except Exception as err:
        msg = "Could not convert to object !"
        msg += str(err)
        warnings.warn(msg)
    return diag

sig = inspect.signature(DiagLoader)
kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
params = list(sig.parameters.values())
params = params + [params[0].replace(name='Name', default=None, kind=kind)]
load_Diag.__signature__ = sig.replace(parameters=params)

del sig, params, kind
