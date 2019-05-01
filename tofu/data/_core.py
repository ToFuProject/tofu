# -*- coding: utf-8 -*-

# Built-in
import sys
import os
import itertools as itt
import copy
import warnings
from abc import ABCMeta, abstractmethod
if sys.version[0] == '3':
    import inspect
else:
    # Python 2 back-porting
    import funcsigs as inspect

# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation as mplTri
from matplotlib.tri import LinearTriInterpolator as mplTriLinInterp
try:
    import pandas as pd
except Exception:
    warnings.warn("pandas could not be imported => no get_summary()")


# tofu
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.data._comp as _comp
    import tofu.data._plot as _plot
    import tofu.data._def as _def
except Exception:
    from . import _comp as _comp
    from . import _plot as _plot
    from . import _def as _def

__all__ = ['DataCam1D','DataCam2D',
           'DataCam1DSpectral','DataCam2DSpectral',
           'Plasma2D']


#############################################
#       utils
#############################################

def _format_ind(ind=None, n=None):
    if ind is None:
        ind = np.ones((n,),dtype=bool)
    else:
        lInt = [int,np.int64]
        if type(ind) in lInt:
            ii = np.zeros((n,),dtype=bool)
            ii[int(ii)] = True
            ind = ii
        else:
            assert hasattr(ind,'__iter__')
            if type(ind[0]) in [bool,np.bool_]:
                ind = np.asarray(ind).astype(bool)
                assert ind.size==n
            elif type(ind[0]) in lInt:
                ind = np.asarray(ind).astype(int)
                ii = np.zeros((n,),dtype=bool)
                ii[ind] = True
                ind = ii
            else:
                msg = "Index must be a int, or an iterable of bool or int !"
                raise Exception(msg)
    return ind


def _select_ind(v, ref, nRef):
    ltypes = [int,float,np.int64,np.float64]
    C0 = type(v) in ltypes
    C1 = type(v) is np.ndarray
    C2 = type(v) is list
    C3 = type(v) is tuple
    assert v is None or np.sum([C0,C1,C2,C3])==1
    nnRef = 1 if ref.ndim==1 else ref.shape[0]
    ind = np.zeros((nnRef,nRef),dtype=bool)
    if v is None:
        ind = ~ind
    elif C0 or C1:
        if C0:
            v = np.r_[v]
        # Traditional :
        #for vv in v:
        #    ind[np.nanargmin(np.abs(ref-vv))] = True
        # Faster with digitize :
        if ref.ndim==1:
            ind[0,np.digitize(v, (ref[1:]+ref[:-1])/2.)] = True
        elif ref.ndim==2:
            for ii in range(0,ref.shape[0]):
                ind[ii,np.digitize(v, (ref[ii,1:]+ref[ii,:-1])/2.)] = True

    elif C2 or C3:
        c0 = len(v)==2 and all([type(vv) in ltypes for vv in v])
        c1 = all([(type(vv) is type(v) and len(vv)==2
                   and all([type(vvv) in ltypes for vvv in vv]))
                  for vv in v])
        assert c0!=c1
        if c0:
            v = [v]
        for vv in v:
            ind = ind | ((ref>=vv[0]) & (ref<=vv[1]))
        if C3:
            ind = ~ind
    if ref.ndim==1:
        ind = ind.squeeze()
    return ind




#############################################
#       class
#############################################

class DataAbstract(utils.ToFuObject):

    __metaclass__ = ABCMeta

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dtreat':{'order':['mask','interp-indt','interp-indch','data0','dfit',
                                'indt', 'indch', 'indlamb', 'interp-t']}}

    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, **kwdargs):
        # Python 2
        super(DataAbstract,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(DataAbstract._ddef)
        #cls._dplot = copy.deepcopy(Struct._dplot)
        #cls._set_color_ddef(cls._color)


    def __init__(self, data=None, t=None, X=None, lamb=None,
                 dchans=None, dlabels=None, dX12='geom',
                 Id=None, Name=None, Exp=None, shot=None, Diag=None,
                 dextra=None, lCam=None, config=None,
                 fromdict=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        #self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(DataAbstract,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(DataAbstract,self)._reset()
        self._ddataRef = dict.fromkeys(self._get_keys_ddataRef())
        self._dtreat = dict.fromkeys(self._get_keys_dtreat())
        self._ddata = dict.fromkeys(self._get_keys_ddata())
        self._dlabels = dict.fromkeys(self._get_keys_dlabels())
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dchans = dict.fromkeys(self._get_keys_dchans())
        self._dextra = dict.fromkeys(self._get_keys_dextra())
        if self._is2D():
            self._dX12 = dict.fromkeys(self._get_keys_dX12())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None,
                               Diag=None, include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot, Diag = Id.Name, Id.Exp, Id.shot, Id.Diag
        assert type(Name) is str, Name
        assert type(Diag) is str, Diag
        assert type(Exp) is str, Exp
        if include is None:
            include = cls._ddef['Id']['include']
        assert shot is None or type(shot) in [int,np.int64]
        if shot is None:
            if 'shot' in include:
                include.remove('shot')
        else:
            shot = int(shot)
            if 'shot' not in include:
                include.append('shot')
        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot,
                        'Diag':Diag, 'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_ddataRef():
        largs = ['data','t',
                 'X', 'indtX',
                 'lamb', 'indtlamb', 'indXlamb', 'indtXlamb']
        return largs

    @staticmethod
    def _get_largs_ddata():
        largs = []
        return largs

    @staticmethod
    def _get_largs_dtreat():
        largs = ['dtreat']
        return largs

    @staticmethod
    def _get_largs_dlabels():
        largs = ['dlabels']
        return largs

    @staticmethod
    def _get_largs_dgeom():
        largs = ['lCam','config']
        return largs

    @staticmethod
    def _get_largs_dX12():
        largs = ['dX12']
        return largs

    @staticmethod
    def _get_largs_dchans():
        largs = ['dchans']
        return largs

    @staticmethod
    def _get_largs_dextra():
        largs = ['dextra']
        return largs


    ###########
    # Get check and format inputs
    ###########



    def _checkformat_inputs_ddataRef(self, data=None, t=None,
                                     X=None, indtX=None,
                                     lamb=None, indtlamb=None,
                                     indXlamb=None, indtXlamb=None):
        assert data is not None
        data = np.asarray(data).squeeze()

        if t is not None:
            t = np.asarray(t).squeeze()
        if X is not None:
            X = np.asarray(X).squeeze()
        if indtX is not None:
            indtX = np.asarray(indtX, dtype=int).squeeze()
        if lamb is not None:
            lamb = np.asarray(lamb).squeeze()
        if indtlamb is not None:
            indtlamb = np.asarray(indtlamb, dtype=int).squeeze()
        if indXlamb is not None:
            indXlamb = np.asarray(indXlamb, dtype=int).squeeze()
        if indtXlamb is not None:
            indtXlamb = np.asarray(indtXlamb, dtype=int).squeeze()

        ndim = data.ndim
        assert ndim in [2,3]
        if not self._isSpectral():
            msg = "self is not of spectral type"
            msg += "\n  => the data cannot be 3D ! (ndim)"
            assert ndim==2, msg

        nt = data.shape[0]
        if t is None:
            t = np.arange(0,nt)
        else:
            assert t.shape==(nt,)

        n1 = data.shape[1]
        if ndim==2:
            lC = [X is None, lamb is None]
            assert any(lC)
            if all(lC):
                if self._isSpectral():
                    X = np.array([0])
                    lamb = np.arange(0,n1)
                    data = data.reshape((nt,1,n1))
                else:
                    X = np.arange(0,n1)
            elif lC[0]:
                assert self._isSpectral()
                X = np.array([0])
                data = data.reshape((nt,1,n1))
                assert lamb.ndim in [1,2]
                if lamb.ndim==1:
                    assert lamb.size==n1
                elif lamb.ndim==2:
                    assert lamb.shape[1]==n1
            else:
                assert not self._isSpectral()
                assert X.ndim in [1,2]
                assert X.shape[-1]==n1
        else:
            assert self._isSpectral()
            n2 = data.shape[2]
            lC = [X is None, lamb is None]
            if lC[0]:
                X = np.arange(0,n1)
            else:
                assert X.ndim in [1,2]
                assert X.shape[-1]==n1

            if lC[1]:
                lamb = np.arange(0,n2)
            else:
                assert lamb.ndim in [1,2]
                if lamb.ndim==1:
                    assert lamb.size==n2
                else:
                    assert lamb.shape[1]==n2
        if X.ndim==1:
            X = np.array([X])
        if lamb is not None and lamb.ndim==1:
            lamb = np.array([lamb])


        # Get shapes
        nt, nch = data.shape[:2]
        nnch = X.shape[0]
        if data.ndim==3:
            nnlamb, nlamb = lamb.shape
        else:
            nnlamb, nlamb = 0, 0

        # Check indices
        if indtX is not None:
            assert indtX.shape==(nt,)
            assert np.min(indtX)>=0 and np.max(indtX)<=nnch
        lC = [indtlamb is None, indXlamb is None, indtXlamb is None]
        assert lC[2] or (~lC[2] and np.sum(lC[:2])==2)
        if lC[2]:
            if not lC[0]:
                assert indtlamb.shape==(nt,)
                assert inp.min(indtlamb)>=0 and np.max(indtlamb)<=nnlamb
            if not lC[1]:
                assert indXlamb.shape==(nch,)
                assert inp.min(indXlamb)>=0 and np.max(indXlamb)<=nnlamb
        else:
            assert indtXlamb.shape==(nt,nch)
            assert inp.min(indtXlamb)>=0 and np.max(indtXlamb)<=nnlamb

        # Check consistency X/lamb shapes vs indices
        if X is not None and indtX is None:
            assert nnch in [1,nt]
            if lamb is not None:
                if all([ii is None for ii in [indtlamb,indXlamb,indtXlamb]]):
                    assert nnlamb in [1,nch]

        l = [data, t, X, lamb, nt, nch, nlamb, nnch, nnlamb,
             indtX, indtlamb, indXlamb, indtXlamb]
        return l

    def _checkformat_inputs_XRef(self, X=None, indtX=None, indXlamb=None):
        if X is not None:
            X = np.asarray(X).squeeze()
        if indtX is not None:
            indtX = np.asarray(indtX).squeeze()
        if indXlamb is not None:
            indXlamb = np.asarray(indXlamb).squeeze()

        ndim = self._ddataRef['data'].ndim
        nt, n1 = self._ddataRef['data'].shape[:2]

        if ndim==2:
            if X is None:
                if self._isSpectral():
                    X = np.array([0])
                else:
                    X = np.arange(0,n1)
            else:
                assert not self._isSpectral()
                assert X.ndim in [1,2]
                assert X.shape[-1]==n1
        else:
            if X is None:
                X = np.arange(0,n1)
            else:
                assert X.ndim in [1,2]
                assert X.shape[-1]==n1
        if X.ndim==1:
            X = np.array([X])

        # Get shapes
        nnch, nch = X.shape

        # Check indices
        if indtX is None:
            indtX = self._ddataRef['indtX']
        if indtX is not None:
            assert indtX.shape==(nt,)
            assert inp.argmin(indtX)>=0 and np.argmax(indtX)<=nnch
        if indXlamb is None:
            indXlamb = self._ddataRef['indXlamb']
        if indtXlamb is None:
            indtXlamb = self._ddataRef['indtXlamb']

        if indtXlamb is not None:
            assert indXlamb is None
            assert indXlamb.shape==(nch,)
            assert (np.argmin(indXlamb)>=0
                    and np.argmax(indXlamb)<=self._ddataRef['nnlamb'])
        else:
            assert indXlamb is None
            assert indtXlamb.shape==(nt,nch)
            assert (np.argmin(indtXlamb)>=0
                    and np.argmax(indtXlamb)<=self._ddataRef['nnlamb'])

        return X, nnch, indtX, indXlamb, indtXlamb


    def _checkformat_inputs_dlabels(self, dlabels=None):
        if dlabels is None:
            dlabels = {}
        assert type(dlabels) is dict
        lk = ['data','t','X']
        if self._isSpectral():
            lk.append('lamb')
        for k in lk:
            if not k in dlabels.keys():
                dlabels[k] = {'name': k, 'units':'a.u.'}
            assert type(dlabels[k]) is dict
            assert all([s in dlabels[k].keys() for s in ['name','units']])
            assert type(dlabels[k]['name']) is str
            assert type(dlabels[k]['units']) is str
        return dlabels

    def _checkformat_inputs_dtreat(self, dtreat=None):
        if dtreat is None:
            dtreat = {}
        assert type(dtreat) is dict
        lk0 = self._get_keys_dtreat()
        lk = dtreat.keys()
        for k in lk:
            assert k in lk0
        for k in lk0:
            if k not in lk:
                if k in self._ddef['dtreat'].keys():
                    dtreat[k] = self._ddef['dtreat'][k]
                else:
                    dtreat[k] = None
            if k == 'order':
                if dtreat[k] is None:
                    dtreat[k] = self.__class__._ddef['dtreat']['order']
                assert type(dtreat[k]) is list
                assert dtreat[k][-1] == 'interp-t'
                assert all([ss in dtreat[k][-4:-1]
                            for ss in ['indt','indch','indlamb']])
        return dtreat

    def _checkformat_inputs_dgeom(self, lCam=None, config=None):
        if config is not None:
            assert lCam is None
            nC = 0
        elif lCam is None:
            nC = 0
        else:
            if type(lCam) is not list:
                lCam = [lCam]
            nC = len(lCam)
            # Check type consistency
            lc = [cc._is2D() == self._is2D() for cc in lCam]
            if not all(lc):
                ls = ['%s : %s'%(cc.Id.Name,cc.Id.Cls) for cc in lCam]
                msg = "%s (%s) fed wrong lCam:\n"%(self.Id.Name,self.Id.Cls)
                msg += "    - " + "\n    - ".join(ls)
                raise Exception(msg)
            # Check config consistency
            lconf = [cc.config for cc in lCam]
            if not all([cc is not None for cc in lconf]):
                msg = "The provided Cams should have a config !"
                raise Exception(msg)
            config = [cc for cc in lconf if cc is not None][0].copy()

            # To be finished after modifying __eq__ in tf.utils
            lexcept = ['dvisible','dcompute','color']
            msg = "The following Cam do not have a consistent config:"
            flag = False
            for cc in lCam:
                if not cc.config.__eq__(config, lexcept=lexcept):
                    msg += "\n    {0}".format(cc.Id.Name)
                    flag = True
            if flag:
                raise Exception(msg)

            # Check number of channels wrt data
            nR = np.sum([cc._dgeom['nRays'] for cc in lCam])
            if not nR == self._ddataRef['nch']:
                msg = "Total nb. of rays from lCam != data.shape[1] !"
                raise Exception(msg)
        return config, lCam, nC

    def _checkformat_inputs_dchans(self, dchans=None):
        assert dchans is None or isinstance(dchans,dict)
        if dchans is None:
            dchans = {}
            if self._dgeom['lCam'] is not None:
                ldchans = [cc._dchans for cc in self._dgeom['lCam']]
                for k in ldchans[0].keys():
                    assert ldchans[0][k].ndim in [1,2]
                    if ldchans[0][k].ndim==1:
                        dchans[k] = np.concatenate([dd[k] for dd in ldchans])
                    else:
                        dchans[k] = np.concatenate([dd[k]
                                                    for dd in ldchans], axis=1)
        else:
            for k in dchans.keys():
                arr = np.asarray(dchans[k]).ravel()
                assert arr.size==self._ddata['nch']
                dchans[k] = arr
        return dchans

    def _checkformat_inputs_dextra(self, dextra=None):
        assert dextra is None or isinstance(dextra,dict)
        if dextra is not None:
            for k in dextra.keys():
                assert isinstance(dextra[k],dict)
                assert 't' in dextra[k].keys()
        return dextra

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_ddataRef():
        lk = ['data', 't', 'X', 'lamb', 'nt', 'nch', 'nlamb', 'nnch', 'nnlamb',
              'indtX', 'indtlamb', 'indXlamb', 'indtXlamb']
        return lk

    @staticmethod
    def _get_keys_ddata():
        lk = ['data', 't', 'X', 'lamb', 'nt', 'nch', 'nlamb', 'nnch', 'nnlamb',
              'indtX', 'indtlamb', 'indXlamb', 'indtXlamb', 'uptodate']
        return lk

    @staticmethod
    def _get_keys_dtreat():
        lk = ['order','mask-ind', 'mask-val', 'interp-indt', 'interp-indch',
              'data0-indt', 'data0-Dt', 'data0-data',
              'dfit', 'indt',  'indch', 'indlamb', 'interp-t']
        return lk

    @classmethod
    def _get_keys_dlabels(cls):
        lk = ['data','t','X']
        if cls._isSpectral():
            lk.append('lamb')
        return lk

    @staticmethod
    def _get_keys_dgeom():
        lk = ['config', 'lCam', 'nC']
        return lk

    @staticmethod
    def _get_keys_dX12():
        lk = ['from', 'x1','x2','n1', 'n2',
              'ind1', 'ind2', 'indr']
        return lk

    @staticmethod
    def _get_keys_dchans():
        lk = []
        return lk

    @staticmethod
    def _get_keys_dextra():
        lk = []
        return lk

    ###########
    # _init
    ###########

    def _init(self, data=None, t=None, X=None, lamb=None, dtreat=None, dchans=None,
              dlabels=None, dextra=None, lCam=None, config=None, **kwargs):
        kwdargs = locals()
        kwdargs.update(**kwargs)
        largs = self._get_largs_ddataRef()
        kwddataRef = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dtreat()
        kwdtreat = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dlabels()
        kwdlabels = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dgeom()
        kwdgeom = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dchans()
        kwdchans = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dextra()
        kwdextra = self._extract_kwdargs(kwdargs, largs)
        self._set_ddataRef(**kwddataRef)
        self.set_dtreat(**kwdtreat)
        self._set_ddata()
        self._set_dlabels(**kwdlabels)
        self._set_dgeom(**kwdgeom)
        if self._is2D():
            kwdX12 = self._extract_kwdargs(kwdargs, self._get_largs_dX12())
            self.set_dX12(**kwdX12)
        self.set_dchans(**kwdchans)
        self.set_dextra(**kwdextra)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_ddataRef(self, data=None, t=None,
                     X=None, indtX=None,
                     lamb=None, indtlamb=None, indXlamb=None, indtXlamb=None):
        kwdargs = locals()
        del kwdargs['self']
        lout = self._checkformat_inputs_ddataRef(**kwdargs)
        data, t, X, lamb, nt, nch, nlamb, nnch, nnlamb = lout[:9]
        indtX, indtlamb, indXlamb, indtXlamb = lout[9:]

        self._ddataRef = {'data':data, 't':t, 'X':X, 'lamb':lamb,
                          'nt':nt, 'nch':nch, 'nlamb':nlamb,
                          'nnch':nnch, 'nnlamb':nnlamb,
                          'indtX':indtX, 'indtlamb':indtlamb,
                          'indXlamb':indXlamb, 'indtXlamb':indtXlamb}

    def set_dtreat(self, dtreat=None):
        dtreat = self._checkformat_inputs_dtreat(dtreat=dtreat)
        self._dtreat = dtreat

    def _set_dlabels(self, dlabels=None):
        dlabels = self._checkformat_inputs_dlabels(dlabels=dlabels)
        self._dlabels.update(dlabels)

    def _set_dgeom(self, lCam=None, config=None):
        config, lCam, nC = self._checkformat_inputs_dgeom(lCam=lCam,
                                                          config=config)
        self._dgeom = {'lCam':lCam, 'nC':nC, 'config':config}

    def set_dchans(self, dchans=None, method='set'):
        """ Set (or update) the dchans dict

        dchans is a dict of np.ndarrays of len() = self.nch containing
        channel-specific information

        Use the kwarg 'method' to set / update the dict

        """
        assert method in ['set','update']
        dchans = self._checkformat_inputs_dchans(dchans=dchans)
        if method == 'set':
            self._dchans = dchans
        else:
            self._dchans.update(dchans)

    def set_dextra(self, dextra=None, method='set'):
        """ Set (or update) the dextra dict

        dextra is a dict of nested dict
        It contains all extra signal that can help interpret the data
            e.g.: heating power time traces, plasma current...
        Each nested dict should have the following fields:
            't'    : 1D np.ndarray (time vector)
            'data' : 1D np.ndarray (data time trace)
            'name' : str (used as label in legend)
            'units': str (used n parenthesis in legend after name)

        Use the kwarg 'method' to set / update the dict

        """
        assert method in ['set','update']
        dextra = self._checkformat_inputs_dextra(dextra=dextra)
        if method == 'set':
            self._dextra = dextra
        else:
            self._dextra.update(dextra)

    ###########
    # strip dictionaries
    ###########

    def _strip_ddata(self, strip=0):
        if self._dstrip['strip']==strip:
            return

        if strip in [0,1] and self._dstrip['strip'] in [2]:
            self._set_ddata()
        elif strip in [2] and self._dstrip['strip'] in [0,1]:
            self.clear_ddata()

    def _strip_dgeom(self, strip=0, force=False, verb=True):
        if self._dstrip['strip']==strip:
            return

        if strip in [0] and self._dstrip['strip'] in [1,2]:
            lC, config = None, None
            if self._dgeom['lCam'] is not None:
                assert type(self._dgeom['lCam']) is list
                assert all([type(ss) is str for ss in self._dgeom['lCam']])
                lC = []
                for ii in range(0,len(self._dgeom['lCam'])):
                    lC.append(utils.load(self._dgeom['lCam'][ii], verb=verb))

            elif self._dgeom['config'] is not None:
                assert type(self._dgeom['config']) is str
                config = utils.load(self._dgeom['config'], verb=verb)

            self._set_dgeom(lCam=lC, config=config)

        elif strip in [1,2] and self._dstrip['strip'] in [0]:
            if self._dgeom['lCam'] is not None:
                lpfe = []
                for cc in self._dgeom['lCam']:
                    path, name = cc.Id.SavePath, cc.Id.SaveName
                    pfe = os.path.join(path, name+'.npz')
                    lf = os.listdir(path)
                    lf = [ff for ff in lf if name+'.npz' in ff]
                    exist = len(lf)==1
                    if not exist:
                        msg = """BEWARE:
                            You are about to delete the lCam objects
                            Only the path/name to saved a object will be kept

                            But it appears that the following object has no
                            saved file where specified (obj.Id.SavePath)
                            Thus it won't be possible to retrieve it
                            (unless available in the current console:"""
                        msg += "\n    - {0}".format(pfe)
                        if force:
                            warning.warn(msg)
                        else:
                            raise Exception(msg)
                    lpfe.append(pfe)
                self._dgeom['lCam'] = lpfe
                self._dgeom['config'] = None

            elif self._dgeom['config'] is not None:
                path = self._dgeom['config'].Id.SavePath
                name = self._dgeom['config'].Id.SaveName
                pfe = os.path.join(path, name+'.npz')
                lf = os.listdir(path)
                lf = [ff for ff in lf if name+'.npz' in ff]
                exist = len(lf)==1
                if not exist:
                    msg = """BEWARE:
                        You are about to delete the config object
                        Only the path/name to saved a object will be kept

                        But it appears that the following object has no
                        saved file where specified (obj.Id.SavePath)
                        Thus it won't be possible to retrieve it
                        (unless available in the current console:"""
                    msg += "\n    - {0}".format(pfe)
                    if force:
                        warning.warn(msg)
                    else:
                        raise Exception(msg)
                self._dgeom['config'] = pfe


    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2,3]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: dgeom pathfiles
                 2: dgeom pathfiles + clear data
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(DataAbstract,self).strip(strip=strip, verb=verb)

    def _strip(self, strip=0, verb=True):
        self._strip_ddata(strip=strip)
        self._strip_dgeom(strip=strip, verb=verb)

    def _to_dict(self):
        dout = {'ddataRef':{'dict':self._ddataRef, 'lexcept':None},
                'ddata':{'dict':self._ddata, 'lexcept':None},
                'dtreat':{'dict':self._dtreat, 'lexcept':None},
                'dlabels':{'dict':self._dlabels, 'lexcept':None},
                'dgeom':{'dict':self._dgeom, 'lexcept':None},
                'dchans':{'dict':self._dchans, 'lexcept':None},
                'dextra':{'dict':self._dextra, 'lexcept':None}}
        if self._is2D():
            dout['dX12'] = {'dict':self._dX12, 'lexcept':None}
        return dout

    def _from_dict(self, fd):
        self._ddataRef.update(**fd['ddataRef'])
        self._ddata.update(**fd['ddata'])
        self._dtreat.update(**fd['dtreat'])
        self._dlabels.update(**fd['dlabels'])
        self._dgeom.update(**fd['dgeom'])
        self._dextra.update(**fd['dextra'])
        if 'dchans' not in fd.keys():
            fd['dchans'] = {}
        self._dchans.update(**fd['dchans'])
        if self._is2D():
            self._dX12.update(**fd['dX12'])


    ###########
    # properties
    ###########

    @property
    def ddataRef(self):
        return self._ddataRef
    @property
    def ddata(self):
        if not self._ddata['uptodate']:
            self._set_ddata()
        return self._ddata
    @property
    def dtreat(self):
        return self._dtreat
    @property
    def dlabels(self):
        return self._dlabels
    @property
    def dgeom(self):
        return self._dgeom
    @property
    def dextra(self):
        return self._dextra

    def get_ddata(self, key):
        if not self._ddata['uptodate']:
            self._set_ddata()
        return self._ddata[key]
    @property
    def data(self):
        return self.get_ddata('data')
    @property
    def t(self):
        return self.get_ddata('t')
    @property
    def X(self):
        return self.get_ddata('X')
    @property
    def nt(self):
        return self.get_ddata('nt')
    @property
    def nch(self):
        return self.get_ddata('nch')

    @property
    def config(self):
        return self._dgeom['config']
    @property
    def lCam(self):
        return self._dgeom['lCam']

    @property
    def _isLOS(self):
        c0 = self._dgeom['lCam'] is not None
        if c0:
            c0 = all([cc._isLOS() for cc in self.dgeom['lCam']])
        return c0

    @abstractmethod
    def _isSpectral(self):
        return 'spectral' in self.__class__.name.lower()
    @abstractmethod
    def _is2D(self):
        return '2d' in self.__class__.__name__.lower()

    ###########
    # Hidden and public methods for ddata
    ###########

    def set_XRef(self, X=None, indtX=None, indtXlamb=None):
        """ Reset the reference X

        Useful if to replace channel indices by a time-vraying quantity
        e.g.: distance to the magnetic axis

        """
        out = self._checkformat_inputs_XRef(X=X, indtX=indtX,
                                            indXlamb=indtXlamb)
        X, nnch, indtX, indXlamb, indtXlamb = out
        self._ddataRef['X'] = X
        self._ddataRef['nnch'] = nnch
        self._ddataRef['indtX'] = indtX
        self._ddataRef['indtXlamb'] = indtXlamb
        self._ddata['uptodate'] = False

    def set_dtreat_indt(self, t=None, indt=None):
        """ Store the desired index array for the time vector

        If an array of indices (refering to self.ddataRef['t'] is not provided,
        uses self.select_t(t=t) to produce it

        """
        lC = [indt is not None, t is not None]
        if all(lC):
            msg = "Please provide either t or indt (or none)!"
            raise Exception(msg)

        if lC[1]:
            ind = self.select_t(t=t, out=bool)
        else:
            ind = _format_ind(indt, n=self._ddataRef['nt'])
        self._dtreat['indt'] = ind
        self._ddata['uptodate'] = False

    def set_dtreat_indch(self, indch=None):
        """ Store the desired index array for the channels

        If None => all channels
        Must be a 1d array

        """
        if indch is not None:
            indch = np.asarray(indch)
            assert indch.ndim==1
        indch = _format_ind(indch, n=self._ddataRef['nch'])
        self._dtreat['indch'] = indch
        self._ddata['uptodate'] = False

    def set_dtreat_indlamb(self, indlamb=None):
        """ Store the desired index array for the wavelength

        If None => all wavelengths
        Must be a 1d array

        """
        if not self._isSpectral():
            msg = "The wavelength can only be set with DataSpectral object !"
            raise Exception(msg)
        if indlamb is not None:
            indlamb = np.asarray(indlamb)
            assert indlamb.ndim==1
            indlamb = _format_ind(indlamb, n=self._ddataRef['nlamb'])
        self._dtreat['indlamb'] = indlamb
        self._ddata['uptodate'] = False

    def set_dtreat_mask(self, ind=None, val=np.nan):
        assert ind is None or hasattr(ind,'__iter__')
        assert type(val) in [int,float,np.int64,np.float64]
        if ind is not None:
            ind = _format_ind(ind, n=self._ddataRef['nch'])
        self._dtreat['mask-ind'] = ind
        self._dtreat['mask-val'] = val
        self._ddata['uptodate'] = False

    def set_dtreat_data0(self, data0=None, Dt=None, indt=None):
        lC = [data0 is not None, Dt is not None, indt is not None]
        assert np.sum(lC) <= 1

        if any(lC):
            if lC[0]:
                data0 = np.asarray(data0)
                if self._isSpectral():
                    shape = (self._ddataRef['nch'],self._ddataRef['nlamb'])
                else:
                    shape = (self._ddataRef['nch'],)
                    data0 = data0.ravel()
                    if not data0.shape == shape:
                        msg = "Provided data0 has wrong shape !\n"
                        msg += "    - Expected: %s\n"%str(shape)
                        msg += "    - Provided: %s"%data0.shape
                        raise Exception(msg)
                Dt, indt = None, None
            else:
                if lC[2]:
                    indt = _format_ind(indt, n=self._ddataRef['nt'])
                else:
                    indt = self.select_t(t=Dt, out=bool)
                if np.any(indt):
                    if self._isSpectral():
                        data0 = self._ddataRef['data'][indt,:,:]
                    else:
                        data0 = self._ddataRef['data'][indt,:]
                    if np.sum(indt)>1:
                        data0 = np.nanmean(data0,axis=0)
        self._dtreat['data0-indt'] = indt
        self._dtreat['data0-Dt'] = Dt
        self._dtreat['data0-data'] = data0
        self._ddata['uptodate'] = False

    def set_dtreat_interp_indt(self, indt=None):
        """ Set the indices of the times for which to interpolate data

        The index can be provided as:
            - A 1d np.ndarray of boolean or int indices
                => interpolate data at these times for all channels
            - A dict with:
                * keys = int indices of channels
                * values = array of int indices of times at which to interpolate

        Time indices refer to self.ddataRef['t']
        Channel indices refer to self.ddataRef['X']
        """
        lC = [indt is None, type(indt) in [np.ndarray,list], type(indt) is dict]
        assert any(lC)
        if lC[2]:
            lc = [type(k) is int and k<self._ddataRef['nch'] for k in indt.keys()]
            assert all(lc)
            for k in indt.keys():
                assert hasattr(indt[k],'__iter__')
                indt[k] = _format_ind(indt[k], n=self._ddataRef['nt'])
        elif lC[1]:
            indt = np.asarray(indt)
            assert indt.ndim==1
            indt = _format_ind(indt, n=self._ddataRef['nt'])
        self._dtreat['interp-indt'] = indt
        self._ddata['uptodate'] = False

    def set_dtreat_interp_indch(self, indch=None):
        """ Set the indices of the channels for which to interpolate data

        The index can be provided as:
            - A 1d np.ndarray of boolean or int indices of channels
                => interpolate data at these channels for all times
            - A dict with:
                * keys = int indices of times
                * values = array of int indices of chan. for which to interpolate

        Time indices refer to self.ddataRef['t']
        Channel indices refer to self.ddataRef['X']
        """
        lC = [indch is None, type(indch) in [np.ndarray,list], type(indch) is dict]
        assert any(lC)
        if lC[2]:
            lc = [type(k) is int and k<self._ddataRef['nt'] for k in indch.keys()]
            assert all(lc)
            for k in indch.keys():
                assert hasattr(indch[k],'__iter__')
                indch[k] = _format_ind(indch[k], n=self._ddataRef['nch'])
        elif lC[1]:
            indch = np.asarray(indch)
            assert indch.ndim==1
            indch = _format_ind(indch, n=self._ddataRef['nch'])
        self._dtreat['interp-indch'] = indch
        self._ddata['uptodate'] = False

    def set_dtreat_dfit(self, dfit=None):
        """ Set the fitting dictionnary

        A dict contaning all parameters for fitting the data
        Valid dict content includes:
            - 'type': str
                'fft':  A fourier filtering
                'svd':  A svd filtering

        """
        warnings.warn("Not implemented yet !, dfit forced to None")
        dfit = None

        assert dfit is None or isinstance(dfit,dict)
        if isinstance(dfit,dict):
            assert 'type' in dfit.keys()
            assert dfit['type'] in ['svd','fft']

        self._dtreat['dfit'] = dfit
        self._ddata['uptodate'] = False

    def set_dtreat_interpt(self, t=None):
        """ Set the time vector on which to interpolate the data """
        if t is not None:
            t = np.unique(np.asarray(t, dtype=float).ravel())
        self._dtreat['interp-t'] = t

    @staticmethod
    def _mask(data, mask_ind, mask_val):
        if mask_ind is not None:
            if data.ndim==2:
                data[:,mask_ind] = mask_val
            elif data.ndim==3:
                data[:,mask_ind,:] = mask_val
        return data

    @staticmethod
    def _interp_indt(data, ind, t):
        msg = "interp not coded yet for 3d data !"
        assert data.ndim==2, msg
        if type(ind) is dict:
            for kk in ind.keys():
                data[ind[kk],kk] = np.interp(t[ind[kk]],
                                             t[~ind[kk]], data[~ind[kk],kk],
                                             right=np.nan, left=np.nan)
        elif isinstance(ind,np.ndarray):
            for ii in range(0,data.shape[1]):
                data[ind,ii] = np.interp(t[ind], t[~ind], data[~ind,ii])

        return data

    @staticmethod
    def _interp_indch(data, ind, X):
        msg = "interp not coded yet for 3d data !"
        assert data.ndim==2, msg
        if type(ind) is dict:
            for kk in ind.keys():
                data[kk,ind[kk]] = np.interp(X[ind[kk]],
                                             X[~ind[kk]], data[kk,~ind[kk]],
                                             right=np.nan, left=np.nan)
        elif isinstance(ind,np.ndarray):
            for ii in range(0,data.shape[0]):
                data[ii,ind] = np.interp(X[ind], X[~ind], data[ii,~ind])

        return data

    @staticmethod
    def _data0(data, data0):
        if data0 is not None:
            if data.shape == data0.shape:
                data = data - data0
            elif data.ndim == 2:
                data = data - data0[np.newaxis,:]
            if data.ndim == 3:
                data = data - data0[np.newaxis,:,:]
        return data

    @staticmethod
    def _dfit(data, dfit):
        if dfit is not None:
            if dfit['type']=='svd':
                #data = _comp.()
                pass
            elif dfit['type']=='svd':
                #data = _comp.()
                pass
        return data

    @staticmethod
    def _indt(data, t=None, X=None, nnch=None,
              indtX=None, indtlamb=None, indtXlamb=None, indt=None):
        nt0 = t.size
        if data.ndim==2:
            data = data[indt,:]
        elif data.ndim==3:
            data = data[indt,:,:]
        if t is not None:
            t = t[indt]
        if X is not None and X.ndim == 2 and X.shape[0] == nt0:
            X = X[indt,:]
            nnch = indt.sum()
        if indtX is not None:
            indtX = indtX[indt]
        if indtlamb is not None:
            indtlamb = indtlamb[indt]
        elif indtXlamb is not None:
            indtXlamb = indtXlamb[indt,:]
        return data, t, X, indtX, indtlamb, indtXlamb, nnch

    @staticmethod
    def _indch(data, X=None,
               indXlamb=None, indtXlamb=None, indch=None):
        if data.ndim==2:
            data = data[:,indch]
        elif data.ndim==3:
            data = data[:,indch,:]
        if X is not None:
            X = X[indch] if X.ndim==1 else X[:,indch]
        if indXlamb is not None:
            indXlamb = indXlamb[indch]
        elif indtXlamb is not None:
            indtXlamb = indtXlamb[:,indch]
        return data, X, indXlamb, indtXlamb

    @staticmethod
    def _indlamb(data, lamb=None,
                 indlamb=None):
        data = data[:,:,indlamb]
        if lamb is not None:
            lamb = lamb[indlamb] if lamb.ndim==1 else lamb[:,indlamb]
        return data, lamb

    @staticmethod
    def _interp_t(data, t, indtX=None,
                  indtlamb=None, indtXlamb=None, interpt=None, kind='linear'):
        f = scp.interp1d(t, data, kind=kind, axis=0, copy=True,
                         bounds_error=True, fill_value=np.nan, assume_sorted=False)
        d = f(data)

        lC = [indtX is not None, indtlamb is not None, indtXlamb is not None]
        if any(lC):
            indt = np.digitize(t, (interpt[1:]+interpt[:-1])/2.)
            if lC[0]:
                indtX = indtX[indt]
            if lC[1]:
                indtlamb = indtlamb[indt]
            elif lC[2]:
                indtXlamb = indtXlamb[indt,:]
        return d, interpt, indtX, indtlamb, indtXlamb




    def _get_ddata(self, key):
        if not self._ddata['uptodate']:
            self._set_ddata()
        return self._ddata[key]

    def set_dtreat_order(self, order=None):
        """ Set the order in which the data treatment should be performed

        Provide an ordered list of keywords indicating the order in which
         you wish the data treatment steps to be performed.
        Each keyword corresponds to a step.
        Available steps are (in default order):
            - 'mask' :
            - 'interp_indt' :
            - 'interp_indch' :
            - 'data0' :
            - 'dfit' :
            - 'indt' :
            - 'indch' :
            - 'interp_t':

        All steps are performed on the stored reference self.dataRef['data']
        Thus, the time and channels restriction must be the last 2 steps before
        interpolating on an external time vector
        """
        if order is None:
            order = list(self._ddef['dtreat']['order'])
        assert type(order) is list and all([type(ss) is str for ss in order])
        if not all([ss in ['indt','indch','indlamb'] for ss in order][-4:-1]):
            msg = "indt and indch must be the treatment steps -2 and -3 !"
            raise Exception(msg)
        if not order[-1]=='interp-t':
            msg = "interp-t must be the last treatment step !"
            raise Exception(msg)
        self._dtreat['order'] = order
        self._ddata['uptodate'] = False

    def _get_treated_data(self):
        """ Produce a working copy of the data based on the treated reference

        The reference data is always stored and untouched in self.ddataRef
        You always interact with self.data, which returns a working copy.
        That working copy is the reference data, eventually treated along the
            lines defined (by the user) in self.dtreat
        By reseting the treatment (self.reset()) all data treatment is
        cancelled and the working copy returns the reference data.

        """
        # --------------------
        # Copy reference data
        d = self._ddataRef['data'].copy()
        t, X = self._ddataRef['t'].copy(), self._ddataRef['X'].copy()
        lamb = self._ddataRef['lamb']
        if lamb is not None:
            lamb = lamb.copy()

        indtX = self._ddataRef['indtX']
        if indtX is not None:
            indtX = indtX.copy()
        indtlamb = self._ddataRef['indtlamb']
        if indtlamb is not None:
            indtlamb = indtlamb.copy()
        indXlamb = self._ddataRef['indXlamb']
        if indXlamb is not None:
            indXlamb = indXlamb.copy()
        indtXlamb = self._ddataRef['indtXlamb']
        if indtXlamb is not None:
            indtXlamb = indtXlamb.copy()

        nnch = self._ddataRef['nnch']
        # --------------------
        # Apply data treatment
        for kk in self._dtreat['order']:
            # data only
            if kk=='mask' and self._dtreat['mask-ind'] is not None:
                d = self._mask(d, self._dtreat['mask-ind'],
                               self._dtreat['mask-val'])
            if kk=='interp_indt':
                d = self._interp_indt(d, self._dtreat['interp-indt'],
                                      self._ddataRef['t'])
            if kk=='interp_indch':
                d = self._interp_indch(d, self._dtreat['interp-indch'],
                                       self._ddataRef['X'])
            if kk=='data0':
                d = self._data0(d, self._dtreat['data0-data'])
            if kk=='dfit' and self._dtreat['dfit'] is not None:
                d = self._dfit(d, **self._dtreat['dfit'])

            # data + others
            if kk=='indt' and self._dtreat['indt'] is not None:
                d,t,X, indtX,indtlamb,indtXlamb, nnch = self._indt(d, t, X,
                                                                   nnch, indtX,
                                                                   indtlamb, indtXlamb,
                                                                   self._dtreat['indt'])
            if kk=='indch' and self._dtreat['indch'] is not None:
                d,X, indXlamb,indtXlamb = self._indch(d, X, indXlamb, indtXlamb,
                                                      self._dtreat['indch'])
            if kk=='indlamb' and self._dtreat['indlamb'] is not None:
                d, lamb = self._indch(d, lamb, self._dtreat['indlamb'])
            if kk=='interp_t' and self._dtreat['interp-t'] is not None:
                d,t, indtX,indtlamb,indtXlamb\
                        = self._interp_t(d, t, indtX, indtlamb, indtXlamb,
                                         self._dtreat['interp-t'], kind='linear')
        # --------------------
        # Safety check
        if d.ndim==2:
            (nt, nch), nlamb = d.shape, 0
        else:
            nt, nch, nlamb = d.shape
        assert d.ndim in [2,3]
        assert t.shape==(nt,)
        assert X.shape==(nnch, nch)
        if lamb is not None:
            assert lamb.shape==(self._ddataRef['nnlamb'], nlamb)

        lout = [d, t, X, lamb, nt, nch, nlamb,
                indtX, indtlamb, indXlamb, indtXlamb, nnch]
        return lout

    def _set_ddata(self):
        if not self._ddata['uptodate']:
            data, t, X, lamb, nt, nch, nlamb,\
                    indtX, indtlamb, indXlamb, indtXlamb,\
                    nnch = self._get_treated_data()
            self._ddata['data'] = data
            self._ddata['t'] = t
            self._ddata['X'] = X
            self._ddata['lamb'] = lamb
            self._ddata['nt'] = nt
            self._ddata['nch'] = nch
            self._ddata['nlamb'] = nlamb
            self._ddata['nnch'] = nnch
            self._ddata['nnlamb'] = self._ddataRef['nnlamb']
            self._ddata['indtX'] = indtX
            self._ddata['indtlamb'] = indtlamb
            self._ddata['indXlamb'] = indXlamb
            self._ddata['indtXlamb'] = indtXlamb
            self._ddata['uptodate'] = True


    def clear_ddata(self):
        """ Clear the working copy of data

        Harmless, as it preserves the reference copy and the treatment dict
        Use only to free some memory

        """
        self._ddata = dict.fromkeys(self._get_keys_ddata())
        self._ddata['uptodate'] = False

    def clear_dtreat(self, force=False):
        """ Clear all treatment parameters in self.dtreat

        Subsequently also clear the working copy of data
        The working copy of data is thus reset to the reference data
        """
        lC = [self._dtreat[k] is not None for k in self._dtreat.keys()
              if k != 'order']
        if any(lC) and not force:
            msg = """BEWARE : You are about to delete the data treatment
                              i.e.: to clear self.dtreat (and also self.ddata)
                              Are you sure ?
                              If yes, use self.clear_dtreat(force=True)"""
            raise Exception(msg)
        dtreat = dict.fromkeys(self._get_keys_dtreat())
        self._dtreat = self._checkformat_inputs_dtreat(dtreat)
        self.clear_ddata()

    def dchans(self, key=None):
        """ Return the dchans updated with indch

        Return a dict with all keys if key=None

        """
        if self._dtreat['indch'] is None or np.all(self._dtreat['indch']):
            dch = dict(self._dchans) if key is None else self._dchans[key]
        else:
            dch = {}
            lk = self._dchans.keys() if key is None else [key]
            for kk in lk:
                if self._dchans[kk].ndim==1:
                    dch[kk] = self._dchans[kk][self._dtreat['indch']]
                elif self._dchans[kk].ndim==2:
                    dch[kk] = self._dchans[kk][:,self._dtreat['indch']]
                else:
                    msg = "Don't know how to treat self._dchans[%s]:"%kk
                    msg += "\n  shape = %s"%(kk,str(self._dchans[kk].shape))
                    warnings.warn(msg)
            if key is not None:
                dch = dch[key]
        return dch


    ###########
    # Other public methods
    ###########

    def select_t(self, t=None, out=bool):
        """ Return a time index array

        Return a boolean or integer index array, hereafter called 'ind'
        The array refers to the reference time vector self.ddataRef['t']

        Parameters
        ----------
        t :     None / float / np.ndarray / list / tuple
            The time values to be selected:
                - None : ind matches all time values
                - float : ind is True only for the time closest to t
                - np.ndarray : ind is True only for the times closest to t
                - list (len()==2): ind is True for the times inside [t[0],t[1]]
                - tuple (len()==2): ind is True for times outside ]t[0];t[1][
        out :   type
            Specifies the type of the output index array:
                - bool : return a boolean array of shape (self.ddataRef['nt'],)
                - int : return the array as integers indices

        Return
        ------
        ind :   np.ndarray
            The array of indices, of dtype specified by keywordarg out

        """
        assert out in [bool,int]
        ind = _select_ind(t, self._ddataRef['t'], self._ddataRef['nt'])
        if out is int:
            ind = ind.nonzero()[0]
        return ind


    def select_ch(self, val=None, key=None, log='any', touch=None, out=bool):
        """ Return a channels index array

        Return a boolean or integer index array, hereafter called 'ind'
        The array refers to the reference channel/'X' vector self.ddataRef['X']

        There are 3 different ways of selecting channels, by refering to:
            - The 'X' vector/array values in self.dataRef['X']
            - The dict of channels keys/values (if self.dchans != None)
            - which element each LOS touches (if self.lCam != None)

        Parameters
        ----------
        val :   None / str / float / np.array / list / tuple
            The value against which to dicriminate.
            Behaviour depends whether key is provided:
                - key is None => val compares vs self.ddataRef['X']
                - key provided => val compares vs self.dchans[key]
            If key is None, the behaviour is similar to self.select_indt():
                - None : ind matches all channels
                - float : ind is True only for X closest to val
                - np.ndarray : ind is True only for X closest to val
                - list (len()==2): ind is True for X inside [val[0],val[1]]
                - tuple (len()==2): ind is True for X outside ]val[0];val[1][

        key :   None / str
            If provided, dict key to indicate which self.dchans[key] to use

        log :   str
            If key provided, val can be a list of criteria
            Then, log indicates whether all / any / none should be matched

        touch : None
            If key and val are None, return the indices of the LOS touching the
            elements indicated in touch.
            Requires that self.dgeom['lCam'] is not None (tf.geom.Cam.select())

        out :   type
            Specifies the type of the output index array:
                - bool : return a boolean array of shape (self.ddataRef['nt'],)
                - int : return the array as integers indices

        Return
        ------
        ind :   np.ndarray
            The array of indices, of dtype specified by keywordarg out

        """
        assert out in [int,bool]
        assert log in ['any','all','not']
        lc = [val is None, key is None, touch is None]
        lC = [all(lc), all(lc[:2]) and not lc[2],
              not lc[0] and all(lc[1:]), not any(lc[:2]) and lc[2]]
        assert np.sum(lC)==1

        if lC[0]:
            # get all channels
            ind = np.ones((self._ddataRef['nch'],),dtype=bool)

        elif lC[1]:
            # get touch
            if self._dgeom['lCam'] is None:
                msg = "self.dgeom['lCam'] must be set to use touch !"
                raise Exception(msg)
            if any([type(cc) is str for cc in self._dgeom['lCam']]):
                msg = "self.dgeom['lCam'] contains pathfiles !"
                msg += "\n  => Run self.strip(0)"
                raise Exception(msg)
            ind = []
            for cc in self._dgeom['lCam']:
                ind.append(cc.select(touch=touch, log=log, out=bool))
            if len(ind)==1:
                ind = ind[0]
            else:
                ind = np.concatenate(tuple(ind))

        elif lC[2]:
            # get values on X
            if self._ddataRef['nnch']==1:
                ind = _select_ind(val, self._ddataRef['X'], self._ddataRef['nch'])
            else:
                ind = np.zeros((self._ddataRef['nt'],self._ddataRef['nch']),dtype=bool)
                for ii in range(0,self._ddataRef['nnch']):
                    iind = self._ddataRef['indtX']==ii
                    ind[iind,:] =  _select_ind(val, self._ddataRef['X'],
                                               self._ddataRef['nch'])[np.newaxis,:]

        else:
            if not (type(key) is str and key in self._dchans.keys()):
                msg = "Provided key not valid!\n"
                msg += "    - key: %s\n"%str(key)
                msg += "Please provide a valid key of self.dchans():\n"
                msg += "    - " + "\n    - ".join(self._dchans.keys())
                raise Exception(msg)

            ltypes = [str,int,float,np.int64,np.float64]
            C0 = type(val) in ltypes
            C1 = type(val) in [list,tuple,np.ndarray]
            assert C0 or C1
            if C0:
                val = [val]
            else:
                assert all([type(vv) in ltypes for vv in val])
            ind = np.vstack([self._dchans[key]==ii for ii in val])
            if log=='any':
                ind = np.any(ind,axis=0)
            elif log=='all':
                ind = np.all(ind,axis=0)
            else:
                ind = ~np.any(ind,axis=0)
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def select_lamb(self, lamb=None, out=bool):
        """ Return a wavelength index array

        Return a boolean or integer index array, hereafter called 'ind'
        The array refers to the reference time vector self.ddataRef['lamb']

        Parameters
        ----------
        lamb :     None / float / np.ndarray / list / tuple
            The time values to be selected:
                - None : ind matches all wavelength values
                - float : ind is True only for the wavelength closest to lamb
                - np.ndarray : ind True only for the wavelength closest to lamb
                - list (len()==2): ind True for wavelength in [lamb[0],lamb[1]]
                - tuple (len()==2): ind True for wavelength outside ]t[0];t[1][
        out :   type
            Specifies the type of the output index array:
                - bool : return a boolean array of shape (self.ddataRef['nlamb'],)
                - int : return the array as integers indices

        Return
        ------
        ind :   np.ndarray
            The array of indices, of dtype specified by keywordarg out

        """
        if not self._isSpectral():
            msg = ""
            raise Exception(msg)
        assert out in [bool,int]
        ind = _select_ind(lamb, self._ddataRef['lamb'], self._ddataRef['nlamb'])
        if out is int:
            ind = ind.nonzero()[0]
        return ind



    def plot(self, key=None,
             cmap=None, ms=4, vmin=None, vmax=None,
             vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
             ntMax=None, nchMax=None, nlbdMax=3,
             lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
             inct=[1,10], incX=[1,5], inclbd=[1,10],
             fmt_t='06.3f', fmt_X='01.0f',
             invert=True, Lplot='In', dmarker=None,
             Bck=True, fs=None, dmargin=None, wintit=None, tit=None,
             fontsize=None, labelpad=None, draw=True, connect=True):
        """ Plot the data content in a generic interactive figure  """
        kh = _plot.Data_plot(self, key=key, indref=0,
                             cmap=cmap, ms=ms, vmin=vmin, vmax=vmax,
                             vmin_map=vmin_map, vmax_map=vmax_map,
                             cmap_map=cmap_map, normt_map=normt_map,
                             ntMax=ntMax, nchMax=nchMax, nlbdMax=nlbdMax,
                             lls=lls, lct=lct, lcch=lcch, lclbd=lclbd, cbck=cbck,
                             inct=inct, incX=incX, inclbd=inclbd,
                             fmt_t=fmt_t, fmt_X=fmt_X, Lplot=Lplot,
                             invert=invert, dmarker=dmarker, Bck=Bck,
                             fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                             fontsize=fontsize, labelpad=labelpad,
                             draw=draw, connect=connect)
        return kh

    def plot_compare(self, lD, key=None,
                     cmap=None, ms=4, vmin=None, vmax=None,
                     vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
                     ntMax=None, nchMax=None, nlbdMax=3,
                     lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
                     inct=[1,10], incX=[1,5], inclbd=[1,10],
                     fmt_t='06.3f', fmt_X='01.0f', fmt_l='07.3f',
                     invert=True, Lplot='In', dmarker=None,
                     sharey=True, sharelamb=True,
                     Bck=True, fs=None, dmargin=None, wintit=None, tit=None,
                     fontsize=None, labelpad=None, draw=True, connect=True):
        """ Plot several Data instances of the same diag

        Useful to compare :
                - the diag data for 2 different shots
                - experimental vs synthetic data for the same shot

        """
        C0 = isinstance(lD,list)
        C0 = C0 and all([issubclass(dd.__class__,DataAbstract) for dd in lD])
        C1 = issubclass(lD.__class__,DataAbstract)
        assert C0 or C1, 'Provided first arg. must be a tf.data.DataAbstract or list !'
        lD = [lD] if C1 else lD
        kh = _plot.Data_plot([self]+lD, key=key, indref=0,
                             cmap=cmap, ms=ms, vmin=vmin, vmax=vmax,
                             vmin_map=vmin_map, vmax_map=vmax_map,
                             cmap_map=cmap_map, normt_map=normt_map,
                             ntMax=ntMax, nchMax=nchMax, nlbdMax=nlbdMax,
                             lls=lls, lct=lct, lcch=lcch, lclbd=lclbd, cbck=cbck,
                             inct=inct, incX=incX, inclbd=inclbd,
                             fmt_t=fmt_t, fmt_X=fmt_X, fmt_l=fmt_l, Lplot=Lplot,
                             invert=invert, dmarker=dmarker, Bck=Bck,
                             sharey=sharey, sharelamb=sharelamb,
                             fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                             fontsize=fontsize, labelpad=labelpad,
                             draw=draw, connect=connect)
        return kh

    def plot_combine(self, lD, key=None, Bck=True, indref=0,
                  cmap=None, ms=4, vmin=None, vmax=None,
                  vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
                  ntMax=None, nchMax=None, nlbdMax=3,
                  inct=[1,10], incX=[1,5], inclbd=[1,10],
                  lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
                  fmt_t='06.3f', fmt_X='01.0f',
                  invert=True, Lplot='In', dmarker=None,
                  fs=None, dmargin=None, wintit=None, tit=None,
                  fontsize=None, labelpad=None, draw=True, connect=True):
        """ Plot several Data instances of different diags

        Useful to visualize several diags for the same shot

        """
        C0 = isinstance(lD,list)
        C0 = C0 and all([issubclass(dd.__class__,DataAbstract) for dd in lD])
        C1 = issubclass(lD.__class__,DataAbstract)
        assert C0 or C1, 'Provided first arg. must be a tf.data.DataAbstract or list !'
        lD = [lD] if C1 else lD
        kh = _plot.Data_plot_combine([self]+lD, key=key, Bck=Bck,
                                     indref=indref, cmap=cmap, ms=ms,
                                     vmin=vmin, vmax=vmax,
                                     vmin_map=vmin_map, vmax_map=vmax_map,
                                     cmap_map=cmap_map, normt_map=normt_map,
                                     ntMax=ntMax, nchMax=nchMax,
                                     inct=inct, incX=incX,
                                     lls=lls, lct=lct, lcch=lcch, cbck=cbck,
                                     fmt_t=fmt_t, fmt_X=fmt_X,
                                     invert=invert, Lplot=Lplot,
                                     dmarker=dmarker, fs=fs, dmargin=dmargin,
                                     wintit=wintit, tit=tit, fontsize=fontsize,
                                     labelpad=labelpad, draw=draw,
                                     connect=connect)
        return kh


    def calc_spectrogram(self, fmin=None,
                         method='scipy-fourier', deg=False,
                         window='hann', detrend='linear',
                         nperseg=None, noverlap=None,
                         boundary='constant', padded=True,
                         wave='morlet', warn=True):
        """ Return the power spectrum density for each channel

        The power spectrum density is computed with the chosen method

        Parameters
        ----------
        fmin :  None / float
            The minimum frequency of interest
            If None, set to 5/T, where T is the whole time interval
            Used to constrain the number of points per window
        deg :   bool
            Flag indicating whether to return the phase in deg (vs rad)
        method : str
            Flag indicating which method to use for computation:
                - 'scipy-fourier':  uses scipy.signal.spectrogram()
                    (windowed fast fourier transform)
                - 'scipy-stft':     uses scipy.signal.stft()
                    (short time fourier transform)
                - 'scipy-wavelet':  uses scipy.signal.cwt()
                    (continuous wavelet transform)
            The following keyword args are fed to one of these scipy functions
            See the corresponding online scipy documentation for details on
            each function and its arguments
        window : None / str / tuple
            If method='scipy-fourier'
            Flag indicating which type of window to use
        detrend : None / str
            If method='scipy-fourier'
            Flag indicating whether and how to remove the trend of the signal
        nperseg :   None / int
            If method='scipy-fourier'
            Number of points to the used for each window
            If None, deduced from fmin
        noverlap:
            If method='scipy-fourier'
            Number of points on which successive windows should overlap
            If None, nperseg-1
        boundary:
            If method='scipy-stft'

        padded :
            If method='scipy-stft'
            d
        wave: None / str
            If method='scipy-wavelet'

        Return
        ------
        tf :    np.ndarray
            Time vector of the spectrogram (1D)
        f:      np.ndarray
            frequency vector of the spectrogram (1D)
        lspect: list of np.ndarrays
            list of () spectrograms

        """
        if self._isSpectral():
            msg = "spectrogram not implemented yet for spectral data class"
            raise Exception(msg)
        tf, f, lpsd, lang = _comp.spectrogram(self.data, self.t,
                                              fmin=fmin, deg=deg,
                                              method=method, window=window,
                                              detrend=detrend, nperseg=nperseg,
                                              noverlap=noverlap, boundary=boundary,
                                              padded=padded, wave=wave,
                                              warn=warn)
        return tf, f, lpsd, lang

    def plot_spectrogram(self, fmin=None, fmax=None,
                         method='scipy-fourier', deg=False,
                         window='hann', detrend='linear',
                         nperseg=None, noverlap=None,
                         boundary='constant', padded=True, wave='morlet',
                         invert=True, plotmethod='imshow',
                         cmap_f=None, cmap_img=None,
                         ms=4, ntMax=None, nfMax=None,
                         Bck=True, fs=None, dmargin=None, wintit=None,
                         tit=None, vmin=None, vmax=None, normt=False,
                         draw=True, connect=True, returnspect=False, warn=True):
        """ Plot the spectrogram of all channels with chosen method

        All non-plotting arguments are fed to self.calc_spectrogram()
        see self.calc_spectrogram? for details

        Parameters
        ----------

        Return
        ------
        kh :    tofu.utils.HeyHandler
            The tofu KeyHandler object handling figure interactivity
        """
        if self._isSpectral():
            msg = "spectrogram not implemented yet for spectral data class"
            raise Exception(msg)
        tf, f, lpsd, lang = _comp.spectrogram(self.data, self.t,
                                              fmin=fmin, deg=deg,
                                              method=method, window=window,
                                              detrend=detrend, nperseg=nperseg,
                                              noverlap=noverlap, boundary=boundary,
                                              padded=padded, wave=wave,
                                              warn=warn)
        kh = _plot.Data_plot_spectrogram(self, tf, f, lpsd, lang, fmax=fmax,
                                         invert=invert, plotmethod=plotmethod,
                                         cmap_f=cmap_f, cmap_img=cmap_img,
                                         ms=ms, ntMax=ntMax,
                                         nfMax=nfMax, Bck=Bck, fs=fs,
                                         dmargin=dmargin, wintit=wintit,
                                         tit=tit, vmin=vmin, vmax=vmax,
                                         normt=normt, draw=draw,
                                         connect=connect)
        if returnspect:
            return kh, tf, f, lpsd, lang
        else:
            return kh

    def calc_svd(self, lapack_driver='gesdd'):
        """ Return the SVD decomposition of data

        The input data np.ndarray shall be of dimension 2,
            with time as the first dimension, and the channels in the second
            Hence data should be of shape (nt, nch)

        Uses scipy.linalg.svd(), with:
            full_matrices = True
            compute_uv = True
            overwrite_a = False
            check_finite = True

        See scipy online doc for details

        Return
        ------
        chronos:    np.ndarray
            First arg (u) returned by scipy.linalg.svd()
            Contains the so-called 'chronos', of shape (nt, nt)
                i.e.: the time-dependent part of the decoposition
        s:          np.ndarray
            Second arg (s) returned by scipy.linalg.svd()
            Contains the singular values, of shape (nch,)
                i.e.: the channel-dependent part of the decoposition
        topos:      np.ndarray
            Third arg (v) returned by scipy.linalg.svd()
            Contains the so-called 'topos', of shape (nch, nch)
                i.e.: the channel-dependent part of the decoposition

        """
        if self._isSpectral():
            msg = "svd not implemented yet for spectral data class"
            raise Exception(msg)
        chronos, s, topos = _comp.calc_svd(self.data, lapack_driver=lapack_driver)
        return u, s, v

    def plot_svd(self, lapack_driver='gesdd', modes=None, key=None, Bck=True,
                 Lplot='In', cmap=None, vmin=None, vmax=None,
                 cmap_topos=None, vmin_topos=None, vmax_topos=None,
                 ntMax=None, nchMax=None, ms=4,
                 inct=[1,10], incX=[1,5], incm=[1,5],
                 lls=None, lct=None, lcch=None, lcm=None, cbck=None,
                 invert=False, fmt_t='06.3f', fmt_X='01.0f', fmt_m='03.0f',
                 fs=None, dmargin=None, labelpad=None, wintit=None, tit=None,
                 fontsize=None, draw=True, connect=True):
        """ Plot the chosen modes of the svd decomposition

        All modes will be plotted, the keyword 'modes' is only used to
        determine the reference modes for computing a common scale for
        vizualisation

        Runs self.calc_svd() and then plots the result in an interactive figure

        """
        if self._isSpectral():
            msg = "svd not implemented yet for spectral data class"
            raise Exception(msg)
        # Computing (~0.2 s for 50 channels 1D and 1000 times)
        chronos, s, topos = _comp.calc_svd(self.data, lapack_driver=lapack_driver)

        # Plotting (~11 s for 50 channels 1D and 1000 times)
        kh = _plot.Data_plot_svd(self, chronos, s, topos, modes=modes,
                                 key=key, Bck=Bck, Lplot=Lplot,
                                 cmap=cmap, vmin=vmin, vmax=vmax,
                                 cmap_topos=cmap_topos, vmin_topos=vmin_topos,
                                 vmax_topos=vmax_topos,
                                 ntMax=ntMax, nchMax=nchMax, ms=ms,
                                 inct=inct, incX=incX, incm=incm,
                                 lls=lls, lct=lct, lcch=lcch, lcm=lcm, cbck=cbck,
                                 invert=invert, fmt_t=fmt_t, fmt_X=fmt_X, fmt_m=fmt_m,
                                 fs=fs, dmargin=dmargin, labelpad=labelpad, wintit=wintit,
                                 tit=tit, fontsize=fontsize, draw=draw,
                                 connect=connect)
        return kh

    def save(self, path=None, name=None,
             strip=None, deep=False, mode='npz',
             compressed=False, verb=True, return_pfe=False):
        if deep is False:
            self.strip(1)
        out = super(DataAbstract, self).save(path=path, name=name,
                                             deep=deep, mode=mode,
                                             strip=strip, compressed=compressed,
                                             return_pfe=return_pfe, verb=verb)
        return out
















############################################ To be finished


"""
    def _get_LCam(self):
        if self.geom is None or self.geom['LCam'] is None:
            lC = None
        else:
            if np.all(self.indch):
                lC = self.geom['LCam']
            else:
                import tofu.geom as tfg
                inds = [self.geom['LCam'][ii].nRays
                        for ii in range(len(self.geom['LCam']-1))]
                lind = self.indch.split(inds)
                lC = [cc.get_subset(indch=iind) for iind in lind]
        return lC



    def __abs__(self):
        opfunc = lambda x: np.abs(x)
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __sub__(self, other):
        opfunc = lambda x, y: x-y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __rsub__(self, other):
        opfunc = lambda x, y: x-y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __add__(self, other):
        opfunc = lambda x, y: x+y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __radd__(self, other):
        opfunc = lambda x, y: x+y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __mul__(self, other):
        opfunc = lambda x, y: x*y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __rmul__(self, other):
        opfunc = lambda x, y: x*y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __truediv__(self, other):
        opfunc = lambda x, y: x/y
        data = _recreatefromoperator(self, other, opfunc)
        return data

    def __pow__(self, other):
        opfunc = lambda x, y: x**y
        data = _recreatefromoperator(self, other, opfunc)
        return data









def _compare_(ls, null=None):
    ind = np.nonzero([ss is not null for ss in ls])[0]
    if ind.size>0:
        if all([ls[ind[0]]==ls[ind[ii]] for ii in range(1,len(ind))]):
            s = ls[ind[0]]
        else:
            s = null
    else:
        s = null
    return s

def _compare_dchans(ldch):
    ind = np.nonzero([dd is not None for dd in ldch])[0]
    if ind.size>0:
        All = True
        dch = ldch[ind[0]]
        for ii in range(1,len(ind)):
            if any([not kk in ldch[ind[ii]].keys() for kk in dch.keys()]):
                All = False
                break
            if any([not kk in dch.keys() for kk in ldch[ind[ii]].keys()]):
                All = False
                break
            for kk in dch.keys():
                if not dch[kk].shape==ldch[ind[ii]][kk].shape:
                    All = False
                    break
                if not dch[kk].dtype==ldch[ind[ii]][kk].dtype:
                    All = False
                    break
                C = all([dch[kk][jj]==ldch[ind[ii]][kk][jj]
                         for jj in range(len(dch[kk]))])
                if not C:
                    All = False
                    break
            if All is False:
                break

        if All is False:
            dch = None
    else:
        dch = None
    return dch


def _compare_lCam(ld, atol=1.e-12):
    lLC = [dd.geom['LCam'] for dd in ld]
    ind = np.nonzero([lc is not None for lc in lLC])[0]
    lC = None
    if ind.size>0:
        All = True
        lD = [np.concatenate(tuple([cc.D for cc in lLC[ind[ii]]]),axis=1)
              for ii in ind]
        lu = [np.concatenate(tuple([cc.u for cc in lLC[ind[ii]]]),axis=1)
              for ii in ind]
        for ii in range(1,len(ind)):
            CD = np.any(~np.isclose(lD[0],lD[ii],
                                    atol=atol, rtol=0., equal_nan=True))
            Cu = np.any(~np.isclose(lu[0],lu[ii],
                                    atol=atol, rtol=0., equal_nan=True))
            Cind = not np.all(ld[ind[0]].indch==ld[ind[ii]].indch)
            if CD or Cu or Cind:
                All = False
                break
        if All:
            lC = ld[ind[0]]._get_LCam()
    return lC



def _extractCommonParams(ld):

    # data size
    lnt, lnch = np.array([(dd.data.shape[0],dd.Ref['data'].shape[1]) for dd in ld]).T
    assert all([lnt[0]==nt for nt in lnt[1:]]), "Different data.shape[0] !"
    assert all([lnch[0]==nch for nch in lnch[1:]]), "Different data.shape[1] !"

    # Time vector
    lt = [dd.t for dd in ld]
    ind = np.nonzero([tt is not None for tt in lt])[0]
    if ind.size>0:
        if all([np.allclose(lt[ind[ii]],lt[ind[0]]) for ii in range(len(ind))]):
            t = lt[ind[0]]
        else:
            warnings.warn("\n Beware : the time vectors seem to differ !")
            t = None
    else:
        t = None

    # Channels
    indch = np.vstack([dd.indch for dd in ld])
    assert np.all(np.all(indch,axis=0) | np.all(~indch,axis=0)), "Different indch !"
    LCam = _compare_lCam(ld)

    if LCam is None:
        dchans = _compare_dchans([dd.dchans() for dd in ld])
    else:
        dchans = None

    # dlabels, Id, Exp, shot, Diag, SavePath
    dlabels = _compare_([dd._dlabels for dd in ld], null={})
    Id = ' '.join([dd.Id.Name for dd in ld])
    Exp = _compare_([dd.Id.Exp for dd in ld])
    shot = _compare_([dd.Id.shot for dd in ld])
    Diag = _compare_([dd.Id.Diag for dd in ld])
    SavePath = _compare_([dd.Id.SavePath for dd in ld])

    return t, LCam, dchans, dlabels, Id, Exp, shot, Diag, SavePath




def _recreatefromoperator(d0, other, opfunc):
    if type(other) in [int,float,np.int64,np.float64]:
        d = opfunc(d0.data, other)

        #  Fix LCam and dchans
        #t, LCam, dchans = d0.t, d0._get_LCam(), d0.dchans(d0.indch)
        out = _extractCommonParams([d0, d0])
        t, LCam, dchans, dlabels, Id, Exp, shot, Diag, SavePath = out

        #dlabels = d0._dlabels
        #Id, Exp, shot = d0.Id.Name, d0.Id.Exp, d0.Id.shot
        #Diag, SavePath = d0.Id.Diag, d0.Id.SavePath
    elif issubclass(other.__class__, Data):
        assert other.__class__==d0.__class__, 'Same class is expected !'
        try:
            d = opfunc(d0.data, other.data)
        except Exception as err:
            print("\n data shapes not matching !")
            raise err
        out = _extractCommonParams([d0, other])
        t, LCam, dchans, dlabels, Id, Exp, shot, Diag, SavePath = out
    else:
        raise NotImplementedError

    kwdargs = dict(t=t, dchans=dchans, LCam=LCam, dlabels=dlabels,
                   Id=Id, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath)

    if '1D' in d0.Id.Cls:
        data = Data1D(d, **kwdargs)
    elif '2D' in d0.Id.Cls:
        data = Data2D(d, **kwdargs)
    else:
        data = Data(d, **kwdargs)
    return data

"""





#####################################################################
#               Data1D and Data2D
#####################################################################

sig = inspect.signature(DataAbstract)
params = sig.parameters


class DataCam1D(DataAbstract):
    """ Data object used for 1D cameras or list of 1D cameras  """
    @classmethod
    def _isSpectral(cls):  return False
    @classmethod
    def _is2D(cls):        return False

lp = [p for p in params.values() if p.name not in ['lamb','dX12']]
DataCam1D.__signature__ = sig.replace(parameters=lp)

class DataCam1DSpectral(DataCam1D):
    """ Data object used for 1D cameras or list of 1D cameras  """
    @classmethod
    def _isSpectral(cls):  return True

    @property
    def lamb(self):
        return self.get_ddata('lamb')
    @property
    def nlamb(self):
        return self.get_ddata('nlamb')

lp = [p for p in params.values() if p.name not in ['dX12']]
DataCam1D.__signature__ = sig.replace(parameters=lp)


class DataCam2D(DataAbstract):
    """ Data object used for 2D cameras or list of 2D cameras  """

    @classmethod
    def _isSpectral(cls):  return False
    @classmethod
    def _is2D(cls):        return True

    def _checkformat_dX12(self, dX12=None):
        lc = [dX12 is None, dX12 == 'geom' or dX12 == {'from':'geom'},
              isinstance(dX12, dict) and dX12 != {'from':'geom'}]
        if not np.sum(lc) == 1:
            msg = "dX12 must be either:\n"
            msg += "    - None\n"
            msg += "    - 'geom' : will be derived from the cam geometry\n"
            msg += "    - dict : containing {'x1'  : array of coords.,\n"
            msg += "                         'x2'  : array of coords.,\n"
            msg += "                         'ind1': array of int indices,\n"
            msg += "                         'ind2': array of int indices}"
            raise Exception(msg)

        if lc[1]:
            ls = self._get_keys_dX12()
            c0 = self._dgeom['lCam'] is not None
            c1 = c0 and len(self._dgeom['lCam']) == 1
            c2 = c1 and self._dgeom['lCam'][0].dX12 is not None
            if not c2:
                msg = "dX12 cannot be derived from dgeom['lCam'][0].dX12 !"
                raise Exception(msg)
            dX12 = {'from':'geom'}

        elif lc[2]:
            ls = ['x1','x2','ind1','ind2']
            assert all([ss in dX12.keys() for ss in ls])
            x1 = np.asarray(dX12['x1']).ravel()
            x2 = np.asarray(dX12['x2']).ravel()
            n1, n2 = x1.size, x2.size
            ind1, ind2, indr = self._get_ind12r_n12(ind1=dX12['ind1'],
                                                    ind2=dX12['ind2'],
                                                    n1=n1, n2=n2)
            dX12 = {'x1':x1, 'x2':x2, 'n1':n1, 'n2':n2,
                    'ind1':ind1, 'ind2':ind2, 'indr':indr, 'from':'self'}
        return dX12

    def set_dX12(self, dX12=None):
        dX12 = self._checkformat_dX12(dX12)
        self._dX12.update(dX12)

    @property
    def dX12(self):
        if self._dX12 is not None and self._dX12['from'] == 'geom':
            dX12 = self._dgeom['lCam'][0].dX12
        else:
            dX12 = self._dX12
        return dX12

    def get_X12plot(self, plot='imshow'):
        assert self.dX12 is not None
        if plot == 'imshow':
            x1, x2 = self.dX12['x1'], self.dX12['x2']
            x1min, Dx1min = x1[0], 0.5*(x1[1]-x1[0])
            x1max, Dx1max = x1[-1], 0.5*(x1[-1]-x1[-2])
            x2min, Dx2min = x2[0], 0.5*(x2[1]-x2[0])
            x2max, Dx2max = x2[-1], 0.5*(x2[-1]-x2[-2])
            extent = (x1min - Dx1min, x1max + Dx1max,
                      x2min - Dx2min, x2max + Dx2max)
            indr = self.dX12['indr']
            return x1, x2, indr, extent

lp = [p for p in params.values() if p.name not in ['lamb']]
DataCam2D.__signature__ = sig.replace(parameters=lp)



class DataCam2DSpectral(DataCam2D):
    """ Data object used for 1D cameras or list of 1D cameras  """
    @classmethod
    def _isSpectral(cls):  return True

    @property
    def lamb(self):
        return self.get_ddata('lamb')
    @property
    def nlamb(self):
        return self.get_ddata('nlamb')

lp = [p for p in params.values()]
DataCam2D.__signature__ = sig.replace(parameters=lp)







#####################################################################
#####################################################################
#####################################################################
#               Plasma2D
#####################################################################
#####################################################################



class Plasma2D(utils.ToFuObject):
    """ A generic class for handling 2D (and 1D) plasma profiles

    Provides:
        - equilibrium-related quantities
        - any 1d profile (can be remapped on 2D equilibrium)
        - spatial interpolation methods

    """
    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dtreat':{'order':['mask','interp-indt','interp-indch','data0','dfit',
                                'indt', 'indch', 'indlamb', 'interp-t']}}

    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, **kwdargs):
        # Python 2
        super(Plasma2D,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Plasma2D._ddef)
        #cls._dplot = copy.deepcopy(Struct._dplot)
        #cls._set_color_ddef(cls._color)


    def __init__(self, dtime=None, dradius=None, d1d=None,
                 d2d=None, dmesh=None, config=None,
                 Id=None, Name=None, Exp=None, shot=None,
                 fromdict=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        #self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(Plasma2D,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Plasma2D,self)._reset()
        self._dgroup = dict.fromkeys(self._get_keys_dgroup())
        self._dindref = dict.fromkeys(self._get_keys_dindref())
        self._ddata = dict.fromkeys(self._get_keys_ddata())
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None,
                               include=None, **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot = Id.Name, Id.Exp, Id.shot
        assert type(Name) is str, Name
        assert type(Exp) is str, Exp
        if include is None:
            include = cls._ddef['Id']['include']
        assert shot is None or type(shot) in [int,np.int64]
        if shot is None:
            if 'shot' in include:
                include.remove('shot')
        else:
            shot = int(shot)
            if 'shot' not in include:
                include.append('shot')
        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot,
                        'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dindrefdatagroup():
        largs = ['dtime', 'dradius', 'dmesh', 'd1d', 'd2d']
        return largs

    @staticmethod
    def _get_largs_dgeom():
        largs = ['config']
        return largs

    ###########
    # Get check and format inputs
    ###########

    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @staticmethod
    def _extract_dnd(dnd, k0):
        lk = dnd[k0].keys()
        if type(dnd[k0]) is dict:
            assert 'data' in lk
            data = dnd[k0]['data']
            if 'units' in lk:
                units = dnd[k0]['units']
            else:
                units = 'a.u.'
            if 'quant' in lk:
                quant = dnd[k0]['quant']
            else:
                quant = k0
            if 'name' in lk:
                name = dnd[k0]['name']
            else:
                name = k0
        else:
            data = dnd[k0]
            units = 'a.u.'
            quant = k0
            name = k0
        return data, units, quant, name

    @staticmethod
    def _checkformat_dtrm(dtime, dradius, dmesh):
        dd = {'dtime':dtime, 'dradius':dradius, 'dmesh':dmesh}
        lc = [type(di) is dict for di in dd.values()]
        if not all(lc):
            ls = ["    - %s : %s"%(kk, type(vv)) for kk, vv in dd.items()]
            msg = "All inputs should be dict !\n"
            msg += "\n".join(ls)
            raise exception(msg)

        # dtime
        for k0, v0 in dtime.items():
            c0 = type(k0) is str and 't' in v0.keys()
            c0 &= v0['t'] is not None
            try:
                dtime[k0]['t'] = np.asarray(v0['t']).ravel()
            except Exception:
                c0 = False
            if not c0:
                msg = "Arg dtime must be a dict of nested dict such that:\n"
                msg += "    - each key is a str (name of the radius)\n"
                msg += "    - each nested dict has one item ('t',np.ndarray)\n"
                msg += "e.g.:  dtime = {'t1':{'t':np.array([0,...,10])},\n"
                msg += "                't2':{'t':np.array([0.1,...5.5])}}"
                raise Exception(msg)

        # dradius
        for k0, v0 in dradius.items():
            c0 = type(k0) is str
            c0 &= type(v0) is dict and len(v0.keys()) == 1
            c0 &= 'size' in v0.keys() and type(v0['size']) is int
            if not c0:
                msg = "Arg dradius must be a dict of nested dict such that:\n"
                msg += "    - each key is a str (name of the radius)\n"
                msg += "    - each nested dict has one item ('size',int)\n"
                msg += "e.g.:  dradius = {'radius1':{'size}:10,\n"
                msg += "                  'radius2':{'size':100},\n"
                msg += "                  'radius3':{'size':1000}}"
                raise Exception(msg)

        # dmesh
        for k0, v0 in dmesh.items():
            assert type(k0) is str
            assert type(v0) is dict
            assert all([type(k1) is str for k1 in v0.keys()])
            ls = ['type','ftype','nodes','faces','nfaces','nnodes']
            assert all([ss in v0.keys() for ss in ls])
            dmesh[k0]['nodes'] = np.atleast_2d(dmesh[k0]['nodes']).astype(float)
            dmesh[k0]['faces'] = np.atleast_2d(dmesh[k0]['faces']).astype(int)
            assert dmesh[k0]['nodes'].shape == (v0['nnodes'],2)
            assert np.max(dmesh[k0]['faces']) < v0['nnodes']
            assert v0['type'] == 'tri'  # Only triangular meshes so far
            if v0['type'] == 'tri':
                assert dmesh[k0]['faces'].shape == (v0['nfaces'],3)
                if 'mpltri' not in v0.keys() or v0['mpltri'] is None:
                    dmesh[k0]['mpltri'] = mplTri(dmesh[k0]['nodes'][:,0],
                                                 dmesh[k0]['nodes'][:,1],
                                                 dmesh[k0]['faces'])
                assert isinstance(dmesh[k0]['mpltri'], mplTri)
                assert v0['ftype'] == 'linear'  # Only linear interp so far
                if v0['ftype'] == 'linear':
                    dmesh[k0]['size'] = v0['nnodes']
        return dtime, dradius, dmesh

    def _checkformat_inputs_dgeom(self, config=None):
        if config is not None:
            assert issubclass(config.__class__, utils.ToFuObject)
        return config

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgroup():
        lk = ['time', 'radius', 'mesh']
        return lk

    @staticmethod
    def _get_keys_dindref():
        lk = []
        return lk

    @staticmethod
    def _get_keys_ddata():
        lk = []
        return lk

    @staticmethod
    def _get_keys_dgeom():
        lk = ['config']
        return lk


    ###########
    # _init
    ###########

    def _init(self, dtime=None, dradius=None, dmesh=None, d1d=None, d2d=None,
              config=None, **kwargs):
        kwdargs = locals()
        kwdargs.update(**kwargs)
        largs = self._get_largs_dindrefdatagroup()
        kwdindrefdatagroup = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dgeom()
        kwdgeom = self._extract_kwdargs(kwdargs, largs)
        self._set_dindrefdatagroup(**kwdindrefdatagroup)
        self.set_dgeom(**kwdgeom)
        self._dstrip['strip'] = 0


    ###########
    # set dictionaries
    ###########

    def _set_dindrefdatagroup(self, dtime=None, dradius=None, dmesh=None,
                              d1d=None, d2d=None):

        # Check dtime is not None
        out = self._checkformat_dtrm(dtime=dtime, dradius=dradius, dmesh=dmesh)
        dtime, dradius, dmesh = out

        dgroup, dindref, ddata = {}, {}, {}
        # Get indt
        for k0 in dtime.keys():
            idt = str(id(dtime[k0]))
            dindref[idt] = {'size':dtime[k0]['t'].size,
                            'name':k0,
                            'group':'time'}

            ddata[idt] = {'data':dtime[k0]['t'],
                          'quant':'time', 'name':k0, 'units':'s',
                          'indref':(idt,)}

        # get radius
        for k0, v0 in dradius.items():
            idr = str(id(v0))
            dindref[idr] = {'size':v0['size'],
                            'name':k0,
                            'group':'radius'}

        # Get d1d
        iddref = None
        if d1d is not None:
            for k0 in d1d.keys():
                idd = str(id(d1d[k0]))
                data, units, quant, name = self._extract_dnd(d1d, k0)

                if data is None:
                    msg = "Provided data is None:\n"
                    msg += "    - d1d[%s]"%k0
                    raise Exception(msg)

                # data
                shape = data.shape
                lrefrad = [kk for kk, vv in dindref.items()
                           if (vv['group'] == 'radius'
                               and vv['size'] == shape[-1]
                               and vv['name'] == d1d[k0]['radius'])]
                assert len(lrefrad) == 1
                if len(shape) == 1:
                    indref = (lrefrad[0],)
                else:
                    kt = [kk for kk,vv in dindref.items()
                         if (vv['group'] == 'time'
                             and vv['size'] == shape[0]
                             and vv['name'] == d1d[k0]['time'])]
                    assert len(kt) == 1
                    indref = (kt[0], lrefrad[0])
                ddata[idd] = {'data':data,
                              'quant':quant, 'name':name, 'units':units,
                              'indref':indref}

        # dmesh ref
        if dmesh is not None:
            for k0 in dmesh.keys():
                idm = str(id(dmesh[k0]))
                dindref[idm] = {'size':dmesh[k0]['size'],
                                'name':k0,
                                'group':'mesh'}

                ddata[idm] = {'data':dmesh[k0],
                              'quant':k0, 'name':k0, 'units':'a.u.',
                              'indref':(idm,)}

        # d2d
        if d2d is not None:
            for k0 in d2d.keys():
                idd2 = str(id(d2d[k0]))
                data, units, quant, name = self._extract_dnd(d2d, k0)
                shape = data.shape
                lrefrad = [kk for kk, vv in dindref.items()
                           if (vv['group'] == 'mesh'
                               and vv['size'] == shape[-1]
                               and vv['name'] == d2d[k0]['mesh'])]
                assert len(lrefrad) == 1
                if len(shape) == 1:
                    indref = (lrefrad[0],)
                else:
                    kt = [kk for kk,vv in dindref.items()
                         if (vv['group'] == 'time'
                             and vv['size'] == shape[0]
                             and vv['name'] == d2d[k0]['time'])]
                    assert len(kt) == 1
                    indref = (kt[0], lrefrad[0])
                ddata[idd2] = {'data':data,
                               'quant':quant, 'name':name, 'units':units,
                               'indref':indref}

        # dgroup
        dgroup = {'time':{'indref':idt},
                  'radius':{'indref':idr},
                  'mesh':{'indref':idm}}

        # Complement
        self._complement(dgroup, dindref, ddata)

        # Update dict
        self._dgroup = dgroup
        self._dindref = dindref
        self._ddata = ddata


    @classmethod
    def _complement(cls, dgroup, dindref, ddata):

        # --------------
        # ddata
        lkstr = ['quant','name','units']
        for id_, vd in ddata.items():
            assert all([kk in vd.keys() for kk in lkstr])
            assert all([type(vd[kk]) is str for kk in lkstr])
            linind = [ii in dindref.keys() for ii in vd['indref']]
            if not all(linind):
                msg = "In ddata[%s], indref not in dindref.keys():\n"%str(id_)
                msg += "    - quant: %s\n"%vd['quant']
                msg += "    - name : %s\n"%vd['name']
                msg += "    - units : %s\n"%vd['units']
                if type(vd['data']) is np.ndarray:
                    msg += "    - shape : %s\n"%str(vd['data'].shape)
                msg += "    - indref : %s\n"%str(vd['indref'])
                msg += "  dindref.keys() = %s"%str(dindref.keys())
                raise Exception(msg)
            ddata[id_]['lgroup'] = list(set([dindref[ii]['group']
                                             for ii in vd['indref']]))
            assert all([ii in dindref.keys() for ii in vd['indref']])
            assert 'data' in vd.keys()
            type_ = type(vd['data'])
            shape = tuple([dindref[ii]['size'] for ii in vd['indref']])
            if len(shape) == 1:
                c0 = type_ is dict and 'mesh' in ddata[id_]['lgroup']
                c1 = not c0 and len(vd['data']) == shape[0]
                assert c0 or c1
            else:
                assert type(vd['data']) is np.ndarray
                assert vd['data'].shape == shape
        lni = sorted([(vd['name'],vd['indref']) for vd in ddata.values()])
        if not len(set(lni)) == len(lni):
            msg = "Names / indref tuples for data should be unique !\n"
            msg += "    - %s"%str(lni)
            raise Exception(msg)

        # --------------
        # dindref
        for id_ in dindref.keys():
            dindref[id_]['liddata'] = [kk for kk, vv in ddata.items()
                                       if id_ in vv['indref']]
            assert dindref[id_]['group'] in dgroup.keys()

        # --------------
        # dgroup
        for gg, vg in dgroup.items():
            lidindref = [id_ for id_, vv in dindref.items() if vv['group'] == gg]
            liddata = [id_ for id_ in ddata.keys()
                       if any([id_ in dindref[vref]['liddata']
                               for vref in lidindref])]
            assert vg['indref'] in lidindref
            dgroup[gg]['lidindref'] = lidindref
            dgroup[gg]['liddata'] = liddata


    def set_dgeom(self, config=None):
        config = self._checkformat_inputs_dgeom(config=config)
        self._dgeom = {'config':config}


    ###########
    # strip dictionaries
    ###########

    def _strip_ddata(self, strip=0):
        pass


    def _strip_dgeom(self, strip=0, force=False, verb=True):
        if self._dstrip['strip']==strip:
            return

        if strip in [0] and self._dstrip['strip'] in [1]:
            config = None
            if self._dgeom['config'] is not None:
                assert type(self._dgeom['config']) is str
                config = utils.load(self._dgeom['config'], verb=verb)

            self._set_dgeom(config=config)

        elif strip in [1] and self._dstrip['strip'] in [0]:
            if self._dgeom['config'] is not None:
                path = self._dgeom['config'].Id.SavePath
                name = self._dgeom['config'].Id.SaveName
                pfe = os.path.join(path, name+'.npz')
                lf = os.listdir(path)
                lf = [ff for ff in lf if name+'.npz' in ff]
                exist = len(lf)==1
                if not exist:
                    msg = """BEWARE:
                        You are about to delete the config object
                        Only the path/name to saved a object will be kept

                        But it appears that the following object has no
                        saved file where specified (obj.Id.SavePath)
                        Thus it won't be possible to retrieve it
                        (unless available in the current console:"""
                    msg += "\n    - {0}".format(pfe)
                    if force:
                        warning.warn(msg)
                    else:
                        raise Exception(msg)
                self._dgeom['config'] = pfe

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: dgeom pathfiles
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(Plasma2D,self).strip(strip=strip, verb=verb)

    def _strip(self, strip=0, verb=True):
        self._strip_dgeom(strip=strip, verb=verb)

    def _to_dict(self):
        dout = {'dgroup':{'dict':self._dgroup, 'lexcept':None},
                'dindref':{'dict':self._dindref, 'lexcept':None},
                'ddata':{'dict':self._ddata, 'lexcept':None},
                'dgeom':{'dict':self._dgeom, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dgroup.update(**fd['dgroup'])
        self._dindref.update(**fd['dindref'])
        self._ddata.update(**fd['ddata'])
        self._dgeom.update(**fd['dgeom'])


    ###########
    # properties
    ###########

    @property
    def dgroup(self):
        return self._dgroup
    @property
    def dindref(self):
        return self._dindref
    @property
    def ddata(self):
        return self._ddata
    @property
    def dtime(self):
        return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     if vv['group'] == 'time'])
    @property
    def dradius(self):
        return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     if vv['group'] == 'radius'])
    @property
    def dmesh(self):
        return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     if vv['group'] == 'mesh'])
    @property
    def config(self):
        return self._dgeom['config']

    #---------------------
    # Read-only for internal use
    #---------------------

    @property
    def _lquantboth(self):
        """ Return list of quantities available both in 1d and 2d """
        lq1 = [self._ddata[vd]['quant'] for vd in self._dgroup['radius']['liddata']]
        lq2 = [self._ddata[vd]['quant'] for vd in self._dgroup['mesh']['liddata']]
        lq = list(set(lq1).intersection(lq2))
        return lq

    def _get_liddata(self, quant=None, name=None, units=None,
                     indref=None, group=None, log='all'):
        assert log in ['all','any']
        lid = np.array(list(self._ddata.keys()))
        ind = np.ones((5,len(lid)),dtype=bool)
        if quant is not None:
            ind[0,:] = [self._ddata[id_]['quant'] == quant for id_ in lid]
        if name is not None:
            ind[1,:] = [self._ddata[id_]['name'] == name for id_ in lid]
        if units is not None:
            ind[2,:] = [self._ddata[id_]['units'] == units for id_ in lid]
        if indref is not None:
            ind[3,:] = [indref in self._ddata[id_]['indref'] for id_ in lid]
        if group is not None:
            ind[4,:] = [group in self._ddata[id_]['lgroup'] for id_ in lid]

        if log == 'all':
            ind = np.all(ind, axis=0)
        else:
            ind = np.any(ind, axis=0)
        if np.any(ind):
            lid = lid[ind.nonzero()[0]]
        else:
            lid = np.array([],dtype=int)
        return lid


    #---------------------
    # Methods for showing data
    #---------------------

    def get_summary(self, max_columns=100, width=1000,
                    verb=True, Return=False):
        """ Summary description of the object content as a pandas DataFrame """
        # # Make sure the data is accessible
        # msg = "The data is not accessible because self.strip(2) was used !"
        # assert self._dstrip['strip']<2, msg

        # -----------------------
        # Build the list
        data = []
        for k0,v0 in self._dgroup.items():
            lu = [k0, v0['indref']]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'indref']
        df0 = pd.DataFrame(data, columns=col)

        # -----------------------
        # Build the list
        data = []
        for k0,v0 in self._dindref.items():
            lu = [k0, v0['group'], v0['size']]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'group', 'size']
        df1 = pd.DataFrame(data, columns=col)

        # -----------------------
        # Build the list
        data = []
        for k0,v0 in self._ddata.items():
            if type(v0['data']) is np.ndarray:
                shape = v0['data'].shape
            else:
                shape = v0['data'].__class__.__name__
            lu = [k0, v0['quant'], v0['name'], v0['units'], shape,
                  v0['indref'], v0['lgroup']]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'quant', 'name', 'units', 'shape', 'indref', 'lgroup']
        df2 = pd.DataFrame(data, columns=col)
        pd.set_option('display.max_columns',max_columns)
        pd.set_option('display.width',width)

        if verb:
            sep = "\n------------\n"
            print("dgroup", sep, df0, "\n")
            print("dindref", sep, df1, "\n")
            print("ddata", sep, df2, "\n")
        if Return:
            return df0, df1, df2


    #---------------------
    # Methods for interpolation
    #---------------------


    def _get_quantrefid(self, quant, ref):

        # Get relevant lists
        l1dn = self._get_liddata(name=quant, group='radius')
        l1dq = self._get_liddata(quant=quant, group='radius')

        l2dn = self._get_liddata(name=quant, group='mesh')
        l2dq = self._get_liddata(quant=quant, group='mesh')

        # Check if quant 2d
        idref1d, idref2d = None, None
        if len(l2dn) > 0:
            if len(l2dn) > 1:
                msg = "Ambiguous 2d name:\n"
                msg += "    - name : %s\n"%quant
                msg += "    - matches : %s\n"%str(l2dn)
                raise Exception(msg)
            idquant = l2dn[0]
            return idquant, idref1d, idref2d
        if len(l2dq) > 0:
            if len(l2dq) > 1:
                msg = "Ambiguous 2d quant:\n"
                msg += "    - quant : %s\n"%quant
                msg += "    - matches : %s\n"%str(l2dq)
                msg = "Ambiguous 2d quant"
                raise Exception(msg)
            idquant = l2dq[0]
            return idquant, idref1d, idref2d

        # Check if quant 1d
        if len(l1dn) > 0:
            if len(l1dn) > 1:
                msg = "Ambiguous 1d name:\n"
                msg += "    - name : %s\n"%quant
                msg += "    - matches : %s\n"%str(l1dn)
                msg = "Ambiguous 1d name"
                raise Exception(msg)
            idquant = l1dn[0]
        elif len(l1dq) > 0:
            if len(l1dq) > 1:
                msg = "Ambiguous 1d quant:\n"
                msg += "    - quant : %s\n"%quant
                msg += "    - matches : %s\n"%str(l1dn)
                msg = "Ambiguous 1d quant"
                raise Exception(msg)
            idquant = l1dq[0]
        else:
            msg = "Quant %s matches no name / quant in 2d nor 1d!"%quant
            raise Exception(msg)

        # Get associated ref
        assert ref is not None, "ref must be provided to interpolate %s"%quant

        l1dn = self._get_liddata(name=ref, group='radius')
        l1dq = self._get_liddata(quant=ref, group='radius')

        l2dn = self._get_liddata(name=ref, group='mesh')
        l2dq = self._get_liddata(quant=ref, group='mesh')

        cn = len(l1dn) >= 1 and len(l2dn) >= 1
        cq = len(l1dq) >= 1 and len(l2dq) >= 1
        if not (cn or cq):
            msg = "Ref %s must match either a name or a quantity !\n"%ref
            msg += "    and it should be available both as 1d and 2d!\n"
            msg += "  => check self.get_summary()"
            raise Exception(msg)
        if cn:
            if not (len(l1dn) == 1 and len(l2dn) == 1):
                msg = "Ambiguous ref name !"
                raise Exception(msg)
            idref1d = l1dn[0]
            idref2d = l2dn[0]
        else:
            idref1d = l1dq[0]
            idref2d = l2dq[0]

        return idquant, idref1d, idref2d


    def _get_indtmult(self, t=None, idquant=None, idref1d=None, idref2d=None):

        # Get time vectors and bins
        idtq = self._ddata[idquant]['indref'][0]
        tq = self._ddata[idtq]['data']
        if idref1d is not None:
            tbinq = 0.5*(tq[1:]+tq[:-1])
            idtr1 = self._ddata[idref1d]['indref'][0]
            tr1 = self._ddata[idtr1]['data']
            tbinr1 = 0.5*(tr1[1:]+tr1[:-1])
        if idref2d is not None and idref2d != idref1d:
            idtr2 = self._ddata[idref2d]['indref'][0]
            tr2 = self._ddata[idtr2]['data']
            tbinr2 = 0.5*(tr2[1:]+tr2[:-1])

        # Get tbinall and tall
        if idref1d is None:
            tbinall = tbinq
            tall = tq
        else:
            if idref2d is None:
                tbinall = np.unique(np.r_[tbinq,tbinr1])
            else:
                tbinall = np.unique(np.r_[tbinq,tbinr1,tbinr2])
            tall = np.r_[tbinall[0] - 0.5*(tbinall[1]-tbinall[0]),
                         0.5*(tbinall[1:]+tbinall[:-1]),
                         tbinall[-1] + 0.5*(tbinall[-1]-tbinall[-2])]

        # Get indtqr1r2 (tall with respect to tq, tr1, tr2)
        indtq, indtr1, indtr2 = None, None, None
        indtq = np.digitize(tall, tbinq)
        if idref1d is None:
            assert np.all(indtq == np.arange(0,tall.size))
        if idref1d is not None:
            indtr1 = np.digitize(tall, tbinr1)
        if idref2d is not None:
            indtr2 = np.digitize(tall, tbinr2)

        # Get indt (t with respect to tbinall)
        indt, indtu = None, None
        if t is not None:
            indt = np.digitize(t, tbinall)
            indtu = np.unique(indt)

            # Update
            tall = tall[indtu]
            indtq = indtq[indtu]
            if idref1d is not None:
                indtr1 = indtr1[indtu]
            if idref2d is not None:
                indtr2 = indtr2[indtu]
        ntall = tall.size
        return tall, ntall, indt, indtu, indtq, indtr1, indtr2


    def _interp_pts2profile(self, ptsRZ,
                            idquant, idref1d, idref2d, t=None,
                            interp_t='nearest',
                            interp_space='linear',
                            fill_value=np.nan):

        # Get idmesh
        if idref1d is None:
            lidmesh = [qq for qq in self._ddata[idquant]['indref']
                       if self._dindref[qq]['group'] == 'mesh']
        else:
            lidmesh = [qq for qq in self._ddata[idref2d]['indref']
                       if self._dindref[qq]['group'] == 'mesh']
        assert len(lidmesh) == 1
        idmesh = lidmesh[0]

        # Get mesh
        mpltri = self._ddata[idmesh]['data']['mpltri']
        trifind = mpltri.get_trifinder()
        r, z = ptsRZ[0], ptsRZ[1]

        # Get common time indices
        if interp_t == 'nearest':
            out = self._get_indtmult(t=t, idquant=idquant,
                                     idref1d=idref1d, idref2d=idref2d)
            tall, ntall, indt, indtu, indtq, indtr1, indtr2 = out

        # Prepare output
        shapeval = list(ptsRZ.shape)
        shapeval[0] = ntall
        val = np.full(tuple(shapeval), np.nan)

        # Interpolate
        # Note : Maybe consider using scipy.LinearNDInterpolator ?
        vquant = self._ddata[idquant]['data']
        if idref1d is None:
            if t is None:
                for ii in range(0,ntall):
                    val[ii,...] = mplTriLinInterp(mpltri,
                                                  vquant[indtq[ii],:],
                                                  trifinder=trifind)(r,z)
            else:
                for ii in range(0,ntall):
                    ind = indt == indtu[ii]
                    val[ind,...] = mplTriLinInterp(mpltri,
                                                   vquant[indtq[ii],:],
                                                   trifinder=trifind)(r,z)

        else:
            vr2 = self._ddata[idref2d]['data']
            vr1 = self._ddata[idref1d]['data']
            if t is None:
                for ii in range(0,ntall):
                    # get ref values for mapping
                    vii = mplTriLinInterp(mpltri,
                                          vr2[indtr2[ii],:],
                                          trifinder=trifind)(r,z)

                    # interpolate 1d
                    val[ii,...] = scpinterp.interp1d(vr1[indtr1[ii],:],
                                                     vquant[indtq[ii],:],
                                                     kind=interp_space,
                                                     bounds_error=False,
                                                     fill_value=fill_value)(np.asarray(vii))
            else:
                for ii in range(0,ntall):
                    # get ref values for mapping
                    vii = mplTriLinInterp(mpltri,
                                          vr2[indtr2[ii],:],
                                          trifinder=trifind)(r,z)

                    # interpolate 1d
                    ind = indt == indtu[ii]
                    val[ind,...] = scpinterp.interp1d(vr1[indtr1[ii],:],
                                                      vquant[indtq[ii],:],
                                                      kind=interp_space,
                                                      bounds_error=False,
                                                      fill_value=fill_value)(np.asarray(vii))


        # return time
        if t is None:
            t = tall
        return val, t


    def interp_pts2profile(self, quant, ptsRZ=None, t=None, ref=None,
                           interp_t='nearest', interp_space='linear',
                           fill_value=np.nan):
        """ Return the value of the desired profiles_1d quantity

        For the desired inputs points (pts):
            - pts are in (R,Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check inputs
        assert interp_t == 'nearest', "Only 'nearest' available so far!"

        # Check requested quant is available in 2d or 1d
        idquant, idref1d, idref2d = self._get_quantrefid(quant, ref)

        # Check the ptsRZ is (2,...) array of floats
        if ptsRZ is None:
            if idref1d is None:
                ptsRZ = self.dmesh[idquant]['data']['nodes'].T
            else:
                ptsRZ = self.dmesh[idref2]['data']['nodes'].T

        ptsRZ = np.atleast_2d(ptsRZ)
        if ptsRZ.shape[0] != 2:
            msg = "ptsRZ must ba np.ndarray of (R,Z) points coordinates\n"
            msg += "Can be multi-dimensional, but the 1st dimension is (R,Z)\n"
            msg += "    - Expected shape : (2,...)\n"
            msg += "    - Provided shape : %s"%str(ptsRZ.shape)
            raise Exception(msg)

        # Interpolation (including time broadcasting)
        val, t = self._interp_pts2profile(ptsRZ, idquant,
                                          idref1d, idref2d, t=t,
                                          interp_t=interp_t,
                                          interp_space=interp_space,
                                          fill_value=fill_value)
        return val, t


    #---------------------
    # Methods for getting data
    #---------------------

    def _get_idXq(self, X):
        l1dn = self._get_liddata(name=X, group='radius')
        l1dq = self._get_liddata(quant=X, group='radius')
        if not (len(l1dn) >= 1 or len(l1dq) >= 1):
            msg = "No match for X = %s in group radius !"%X
            raise Exception(msg)
        if len(l1dn) > 1:
            msg = "Several matches for X = %s ,by name, in group radius"%X
            raise Exception(msg)
        if len(l1dn) == 1:
            idX = l1dn[0]
        else:
            if len(l1dq) > 1:
                msg = "Several matches for X = %s ,by quant, in group radius"%X
                raise Exception(msg)
            idX = l1dq[0]
        return idX

    def get_Data(self, lquant, X=None, ref=None,
                 remap=False, res=0.01, interp_space='linear'):

        try:
            import tofu.data as tfd
        except Exception:
            from .. import data as tfd

        # Check and format input
        assert type(lquant) in [str,list]
        if type(lquant) is str:
            lquant = [lquant]
        nquant = len(lquant)

        # Get X if common
        c0 = type(X) is str or (type(X) is list and len(X) == nquant)
        if not c0:
            msg = "X must be specified, either as :\n"
            msg += "    - a str (name or quant)\n"
            msg += "    - a list of str\n"
            msg += "    Provided: %s"%str(X)
            raise Exception(msg)

        if type(X) is str:
            idX = self._get_idXq(X)

        # prepare remap pts
        if remap:
            assert self.config is not None
            refS = list(self.config.dStruct['dObj']['Ves'].values())[0]
            ptsRZ, x1, x2, extent = refS.get_sampleCross(res, mode='imshow')
            dmap = {'t':None, 'data2D':None, 'extent':extent}
            if ref is None and X in self._lquantboth:
                ref = X

        # Define Data
        dcommon = dict(Exp=self.Id.Exp, shot=self.Id.shot,
                       Diag='profiles1d', config=self.config)

        # Get output
        dextra = None
        lout = [None for qq in lquant]
        for ii in range(0,nquant):
            qq = lquant[ii]
            if remap:
                # Check requested quant is available in 2d or 1d
                idq, idrefd1, idref2d = self._get_quantrefid(qq, ref)
            else:
                idq = self._get_idXq(qq)
            if idq not in self._dgroup['radius']['liddata']:
                msg = "Only 1d quantities can be turned into tf.data.Data !\n"
                msg += "    - %s is not a radius-dependent quantity"%qq
                raise Exception(msg)
            idt = self._ddata[idq]['indref'][0]

            if type(X) is list:
                idX = self._get_idXq(X[ii])

            if remap:
                dmapii = dict(dmap)
                val, tii = self.interp_pts2profile(qq, ptsRZ=ptsRZ, ref=ref,
                                                   interp_space=interp_space)
                dmapii['data2D'], dmapii['t'] = val, tii
                dextra = {'map':dmapii}
            lout[ii] = DataCam1D(Name = qq,
                                 data = self._ddata[idq]['data'],
                                 t = self._ddata[idt]['data'],
                                 X = self._ddata[idX]['data'],
                                 dextra = dextra, **dcommon)
        if nquant == 1:
            lout = lout[0]
        return lout


    #---------------------
    # Methods for plotting data
    #---------------------

    def plot(self, lquant, X=None, ref=None,
             remap=False, res=0.01, interp_space='linear'):
        lDat = self.get_Data(lquant, X=X, remap=remap,
                             ref=ref, res=res, interp_space=interp_space)
        if type(lDat) is list:
            kh = lDat[0].plot_combine(lDat[1:])
        else:
            kh = lDat.plot()
        return kh

    def plot_combine(self, lquant, lData, X=None, ref=None,
                     remap=False, res=0.01, interp_space='linear'):
        lDat = self.get_Data(lquant, X=X, remap=remap,
                             ref=ref, res=res, interp_space=interp_space)
        if type(lDat) is list:
            lData = lDat[1:] + lData
        kh = lDat[0].plot_combine(lData)
        return kh
