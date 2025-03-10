# -*- coding: utf-8 -*-

# Built-in
import sys
import os
import itertools as itt
import copy
import warnings
# from abc import ABCMeta, abstractmethod
import inspect

# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation as mplTri


# tofu
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.data._comp as _comp
    import tofu.data._plot as _plot
    import tofu.data._def as _def
    import tofu._physics as _physics
except Exception:
    from . import _comp as _comp
    from . import _plot as _plot
    from . import _def as _def
    from .. import _physics as _physics

__all__ = ['DataCam1D','DataCam2D',
           'DataCam1DSpectral','DataCam2DSpectral']
           # 'Plasma2D']
_INTERPT = 'zero'


#############################################
#       utils
#############################################

def _format_ind(ind=None, n=None):
    """Helper routine to convert selected channels (as numbers) in `ind` to
    a boolean array format.

    Parameters
    ----------
    ind : integer, or list of integers
        A channel or a list of channels that the user wants to select.
    n : integer, or None
        The total number of available channels.

    Returns
    -------
    ind : ndarray of booleans, size (n,)
        The array with the selected channels set to True, remaining ones set
        to False


    Examples
    --------

    >>> _format_ind(ind=[0, 3], n=4)
    [True, False, False, True]

    """
    if ind is None:
        ind = np.ones((n,),dtype=bool)
    else:
        # list of accepted integer types
        lInt = (int, np.integer)
        if isinstance(ind, lInt):
            ii = np.zeros((n,),dtype=bool)
            ii[int(ii)] = True
            ind = ii
        else:
            assert hasattr(ind,'__iter__')
            if isinstance(ind[0], (bool, np.bool_)):
                ind = np.asarray(ind).astype(bool)
                assert ind.size==n
            elif isinstance(ind[0], lInt):
                ind = np.asarray(ind).astype(int)
                ii = np.zeros((n,),dtype=bool)
                ii[ind] = True
                ind = ii
            else:
                msg = ("Index must be int, or an iterable of bool or int "
                       "(first element of index has"
                       " type: {})!".format(type(ind[0]))
                       )
                raise Exception(msg)
    return ind


def _select_ind(v, ref, nRef):
    ltypes = (int, float, np.integer)
    C0 = np.isscalar(v) and isinstance(v, ltypes)
    C1 = isinstance(v, np.ndarray)
    C2 = isinstance(v, list)
    C3 = isinstance(v, tuple)
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
        c0 = len(v)==2 and all([isinstance(vv, ltypes) for vv in v])
        c1 = all([(type(vv) is type(v) and len(vv)==2
                   and all([isinstance(vvv, ltypes) for vvv in vv]))
                  for vv in v])
        assert c0!=c1
        if c0:
            v = [v]
        for vv in v:
            ind = ind | ((ref>=vv[0]) & (ref<=vv[1]))
        if C3:
            ind = ~ind
    if ref.ndim == 1:
        ind = np.atleast_1d(ind.squeeze())
    return ind




#############################################
#       class
#############################################

class DataAbstract(utils.ToFuObject):

    # __metaclass__ = ABCMeta

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
                 fromdict=None, sep=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

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
            if not isinstance(Id, utils.ID):
                msg = ("Arg Id must be a utils.ID instance!\n"
                       + "\t- provided: {}".format(Id))
                raise Exception(msg)
            Name, Exp, shot, Diag = Id.Name, Id.Exp, Id.shot, Id.Diag
        assert type(Name) is str, Name
        assert type(Diag) is str, Diag
        assert type(Exp) is str, Exp
        if include is None:
            include = cls._ddef['Id']['include']
        assert shot is None or isinstance(shot, (int, np.integer))
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
        if data is None:
            msg = "data can not be None!"
            raise Exception(msg)
        data = np.atleast_1d(np.asarray(data).squeeze())

        if data.ndim == 1:
            data = data.reshape((1, data.size))
        if t is not None:
            t = np.atleast_1d(np.asarray(t).squeeze())
        if X is not None:
            X = np.atleast_1d(np.asarray(X).squeeze())
        if indtX is not None:
            indtX = np.atleast_1d(np.asarray(indtX, dtype=int).squeeze())
        if lamb is not None:
            lamb = np.atleast_1d(np.asarray(lamb).squeeze())
        if indtlamb is not None:
            indtlamb = np.atleast_1d(np.asarray(indtlamb, dtype=int).squeeze())
        if indXlamb is not None:
            indXlamb = np.atleast_1d(np.asarray(indXlamb, dtype=int).squeeze())
        if indtXlamb is not None:
            indtXlamb = np.atleast_1d(np.asarray(indtXlamb,
                                                 dtype=int).squeeze())

        ndim = data.ndim
        assert ndim in [2,3]
        if not self._isSpectral():
            msg = ("self is not of spectral type\n"
                   + "  => the data cannot be 3D ! (ndim)")
            assert ndim==2, msg

        nt = data.shape[0]
        if t is None:
            t = np.arange(0,nt)
        else:
            if t.shape != (nt,):
                msg = ("Wrong time dimension\n"
                       + "\t- t.shape = {}\n".format(t.shape)
                       + "\t- nt = {}".format(nt))
                raise Exception(msg)

        n1 = data.shape[1]
        if ndim==2:
            lC = [X is None, lamb is None]
            if not any(lC):
                msg = "Please provide at least X or lamb (both are None)!"
                raise Exception(msg)
            if all(lC):
                if self._isSpectral():
                    X = np.array([0])
                    lamb = np.arange(0,n1)
                    data = data.reshape((nt,1,n1))
                else:
                    X = np.arange(0,n1)
            elif lC[0]:
                if not self._isSpectral():
                    msg = "lamb provided => self._isSpectral() must be True!"
                    raise Exception(msg)
                X = np.array([0])
                data = data.reshape((nt, 1, n1))
                if lamb.ndim not in [1, 2]:
                    msg = ("lamb.ndim must be in [1, 2]\n"
                           + "\t- lamb.shape = {}".format(lamb.shape))
                    raise Exception(msg)
                if lamb.ndim == 1:
                    if lamb.size != n1:
                        msg = ("lamb has wrong size!\n"
                               + "\t- expected: {}".format(n1)
                               + "\t- provided: {}".format(lamb.size))
                        raise Exception(msg)
                elif lamb.ndim == 2:
                    if lamb.shape[1] != n1:
                        msg = ("lamb has wrong shape!\n"
                               + "\t- expected: (.., {})".format(n1)
                               + "\t- provided: {}".format(lamb.shape))
                        raise Exception(msg)
            else:
                if self._isSpectral():
                    msg = "object cannot be spectral!"
                    raise Exception(msg)
                if X.ndim not in [1, 2] or X.shape[-1] != n1:
                    msg = ("X.ndim should be in [1, 2]\n"
                           + "\t- expected: (..., {})\n".format(n1)
                           + "\t- provided: {}".format(X.shape))
                    raise Exception(msg)
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
                assert np.min(indtlamb) >= 0 and np.max(indtlamb) <= nnlamb
            if not lC[1]:
                assert indXlamb.shape == (nch,)
                assert np.min(indXlamb) >= 0 and np.max(indXlamb) <= nnlamb
        else:
            assert indtXlamb.shape == (nt, nch)
            assert np.min(indtXlamb) >= 0 and np.max(indtXlamb) <= nnlamb

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
            X = np.atleast_1d(np.asarray(X).squeeze())
        if indtX is not None:
            indtX = np.atleast_1d(np.asarray(indtX).squeeze())
        if indXlamb is not None:
            indXlamb = np.atleast_1d(np.asarray(indXlamb).squeeze())

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
            assert indtX.shape == (nt,)
            assert np.argmin(indtX) >= 0 and np.argmax(indtX) <= nnch
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
                if not (type(dextra[k]) is dict and 't' in dextra[k].keys()):
                    msg = "All dextra values should be dict with 't':\n"
                    msg += "    - dextra[%s] = %s"%(k,str(dextra[k]))
                    raise Exception(msg)
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
                            warnings.warn(msg)
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
                        warnings.warn(msg)
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

    # @abstractmethod
    def _isSpectral(self):
        return 'spectral' in self.__class__.name.lower()
    # @abstractmethod
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
        assert isinstance(val, (int, float, np.integer))
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
        f = scpinterp.interp1d(t, data, kind=kind, axis=0, copy=True,
                               bounds_error=True, fill_value=np.nan,
                               assume_sorted=False)
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
        lc = [d.ndim in [2,3], t.shape == (nt,), X.shape == (nnch, nch)]
        if not all(lc):
            msg = "Data, X, t shape unconsistency:\n"
            msg += "    - data.shape: %s\n"%str(d.shape)
            msg += "    - X.shape:     %s\n"%str(X.shape)
            msg += "    - (nnch, nch): %s\n"%str((nnch,nch))
            msg += "    - t.shape: %s\n"%str(t.shape)
            msg += "    - nt :     %s"%str(nt)
            raise Exception(msg)
        if lamb is not None:
            assert lamb.shape == (self._ddataRef['nnlamb'], nlamb)

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

            ltypes = (str, int, float,np.integer)
            C0 = isinstance(val, ltypes)
            C1 = type(val) in [list,tuple,np.ndarray]
            assert C0 or C1
            if C0:
                val = [val]
            else:
                assert all([isinstance(vv, ltypes) for vv in val])
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
             bck=True, fs=None, dmargin=None, wintit=None, tit=None,
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
                             invert=invert, dmarker=dmarker, bck=bck,
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
                     bck=True, fs=None, dmargin=None, wintit=None, tit=None,
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
                             invert=invert, dmarker=dmarker, bck=bck,
                             sharey=sharey, sharelamb=sharelamb,
                             fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                             fontsize=fontsize, labelpad=labelpad,
                             draw=draw, connect=connect)
        return kh

    def plot_combine(self, lD, key=None, bck=True, indref=0,
                  cmap=None, ms=4, vmin=None, vmax=None,
                  vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
                  ntMax=None, nchMax=None, nlbdMax=3,
                  inct=[1,10], incX=[1,5], inclbd=[1,10],
                  lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
                  fmt_t='06.3f', fmt_X='01.0f',
                  invert=True, Lplot='In', dmarker=None,
                  fs=None, dmargin=None, wintit=None, tit=None, sharex=False,
                  fontsize=None, labelpad=None, draw=True, connect=True):
        """ Plot several Data instances of different diags

        Useful to visualize several diags for the same shot

        """
        C0 = isinstance(lD,list)
        C0 = C0 and all([issubclass(dd.__class__,DataAbstract) for dd in lD])
        C1 = issubclass(lD.__class__,DataAbstract)
        assert C0 or C1, 'Provided first arg. must be a tf.data.DataAbstract or list !'
        lD = [lD] if C1 else lD
        kh = _plot.Data_plot_combine([self]+lD, key=key, bck=bck,
                                     indref=indref, cmap=cmap, ms=ms,
                                     vmin=vmin, vmax=vmax,
                                     vmin_map=vmin_map, vmax_map=vmax_map,
                                     cmap_map=cmap_map, normt_map=normt_map,
                                     ntMax=ntMax, nchMax=nchMax,
                                     inct=inct, incX=incX,
                                     lls=lls, lct=lct, lcch=lcch, cbck=cbck,
                                     fmt_t=fmt_t, fmt_X=fmt_X,
                                     invert=invert, Lplot=Lplot, sharex=sharex,
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
                         bck=True, fs=None, dmargin=None, wintit=None,
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
                                         nfMax=nfMax, bck=bck, fs=fs,
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
        return chronos, s, topos


    def extract_svd(self, modes=None, lapack_driver='gesdd', out=object):
        """ Extract, as Data object, the filtered signal using selected modes

        The svd (chronos, s, topos) is computed,
        The selected modes are used to re-construct a filtered signal, using:
            data = chronos[:,modes] @ (s[None,modes] @ topos[modes,:]

        The result is exported a an array or a Data object on the same class
        """
        if self._isSpectral():
            msg = "svd not implemented yet for spectral data class"
            raise Exception(msg)
        msg = None
        if modes is not None:
            try:
                modes = np.r_[modes].astype(int)
            except Exception as err:
                msg = str(err)
        else:
            msg = "Arg mode cannot be None !"
        if msg is not None:
            msg += "\n\nArg modes must a positive int or a list of such!\n"
            msg += "    - Provided: %s"%str(modes)
            raise Exception(msg)

        chronos, s, topos = _comp.calc_svd(self.data, lapack_driver=lapack_driver)
        data = np.matmul(chronos[:, modes], (s[modes, None] * topos[modes, :]))
        if out is object:
            data = self.__class__(data=data, t=self.t, X=self.X,
                                  lCam=self.lCam, config=self.config,
                                  Exp=self.Id.Exp, Diag=self.Id.Diag,
                                  shot=self.Id.shot,
                                  Name=self.Id.Name + '-svd%s'%str(modes))
        return data


    def plot_svd(self, lapack_driver='gesdd', modes=None, key=None, bck=True,
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
                                 key=key, bck=bck, Lplot=Lplot,
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


    def save_to_imas(self, ids=None, shot=None, run=None, refshot=None, refrun=None,
                     user=None, database=None, version=None, occ=None,
                     dryrun=False, deep=True, verb=True,
                     restore_size=True, forceupdate=False,
                     path_data=None, path_X=None,
                     config_description_2d=None, config_occ=None):
       import tofu.imas2tofu as _tfimas
       _tfimas._save_to_imas(self, tfversion=__version__,
                             shot=shot, run=run, refshot=refshot,
                             refrun=refrun, user=user, database=database,
                             version=version, occ=occ, dryrun=dryrun, verb=verb,
                             ids=ids, deep=deep,
                             restore_size=restore_size,
                             forceupdate=forceupdate,
                             path_data=path_data, path_X=path_X,
                             config_description_2d=config_description_2d,
                             config_occ=config_occ)


    #----------------------------
    # Operator overloading section

    @staticmethod
    def _extract_common_params(obj0, obj1=None):
        if obj1 is None:
            Id = obj0.Id.copy()
            Id._dall['Name'] += 'modified'
            dcom = {'Id':Id,
                    'dchans':obj0._dchans, 'dlabels':obj0.dlabels,
                    't':obj0.t, 'X':obj0.X,
                    'lCam':obj0.lCam, 'config':obj0.config,
                    'dextra':obj0.dextra}
            if dcom['lCam'] is not None:
                dcom['config'] = None
        else:
            ls = ['SavePath', 'Diag', 'Exp', 'shot']
            dcom = {ss:getattr(obj0.Id,ss) for ss in ls
                    if getattr(obj0.Id,ss) == getattr(obj1.Id,ss)}
            if obj0._dchans == obj1._dchans:
                dcom['dchans'] = obj0._dchans
            if obj0.dlabels == obj1.dlabels:
                dcom['dlabels'] = obj0.dlabels
            if obj0.dextra == obj1.dextra:
                dcom['dextra'] = obj0.dextra
            if np.allclose(obj0.t, obj1.t):
                dcom['t'] = obj0.t
            if np.allclose(obj0.X, obj1.X):
                dcom['X'] = obj0.X
            if obj0.lCam is not None and obj1.lCam is not None:
                if all([c0 == c1 for c0, c1 in zip(obj0.lCam, obj1.lCam)]):
                    dcom['lCam'] = obj0.lCam
            if obj0.config == obj1.config:
                dcom['config'] = obj0.config
        return dcom

    @staticmethod
    def _recreatefromoperator(d0, other, opfunc):

        if other is None:
            data = opfunc(d0.data)
            dcom = d0._extract_common_params(d0)

        elif isinstance(other, (int, float, np.integer)):
            data = opfunc(d0.data, other)
            dcom = d0._extract_common_params(d0)

        elif isinstance(other, np.ndarray):
            data = opfunc(d0.data, other)
            dcom = d0._extract_common_params(d0)

        elif issubclass(other.__class__, DataAbstract):
            if other.__class__.__name__ != d0.__class__.__name__:
                msg = 'Operator overloaded only for same-class instances:\n'
                msg += "    - provided: %s and %s"%(d0.__class__.__name__,
                                                    other.__class__.__name__)
                raise Exception(msg)
            try:
                data = opfunc(d0.data, other.data)
            except Exception as err:
                msg = str(err)
                msg += "\n\ndata shapes not matching !"
                raise Exception(msg)

            dcom = d0._extract_common_params(d0, other)
        else:
            msg = "Behaviour not implemented !"
            raise NotImplementedError(msg)

        return d0.__class__(data=data, Name='New', **dcom)


    def __abs__(self):
        opfunc = lambda x: np.abs(x)
        data = self._recreatefromoperator(self, None, opfunc)
        return data

    def __sub__(self, other):
        opfunc = lambda x, y: x-y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __rsub__(self, other):
        opfunc = lambda x, y: x-y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __add__(self, other):
        opfunc = lambda x, y: x+y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __radd__(self, other):
        opfunc = lambda x, y: x+y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __mul__(self, other):
        opfunc = lambda x, y: x*y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __rmul__(self, other):
        opfunc = lambda x, y: x*y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __truediv__(self, other):
        opfunc = lambda x, y: x/y
        data = self._recreatefromoperator(self, other, opfunc)
        return data

    def __pow__(self, other):
        opfunc = lambda x, y: x**y
        data = self._recreatefromoperator(self, other, opfunc)
        return data












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
# DataCam1D.__signature__ = sig.replace(parameters=lp)

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
# DataCam1D.__signature__ = sig.replace(parameters=lp)


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
            msg = ("dX12 must be either:\n"
                   + "\t- None\n"
                   + "\t- 'geom' : will be derived from the cam geometry\n"
                   + "\t- dict : containing {'x1'  : array of coords.,\n"
                   + "\t                     'x2'  : array of coords.,\n"
                   + "\t                     'ind1': array of int indices,\n"
                   + "\t                     'ind2': array of int indices}")
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
# DataCam2D.__signature__ = sig.replace(parameters=lp)



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
# DataCam2D.__signature__ = sig.replace(parameters=lp)


# ####################################################################
# ####################################################################
#               Plasma2D
# ####################################################################
# ####################################################################


# class Plasma2D(utils.ToFuObject):
    # """ A generic class for handling 2D (and 1D) plasma profiles

    # Provides:
        # - equilibrium-related quantities
        # - any 1d profile (can be remapped on 2D equilibrium)
        # - spatial interpolation methods

    # """
    # # Fixed (class-wise) dictionary of default properties
    # _ddef = {'Id': {'include': ['Mod', 'Cls', 'Exp', 'Diag',
                                # 'Name', 'shot', 'version']},
             # 'dtreat': {'order': ['mask', 'interp-indt', 'interp-indch',
                                  # 'data0', 'dfit',
                                  # 'indt', 'indch', 'indlamb', 'interp-t']}}

    # def __init_subclass__(cls, **kwdargs):
        # # Does not exist before Python 3.6 !!!
        # # Python 2
        # super(Plasma2D, cls).__init_subclass__(**kwdargs)
        # # Python 3
        # # super().__init_subclass__(**kwdargs)
        # cls._ddef = copy.deepcopy(Plasma2D._ddef)
        # # cls._dplot = copy.deepcopy(Struct._dplot)
        # # cls._set_color_ddef(cls._color)


    # def __init__(self, dtime=None, dradius=None, d0d=None, d1d=None,
                 # d2d=None, dmesh=None, config=None,
                 # Id=None, Name=None, Exp=None, shot=None,
                 # fromdict=None, sep=None, SavePath=os.path.abspath('./'),
                 # SavePath_Include=tfpf.defInclude):

        # # Create a dplot at instance level
        # #self._dplot = copy.deepcopy(self.__class__._dplot)

        # kwdargs = locals()
        # del kwdargs['self']
        # # super()
        # super(Plasma2D,self).__init__(**kwdargs)

    # def _reset(self):
        # # super()
        # super(Plasma2D,self)._reset()
        # self._dgroup = dict.fromkeys(self._get_keys_dgroup())
        # self._dindref = dict.fromkeys(self._get_keys_dindref())
        # self._ddata = dict.fromkeys(self._get_keys_ddata())
        # self._dgeom = dict.fromkeys(self._get_keys_dgeom())

    # @classmethod
    # def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               # Exp=None, shot=None,
                               # include=None, **kwdargs):
        # if Id is not None:
            # assert isinstance(Id,utils.ID)
            # Name, Exp, shot = Id.Name, Id.Exp, Id.shot
        # assert type(Name) is str, Name
        # assert type(Exp) is str, Exp
        # if include is None:
            # include = cls._ddef['Id']['include']
        # assert shot is None or type(shot) in [int,np.int64]
        # if shot is None:
            # if 'shot' in include:
                # include.remove('shot')
        # else:
            # shot = int(shot)
            # if 'shot' not in include:
                # include.append('shot')
        # kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot,
                        # 'include':include})
        # return kwdargs

    # ###########
    # # Get largs
    # ###########

    # @staticmethod
    # def _get_largs_dindrefdatagroup():
        # largs = ['dtime', 'dradius', 'dmesh', 'd0d', 'd1d', 'd2d']
        # return largs

    # @staticmethod
    # def _get_largs_dgeom():
        # largs = ['config']
        # return largs

    # ###########
    # # Get check and format inputs
    # ###########

    # #---------------------
    # # Methods for checking and formatting inputs
    # #---------------------

    # @staticmethod
    # def _extract_dnd(dnd, k0,
                     # dim_=None, quant_=None, name_=None,
                     # origin_=None, units_=None):
        # # Set defaults
        # dim_ =   k0 if dim_ is None else dim_
        # quant_ = k0 if quant_ is None else quant_
        # name_ =  k0 if name_ is None else name_
        # origin_ = 'unknown' if origin_ is None else origin_
        # units_ = 'a.u.' if units_ is None else units_

        # # Extrac
        # dim = dnd[k0].get('dim', None)
        # if dim is None:
            # dim = dim_
        # quant = dnd[k0].get('quant', None)
        # if quant is None:
            # quant = quant_
        # origin = dnd[k0].get('origin', None)
        # if origin is None:
            # origin = origin_
        # name = dnd[k0].get('name', None)
        # if name is None:
            # name = name_
        # units = dnd[k0].get('units', None)
        # if units is None:
            # units = units_
        # return dim, quant, origin, name, units

    # @staticmethod
    # def _checkformat_dtrm(dtime=None, dradius=None, dmesh=None,
                          # d0d=None, d1d=None, d2d=None):

        # dd = {'dtime':dtime, 'dradius':dradius, 'dmesh':dmesh,
              # 'd0d':d0d, 'd1d':d1d, 'd2d':d2d}

        # # Define allowed keys for each dict
        # lkok = ['data', 'dim', 'quant', 'name', 'origin', 'units',
                # 'depend']
        # lkmeshmax = ['type', 'ftype', 'nodes', 'faces', 'R', 'Z', 'shapeRZ',
                     # 'nfaces', 'nnodes', 'mpltri', 'size', 'ntri']
        # lkmeshmin = ['type', 'ftype']
        # dkok = {'dtime': {'max':lkok, 'min':['data'], 'ndim':[1]},
                # 'dradius':{'max':lkok, 'min':['data'], 'ndim':[1,2]},
                # 'd0d':{'max':lkok, 'min':['data'], 'ndim':[1,2,3]},
                # 'd1d':{'max':lkok, 'min':['data'], 'ndim':[1,2]},
                # 'd2d':{'max':lkok, 'min':['data'], 'ndim':[1,2]}}
        # dkok['dmesh'] = {'max':lkok + lkmeshmax, 'min':lkmeshmin}

        # # Check each dict independently
        # for dk, dv in dd.items():
            # if dv is None or len(dv) == 0:
                # dd[dk] = {}
                # continue
            # c0 = type(dv) is not dict or any([type(k0) is not str
                                              # for k0 in dv.keys()])
            # c0 = any([type(k0) is not str or type(v0) is not dict
                      # for k0, v0 in dv.items()])
            # if c0:
                # msg = "Arg %s must be a dict with:\n"
                # msg += "    - (key, values) of type (str, dict)"
                # raise Exception(msg)

            # for k0, v0 in  dv.items():
                # c0 = any([k1 not in dkok[dk]['max'] for k1 in v0.keys()])
                # c0 = c0 or any([v0.get(k1,None) is None
                                # for k1 in dkok[dk]['min']])
                # if c0:
                    # msg = "Arg %s[%s] must be a dict with keys in:\n"%(dk,k0)
                    # msg += "    - %s\n"%str(dkok[dk]['max'])
                    # msg += "And with at least the following keys:\n"
                    # msg += "    - %s\n"%str(dkok[dk]['min'])
                    # msg += "Provided:\n"
                    # msg += "    - %s\n"%str(v0.keys())
                    # msg += "Missing:\n"
                    # msg += "    - %s\n"%str(set(dkok[dk]['min']).difference(v0.keys()))
                    # msg += "Non-valid:\n"
                    # msg += "    - %s"%str(set(v0.keys()).difference(dkok[dk]['max']))
                    # raise Exception(msg)
                # if 'data' in dkok[dk]['min']:
                    # dd[dk][k0]['data'] = np.atleast_1d(np.squeeze(v0['data']))
                    # if dd[dk][k0]['data'].ndim not in dkok[dk]['ndim']:
                        # msg = "%s[%s]['data'] has wrong dimensions:\n"%(dk,k0)
                        # msg += "    - Expected: %s\n"%str(dkok[dk]['ndim'])
                        # msg += "    - Provided: %s"%str(dd[dk][k0]['data'].ndim)
                        # raise Exception(msg)

                # # mesh
                # if dk == 'dmesh':

                    # lmok = ['rect', 'tri', 'quadtri']
                    # if v0['type'] not in lmok:
                        # msg = ("Mesh['type'] should be in {}\n".format(lmok)
                               # + "\t- Provided: {}".format(v0['type']))
                        # raise Exception(msg)

                    # if v0['type'] == 'rect':
                        # c0 = all([ss in v0.keys() and v0[ss].ndim in [1, 2]
                                  # for ss in ['R', 'Z']])
                        # if not c0:
                            # msg = ("A mesh of type 'rect' must have attr.:\n"
                                   # + "\t- R of dim in [1, 2]\n"
                                   # + "\t- Z of dim in [1, 2]")
                            # raise Exception(msg)
                        # shapeu = np.unique(np.r_[v0['R'].shape, v0['Z'].shape])

                        # shapeRZ = v0['shapeRZ']
                        # if shapeRZ is None:
                            # shapeRZ = [None, None]
                        # else:
                            # shapeRZ = list(shapeRZ)
                        # if v0['R'].ndim == 1:
                            # if np.any(np.diff(v0['R']) <= 0.):
                                # msg = "Non-increasing R"
                                # raise Exception(msg)
                            # R = v0['R']
                        # else:
                            # lc = [np.all(np.diff(v0['R'][0, :])) > 0.,
                                  # np.all(np.diff(v0['R'][:, 0])) > 0.]
                            # if np.sum(lc) != 1:
                                # msg = "Impossible to know R dimension!"
                                # raise Exception(msg)
                            # if lc[0]:
                                # R = v0['R'][0, :]
                                # if shapeRZ[1] is None:
                                    # shapeRZ[1] = 'R'
                                # if shapeRZ[1] != 'R':
                                    # msg = "Inconsistent shapeRZ"
                                    # raise Exception(msg)
                            # else:
                                # R = v0['R'][:, 0]
                                # if shapeRZ[0] is None:
                                    # shapeRZ[0] = 'R'
                                # if shapeRZ[0] != 'R':
                                    # msg = "Inconsistent shapeRZ"
                                    # raise Exception(msg)
                        # if v0['Z'].ndim == 1:
                            # if np.any(np.diff(v0['Z']) <= 0.):
                                # msg = "Non-increasing Z"
                                # raise Exception(msg)
                            # Z = v0['Z']
                        # else:
                            # lc = [np.all(np.diff(v0['Z'][0, :])) > 0.,
                                  # np.all(np.diff(v0['Z'][:, 0])) > 0.]
                            # if np.sum(lc) != 1:
                                # msg = "Impossible to know R dimension!"
                                # raise Exception(msg)
                            # if lc[0]:
                                # Z = v0['Z'][0, :]
                                # if shapeRZ[1] is None:
                                    # shapeRZ[1] = 'Z'
                                # if shapeRZ[1] != 'Z':
                                    # msg = "Inconsistent shapeRZ"
                                    # raise Exception(msg)
                            # else:
                                # Z = v0['Z'][:, 0]
                                # if shapeRZ[0] is None:
                                    # shapeRZ[0] = 'Z'
                                # if shapeRZ[0] != 'Z':
                                    # msg = "Inconsistent shapeRZ"
                                    # raise Exception(msg)
                        # shapeRZ = tuple(shapeRZ)
                        # if shapeRZ not in [('R', 'Z'), ('Z', 'R')]:
                            # msg = "Inconsistent shapeRZ"
                            # raise Exception(msg)

                        # if None in shapeRZ:
                            # msg = ("Please provide shapeRZ "
                                   # + " = ('R', 'Z') or ('Z', 'R')\n"
                                   # + "Could not be inferred from data itself")
                            # raise Exception(msg)

                        # def trifind(r, z,
                                    # Rbin=0.5*(R[1:] + R[:-1]),
                                    # Zbin=0.5*(Z[1:] + Z[:-1]),
                                    # nR=R.size, nZ=Z.size,
                                    # shapeRZ=shapeRZ):
                            # indR = np.searchsorted(Rbin, r)
                            # indZ = np.searchsorted(Zbin, z)
                            # if shapeRZ == ('R', 'Z'):
                                # indpts = indR*nZ + indZ
                            # else:
                                # indpts = indZ*nR + indR
                            # indout = ((r < R[0]) | (r > R[-1])
                                      # | (z < Z[0]) | (z > Z[-1]))
                            # indpts[indout] = -1
                            # return indpts

                        # dd[dk][k0]['R'] = R
                        # dd[dk][k0]['Z'] = Z
                        # dd[dk][k0]['shapeRZ'] = shapeRZ
                        # dd[dk][k0]['nR'] = R.size
                        # dd[dk][k0]['nZ'] = Z.size
                        # dd[dk][k0]['trifind'] = trifind
                        # if dd[dk][k0]['ftype'] != 0:
                            # msg = "Linear interpolation not handled yet !"
                            # raise Exception(msg)
                        # dd[dk][k0]['size'] = R.size*Z.size

                    # else:
                        # ls = ['nodes', 'faces']
                        # if not all([s in v0.keys() for s in ls]):
                            # msg = ("The following keys should be in dmesh:\n"
                                   # + "\t- {}".format(ls))
                            # raise Exception(msg)
                        # func = np.atleast_2d
                        # dd[dk][k0]['nodes'] = func(v0['nodes']).astype(float)
                        # dd[dk][k0]['faces'] = func(v0['faces']).astype(int)
                        # nnodes = dd[dk][k0]['nodes'].shape[0]
                        # nfaces = dd[dk][k0]['faces'].shape[0]

                        # # Test for duplicates
                        # nodesu = np.unique(dd[dk][k0]['nodes'], axis=0)
                        # facesu = np.unique(dd[dk][k0]['faces'], axis=0)
                        # lc = [nodesu.shape[0] != nnodes,
                              # facesu.shape[0] != nfaces]
                        # if any(lc):
                            # msg = "Non-valid mesh {0}[{1}]: \n".format(dk, k0)
                            # if lc[0]:
                                # ndup = nnodes - nodesu.shape[0]
                                # ndsh = dd[dk][k0]['nodes'].shape
                                # undsh = nodesu.shape
                                # msg += (
                                    # "  Duplicate nodes: {}\n".format(ndup)
                                    # + "\t- nodes.shape: {}\n".format(ndsh)
                                    # + "\t- unique shape: {}\n".format(undsh))
                            # if lc[1]:
                                # ndup = str(nfaces - facesu.shape[0])
                                # facsh = str(dd[dk][k0]['faces'].shape)
                                # ufacsh = str(facesu.shape)
                                # msg += (
                                    # "  Duplicate faces: {}\n".format(ndup)
                                    # + "\t- faces.shape: {}\n".format(facsh)
                                    # + "\t- unique shape: {}".format(ufacsh))
                            # raise Exception(msg)

                        # # Test for unused nodes
                        # facesu = np.unique(facesu)
                        # c0 = np.all(facesu >= 0) and facesu.size == nnodes
                        # if not c0:
                            # ino = str([ii for ii in range(0, nnodes)
                                       # if ii not in facesu])
                            # msg = "Unused nodes in {0}[{1}]:\n".format(dk, k0)
                            # msg += "    - unused nodes indices: {}".format(ino)
                            # warnings.warn(msg)

                        # dd[dk][k0]['nnodes'] = dd[dk][k0].get('nnodes', nnodes)
                        # dd[dk][k0]['nfaces'] = dd[dk][k0].get('nfaces', nfaces)

                        # assert dd[dk][k0]['nodes'].shape == (v0['nnodes'], 2)
                        # assert np.max(dd[dk][k0]['faces']) < v0['nnodes']
                        # # Only triangular meshes so far
                        # assert v0['type'] in ['tri', 'quadtri'], v0['type']

                        # if 'tri' in v0['type']:
                            # fshap = dd[dk][k0]['faces'].shape
                            # fshap0 = (v0['nfaces'], 3)
                            # if fshap != fshap0:
                                # msg = ("Wrong shape of {}[{}]\n".format(dk, k0)
                                       # + "\t- Expected: {}\n".format(fshap0)
                                       # + "\t- Provided: {}".format(fshap))
                                # raise Exception(msg)
                            # if v0.get('mpltri', None) is None:
                                # dd[dk][k0]['mpltri'] = mplTri(
                                    # dd[dk][k0]['nodes'][:, 0],
                                    # dd[dk][k0]['nodes'][:, 1],
                                    # dd[dk][k0]['faces'])
                            # assert isinstance(dd[dk][k0]['mpltri'], mplTri)
                            # assert dd[dk][k0]['ftype'] in [0, 1]
                            # ntri = dd[dk][k0]['ntri']
                            # if dd[dk][k0]['ftype'] == 1:
                                # dd[dk][k0]['size'] = dd[dk][k0]['nnodes']
                            # else:
                                # dd[dk][k0]['size'] = int(
                                    # dd[dk][k0]['nfaces'] / ntri
                                # )

        # # Check unicity of all keys
        # lk = [list(dv.keys()) for dv in dd.values()]
        # lk = list(itt.chain.from_iterable(lk))
        # lku = sorted(set(lk))
        # lk = ['{0} : {1} times'.format(kk, str(lk.count(kk)))
              # for kk in lku if lk.count(kk) > 1]
        # if len(lk) > 0:
            # msg = ("Each key of (dtime, dradius, dmesh, d0d, d1d, d2d)"
                   # + " must be unique !\n"
                   # + "The following keys are repeated :\n"
                   # + "    - " + "\n    - ".join(lk))
            # raise Exception(msg)

        # dtime, dradius, dmesh = dd['dtime'], dd['dradius'], dd['dmesh']
        # d0d, d1d, d2d = dd['d0d'], dd['d1d'], dd['d2d']

        # return dtime, dradius, dmesh, d0d, d1d, d2d


    # def _checkformat_inputs_dgeom(self, config=None):
        # if config is not None:
            # assert issubclass(config.__class__, utils.ToFuObject)
        # return config

    # ###########
    # # Get keys of dictionnaries
    # ###########

    # @staticmethod
    # def _get_keys_dgroup():
        # lk = ['time', 'radius', 'mesh']
        # return lk

    # @staticmethod
    # def _get_keys_dindref():
        # lk = []
        # return lk

    # @staticmethod
    # def _get_keys_ddata():
        # lk = []
        # return lk

    # @staticmethod
    # def _get_keys_dgeom():
        # lk = ['config']
        # return lk


    # ###########
    # # _init
    # ###########

    # def _init(self, dtime=None, dradius=None, dmesh=None,
              # d0d=None, d1d=None, d2d=None,
              # config=None, **kwargs):
        # kwdargs = locals()
        # kwdargs.update(**kwargs)
        # largs = self._get_largs_dindrefdatagroup()
        # kwdindrefdatagroup = self._extract_kwdargs(kwdargs, largs)
        # largs = self._get_largs_dgeom()
        # kwdgeom = self._extract_kwdargs(kwdargs, largs)
        # self._set_dindrefdatagroup(**kwdindrefdatagroup)
        # self.set_dgeom(**kwdgeom)
        # self._dstrip['strip'] = 0


    # ###########
    # # set dictionaries
    # ###########

    # @staticmethod
    # def _find_lref(shape=None, k0=None, dd=None, ddstr=None,
                   # dindref=None, lrefname=['t','radius']):
        # if 'depend' in dd[k0].keys():
            # lref = dd[k0]['depend']
        # else:
            # lref = [[kk for kk, vv in dindref.items()
                     # if vv['size'] == sh and vv['group'] in lrefname]
                    # for sh in shape]
            # lref = list(itt.chain.from_iterable(lref))
            # if len(lref) < len(shape):
                # msg = "Maybe not enoough references for %s[%s]:\n"%(ddstr,k0)
                # msg += "    - shape: %s\n"%str(shape)
                # msg += "    - lref:  %s"%str(lref)
                # warnings.warn(msg)

        # if len(lref) > len(shape):
            # msg = "Too many references for %s[%s]:\n"%(ddstr,k0)
            # msg += "    - shape: %s\n"%str(shape)
            # msg += "    - lref:  %s"%str(lref)
            # raise Exception(msg)
        # return lref


    # def _set_dindrefdatagroup(self, dtime=None, dradius=None, dmesh=None,
                              # d0d=None, d1d=None, d2d=None):

        # # Check dtime is not None
        # out = self._checkformat_dtrm(dtime=dtime, dradius=dradius, dmesh=dmesh,
                                     # d0d=d0d, d1d=d1d, d2d=d2d)
        # dtime, dradius, dmesh, d0d, d1d, d2d = out

        # dgroup, dindref, ddata = {}, {}, {}
        # empty = {}
        # # Get indt
        # if dtime is not None:
            # for k0 in dtime.keys():
                # out = self._extract_dnd(dtime,k0,
                                        # dim_='time', quant_='t',
                                        # name_=k0, units_='s')
                # dim, quant, origin, name, units = out

                # assert k0 not in dindref.keys()
                # dtime[k0]['data'] = np.atleast_1d(np.squeeze(dtime[k0]['data']))
                # assert dtime[k0]['data'].ndim == 1

                # dindref[k0] = {'size':dtime[k0]['data'].size,
                               # 'group':'time'}

                # assert k0 not in ddata.keys()
                # ddata[k0] = {'data':dtime[k0]['data'],
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'origin':origin, 'units':units, 'depend':(k0,)}

        # # d0d
        # if d0d is not None:
            # for k0 in d0d.keys():
                # out = self._extract_dnd(d0d,k0)
                # dim, quant, origin, name, units = out

                # # data
                # d0d[k0]['data'] = np.atleast_1d(np.squeeze(d0d[k0]['data']))
                # assert d0d[k0]['data'].ndim >= 1

                # depend = self._find_lref(d0d[k0]['data'].shape, k0, dd=d0d,
                                         # ddstr='d0d', dindref=dindref,
                                         # lrefname=['t'])
                # assert len(depend) == 1 and dindref[depend[0]]['group']=='time'
                # assert k0 not in ddata.keys()
                # ddata[k0] = {'data':d0d[k0]['data'],
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'units':units, 'origin':origin, 'depend':depend}

        # # get radius
        # if dradius is not None:
            # for k0 in dradius.keys():
                # out = self._extract_dnd(dradius, k0, name_=k0)
                # dim, quant, origin, name, units = out
                # assert k0 not in dindref.keys()
                # data = np.atleast_1d(np.squeeze(dradius[k0]['data']))
                # assert data.ndim in [1,2]

                # if len(dradius[k0].get('depend',[1])) == 1:
                    # assert data.ndim == 1
                    # size = data.size
                # else:
                    # lkt = [k for k in dtime.keys() if k in dradius[k0]['depend']]
                    # assert len(lkt) == 1
                    # axist = dradius[k0]['depend'].index(lkt[0])
                    # # Handle cases with only 1 time step
                    # if data.ndim == 1:
                        # assert dindref[lkt[0]]['size'] == 1
                        # data = data.reshape((1, data.size))
                    # size = data.shape[1-axist]
                # dindref[k0] = {'size':size,
                               # 'group':'radius'}

                # assert k0 not in ddata.keys()
                # depend = self._find_lref(data.shape, k0, dd=dradius,
                                         # ddstr='dradius', dindref=dindref,
                                         # lrefname=['t','radius'])
                # ddata[k0] = {'data':data,
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'origin':origin, 'units':units, 'depend':depend}


        # # Get d1d
        # if d1d is not None:
            # for k0 in d1d.keys():
                # out = self._extract_dnd(d1d,k0)
                # dim, quant, origin, name, units = out

                # d1d[k0]['data'] = np.atleast_2d(np.squeeze(d1d[k0]['data']))
                # assert d1d[k0]['data'].ndim == 2

                # # data
                # depend = self._find_lref(d1d[k0]['data'].shape, k0, dd=d1d,
                                         # ddstr='d1d', dindref=dindref,
                                         # lrefname=['t','radius'])
                # assert k0 not in ddata.keys()
                # ddata[k0] = {'data':d1d[k0]['data'],
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'units':units, 'origin':origin, 'depend':depend}

        # # dmesh ref
        # if dmesh is not None:
            # for k0 in dmesh.keys():
                # out = self._extract_dnd(dmesh, k0, dim_='mesh')
                # dim, quant, origin, name, units = out

                # assert k0 not in dindref.keys()
                # dindref[k0] = {'size':dmesh[k0]['size'],
                               # 'group':'mesh'}

                # assert k0 not in ddata.keys()
                # ddata[k0] = {'data':dmesh[k0],
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'units':units, 'origin':origin, 'depend':(k0,)}

        # # d2d
        # if d2d is not None:
            # for k0 in d2d.keys():
                # out = self._extract_dnd(d2d,k0)
                # dim, quant, origin, name, units = out

                # d2d[k0]['data'] = np.atleast_2d(np.squeeze(d2d[k0]['data']))
                # assert d2d[k0]['data'].ndim == 2

                # depend = self._find_lref(d2d[k0]['data'].shape, k0, dd=d2d,
                                         # ddstr='d2d', dindref=dindref,
                                         # lrefname=['t','mesh'])
                # assert k0 not in ddata.keys()
                # ddata[k0] = {'data':d2d[k0]['data'],
                             # 'dim':dim, 'quant':quant, 'name':name,
                             # 'units':units, 'origin':origin, 'depend':depend}

        # # dgroup
        # dgroup = {}
        # if len(dtime) > 0:
            # dgroup['time'] = {'dref': list(dtime.keys())[0]}
        # if len(dradius) > 0:
            # dgroup['radius'] = {'dref': list(dradius.keys())[0]}
        # if len(dmesh) > 0:
            # dgroup['mesh'] = {'dref': list(dmesh.keys())[0]}

        # # Update dict
        # self._dgroup = dgroup
        # self._dindref = dindref
        # self._ddata = ddata
        # # Complement
        # self._complement()

    # def _complement(self):

        # # --------------
        # # ddata
        # for k0, v0 in self.ddata.items():
            # lindout = [ii for ii in v0['depend'] if ii not in self.dindref.keys()]
            # if not len(lindout) == 0:
                # msg = ("ddata[{}]['depend'] keys not in dindref:\n".format(k0)
                       # + "    - " + "\n    - ".join(lindout))
                # raise Exception(msg)

            # self.ddata[k0]['lgroup'] = [self.dindref[ii]['group']
                                        # for ii in v0['depend']]
            # type_ = type(v0['data'])
            # shape = tuple([self.dindref[ii]['size'] for ii in v0['depend']])

            # # if only one dim => mesh or iterable or unspecified
            # if len(shape) == 1 or type_ is dict:
                # c0 = type_ is dict and 'mesh' in self.ddata[k0]['lgroup']
                # c1 = not c0 and len(v0['data']) == shape[0]
                # if not (c0 or c1):
                    # msg = ("Signal {}['data'] should be either:\n".format(k0)
                           # + "\t- dict: a mesh\n"
                           # + "\t- iterable of len() = "
                           # + "{} (shape[0] of ref)\n".format(shape[0])
                           # + "  You provided:\n"
                           # + "\t- type: {}\n".format(type_)
                           # + "\t- len(): {}\n".format(len(v0['data']))
                           # + "\t- {}['data']: {}".format(k0, v0['data']))
                    # raise Exception(msg)
            # else:
                # assert type(v0['data']) is np.ndarray
                # assert v0['data'].shape == shape

        # # --------------
        # # dindref
        # for k0 in self.dindref.keys():
            # self.dindref[k0]['ldata'] = [kk for kk, vv in self.ddata.items()
                                    # if k0 in vv['depend']]
            # assert self.dindref[k0]['group'] in self.dgroup.keys()

        # # --------------
        # # dgroup
        # for gg, vg in self.dgroup.items():
            # lindref = [id_ for id_,vv in self.dindref.items()
                       # if vv['group'] == gg]
            # ldata = [id_ for id_ in self.ddata.keys()
                     # if any([id_ in self.dindref[vref]['ldata']
                             # for vref in lindref])]
            # #assert vg['depend'] in lidindref
            # self.dgroup[gg]['lindref'] = lindref
            # self.dgroup[gg]['ldata'] = ldata


    # def set_dgeom(self, config=None):
        # config = self._checkformat_inputs_dgeom(config=config)
        # self._dgeom = {'config':config}


    # ###########
    # # strip dictionaries
    # ###########

    # def _strip_ddata(self, strip=0):
        # pass


    # def _strip_dgeom(self, strip=0, force=False, verb=True):
        # if self._dstrip['strip']==strip:
            # return

        # if strip in [0] and self._dstrip['strip'] in [1]:
            # config = None
            # if self._dgeom['config'] is not None:
                # assert type(self._dgeom['config']) is str
                # config = utils.load(self._dgeom['config'], verb=verb)

            # self._set_dgeom(config=config)

        # elif strip in [1] and self._dstrip['strip'] in [0]:
            # if self._dgeom['config'] is not None:
                # path = self._dgeom['config'].Id.SavePath
                # name = self._dgeom['config'].Id.SaveName
                # pfe = os.path.join(path, name+'.npz')
                # lf = os.listdir(path)
                # lf = [ff for ff in lf if name+'.npz' in ff]
                # exist = len(lf)==1
                # if not exist:
                    # msg = """BEWARE:
                        # You are about to delete the config object
                        # Only the path/name to saved a object will be kept

                        # But it appears that the following object has no
                        # saved file where specified (obj.Id.SavePath)
                        # Thus it won't be possible to retrieve it
                        # (unless available in the current console:"""
                    # msg += "\n    - {0}".format(pfe)
                    # if force:
                        # warnings.warn(msg)
                    # else:
                        # raise Exception(msg)
                # self._dgeom['config'] = pfe

    # ###########
    # # _strip and get/from dict
    # ###########

    # @classmethod
    # def _strip_init(cls):
        # cls._dstrip['allowed'] = [0,1]
        # nMax = max(cls._dstrip['allowed'])
        # doc = """
                 # 1: dgeom pathfiles
                 # """
        # doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        # cls.strip.__doc__ = doc

    # def strip(self, strip=0, verb=True):
        # # super()
        # super(Plasma2D,self).strip(strip=strip, verb=verb)

    # def _strip(self, strip=0, verb=True):
        # self._strip_dgeom(strip=strip, verb=verb)

    # def _to_dict(self):
        # dout = {'dgroup':{'dict':self._dgroup, 'lexcept':None},
                # 'dindref':{'dict':self._dindref, 'lexcept':None},
                # 'ddata':{'dict':self._ddata, 'lexcept':None},
                # 'dgeom':{'dict':self._dgeom, 'lexcept':None}}
        # return dout

    # def _from_dict(self, fd):
        # self._dgroup.update(**fd['dgroup'])
        # self._dindref.update(**fd['dindref'])
        # self._ddata.update(**fd['ddata'])
        # self._dgeom.update(**fd['dgeom'])


    # ###########
    # # properties
    # ###########

    # @property
    # def dgroup(self):
        # return self._dgroup
    # @property
    # def dindref(self):
        # return self._dindref
    # @property
    # def ddata(self):
        # return self._ddata
    # @property
    # def dtime(self):
        # return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     # if vv['group'] == 'time'])
    # @property
    # def dradius(self):
        # return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     # if vv['group'] == 'radius'])
    # @property
    # def dmesh(self):
        # return dict([(kk, self._ddata[kk]) for kk,vv in self._dindref.items()
                     # if vv['group'] == 'mesh'])
    # @property
    # def config(self):
        # return self._dgeom['config']

    # #---------------------
    # # Read-only for internal use
    # #---------------------

    # @property
    # def _lquantboth(self):
        # """ Return list of quantities available both in 1d and 2d """
        # lq1 = [self._ddata[vd]['quant'] for vd in self._dgroup['radius']['ldata']]
        # lq2 = [self._ddata[vd]['quant'] for vd in self._dgroup['mesh']['ldata']]
        # lq = list(set(lq1).intersection(lq2))
        # return lq

    # def _get_ldata(self, dim=None, quant=None, name=None,
                   # units=None, origin=None,
                   # indref=None, group=None, log='all', return_key=True):
        # assert log in ['all','any','raw']
        # lid = np.array(list(self._ddata.keys()))
        # ind = np.ones((7,len(lid)),dtype=bool)
        # if dim is not None:
            # ind[0,:] = [self._ddata[id_]['dim'] == dim for id_ in lid]
        # if quant is not None:
            # ind[1,:] = [self._ddata[id_]['quant'] == quant for id_ in lid]
        # if name is not None:
            # ind[2,:] = [self._ddata[id_]['name'] == name for id_ in lid]
        # if units is not None:
            # ind[3,:] = [self._ddata[id_]['units'] == units for id_ in lid]
        # if origin is not None:
            # ind[4,:] = [self._ddata[id_]['origin'] == origin for id_ in lid]
        # if indref is not None:
            # ind[5,:] = [depend in self._ddata[id_]['depend'] for id_ in lid]
        # if group is not None:
            # ind[6,:] = [group in self._ddata[id_]['lgroup'] for id_ in lid]

        # if log == 'all':
            # ind = np.all(ind, axis=0)
        # elif log == 'any':
            # ind = np.any(ind, axis=0)

        # if return_key:
            # if np.any(ind):
                # out = lid[ind.nonzero()[0]]
            # else:
                # out = np.array([],dtype=int)
        # else:
            # out = ind, lid
        # return out

    # def _get_keyingroup(self, key, group=None, msgstr=None, raise_=False):

        # if key in self._ddata.keys():
            # lg = self._ddata[key]['lgroup']
            # if group is None or group in lg:
                # return key, None
            # else:
                # msg = ("Required data key does not have matching group:\n"
                       # + "\t- ddata[{}]['lgroup'] = {}\n".format(key, lg)
                       # + "\t- Expected group:  {}".format(group))
                # if raise_:
                    # raise Exception(msg)

        # ind, akeys = self._get_ldata(dim=key, quant=key, name=key, units=key,
                                     # origin=key, group=group, log='raw',
                                     # return_key=False)
        # # Remove indref and group
        # ind = ind[:5,:] & ind[-1,:]

        # # Any perfect match ?
        # nind = np.sum(ind, axis=1)
        # sol = (nind == 1).nonzero()[0]
        # key, msg = None, None
        # if sol.size > 0:
            # if np.unique(sol).size == 1:
                # indkey = ind[sol[0],:].nonzero()[0]
                # key = akeys[indkey][0]
            # else:
                # lstr = "[dim,quant,name,units,origin]"
                # msg = "Several possible matches in {} for {}".format(lstr, key)
        # else:
            # lstr = "[dim,quant,name,units,origin]"
            # msg = "No match in {} for {} in group {}".format(lstr, key, group)

        # if msg is not None:
            # msg += "\n\nRequested {} could not be identified!\n".format(msgstr)
            # msg += "Please provide a valid (unique) key/name/quant/dim:\n\n"
            # msg += self.get_summary(verb=False, return_='msg')
            # if raise_:
                # raise Exception(msg)
        # return key, msg

    # #---------------------
    # # Methods for showing data
    # #---------------------

    # def get_summary(
        # self,
        # sep='  ',
        # line='-',
        # just='l',
        # table_sep=None,
        # verb=True,
        # return_=False,
    # ):
        # """ Summary description of the object content """
        # # # Make sure the data is accessible
        # # msg = "The data is not accessible because self.strip(2) was used !"
        # # assert self._dstrip['strip']<2, msg

        # # -----------------------
        # # Build for ddata
        # col0 = ['group key', 'nb. indref']
        # ar0 = [(k0, len(v0['lindref'])) for k0,v0 in self._dgroup.items()]

        # # -----------------------
        # # Build for ddata
        # col1 = ['indref key', 'group', 'size']
        # ar1 = [(k0, v0['group'], v0['size']) for k0,v0 in self._dindref.items()]

        # # -----------------------
        # # Build for ddata
        # col2 = ['data key', 'origin', 'dim', 'quant',
                # 'name', 'units', 'shape', 'depend', 'lgroup']
        # ar2 = []
        # for k0,v0 in self._ddata.items():
            # if type(v0['data']) is np.ndarray:
                # shape = str(v0['data'].shape)
            # else:
                # shape = v0['data'].__class__.__name__
            # lu = [k0, v0['origin'], v0['dim'], v0['quant'], v0['name'],
                  # v0['units'], shape,
                  # str(v0['depend']), str(v0['lgroup'])]
            # ar2.append(lu)

        # return self._get_summary(
            # [ar0, ar1, ar2],
            # [col0, col1, col2],
            # sep=sep,
            # line=line,
            # table_sep=table_sep,
            # verb=verb,
            # return_=return_,
        # )

    # #---------------------
    # # Methods for adding ref / quantities
    # #---------------------

    # def _checkformat_addref(self, key=None, data=None, group=None,
                            # dim=None, quant=None, units=None,
                            # origin=None, name=None,
                            # comments=None, delimiter=None):
        # # Check data
        # lc = [isinstance(data, np.ndarray),
              # isinstance(data, dict),
              # isinstance(data, str) and os.path.isfile(data)]
        # if not any(lc):
            # msg = ("Arg data must be either:\n"
                   # + "\t- np.ndarray: a 1d array\n"
                   # + "\t- dict:       a dict containing a 2d mesh\n"
                   # + "\t- str:        an absolute path to an existing file\n"
                   # + "You provided:\n{}".format(data))
            # raise Exception(msg)

        # # If file: check content and extract data
        # if lc[2] is True:
            # data = os.path.abspath(data)
            # (data, key, group, units,
             # quant, dim, origin, name) = self._add_ref_from_file(
                 # pfe=data,
                 # key=key, group=group,
                 # dim=dim, quant=quant, units=units, origin=origin, name=name,
                 # comments=comments, delimiter=delimiter)

        # # Check key
        # c0 = type(key) is str and key not in self._ddata.keys()
        # if not c0:
            # msg = ("Arg key must be a str not already in self.ddata.keys()\n"
                   # + "\t- key: {}\n".format(key))
            # raise Exception(msg)

        # # Check group
        # c0 = group in self._dgroup.keys()
        # if not c0:
            # msg = ("Arg group must be str in self.dgroup.keys()\n"
                   # + "\t- group: {}".format(group)
                   # + "\t- available groups: {}".format(self.dgroups.keys()))
            # raise Exception(msg)

        # return data, key, group, units, dim, quant, origin, name

    # @staticmethod
    # def _add_ref_from_file(pfe=None, key=None, group=None,
                           # dim=None, quant=None, units=None,
                           # origin=None, name=None,
                           # comments=None, delimiter=None):
        # if comments is None:
            # comments = '#'

        # lf = ['.mat', '.txt']
        # c0 = pfe[-4:] in lf
        # if not c0:
            # msg = ("Only the following file formats are supported:\n"
                   # + "\n\t- " + "\n\t- ".join(lf) + "\n"
                   # + "You provided: {}".format(pfe))
            # raise Exception(msg)

        # # Extract data
        # if pfe[-4:] == '.mat':
            # # load and check only one 1x1 struct
            # import scipy.io as scpio
            # out = scpio.loadmat(pfe)
            # ls = [ss for ss in out.keys() if '__' not in ss]
            # c0 = (len(ls) == 1
                  # and isinstance(out[ls[0]], np.ndarray)
                  # and len(out[ls[0]]) == 1)
            # if not c0:
                # msg = ("The file should contain a 1x1 matlab struct only!\n"
                       # + "file contains: {}".format(ls))
                # raise Exception(msg)

            # # Get into unique struct and get key / value pairs
            # out = out[ls[0]][0]
            # nk = len(out.dtype)
            # if nk != len(out[0]):
                # msg = ("Non-conform file!\n"
                       # + "\tlen(out.dtype) = {}\n".format(nk)
                       # + "\tlen(out[0] = {}".format(len(out[0])))
                # raise Exception(msg)

            # lvi = [ii for ii in range(nk)
                   # if (out[0][ii].dtype.char == 'U'
                       # and out[0][ii].shape == (1,))]
            # limat = [ii for ii in range(nk) if ii not in lvi]

            # c0 = ((len(limat) == 1 and nk >= 1)
                  # and (out[0][limat[0]].ndim == 2
                       # and 1 in out[0][limat[0]].shape))
            # if not c0:
                # msg = (
                    # "The struct store in {} should contain:\n".format(pfe)
                    # + "\t- at least a (1, N) matrice\n"
                    # + "\t- optionally, the following char str:\n"
                    # + "\t\t- key: unique identifier\n"
                    # + "\t\t- group: 'time', 'radius', 'mesh', ...\n"
                    # + "\t\t- dim: physical dimension (e.g.: 'B flux',)\n"
                    # + "\t\t- quant: 'psi', 'phi, ...\n"
                    # + "\t\t- units: 'Wb', ...\n"
                    # + "\t\t- origin: 'NICE', 'CHEASE'..."
                    # + "\t\t\tby the default the file name\n"
                    # + "\t\t- name: short identifier (e.g.: 1dpsiNICE)\n\n"
                    # + "You provided:\n{}".format(out))
                # raise Exception(msg)
            # dout = {out.dtype.names[ii]: out[0][ii][0] for ii in lvi}
            # data = out[0][limat[0]].ravel()

        # elif pfe[-4:] == '.txt':
            # # data array
            # data = np.loadtxt(pfe, comments=comments, delimiter=delimiter)
            # if not data.ndim == 1:
                # msg = ("data stored in {} is not a 1d array!\n".format(pfe)
                       # + "\t- data.shape = {}".format(data.shape))
                # raise Exception(msg)

            # # params
            # dout = utils.from_txt_extract_params(
                # pfe=pfe,
                # lparams=['key', 'group', 'units',
                         # 'dim', 'quant', 'origin', 'name'],
                # comments=comments)

        # if 'origin' in dout.keys() and dout['origin'] is None:
            # del dout['origin']

        # # Get default values
        # din = {'key': key, 'group': group, 'dim': dim, 'quant': quant,
               # 'units': units, 'origin': origin, 'name': name}
        # for k0, v0 in din.items():
            # if v0 is None:
                # din[k0] = dout.get(k0, pfe) if k0 == 'origin' else dout.get(k0)
            # else:
                # if dout.get(k0) is not None:
                    # if din[k0] != dout[k0]:
                        # msg = ("Non-matching values of {}:\n".format(k0)
                               # + "{}\n".format(pfe)
                               # + "\t- kwdarg: {}\n".format(din[k0])
                               # + "\t- file: {}".format(dout[k0]))
                        # warnings.warn(msg)
        # return (data, din['key'], din['group'], din['units'],
                # din['quant'], din['dim'], din['origin'], din['name'])

    # def add_ref(self, key=None, data=None, group=None,
                # dim=None, quant=None, units=None, origin=None, name=None,
                # comments=None, delimiter=None):
        # """ Add a reference

        # The reference data is contained in data, which can be:
            # - np.array: a 1d profile
            # - dict: for mesh
            # - str:  absolute path to a file, holding a 1d profile

        # Please also provide (if not included in file if data is a str):
            # - key: unique str identifying the data
            # - group: str identifying the reference group (self.dgroup.keys())
        # If data is a str to a file, key and group (and others) can be included
        # in the file

        # Parameters dim, quant, units, origin and name are optional
        # Parameters comments and delimiter and only used if data is the path to
        # a .txt file (fed to np.loadtxt)
        # """
        # # Check inputs
        # (data, key, group, units,
         # dim, quant, origin, name) = self._checkformat_addref(
             # data=data, key=key, group=group, units=units,
             # dim=dim, quant=quant, origin=origin, name=name,
             # comments=comments, delimiter=delimiter)

        # # Format inputs
        # out = self._extract_dnd({key: {
            # 'dim': dim, 'quant': quant, 'name': name,
            # 'units': units, 'origin': origin
        # }},
            # key)
        # dim, quant, origin, name, units = out
        # if type(data) is np.ndarray:
            # size = data.shape[0]
        # else:
            # assert data['ftype'] in [0, 1]
            # size = data['nnodes'] if data['ftype'] == 1 else data['nfaces']

        # # Update attributes
        # self._dindref[key] = {'group': group, 'size': size, 'ldata': [key]}

        # self._ddata[key] = {'data': data,
                            # 'dim': dim, 'quant': quant, 'units': units,
                            # 'origin': origin, 'name': name,
                            # 'depend': (key,), 'lgroup': [group]}

        # # Run global consistency check and complement if necessary
        # self._complement()

    # def add_quantity(self, key=None, data=None, depend=None,
                     # dim=None, quant=None, units=None,
                     # origin=None, name=None):
        # """ Add a quantity """
        # c0 = type(key) is str and key not in self._ddata.keys()
        # if not c0:
            # msg = "key must be a str not already in self.ddata.keys()!\n"
            # msg += "    - Provided: %s"%str(key)
            # raise Exception(msg)
        # if type(data) not in [np.ndarray, dict]:
            # msg = "data must be either:\n"
            # msg += "    - np.ndarray\n"
            # msg += "    - dict (mesh)\n"
            # msg += "\n    Provided: %s"%str(type(data))
            # raise Exception(msg)
        # out = self._extract_dnd({key:{'dim':dim, 'quant':quant, 'name':name,
                                 # 'units':units, 'origin':origin}}, key)
        # dim, quant, origin, name, units = out
        # assert type(depend) in [list,str,tuple]
        # if type(depend) is str:
            # depend = (depend,)
        # for ii in range(0,len(depend)):
            # assert depend[ii] in self._dindref.keys()
        # lgroup = [self._dindref[dd]['group'] for dd in depend]
        # self._ddata[key] = {'data':data,
                            # 'dim':dim, 'quant':quant, 'units':units,
                            # 'origin':origin, 'name':name,
                            # 'depend':tuple(depend), 'lgroup':lgroup}
        # self._complement()


    # #---------------------
    # # Method for getting time of a quantity
    # #---------------------

    # def get_time(self, key):
        # """ Return the time vector associated to a chosen quantity (identified
        # by its key)"""

        # if key not in self._ddata.keys():
            # msg = "Provided key not in self.ddata.keys() !\n"
            # msg += "    - Provided: %s\n"%str(key)
            # msg += "    - Available: %s\n"%str(self._ddata.keys())
            # raise Exception(msg)

        # indref = self._ddata[key]['depend'][0]
        # t = [kk for kk in self._dindref[indref]['ldata']
             # if (self._ddata[kk]['depend'] == (indref,)
                 # and self._ddata[kk]['quant'] == 't')]
        # if len(t) != 1:
            # msg = "No / several macthing time vectors were identified:\n"
            # msg += "    - Provided: %s\n"%key
            # msg += "    - Found: %s"%str(t)
            # raise Exception(msg)
        # return t[0]


    # def get_time_common(self, lkeys, choose=None):
        # """ Return the common time vector to several quantities

        # If they do not have a common time vector, a reference one is choosen
        # according to criterion choose
        # """
        # # Check all data have time-dependency
        # dout = {kk: {'t':self.get_time(kk)} for kk in lkeys}
        # dtu = dict.fromkeys(set([vv['t'] for vv in dout.values()]))
        # for kt in dtu.keys():
            # dtu[kt] = {'ldata':[kk for kk in lkeys if dout[kk]['t'] == kt]}
        # if len(dtu) == 1:
            # tref = list(dtu.keys())[0]
        # else:
            # lt, lres = zip(*[(kt,np.mean(np.diff(self._ddata[kt]['data'])))
                             # for kt in dtu.keys()])
            # if choose is None:
                # choose  = 'min'
            # if choose == 'min':
                # tref = lt[np.argmin(lres)]
        # return dout, dtu, tref

    # @staticmethod
    # def _get_time_common_arrays(dins, choose=None):
        # dout = dict.fromkeys(dins.keys())
        # dtu = {}
        # for k, v in dins.items():
            # c0 = type(k) is str
            # c0 = c0 and all([ss in v.keys() for ss in ['val','t']])
            # c0 = c0 and all([type(v[ss]) is np.ndarray for ss in ['val','t']])
            # c0 = c0 and v['t'].size in v['val'].shape
            # if not c0:
                # msg = "dins must be a dict of the form (at least):\n"
                # msg += "    dins[%s] = {'val': np.ndarray,\n"%str(k)
                # msg += "                't':   np.ndarray}\n"
                # msg += "Provided: %s"%str(dins)
                # raise Exception(msg)

            # kt, already = id(v['t']), True
            # if kt not in dtu.keys():
                # lisclose = [kk for kk, vv in dtu.items()
                            # if (vv['val'].shape == v['t'].shape
                                # and np.allclose(vv['val'],v['t']))]
                # assert len(lisclose) <= 1
                # if len(lisclose) == 1:
                    # kt = lisclose[0]
                # else:
                    # already = False
                    # dtu[kt] = {'val':np.atleast_1d(v['t']).ravel(),
                               # 'ldata':[k]}
            # if already:
                # dtu[kt]['ldata'].append(k)
            # assert dtu[kt]['val'].size == v['val'].shape[0]
            # dout[k] = {'val':v['val'], 't':kt}

        # if len(dtu) == 1:
            # tref = list(dtu.keys())[0]
        # else:
            # lt, lres = zip(*[(kt,np.mean(np.diff(dtu[kt]['val'])))
                             # for kt in dtu.keys()])
            # if choose is None:
                # choose  = 'min'
            # if choose == 'min':
                # tref = lt[np.argmin(lres)]
        # return dout, dtu, tref

    # def _interp_on_common_time(self, lkeys,
                               # choose='min', interp_t=None, t=None,
                               # fill_value=np.nan):
        # """ Return a dict of time-interpolated data """
        # dout, dtu, tref = self.get_time_common(lkeys)
        # if type(t) is np.ndarray:
            # tref = np.atleast_1d(t).ravel()
            # tr = tref
            # ltu = dtu.keys()
        # else:
            # if type(t) is str:
                # tref = t
            # tr = self._ddata[tref]['data']
            # ltu = set(dtu.keys())
            # if tref in dtu.keys():
                # ltu = ltu.difference([tref])

        # if interp_t is None:
            # interp_t = _INTERPT

        # # Interpolate
        # for tt in ltu:
            # for kk in dtu[tt]['ldata']:
                # dout[kk]['val'] = scpinterp.interp1d(self._ddata[tt]['data'],
                                                     # self._ddata[kk]['data'],
                                                     # kind=interp_t, axis=0,
                                                     # bounds_error=False,
                                                     # fill_value=fill_value)(tr)

        # if type(tref) is not np.ndarray and tref in dtu.keys():
            # for kk in dtu[tref]['ldata']:
                 # dout[kk]['val'] = self._ddata[kk]['data']

        # return dout, tref

    # def _interp_on_common_time_arrays(self, dins,
                                      # choose='min', interp_t=None, t=None,
                                      # fill_value=np.nan):
        # """ Return a dict of time-interpolated data """
        # dout, dtu, tref = self._get_time_common_arrays(dins)
        # if type(t) is np.ndarray:
            # tref = np.atleast_1d(t).ravel()
            # tr = tref
            # ltu = dtu.keys()
        # else:
            # if type(t) is str:
                # assert t in dout.keys()
                # tref = dout[t]['t']
            # tr = dtu[tref]['val']
            # ltu = set(dtu.keys()).difference([tref])

        # if interp_t is None:
            # interp_t = _INTERPT

        # # Interpolate
        # for tt in ltu:
            # for kk in dtu[tt]['ldata']:
                # dout[kk]['val'] = scpinterp.interp1d(dtu[tt]['val'],
                                                     # dout[kk]['val'],
                                                     # kind=interp_t, axis=0,
                                                     # bounds_error=False,
                                                     # fill_value=fill_value)(tr)
        # return dout, tref

    # def interp_t(self, dkeys,
                 # choose='min', interp_t=None, t=None,
                 # fill_value=np.nan):
        # # Check inputs
        # assert type(dkeys) in [list,dict]
        # if type(dkeys) is list:
            # dkeys = {kk:{'val':kk} for kk in dkeys}
        # lc = [(type(kk) is str
               # and type(vv) is dict
               # and type(vv.get('val',None)) in [str,np.ndarray])
              # for kk,vv in dkeys.items()]
        # assert all(lc), str(dkeys)

        # # Separate by type
        # dk0 = dict([(kk,vv) for kk,vv in dkeys.items()
                    # if type(vv['val']) is str])
        # dk1 = dict([(kk,vv) for kk,vv in dkeys.items()
                    # if type(vv['val']) is np.ndarray])
        # assert len(dkeys) == len(dk0) + len(dk1), str(dk0) + '\n' + str(dk1)


        # if len(dk0) == len(dkeys):
            # lk = [v['val'] for v in dk0.values()]
            # dout, tref = self._interp_on_common_time(lk, choose=choose,
                                                     # t=t, interp_t=interp_t,
                                                     # fill_value=fill_value)
            # dout = {kk:{'val':dout[vv['val']]['val'], 't':dout[vv['val']]['t']}
                    # for kk,vv in dk0.items()}
        # elif len(dk1) == len(dkeys):
            # dout, tref = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                            # t=t, interp_t=interp_t,
                                                            # fill_value=fill_value)

        # else:
            # lk = [v['val'] for v in dk0.values()]
            # if type(t) is np.ndarray:
                # dout, tref =  self._interp_on_common_time(lk, choose=choose,
                                                       # t=t, interp_t=interp_t,
                                                       # fill_value=fill_value)
                # dout1, _   = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                              # t=t, interp_t=interp_t,
                                                              # fill_value=fill_value)
            # else:
                # dout0, dtu0, tref0 = self.get_time_common(lk,
                                                          # choose=choose)
                # dout1, dtu1, tref1 = self._get_time_common_arrays(dk1,
                                                                  # choose=choose)
                # if type(t) is str:
                    # lc = [t in dtu0.keys(), t in dout1.keys()]
                    # if not any(lc):
                        # msg = "if t is str, it must refer to a valid key:\n"
                        # msg += "    - %s\n"%str(dtu0.keys())
                        # msg += "    - %s\n"%str(dout1.keys())
                        # msg += "Provided: %s"%t
                        # raise Exception(msg)
                    # if lc[0]:
                        # t0, t1 = t, self._ddata[t]['data']
                    # else:
                        # t0, t1 = dtu1[dout1[t]['t']]['val'], t
                    # tref = t
                # else:
                    # if choose is None:
                        # choose = 'min'
                    # if choose == 'min':
                        # t0 = self._ddata[tref0]['data']
                        # t1 = dtu1[tref1]['val']
                        # dt0 = np.mean(np.diff(t0))
                        # dt1 = np.mean(np.diff(t1))
                        # if dt0 < dt1:
                            # t0, t1, tref = tref0, t0, tref0
                        # else:
                            # t0, t1, tref = t1, tref1, tref1

                # dout, tref =  self._interp_on_common_time(lk, choose=choose,
                                                          # t=t0, interp_t=interp_t,
                                                          # fill_value=fill_value)
                # dout = {kk:{'val':dout[vv['val']]['val'],
                            # 't':dout[vv['val']]['t']}
                        # for kk,vv in dk0.items()}
                # dout1, _   = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                                # t=t1, interp_t=interp_t,
                                                                # fill_value=fill_value)
            # dout.update(dout1)

        # return dout, tref

    # #---------------------
    # # Methods for computing additional plasma quantities
    # #---------------------


    # def _fill_dins(self, dins):
        # for k in dins.keys():
            # if type(dins[k]['val']) is str:
                # assert dins[k]['val'] in self._ddata.keys()
            # else:
                # dins[k]['val'] = np.atleast_1d(dins[k]['val'])
                # assert dins[k]['t'] is not None
                # dins[k]['t'] = np.atleast_1d(dins[k]['t']).ravel()
                # assert dins[k]['t'].size == dins[k]['val'].shape[0]
        # return dins

    # @staticmethod
    # def _checkformat_shapes(dins):
        # shape = None
        # for k in dins.keys():
            # dins[k]['shape'] = dins[k]['val'].shape
            # if shape is None:
                # shape = dins[k]['shape']
            # if dins[k]['shape'] != shape:
                # if dins[k]['val'].ndim > len(shape):
                    # shape = dins[k]['shape']

        # # Check shape consistency for broadcasting
        # assert len(shape) in [1,2]
        # if len(shape) == 1:
            # for k in dins.keys():
                # assert dins[k]['shape'][0] in [1,shape[0]]
                # if dins[k]['shape'][0] < shape[0]:
                    # dins[k]['val'] = np.full((shape[0],), dins[k]['val'][0])
                    # dins[k]['shape'] = dins[k]['val'].shape

        # elif len(shape) == 2:
            # for k in dins.keys():
                # if len(dins[k]['shape']) == 1:
                    # if dins[k]['shape'][0] not in [1]+list(shape):
                        # msg = "Non-conform shape for dins[%s]:\n"%k
                        # msg += "    - Expected: (%s,...) or (1,)\n"%str(shape[0])
                        # msg += "    - Provided: %s"%str(dins[k]['shape'])
                        # raise Exception(msg)
                    # if dins[k]['shape'][0] == 1:
                        # dins[k]['val'] = dins[k]['val'][None,:]
                    # elif dins[k]['shape'][0] == shape[0]:
                        # dins[k]['val'] = dins[k]['val'][:,None]
                    # else:
                        # dins[k]['val'] = dins[k]['val'][None,:]
                # else:
                    # assert dins[k]['shape'] == shape
                # dins[k]['shape'] = dins[k]['val'].shape
        # return dins



    # def compute_bremzeff(self, Te=None, ne=None, zeff=None, lamb=None,
                         # tTe=None, tne=None, tzeff=None, t=None,
                         # interp_t=None):
        # """ Return the bremsstrahlung spectral radiance at lamb

        # The plasma conditions are set by:
            # - Te   (eV)
            # - ne   (/m3)
            # - zeff (adim.)

        # The wavelength is set by the diagnostics
            # - lamb (m)

        # The vol. spectral emis. is returned in ph / (s.m3.sr.m)

        # The computation requires an intermediate : gff(Te, zeff)
        # """
        # dins = {'Te':{'val':Te, 't':tTe},
                # 'ne':{'val':ne, 't':tne},
                # 'zeff':{'val':zeff, 't':tzeff}}
        # lc = [vv['val'] is None for vv in dins.values()]
        # if any(lc):
            # msg = "All fields should be provided:\n"
            # msg += "    - %s"%str(dins.keys())
            # raise Exception(msg)
        # dins = self._fill_dins(dins)
        # dins, t = self.interp_t(dins, t=t, interp_t=interp_t)
        # lamb = np.atleast_1d(lamb)
        # dins['lamb'] = {'val':lamb}
        # dins = self._checkformat_shapes(dins)

        # val, units = _physics.compute_bremzeff(dins['Te']['val'],
                                               # dins['ne']['val'],
                                               # dins['zeff']['val'],
                                               # dins['lamb']['val'])
        # return val, t, units

    # def compute_fanglev(self, BR=None, BPhi=None, BZ=None,
                        # ne=None, lamb=None, t=None, interp_t=None,
                        # tBR=None, tBPhi=None, tBZ=None, tne=None):
        # """ Return the vector faraday angle at lamb

        # The plasma conditions are set by:
            # - BR    (T) , array of R component of B
            # - BRPhi (T) , array of phi component of B
            # - BZ    (T) , array of Z component of B
            # - ne    (/m3)

        # The wavelength is set by the diagnostics
            # - lamb (m)

        # The vector faraday angle is returned in T / m
        # """
        # dins = {'BR':  {'val':BR,   't':tBR},
                # 'BPhi':{'val':BPhi, 't':tBPhi},
                # 'BZ':  {'val':BZ,   't':tBZ},
                # 'ne':  {'val':ne,   't':tne}}
        # dins = self._fill_dins(dins)
        # dins, t = self.interp_t(dins, t=t, interp_t=interp_t)
        # lamb = np.atleast_1d(lamb)
        # dins['lamb'] = {'val':lamb}
        # dins = self._checkformat_shapes(dins)

        # val, units = _physics.compute_fangle(BR=dins['BR']['val'],
                                             # BPhi=dins['BPhi']['val'],
                                             # BZ=dins['BZ']['val'],
                                             # ne=dins['ne']['val'],
                                             # lamb=dins['lamb']['val'])
        # return val, t, units



    # #---------------------
    # # Methods for interpolation
    # #---------------------


    # def _get_quantrefkeys(self, qq, ref1d=None, ref2d=None):

        # # Get relevant lists
        # kq, msg = self._get_keyingroup(qq, 'mesh', msgstr='quant', raise_=False)
        # if kq is not None:
            # k1d, k2d = None, None
        # else:
            # kq, msg = self._get_keyingroup(qq, 'radius', msgstr='quant', raise_=True)
            # if ref1d is None and ref2d is None:
                # msg = "quant %s needs refs (1d and 2d) for interpolation\n"%qq
                # msg += "  => ref1d and ref2d cannot be both None !"
                # raise Exception(msg)
            # if ref1d is None:
                # ref1d = ref2d
            # k1d, msg = self._get_keyingroup(ref1d, 'radius',
                                            # msgstr='ref1d', raise_=False)
            # if k1d is None:
                # msg += "\n\nInterpolation of %s:\n"%qq
                # msg += "  ref could not be identified among 1d quantities\n"
                # msg += "    - ref1d : %s"%ref1d
                # raise Exception(msg)
            # if ref2d is None:
                # ref2d = ref1d
            # k2d, msg = self._get_keyingroup(ref2d, 'mesh',
                                            # msgstr='ref2d', raise_=False)
            # if k2d is None:
                # msg += "\n\nInterpolation of %s:\n"
                # msg += "  ref could not be identified among 2d quantities\n"
                # msg += "    - ref2d: %s"%ref2d
                # raise Exception(msg)

            # q1d, q2d = self._ddata[k1d]['quant'], self._ddata[k2d]['quant']
            # if q1d != q2d:
                # msg = "ref1d and ref2d must be of the same quantity !\n"
                # msg += "    - ref1d (%s):   %s\n"%(ref1d, q1d)
                # msg += "    - ref2d (%s):   %s"%(ref2d, q2d)
                # raise Exception(msg)

        # return kq, k1d, k2d


    # def _get_indtmult(self, idquant=None, idref1d=None, idref2d=None):

        # # Get time vectors and bins
        # idtq = self._ddata[idquant]['depend'][0]
        # tq = self._ddata[idtq]['data']
        # tbinq = 0.5*(tq[1:]+tq[:-1])
        # if idref1d is not None:
            # idtr1 = self._ddata[idref1d]['depend'][0]
            # tr1 = self._ddata[idtr1]['data']
            # tbinr1 = 0.5*(tr1[1:]+tr1[:-1])
        # if idref2d is not None and idref2d != idref1d:
            # idtr2 = self._ddata[idref2d]['depend'][0]
            # tr2 = self._ddata[idtr2]['data']
            # tbinr2 = 0.5*(tr2[1:]+tr2[:-1])

        # # Get tbinall and tall
        # if idref1d is None:
            # tbinall = tbinq
            # tall = tq
        # else:
            # if idref2d is None:
                # tbinall = np.unique(np.r_[tbinq,tbinr1])
            # else:
                # tbinall = np.unique(np.r_[tbinq,tbinr1,tbinr2])
            # tall = np.r_[tbinall[0] - 0.5*(tbinall[1]-tbinall[0]),
                         # 0.5*(tbinall[1:]+tbinall[:-1]),
                         # tbinall[-1] + 0.5*(tbinall[-1]-tbinall[-2])]

        # # Get indtqr1r2 (tall with respect to tq, tr1, tr2)
        # indtq, indtr1, indtr2 = None, None, None
        # if tbinq.size > 0:
            # indtq = np.digitize(tall, tbinq)
        # else:
            # indtq = np.r_[0]
        # if idref1d is None:
            # assert np.all(indtq == np.arange(0,tall.size))
        # if idref1d is not None:
            # if tbinr1.size > 0:
                # indtr1 = np.digitize(tall, tbinr1)
            # else:
                # indtr1 = np.r_[0]
        # if idref2d is not None:
            # if tbinr2.size > 0:
                # indtr2 = np.digitize(tall, tbinr2)
            # else:
                # indtr2 = np.r_[0]

        # ntall = tall.size
        # return tall, tbinall, ntall, indtq, indtr1, indtr2

    # @staticmethod
    # def _get_indtu(t=None, tall=None, tbinall=None,
                   # idref1d=None, idref2d=None,
                   # indtr1=None, indtr2=None):
        # # Get indt (t with respect to tbinall)
        # indt, indtu = None, None
        # if t is not None:
            # if len(t) == len(tall) and np.allclose(t, tall):
                # indt = np.arange(0, tall.size)
                # indtu = indt
            # else:
                # indt = np.digitize(t, tbinall)
                # indtu = np.unique(indt)
                # # Update
                # tall = tall[indtu]

            # if idref1d is not None:
                # assert indtr1 is not None
                # indtr1 = indtr1[indtu]
            # if idref2d is not None:
                # assert indtr2 is not None
                # indtr2 = indtr2[indtu]
        # ntall = tall.size
        # return tall, ntall, indt, indtu, indtr1, indtr2

    # def get_tcommon(self, lq, prefer='finer'):
        # """ Check if common t, else choose according to prefer

        # By default, prefer the finer time resolution

        # """
        # if type(lq) is str:
            # lq = [lq]
        # t = []
        # for qq in lq:
            # ltr = [kk for kk in self._ddata[qq]['depend']
                   # if self._dindref[kk]['group'] == 'time']
            # assert len(ltr) <= 1
            # if len(ltr) > 0 and ltr[0] not in t:
                # t.append(ltr[0])
        # assert len(t) >= 1
        # if len(t) > 1:
            # dt = [np.nanmean(np.diff(self._ddata[tt]['data'])) for tt in t]
            # if prefer == 'finer':
                # ind = np.argmin(dt)
            # else:
                # ind = np.argmax(dt)
        # else:
            # ind = 0
        # return t[ind], t

    # def _get_tcom(self, idquant=None, idref1d=None,
                  # idref2d=None, idq2dR=None):
        # if idquant is not None:
            # out = self._get_indtmult(idquant=idquant,
                                     # idref1d=idref1d, idref2d=idref2d)
        # else:
            # out = self._get_indtmult(idquant=idq2dR)
        # return out

    # def _get_finterp(
        # self,
        # idquant=None, idref1d=None, idref2d=None,
        # idq2dR=None, idq2dPhi=None, idq2dZ=None,
        # interp_t=None, interp_space=None,
        # fill_value=None, ani=False, Type=None,
    # ):

        # if interp_t is None:
            # interp_t = 'nearest'

        # # Get idmesh
        # if idquant is not None:
            # if idref1d is None:
                # lidmesh = [qq for qq in self._ddata[idquant]['depend']
                           # if self._dindref[qq]['group'] == 'mesh']
            # else:
                # lidmesh = [qq for qq in self._ddata[idref2d]['depend']
                           # if self._dindref[qq]['group'] == 'mesh']
        # else:
            # assert idq2dR is not None
            # lidmesh = [qq for qq in self._ddata[idq2dR]['depend']
                       # if self._dindref[qq]['group'] == 'mesh']
        # assert len(lidmesh) == 1
        # idmesh = lidmesh[0]

        # # Get common time indices
        # if interp_t == 'nearest':
            # out = self._get_tcom(idquant, idref1d, idref2d, idq2dR)
            # tall, tbinall, ntall, indtq, indtr1, indtr2 = out

        # # Get mesh
        # if self._ddata[idmesh]['data']['type'] == 'rect':
            # mpltri = None
            # trifind = self._ddata[idmesh]['data']['trifind']
        # else:
            # mpltri = self._ddata[idmesh]['data']['mpltri']
            # trifind = mpltri.get_trifinder()

        # # # Prepare output

        # # Interpolate
        # # Note : Maybe consider using scipy.LinearNDInterpolator ?
        # if idquant is not None:
            # vquant = self._ddata[idquant]['data']
            # if idref1d is not None:
                # vr1 = self._ddata[idref1d]['data']
                # vr2 = self._ddata[idref2d]['data']

            # c0 = (
                # self._ddata[idmesh]['data']['type'] == 'quadtri'
                # and self._ddata[idmesh]['data']['ntri'] > 1
            # )
            # if c0:
                # vquant = np.repeat(
                    # vquant,
                    # self._ddata[idmesh]['data']['ntri'],
                    # axis=0,
                # )
                # if idref1d is not None:
                    # vr2 = np.repeat(
                        # vr2,
                        # self._ddata[idmesh]['data']['ntri'],
                        # axis=0,
                    # )
        # else:
            # vq2dR   = self._ddata[idq2dR]['data']
            # vq2dPhi = self._ddata[idq2dPhi]['data']
            # vq2dZ   = self._ddata[idq2dZ]['data']

        # if interp_space is None:
            # interp_space = self._ddata[idmesh]['data']['ftype']

        # # get interpolation function
        # if ani:
            # # Assuming same mesh and time vector for all 3 components
            # func = _comp.get_finterp_ani(
                # idq2dR, idq2dPhi, idq2dZ,
                # idmesh=idmesh, vq2dR=vq2dR,
                # vq2dZ=vq2dZ, vq2dPhi=vq2dPhi,
                # tall=tall, tbinall=tbinall,
                # ntall=ntall,
                # interp_t=interp_t,
                # interp_space=interp_space,
                # fill_value=fill_value,
                # indtq=indtq, trifind=trifind,
                # Type=Type, mpltri=mpltri,
            # )
        # else:
            # func = _comp.get_finterp_isotropic(
                # idquant, idref1d, idref2d,
                # vquant=vquant,
                # vr1=vr1,
                # vr2=vr2,
                # interp_t=interp_t,
                # interp_space=interp_space,
                # fill_value=fill_value,
                # idmesh=idmesh,
                # tall=tall, tbinall=tbinall,
                # ntall=ntall, mpltri=mpltri,
                # indtq=indtq, indtr1=indtr1,
                # indtr2=indtr2, trifind=trifind,
            # )

        # return func


    # def _checkformat_qr12RPZ(self, quant=None, ref1d=None, ref2d=None,
                             # q2dR=None, q2dPhi=None, q2dZ=None):
        # lc0 = [quant is None, ref1d is None, ref2d is None]
        # lc1 = [q2dR is None, q2dPhi is None, q2dZ is None]
        # if np.sum([all(lc0), all(lc1)]) != 1:
            # msg = "Please provide either (xor):\n"
            # msg += "    - a scalar field (isotropic emissivity):\n"
            # msg += "        quant : scalar quantity to interpolate\n"
            # msg += "                if quant is 1d, intermediate reference\n"
            # msg += "                fields are necessary for 2d interpolation\n"
            # msg += "        ref1d : 1d reference field on which to interpolate\n"
            # msg += "        ref2d : 2d reference field on which to interpolate\n"
            # msg += "    - a vector (R,Phi,Z) field (anisotropic emissivity):\n"
            # msg += "        q2dR :  R component of the vector field\n"
            # msg += "        q2dPhi: R component of the vector field\n"
            # msg += "        q2dZ :  Z component of the vector field\n"
            # msg += "        => all components have teh same time and mesh !\n"
            # raise Exception(msg)

        # # Check requested quant is available in 2d or 1d
        # if all(lc1):
            # idquant, idref1d, idref2d = self._get_quantrefkeys(quant, ref1d, ref2d)
            # idq2dR, idq2dPhi, idq2dZ = None, None, None
            # ani = False
        # else:
            # idq2dR, msg   = self._get_keyingroup(q2dR, 'mesh', msgstr='quant',
                                              # raise_=True)
            # idq2dPhi, msg = self._get_keyingroup(q2dPhi, 'mesh', msgstr='quant',
                                              # raise_=True)
            # idq2dZ, msg   = self._get_keyingroup(q2dZ, 'mesh', msgstr='quant',
                                              # raise_=True)
            # idquant, idref1d, idref2d = None, None, None
            # ani = True
        # return idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani


    # def get_finterp2d(self, quant=None, ref1d=None, ref2d=None,
                      # q2dR=None, q2dPhi=None, q2dZ=None,
                      # interp_t=None, interp_space=None,
                      # fill_value=None, Type=None):
        # """ Return the function interpolating (X,Y,Z) pts on a 1d/2d profile

        # Can be used as input for tf.geom.CamLOS1D/2D.calc_signal()

        # """
        # # Check inputs
        # msg = "Only 'nearest' available so far for interp_t!"
        # assert interp_t == 'nearest', msg
        # out = self._checkformat_qr12RPZ(quant=quant, ref1d=ref1d, ref2d=ref2d,
                                        # q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ)
        # idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = out


        # # Interpolation (including time broadcasting)
        # func = self._get_finterp(idquant=idquant, idref1d=idref1d,
                                 # idref2d=idref2d, idq2dR=idq2dR,
                                 # idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
                                 # interp_t=interp_t, interp_space=interp_space,
                                 # fill_value=fill_value, ani=ani, Type=Type)
        # return func


    # def interp_pts2profile(self, pts=None, vect=None, t=None,
                           # quant=None, ref1d=None, ref2d=None,
                           # q2dR=None, q2dPhi=None, q2dZ=None,
                           # interp_t=None, interp_space=None,
                           # fill_value=None, Type=None):
        # """ Return the value of the desired profiles_1d quantity

        # For the desired inputs points (pts):
            # - pts are in (X,Y,Z) coordinates
            # - space interpolation is linear on the 1d profiles
        # At the desired input times (t):
            # - using a nearest-neighbourg approach for time

        # """
        # # Check inputs
        # # msg = "Only 'nearest' available so far for interp_t!"
        # # assert interp_t == 'nearest', msg

        # # Check requested quant is available in 2d or 1d
        # out = self._checkformat_qr12RPZ(quant=quant, ref1d=ref1d, ref2d=ref2d,
                                        # q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ)
        # idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = out

        # # Check the pts is (2,...) array of floats
        # if pts is None:
            # if ani:
                # idmesh = [id_ for id_ in self._ddata[idq2dR]['depend']
                          # if self._dindref[id_]['group'] == 'mesh'][0]
            # else:
                # if idref1d is None:
                    # idmesh = [id_ for id_ in self._ddata[idquant]['depend']
                              # if self._dindref[id_]['group'] == 'mesh'][0]
                # else:
                    # idmesh = [id_ for id_ in self._ddata[idref2d]['depend']
                              # if self._dindref[id_]['group'] == 'mesh'][0]
            # if self.dmesh[idmesh]['data']['type'] == 'rect':
                # if self.dmesh[idmesh]['data']['shapeRZ'] == ('R', 'Z'):
                    # R = np.repeat(self.dmesh[idmesh]['data']['R'],
                                  # self.dmesh[idmesh]['data']['nZ'])
                    # Z = np.tile(self.dmesh[idmesh]['data']['Z'],
                                # self.dmesh[idmesh]['data']['nR'])
                # else:
                    # R = np.tile(self.dmesh[idmesh]['data']['R'],
                                # self.dmesh[idmesh]['data']['nZ'])
                    # Z = np.repeat(self.dmesh[idmesh]['data']['Z'],
                                  # self.dmesh[idmesh]['data']['nR'])
                # pts = np.array(
                    # [R, np.zeros((self.dmesh[idmesh]['data']['size'],)), Z])
            # else:
                # pts = self.dmesh[idmesh]['data']['nodes']
                # pts = np.array(
                    # [pts[:, 0], np.zeros((pts.shape[0],)), pts[:, 1]])

        # pts = np.atleast_2d(pts)
        # if pts.shape[0] != 3:
            # msg = "pts must be np.ndarray of (X,Y,Z) points coordinates\n"
            # msg += "Can be multi-dimensional, but the 1st dimension is (X,Y,Z)\n"
            # msg += "    - Expected shape : (3,...)\n"
            # msg += "    - Provided shape : %s"%str(pts.shape)
            # raise Exception(msg)

        # # Check t
        # lc = [t is None, type(t) is str, type(t) is np.ndarray]
        # assert any(lc)
        # if lc[1]:
            # assert t in self._ddata.keys()
            # t = self._ddata[t]['data']

        # # Interpolation (including time broadcasting)
        # # this is the second slowest step (~0.08 s)
        # func = self._get_finterp(
            # idquant=idquant, idref1d=idref1d, idref2d=idref2d,
            # idq2dR=idq2dR, idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
            # interp_t=interp_t, interp_space=interp_space,
            # fill_value=fill_value, ani=ani, Type=Type,
        # )

        # # This is the slowest step (~1.8 s)
        # val, t = func(pts, vect=vect, t=t)
        # return val, t


    # def calc_signal_from_Cam(self, cam, t=None,
                             # quant=None, ref1d=None, ref2d=None,
                             # q2dR=None, q2dPhi=None, q2dZ=None,
                             # Brightness=True, interp_t=None,
                             # interp_space=None, fill_value=None,
                             # res=0.005, DL=None, resMode='abs', method='sum',
                             # ind=None, out=object, plot=True, dataname=None,
                             # fs=None, dmargin=None, wintit=None, invert=True,
                             # units=None, draw=True, connect=True):

        # if 'Cam' not in cam.__class__.__name__:
            # msg = "Arg cam must be tofu Camera instance (CamLOS1D, CamLOS2D...)"
            # raise Exception(msg)

        # return cam.calc_signal_from_Plasma2D(self, t=t,
                                             # quant=quant, ref1d=ref1d, ref2d=ref2d,
                                             # q2dR=q2dR, q2dPhi=q2dPhi,
                                             # q2dZ=q2dZ,
                                             # Brightness=Brightness,
                                             # interp_t=interp_t,
                                             # interp_space=interp_space,
                                             # fill_value=fill_value, res=res,
                                             # DL=DL, resMode=resMode,
                                             # method=method, ind=ind, out=out,
                                             # pot=plot, dataname=dataname,
                                             # fs=fs, dmargin=dmargin,
                                             # wintit=wintit, invert=invert,
                                             # units=units, draw=draw,
                                             # connect=connect)


    # #---------------------
    # # Methods for getting data
    # #---------------------

    # def get_dextra(self, dextra=None):
        # lc = [dextra is None, dextra == 'all', type(dextra) is dict,
              # type(dextra) is str, type(dextra) is list]
        # assert any(lc)
        # if dextra is None:
            # dextra = {}

        # if dextra == 'all':
            # dextra = [k for k in self._dgroup['time']['ldata']
                      # if (self._ddata[k]['lgroup'] == ['time']
                          # and k not in self._dindref.keys())]

        # if type(dextra) is str:
            # dextra = [dextra]

        # # get data
        # if type(dextra) is list:
            # for ii in range(0,len(dextra)):
                # if type(dextra[ii]) is tuple:
                    # ee, cc = dextra[ii]
                # else:
                    # ee, cc = dextra[ii], None
                # ee, msg = self._get_keyingroup(ee, 'time', raise_=True)
                # if self._ddata[ee]['lgroup'] != ['time']:
                    # msg = "time-only dependent signals allowed in dextra!\n"
                    # msg += "    - %s : %s"%(ee,str(self._ddata[ee]['lgroup']))
                    # raise Exception(msg)
                # idt = self._ddata[ee]['depend'][0]
                # key = 'data' if self._ddata[ee]['data'].ndim == 1 else 'data2D'
                # dd = {key: self._ddata[ee]['data'],
                      # 't': self._ddata[idt]['data'],
                      # 'label': self._ddata[ee]['name'],
                      # 'units': self._ddata[ee]['units']}
                # if cc is not None:
                    # dd['c'] = cc
                # dextra[ii] = (ee, dd)
            # dextra = dict(dextra)
        # return dextra

    # def get_Data(self, lquant, X=None, ref1d=None, ref2d=None,
                 # remap=False, res=0.01, interp_space=None, dextra=None):

        # try:
            # import tofu.data as tfd
        # except Exception:
            # from .. import data as tfd

        # # Check and format input
        # assert type(lquant) in [str,list]
        # if type(lquant) is str:
            # lquant = [lquant]
        # nquant = len(lquant)

        # # Get X if common
        # c0 = type(X) is str
        # c1 = type(X) is list and (len(X) == 1 or len(X) == nquant)
        # if not (c0 or c1):
            # msg = ("X must be specified, either as :\n"
                   # + "    - a str (name or quant)\n"
                   # + "    - a list of str\n"
                   # + "    Provided: {}".format(X))
            # raise Exception(msg)
        # if c1 and len(X) == 1:
            # X = X[0]

        # if type(X) is str:
            # idX, msg = self._get_keyingroup(X, 'radius', msgstr='X', raise_=True)

        # # prepare remap pts
        # if remap:
            # assert self.config is not None
            # refS = list(self.config.dStruct['dObj']['Ves'].values())[0]
            # ptsRZ, x1, x2, extent = refS.get_sampleCross(res, mode='imshow')
            # dmap = {'t':None, 'data2D':None, 'extent':extent}
            # if ref is None and X in self._lquantboth:
                # ref = X

        # # Define Data
        # dcommon = dict(Exp=self.Id.Exp, shot=self.Id.shot,
                       # Diag='profiles1d', config=self.config)

        # # dextra
        # dextra = self.get_dextra(dextra)

        # # Get output
        # lout = [None for qq in lquant]
        # for ii in range(0,nquant):
            # qq = lquant[ii]
            # if remap:
                # # Check requested quant is available in 2d or 1d
                # idq, idrefd1, idref2d = self._get_quantrefkeys(qq, ref1d, ref2d)
            # else:
                # idq, msg = self._get_keyingroup(qq, 'radius',
                                                # msgstr='quant', raise_=True)
            # if idq not in self._dgroup['radius']['ldata']:
                # msg = "Only 1d quantities can be turned into tf.data.Data !\n"
                # msg += "    - %s is not a radius-dependent quantity"%qq
                # raise Exception(msg)
            # idt = self._ddata[idq]['depend'][0]

            # if type(X) is list:
                # idX, msg = self._get_keyingroup(X[ii], 'radius',
                                                # msgstr='X', raise_=True)

            # dlabels = {'data':{'name': self._ddata[idq]['name'],
                               # 'units': self._ddata[idq]['units']},
                       # 'X':{'name': self._ddata[idX]['name'],
                            # 'units': self._ddata[idX]['units']},
                       # 't':{'name': self._ddata[idt]['name'],
                            # 'units': self._ddata[idt]['units']}}

            # dextra_ = dict(dextra)
            # if remap:
                # dmapii = dict(dmap)
                # val, tii = self.interp_pts2profile(qq, ptsRZ=ptsRZ, ref=ref,
                                                   # interp_space=interp_space)
                # dmapii['data2D'], dmapii['t'] = val, tii
                # dextra_['map'] = dmapii
            # lout[ii] = DataCam1D(Name = qq,
                                 # data = self._ddata[idq]['data'],
                                 # t = self._ddata[idt]['data'],
                                 # X = self._ddata[idX]['data'],
                                 # dextra = dextra_, dlabels=dlabels, **dcommon)
        # if nquant == 1:
            # lout = lout[0]
        # return lout


    # #---------------------
    # # Methods for plotting data
    # #---------------------

    # def plot(self, lquant, X=None,
             # ref1d=None, ref2d=None,
             # remap=False, res=0.01, interp_space=None,
             # sharex=False, bck=True):
        # lDat = self.get_Data(lquant, X=X, remap=remap,
                             # ref1d=ref1d, ref2d=ref2d,
                             # res=res, interp_space=interp_space)
        # if type(lDat) is list:
            # kh = lDat[0].plot_combine(lDat[1:], sharex=sharex, bck=bck)
        # else:
            # kh = lDat.plot(bck=bck)
        # return kh

    # def plot_combine(self, lquant, lData=None, X=None,
                     # ref1d=None, ref2d=None,
                     # remap=False, res=0.01, interp_space=None,
                     # sharex=False, bck=True):
        # """ plot combining several quantities from the Plasma2D itself and
        # optional extra list of Data instances """
        # lDat = self.get_Data(lquant, X=X, remap=remap,
                             # ref1d=ref1d, ref2d=ref2d,
                             # res=res, interp_space=interp_space)
        # if lData is not None:
            # if type(lDat) is list:
                # lData = lDat[1:] + lData
            # else:
                # lData = lDat[1:] + [lData]
        # kh = lDat[0].plot_combine(lData, sharex=sharex, bck=bck)
        # return kh
