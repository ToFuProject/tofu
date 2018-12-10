# -*- coding: utf-8 -*-

# Built-in
import os
import itertools as itt
import warnings
#from abc import ABCMeta, abstractmethod

# Common
import numpy as np
import matplotlib.pyplot as plt

# tofu
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.data._plot as _plot
    import tofu.data._def as _def
except Exception:
    from . import _plot as _plot
    from . import _def as _def

__all__ = ['Data1D','Data2D','DataSpectro']


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
    ind = np.zeros((nRef,),dtype=bool)
    if v is None:
        ind = ~ind
    elif C0 or C1:
        if C0:
            v = np.r_[v]
        for vv in v:
            ind[np.nanargmin(np.abs(ref-vv))] = True
    elif C2 or C3:
        c0 = len(v)==2 and all([type(vv) in ltypes for vv in v])
        c1 = all([(type(vv) is type(v)
                   and all([type(vvv) in ltypes for vvv in vv]))
                  for vv in v])
        assert c0!=c1
        if c0:
            v = [v]
        for vv in v:
            ind = ind | ((ref>=v[0]) & (ref<=v[1]))
        if C3:
            ind = ~ind
    return ind




#############################################
#       class
#############################################

class Data(utils.ToFuObject):

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'Type':'1D',
                   'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dtreat':{'order':['mask','interp_t','interp_ch','data0','fft',
                                'indt', 'indch']}}

    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, **kwdargs):
        # Python 2
        super(Data,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Data._ddef)
        #cls._dplot = copy.deepcopy(Struct._dplot)
        #cls._set_color_ddef(cls._color)


    def __init__(self, data=None, t=None, X=None, lamb=None,
                 dchans=None, dunits=None,
                 Id=None, Name=None, Exp=None, shot=None, Diag=None,
                 dextra=None, lCam=None, config=None, Type=None,
                 fromdict=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(Data,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Data,self)._reset()
        self._ddataRef = dict.fromkeys(self._get_keys_ddataRef())
        self._ddata = dict.fromkeys(self._get_keys_ddata())
        self._dtreat = dict.fromkeys(self._get_keys_dtreat())
        self._dunits = dict.fromkeys(self._get_keys_dunits())
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dchans = dict.fromkeys(self._get_keys_dchans())
        self._dextra = dict.fromkeys(self._get_keys_dextra())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None, Type=None,
                               Diag=None, include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot = Id.Name, Id.Exp, Id.shot
            Type, Diag = Id.Type, Id.Diag
        assert type(Name) is str
        assert type(Diag) is str
        assert type(Exp) is str
        if Type is None:
            Type = cls._def['Id']['Type']
        assert Type in ['1D','2D','1DSpectral','2DSpectral']
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
        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot, 'Type':Type,
                        'include':include})
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
        largs = []
        return largs

    @staticmethod
    def _get_largs_dunits():
        largs = ['dunits']
        return largs

    @staticmethod
    def _get_largs_dgeom():
        largs = ['lCam','config']
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
                                  lamb=None, indtlamb=None, indXlamb=None):
        assert data is not None
        data = np.asarray(data).sqeeze()
        if t is not None:
            t = np.asarray(t).sqeeze()
        if X is not None:
            X = np.asarray(X).sqeeze()
        if indtX is not None:
            indtX = np.asarray(indtX).sqeeze()
        if lamb is not None:
            lamb = np.asarray(lamb).sqeeze()
        if indtlamb is not None:
            indtlamb = np.asarray(indtlamb).sqeeze()
        if indXlamb is not None:
            indXlamb = np.asarray(indXlamb).sqeeze()

        ndim = data.ndim
        assert ndim in [2,3]
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
                if 'spectral' in self.Id.Type.lower():
                    X = np.array([0])
                    lamb = np.arange(0,n1)
                    data = data.reshape((nt,1,n1))
                else:
                    X = np.arange(0,n1)
            elif lC[0]:
                assert 'spectral' in self.Id.Type.lower()
                X = np.array([0])
                data = data.reshape((nt,1,n1))
                assert lamb.ndim in [1,2]
                if lamb.ndim==1:
                    assert lamb.size==n1
                elif lamb.ndim==2:
                    assert lamb.shape[1]==n1
            else:
                assert 'spectral' not in self.Id.Type.lower()
                assert X.ndim in [1,2]
                if X.ndim==1:
                    assert X.size==n1
                elif X.ndim==2:
                    assert X.shape[1]==n1
        else:
            assert 'spectral' in self.Id.Type.lower()
            n2 = data.shape[2]
            lC = [X is None, lamb is None]
            if lC[0]:
                X = np.arange(0,n1)
            else:
                assert X.ndim in [1,2]
                if X.ndim==1:
                    assert X.size==n1
                else:
                    assert X.shape[1]==n1

            if lC[1]:
                lamb = np.arange(0,n2)
            else:
                assert lamb.ndim in [1,2]
                if lamb.ndim==1:
                    assert lamb.size==n2
                else:
                    assert lamb.shape[1]==n2

        # Get shapes
        if data.ndim==2:
            (nt,nX), nlamb = data.shape, 0
        else:
            nt, nX, nlamb = data.shape
        nnX = 1 if X.ndim==1 else X.shape[0]
        nnlamb = 1 if lamb.ndim==1 else lamb.shape[0]

        # Check indices
        if indtX is not None:
            assert indtX.shape==(nt,)
            assert inp.argmin(indtX)>=0 and np.argmax(indtX)=<nnX
        lC = [indtlamb is None, indXlamb is None, indtXlamb is None]
        assert lC[2] or (~lC[2] and np.sum(lC[:2])==2)
        if lC[2]:
            if ~lC[0]:
                assert indtlamb.shape==(nt,)
                assert inp.argmin(indtlamb)>=0 and np.argmax(indtlamb)=<nnlamb
            if ~lC[1]:
                assert indXlamb.shape==(nX,)
                assert inp.argmin(indXlamb)>=0 and np.argmax(indXlamb)=<nnlamb
        else:
            assert indtXlamb.shape==(nt,nX)
            assert inp.argmin(indtXlamb)>=0 and np.argmax(indtXlamb)=<nnlamb

        l = [data, t, X, lamb, nt, nX, nlamb, nnX, nnlamb,
             indtX, indtlamb, indXlamb, indtXlamb]
        return l


    def _checkformat_inputs_dunits(self, dunits=None):
        if dunits is None:
            dunits = {}
        assert type(dunits) is dict
        lk = ['data','t','X']
        for k in lk:
            if not k in dunits.keys():
                dunits[k] = 'a.u.'
            assert type(dunits[k]) is str
        if 'spectral' in self.Id.Type.lower():
            if 'lamb' not in dunits.keys():
                dunits['lamb'] = 'a.u.'
            assert type(dunits['lamb']) is str
        return dunits

    def _checkformat_inputs_dtreat(self, dtreat=None):
        if dtreat is None:
            dtreat = {}
        assert type(dtreat) is dict
        if 'order' not in dtreat.keys():
            dtreat['order'] = list(self._ddef['dtreat']['order'])
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
            for dd in ['1d','2d']:
                if dd in self.Id.Type.lower():
                    lc = [dd in cc.Id.Type.lower()i for cc in lCam]
                    if not all(lc):
                        msg = "The following cameras have wrong class (%s)"%dd
                        lm = ['%s: %s'%s(cc.Id.Name,cc.Id.Cls) for cc in lCam]
                        msg += "\n    " + "\n    ".join(lm)
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
            if not nR==self._ddata['nX']:
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
                assert arr.size==self._ddata['nX']
                dchans[k] = arr
        return dchans

    def _checkformat_inputs_dextra(self, dextra=None):
        assert dextra is None or isinstance(dextra,dict)
        if dextra is not None:
            for k in dextra.keys():
                assert isinstance(dextra[k],dict)
                assert 't' in dextra[k].keys()

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_ddataRef():
        lk = ['data', 't', 'X', 'lamb', 'nt', 'nX', 'nlamb', 'nnX', 'nnlamb',
              'indtX', 'indtlamb', 'indXlamb', 'indtXlamb']

    @staticmethod
    def _get_keys_ddata():
        lk = ['data', 't', 'X', 'lamb', 'nt', 'nX', 'nlamb', 'nnX', 'nnlamb',
              'indtX', 'indtlamb', 'indXlamb', 'indtXlamb']

    @staticmethod
    def _get_keys_dtreat():
        lk = ['indt','indch']

    @staticmethod
    def _get_keys_dunits():
        lk = ['data','t','X','lamb']

    @staticmethod
    def _get_keys_dgeom():
        lk = ['config', 'lCam', 'nC']
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

    def _init(self, data=None, t=None, X=None, lamb=None, dchans=None,
              dunits=None, dextra=None, lCam=None, config=None, **kwdargs):
        largs = self._get_largs_ddataRef()
        kwddataRef = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dunits()
        kwdunits = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dgeom()
        kwdgeom = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dchans()
        kwdchans = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dextra()
        kwdextra = self._extract_kwdargs(locals(), largs)
        self._set_ddataRef(**kwddataRef)
        self.compute_data()
        self._set_dunits(**kwdunits)
        self._set_dgeom(**kwdgeom)
        self.set_dchans(**kwdchans)
        self.set_dextra(**kwdextra)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dataRef(self, data=None, t=None,
                     X=None, indtX=None,
                     lamb=None, indtlamb=None, indXlamb=None, indtXlamb=None):
        kwdargs = locals()
        del kwdargs['self']
        lout = self._checkformat_inputs_ddataRef(**kwdargs)                                                 )
        data, t, X, lamb, nt, nX, nlamb, nnX, nnlamb = lout[:9]
        indtX, indtlamb, indXlamb, indtXlamb = lout[9:]

        self._dataRef = {'data':data, 't':t, 'X':X, 'lamb':lamb,
                         'nt':nt, 'nX':nX, 'nlamb':nlamb,
                         'nnX':nnX, 'nnlamb':nnlamb,
                         'indtX':indtX, 'indtlamb':indtlamb,
                         'indXlamb':indXlamb, 'indtXlamb':indtXlamb}

    def set_dtreat(self, dtreat=None):
        dtreat = self._checkformat_inputs_dtreat(dtreat=dtreat)
        self._dtreat = dtreat

    def _set_dunits(self, dunits=None):
        dunits = self._checkformat_inputs_dunits(dunits=dunits)
        self._dunits = dunits

    def _set_dgeom(self, lCam=None, config=None):
        config, lCam, nC = self._checkformat_inputs_dgeom(lCam=lCam,
                                                          config=config)
        self._dgeom = {'lCam':lCam, 'nC':nC, 'config':config}

    def set_dchans(self, dchans=None):
        dchans = self._checkformat_inputs_dchans(dchans=dchans)
        self._dchans = dchans

    def set_dextra(self, dextra=None):
        dextra = self._checkformat_inputs_dextra(dextra=dextra)
        self._dextra = dextra

    ###########
    # strip dictionaries
    ###########

    def _strip_ddata(self, strip=0):
        if self._dstrip['strip']==strip:
            return

        if strip in [0,2]:
            self.compute_data()
        elif strip in [1,3]:
            self.clear_data()

    def _strip_dgeom(self, strip=0, force=False):
        if self._dstrip['strip']==strip:
            return

        if strip in [0,1] and self._dstrip['strip'] in [2,3]:
            lC, config = None, None
            if self._dgeom['lCam'] is not None:
                assert type(self._dgeom['lCam']) is list
                assert all([type(ss) is str for ss in self._dgeom['lCam']])
                lC = []
                for ii in range(0,len(self._dgeom['lCam']))
                    lC.append(utils.load(self._dgeom['lCam'][ii]))

            elif self._dgeom['config'] is not None:
                assert type(self._dgeom['config']) is str
                config = utils.load(self._dgeom['config'])

            self._set_dgeom(lCam=lC, config=config)

        elif strip in [2,3] and self._dstrip['strip'] in [0,1]:
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
                 1: clear data
                 2: dgeom all pathfile (=> tf.geom.Rays.strip(-1))
                 3: dgeom all pathfile + data clear
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0):
        # super()
        super(Data,self).strip(strip=strip)

    def _strip(self, strip=0):
        self._strip_ddata(strip=strip)
        self._strip_dgeom(strip=strip)

    def _to_dict(self):
        dout = {'ddataRef':{'dict':self._ddataRef, 'lexcept':None},
                'ddata':{'dict':self._ddata, 'lexcept':None},
                'dtreat':{'dict':self._dtreat, 'lexcept':None},
                'dunits':{'dict':self._dunits, 'lexcept':None},
                'dgeom':{'dict':self._dgeom, 'lexcept':None},
                'dchans':{'dict':self._dchans, 'lexcept':None},
                'dextra':{'dict':self._dextra, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._ddataRef.update(**fd['ddataRef'])
        self._ddata.update(**fd['ddata'])
        self._dtreat.update(**fd['dtreat'])
        self._dunits.update(**fd['dunits'])
        self._dgeom.update(**fd['dgeom'])
        self._dchans.update(**fd['dchans'])
        self._dextra.update(**fd['dextra'])


    ###########
    # properties
    ###########

    @property
    def ddataRef(self):
        return self._ddataRef
    @property
    def ddata(self):
        return self._ddata
    @property
    def dtreat(self):
        return self._dtreat
    @property
    def dunits(self):
        return self._dunits
    @property
    def dchans(self):
        return self._dchans
    @property
    def dgeom(self):
        return self._dgeom
    @property
    def dextra(self):
        return self._dextra


    @property
    def data(self):
        return self._get_data()

    @property
    def t(self):
        if self._ddata['t'] is None:
            t = self._ddataRef['t']
        else:
            t = self._ddata['t'][self.indt]
        return t




    @property
    def indt(self):
        if self._indt is None:
            return np.ones((self._Ref['nt'],),dtype=bool)
        else:
            return self._indt
    @property
    def indch(self):
        if self._indch is None:
            return np.ones((self._Ref['nch'],),dtype=bool)
        else:
            return self._indch
    @property
    def mask(self):
        return self._mask
    @property
    def nt(self):
        if self._nt is None:
            return int(np.sum(self.indt))
        else:
            return self._nt
    @property
    def nch(self):
        return int(np.sum(self.indch))
    @property
    def data0(self):
        return self._data0
    @property
    def LCam(self):
        return self._get_LCam()
    @property
    def treatment(self):
        d = {'indt':self.indt, 'indch':self.indch,
             'DtRef':self._DtRef, 'fft':self._fft}
        return d
    @property
    def dextra(self):
        return self._dextra


    ###########
    # public methods
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
        indt = _select_ind(t, self._ddataRef['t'], self._ddataRef['nt'])
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indt(self, t=None, indt=None):
        """ Store the desired index array for the time vector

        If an array of indices (refering to self.ddataRef['t'] is not provided,
        uses self.select_t(t=t) to produce it

        """
        lC = [indt is None,t is None]
        assert np.sum(lC)>=1
        if all(lC):
            ind = None
        elif C[0]:
            ind = self.select_t(t=t, out=bool)
        elif C[1]:
            ind = _format_ind(indt, n=self._Ref['nt'])
        self._dtreat['indt'] = ind

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
        lC = [all(lc), all(lc[:2]) and ~lc[2],
              ~lc[0] and all(lc[1:]), ~any(lc[:2]) and lc[2]]
        assert np.sum(lC)==1

        if lC[0]:
            # get all channels
            ind = np.ones((self._ddataRef['nX'],),dtype=bool)

        elif lC[1]:
            # get touch
            if self._dgeom['lCam'] is None:
                msg = "self.dgeom['lCam'] must be set to use touch !"
                raise Exception(msg)
            if any([type(cc) is str for ss in self._dgeom['lCam']]):
                msg = "self.dgeom['lCam'] contains pathfiles !"
                msg += "\n  => Run self.strip(0)"
                raise Exception(msg)
            ind = []
            for cc in self._dgeom['LCam']:
                ind.append(cc.select(touch=touch, log=log, out=bool))
            if len(ind)==1:
                ind = ind[0]
            else:
                ind = np.concatenate(tuple(ind))

        elif lC[2]:
            # get values on X
            if self._ddataRef['nnX']==1:
                ind = _select_ind(val, self._ddataRef['X'], self._ddataRef['nX'])
            else:
                ind = np.zeros((self._ddataRef['nt'],self._ddataRef['nX']),dtype=bool)
                for ii range(0,self._ddataRef['nnX']):
                    iind = self._ddataRef['indtX']==ii
                    ind[iind,:] =  _select_ind(val, self._ddataRef['X'],
                                               self._ddataRef['nX'])[np.newaxis,:]

        else:
            assert type(key) is str and key in self._dchans['dchans'].keys()
            ltypes = [str,int,float,np.int64,np.float64]
            C0 = type(val) in ltypes
            C1 = type(val) in [list,tuple,np.ndarray]
            assert C0 or C1
            if C0:
                val = [val]
            else:
                assert all([type(vv) in ltypes for vv in val])
            ind = np.vstack([self.Ref['dchans'][key]==ii for ii in val])
            if log=='any':
                ind = np.any(ind,axis=0)
            elif log=='all':
                ind = np.all(ind,axis=0)
            else:
                ind = ~np.any(ind,axis=0)
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indch(self, indch=None):
        """ Store the desired index array for the channels

        If None => all channels
        Must be a 1d array

        """
        if indch is not None:
            indch = np.asarray(indch)
            assert indch.ndim==1
            indch = _format_ind(indch, n=self._ddataRef['nX'])
        self._dtreat['indch'] = indch

    def set_mask(self, ind=None, val=np.nan):
        assert ind is None or hasattr(ind,'__iter__')
        assert type(val) in [int,float,np.int64,np.float64]
        if ind is not None:
            ind = _format_ind(ind, n=self._ddataRef['nX'])
        self._dtreat['mask-ind'] = ind
        self._dtreat['mask-val'] = val

    def set_data0(self, data0=None, Dt=None, indt=None):
        assert self._ddataRef['nt']>1, "Useless if only one data slice !"
        C = [data0 is None, Dt is None, indt is None]
        assert np.sum(C)>=2
        if data0 is not None:
            data0 = np.asarray(data0).ravel()
            assert data0.shape==(self._ddataRef['nX'],)
            Dt, indt = None, None
        else:
            if indt is not None:
                indt = _format_ind(indt, n=self._ddataRef['nt'])
            else:
                indt = self.select_t(t=Dt, out=bool)
            if np.any(indt):
                data0 = self._ddataRef['data'][indt,:]
                if np.sum(indt)>1:
                    data0 = np.nanmean(data,axis=0)
        self._dtreat['data0-indt'] = indt
        self._dtreat['data0-data'] = data0
        self._dtreat['data0-Dt'] = Dt

    def set_interp_ch(self, indch=None):
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
        assert indch is None or type(indch) in [np.ndarray, list, dict]
        if isinstance(indch,dict):
            C = [type(k) is int and k<self._ddataRef['nt'] for k in indch.keys()]
            assert all(C)
            for k in indch.keys():
                assert hasattr(indch[k],'__iter__')
                indch[k] = _format_ind(indch[k], n=self._ddataRef['nX'])
        else:
            indch = np.asarray(indch)
            assert indch.ndim==1:
            indch = _format_ind(indch, n=self._ddataRef['nX'])
        self._dtreat['interp-indch'] = indch
        self._ddata['uptodate'] = False

    def set_interp_t(self, indt=None):
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
        assert indt is None or type(indt) in [np.ndarray, list, dict]
        if isinstance(indt,dict):
            C = [type(k) is int and k<self._ddataRef['nX'] for k in indt.keys()]
            assert all(C)
            for k in indt.keys():
                assert hasattr(indt[k],'__iter__')
                indt[k] = _format_ind(indt[k], n=self._ddataRef['nt'])
        else:
            indt = np.asarray(indt)
            assert indt.ndim==1:
            indt = _format_ind(indt, n=self._ddataRef['nt'])
        self._dtreat['interp-indt'] = indt
        self._ddata['uptodate'] = False

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
            if data.shape==data0.shape:
                data = data - data0
            elif data.ndim==2:
                data = data - data0[np.newaxis,:]
            if data.ndim==3:
                data = data - data0[np.newaxis,:,np.newaxis]
        return data

    @staticmethod
    def _fft(data, df=None, harm=None, dfEx=None, harmEx=None):



    @staticmethod
    def _indt(data, indt):
        if indt is not None:
            if data.ndim==2:
                data = data[indt,:]
            elif data.ndim==3:
                data = data[indt,:,:]
        return data

    @staticmethod
    def _indch(data, indch):
        if indt is not None:
            if data.ndim==2:
                data = data[:,indch]
            elif data.ndim==3:
                data = data[:,indch,:]
        return data

    def _get_data(self):
        if self._ddata['uptodate']:
            data = self._ddata['data']
        else:
            data = self._get_treated_data()
            self._ddata['data'] = data
            self._ddata['uptodate'] = True
        return data

    def set_dtreat_order(self, order=None):
        """ Set the order in which the data treatment should be performed

        Provide an ordered list of keywords indicating the order in which
         you wish the data treatment steps to be performed.
        Each keyword corresponds to a step.
            - 'mask' :
            - 'interp_t' :
            - 'interp_ch' :
            - 'data0' :
            - 'fft' :
            - 'indt' :
            - 'indch' :

        All steps are performed on the stored reference self.dataRef['data']
        Thus, the time and channels restriction must be the last 2 steps
        """
        if order is None:
            order = list(self._ddef['dtreat']['order'])
        assert type(order) is list and all([type(ss) is str for ss in order])
        C = [ss in ['indt','indch'] for ss in self._dtreat['order'][-2:]]
        if not all(C):
            msg = "indt and indch must be the last 2 treatment steps !"
            raise Exception(msg)
        self._dtreat['order'] = order

    def _get_treated_data(self):
        """ Produce a working copy of the data based on the treated reference

        The reference data is always stored and untouched in self.ddataRef
        You always interact with self.data, which returns a working copy.
        That working copy is the reference data, eventually treated along the
            lines defined (by the user) in self.dtreat
        By reseting the treatment (self.reset()) all data treatment is
        cancelled and the working copy returns the reference data.

        """
        indt, indch = self._dtreat['interp_indt'], self._dtreat['interp_indch']
        C0 = indch is None
        C1 = indt is None
        C2 = self._dtreat['fft-df'] is None
        C3 = self._dtreat['svd-modes'] is None
        if np.sum([C2,C3])==0:
            msg = "You cannot do both a fft and svd filtering, choose one"
            msg += "\n  => remove fft by self.set_fft()"
            msg += "\n  => remove svd by self.set_svd()"
            raise Exception(msg)
        d = self._ddataRef['data'].copy()
        for kk in self._dtreat['order']:
            if kk=='mask' and self._dtreat['mask-ind'] is not None:
                d = self._mask(d, self._dtreat['mask-ind'],
                               self._dtreat['mask-val'])
            if kk=='interp_t':
                d = self._interp_indt(d, self._dtreat['interp-indt'],
                                      self._ddataRef['t'])
            if kk=='interp_ch':
                d = self._interp_indt(d, self._dtreat['interp-indch'],
                                      self._ddataRef['X'])
            if kk=='data0':
                d = self._data0(d, self._dtreat['data0-data'],
                                self._dtreat['data-val'])
            if kk=='fft':
                d = self._fft(d, self._dtreat['fft-df'], self._dtreat['fft-'])
            if kk=='indt':
                d = self._indt(d, self._dtreat['indt'])
            if kk=='indch':
                d = self._indch(d, self._dtreat['indt'])
        return d



    def get_treatment(self):




    def reset_treatment(self):
        self.set_indch()
        self.set_indt()
        self.set_data0()
        self.set_interp_t()
        self.set_fft()
        self._set_data()
        self._ddata = dict.fromkeys(self._get_keys_ddata())
        self.compute_data()





    def plot_spectrogram(self, fmin=None,
                         method='scipy-fourier',
                         window='hann', detrend='linear',
                         nperseg=None, noverlap=None, stft=False,
                         boundary='constant', padded=True, wave='morlet'):
        """ Return the power spectrum density for each channel

        The power spectrum density is computed with the chosen method

        Parameters
        ----------
        fmin :  None / float
            The minimum frequency of interest
            If None, set to 5/T, where T is the whole time interval
            Used to constrain the number of points per window
        method : str
            Flag indicating which method to use for computation:
                - 'scipy-fourier':  uses scipy.signal.spectrogram()
                    (windowed fast fourier transform)
                - 'scipy-stft':     uses scipy.signal.stft()
                    (short time fourier transform)
                - 'scipy-wavelet':  uses scipy.signal.cwt()
                    (continuous wavelet transform)
            The following keyword args are fed to one of these scipy functions
            See the corresponding online scipy documentation for details on the
            function and its arguments
        window : None / str / tuple
            If method='scipy-fourier'
            Flag indicating which type of window to use
        detrend : None / str
            If method='scipy-fourier'
            Flag indicating whether and how to remove the trend of the signal
        nperseg :   None / int
            If method='scipy-fourier'
            Number of points to the used for each window
        noverlap:
            If method='scipy-fourier'
            Number of points on which successive windows should overlap
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
            d
        f:      np.ndarray
            d
        lspect: np.ndarray
            d

        """


        tf, f, lspect = _comp.spectrogram(self.data, self.t, method=method)
        return tf, f, lspect


    def plot_spectrogram():
        """   """
        tf, f, lspect =




















############################################ To be finished


    def dchans(self, key=None):
        """ List the channels selected by self.indch

        Return their indices (default) or the chosen criterion
        """
        if self.geom is None or self.geom['LCam'] is None:
            dchans = None
        elif self._Ref['dchans']=={}:
            dchans = self._Ref['dchans']
        else:
            assert key in [None]+list(self._Ref['dchans'].keys())
            ind = self.indch.nonzero()[0]
            if key is None:
                lK = self._Ref['dchans'].keys()
                dchans = {}
                for kk in lK:
                    if self._Ref['dchans'][kk].ndim==1:
                        dchans[kk] = self._Ref['dchans'][kk][ind]
                    else:
                        dchans[kk] = self._Ref['dchans'][kk][:,ind]
            else:
                if self._Ref['dchans'][key].ndim==1:
                    dchans = self._Ref['dchans'][key][ind]
                else:
                    dchans = self._Ref['dchans'][key][:,ind]
        return dchans

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






    def _set_data(self):
        if self._fft is None and self._interp_t is None:
            d = None
        else:
            d = self._calc_data_core()
            # Get fft
            if self._fft is not None:
                d = _comp.get_fft(d, **self._fft)
            if self._interp_t is not None:
                t = self._interp_t
                d = np.vstack([np.interp(t, self.t, d[:,ii],
                                         left=np.nan, right=np.nan)
                               for ii in range(d.shape[1])]).T
                self._t = t
                self._nt = t.size
            else:
                self._t = None
                self._nt = None
        self._data = d


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

    def plot(self, key=None, invert=None, plotmethod='imshow',
             cmap=plt.cm.gray, ms=4, ntMax=None, nchMax=None, nlbdMax=3,
             Bck=True, fs=None, dmargin=None, wintit=None, tit=None,
             vmin=None, vmax=None, normt=False, draw=True, connect=True):
        """ Plot the data content in a predefined figure  """
        KH = _plot.Data_plot(self, key=key, invert=invert, Bck=Bck,
                             ntMax=ntMax, nchMax=nchMax, nlbdMax=nlbdMax,
                             plotmethod=plotmethod, cmap=cmap, ms=ms,
                             fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                             vmin=vmin, vmax=vmax, normt=normt,
                             draw=draw, connect=connect)
        return KH

    def plot_compare(self, lD, key=None, invert=None, plotmethod='imshow',
                     cmap=plt.cm.gray, ms=4, ntMax=None, nchMax=None, nlbdMax=3,
                     Bck=True, indref=0, fs=None, dmargin=None,
                     vmin=None, vmax=None, normt=False,
                     wintit=None, tit=None, fontsize=None,
                     draw=True, connect=True):
        """ Plot several Data instances of the same diag

        Useful to compare :
                - the diag data for 2 different shots
                - experimental vs synthetic data for the same shot

        """
        C0 = isinstance(lD,list)
        C0 = C0 and all([issubclass(dd.__class__,Data) for dd in lD])
        C1 = issubclass(lD.__class__,Data)
        assert C0 or C1, 'Provided first arg. must be a tf.data.Data or list !'
        lD = [lD] if C1 else lD
        KH = _plot.Data_plot([self]+lD, key=key, invert=invert, Bck=Bck,
                             ntMax=ntMax, nchMax=nchMax, nlbdMax=nlbdMax,
                             plotmethod=plotmethod, cmap=cmap, ms=ms,
                             fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                             vmin=vmin, vmax=vmax, normt=normt,
                             fontsize=fontsize, indref=indref,
                             draw=draw, connect=connect)
        return KH

    def plot_combine(self, lD, key=None, invert=None, plotmethod='imshow',
                     cmap=plt.cm.gray, ms=4, ntMax=None, nchMax=None, nlbdMax=3,
                     Bck=True, indref=0, fs=None, dmargin=None,
                     vmin=None, vmax=None, normt=False,
                     wintit=None, tit=None, fontsize=None,
                     draw=True, connect=True):
        """ Plot several Data instances of different diags

        Useful to visualize several diags for the same shot

        """
        C0 = isinstance(lD,list)
        C0 = C0 and all([issubclass(dd.__class__,Data) for dd in lD])
        C1 = issubclass(lD.__class__,Data)
        assert C0 or C1, 'Provided first arg. must be a tf.data.Data or list !'
        lD = [lD] if C1 else lD
        KH = _plot.Data_plot_combine([self]+lD, key=key, invert=invert, Bck=Bck,
                                     ntMax=ntMax, nchMax=nchMax, nlbdMax=nlbdMax,
                                     plotmethod=plotmethod, cmap=cmap, ms=ms,
                                     fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                                     vmin=vmin, vmax=vmax, normt=normt,
                                     indref=indref, fontsize=fontsize,
                                     draw=draw, connect=connect)
        return KH







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

    # dunits, Id, Exp, shot, Diag, SavePath
    dunits = _compare_([dd._dunits for dd in ld], null={})
    Id = ' '.join([dd.Id.Name for dd in ld])
    Exp = _compare_([dd.Id.Exp for dd in ld])
    shot = _compare_([dd.Id.shot for dd in ld])
    Diag = _compare_([dd.Id.Diag for dd in ld])
    SavePath = _compare_([dd.Id.SavePath for dd in ld])

    return t, LCam, dchans, dunits, Id, Exp, shot, Diag, SavePath




def _recreatefromoperator(d0, other, opfunc):
    if type(other) in [int,float,np.int64,np.float64]:
        d = opfunc(d0.data, other)

        #  Fix LCam and dchans
        #t, LCam, dchans = d0.t, d0._get_LCam(), d0.dchans(d0.indch)
        out = _extractCommonParams([d0, d0])
        t, LCam, dchans, dunits, Id, Exp, shot, Diag, SavePath = out

        #dunits = d0._dunits
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
        t, LCam, dchans, dunits, Id, Exp, shot, Diag, SavePath = out
    else:
        raise NotImplementedError

    kwdargs = dict(t=t, dchans=dchans, LCam=LCam, dunits=dunits,
                   Id=Id, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath)

    if '1D' in d0.Id.Cls:
        data = Data1D(d, **kwdargs)
    elif '2D' in d0.Id.Cls:
        data = Data2D(d, **kwdargs)
    else:
        data = Data(d, **kwdargs)
    return data







#####################################################################
#               Data1D and Data2D
#####################################################################


class Data1D(Data):
    """ Data object used for 1D cameras or list of 1D cameras  """
    def __init__(self, data=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None, dextra=None,
                 LCam=None, Ves=None, LStruct=None, fromdict=None,
                 SavePath=os.path.abspath('./')):
        Data.__init__(self, data, t=t, dchans=dchans, dunits=dunits,
                 Id=Id, Exp=Exp, shot=shot, Diag=Diag, dextra=dextra, CamCls='1D',
                 LCam=LCam, Ves=Ves, LStruct=LStruct, fromdict=fromdict, SavePath=SavePath)



class Data2D(Data):
    """ Data object used for 1D cameras or list of 1D cameras  """
    def __init__(self, data=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None, dextra=None,
                 LCam=None, Ves=None, LStruct=None, X12=None, fromdict=None,
                 SavePath=os.path.abspath('./')):
        Data.__init__(self, data, t=t, dchans=dchans, dunits=dunits,
                      Id=Id, Exp=Exp, shot=shot, Diag=Diag, dextra=dextra,
                      LCam=LCam, Ves=Ves, LStruct=LStruct, CamCls='2D',
                      fromdict=fromdict, SavePath=SavePath)
        self.set_X12(X12)

    def set_X12(self, X12=None):
        X12 = X12 if (self.geom is None or self.geom['LCam'] is None) else None
        if X12 is not None:
            X12 = np.asarray(X12)
            assert X12.shape==(2,self.Ref['nch'])
        self._X12 = X12

    def get_X12(self, out='1d'):
        if self._X12 is None:
            C0 = self.geom is not None
            C0 = C0 and self.geom['LCam'] is not None
            msg = "X12 must be set for plotting if LCam not provided !"
            assert C0, msg
            X12, DX12 = self.geom['LCam'][0].get_X12(out=out)
        else:
            X12 = self._X12
            if out.lower()=='2d':
                x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
                X12 = [x1u,x2u,ind]
            else:
                DX12 = None
        return X12, DX12









#####################################################################
#               DataSpectro
#####################################################################








class DataSpectro(Data):
    """ data should be provided in (nt,nlamb,chan) format  """

    def __init__(self, data=None, lamb=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None, dMag=None,
                 LCam=None, CamCls='1D', fromdict=None,
                 SavePath=os.path.abspath('./')):

        self._Done = False
        if fromdict is None:
            msg = "Provide either dchans or LCam !"
            assert np.sum([dchans is None, LCam is None])>=1, msg
            self._set_dataRef(data, lamb=lamb, t=t, dchans=dchans, dunits=dunits)
            self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
            self._set_LCam(LCam=LCam, CamCls=CamCls)
            self._dMag = dMag
        else:
            self._fromdict(fromdict)
        self._Done = True

    @property
    def indlamb(self):
        if self._indlamb is None:
            return np.ones((self._Ref['nlamb'],),dtype=bool)
        else:
            return self._indlamb
    @property
    def lamb(self):
        if self._Ref['lamb'] is None:
            return None
        elif self._lamb is None:
            return self._Ref['lamb'][self.indlamb]
        else:
            return self._lamb
    @property
    def nlamb(self):
        return int(np.sum(self.indlamb))

    def _check_inputs(self, Id=None, data=None, lamb=None, t=None, dchans=None,
                      dunits=None, Exp=None, shot=None, Diag=None, LCam=None,
                      CamCls=None, SavePath=None):
        _DataSpectro_check_inputs(data=data, lamb=lamb, t=t,
                                  dchans=dchans, LCam=LCam)
        _Data_check_inputs(Id=Id, dunits=dunits,
                           Exp=Exp, shot=shot, Diag=Diag,
                           CamCls=CamCls, SavePath=SavePath)

    def _set_dataRef(self, data, lamb=None, t=None, dchans=None, dunits=None):
        self._check_inputs(data=data, lamb=lamb, t=t, dchans=dchans, dunits=dunits)
        if data.ndim==1:
            nt, nlamb, nch = 1, data.size, 1
            data = data.reshape((nt,nlamb,nch))
            if t is not None:
                t = np.asarray(t) if hasattr(t,'__iter__') else np.asarray([t])
            if lamb is not None:
                lamb = np.asarray(lamb)
        else:
            nt, nlamb, nch= data.shape
            t = np.asarray(t)
            lamb = np.asarray(lamb)
        self._Ref = {'data':data, 'lamb':lamb, 't':t,
                     'nt':nt, 'nlamb':nlamb, 'nch':nch}
        dchans = {} if dchans is None else dchans
        lK = sorted(dchans.keys())
        dchans = dict([(kk,np.asarray(dchans[kk])) for kk in lK])
        self._Ref['dchans'] = dchans

        self._dunits = {} if dunits is None else dunits
        self._data, self._t, self._lamb = None, None, None
        self._indt, self._indlamb, self._indch = None, None, None
        self._data0 = {'data':None,'t':None,'Dt':None}
        self._fft, self._interp_t = None, None
        self._indt_corr, self._indch_corr = None, None

    def select_lamb(self, lamb=None, out=bool):
        assert out in [bool,int]
        C0 = type(lamb) in [int,float,np.int64,np.float64]
        C1 = type(lamb) in [list,np.ndarray] and len(lamb)==2
        C2 = type(lamb) is tuple and len(lamb)==2
        assert lamb is None or C0 or C1 or C2
        ind = np.zeros((self._Ref['nlamb'],),dtype=bool)
        if lamb is None or self._Ref['lamb'] is None:
            ind = ~ind
        elif C0:
            ind[np.nanargmin(np.abs(self._Ref['lamb']-lamb))] = True
        elif C1 or C2:
            ind[(self._Ref['lamb']>=lamb[0]) & (self._Ref['lamb']<=lamb[1])] = True
            if C2:
                ind = ~ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indtlamb(self, indlamb=None, lamb=None):
        C = [indlamb is None, lamb is None]
        assert np.sum(C)>=1
        if all(C):
            ind = np.ones((self._Ref['nlamb'],),dtype=bool)
        elif C[0]:
            ind = self.select_lamb(lamb=lamb, out=bool)
        elif C[1]:
            ind = _format_ind(indlamb, n=self._Ref['nlamb'])
        self._indlamb = ind

    def set_data0(self, data0=None, Dt=None, indt=None):
        assert self._Ref['nt']>1, "Useless if only one data slice !"
        C = [data0 is None, Dt is None, indt is None]
        assert np.sum(C)>=2
        if all(C):
            data, t, indt = None, None, None
        elif data0 is not None:
            assert np.asarray(data0).shape==(self._Ref['nlamb'],self._Ref['nch'])
            data = np.asarray(data0)
            Dt, indt = None, None
        else:
            if indt is not None:
                indt = _format_ind(indt, n=self._Ref['nt'])
            else:
                indt = self.select_t(t=Dt, out=bool)
            if indt is not None and np.any(indt):
                data = self._Ref['data'][indt,:,:]
                if np.sum(indt)>1:
                    data = np.nanmean(data,axis=0)
                if self._Ref['t'] is not None:
                    Dt = [np.nanmin(self._Ref['t'][indt]),
                          np.nanmax(self._Ref['t'][indt])]
                else:
                    Dt = None
            else:
                data = None
                Dt = None
        self._data0 = {'indt':indt,'data':data, 'Dt':Dt}

    def _calc_data_core(self):
        # Get time interval
        d = self._Ref['data'][self.indt,:,:]
        # Substract reference time data
        if self._data0['data'] is not None:
            d = d - self._data0['data'][np.newaxis,:,:]
        # Get desired wavelenght interval
        d = d[:,self.indlamb,:]
        # Get desired channels
        d = d[:,:,self.indch]
        return d

    def _set_data(self):
        if self._fft is None and self._interp_t is None:
            d = None
        else:
            d = self._calc_data_core()
            # Get fft
            if self._fft is not None:
                d = _comp.get_fft(d, **self._fft)
            if self._interp_t is not None:
                t = self._interp_t
                dn = np.full((t.size,d.shape[1],d.shape[2]),np.nan)
                for ii in range(d.shape[2]):
                    for jj in range(0,d.shape[1]):
                        dn[:,jj,ii] = np.interp(t, self.t, d[:,jj,ii],
                                                left=np.nan, right=np.nan)
                d = dn
                self._t = t
            else:
                self._t = None
        self._data = d


    #def get_fft(self, DF=None, Harm=True, DFEx=None, HarmEx=True, Calc=True):
        #self._set_data()

    def __sub__(self, other):
        opfunc = lambda x, y: x-y
        data = _recreatefromoperator(self, other, opfunc)
        return data


    def plot(self, key=None, invert=None, plotmethod='imshow',
             cmap=plt.cm.gray, ms=4, Max=None,
             fs=None, dmargin=None, wintit='tofu',
             draw=True, connect=True):
        """ Plot the data content in a predefined figure  """
        dax, KH = _plot.Data_plot(self, key=key, invert=invert, Max=Max,
                                  plotmethod=plotmethod, cmap=cmap, ms=ms,
                                  fs=fs, dmargin=dmargin, wintit=wintit,
                                  draw=draw, connect=connect)
        return dax, KH





def _DataSpectro_check_inputs(data=None, lamb=None, t=None,
                              dchans=None, LCam=None):
    if data is not None:
        assert type(data) is np.ndarray and data.ndim ==3
        if t is not None:
            assert data.shape[0]==t.size
        if lamb is not None:
            assert data.shape[1]==lamb.size
    if t is not None:
        assert type(t) is np.ndarray and t.ndim==1
        if data is not None:
                assert data.shape[0]==t.size
    if lamb is not None:
        assert type(lamb) is np.ndarray and lamb.ndim==1 and lamb.size>1
        if data is not None:
            assert data.shape[1]==lamb.size
    if dchans is not None:
        assert type(dchans) is dict
        if data is not None:
            for kk in dchans.keys():
                assert hasattr(dchans[kk],'__iter__')
                assert len(dchans[kk])==data.shape[-1]
    if LCam is not None:
        assert type(LCam) is list or issubclass(LCam.__class__,object)
        if type(LCam) is list:
            lCls = ['LOSCam1D','Cam1D','LOSCam2D','Cam2D']
            assert all([cc.Id.Cls in lCls for cc in LCam])
            assert all([issubclass(cc.__class__,object) for cc in LCam])
            if len(LCam)>1:
                msg = "Cannot associate mulitple 2D cameras !"
                assert all([c.Id.Cls in ['LOSCam1D','Cam1D'] for c in LCam]),msg
                msg = "Cannot associate cameras of different types !"
                assert all([cc.Id.Cls==LCam[0].Id.Cls for cc in LCam]), msg
                lVes = [cc.Ves for cc in LCam]
                C0 = all([vv is None for vv in lVes])
                C1 = all([tfu.dict_cmp(vv._todict(),lVes[0]._todict())
                          for vv in lVes])
                assert C0 or C1
                lK = [sorted(cc.dchans.keys() for cc in LCam)]
                assert all([lk==lK[0] for lk in lK])
            if data is not None:
                assert np.sum([cc.nRays for cc in LCam])==data.shape[-1]
        else:
            assert LCam.Id.Cls in ['LOSCam1D','LOSCam2D','Cam1D','Cam2D']
            if data is not None:
                assert LCam.nRays==data.shape[-1]
