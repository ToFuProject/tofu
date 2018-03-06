# -*- coding: utf-8 -*-

# Built-in
import os
import itertools as itt

# Common
import numpy as np
import matplotlib.pyplot as plt

# tofu
import tofu.pathfile as tfpf
import tofu.utils as tfu
try:
    import tofu.data._plot as _plot
except Exception:
    from . import _plot as _plot

__all__ = ['Data1D','Data2D']





class Data(object):


    def __init__(self, data=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None,
                 LCam=None, CamCls='1D', fromdict=None,
                 SavePath=os.path.abspath('./')):

        self._Done = False
        if fromdict is None:
            msg = "Provide either dchans or LCam !"
            assert np.sum([dchans is None, LCam is None])>=1, msg
            self._set_dataRef(data, t=t, dchans=dchans, dunits=dunits)
            self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
            self._set_LCam(LCam=LCam, CamCls=CamCls)
        else:
            self._fromdict(fromdict)
        self._Done = True


    def _fromdict(self, fd):
        _Data_check_fromdict(fd)
        self._Id = tfpf.ID(fromdict=fd['Id'])
        self._Ref = fd['Ref']
        self._dunits = fd['dunits']
        self._indt, self._indch = fd['indt'], fd['indch']
        self._data0 = fd['data0']
        self._CamCls = fd['CamCls']
        self._fft = fd['fft']
        if fd['geom'] is None:
            self._geom = None
        else:
            import tofu.geom as tfg
            if '1D' in fd['CamCls']:
                LCam = [tfg.LOSCam1D(fromdict=cc) for cc in fd['geom']]
            else:
                LCam = [tfg.LOSCam2D(fromdict=cc) for cc in fd['geom']]
            self._set_LCam(LCam=LCam, CamCls=fd['CamCls'])

    def _todict(self):
        out = {'Id':self.Id._todict(),
               'Ref':self._Ref,
               'dunits':self._dunits,
               'indt':self._indt, 'indch':self._indch,
               'data0':self._data0,
               'CamCls':self._CamCls,
               'fft':self._fft}
        if self.geom is None:
            geom = None
        else:
            geom = [cc._todict() for cc in self.geom['LCam']]
        out['geom'] = geom
        return out

    @property
    def Id(self):
        return self._Id
    @property
    def shot(self):
        return self._Id._shot
    @property
    def geom(self):
        return self._geom
    @property
    def units(self):
        return self._dunits
    @property
    def Ref(self):
        return self._Ref
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
    def t(self):
        if self._Ref['t'] is None:
            return None
        else:
            return self._Ref['t'][self.indt]
    @property
    def data(self):
        return self._get_data()
    @property
    def nt(self):
        return int(np.sum(self.indt))
    @property
    def nch(self):
        return int(np.sum(self.indch))
    @property
    def data0(self):
        return self._data0
    @property
    def treatment(self):
        d = {'indt':self.indt, 'indch':self.indch,
             'DtRef':self._DtRef, 'fft':self._fft}
        return d

    def _check_inputs(self, Id=None, data=None, t=None, dchans=None,
                      dunits=None, Exp=None, shot=None, Diag=None, LCam=None,
                      CamCls=None, SavePath=None):
        _Data_check_inputs(Id=Id, data=data, t=t, dchans=dchans, dunits=dunits,
                           Exp=Exp, shot=shot, Diag=Diag, LCam=LCam,
                           CamCls=CamCls, SavePath=SavePath)

    def _set_Id(self, Id, Exp=None, Diag=None, shot=None, SavePath='./'):
        dd = {'Exp':Exp, 'Diag':Diag, 'shot':shot, 'SavePath':SavePath}
        if self._Done:
            tfpf._get_FromItself(self.Id, dd)
        tfpf._check_NotNone({'Id':Id})
        self._check_inputs(Id=Id)
        if type(Id) is str:
            Id = tfpf.ID(self.__class__, Id, **dd)
        self._Id = Id

    def _set_dataRef(self, data, t=None, dchans=None, dunits=None):
        self._check_inputs(data=data, t=t, dchans=dchans, dunits=dunits)
        if data.ndim==1:
            nt, nch = 1, data.size
            data = data.reshape((1,nch))
            if t is not None:
                t = np.asarray(t) if hasattr(t,'__iter__') else np.asarray([t])
        else:
            nt, nch= data.shape
            t = np.asarray(t)
        self._Ref = {'data':data, 't':t,
                     'nt':nt, 'nch':nch}
        dchans = {} if dchans is None else dchans
        lK = sorted(dchans.keys())
        dchans = dict([(kk,np.asarray(dchans[kk])) for kk in lK])
        self._Ref['dchans'] = dchans

        self._dunits = {} if dunits is None else dunits
        self._data = None
        self._indt, self._indch = None, None
        self._data0 = {'data':None,'t':None,'Dt':None}
        self._fft = None
        self._indt_corr, self._indch_corr = None, None

    def _set_LCam(self, LCam=None, CamCls='1D'):
        self._check_inputs(LCam=LCam, CamCls=CamCls)
        if LCam is None:
            self._geom = None
            self._CamCls = CamCls
            if 'data' not in self._dunits.keys():
                self._dunits['data'] = r"a.u."
        else:
            LCam = LCam if type(LCam) is list else [LCam]
            # Set up dchans
            dchans = {}
            if LCam[0].dchans is not None:
                for kk in LCam[0].dchans.keys():
                    dchans[kk] = np.r_[[cc.dchans[kk] for cc in LCam]].ravel()
            self._Ref['dchans'] = dchans
            Ves = LCam[0].Ves
            lS = [c.LStruct for c in LCam if c.LStruct is not None]
            if len(lS)==0:
                lS = None
            else:
                lS = lS[0] if len(lS)==1 else list(itt.chain.from_iterable(lS))
                lSP = [os.path.join(s.Id.SavePath,s.Id.SaveName) for s in lS]
                lS = [lS[lSP.index(ss)] for ss in list(set(lSP))]
            self._geom = {'Ves':Ves, 'LStruct':lS, 'LCam':LCam}
            CamCls = LCam[0].Id.Cls
            self._CamCls = CamCls

            if 'data' not in self._dunits.keys():
                self._dunits['data'] = r"$W/m^2$" if 'LOS' in CamCls else r"$W$"

            LObj = []
            if Ves is not None:
                LObj += [Ves.Id]
            if lS is not None:
                LObj += [ss.Id for ss in lS]
            LObj += [cc.Id for cc in LCam]
            if len(LObj)>0:
                self.Id.set_LObj(LObj)

    def dchans(self, key=None):
        """ List the channels selected by self.indch

        Return their indices (default) or the chosen criterion
        """
        if self.geom is None:
            dchans = None
        elif self._Ref['dchans']=={}:
            dchans = self._Ref['dchans']
        else:
            assert key in [None]+list(self._Ref['dchans'].keys())
            ind = self.indch.nonzero()[0]
            if key is None:
                lK = self._Ref['dchans'].keys()
                dchans = dict([(kk,self._Ref['dchans'][kk][ind]) for kk in lK])
            else:
                dchans = self._Ref['dchans'][key][ind]
        return dchans

    def select_t(self, t=None, out=bool):
        assert out in [bool,int]
        C0 = type(t) in [int,float,np.int64,np.float64]
        C1 = type(t) in [list,np.ndarray] and len(t)==2
        C2 = type(t) is tuple and len(t)==2
        assert t is None or C0 or C1 or C2
        ind = np.zeros((self._Ref['nt'],),dtype=bool)
        if t is None or self._Ref['t'] is None:
            ind = ~ind
        elif C0:
            ind[np.nanargmin(np.abs(self._Ref['t']-t))] = True
        elif C1 or C2:
            ind[(self._Ref['t']>=t[0]) & (self._Ref['t']<=t[1])] = True
            if C2:
                ind = ~ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indt(self, indt=None, t=None):
        C = [indt is None,t is None]
        assert np.sum(C)>=1
        if all(C):
            ind = np.ones((self._Ref['nt'],),dtype=bool)
        elif C[0]:
            ind = self.select_t(t=t, out=bool)
        elif C[1]:
            ind = _format_ind(indt, n=self._Ref['nt'])
        self._indt = ind

    def select_ch(self, key=None, val=None, log='any', touch=None, out=bool):
        assert out in [int,bool]
        assert log in ['any','all','not']
        C = [key is None,touch is None]
        assert np.sum(C)>=1
        if np.sum(C)==2:
            ind = np.ones((self.nRays,),dtype=bool)
        else:
            if key is not None:
                assert type(key) is str and key in self.Ref['dchans'].keys()
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
            elif touch is not None:
                assert self._geom is not None, "Geometry (LCam) not defined !"
                ind = []
                for cc in self._geom['LCam']:
                    ind.append(cc.select(touch=touch, log=log, out=bool))
                if len(ind)==1:
                    ind = ind[0]
                else:
                    ind = np.concatenate(tuple(ind))
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indch(self, indch=None, key=None, val=None, touch=None, log='any'):
        C = [indch is None, key is None, touch is None]
        assert np.sum(C)>=2
        if all(C):
            ind = np.ones((self._Ref['nch'],),dtype=bool)
        elif C[0]:
            ind = self.select_ch(key=key, val=val, touch=touch, log=log, out=bool)
        elif C[1]:
            ind = _format_ind(indch, n=self._Ref['nch'])
        self._indch = ind

    def set_data0(self, data0=None, Dt=None, indt=None):
        assert self._Ref['nt']>1, "Useless if only one data slice !"
        C = [data0 is None, Dt is None, indt is None]
        assert np.sum(C)>=2
        if all(C):
            data, t, indt = None, None, None
        elif data0 is not None:
            assert np.asarray(data0).shape==(self._Ref['nch'],)
            data = np.asarray(data0)
            Dt, indt = None, None
        else:
            if indt is not None:
                indt = _format_ind(indt, n=self._Ref['nt'])
            else:
                indt = self.select_t(t=Dt, out=bool)
            if indt is not None and np.any(indt):
                data = self._Ref['data'][indt,:]
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

    def _set_data(self):
        if self._fft is None:
            d = None
        else:
            # Get time interval
            d = self._Ref['data'][self.indt,:]
            # Substract reference time data
            if self._data0['data'] is not None:
                d = d - self._data0['data'][np.newaxis,:]
            # Get desired channels
            d = d[:,self.indch]
            # Get fft
            d = _comp.get_fft(d, **self._fft)
        self._data = d

    def _get_data(self):
        if self._data is None:
            # Get time interval
            d = self._Ref['data'][self.indt,:]
            # Substract reference time data
            if self._data0['data'] is not None:
                d = d - self._data0['data'][np.newaxis,:]
            # Get desired channels
            d = d[:,self.indch]
        else:
            d = self._data
        return d


    #def get_fft(self, DF=None, Harm=True, DFEx=None, HarmEx=True, Calc=True):
        #self._set_data()

    def plot(self, key=None,
             cmap=plt.cm.gray, ms=4,
             Max=None, fs=None):
        dax, KH = _plot.Data_plot(self, key=key,
                                  cmap=cmap, ms=ms,
                                  Max=Max, fs=fs)
        return dax, KH


    def save(self, SaveName=None, Path=None,
             Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under SaveName

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file
            If None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file
            If None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying how to save the object:
                'npz': as a numpy array file (recommended)
        compressed :    bool
            Flag, used when Mode='npz', indicates whether to use:
                - False : np.savez
                - True :  np.savez_compressed (slower but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path,
                          Mode=Mode, compressed=compressed, Print=Print)









def _Data_check_inputs(Id=None, data=None, t=None, dchans=None,
                      dunits=None, Exp=None, shot=None, Diag=None, LCam=None,
                      CamCls=None, SavePath=None):
    if Id is not None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID !"
    if data is not None:
        assert type(data) is np.ndarray and data.ndim in [1,2]
        if t is not None:
            if t.size==1:
                assert data.ndim==1 or data.shape[0]==1
            else:
                assert data.ndim==2 and data.shape[0]==t.size
    if t is not None:
        assert type(t) is np.ndarray and t.ndim==1
        if data is not None:
            if t.size>1:
                assert data.ndim==2 and data.shape[0]==t.size
            else:
                assert data.ndim==1 or (data.ndim==2 and data.shape[0]==t.size)
    if dchans is not None:
        assert type(dchans) is dict
        if data is not None:
            for kk in dchans.keys():
                assert hasattr(dchans[kk],'__iter__')
                assert len(dchans[kk])==data.shape[1]
    if dunits is not None:
        assert type(dunits) is dict
        assert all([type(dunits[kk]) is str for kk in dunits.keys()])
    if Exp is not None:
        assert type(Exp) is str
    if Diag is not None:
        assert type(Diag) is str
    if shot is not None:
        assert type(shot) is int
    if SavePath is not None:
        assert type(SavePath) is str
    if CamCls is not None:
        assert CamCls in ['1D','2D','LOSCam1D','LOSCam2D']
    if LCam is not None:
        assert type(LCam) is list or issubclass(LCam.__class__,object)
        if type(LCam) is list:
            lCls = ['LOSCam1D','Cam1D','LOSCam2D','Cam2D']
            assert all([cc.Id.Cls in lCls for cc in LCam])
            assert all([issubclass(cc.__class__,object) for cc in LCam])
            if len(LCam)>1:
                msg = "Cannot associate mulitple 2D cameras !"
                assert all([cc.Id.Cls in ['LOSCam1D','Cam1D'] for cc in LCam]), msg
                msg = "Cannot associate cameras of different types !"
                assert all([cc.Id.Cls==LCam[0].Id.Cls for cc in LCam]), msg
                lVes = [cc.Ves for cc in LCam]
                C0 = all([vv is None for vv in lVes])
                C1 = all([tfu.dict_cmp(vv._todict(),lVes[0]._todict())
                          for vv in lVes])
                assert C0 or C1
                lK = [sorted(cc.dchans.keys() for cc in LCam)]
                assert all([lk==lK[0] for lk in lK])
        else:
            assert LCam.Id.Cls in ['LOSCam1D','LOSCam2D','Cam1D','Cam2D']


def _Data_check_fromdict(fd):
    assert type(fd) is dict, "Arg from dict must be a dict !"
    k0 = {'Id':dict, 'Ref':dict, 'dunits':[None,dict],
          'indt':[None,np.ndarray], 'indch':[None,np.ndarray],
          'data0':[None,dict], 'CamCls':str, 'fft':[None,dict],
          'geom':[None,list]}
    keys = list(fd.keys())
    for kk in k0:
        assert kk in keys, "%s must be a key of fromdict"%kk
        typ = type(fd[kk])
        C = typ is k0[kk] or typ in k0[kk] or fd[kk] in k0[kk]
        assert C, "Wrong type of fromdict[%s]: %s"%(kk,str(typ))+str(fd[kk])

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




#####################################################################
#               Data1D and Data2D
#####################################################################


class Data1D(Data):
    """ Data object used for 1D cameras or list of 1D cameras  """
    def __init__(self, data=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None,
                 LCam=None, fromdict=None,
                 SavePath=os.path.abspath('./')):
        Data.__init__(self, data, t=t, dchans=dchans, dunits=dunits,
                 Id=Id, Exp=Exp, shot=shot, Diag=Diag, CamCls='1D',
                 LCam=LCam, fromdict=fromdict, SavePath=SavePath)



class Data2D(Data):
    """ Data object used for 1D cameras or list of 1D cameras  """
    def __init__(self, data=None, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None,
                 LCam=None, X12=None, fromdict=None,
                 SavePath=os.path.abspath('./')):
        Data.__init__(self, data, t=t, dchans=dchans, dunits=dunits,
                      Id=Id, Exp=Exp, shot=shot, Diag=Diag,
                      LCam=LCam, CamCls='2D', fromdict=fromdict, SavePath=SavePath)
        self.set_X12(X12)

    def set_X12(self, X12=None):
        X12 = X12 if self.geom is None else None
        if X12 is not None:
            X12 = np.asarray(X12)
            assert X12.shape==(2,self.Ref['nch'])
        self._X12 = X12

    def get_X12(self, out='1d'):
        if self._X12 is None:
            msg = "X12 must be set for plotting if LCam not provided !"
            assert self.geom is not None, msg
            X12, DX12 = self.geom['LCam'][0].get_X12(out=out)
        else:
            X12 = self._X12
            if out.lower()=='2d':
                x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
                X12 = [x1u,x2u,ind]
            else:
                DX12 = None
        return X12, DX12

    def plot(self, key=None, invert=True, cmap=plt.cm.gray, ms=4,
             Max=None, fs=None, plotmethod='imshow'):
        dax, KH = _plot.Data_plot(self, key=key, invert=invert,
                                  plotmethod=plotmethod,
                                  cmap=cmap, ms=ms, Max=Max, fs=fs)
        return dax, KH
