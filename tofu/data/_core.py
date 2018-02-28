# -*- coding: utf-8 -*-

# Built-in
import os
import itertools as itt

# Common
import numpy as np

# tofu
import tofu.pathfile as tfpf
try:
    import tofu.data._plot as _plot
except Exception:
    from . import _plot as _plot

__all__ = ['Data']





class Data(object):


    def __init__(self, data, t=None, dchans=None, dunits=None,
                 Id=None, Exp=None, shot=None, Diag=None,
                 LCam=None, CamCls='1D', fromdict=None, SavePath='./'):

        self._Done = False
        if fromdict is None:
            msg = "Provide either dchans or LCam !"
            assert np.sum([dchans is None, LCam is None])>=1, msg
            self._set_data(data, t=t, dchans=dchans, dunits=dunits)
            self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
            self._set_LCam(LCam=LCam, CamCls=CamCls)
        else:
            self._fromdict(fromdict)
        self._Done = True


    def _fromdict(self, fd):
        _Data_check_fromdict(fd)
        self._Id = tfpf.ID(fromdict=fd['Id'])

    def _todict(self):
        out = {'Id':self.Id._todict()}
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
        return self._Ref['t'][self.indt]
    @property
    def data(self):
        # Get time interval
        d = self._Ref['data'][self.indt,:]
        # Substract reference time data
        if self._DtRef is not None:
            d = d - self._DtRef_data[np.newaxis,:]
        # Get desired channels
        d = d[:,self.indch]
        # Get FFT-filtered data
        #if self._fft is not None:
        #    d =
        return d
    @property
    def nt(self):
        return int(np.sum(self.indt))
    @property
    def nch(self):
        return int(np.sum(self.indch))
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

    def _set_data(self, data, t=None, dchans=None, dunits=None):
        self._check_inputs(data=data, t=t, dchans=dchans, dunits=dunits)
        self._Ref = {'data':data, 't':t,
                     'nt':data.shape[0], 'nch':data.shape[1]}
        dchans = {} if dchans is None else dchans
        lK = sorted(dchans.keys())
        dchans = dict([(kk,np.asarray(dchans[kk])) for kk in lK])
        self._Ref['dchans'] = dchans

        self._dunits = {} if dunits is None else dunits
        self._indt, self._indch = None, None
        self._DtRef, self._fft = None, None
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


    def select_t(self, t=None, out=bool):
        assert out in [bool,int]
        C0 = type(t) in [int,float,np.int64,np.float64]
        C1 = type(t) in [list,np.ndarray] and len(t)==2
        C2 = type(t) is tuple and len(t)==2
        assert t is None or C0 or C1 or C2
        ind = np.zeros((self._Ref['nt'],),dtype=bool)
        if t is None:
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
            Ints = [int,np.int64]
            C0 = type(indt) in Ints
            if C0:
                ii = np.zeros((self._Ref['nt'],),dtype=bool)
                ii[indt] = True
                ind = ii
            else:
                if type(indt[0]) in [bool,np.bool_]:
                    ind = np.asarray(indt)
                    assert ind.size==self._Ref['nt']
                else:
                    ii = np.zeros((self._Ref['nt'],),dtype=bool)
                    ii[np.asarray(indt)] = True
                    ind = ii
        self._indt = ind

    def dchans(self, key=None):
        """ List the channels selected by self.indch

        Return their indices (default) or the chosen criterion
        """
        if self._Ref['dchans']=={}:
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

    def select_ch(self, key=None, val=None, log='any', touch=None, out=bool):
        assert out in [int,bool]
        assert log in ['any','all','not']
        C = [key is None,touch is None]
        assert np.sum(C)>=1
        if np.sum(C)==2:
            ind = np.ones((self.nRays,),dtype=bool)
        else:
            if key is not None:
                assert type(key) is str and key in self._dchans.keys()
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
            elif touch is not None:
                assert self._geom is not None, "Geometry (LCam) not defined !"
                VesOk, SOk = self.Ves is not None, self.LStruct is not None
                SNames = [ss.Id.Name for ss in self.LStruct] if SOk else None
                ind = []
                for cc in self._geom['LCam']:
                    ind.append(_comp.Rays_touch(VesOk, SOk, cc.geom['IndOut'],
                                                SNames, touch=touch))
                ind = np.concatenate(tuple(ind))
                ind = ~ind if log=='not' else ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def set_indch(self, indch=None, key=None, val=None, log='any'):
        C = [indch is None,t is None]
        assert np.sum(C)>=1
        if all(C):
            ind = np.ones((self._Ref['nch'],),dtype=bool)
        elif C[0]:
            ind = self.select_ch(key=key, val=val, log=log, out=bool)
        elif C[1]:
            Ints = [int,np.int64]
            C0 = type(indch) in Ints
            if C0:
                ii = np.zeros((self._Ref['nch'],),dtype=bool)
                ii[indch] = True
                ind = ii
            else:
                if type(indch[0]) in [bool,np.bool_]:
                    ind = np.asarray(indch)
                    assert ind.size==self._Ref['nch']
                else:
                    ii = np.zeros((self._Ref['nch'],),dtype=bool)
                    ii[np.asarray(indch)] = True
                    ind = ii
        self._indch = ind

    def set_DtRef(self, DtRef=None):
        self._DtRef = DtRef
        if DtRef is not None:
            indt = self.select_t(t=DtRef, out=bool)
            if np.any(indt):
                dd = self._Ref['data'][indt,:]
                if np.sum(indt)>1:
                    dd = np.nanmean(dd,axis=0)
                self._DtRef_data = dd
        else:
            self._DtRef_data = None

    #def get_fft(self, DF=None, Harm=True, DFEx=None, HarmEx=True, Calc=True):


    def plot(self, key=None, Max=4, a4=False):
        Lax = _plot.Data_plot(self, key=key, Max=Max, a4=a4)
        return Lax


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
        assert type(data) is np.ndarray and data.ndim==2
    if t is not None:
        assert type(t) is np.ndarray and t.ndim==1
        if data is not None:
            assert data.shape[0]==t.size
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
        assert CamCls in ['1D','2D']
    if LCam is not None:
        assert type(LCam) is list or issubclass(LCam.__class__,object)
        if issubclass(LCam.__class__,object):
            assert LCam.Id.Cls in ['LOSCam1D','LOSCam2D','Cam1D','Cam2D']
        if type(LCam) is list:
            assert all([issubclass(cc.__class__,object) for cc in LCam])
            msg = "Cannot associate mulitple 2D cameras !"
            assert all([cc.Id.Cls in ['LOSCam1D','Cam1D'] for cc in LCam]), msg
            msg = "Cannot associate cameras of different types !"
            assert all([cc.Id.Cls==LCam[0].Id.Cls for cc in LCam]), msg
            lVes = [cc.Ves for ss in LCam]
            C0 = all([vv is None for vv in lVes])
            C1 = all([vv._todict()==lVes[0]._todict() for vv in lVes])
            assert C0 or C1
            lK = [sorted(cc.dchans.keys() for cc in LCam)]
            assert all([lk==LK[0] for lk in lK])


def _Data_check_fromdict(fd):
    assert type(fd) is dict, "Arg from dict must be a dict !"
    k0 = {'Id':dict, 'geom':dict, 'LNames':[None,list],
          'Ves':[None,dict], 'LStruct':[None,list]}
    keys = list(fd.keys())
    for kk in k0:
        assert kk in keys, "%s must be a key of fromdict"%kk
        typ = type(fd[kk])
        C = typ is k0[kk] or typ in k0[kk] or fd[kk] in k0[kk]
        assert C, "Wrong type of fromdict[%s]: %s"%(kk,str(typ))
