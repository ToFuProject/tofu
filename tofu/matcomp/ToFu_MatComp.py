# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:43:27 2014

@author: didiervezinet
"""

import numpy as np
import ToFu_Geom as TFG
import ToFu_Mesh as TFM
import ToFu_PathFile as TFPF
import ToFu_Defaults as TFD
import General_Geom_cy as GG
import scipy.integrate as scpinteg
import Polygon as plg
import matplotlib.pyplot as plt
import matplotlib.colors as mpcl
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection as PcthColl
import scipy.sparse as scpsp
import datetime as dtm
import cPickle as pck # for saving objects
import os
import itertools as itt
import math
import warnings
import copy


from mpl_toolkits.mplot3d import Axes3D


"""
###############################################################################
###############################################################################
                Geometry Matrix class
###############################################################################
"""


class GMat2D(object):

    def __init__(self, Id, BF2=None, LD=None, Mat=None, indMat=None, MatLOS=None, Calcind=True, Calc=True, CalcLOS=True, Iso='Iso', StoreBF2=False, StoreLD=False, StoreTor=True, dtime=None, Fast=True, Verb=True,
            SubPind=TFD.GMindMSubP, ModeLOS=TFD.GMMatLOSMode, epsrelLOS=TFD.GMMatLOSeps, SubPLOS=TFD.GMMatLOSSubP, SubModeLOS=TFD.GMMatLOSSubPMode,
            Mode=TFD.GMMatMode, epsrel=TFD.GMMatepsrel, SubP=TFD.GMMatMSubP, SubMode=TFD.GMMatMSubPMode, SubTheta=TFD.GMMatMSubTheta, SubThetaMode=TFD.GMMatMSubThetaMode):

        self._init_CompParam(Iso=Iso, Mode=Mode, epsrel=epsrel, SubP=SubP, SubMode=SubMode, SubTheta=SubTheta, SubThetaMode=SubThetaMode, Fast=Fast, SubPind=SubPind, ModeLOS=ModeLOS, epsrelLOS=epsrelLOS, SubPLOS=SubPLOS, SubModeLOS=SubModeLOS)
        self.set_Id(Id, dtime=dtime)
        if not BF2 is None:
            self._set_BF2(BF2, Calcind=False, Calc=False, CalcLOS=False)
        if not LD is None:
            self._set_LD(LD, Calcind=False, Calc=False, CalcLOS=False)
        if not None in [BF2,LD]:
            self.Store(StoreBF2=StoreBF2, StoreLD=StoreLD, StoreTor=StoreTor)
            self._CalcAllMat(indMat=indMat, Mat=Mat, MatLOS=MatLOS, Calcind=Calcind, Calc=Calc, CalcLOS=CalcLOS, Verb=Verb)

    @property
    def Id(self):
        return self._Id
    @Id.setter
    def Id(self,Val):
        self.set_Id(Val)
    @property
    def BF2(self):
        if self._BF2 is None:
            return TFPF.Open(self.Id.LObj['BF2D']['SavePath'][0]+self.Id.LObj['BF2D']['SaveName'][0]+'.npz')
        else:
            return self._BF2
    @BF2.setter
    def BF2(self,Val):
        self._set_BF2(Val)
    @property
    def BF2_Deg(self):
        return self._BF2_Deg
    @property
    def BF2_NFunc(self):
        return self._BF2_NFunc
    @property
    def Ves(self):
        if self._Tor is None:
            return TFPF.Open(self.Id.LObj['Ves']['SavePath'][0]+self.Id.LObj['Ves']['SaveName'][0]+'.npz')
        else:
            return self._Tor
    @property
    def LD(self):
        if self._LD is None:
            LD = []
            for ii in range(0,self.LD_nDetect):
                LD.append(TFPF.Open(self.Id.LObj['Detect']['SavePath'][ii]+self.Id.LObj['Detect']['SaveName'][ii]+'.npz'))
            return LD
        else:
            return self._LD
    @LD.setter
    def LD(self,Val):
        self._set_LD(Val)
    @property
    def LD_nDetect(self):
        return self._LD_nDetect
    @property
    def Mat(self):
        return self._Mat_csr
    @property
    def MatLOS(self):
        return self._MatLOS_csr

    def _init_CompParam(self, Iso=None, Mode=None, epsrel=None, SubP=None, SubMode=None, SubTheta=None, SubThetaMode=None, Fast=True, SubPind=None, ModeLOS=None, epsrelLOS=None, SubPLOS=None, SubModeLOS=None):
        assert all([mm is None or mm in ['quad','simps','trapz'] for mm in [Mode,ModeLOS]]), "Args Mode and ModeLOS must be in ['quad','simps','trapz'] !"
        assert Iso is None or Iso in ['Iso'], "Arg Iso must be in ['Iso'] !"
        assert all([aa is None or (type(aa) in [int,float,np.float64,np.float32] and aa > 0.) for aa in [epsrel,SubP,SubTheta,SubPind,epsrelLOS,SubPLOS]]), "Args [epsrel,SubP,SubTheta,SubPind,epsrelLOS,SubPLOS] must be int, float or np.float64/32 !"
        assert all([ss is None or ss.lower() in ['rel','abs'] for ss in [SubMode,SubThetaMode,SubModeLOS]]), "Args [SubMode,SubThetaMode,SubModeLOS] must be in ['rel','abs'] !"
        assert type(Fast) is bool, "Arg Fast must be a bool !"
        for ss, vv in zip(['Iso','Mode','epsrel','SubP','SubMode','SubTheta','SubThetaMode','Fast'],[Iso,Mode,epsrel,SubP,SubMode,SubTheta,SubThetaMode,Fast]):
            if not vv is None:
                setattr(self,'_Mat_'+ss, vv)
        if not SubPind is None:
            self._indMat_SubP = SubPind
        for ss, vv in zip(['Mode','epsrel','SubMode','SubP'],[ModeLOS,epsrelLOS,SubModeLOS,SubPLOS]):
            if not vv is None:
                setattr(self,'_MatLOS_'+ss, vv)

    def _CalcAllMat(self, indMat=None, Mat=None, MatLOS=None, Calcind=True, Calc=True, CalcLOS=True, Verb=True):
        if True in [Calcind,Calc,CalcLOS]:
            print "    Computing GMat2D :", self.Id.Name
        if Calcind or True in [Calc,CalcLOS]:
            self._set_indMat(indMat=indMat, Verb=Verb)
        if CalcLOS:
            self._set_MatLOS(MatLOS=MatLOS, Verb=Verb)
        if Calc:
            self._set_Mat(Mat=Mat, Verb=Verb)


    def set_Id(self,Val, dtime=None):
        assert type(Val) is str or isinstance(Val,TFPF.ID), "Arg Id should be string or an TFPF.ID instance !"
        if type(Val) is str:
            if self._Mode=='quad':
                ext = self._Mode + str(self._epsrel)
            else:
                ext = self._Mode + '_SubP' + str(self._SubP) + '_SubMode' + self._SubMode + '_SubTheta' + str(self._SubTheta)
            Val = TFPF.ID('GMat2D',Val+'_'+ext, dtime=dtime)
        self._Id = Val

    def _set_BF2(self,BF2, Store=True, Calcind=True, Calc=True, CalcLOS=True):
        assert isinstance(BF2,TFM.BF2D), "Arg BF2 must be a TFM.BF2D instance !"
        self._Id.set_LObj([BF2.Id])
        self._BF2_Deg = BF2.Deg
        self._BF2_NCents = BF2.Mesh.NCents
        self._BF2_NFunc = BF2.NFunc
        self._BF2 = BF2 if Store else None
        self._CalcAllMat(Calcind=Calcind, Calc=Calc, CalcLOS=CalcLOS)

    def _set_LD(self, LDetect, StoreTor=True, StoreLD=True, Calcind=True, Calc=True, CalcLOS=True):
        assert isinstance(LDetect,TFG.GDetect) or isinstance(LDetect,TFG.Detect) or (type(LDetect) is list and (all([isinstance(gd,TFG.GDetect) for gd in LDetect]) or all([isinstance(gd,TFG.Detect) for gd in LDetect]))), "Arg GD must be a TFG.GDetect or TFG.Detect instance or a list of such !"
        if isinstance(LDetect,TFG.GDetect):
            LDetect = LDetect.LDetect
        if isinstance(LDetect,TFG.Detect):
            LDetect = [LDetect]
        assert all([TFPF.CheckSameObj(LDetect[0].Ves,dd.Ves, ['Poly']) for dd in LDetect]), "All Detect objects must have the same Ves object !"
        self._Id.set_LObj([LDetect[0].Ves.Id])
        self._Tor = LDetect[0].Ves if StoreTor else None
        self._LD_Id = [ll.Id for ll in LDetect]
        self._Id.set_LObj(self._LD_Id)
        self._LD_nDetect = len(LDetect)
        self._LD = LDetect if StoreLD else None
        self._CalcAllMat(Calcind=Calcind, Calc=Calc, CalcLOS=CalcLOS)

    def Store(self, StoreBF2=True, StoreLD=False, StoreTor=True):
        assert type(StoreBF2) is bool and type(StoreLD) is bool, "Args StoreBF2 and StoreLD must be bool !"
        if self._BF2 is None and StoreBF2:
            self._BF2 = self.BF2
        elif not self._BF2 is None and not StoreBF2:
            self._BF2 = None
        if self._LD is None and StoreLD:
            self._LD = self.LD
        elif not self._LD is None and not StoreLD:
            self._LD = None
        if self._Tor is None and StoreTor:
            self._Tor = self.Ves
        elif not self._Tor is None and not StoreTor:
            self._Tor = None

    def get_Tor(self):
        try:
            PathFileExt = self.Id.LObj['Ves']['SavePath'][0]+self.Id.LObj['Ves']['SaveName'][0]+'.npz'
            tor = TFPF.Open(PathFileExt)
            print "Detect "+ self.Id.Name +" : associated Ves object was loaded from "+self.Id.LObj['Ves']['SavePath'][0]+self.Id.LObj['Ves']['SaveName'][0]+".npz"
            return tor
        except:
            try:
                warnings.warn("Detect "+ self.Id.Name +" : associated Ves object could not be loaded from "+self.Id.LObj['Ves']['SavePath'][0]+self.Id.LObj['Ves']['SaveName'][0]+".npz")
            except:
                warnings.warn("Detect "+ self.Id.Name +" : associated Ves object could not be loaded (no matching PathFileExt) !")

    def select(self, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='or', InOut='In', Out=bool):
        return TFPF.SelectFromIdLObj(self.Id.LObj['Detect'], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=Out)

    def _set_indMat(self, indMat=None, SubP=None, Verb=True):
        assert indMat is None or (isinstance(indMat,np.ndarray) and indMat.dtype.name=='bool') or isinstance(indMat,scpsp.csr_matrix), "Arg indMat must be a np.ndarray instance with dtype=bool !"
        self._init_CompParam(SubPind=SubP)
        if indMat is None:
            indMat = Calc_indMatrix_2D(self.BF2, self.LD, SubP=self._indMat_SubP, Verb=Verb, Test=True)
        else:
            assert indMat.ndim==2 and indMat.shape == (self.LD_nDetect,self._BF2_NCents), "Arg indMat not the right shape !"
        self._indMat = indMat

    def _set_Mat(self, Mat=None, Iso=None, Mode=None, epsrel=None, SubP=None, SubMode=None, SubTheta=None, SubThetaMode=None, Fast=True, Verb=True, Test=True):
        assert Mat is None or isinstance(Mat,np.ndarray) or isinstance(Mat,scpsp.csr_matrix), "Arg Mat must be a np.ndarray or scp.sparse.csr_matrix instance !"
        self._init_CompParam( Iso=Iso, Mode=Mode, epsrel=epsrel, SubP=SubP, SubMode=SubMode, SubTheta=SubTheta, SubThetaMode=SubThetaMode, Fast=Fast)
        if Mat is None:
            if self._Mat_Iso=='Iso':
                Mat = Calc_GeomMatrix_2D_Iso(self.BF2, self.LD, self._indMat, Mode=self._Mat_Mode, epsrel=self._Mat_epsrel, SubP=self._Mat_SubP, SubMode=self._Mat_SubMode, SubTheta=self._Mat_SubTheta, SubThetaMode=self._Mat_SubThetaMode, Fast=self._Mat_Fast, Verb=Verb, Test=Test)
            else:
                print "Not coded yet"
        else:
            assert Mat.ndim == 2 and Mat.shape == (self.LD_nDetect,self.BF2_NFunc), "Arg Mat not the right shape !"
        self._Mat_csr = scpsp.csr_matrix(Mat)

    def _set_MatLOS(self, MatLOS=None, BF2=None, LD=None, Mode=None, epsrel=None, SubP=None, SubMode=None, Verb=True, Test=True):
        assert MatLOS is None or isinstance(MatLOS,np.ndarray) or isinstance(MatLOS,scpsp.csr_matrix), "Arg MatLOS must be a np.ndarray or scp.sparse.csr_matrix instance !"
        self._init_CompParam(ModeLOS=Mode, epsrelLOS=epsrel, SubPLOS=SubP, SubModeLOS=SubMode)
        if MatLOS is None:
            MatLOS = Calc_GeomMatrix_LOS_Iso(self.BF2, self.LD, self._indMat, Mode=self._MatLOS_Mode, epsrel=self._MatLOS_epsrel, SubP=self._MatLOS_SubP, SubMode=self._MatLOS_SubMode, Verb=Verb, Test=Test)
        else:
            assert MatLOS.ndim==2 and MatLOS.shape == (self.LD_nDetect,self.BF2_NFunc), "Arg MatLOS not the right shape !"
        self._MatLOS_csr = scpsp.csr_matrix(MatLOS)

    def get_SubGMat2D(self, Name=None, ind=None, Val=None, Crit='Name',PreExp=None,PostExp=None,Log='or',InOut='In'):
        assert ind is None or isinstance(ind,np.ndarray) and 'int' in ind.dtype.name, "Arg ind must be a np.ndarray of int !"
        if ind is None:
            ind = TFPF.SelectFromIdLObj(self.Id.LObj['Detect'], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
        indMat = self._indMat[ind,:]
        Mat = self._Mat_csr.todense()[ind,:]
        MatLOS = self._MatLOS_csr.todense()[ind,:]
        Id = copy.deepcopy(self.Id)
        if Name is None:
            Name = Id.Name+'_Reduced'
        Id.set_Name(Name)
        for kk in Id._LObj['Detect'].keys():
            Id._LObj['Detect'][kk] = [self.Id.LObj['Detect'][kk][ii] for ii in ind]
        GM = GMat2D(Id, BF2=None, LD=None, Mat=None, indMat=None, MatLOS=None, Calcind=False, Calc=False, CalcLOS=False)
        GM._init_CompParam(Mode=self._Mat_Mode, epsrel=self._Mat_epsrel, SubP=self._Mat_SubP, SubMode=self._Mat_SubMode, SubTheta=self._Mat_SubTheta, SubThetaMode=self._Mat_SubThetaMode, Fast=self._Mat_Fast, Iso=self._Mat_Iso,
                SubPind=self._indMat_SubP, ModeLOS=self._MatLOS_Mode, epsrelLOS=self._MatLOS_epsrel, SubPLOS=self._MatLOS_SubP, SubModeLOS=self._MatLOS_SubMode)
        GM._BF2 = None
        GM._BF2_NFunc = self.BF2_NFunc
        GM._BF2_Deg = self.BF2_Deg
        GM._BF2_NCents = self._BF2_NCents
        GM._Tor = None
        GM._LD = None
        GM._LD_nDetect = ind.size
        GM._set_indMat(indMat=indMat, Verb=False)
        GM._set_MatLOS(MatLOS=MatLOS, Verb=False)
        GM._set_Mat(Mat=Mat, Verb=False)
        return GM

    def get_Sig(self, Coefs=1., LOS=False):
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==self.BF2_NFunc), "Arg Coefs must be a (self.BF2_NFunc,) np.ndarray !"
        if Coefs.ndim==2:
            Coefs = Coefs.flatten()
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.BF_NFunc,))
        if LOS:
            return np.array(self._MatLOS_csr.dot(Coefs))
        else:
            return np.array(self._Mat_csr.dot(Coefs))

    def plot_OneDetect_PolProj(self, EltDet='PCL', Name=None, Val=None, Crit='Name',PreExp=None,PostExp=None,Log='or',InOut='In', ind=None, axP=None, axM=None, axBF=None, Mask=False, TLOS=False, SubP=TFD.GMPlotDetSubP, SubMode=TFD.GMPlotDetSubPMode, Cdict=TFD.GMPlotDetCd, Ldict=TFD.GMPlotDetLd, LdictLOS=TFD.GMPlotDetLOSd, KWArgMesh=TFD.GMPlotDetKWArgMesh, KWArgDet=TFD.GMPlotDetKWArgDet, a4=False):
        assert not (self._Mat_csr is None and self._MatLOS_csr is None) and not (self._Mat_csr is None and not TLOS), "Geometry matrix not computed yet !"
        assert ind is None or type(ind) is int, "Arg int must be an integer (index of detector to be plotted) !"
        if ind is None:
            #ind = TFPF.SelectFromListId(self._LD_Id, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)[0]
            ind = TFPF.SelectFromIdLObj(self.Id.LObj['Detect'], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)[0]
        D = self.LD[ind]
        BF2 = self.BF2
        Ti = np.array(self._Mat_csr[ind,:].todense()).flatten() if not self._Mat_csr is None else None
        TiLOS = np.array(self._MatLOS_csr[ind,:].todense()).flatten() if (not self._MatLOS_csr is None and TLOS) else None
        axP, axM, axBF = Plot_GeomMatrix_Detect(D, BF2, Ti, axPol=axP, axMesh=axM, axBF=axBF, Mask=Mask, indMask=self._indMat[ind,:], TLOS=TLOS, TiLOS=TiLOS, SubP=SubP, SubMode=SubMode, Cdict=Cdict, Ldict=Ldict, LdictLOS=LdictLOS, Test=False, a4=a4)
        if not axP is None:
            axP = BF2.Mesh.plot(ax=axP, **KWArgMesh)
            axP = D.plot(Lax=[axP], Proj='Pol', **KWArgDet)
        return axP, axM, axBF

    def plot_OneBF_PolProj(self, ind, axP=None, axD=None, axDred=None, TLOS=False, SubP=TFD.GMPlotBFSubP, SubMode=TFD.GMPlotBFSubPMode,
            Cdict=TFD.GMPlotDetCd, Ldict=TFD.GMPlotDetLd, LdictLOS=TFD.GMPlotDetLOSd, KWArgMesh=TFD.GMPlotDetKWArgMesh, KWArgTor=TFD.GMPlotDetKWArgTor, KWArgDet=TFD.GMPlotBFDetd, a4=False):
        assert type(ind) is int, "Arg int must be an int !"
        assert not (self._Mat_csr is None and self._MatLOS_csr is None) and not (self._Mat_csr is None and not TLOS), "Geometry matrix not computed yet !"
        Ti = np.array(self._Mat_csr[:,ind].todense()).flatten() if not self._Mat_csr is None else None
        TiLOS = np.array(self._MatLOS_csr[:,ind].todense()).flatten() if (not self._MatLOS_csr is None and TLOS) else None
        LD = self.LD
        axP, axD, axDred = Plot_GeomMatrix_BFunc(LD, self.Id.LObj['Detect']['Name'], ind, Ti=Ti, axPol=axP, axD=axD, axDred=axDred, TLOS=TLOS, TiLOS=TiLOS, Ldict=Ldict, LdictLOS=LdictLOS, KWArgDet=KWArgDet, Test=True, a4=a4)
        if not axP is None:
            BF2 = self.BF2
            axP = BF2.plot_Ind(Ind=ind, ax=axP, Elt='S', EltM='')
            axP = BF2.Mesh.plot(ax=axP, **KWArgMesh)
            axP = LD[0].Ves.plot(Lax=[axP], Proj='Pol', **KWArgTor)
        return axP, axD, axDred

    def plot_sum(self, axD=None, axBF=None, TLOS=False, Ldict=TFD.GMPlotDetLd, LdictLOS=TFD.GMPlotDetLOSd, LegDict=TFD.TorLegd, a4=False):
        assert axD is None or isinstance(axD,plt.Axes), "Arg axD must be None, 'None' or a plt.Axes instance !"
        assert axBF is None or isinstance(axBF,plt.Axes), "Arg axBF must be None, 'None' or a plt.Axes instance !"
        assert type(Ldict) is dict, "Arg Ldict must be a dict !"
        if  axD is None and axBF is None:
            axD, axBF = TFD.Plot_BF2_sum_DefAxes(a4=a4)
        if not axD is None:
            axD.plot(np.arange(0,self.LD_nDetect), self._Mat_csr.sum(axis=1), label=self.Id.NameLTX, **Ldict)
            if TLOS:
                axD.plot(np.arange(0,self.LD_nDetect), self._MatLOS_csr.sum(axis=1), label=self.Id.NameLTX+' LOS', **LdictLOS)
            axD.set_xlim(0,self.LD_nDetect-1)
            if not LegDict is None:
                axD.legend(**LegDict)
        if not axBF is None:
            axBF.plot(np.arange(0,self.BF2_NFunc), self._Mat_csr.sum(axis=0).T, label=self.Id.NameLTX, **Ldict)
            if TLOS:
                axBF.plot(np.arange(0,self.BF2_NFunc), self._MatLOS_csr.sum(axis=0).T, label=self.Id.NameLTX+' LOS', **LdictLOS)
            axBF.set_xlim(0,self.BF2_NFunc-1)
            if not LegDict is None:
                axBF.legend(**LegDict)
        axD.figure.canvas.draw()
        return axD, axBF

    def plot_Sig(self, Coefs=1., ind=0, ax1=None, ax2=None, ax3=None, ax4=None, TLOS=False, LOSRef=None, SubP=TFD.GMSigPlotSubP, SubMode=TFD.GMSigPlotSubPMode, NC=TFD.GMSigPlotNC,
            Sdict=TFD.GMSigPlotSd, SLOSdict=TFD.GMSigPlotSd.copy(), LOSdict=TFD.GMSigPlotSLOSd.copy(), Cdict=TFD.GMSigPlotCd, a4=False):
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==self.BF2_NFunc), "Arg Coefs must be a (self.BF2_NFunc,) np.ndarray !"
        assert all([type(bb) is bool for bb in [TLOS]]), "Args [TLOS] must be bool !"
        if Coefs.ndim==2:
            Coefs = Coefs.flatten()
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.BF_NFunc,))
        BF2, LD = self.BF2, self.LD
        Mes = self.get_Sig(Coefs,LOS=False)
        if TLOS:
            MesLOS = self.get_Sig(Coefs,LOS=True)
            if LOSRef is None:
                LOSRef = LD[0]._LOSRef

        if all([aa is None for aa in [ax1,ax2,ax3,ax4]]):
            ax1, ax2, ax3, ax4 = TFD.Plot_SynDiag_DefAxes(a4=a4)
        if ax1 is None:
            ax1 = TFD.Plot_SynDiag_DefAxes(a4=a4)[0]
        if not ax2 is None:
            ax2 = BF2.Mesh.plot(ax=ax2, Elt='M')
            ax2 = LD[0].Ves.plot(Lax=[ax2],Proj='Pol')
        if not ax3 is None:
            ax3 = BF2.plot(ax=ax3, Deriv=0, Coefs=Coefs, SubP=SubP, SubMode=SubMode, NC=NC, Totdict=Cdict)
            ax3 = LD[0].Ves.plot(Lax=[ax3], Elt='P',Proj='Pol')
            XL, YL = ax3.get_xlim(), ax3.get_ylim()

        Xticks, XTicksLab = [], []
        if not ax4 is None:
            ax4 = LD[0].Ves.plot(Lax=[ax4], Elt='P',Proj='Pol')
        LCams = sorted(list(set([dd.Id.USRdict['CamHead'] for dd in LD])))
        II = 1
        for ii in range(0,len(LCams)):
            ind = [jj for jj in range(0,len(LD)) if LD[jj].Id.USRdict['CamHead']==LCams[ii]]
            ll = ax1.plot(np.arange(II,II+len(ind)), Mes[ind], label=LCams[ii], **Sdict)
            if TLOS:
                SLOSdict['c'], SLOSdict['ls'] = ll[0].get_color(), '--'
                ax1.plot(np.arange(II,II+len(ind)), MesLOS[ind], label=LCams[ii]+' LOS '+LOSRef, **SLOSdict)
            if not ax4 is None:
                LOSdict['Ldict']['c'] = ll[0].get_color()
                ax4 = TFG.Plot_PolProj_GLOS([LD[jj].LOS[LOSRef]['LOS'] for jj in ind], Leg=LCams[ii]+' '+LOSRef, ax=ax4, **LOSdict)
                ax4.set_xlim(XL), ax4.set_ylim(YL)
            Xticks.append(range(II,II+len(ind)))
            XTicksLab.append([LD[jj].Id.Name for jj in ind])
            II += len(ind)
        Xticks = list(itt.chain.from_iterable(Xticks))
        XTicksLab = list(itt.chain.from_iterable(XTicksLab))
        ax1.set_xlim(0,self.LD_nDetect+1)
        ax1.set_xticks(Xticks)
        ax1.set_xticklabels(XTicksLab, rotation=40)
        if not ax2 is None:
            ax2.set_xlim(XL), ax2.set_ylim(YL)
        plt.gcf().canvas.draw()
        return ax1, ax2, ax3, ax4

    def save(self,SaveName=None,Path=None,Mode='npz'):
        if Path is None:
            Path = self.Id.SavePath
        else:
            assert type(Path) is str, "Arg Path must be a str !"
            self._Id.SavePath = Path
        if SaveName is None:
            SaveName = self.Id.SaveName
        else:
            assert type(SaveName) is str, "Arg SaveName must be a str !"
            self.Id.SaveName = SaveName
        Ext = '.npz' if 'npz' in Mode else '.pck'
        TFPF.save(self, Path+SaveName+Ext)


"""
###############################################################################
###############################################################################
                Computation functions
###############################################################################
"""


def Calc_GeomMatrix_2D_Iso(BF2, LD, indMat, Mode='quad', epsrel=TFD.GMMatepsrel, SubP=TFD.GMindMSubP, SubMode=TFD.GMindMSubPMode, SubTheta=TFD.GMindMSubTheta, SubThetaMode=TFD.GMindMSubThetaMode, Fast=True, Verb=True, Test=True):
    if Test:
        assert isinstance(BF2,TFM.BF2D), "Arg BF2 must be a TFM.BF2D instance !"
        assert Mode=='quad' or Mode=='simps' or Mode=='trapz', "Arg Mode must be 'quad', 'simps' or 'trapz' !"
        assert isinstance(LD,TFG.GDetect) or isinstance(LD,TFG.Detect) or (type(LD) is list and (all([isinstance(gd,TFG.GDetect) for gd in LD]) or all([isinstance(gd,TFG.Detect) for gd in LD]))), "Arg LD must be a TFG.GDetect or TFG.Detect instance or a list of such !"
    if isinstance(LD,TFG.GDetect):
        LD = LD.LDetect
    if isinstance(LD,TFG.Detect):
        LD = [LD]

    ND = len(LD)
    T = np.zeros((ND,BF2.NFunc))
    err = np.nan*np.ones((ND,BF2.NFunc))
    if Mode=='quad':    # Check and debug
        RZLim = BF2._get_Func_SuppBounds()
        for ii in range(0,ND):
            if Fast and LD[ii]._SAngPol_Reg:
                if Verb:
                    print "        Computing Mat for ", LD[ii].Id.Name, "(fast lane)"
                ind = np.unique(BF2._Cents_Funcind[:,indMat[ii,:]])
                ind = ind[~np.isnan(ind)].astype(int)
                for jj in range(0,ind.size):
                    #print "        Computing Geom. matrix for " + LD[ii].Id.Name + " and BFunc nb. " + str(jj+1) + " / " + str(ind.size)
                    ZMin = lambda y, jj=jj: RZLim[2,ind[jj]]
                    ZMax = lambda y, jj=jj: RZLim[3,ind[jj]]
                    FF = LD[ii]._get_SAngIntMax(Proj='Pol', SAng='Int')
                    if BF2.Deg==0:
                        def FuncSABF(x3,x2,ii=ii,jj=jj):
                            return x2 * FF(np.array([[x2],[x3]]))[0]
                    else:
                        def FuncSABF(x3,x2,ii=ii,jj=jj):
                            return x2 * BF2._LFunc[ind[jj]](np.array([[x2],[x3]]))[0] * FF(np.array([[x2],[x3]]))[0]
                    T[ii,ind[jj]], err[ii,ind[jj]] = GG.dblquad_custom(FuncSABF, RZLim[0,ind[jj]], RZLim[1,ind[jj]], ZMin, ZMax, epsrel=epsrel)
            else:
                if Verb:
                    print "        Computing Mat for ", LD[ii].Id.Name, "(slow lane)"
                LPolys = [LD[ii].Poly]+[aa.Poly for aa in LD[ii].LApert]
                ind = np.unique(BF2._Cents_Funcind[:,indMat[ii,:]])
                ind = ind[~np.isnan(ind)].astype(int)
                for jj in range(0,ind.size):
                    #print "    Computing Geom. matrix for " + LD[ii].Id.Name + " and BFunc nb. " + str(jj+1) + " / " + str(ind.size)
                    RMin = lambda x, jj=jj: RZLim[0,ind[jj]]
                    RMax = lambda x, jj=jj: RZLim[1,ind[jj]]
                    ZMin = lambda x,y, jj=jj: RZLim[2,ind[jj]]
                    ZMax = lambda x,y, jj=jj: RZLim[3,ind[jj]]

                    if BF2.Deg==0:
                        def FuncSABF(x3,x2,x1,ii=ii,jj=jj):
                            P = np.array([np.cos(x1)*x2,np.sin(x1)*x2,x3])
                            return x2 * GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, P, LD[ii].SAng_P.flatten(), LD[ii].SAng_nP.flatten(), LD[ii].SAng_e1.flatten(), LD[ii].SAng_e2.flatten())[0]
                    else:
                        def FuncSABF(x3,x2,x1,ii=ii,jj=jj):
                            P = np.array([np.cos(x1)*x2,np.sin(x1)*x2,x3])
                            return x2 * BF2._LFunc[ind[jj]](np.array([[x2],[x3]]))[0] * GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, P, LD[ii].SAng_P.flatten(), LD[ii].SAng_nP.flatten(), LD[ii].SAng_e1.flatten(), LD[ii].SAng_e2.flatten())[0]
                    T[ii,ind[jj]], err[ii,ind[jj]] = GG.tplquad_custom(FuncSABF, LD[ii].Span_Theta[0], LD[ii].Span_Theta[1], RMin, RMax, ZMin, ZMax, epsrel=epsrel)
                    """
                    except Warning:
                        R = np.linspace(RMin(0),RMax(0),30)
                        Z = np.linspace(ZMin(0,0),ZMax(0,0),30)
                        Theta = np.linspace(LD[ii].Span_Theta[0],LD[ii].Span_Theta[1],20)
                        R, Theta, Z = np.meshgrid(R, Theta, Z)
                        Val = np.zeros(R.shape)
                        for ll in range(0,R.shape[0]):
                            for gg in range(0,R.shape[1]):
                                for kk in range(0,R.shape[2]):
                                    Val[ll,gg,kk] = FuncSABF(Z[ll,gg,kk],R[ll,gg,kk],Theta[ll,gg,kk])
                        indT = np.any(np.any(Val>0,2),1).nonzero()[0][2]
                        plt.figure(), plt.contourf(R[indT,:,:],Z[indT,:,:],Val[indT,:,:]), plt.draw()

                        def FuncSABF2D(x3,x2,ii=ii,jj=jj):
                                P = np.array([[np.cos(Theta[indT,0,0])*x2],[np.sin(Theta[indT,0,0])*x2],[x3]])
                                return x2 * BF2._LFunc[ind[jj]](np.array([[x2],[x3]]))[0] * TFG.Calc_SolAngVect_DetectApert_Fast(LD[ii], P, PBary, nPtemp, e1, e2)[0]

                        A, res = GG.dblquad_custom(FuncSABF2D, RZLim[0,ind[jj]], RZLim[1,ind[jj]], lambda x,jj=jj:RZLim[2,ind[jj]], lambda x,jj=jj:RZLim[3,ind[jj]],epsabs=0,epsrel=1e-2, limit=100)
                    """
    else:
        MinMesh = min(np.min(BF2.Mesh.MeshR._Lengths),np.min(BF2.Mesh.MeshZ._Lengths))
        Supps = BF2._get_Func_SuppBounds()
        if SubMode.lower()=='rel':
            DMin = SubP*min(np.min([np.min(dd._ConeWidth) for dd in LD]), MinMesh)
        else:
            DMin = SubP
        DMin = max(DMin,TFD.GMMatDMinInf)
        for ii in range(0,ND):
            ind = np.unique(BF2._Cents_Funcind[:,indMat[ii,:]])
            ind = ind[~np.isnan(ind)].astype(int)
            if Fast and LD[ii]._SAngPol_Reg:
                if Verb:
                    print "        Computing Mat for " + LD[ii].Id.Name + "  ({0}) BFunc,  DMin = {1} m)".format(ind.size,DMin)
                if Mode=='simps':
                    FF = GG.dblsimps_custom
                elif Mode=='trapz':
                    FF = GG.dbltrapz_custom
                elif Mode=='nptrapz':
                    FF = GG.dblnptrapz_custom
                for jj in range(0,ind.size):
                    NR, NZ = math.ceil((Supps[1,ind[jj]]-Supps[0,ind[jj]])/DMin), math.ceil((Supps[3,ind[jj]]-Supps[2,ind[jj]])/DMin)
                    if BF2.Deg>0:
                        R, Z = np.linspace(Supps[0,ind[jj]],Supps[1,ind[jj]],NR), np.linspace(Supps[2,ind[jj]],Supps[3,ind[jj]],NZ)
                    else:
                        R, Z = np.linspace(Supps[0,ind[jj]],Supps[1,ind[jj]]-DMin/1000.,NR), np.linspace(Supps[2,ind[jj]],Supps[3,ind[jj]]-DMin/1000.,NZ)
                    PtsRZ = np.array([np.tile(R,(NZ,1)).T.flatten(), np.tile(Z,(NR,1)).flatten()])
                    Val = BF2._LFunc[ind[jj]](PtsRZ) * LD[ii]._get_SAngIntMax(Proj='Pol', SAng='Int')(PtsRZ)  # !!! R was already included in SAngInt !!!
                    T[ii,ind[jj]] = FF(Val.reshape((NR,NZ)), x=R, y=Z)
            else:
                nind = math.ceil(ind.size/5.)
                P, nP, e1, e2 = LD[ii]._SAngPlane
                LPolys = [LD[ii].Poly]+[aa.Poly for aa in LD[ii].LApert]
                if SubThetaMode.lower()=='rel':
                    NT= np.round(1./SubTheta)
                else:
                    NT= np.round((LD[ii]._Span_Theta[1]-LD[ii]._Span_Theta[0])/SubTheta)
                Theta = np.linspace(LD[ii]._Span_Theta[0], LD[ii]._Span_Theta[1], NT, endpoint=True)
                if Mode=='simps':
                    FF = GG.tplsimps_custom
                elif Mode=='trapz':
                    FF = GG.tpltrapz_custom
                elif Mode=='nptrapz':
                    FF = GG.tplnptrapz_custom
                for jj in range(0,ind.size):
                    if Verb and (jj==0 or (jj+1)%nind==0):
                        print "        Computing Mat for " + LD[ii].Id.Name + " and BFunc nb. " + str(jj+1) + " / " + str(ind.size)
                    NR, NZ = math.ceil((Supps[1,ind[jj]]-Supps[0,ind[jj]])/DMin), math.ceil((Supps[3,ind[jj]]-Supps[2,ind[jj]])/DMin)
                    if BF2.Deg>0:
                        R, Z = np.linspace(Supps[0,ind[jj]],Supps[1,ind[jj]],NR), np.linspace(Supps[2,ind[jj]],Supps[3,ind[jj]],NZ)
                    else:
                        R, Z = np.linspace(Supps[0,ind[jj]],Supps[1,ind[jj]]-DMin/1000.,NR), np.linspace(Supps[2,ind[jj]],Supps[3,ind[jj]]-DMin/1000.,NZ)
                    Rvf, Thetavf, Zvf = np.meshgrid(R, Theta, Z, indexing='ij')
                    Rvf, Thetavf, Zvf = Rvf.flatten(order='F'), Thetavf.flatten(order='F'), Zvf.flatten(order='F')
                    Points = np.array([Rvf*np.cos(Thetavf), Rvf*np.sin(Thetavf), Zvf])
                    inds = LD[ii]._isOnGoodSide(Points) & LD[ii].isInside(Points,In='(X,Y,Z)')
                    Val = np.zeros((NR*NT*NZ,))
                    if np.any(inds):
                        Val[inds] = Rvf[inds] * BF2._LFunc[ind[jj]](np.array([Rvf[inds],Zvf[inds]])) * GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,inds], P, nP, e1, e2)[0]
                    T[ii,ind[jj]] = FF(Val.reshape((NR,NT,NZ),order='F'), x=R, y=Theta, z=Z)
    return T


def Calc_indMatrix_2D(BF2, LD, SubP=TFD.GMindMSubP, Verb=True, Test=True):
    if Test:
        assert isinstance(BF2,TFM.BF2D), "Arg BF2 must be a TFM.BF2D instance !"
        assert isinstance(LD,TFG.GDetect) or isinstance(LD,TFG.Detect) or (type(LD) is list and (all([isinstance(gd,TFG.GDetect) for gd in LD]) or all([isinstance(gd,TFG.Detect) for gd in LD]))), "Arg LD must be a TFG.GDetect or TFG.Detect instance or a list of such !"
    if isinstance(LD,TFG.GDetect):
        LD = LD.LDetect
    if isinstance(LD,TFG.Detect):
        LD = [LD]
    ND = len(LD)
    MinDL = SubP*min(np.min([np.min(dd._ConeWidth) for dd in LD]), min(np.min(BF2.Mesh.MeshR._Lengths),np.min(BF2.Mesh.MeshZ._Lengths)))
    MinDL = max(MinDL,TFD.GMMatDMinInf)
    indMat = np.zeros((ND,BF2.Mesh.NCents),dtype=bool)
    N = math.ceil(BF2.Mesh.NCents/5.)
    for jj in range(0,BF2.Mesh.NCents):
        if Verb and (jj==0 or (jj+1)%N==0):
            print "        Computing ind matrix for ", jj+1, "/", BF2.Mesh.NCents
        R,Z = np.unique(BF2.Mesh.Knots[0,BF2.Mesh._Cents_Knotsind[:,jj]]), np.unique(BF2.Mesh.Knots[1,BF2.Mesh._Cents_Knotsind[:,jj]])
        DR, DZ = list(np.linspace(R[0],R[1],math.ceil(abs(R[1]-R[0])/MinDL),endpoint=True)), list(np.linspace(Z[0],Z[1],math.ceil(abs(Z[1]-Z[0])/MinDL),endpoint=True))
        NR, NZ = len(DR), len(DZ)
        for ii in range(0,ND):
            indMat[ii,jj] = np.any(LD[ii].isInside(np.array([[R[0]]*NZ+[R[1]]*NZ+DR+DR, DZ+DZ+[Z[0]]*NR+[Z[1]]*NR]),In='(R,Z)'))
    return indMat


"""
def Calc_indMatrix_2D(BF2, LD, SubP=TFD.GMindMSubP, SubMode=TFD.GMindMSubPMode, SubTheta=TFD.GMindMSubTheta, SubThetaMode=TFD.GMindMSubThetaMode, Test=True):       # Not used ?
    if Test:
        assert isinstance(BF2,TFM.BF2D), "Arg BF2 must be a TFM.BF2D instance !"
        assert isinstance(LD,TFG.Detect) or (type(LD) is list and [isinstance(dd,TFG.Detect) for dd in LD]), "Arg LD must be a TFG.Detect instance or a list of such !"
        # assert Method in ['Vis','ConePoly'], "Arg Method must be in ['Vis','ConePoly'] !"

    SubMode, SubThetaMode = SubMode.lower(), SubThetaMode.lower()
    if isinstance(LD,TFG.Detect):
        LD = [LD]
    ND = len(LD)
    indMat = np.zeros((ND,BF2.Mesh.NCents),dtype=bool)
    for ii in range(0,ND):
        print "    Computing ind matrix for " + LD[ii].Id.Name
        if SubThetaMode=='rel':
            NT= np.round(1./SubTheta)
        else:
            NT= np.round((LD[ii].Span_Theta[1]-LD[ii].Span_Theta[0])/SubTheta)
        Theta = np.linspace(LD[ii].Span_Theta[0], LD[ii].Span_Theta[1], NT, endpoint=True)
        #LPolys = [LD[ii].Poly] + [aa.Poly for aa in LD[ii].LApert]
        for jj in range(0,BF2.Mesh.NCents):
            R, Z = TFM.Calc_SumMeshGrid2D(np.unique(BF2.Mesh.Knots[0,BF2.Mesh._Cents_Knotsind[:,jj]]), np.unique(BF2.Mesh.Knots[1,BF2.Mesh._Cents_Knotsind[:,jj]]), SubP=SubP, SubMode=SubMode, Test=False)
            NR, NZ = R.size, Z.size
            RZ = R.reshape((NR,1)).dot(np.ones((1,NZ))).flatten()
            RT = R.reshape((NR,1)).dot(np.ones((1,NT))).flatten()
            TR = np.ones((NR,1)).dot(Theta.reshape((1,NT))).flatten()
            TZ = Theta.reshape((NT,1)).dot(np.ones((1,NZ))).flatten()
            ZR = np.ones((NR,1)).dot(Z.reshape((1,NZ))).flatten()
            ZT = np.ones((NT,1)).dot(Z.reshape((1,NZ))).flatten()
            RR = np.concatenate((R[0]*np.ones((NR*NT,)), R[-1]*np.ones((NR*NT,)), RZ, RZ, RT, RT))
            TT = np.concatenate((TZ, TZ, Theta[-1]*np.ones((NR*NZ,)), Theta[-1]*np.ones((NR*NZ,)), TR, TR))
            Points = np.array([RR*np.cos(TT), RR*np.sin(TT), np.concatenate((ZT,TZ,ZR,ZR,Z[0]*np.ones((NR*NT,)),Z[-1]*np.ones((NR*NT,))))])
            indMat[ii,jj] = np.any(LD[ii].isinside_ConePoly(Points))
    return indMat
"""


def Calc_GeomMatrix_LOS_Iso(BF2, LD, indMat, Mode='quad', epsrel=TFD.GMMatepsrel, SubP=TFD.GMindMSubP, SubMode='Rel', LOSRef=None, Verb=True, Test=True):
    if Test:
        assert isinstance(BF2,TFM.BF2D), "Arg BF2 must be a TFM.BF2D instance !"
        assert isinstance(LD,TFG.GDetect) or isinstance(LD,TFG.Detect) or (type(LD) is list and (all([isinstance(gd,TFG.GDetect) for gd in LD]) or all([isinstance(gd,TFG.Detect) for gd in LD]))), "Arg LD must be a TFG.GDetect or TFG.Detect instance or a list of such !"
        assert Mode in ['quad','simps','trapz','nptrapz'], "Arg Mode must be in ['quad','simps','trapz','nptrapz'] !"
    SubMode = SubMode.lower()

    if isinstance(LD,TFG.GDetect):
        LD = LD.LDetect
    if isinstance(LD,TFG.Detect):
        LD = [LD]
    ND = len(LD)
    T = np.zeros((ND,BF2.NFunc))
    err = np.nan*np.ones((ND,BF2.NFunc))
    Poly = BF2._get_Func_Supps()
    Vin = np.array([[0.,-1.,0.,1.],[1.,0.,-1.,0.]])
    if LOSRef is None:
        LOSRef = LD[0]._LOSRef
    if Mode=='quad':
        for ii in range(0,ND):
            if Verb:
                print "        Computing MatLOS for " + LD[ii].Id.Name
            ThetaLim = (LD[ii]._Span_Theta[0], LD[ii]._Span_Theta[1])
            ind = np.unique(BF2._Cents_Funcind[:,indMat[ii,:]])
            ind = ind[~np.isnan(ind)].astype(int)
            for jj in range(0,ind.size):
                SIn, SOut = GG.Calc_InOut_LOS_PIO(LD[ii].LOS[LOSRef]['LOS'].D.reshape((3,1)), LD[ii].LOS[LOSRef]['LOS'].u.reshape((3,1)), Poly[ind[jj]], Vin)
                if SIn.shape[1]>0 and SOut.shape[1]>0:
                    if SIn.shape[1]>1:
                        sp = np.sum((SIn-np.tile(LD[ii].LOS[LOSRef]['LOS'].D,(SIn.shape[1],1)).T)**2,axis=0)
                        SIn = SIn[:,np.argmin(sp)]
                    if SOut.shape[1]>1:
                        sp = np.sum((SOut-np.tile(LD[ii].LOS[LOSRef]['LOS'].D,(SOut.shape[1],1)).T)**2,axis=0)
                        SOut = SOut[:,np.argmin(sp)]
                    Thetas = (np.arctan2(SIn[1],SIn[0]), np.arctan2(SOut[1],SOut[0]))
                    if Thetas[0]>=ThetaLim[0] and Thetas[0]<=ThetaLim[1] and Thetas[1]>=ThetaLim[0] and Thetas[1]<=ThetaLim[1]:
                        SIn, SOut = SIn.flatten(), SOut.flatten()
                        DMax = np.linalg.norm(SOut-SIn)
                        nS = (SOut-SIn)/DMax
                        def FuncSABF(ss):
                            P = SIn + nS*ss
                            return BF2._LFunc[ind[jj]](np.array([[np.hypot(P[0],P[1])],[P[2]]]))[0]
                        T[ii,ind[jj]], err[ii,ind[jj]] = scpinteg.quad(FuncSABF, 0, DMax, epsabs=0., epsrel=epsrel)
            T[ii,:] = T[ii,:] * LD[ii].LOS[LOSRef]['Etend']
    else:
        if Mode=='trapz':
            F = scpinteg.trapz
        elif Mode=='simps':
            F = scpinteg.simps
        elif Mode=='nptrapz':
            F = np.trapz
        for ii in range(0,ND):
            if Verb:
                print "        Computing MatLOS for " + LD[ii].Id.Name
            NP = round(1./SubP)
            ind = np.unique(BF2._Cents_Funcind[:,indMat[ii,:]])
            ind = ind[~np.isnan(ind)].astype(int)
            for jj in range(0,ind.size):
                SIn, SOut = GG.Calc_InOut_LOS_PIO( LD[ii].LOS[LOSRef]['LOS'].D.reshape((3,1)), LD[ii].LOS[LOSRef]['LOS'].u.reshape((3,1)), Poly[ind[jj]], Vin)
                if SIn.shape[1]>0 and SOut.shape[1]>0:
                    if SIn.shape[1]>1:
                        sp = np.sum((SIn-np.tile(LD[ii].LOS[LOSRef]['LOS'].D,(SIn.shape[1],1)).T)**2,axis=0)
                        SIn = SIn[:,np.argmin(sp)]
                    if SOut.shape[1]>1:
                        sp = np.sum((SOut-np.tile(LD[ii].LOS[LOSRef]['LOS'].D,(SOut.shape[1],1)).T)**2,axis=0)
                        SOut = SOut[:,np.argmin(sp)]
                    Thetas = (np.arctan2(SIn[1,0],SIn[0,0]), np.arctan2(SOut[1,0],SOut[0,0]))
                    if Thetas[0]>=LD[ii]._Span_Theta[0] and Thetas[0]<=LD[ii]._Span_Theta[1] and Thetas[1]>=LD[ii]._Span_Theta[0] and Thetas[1]<=LD[ii]._Span_Theta[1]:
                        SIn, SOut = SIn.flatten(), SOut.flatten()
                        DMax = np.linalg.norm(SOut-SIn)
                        nS = (SOut-SIn)/DMax
                        DV = DMax*SubP if SubMode=='rel' else DMax/SubP
                        NP = math.ceil(DMax/DV)
                        ss = np.linspace(0,DMax,NP,endpoint=True)
                        Val = BF2._LFunc[ind[jj]](np.array([np.hypot(SIn[0]+nS[0]*ss,SIn[1]+nS[1]*ss), SIn[2]+nS[2]*ss]))
                        T[ii,ind[jj]] = F(Val, x=ss)
            T[ii,:] = T[ii,:] * LD[ii].LOS[LOSRef]['Etend']
    return T


def Regroup_GMat2D(LGMat, Id=None, dtime=None, CompGM=True):           # Finish update and debugging !!!!
    assert type(LGMat) is list and all([isinstance(GMat,GMat2D) for GMat in LGMat]), "Arg LGMat must be a list of GMat2D instances !"
    assert all([L.Id.BF2_SaveName==LGMat[0].Id.BF2_SaveName for L in LGMat]), "All GeomMat2 instances must have the same TFM.BF2D instance !"
    assert all([L.Mat_Mode==LGMat[0].Mat_Mode and L.Mat_epsrel==LGMat[0].Mat_epsrel and L.Mat_SubP==LGMat[0].Mat_SubP and L.Mat_SubMode==LGMat[0].Mat_SubMode and L.Mat_SubTheta==LGMat[0].Mat_SubTheta]), "All GeomMat2 instances must have the same computation mode and parameters for Mat !"
    assert all([L.Mat_Iso==LGMat[0].Mat_Iso for L in LGMat]), "All GeomMat2 instances must have the same Mat_Iso !"
    assert all([L.MatLOS_Mode==LGMat[0].MatLOS_Mode and L.MatLOS_epsrel==LGMat[0].MatLOS_epsrel and L.MatLOS_SubP==LGMat[0].MatLOS_SubP and L.MatLOS_SubMode==LGMat[0].MatLOS_SubMode]), "All GeomMat2 instances must have the same computation mode and parameters for MatLOS !"
    assert all([L.indMat_SubP==LGMat[0].indMat_SubP and L.indMat_SubMode==LGMat[0].indMat_SubMode and L.indMat_SubTheta==LGMat[0].indMat_SubTheta]), "All GeomMat2 instances must have the same computation parameters for indMat !"
    assert Id is None or type(Id) is str, "Arg Id must be None or a str !"
    assert dtime is None or isinstance(dtime,dtm.datetime), "Arg dtime must be None or a dtm.datetime instance !"

    LD = []
    GDetect_nGDetect, GDetect_Names, GDetect_SaveNames, LDetect_GD_SaveNames, LDetect_GD_Names = 0, [], [], [], []
    indMat, Mat, MatLOS = [], [], []
    for ii in range(0,len(LGMat)):
        LD += LGMat[ii].get_LD()
        GDetect_nGDetect += LGMat[ii].Id.GDetect_nGDetect
        GDetect_Names += LGMat[ii].Id.GDetect_Names
        GDetect_SaveNames += LGMat[ii].Id.GDetect_SaveNames
        LDetect_GD_SaveNames += LGMat[ii].Id.LDetect_GD_SaveNames
        LDetect_GD_Names += LGMat[ii].Id.LDetect_GD_Names
        indMat.append(LGMat[ii].indMat)
        Mat.append(LGMat[ii].Mat_csr.todense())
        MatLOS.append(LGMat[ii].MatLOS_csr.todense())
    indMat = np.concatenate(tuple(indMat),axis=0)
    Mat = np.concatenate(tuple(Mat),axis=0)
    MatLOS = np.concatenate(tuple(MatLOS),axis=0)

    if Id is None:
        Id = ''
        for ii in range(0,len(LGMat)):
            Id += LGMat[ii].Id.Name+'_'
        Id = Id[:-1]
    if dtime is None:
        dts = [mm.Id.dtime for mm in LGMat]
        if all([dd==dts[0] for dd in dts]):
            dtime = dts[0]
        else:
            dtime = dtm.datetime.now()

    GM = GMat2D(Id, LGMat[0].get_BF2(), LD, Mat=Mat, indMat=indMat, MatLOS=MatLOS, CompGM=CompGM, Iso=LGMat[0].Mat_Iso, dTime=dtime,
            Mode=LGMat[0].Mat_Mode, epsrel=LGMat[0].Mat_epsrel, SubP=LGMat[0].Mat_SubP, SubMode=LGMat[0].Mat_SubMode, SubTheta=LGMat[0].Mat_SubTheta, SubThetaMode=LGMat[0].Mat_SubThetaMode,
            SubPind=LGMat[0].indMat_SubP, SubModeind=LGMat[0].indMat_SubMode, SubThetaind=LGMat[0].indMat_SubTheta, SubThetaModeind=LGMat[0].indMat_SubThetaMode,
            ModeLOS=LGMat[0].MatLOS_Mode, epsrelLOS=LGMat[0].MatLOS_epsrel, SubPLOS=LGMat[0].MatLOS_SubP, SubModeLOS=LGMat[0].MatLOS_SubMode)

    GM.Id.GDetect_nGDetect = GDetect_nGDetect
    GM.Id.GDetect_Names = GDetect_Names
    GM.Id.GDetect_SaveNames = GDetect_SaveNames
    GM.Id.LDetect_GD_SaveNames = LDetect_GD_SaveNames
    GM.Id.LDetect_GD_Names = LDetect_GD_Names
    print "    Created (but not saved) : "+GM.Id.SaveName
    return GM


"""
###############################################################################
###############################################################################
                Plotting functions
###############################################################################
"""

def Plot_GeomMatrix_Detect(D, BF2, Ti=None, axPol=None, axMesh=None, axBF=None, Mask=False, indMask='', TLOS=False, TiLOS=None, SubP=TFD.GMPlotDetSubP, SubMode=TFD.GMPlotDetSubPMode,
        Cdict=TFD.GMPlotDetCd, Ldict=TFD.GMPlotDetLd, LdictLOS=TFD.GMPlotDetLOSd, Test=True, a4=False):
    if Test:
        assert isinstance(D, TFG.Detect), "Arg D must be a Detect instance !"
        assert isinstance(BF2, TFM.BF2D), "Arg BF2 must be a BF2D instance !"
        assert not (Ti is None and TiLOS is None), "No geometry matrix available !"
        assert Ti is None or (isinstance(Ti, np.ndarray) and (Ti.shape==(BF2.NFunc,) or Ti.shape==(1,BF2.NFunc))), "Arg Ti must be a (N,) or (1,N) np.ndarray (line of the geometry matrix) !"
        assert all([ax is None or isinstance(ax,plt.Axes) for ax in [axPol,axMesh,axBF]]), "Args axPol, axMesh and axBF must be plt.Axes instances or None !"
        assert type(SubP) is float, "Arg SubP must be a float !"
        assert SubMode.lower() in ['rel','abs'], "Arg SubMode must be 'rel' or 'abs' !"
        assert type(Cdict) is dict, "Arg Cdict must be dict !"
        assert type(Ldict) is dict, "Arg Ldict must be dict !"
        assert type(Mask) is bool and ((Mask and isinstance(indMask,np.ndarray) and indMask.size==BF2.NFunc and indMask.dtype.name=='bool') or ~Mask), "Arg Mask must be a bool and arg indMask must be a np.ndarray of bool !"
    if not Mask:
        indMask = np.ones((BF2.Mesh.NCents,),dtype=bool)
    RZLim = BF2._get_Func_SuppBounds()
    if not Ti is None:
        ind = Ti.nonzero()[0]
        indC = np.unique(BF2._Func_Centsind[:,ind])
        Val = np.zeros((ind.size,BF2.Mesh.NCents))
        for ii in range(0,ind.size):
            Val[ii,indC] = Ti[ind[ii]]*(indMask[indC] & (BF2.Mesh.Cents[0,indC]>=RZLim[0,ind[ii]]) & (BF2.Mesh.Cents[0,indC]<RZLim[1,ind[ii]]) & (BF2.Mesh.Cents[1,indC]>=RZLim[2,ind[ii]]) & (BF2.Mesh.Cents[1,indC]<RZLim[3,ind[ii]]))
        Val = np.sum(Val,axis=0)
    if not TiLOS is None:
        ind = TiLOS.nonzero()[0]
        indCLOS = np.unique(BF2._Func_Centsind[:,ind])
        ValLOS = np.zeros((ind.size,BF2.Mesh.NCents))
        for ii in range(0,ind.size):
            ValLOS[ii,indCLOS] = TiLOS[ind[ii]]*(indMask[indCLOS] & (BF2.Mesh.Cents[0,indCLOS]>=RZLim[0,ind[ii]]) & (BF2.Mesh.Cents[0,indCLOS]<RZLim[1,ind[ii]]) & (BF2.Mesh.Cents[1,indCLOS]>=RZLim[2,ind[ii]]) & (BF2.Mesh.Cents[1,indCLOS]<RZLim[3,ind[ii]]))
        ValLOS = np.sum(ValLOS,axis=0)
    if axPol is None and axMesh is None and axBF is None:
        axPol, axMesh, axBF = TFD.Plot_GeomMatrix_Mesh_DefAxes(a4=a4)

    if not axPol==None:
        indMask = np.logical_and(indMask,Val>0).nonzero()[0] if not Ti is None else np.logical_and(indMask,ValLOS>0).nonzero()[0]
        patch = []
        for ii in range(0,indMask.size):
            Rk = np.unique(BF2.Mesh.Knots[0,BF2.Mesh._Cents_Knotsind[:,indMask[ii]]])
            Zk = np.unique(BF2.Mesh.Knots[1,BF2.Mesh._Cents_Knotsind[:,indMask[ii]]])
            patch.append(patches.Polygon(np.array([[Rk[0], Zk[0]], [Rk[1], Zk[0]], [Rk[1], Zk[1]], [Rk[0], Zk[1]]]), True))
        ppp = PcthColl(patch, **Cdict)
        if not Ti is None:
            ppp.set_array(Val[indC])
        else:
            ppp.set_array(ValLOS[indCLOS])
        CNb = axPol.add_collection(ppp)
        cbar = plt.colorbar(CNb, ax=axPol, orientation='vertical', fraction=0.10, pad=0.05, shrink=0.8, panchor=(1.0,0.0),anchor=(0.0,0.0), extend='neither')
    if not axMesh==None:
        if not Ti is None:
            axMesh.plot(np.arange(0,BF2.Mesh.NCents), Val, label=D.Id.NameLTX, **Ldict)
        if not TiLOS is None:
            axMesh.plot(np.arange(0,BF2.Mesh.NCents), ValLOS, label=D.Id.NameLTX+' LOS', **LdictLOS)
    if not axBF==None:
        if not Ti is None:
            axBF.plot(np.arange(0,BF2.NFunc), Ti, label=D.Id.NameLTX, **Ldict)
        if not TiLOS is None:
            axBF.plot(np.arange(0,BF2.NFunc), TiLOS, label=D.Id.NameLTX+' LOS', **LdictLOS)
    return axPol, axMesh, axBF


def Plot_GeomMatrix_BFunc(LD, Names, Nb, Ti=None, axPol=None, axD=None, axDred=None, TLOS=False, TiLOS=None, Ldict=TFD.GMPlotDetLd, LdictLOS=TFD.GMPlotDetLOSd, KWArgDet=TFD.GMPlotBFDetd, a4=False, Test=True):
    if Test:
        assert type(LD) is list and all([isinstance(D,TFG.Detect) for D in LD]), "Arg LD must be a list of TFG.Detect instances !"
        assert type(Names) is list and all([type(N) is str for N in Names]), "Arg Names must be a list of str !"
        assert all([tt is None or (isinstance(tt,np.ndarray) and (tt.shape==(len(LD),) or tt.shape==(len(LD),1))) for tt in [Ti,TiLOS]]), "Arg Ti must be a (N,) or (N,1) np.ndarray (column of the geometry matrix) !"
        assert not (Ti is None and TiLOS is None), "No geometry matrix was provided !"
        assert all([ax==None or ax=='None' or isinstance(ax,plt.Axes) for ax in [axPol,axD,axDred]]), "Args axPol, axD and axDred must be plt.Axes instances or None !"
        assert type(Ldict) is dict, "Arg Ldict must be dict !"

    if axPol is None and axD is None and axDred is None:
        axPol, axD, axDred = TFD.Plot_GeomMatrix_Mesh_DefAxes(a4=a4)
    nD = len(LD)
    ind = Ti.nonzero()[0] if (not Ti is None) else TiLOS.nonzero()[0]
    if not axPol is None:
        Col = plt.cm.jet
        NC = Col.N
        for ii in range(0,len(ind)):
            cc = Col(int((Ti[ind[ii]]-np.min(Ti))*NC/(np.max(Ti)-np.min(Ti)))) if not Ti is None else Col(int((TiLOS[ind[ii]]-np.min(TiLOS))*NC/(np.max(TiLOS)-np.min(TiLOS))))
            KWArgDet['Ldict']['c'] = cc
            KWArgDet['Conedict']['facecolors'] = (cc[0],cc[1],cc[2],KWArgDet['Conedict']['facecolors'][3])
            LD[ind[ii]].plot(Proj='Pol', Lax=[axPol], **KWArgDet)

    if not axD is None:
        if not Ti is None:
            axD.plot(np.arange(0,nD), Ti, label='BF nb. '+ str(Nb), **Ldict)
        if TLOS and not TiLOS is None:
            axD.plot(np.arange(0,nD), TiLOS, label='BF nb. '+ str(Nb)+' LOS', **LdictLOS)
        axD.set_xlabel('Detect index (starts at 0)')

    if not axDred is None:
        Names = [Names[ii] for ii in ind]
        Names = [Name.replace('_',' ') for Name in Names]
        if not Ti is None:
            axDred.plot(np.arange(1,ind.size+1), Ti[ind], label='BF nb. '+ str(Nb), **Ldict)
        if TLOS and not TiLOS is None:
            axDred.plot(np.arange(1,ind.size+1), TiLOS[ind], label='BF nb. '+ str(Nb)+' LOS', **LdictLOS)
        axDred.set_xlim(0,ind.size+1)
        axDred.set_xticks(np.arange(1,ind.size+1))
        axDred.set_xticklabels(Names, rotation=60)
        axDred.set_xlabel('')
    return axPol, axD, axDred


