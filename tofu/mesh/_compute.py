# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np

import Polygon as plg
import scipy.sparse as scpsp
import scipy.interpolate as scpinterp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection as PcthColl

import datetime as dtm


# ToFu-specific
import tofu.defaults as tfd
import tofu.pathfile as TFPF
from . import _bsplines_cy as _tfm_bs
from tofu.geom import _GG as TFGG    # For Remap_2DFromFlat() only => make local version to get rid of dependency ?




"""
###############################################################################
###############################################################################
                        Default
###############################################################################
"""

BS2Dict_Def = {'linewidth':0.,'rstride':1,'cstride':1, 'antialiased':False}
Tot2Dict_Def = {'color':(0.7,0.7,0.7),'linewidth':0.,'rstride':1,'cstride':1, 'antialiased':False}
SuppDict_Def = {'facecolor':(0.8,0.8,0.8), 'lw':0.}
PMaxDict_Def = {'color':'g', 'marker':'s', 'markersize':8, 'linestyle':'None', 'linewidth':1.}


"""
###############################################################################
###############################################################################
                        Mesh definitions
###############################################################################
"""

############################################
#####     Computing functions for Mesh1D
############################################


def _Mesh1D_set_Knots(Knots):
    Knots = np.asarray(Knots)
    NCents = Knots.size-1
    NKnots = Knots.size
    Cents = (Knots[:-1]+Knots[1:])/2.
    Lengths = np.diff(Knots)
    Length = Knots.max()-Knots.min()
    Bary = 0.5*(Knots[0]+Knots[-1])
    Cents_Knotsind = np.array([np.arange(0,NKnots-1),np.arange(1,NKnots)])
    Knots_Centsind = np.concatenate((np.array([[np.nan],[0]]),np.array([np.arange(0,NCents-1),np.arange(1,NCents)]), np.array([[NCents-1],[np.nan]])), axis=1)
    return NKnots, Knots.astype(float), NCents, Cents.astype(float), Lengths.astype(float), float(Length), float(Bary), Cents_Knotsind.astype(int), Knots_Centsind


def _Mesh1D_sample(Knots, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
    if Test:
        assert hasattr(Knots,'__iter__') and np.asarray(Knots).ndim==1 and np.all(Knots==np.unique(Knots)), "Arg Knots must be a 1-dim iterable of unique values !"
        assert type(Sub) is float, "Arg SubP must be a float !"
        assert SubMode in ['rel','abs'], "Arg SubMode must be in ['rel','abs'] !"

    nKnots = len(Knots)
    if SubMode=='abs':
        sub = int(round((Knots[-1]-Knots[0])/Sub))+1
        X = np.linspace(Knots[0],Knots[-1],sub,endpoint=True)
    else:
        X = np.reshape(Knots,(1,nKnots))
        X = np.concatenate((X[:,:-1],X[:,1:]),axis=0)
        sub = int(round(1./Sub))
        dd = np.delete(np.linspace(0,1,sub,endpoint=False),0)
        Xtemp = np.dot(np.ones((sub-1,1)),X[0:1,:]) + np.dot(dd.reshape((sub-1,1)), np.diff(X,axis=0))
        X = np.concatenate((X[0:1,:],Xtemp,X[1:2,:]),axis=0)
        X = np.unique(X.T.flatten())
    return X


############################################
#####     Computing functions for Mesh2D
############################################



def _Mesh2D_set_Cents_Knotsind(NCents, KnotsX1, KnotsX2, Cents, Knots):
    Cents_Knotsind = np.zeros((4,NCents))
    for ii in range(0,NCents):
        Rl = KnotsX1[KnotsX1 < Cents[0,ii]].max()
        Rr = KnotsX1[KnotsX1 > Cents[0,ii]].min()
        Zb = KnotsX2[KnotsX2 < Cents[1,ii]].max()
        Zt = KnotsX2[KnotsX2 > Cents[1,ii]].min()
        bl = np.where((Knots[0,:]==Rl)&(Knots[1,:]==Zb))[0][0]
        br = np.where((Knots[0,:]==Rr)&(Knots[1,:]==Zb))[0][0]
        tl = np.where((Knots[0,:]==Rl)&(Knots[1,:]==Zt))[0][0]
        tr = np.where((Knots[0,:]==Rr)&(Knots[1,:]==Zt))[0][0]
        Cents_Knotsind[:,ii] = np.array([bl,br,tr,tl])
    return Cents_Knotsind.astype(int)


def _Mesh2D_set_Knots_Centsind(NKnots, CentsX1, CentsX2, Knots, Cents, NCents):
    Knots_Centsind =  np.zeros((4,NKnots))
    for ii in range(0,NKnots):
        indRl = CentsX1 < Knots[0,ii]
        indRr = CentsX1 > Knots[0,ii]
        indZb = CentsX2 < Knots[1,ii]
        indZt = CentsX2 > Knots[1,ii]
        RlBoo, RrBoo, ZbBoo, ZtBoo = np.any(indRl), np.any(indRr), np.any(indZb), np.any(indZt)
        Rl = CentsX1[indRl].max() if RlBoo else False
        Rr = CentsX1[indRr].min() if RrBoo else False
        Zb = CentsX2[indZb].max() if ZbBoo else False
        Zt = CentsX2[indZt].min() if ZtBoo else False
        ind = (Cents[0,:]==Rl)&(Cents[1,:]==Zb) if RlBoo&ZbBoo else False
        bl = np.where(ind)[0][0] if np.any(ind) else -NCents*10
        ind = (Cents[0,:]==Rl)&(Cents[1,:]==Zt) if RlBoo&ZtBoo else False
        tl = np.where(ind)[0][0] if np.any(ind) else -NCents*10
        ind = (Cents[0,:]==Rr)&(Cents[1,:]==Zb) if RrBoo&ZbBoo else False
        br = np.where(ind)[0][0] if np.any(ind) else -NCents*10
        ind = (Cents[0,:]==Rr)&(Cents[1,:]==Zt) if RrBoo&ZtBoo else False
        tr = np.where(ind)[0][0] if np.any(ind) else -NCents*10
        Knots_Centsind[:,ii] = np.array([bl,br,tr,tl])
    return Knots_Centsind.astype(int)


def _Mesh2D_set_SurfVolBary(Knots, Cents_Knotsind, Cents, VType=None):
    SuppX1 = np.array([Knots[0,Cents_Knotsind].min(axis=0), Knots[0,Cents_Knotsind].max(axis=0)])
    SuppX2 = np.array([Knots[1,Cents_Knotsind].min(axis=0), Knots[1,Cents_Knotsind].max(axis=0)])
    Surfs = ((np.diff(SuppX1,axis=0) * np.diff(SuppX2,axis=0)).flatten()).astype(float)
    Surf = float(np.sum(Surfs))
    VolAngs = (0.5*np.diff(SuppX1**2,axis=0) * np.diff(SuppX2,axis=0)).flatten() if VType=='Tor' else None
    VolAng = float(np.sum(VolAngs)) if VType=='Tor' else None
    BaryS = np.sum(Cents*np.tile(Surfs,(2,1)),axis=1, keepdims=False)/Surf
    CentsV = np.array([(2.*np.diff(SuppX1**3,axis=0)/(3.*np.diff(SuppX1**2,axis=0))).flatten(), 0.5*np.sum(SuppX2,axis=0)]) if VType=='Tor' else None
    BaryV = np.sum(CentsV*np.tile(VolAngs,(2,1)),axis=1,keepdims=False)/VolAng if VType=='Tor' else None
    return Surfs, Surf, VolAngs, VolAng, BaryS, BaryV, CentsV


def _Mesh2D_set_BoundPoly(Knots, Cents_Knotsind, NCents):
    Poly = plg.Polygon(Knots[:,Cents_Knotsind[:,0]].T)
    for ii in range(1,NCents):
        Poly = Poly | plg.Polygon(Knots[:,Cents_Knotsind[:,ii]].T)
    Poly = np.array(Poly[0]).T
    return np.concatenate((Poly,Poly[:,0:1]),axis=1)


def _Mesh2D_get_SubMeshPolygon(Cents, Knots, Cents_Knotsind, Poly, InMode='Cents', NLim=1):
    assert isinstance(Poly,np.ndarray) and Poly.ndim==2 and Poly.shape[0]==2, "Arg Poly must be a (2,N) np.ndarray !"
    assert InMode in ['Cents','Knots'], "Arg InMode must be in ['Cents','Knots'] !"
    if InMode=='Cents':
        indIn = Path(Poly.T).contains_points(Cents.T)
    else:
        indIn = Path(Poly.T).contains_points(Knots.T)
        NIn = np.sum(indIn[Cents_Knotsind],axis=0)
        indIn = NIn >= NLim
    return indIn


def _Mesh2D_get_CentBckg(NCentsX1, NCentsX2, CentsX1, CentsX2, Cents, NC):
    NCents = NCentsX1*NCentsX2
    RCentBck = np.tile(CentsX1,(NCentsX2,1)).flatten()
    ZCentBck = np.tile(CentsX2,(NCentsX1,1)).T.flatten()
    indCentBckInMesh = np.zeros((NCents,),dtype=bool)
    NumCentBck = np.nan*np.ones((NCents,))
    for ii in range(0,NC):
        ind = (RCentBck==Cents[0,ii]) & (ZCentBck==Cents[1,ii])
        indCentBckInMesh[ind] = True
        NumCentBck[ind] = ii
    assert np.sum(indCentBckInMesh) == NC, "    Wrong computation of Background Cents in Mesh !"
    return np.array([RCentBck,ZCentBck]), indCentBckInMesh, NumCentBck


def _Mesh2D_get_KnotsBckg(NKX1, NKX2, KnotsX1, KnotsX2, Knots, NK):
    NKnots = NKX1*NKX2
    RKnotBck = np.tile(KnotsX1,(NKX2,1)).flatten()
    ZKnotBck = np.tile(KnotsX2,(NKX1,1)).T.flatten()
    indKnotsBckInMesh = np.zeros((NKnots,),dtype=bool)
    NumKnotBck = np.nan*np.ones((NKnots,))
    for ii in range(0,NK):
        ind = (RKnotBck==Knots[0,ii]) & (ZKnotBck==Knots[1,ii])
        indKnotsBckInMesh[ind] = True
        NumKnotBck[ind] = ii
    assert np.sum(indKnotsBckInMesh) == NK, "    Wrong computation of Background knots in Mesh !"
    return np.array([RKnotBck,ZKnotBck]).astype(float), indKnotsBckInMesh, NumKnotBck


def _Mesh2D_isInside(Pts2D, BoundPoly):
    assert isinstance(Pts2D,np.ndarray) and Pts2D.ndim==2 and Pts2D.shape[0]==2, "Arg Pts2D must be a (2,N) np.ndarray !"
    return Path(BoundPoly.T).contains_points(Pts2D.T)


def _Mesh2D_sample(Knots, Sub=tfd.BF2Sub, SubMode=tfd.BF2SubMode, BoundPoly=None, Test=True):
    if Test:
        assert hasattr(Knots,'__iter__') and np.asarray(Knots).ndim==2 and np.asarray(Knots).shape[0]==2, "Arg Knots must be a 2-dim iterable !"
        assert type(Sub) is float or (hasattr(Sub,'__iter__') and len(Sub)==2 and all([type(ss) is float for ss in Sub])), "Arg Sub must be a float !"
        assert SubMode in ['rel','abs'], "Arg SubMode must be in ['rel','abs'] !"
        assert BoundPoly is None or (type(BoundPoly) is np.ndarray and BoundPoly.ndim==2 and BoundPoly.shape[0]==2), "Arg BoundPoly must be a (2,N) np.ndarray !"

    Knots = np.asarray(Knots)
    Sub = Sub if hasattr(Sub,'__iter__') else (Sub,Sub)
    XX1 = _Mesh1D_sample(np.unique(Knots[0,:]), Sub=Sub[0], SubMode=SubMode, Test=Test)
    XX2 = _Mesh1D_sample(np.unique(Knots[1,:]), Sub=Sub[0], SubMode=SubMode, Test=Test)
    Pts = np.array([np.tile(XX1,(XX2.size,1)).flatten(), np.tile(XX2,(XX1.size,1)).T.flatten()])
    if BoundPoly is not None:
        ind = _Mesh2D_isInside(Pts, BoundPoly)
        Pts = Pts[:,ind]
    return Pts




"""
###############################################################################
###############################################################################
                        LBF1D computing
###############################################################################
"""


def _LBF1D_get_Coefs(LFunc, NFunc, Knots, xx=None, yy=None, ff=None, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
    if Test:
        assert all([ss is None or type(ss) is np.ndarray and ss.ndim==1 for ss in [xx,yy]]), "Args xx and yy must be 1-dim np.ndarray !"
        assert (xx is None and yy is None) or xx.size==yy.size, "Arg xx and yy must be provided together and they must have the same shape !"
        assert ff is None or hasattr(ff,'__call__'), "Arg ff must be a callable function (with one arguments + optional kwdargs) !"
        assert not all([xx is None or yy is None, ff is None]), "You must provide either a function (ff) or sampled data (xx and yy) !"
        assert not all([ff is not None, Knots is None]), "Knots must be provided if ff is provided !"

    if ff is not None:
        xx = _Mesh1D_sample(Knots, Sub=Sub, SubMode=SubMode, Test=Test)
        yy = ff(xx)

    # Keep only points inside the Boundary
    ind = (xx>=np.nanmin(Knots)) & (xx<=np.nanmax(Knots))
    xx, yy = xx[ind], yy[ind]

    A = _tfm_bs.Calc_BF1D_Weights(LFunc, xx)
    Coefs, res, rank, sing = np.linalg.lstsq(A,yy)
    if rank < NFunc:
        xx1 = _Mesh1D_sample(Knots, Sub=Sub, SubMode=SubMode, Test=Test)
        yy1 = scpinterp.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=0., assume_sorted=True)(xx1)
        xx, yy = np.concatenate((xx,xx1)), np.concatenate((yy,yy1))
        xx, ind = np.unique(xx, return_index=True)
        yy = yy[ind]
        A = _tfm_bs.Calc_BF1D_Weights(LFunc, xx)
        Coefs, res, rank, sing = np.linalg.lstsq(A,yy)
    return Coefs, res



def get_IntVal(N=None, Test=True):

    if not hasattr(Coefs,'__iter__'):
        Coefs = Coefs*np.ones((NFunc,),dtype=float)
    Coefs = np.asarray(Coefs)
    Coefs = Coefs if Coefs.ndim==2 else np.tile(Coefs,(1,1))
    Nt = Coefs.shape[0]

    if Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2']:
        A, m = self.get_IntOp(Deriv=Deriv, Method=Method, Mode=Mode, Sparse=True, SpaFormat=None, N=N, Test=Test)
        if m==0:
            Int = [A.dot(cc) for cc in Coefs]
        elif m==1:
            Int = [cc.dot(A.dot(cc)) for cc in Coefs]

    elif Deriv in ['D0ME','D1FI']:
        Int = _tfm_bs.Calc_1D_IntVal_Quad(N=N)

    Int = np.array(Int) if Nt>1 else Int[0]
    return Int





























############################################
#####     Computing functions
############################################


def Calc_SumMeshGrid2D(Knots1, Knots2, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
    if Test:
        assert isinstance(Knots1,np.ndarray) and Knots1.ndim==1, "Arg Knots must be a 1-dim np.ndarray !"
        assert isinstance(Knots1,np.ndarray) and Knots1.ndim==1, "Arg Knots must be a 1-dim np.ndarray !"
        assert type(Sub) is float, "Arg Sub must be a float !"
        assert type(SubMode) is str and SubMode.lower() in ['rel','abs'], "Arg SubMode must be 'Rel' or 'Abs' !"

    SubMode = SubMode.lower()
    X1, X2 = np.unique(Knots1), np.unique(Knots2)
    nKnots1, nKnots2 = X1.size, X2.size

    if SubMode=='abs':
        sub1, sub2 = int(round((X1[-1]-X1[0])/Sub)), int(round((X2[-1]-X2[0])/Sub))
        X1 = np.linspace(X1[0],X1[-1],sub1,endpoint=True)
        X2 = np.linspace(X2[0],X2[-1],sub2,endpoint=True)
    else:
        X1, X2 = X1.reshape((1,nKnots1)), X2.reshape((1,nKnots2))
        X1, X2 = np.concatenate((X1[:,:-1],X1[:,1:]),axis=0), np.concatenate((X2[:,:-1],X2[:,1:]),axis=0)
        sub = int(round(1./Sub))
        dd = np.delete(np.linspace(0,1,sub,endpoint=False),0)
        Xtemp1 = np.dot(np.ones((sub-1,1)),X1[0:1,:]) + np.dot(dd.reshape((sub-1,1)), np.diff(X1,axis=0))
        Xtemp2 = np.dot(np.ones((sub-1,1)),X2[0:1,:]) + np.dot(dd.reshape((sub-1,1)), np.diff(X2,axis=0))
        X1, X2 = np.concatenate((X1[0:1,:],Xtemp1,X1[1:2,:]),axis=0), np.concatenate((X2[0:1,:],Xtemp2,X2[1:2,:]),axis=0)
        X1, X2 = np.unique(X1.T.flatten()), np.unique(X2.T.flatten())
    return X1, X2


"""
def Calc_IntOp_Elementary(Deg1, Deg2, BF1, BF2, K1, K2, D1, D2, Test=True):
    F1 = BSplineDeriv(Deg1, K1, Deriv=D1, Test=False)
    F2 = BSplineDeriv(Deg2, K2, Deriv=D2, Test=False)

def Calc_IntOp_BSpline(Knots, Deg, Deriv=0, Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
    if Test:
        assert isinstance(Knots,np.ndarray) and Knots.ndim==1, "Arg Knots must be a (N,) np.ndarray instance !"
        assert type(Deg) is int and Deg>=0, "Arg Deg must be a positive int !"
        assert (type(Deriv) is int and Deriv<=Deg) or (type(Deriv) is str and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2','D1FI'] and int(Deriv[1])<=Deg), "Arg Deriv must be a int or a str !"

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])

    NKnots = Knots.size
    NFunc = NKnots-Deg-1
    A = np.zeros((NFunc,NFunc))
    if not 'N' in Deriv and not 'FI' in Deriv:
        m = 0
        if Dbis==0:
            if Deg==0:
                A = np.diff(Knots)
            elif Deg==1:
                A = 0.5*(Knots[2:]-Knots[0:-2])
            elif Deg==2:
                Int1 = (Knots[1:-2] - Knots[0:-3])**2 / (3.*(Knots[2:-1]-Knots[0:-3]))
                Int21 = ( Knots[2:-1]**2 - 2.*Knots[1:-2]**2 + Knots[1:-2]*Knots[2:-1] + 3.*Knots[0:-3]*(Knots[1:-2]-Knots[2:-1]) ) / (6.*(Knots[2:-1]-Knots[0:-3]))
                Int22 = ( -2.*Knots[2:-1]**2 + Knots[1:-2]**2 + Knots[1:-2]*Knots[2:-1] + 3.*Knots[3:]*(Knots[2:-1]-Knots[1:-2]) ) / (6.*(Knots[3:]-Knots[1:-2]))
                Int3 = ( Knots[3:] - Knots[2:-1] )**2 / (3.*(Knots[3:]-Knots[1:-2]))
                A = Int1+Int21+Int22+Int3
            elif Deg==3:
                print "NOT CODED YET !"
                Int1 = (Knots[1:-3]-Knots[0:-4])**3/(4.*(Knots[3:-1]-Knots[0:-4])*(Knots[2:-2]-Knots[0:-4]))
                Int211 = ( (Knots[2:-2]-Knots[0:-4])**4/4. - (Knots[1:-3]-Knots[0:-4])**3*(Knots[2:-2]-Knots[1:-3] + (Knots[1:-3]-Knots[0:-4])/4.) ) / ( 3.*(Knots[3:-1]-Knots[0:-4])*(Knots[2:-2]-Knots[0:-4])*(Knots[2:-2]-Knots[1:-3]) )
                Int212 = ( -Knots[2:-2]**4-Knots[1:-3]**4/12. + (Knots[0:-4]+Knots[3:-1])*(Knots[2:-2]**3-Knots[1:-3]*Knots[2:-2]**2/2.+Knots[1:-3]**3/6.) + Knots[1:-3]*Knots[2:-2]**3/3. - Knots[0:-4]*Knots[3:-1]*(Knots[2:-2]**2+Knots[1:-3]**2)/2. + Knots[0:-4]*Knots[1:-3]*Knots[2:-2]*Knots[3:-4] ) / ( (Knots[3:-1]-Knots[0:-4])*(Knots[2:-2]-Knots[1:-3])*(Knots[3:-1]-Knots[1:-3]) )
                Int22 = (Knots[2:-2]-Knots[1:-3])**2*( (Knots[4:]-Knots[2:-2]) + (Knots[2:-2]-Knots[1:-3]) ) / ( (Knots[4:]-Knots[1:-3])*(Knots[3:-1]-Knots[1:-3]) )
                Int31 = ( (Knots[3:-1]-Knots[2:-2])**2*( (Knots[2:-2]-Knots[0:-4]) + (Knots[3:-1]-Knots[2:-2])/4. ) ) / ( 3.*((Knots[3:-1]-Knots[0:-4])*(Knots[3:-1]-Knots[1:-3])) )
                Int321 = 0 # To be finished
                Int322 = 0 # To be finished
                Int4 = 0 # To be finished
                A = Int1+Int211+Int212+Int22+Int31+Int321+Int322+Int4

        elif Dbis>=1:
            A = np.zeros((NFunc,))

    elif 'N2' in Deriv:
        m = 1
        if Dbis==0:
            if Deg==0:
                A = scpsp.diags([np.diff(Knots)],[0],shape=None,format=SpaFormat)
            elif Deg==1:
                d0 = (Knots[2:]-Knots[0:-2])/3.
                dp1 = (Knots[2:-1]-Knots[1:-2])/6.
                A = scpsp.diags([d0,dp1,dp1], [0,1,-1],shape=None,format=SpaFormat)
            elif Deg==2:
                print "NOT CODED YET !"
                d0 = (Knots[1:-2]-Knots[0:-3])**3/(5.*(Knots[2:-1]-Knots[0:-3])**2) + 0 + 0 # To be finished
                dp1 = 0 # To be finished
                dp2 = 0 # To be finished
                A = scpsp.diags([d0,dp1,dp1,dp2,dp2], [0,1,-1,2,-2],shape=None,format=SpaFormat)
            elif Deg==3:
                print "NOT CODED YET !"
                A = 0
        elif Dbis==1:
            if Deg==1:
                print "NOT CODED YET !"
                A = 0
            elif Deg==2:
                print "NOT CODED YET !"
                A = 0
            elif Deg==3:
                print "NOT CODED YET !"
                A = 0
        elif Dbis==2:
            if Deg==2:
                print "NOT CODED YET !"
                A = 0
            elif Deg==3:
                print "NOT CODED YET !"
                A = 0
        elif Dbis==3:
            if Deg==3:
                print "NOT CODED YET !"
                A = 0

        if not Sparse:
            A = np.array(A.todense())

    return A, m
"""




############################################
#####     Plotting functions
############################################


def Plot_BSpline1D(Knots, FTot, LF, Elt='TL', ax='None', Name='', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, LFdict=tfd.BF1Fd, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
    if Test:
        assert ax=='None' or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt) is str, "Arg Elt must be a str !"
        assert hasattr(FTot, '__call__') or FTot is None, "Arg FTot must be a function !"
        assert (type(LF) is list and all([hasattr(ff,'__call__') for ff in LF])) or LF is None, "Arg BS must be a list of functions !"
        assert type(LFdict) is dict, "Arg LFdict must be a dict !"
        assert type(Totdict) is dict, "Arg Totdict must be a dict !"

    X = Calc_SumMeshGrid1D(Knots, SubP=SubP, SubMode=SubMode, Test=Test)
    if ax=='None':
        ax = tfd.Plot_BSpline_DefAxes('1D')

    if 'L' in Elt and not LF is None:
        NBS = len(LF)
        nx = X.size
        Xplot = np.dot(np.ones((NBS,1)),X.reshape((1,nx)))
        Y = np.nan*np.ones((NBS,nx))
        for ii in range(0,NBS):
            Y[ii,:] = LF[ii](X)
        ax.plot(Xplot.T, Y.T, **LFdict)
    if 'T' in Elt and not FTot is None:
        ax.plot(X, FTot(X), label=Name+' Tot', **Totdict)

    ax.set_xlim(np.min(X),np.max(X))
    if not LegDict is None:
        ax.legend(**LegDict)
    ax.figure.canvas.draw()
    return ax



def Plot_BSpline2D(BF2, ax='None', Elt='T', Deriv=0, Coefs=1., DVect=tfd.BF2_DVect_DefR, NC=tfd.BF2PlotNC, PlotMode=tfd.BF2PlotMode, SubP=tfd.BF2PlotSubP, SubMode=tfd.BF2PlotSubMode, Name='Tot', Totdict=tfd.BF2PlotTotd, LegDict=tfd.TorLegd, Test=True):
    if Test:
        assert ax=='None' or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert type(Totdict) is dict, "Arg Totdict must be a dict !"
        assert type(PlotMode) is str and PlotMode in ['contour','contourf','surf'], "Arg PlotMode must be in ['contour','contourf','surf'] !"
    if type(Coefs) in [int,float]:
        Coefs = Coefs*np.array((BF2._NFunc),dtype=float)

    Xplot, Yplot, nx, ny = BF2.get_XYplot(SubP=SubP, SubMode=SubMode)
    PointsRZ = np.array([Xplot.flatten(),Yplot.flatten()])
    ind = BF2.Mesh.isInside(PointsRZ)

    Val = BF2.get_TotVal(PointsRZ,Deriv=Deriv,Coefs=Coefs,DVect=DVect, Test=Test)
    Val[~ind] = np.nan
    Val = Val.reshape(Xplot.shape)

    if PlotMode=='surf':
        if ax=='None':
            ax = tfd.Plot_BSpline_DefAxes('3D')
        if 'T' in Elt:
            ax.plot_surface(Xplot,Yplot,Val, label=Name, **Totdict)
    else:
        if ax=='None':
            ax = tfd.Plot_BSpline_DefAxes('2D')
        if PlotMode=='contour':
            if 'T' in Elt:
                ax.contour(Xplot,Yplot,Val, NC, label=Name, **Totdict)
        elif PlotMode=='contourf':
            if 'T' in Elt:
                CC = ax.contourf(Xplot,Yplot,Val, NC, label=Name, **Totdict)
                plt.colorbar(CC)
    if not LegDict is None:
        ax.legend(**LegDict)
    ax.figure.canvas.draw()
    return ax


def Plot_BSpline2D_Ind(BF2, Ind, ax='None', Elt='LPS', Coefs=1., NC=tfd.BF2PlotNC, PlotMode=tfd.BF2PlotMode, SubP=tfd.BF2PlotSubP, SubMode=tfd.BF2PlotSubMode, Name='Tot', Totdict=tfd.BF2PlotTotd, Sdict=tfd.BF2PlotIndSd, Pdict=tfd.BF2PlotIndPd, LegDict=tfd.TorLegd, Colorbar=True, Test=True):
    if Test:
        assert ax=='None' or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert type(Totdict) is dict, "Arg Totdict must be a dict !"
        assert type(PlotMode) is str and PlotMode in ['contour','contourf'], "Arg PlotMode must be in ['contour','contourf'] !"

    if type(Coefs) is float:
        Coefs = Coefs*np.ones((BF2.NFunc,))

    NInd = Ind.size
    if ax=='None':
        ax = tfd.Plot_BSpline_DefAxes('2D')

    if 'L' in Elt or 'S' in Elt:
        Poly = BF2._get_Func_Supps()
        Poly = [plg.Polygon(Poly[ii].T) for ii in Ind]
        for ii in range(1,Ind.size):
            Poly[0] = Poly[0] + Poly[ii]
        Poly = Poly[0]
        NPoly = len(Poly)

    if 'S' in Elt:  # Plot support
        NPoly = len(Poly)
        patch = [patches.Polygon(np.array(Poly.contour(ii)), True) for ii in range(0,NPoly)]
        ppp = PcthColl(patch, **Sdict)
        ax.add_collection(ppp)

    if 'P' in Elt:  # Plot Points of max value
        PP = BF2._Func_MaxPos[:,Ind]
        ax.plot(PP[0,:],PP[1,:], label=Name+' PMax', **Pdict)

    if 'L' in Elt:  # Plot local value as contour or contourf
        Hull = np.concatenate(tuple([np.array(Poly.contour(ii)) for ii in range(0,NPoly)]),axis=0)
        MinR, MaxR = np.min(Hull[:,0]), np.max(Hull[:,0])
        MinZ, MaxZ = np.min(Hull[:,1]), np.max(Hull[:,1])

        indKR = np.logical_and(BF2._Mesh._MeshR._Knots>=MinR,BF2._Mesh._MeshR._Knots<=MaxR)
        indKZ = np.logical_and(BF2._Mesh._MeshZ._Knots>=MinZ,BF2._Mesh._MeshZ._Knots<=MaxZ)
        KnotsR, KnotsZ = BF2._Mesh._MeshR._Knots[indKR], BF2._Mesh._MeshZ._Knots[indKZ]
        Xplot, Yplot = Calc_SumMeshGrid2D(KnotsR, KnotsZ, SubP=SubP, SubMode=SubMode, Test=True)
        nx, ny = Xplot.size, Yplot.size
        Xplot, Yplot = np.dot(np.ones((ny,1)),Xplot.reshape((1,nx))), np.dot(Yplot.reshape((ny,1)),np.ones((1,nx)))
        PointsRZ = np.array([Xplot.flatten(),Yplot.flatten()])

        Val = np.sum(np.concatenate(tuple([Coefs[Ind[ii]]*BF2._LFunc[Ind[ii]](PointsRZ).reshape((1,PointsRZ.shape[1])) for ii in range(0,NInd)]),axis=0),axis=0)
        indIn = np.array([Poly.isInside(PointsRZ[0,jj],PointsRZ[1,jj]) for jj in range(0,PointsRZ.shape[1])], dtype=bool)
        Val[~indIn] = np.nan
        Val = Val.reshape(Xplot.shape)
        if PlotMode=='contour':
            cc = ax.contour(Xplot,Yplot,Val, NC, label=Name, **Totdict)
        elif PlotMode=='contourf':
            cc = ax.contourf(Xplot,Yplot,Val, NC, label=Name, **Totdict)
        if Colorbar:
            plt.colorbar(cc,ax=ax,orientation='vertical', anchor=(0.,0.), panchor=(1.,0.), shrink=0.65)

    if not LegDict is None:
        ax.legend(**LegDict)
    ax.figure.canvas.draw()
    return ax



"""

def Plot_BSpline2D_mlab(KnotsMultX, KnotsMultY, BSX, BSY, ax1='None', ax2='None',SubP=0.1, SubMode='Rel', BSDict=BS2Dict_Def, TotDict=Tot2Dict_Def):
    assert isinstance(KnotsMultX,np.ndarray) and KnotsMultX.ndim==1, "Arg KnotsMultX must be a 1-dim np.ndarray !"
    assert isinstance(KnotsMultY,np.ndarray) and KnotsMultY.ndim==1, "Arg KnotsMultY must be a 1-dim np.ndarray !"
    assert ax1=='None' or isinstance(ax1,plt.Axes), "Arg ax1 must be a plt.Axes instance !"
    assert ax2=='None' or isinstance(ax2,plt.Axes), "Arg ax2 must be a plt.Axes instance !"
    assert type(SubP) is float, "Arg SubP must be a float !"
    assert SubMode=='Rel' or SubMode=='Abs', "Arg SubMode must be 'Rel' or 'Abs' !"
    assert type(BSX) is list, "Arg BSX must be a list of functions !"
    assert type(BSY) is list, "Arg BSY must be a list of functions !"
    assert type(BSDict) is dict, "Arg BSDict must be a dict !"
    assert type(TotDict) is dict, "Arg TotDict must be a dict !"

    NBSX, NBSY = len(BSX), len(BSY)
    NBS = NBSX*NBSY
    KnotsUniqX, KnotsUniqY = np.unique(KnotsMultX), np.unique(KnotsMultY)
    nKnotsX, nKnotsY = KnotsUniqX.size, KnotsUniqY.size
    X, Y = KnotsUniqX, KnotsUniqY

    if SubMode=='Abs':
        subX, subY = int(round((X[-1]-X[0])/SubP)), int(round((Y[-1]-Y[0])/SubP))
        X = np.linspace(X[0],X[-1],subX,endpoint=True)
        Y = np.linspace(Y[0],Y[-1],subY,endpoint=True)
    else:
        X, Y = X.reshape((1,nKnotsX)), Y.reshape((1,nKnotsY))
        X, Y = np.concatenate((X[:,:-1],X[:,1:]),axis=0), np.concatenate((Y[:,:-1],Y[:,1:]),axis=0)
        sub = int(round(1./SubP))
        dd = np.delete(np.linspace(0,1,sub,endpoint=False),0)
        Xtemp = np.dot(np.ones((sub-1,1)),X[0:1,:]) + np.dot(dd.reshape((sub-1,1)), np.diff(X,axis=0))
        Ytemp = np.dot(np.ones((sub-1,1)),Y[0:1,:]) + np.dot(dd.reshape((sub-1,1)), np.diff(Y,axis=0))
        X, Y = np.concatenate((X[0:1,:],Xtemp,X[1:2,:]),axis=0), np.concatenate((Y[0:1,:],Ytemp,Y[1:2,:]),axis=0)
        X, Y = np.unique(X.T.flatten()), np.unique(Y.T.flatten())
    nx, ny = X.size, Y.size
    Xplot, Yplot = np.dot(np.ones((ny,1)),X.reshape((1,nx))), np.dot(Y.reshape((ny,1)),np.ones((1,nx)))

    Z = np.nan*np.ones((ny,nx,NBS))
    for ii in range(0,NBSX):
        for jj in range(0,NBSY):
            Z[:,:,(ii-1)*NBSX+jj] = BSX[ii](Xplot)*BSY[jj](Yplot)

    if ax1=='None' or ax2=='None':
        f = Plot_BSpline_2Dmlab_DefFig()

    for ii in range(0,NBS):
        ax1.plot_surface(Xplot,Yplot,Z[:,:,ii], **BSDict)

    ax2.plot_surface(Xplot,Yplot,np.sum(Z,axis=2), label='Tot', **TotDict)
    f.draw()
    return ax1, ax2
"""


"""
###############################################################################
###############################################################################
                    MeshBase Objects and properties
###############################################################################
"""


class BF1D(object):
    def __init__(self, Id, Mesh, Deg, dtime=None):
        self.set_Id(Id,Deg=Deg, dtime=dtime)
        self.set_Mesh(Mesh,Deg=Deg)

    @property
    def Mesh(self):
        """Return the knots"""
        return self._Mesh
    @Mesh.setter
    def Mesh(self,Val):
        """Set a new Knot vector and recompute all subsequent attributes"""
        self.set_Mesh(Val)
    @property
    def Deg(self):
        return self._Deg
    @Deg.setter
    def Deg(self,Val):
        self.set_BF(Deg=Val)

    # Read-only attributes
    @property
    def Id(self):
        """Return the Id"""
        return self._Id

    def set_Id(self,Id,Deg=Deg, dtime=None):
        assert type(Id) is str or isinstance(Id,TFPF.ID), "Arg Id should be string or an TFPF.ID instance !"
        if type(Id) is str:
            Id = TFPF.ID('BF1D',Id+'_D{0:01.0f}'.format(Deg), dtime=dtime)
        self._Id = Id

    def set_Mesh(self,Mesh,Deg=None):
        assert isinstance(Mesh,Mesh1D), "Arg Mesh must be a Mesh1D instance !"
        self._Mesh = Mesh
        self.set_BF(Deg=Deg)

    def set_BF(self,Deg=None):
        assert Deg is None or (type(Deg) is int and Deg>=0 and Deg<=2), "Arg Deg must be a int with Deg>=0 and Deg<=2 !"
        if not Deg is None:
            self._Deg = Deg
        self._LFunc, self._Func_Knotsind, self._Func_Centsind, self._Knots_Funcind, self._Cents_Funcind, self._Func_MaxPos = BSpline_LFunc(self._Deg, self._Mesh._Knots)
        #self._LFunc, self._Func_Knotsind, indlK, self._Func_PMax = BSpline(self._Deg, self._Mesh._Knots)
        self._NFunc = len(self._LFunc)

    def _get_Func_Supps(self):
        Knots = self._Mesh._Knots[self._Func_Knotsind]
        return np.array([Knots.min(axis=0),Knots.max(axis=0)])

    def _get_Func_InterFunc(self):
        if self._Deg==0:
            return np.empty((0,))
        indF = np.empty((2*self._Deg+1,self._NFunc))
        return indF

    def get_TotFunc(self, Deriv=0, Coefs=1., Test=True):
        return BSpline_TotFunc(self._Deg, self._Knots, Deriv=Deriv, Coefs=Coefs, Test=Test)

    def get_TotVal(self, Points, Deriv=0, Coefs=1., Test=True):
        TF = BSpline_get_TotFunc(self.Deg, self.Mesh.Knots, Deriv=Deriv, Coefs=Coefs, Test=Test)
        return Val

    def get_Funcs(self, Deriv=0, Test=True):
        return BSpline_LFunc(self._Deg, self._Mesh._Knots, Deriv=Deriv, Test=Test)

    def get_Coefs(self,xx=None,yy=None,ff=None, SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
        assert not all([xx is None or yy is None, ff is None]), "You must provide either a function (ff) or sampled data (xx and yy) !"

        if not ff is None:
            xx = Calc_SumMeshGrid1D(self.Mesh.Knots, SubP=SubP, SubMode=SubMode, Test=Test)
            yy = ff(xx)

        # Keep only points inside the Boundary
        ind = np.logical_and(xx>=np.min(self.Mesh.Knots), xx<=np.max(self.Mesh.Knots))
        xx, yy = xx[ind], yy[ind]

        A = Calc_BF1D_Weights(self.LFunc, xx, Test=False)
        Coefs, res, rank, sing = np.linalg.lstsq(A,yy)
        if rank < self.NFunc:
            xx1 = Calc_SumMeshGrid1D(self.Mesh.Knots, SubP=SubP, SubMode=SubMode, Test=Test)
            yy1 = scpinterp.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=0., assume_sorted=True)(xx1)
            xx, yy = np.concatenate((xx,xx1)), np.concatenate((yy,yy1))
            xx, ind = np.unique(xx, return_index=True)
            yy = yy[ind]
            A = Calc_BF1D_Weights(self.LFunc, xx, Test=False)
            Coefs, res, rank, sing = np.linalg.lstsq(A,yy)
        return Coefs, res

    def get_IntOp(self, Deriv=0, Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
        return Calc_IntOp_BSpline(self.Mesh.Knots, self.Deg, Deriv=Deriv, Sparse=Sparse, SpaFormat=SpaFormat, Test=Test)

    def get_IntVal(self, Deriv=0, Coefs=1., Test=True):
        A, m = Calc_IntOp_BSpline(self.Mesh.Knots, self.Deg, Deriv=Deriv, Sparse=True, Test=Test)
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,))
        if m==0:
            Int = A.dot(Coefs)
        elif m==1:
            Int = Coefs.dot(A.dot(Coefs))
        else:
            print 'Not coded yet !'
        return Int

    def plot(self, ax='None', Coefs=1., Deriv=0, Elt='TL', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, LFdict=tfd.BF1Fd, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,))
        if 'T' in Elt:
            TotF = BSpline_get_TotFunc(self.Deg, self.Mesh.Knots, Deriv=Deriv, Coefs=Coefs, Test=Test)
        else:
            TotF = None
        if (type(Deriv) is int or Deriv in ['D0','D1','D2','D3']) and 'L' in Elt:
            if not type(Deriv) is int:
                Deriv = int(Deriv[1])
            LF1 = BSplineDeriv(self.Deg, self.Mesh.Knots, Deriv=Deriv, Test=Test)
            LF = [lambda x,Coefs=Coefs,ii=ii: Coefs[ii]*LF1[ii](x) for ii in range(0,len(LF1))]
        else:
            LF = None
        return Plot_BSpline1D(self.Mesh.Knots, TotF, LF, ax=ax, Elt=Elt, Name=self.Id.Name+' '+str(Deriv), SubP=SubP, SubMode=SubMode, LFdict=LFdict, Totdict=Totdict, LegDict=LegDict, Test=Test)

    def plot_Ind(self, ax='None', Ind=0, Coefs=1., Elt='LCK', y=0., SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, LFdict=tfd.BF1Fd, Kdict=tfd.M2Kd, Cdict=tfd.M2Cd, LegDict=tfd.TorLegd, Test=True):
        assert type(Ind) in [int,list,np.ndarray], "Arg Ind must be a int, a list of int or a np.ndarray of int or booleans !"
        if type(Ind) is int:
            Ind = [Ind]
            NInd = len(Ind)
        elif type(Ind) is list:
            NInd = len(Ind)
        elif type(Ind) is np.ndarray:
            assert (np.issubdtype(Ind.dtype,bool) and Ind.size==self.NFunc) or np.issubdtype(Ind.dtype,int), "Arg Ind must be a np.ndarray of boolenas with size==self.NFunc or a np.ndarray of int !"
            NInd = Ind.size

        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,))
        if 'L' in Elt:
            LF1 = BSplineDeriv(self.Deg, self.Mesh.Knots, Deriv=0, Test=Test)
            LF = [lambda x,Coefs=Coefs,ii=ii: Coefs[Ind[ii]]*LF1[Ind[ii]](x) for ii in range(0,NInd)]
            ax = Plot_BSpline1D(self.Mesh.Knots, None, LF, ax=ax, Elt=Elt, Name=self.Id.Name, SubP=SubP, SubMode=SubMode, LFdict=LFdict, LegDict=LegDict, Test=Test)
        if 'C' in Elt or 'K' in Elt:
            Cents = self.Mesh.Cents[self.Func_Centsind[:,Ind]].flatten()
            Knots = self.Mesh.Knots[self.Func_Knotsind[:,Ind]].flatten()
            ax = Plot_Mesh1D(Knots, Cents=Cents, y=y, Elt=Elt, Name=self.Id.NameLTX, ax=ax, Kdict=Kdict, Cdict=Cdict, LegDict=LegDict, Test=Test)
        return ax

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



class BF2D(object):

    def __init__(self, Id, Mesh, Deg, dtime=None):
        self.set_Id(Id,Deg=Deg, dtime=dtime)
        self.set_Mesh(Mesh,Deg=Deg)

    @property
    def Id(self):
        return self._Id
    @property
    def Mesh(self):
        return self._Mesh
    @Mesh.setter
    def Mesh(self,Val):
        self.set_Mesh(Val)
    @property
    def Deg(self):
        return self._Deg
    @Deg.setter
    def Deg(self,Val):
        self.set_BF(Deg=Val)
    @property
    def NFunc(self):
        return self._NFunc

    @property
    def Surf(self):
        indin = np.any(~np.isnan(self._Cents_Funcind),axis=0)
        return np.sum(self.Mesh._Surfs[indin])

    def set_Id(self,Id,Deg=Deg, dtime=None):
        assert type(Id) is str or isinstance(Id,TFPF.ID), "Arg Id should be string or an TFPF.ID instance !"
        if type(Id) is str:
            Id = TFPF.ID('BF2D',Id+'_D{0:01.0f}'.format(Deg), dtime=dtime)
        self._Id = Id

    def set_Mesh(self,Mesh,Deg=None):
        assert isinstance(Mesh,Mesh2D), "Arg Mesh must be a Mesh2D instance !"
        self._Mesh = Mesh
        self.set_BF(Deg=Deg)

    def set_BF(self,Deg=None):
        assert Deg is None or type(Deg) is int, "Arg Deg must be a int !"
        if not Deg is None:
            self._Deg = Deg
        BSR, RF_Kind, RF_Cind, RK_Find, RC_Find, RMaxPos = BSpline_LFunc(self._Deg, self._Mesh._MeshR._Knots)
        BSZ, ZF_Kind, ZF_Cind, ZK_Find, ZC_Find, ZMaxPos = BSpline_LFunc(self._Deg, self._Mesh._MeshZ._Knots)
        nBR, nBZ = len(BSR), len(BSZ)
        Func, F_Kind, F_Cind, F_MaxPos = [], [], [], []

        CentBckg, indCentBckInMesh, NumCentBck = self._Mesh._get_CentBckg()
        NCperF, NKperF = (self._Deg+1)**2, (self._Deg+2)**2
        for ii in range(0,nBZ):
            for jj in range(0,nBR):
                inds = ZF_Cind[:,ii].reshape((ZF_Cind.shape[0],1))*self._Mesh._MeshR._NCents + RF_Cind[:,jj]
                if np.all(indCentBckInMesh[inds]):
                    Func.append(lambda RZ, ii=ii,jj=jj: BSR[jj](RZ[0,:])*BSZ[ii](RZ[1,:]))
                    F_Cind.append(NumCentBck[inds].reshape(NCperF,1).astype(int))
                    F_Kind.append(np.unique(self._Mesh._Cents_Knotsind[:,F_Cind[-1][:,0]]).reshape(NKperF,1))
                    F_MaxPos.append(np.array([[RMaxPos[jj]],[ZMaxPos[ii]]]))
        self._LFunc = Func
        self._NFunc = len(Func)
        self._Func_Knotsind = np.concatenate(tuple(F_Kind),axis=1)
        self._Func_Centsind = np.concatenate(tuple(F_Cind),axis=1)
        self._Func_MaxPos = np.concatenate(tuple(F_MaxPos),axis=1)
        self._Cents_Funcind = self._get_Cents_Funcind(Init=True)
        self._Knots_Funcind = self._get_Knots_Funcind(Init=True)
        self._Func_InterFunc = self._get_Func_InterFunc(Init=True)

    def _get_Cents_Funcind(self,Init=True):                  # To be updated with ability to select fraction of BF
        Cent_indFunc = np.nan*np.ones(((self._Deg+1)**2,self._Mesh._NCents))
        for ii in range(0,self._Mesh._NCents):
            inds = np.any(self._Func_Centsind==ii,axis=0)
            Cent_indFunc[:inds.sum(),ii] = inds.nonzero()[0]
        return Cent_indFunc

    def _get_Knots_Funcind(self,Init=True):                  # To be updated with ability to select fraction of BF
        Knots_indFunc = np.nan*np.ones(((self._Deg+2)**2,self._Mesh._NKnots))
        for ii in range(0,self._Mesh._NKnots):
            inds = np.any(self._Func_Knotsind==ii,axis=0)
            Knots_indFunc[:inds.sum(),ii] = inds.nonzero()[0]
        return Knots_indFunc

    def _get_Func_InterFunc(self,Init=True):                  # To be updated with ability to select fraction of BF
        Func_InterFunc = np.nan*np.ones(((2*self._Deg+1)**2-1,self._NFunc))
        for ii in range(0,self.NFunc):
            ind = self._Cents_Funcind[:,self._Func_Centsind[:,ii]].flatten()
            ind = np.unique(ind[~np.isnan(ind)])
            ind = np.delete(ind,(ind==ii).nonzero()[0])
            Func_InterFunc[:ind.size,ii] = ind.astype(int)
        return Func_InterFunc

    def _get_Func_InterFunc(self, Init=False, indFin=None): # To be updated with ability to select fraction of BF
        assert indFin is None or (isnstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"
        if indFin is None and not Init:
            return self._Func_InterFunc
        elif indFin is None:
            indFin = np.arange(0,self._NFunc)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]
        NF = indFin.size
        Func_InterFunc = np.nan*np.ones(((2*self._Deg+1)**2-1,NF))      # Update in progress from here...
        for ii in range(0,NF):
            ind = self._Cents_Funcind[:,self._Func_Centsind[:,ii]].flatten()
            ind = np.unique(ind[~np.isnan(ind)])
            ind = np.delete(ind,(ind==ii).nonzero()[0])
            Func_InterFunc[:ind.size,ii] = ind.astype(int)
        return Func_InterFunc

    def _get_quadPoints(self):
        R = self._Mesh._Knots[0,self._Func_Knotsind]
        Z = self._Mesh._Knots[1,self._Func_Knotsind]
        QuadR, QuadZ = np.zeros((self._Deg+2,self._NFunc)), np.zeros((self._Deg+2,self._NFunc))
        for ii in range(0,self._NFunc):
            QuadR[:,ii] = np.unique(R[:,ii])
            QuadZ[:,ii] = np.unique(Z[:,ii])
        return QuadR, QuadZ

    def _get_Func_SuppBounds(self):
        Func_SuppRZ = np.nan*np.ones((4,self._NFunc))
        RKnots = self._Mesh._Knots[0,self._Func_Knotsind]
        ZKnots = self._Mesh._Knots[1,self._Func_Knotsind]
        Func_SuppRZ = np.concatenate((RKnots.min(axis=0,keepdims=True), np.max(RKnots,axis=0,keepdims=True), np.min(ZKnots,axis=0,keepdims=True), np.max(ZKnots,axis=0,keepdims=True)),axis=0)
        return Func_SuppRZ

    def _get_Func_Supps(self):
        R = self._Mesh._Knots[0,self._Func_Knotsind]
        Z = self._Mesh._Knots[1,self._Func_Knotsind]
        R = np.array([np.min(R,axis=0),np.max(R,axis=0)])
        Z = np.array([np.min(Z,axis=0),np.max(Z,axis=0)])
        return [np.array([[R[0,ii],R[1,ii],R[1,ii],R[0,ii],R[0,ii]],[Z[0,ii],Z[0,ii],Z[1,ii],Z[1,ii],Z[0,ii]]]) for ii in range(0,self._NFunc)]

    def get_SubBFPolygon_indin(self, Poly, NLim=3, Out=bool):
        assert Out in [bool,int], "Arg Out must be in [bool,int] !"
        indMeshout = ~self.Mesh.get_SubMeshPolygon(Poly, NLim=NLim, Out=bool)
        indFout = self._Cents_Funcind[:,indMeshout]
        indFout = np.unique(indFout[~np.isnan(indFout)]).astype(int)
        indF = np.ones((self.NFunc,),dtype=bool)
        indF[indFout] = False
        if Out==int:
            indF = indF.nonzero()[0]
        return indF

    def get_SubBFPolygonind(self, Poly=None, NLim=3, indFin=None):
        assert indFin is None or (isinstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int','int32','int64']), "Arg indFin must be None or a np.ndarray of bool or int !"
        assert (not indFin is None and Poly is None) or (indFin is None and isinstance(Poly,np.ndarray) and Poly.ndim==2 and Poly.shape[0]==2), "If arg indFin is None, arg Poly must be a 2D np.ndarray instance !"
        if indFin is None:
            indFin = self.get_SubBFPolygon_indin(Poly, NLim=NLim, Out=int)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]
        IndMin = np.unique(self._Func_Centsind[:,indFin].flatten())
        Id, IdM = self.Id, self.Mesh.Id
        Id._Name, IdM._Name = Id._Name+'_SubBF', Id._Name+'_SubMesh'
        M = Mesh2D(IdM, self.Mesh, IndMin)
        return BF2D(Id, M, self.Deg, self.Id.dtime)

    def get_TotFunc_FixPoints(self, Points, Deriv=0, Test=True):
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'] !"
        if Deriv in [0,'D0']:
            AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
            def FF(Coefs,AA=AA):
                return AA.dot(Coefs)
        elif Deriv in [1,2,'D1','D2']:
            AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
            def FF(Coefs, DVect,AA=AA):
                dvR = np.hypot(DVect[0,:],DVect[1,:])
                return dvR*AA[0].dot(Coefs) + DVect[2,:]*AA[1].dot(Coefs)
        elif Deriv == 'D0N2':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
                def FF(Coefs,AA=AA):
                    return Coefs.dot(AA.dot(Coefs))
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=0, Test=Test)
                def FF(Coefs,AA=AA):
                    return AA.dot(Coefs)*AA.dot(Coefs)
        elif Deriv=='D2-Gauss':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv='D2N2', Test=Test)
                CC, n = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                AA, BB, CC = (AA[0],AA[1]), BB[2], CC[0]+CC[1]
                def FF(Coefs, AA=AA, BB=BB, CC=CC):
                    return (AA[0].dot(Coefs) * AA[1].dot(Coefs) - Coefs.dot(BB.dot(Coefs))) / (1. + Coefs.dot(CC.dot(Coefs)))**2
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                def FF(Coefs, AA=AA, BB=BB):
                    return (AA[0].dot(Coefs) * AA[1].dot(Coefs) - (AA[2].dot(Coefs))**2) / (1. + (BB[0].dot(Coefs))**2+(BB[1].dot(Coefs))**2)**2
        elif Deriv=='D2-Mean':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                CC, p = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                CC = (CC[0],CC[1],CC[2])
                def FF(Coefs,AA=AA,BB=BB,CC=CC):
                    return ( (1.+Coefs.dot(BB[0].dot(Coefs)))*CC[1].dot(Coefs) - 2.*AA[0].dot(Coefs)*AA[1].dot(Coefs)*CC[2].dot(Coefs) + (1.+Coefs.dot(BB[1].dot(Coefs)))*CC[0].dot(Coefs) ) / (2.*(1. + Coefs.dot((BB[0]+BB[1]).dot(Coefs)))**1.5)
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                def FF(Coefs,AA=AA,BB=BB):
                    return ( (1.+(BB[0].dot(Coefs))**2)*AA[1].dot(Coefs) - 2.*BB[0].dot(Coefs)*BB[1].dot(Coefs)*AA[2].dot(Coefs) + (1.+(BB[1].dot(Coefs))**2)*AA[0].dot(Coefs) ) / (2.*(1. + BB[0].dot(Coefs)**2+(BB[1].dot(Coefs))**2)**1.5)
        else:
            if 'D1' in Deriv:
                try:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                    AA = AA[0]+AA[1]
                    if Deriv=='D1N2':
                        def FF(Coefs,AA=AA):
                            return Coefs.dot(AA.dot(Coefs))
                    elif Deriv=='D1FI':
                        B, n = BF2D_get_Op(self, Points, Deriv='D0', Test=Test)
                        def FF(Coefs,AA=AA,B=B):
                            return Coefs.dot(AA.dot(Coefs))/B.dot(Coefs)
                except MemoryError:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D1', Test=Test)
                    if Deriv=='D1N2':
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2
                    elif Deriv=='D1FI':
                        B, n = BF2D_get_Op(self, Points, Deriv='D0', Test=Test)
                        def FF(Coefs,AA=AA,B=B):
                            return ((AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2)/B.dot(Coefs)
            elif 'D2' in Deriv:
                try:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D2N2', Test=Test)
                    if Deriv=='D2N2-Lapl':
                        AA, B = AA[0]+AA[1], AA[3]
                        def FF(Coefs,AA=AA,B=B):
                            return Coefs.dot(AA.dot(Coefs)) + 2.*B.dot(Coefs)
                    elif Deriv=='D2N2-Vect':
                        AA = AA[0]+AA[1]
                        def FF(Coefs,AA=AA):
                            return Coefs.dot(AA.dot(Coefs))
                except MemoryError:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D2', Test=Test)
                    AA = (AA[0],AA[1])
                    if Deriv=='D2N2-Lapl':
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2 + 2.*AA[0].dot(Coefs)*AA[1].dot(Coefs)
                    elif Deriv=='D2N2-Vect':
                        AA = (AA[0],AA[1])
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2
        return FF

    def get_TotFunc_FixCoefs(self, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
        assert type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim==1 and Coefs.size==self._NFunc), "Arg Coefs must be a float or a (NF,) np.ndarray !"
        return BF2D_get_TotFunc(self, Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)

    def get_TotVal(self, Points, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'] !"
        if type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim==1):
            Coefs = Coefs*np.ones((self._NFunc,))
        if Points.shape==2:
            Points = np.array([Points[0,:],np.zeros((Points.shape[1],)), Points[1,:]])
        FF = self.get_TotFunc_FixPoints(Points, Deriv=Deriv, Test=Test)
        if Coefs.ndim==1:
            if Deriv in [1,2,'D1','D2']:
                dvect = DVect(Points)
                return FF(Coefs,dvect)
            else:
                return FF(Coefs)
        else:
            Nt = Coefs.shape[0]
            Vals = np.empty((Nt,Points.shape[1]))
            if Deriv in [1,2,'D1','D2']:
                dvect = DVect(Points)
                for ii in range(0,Nt):
                    Vals[ii,:] = FF(Coefs[ii,:], dvect)
            else:
                for ii in range(0,Nt):
                    Vals[ii,:] = FF(Coefs[ii,:])
            return Vals

    def get_Coefs(self,xx=None,yy=None, zz=None, ff=None, SubP=tfd.BF2Sub, SubMode=tfd.BF1SubMode, indFin=None, Test=True):         # To be updated to take into account fraction of BF
        assert not all([xx is None or yy is None, ff is None]), "You must provide either a function (ff) or sampled data (xx and yy) !"
        assert indFin is None or (isnstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"
        print self.Id.Name+" : Getting fit coefficients..."
        if indFin is None:
            indFin = np.arange(0,self.NFunc)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]

        if not ff is None:
            xx, yy, nx, ny = self.get_XYplot(SubP=SubP, SubMode=SubMode)        # To update
            xx, yy = xx.flatten(), yy.flatten()
            zz = ff(RZ2Points(np.array([xx,yy])))
        else:
            assert xx.shape==yy.shape==zz.shape, "Args xx, yy and zz must have same shape !"
            xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        # Keep only points inside the Boundary
        ind = self.Mesh.isInside(np.array([xx,yy]))
        xx, yy, zz = xx[ind], yy[ind], zz[ind]

        AA, m = BF2D_get_Op(self, np.array([xx.flatten(),yy.flatten()]), Deriv=0, indFin=indFin, Test=Test)
        Coefs, res, rank, sing = np.linalg.lstsq(AA,zz)
        if rank < indFin.size:
            xx1, yy1, nx1, ny1 = self.get_XYplot(SubP=SubP/2., SubMode=SubMode)
            xx1, yy1 = xx1.flatten(), yy1.flatten()
            ind = self._Mesh.isInside(np.array([xx1,yy1]))
            xx1, yy1 = xx1[ind], yy1[ind]
            zz1 = scpint.interp2d(xx, yy, zzz, kind='linear', bounds_error=False, fill_value=0)(xx1,yy1)
            xx, yy, zz = np.concatenate((xx,xx1)), np.concatenate((yy,yy1)), np.concatenate((zz,zz1))
            AA, m = BF2D_get_Op(self, np.array([xx.flatten(),yy.flatten()]), Deriv=0, Test=Test)
            Coefs, res, rank, sing = np.linalg.lstsq(AA,zz)
        return Coefs, res

    def get_XYplot(self, SubP=tfd.BF2PlotSubP, SubMode=tfd.BF2PlotSubMode):
        Xplot, Yplot = Calc_SumMeshGrid2D(self.Mesh._MeshR.Knots, self.Mesh._MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
        nx, ny = Xplot.size, Yplot.size
        Xplot, Yplot = np.dot(np.ones((ny,1)),Xplot.reshape((1,nx))), np.dot(Yplot.reshape((ny,1)),np.ones((1,nx)))
        return Xplot, Yplot, nx, ny

    def get_IntOp(self, Deriv=0, Mode=tfd.BF2IntMode, Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
        print self.Id.Name+" : Getting integral operator "+str(Deriv)
        return Calc_IntOp_BSpline2D(self, Deriv=Deriv, Mode=Mode, Sparse=Sparse, SpaFormat=SpaFormat, Test=Test)

    def get_IntVal(self, Deriv=0, Mode=tfd.BF2IntMode, Coefs=1., Test=True):
        A, m = Calc_IntOp_BSpline2D(self, Deriv=Deriv, Mode=Mode, Sparse=True, Test=Test)
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,),dtype=float)
        if Coefs.ndim==1:
            if m==0:
                Int = A.dot(Coefs)
            elif m==1:
                Int = Coefs.dot(A.dot(Coefs))
            elif m==2:
                A = A[0]+A[1]
                Int = Coefs.dot(A.dot(Coefs))
            else:
                print 'Not coded yet !'
        else:
            Int = np.nan*np.ones((Coefs.shape[0],))
            if m==0:
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] = A.dot(Coefs[ii,:])
            elif m==1:
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] = Coefs[ii,:].dot(A.dot(Coefs[ii,:]))
            elif m==2:
                A = A[0]+A[1]
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] =Coefs[ii,:].dot(A.dot(Coefs[ii,:]))
            else:
                print 'Not coded yet !'
        return Int

    def get_MinMax(self, Coefs=1., Ratio=0.05, SubP=0.004, SubP1=0.015, SubP2=0.001, TwoSteps=False, SubMode='abs', Deriv='D0', Test=True):                                  # To be deprecated by get_Extrema() when implemented
        return Get_MinMax(self, Coefs=Coefs, Ratio=Ratio, SubP=SubP, SubP1=SubP1, SubP2=SubP2, TwoSteps=TwoSteps, SubMode=SubMode, Deriv=Deriv, Test=Test)

    def get_Extrema(self, Coefs=1., Ratio=0.95, SubP=0.002, SubMode='abs', Deriv='D0', D1N2=True, D2N2=True):
        return Get_Extrema(self, Coefs=Coefs, Ratio=Ratio, SubP=SubP, SubMode=SubMode, D1N2=D1N2, D2N2=D2N2)


    def plot(self, ax='None', Coefs=1., Deriv=0, Elt='T', NC=tfd.BF2PlotNC, DVect=tfd.BF2_DVect_DefR, PlotMode='contourf', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        return Plot_BSpline2D(self, ax=ax, Elt=Elt, Deriv=Deriv, Coefs=Coefs, DVect=DVect, NC=NC, PlotMode=PlotMode, Name=self.Id.Name+' '+str(Deriv), SubP=SubP, SubMode=SubMode, Totdict=Totdict, LegDict=LegDict, Test=Test)

    def plot_fit(self, ax1='None', ax2='None', xx=None, yy=None, zz=None, ff=None, NC=tfd.BF2PlotNC, PlotMode='contourf', Name='', SubP=tfd.BF2Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        Coefs, res = self.get_Coefs(xx=xx,yy=yy, zz=zz, ff=ff, SubP=SubP, SubMode=SubMode, Test=Test)
        if xx is None:
            xx, yy, nxg, nyg = self.get_XYplot(SubP=SubP, SubMode=SubMode)
            zz = ff(RZ2Points(np.array([xx.flatten(),yy.flatten()])))
            zz = zz.reshape((xx.shape))
        else:
            assert xx.ndim==2, "Args xx, yy and zz must be (N,M) plot-friendly !"
        if Name=='':
            Name = self.Id.Name

        Xplot, Yplot, nx, ny = self.get_XYplot(SubP=SubP, SubMode=SubMode)
        PointsRZ = np.array([Xplot.flatten(),Yplot.flatten()])
        ind = self.Mesh.isInside(PointsRZ)
        Val = self.get_TotVal(PointsRZ, Deriv=0, Coefs=Coefs, Test=True)
        Val[~ind] = np.nan
        Val = Val.reshape(Xplot.shape)
        if PlotMode=='surf':
            if ax1=='None' or ax2=='None':
                ax1, ax2 = tfd.Plot_BSplineFit_DefAxes('3D')
            ax1.plot_surface(xx,yy,zz, label='Model', **Totdict)
            ax2.plot_surface(Xplot,Yplot,Val, label=Name, **Totdict)
        else:
            if ax1=='None' or ax2=='None':
                ax1, ax2  = tfd.Plot_BSplineFit_DefAxes('2D')
            if PlotMode=='contour':
                ax1.contour(xx,yy,zz, NC, label='Model', **Totdict)
                ax2.contour(Xplot,Yplot,Val, NC, label=Name, **Totdict)
            elif PlotMode=='contourf':
                ax1.contourf(xx,yy,zz, NC, label='Model', **Totdict)
                ax2.contourf(Xplot,Yplot,Val, NC, label=Name, **Totdict)
        if not LegDict is None:
            ax1.legend(**LegDict)
            ax2.legend(**LegDict)
        ax2.set_title(r"$\chi^2 = "+str(res)+"$")
        ax1.figure.canvas.draw()
        return ax1, ax2

    def plot_Ind(self, Ind=0, Elt='LSP', EltM='BMCK', ax='None', Coefs=1., NC=tfd.BF2PlotNC, PlotMode='contourf', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF2PlotTotd,
            Cdict=tfd.M2Cd, Kdict=tfd.M2Kd, Bckdict=tfd.M2Bckd, Mshdict=tfd.M2Mshd, LegDict=tfd.TorLegd, Colorbar=True, Test=True):
        assert type(Ind) in [int,list,np.ndarray], "Arg Ind must be a int, a list of int or a np.ndarray of int or booleans !"
        if type(Ind) is int:
            Ind = np.array([Ind],dtype=int)
        elif type(Ind) is list:
            Ind = np.array(tuple(Ind),dtype=int)
        elif type(Ind) is np.ndarray:
            assert (np.issubdtype(Ind.dtype,bool) and Ind.size==self._NFunc) or np.issubdtype(Ind.dtype,int), "Arg Ind must be a np.ndarray of boolenas with size==self.NFunc or a np.ndarray of int !"
            if np.issubdtype(Ind.dtype,bool):
                Ind = Ind.nonzero()[0]
        NInd = Ind.size
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self._NFunc,))
        indC, indK = self._Func_Centsind[:,Ind].flatten().astype(int), self._Func_Knotsind[:,Ind].flatten().astype(int)
        ax = self._Mesh.plot(indKnots=indK, indCents=indC, ax=ax, Elt=EltM, Cdict=Cdict, Kdict=Kdict, Bckdict=Bckdict, Mshdict=Mshdict, LegDict=LegDict, Test=Test)
        ax = Plot_BSpline2D_Ind(self, Ind, ax=ax, Coefs=Coefs, Elt=Elt, NC=NC, PlotMode=PlotMode, Name=self._Id.Name, SubP=SubP, SubMode=SubMode, Totdict=Totdict, LegDict=LegDict, Colorbar=Colorbar, Test=Test)
        return ax

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





############################################
#####     Computing functions
############################################


def RZ2Points(PointsRZ, Theta=0.):
        return np.array([PointsRZ[0,:]*np.cos(Theta), PointsRZ[0,:]*np.sin(Theta),PointsRZ[1,:]])

def Points2RZ(Points):
    return np.array([np.sqrt(Points[0,:]**2+Points[1,:]**2),Points[2,:]])


def Calc_BF1D_Weights(LFunc, Points, Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(Points,np.ndarray) and Points.ndim==1, "Arg Points must be a (N,) np.ndarray !"
    NFunc = len(LFunc)
    Wgh = np.zeros((Points.size, NFunc))
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](Points)
    return Wgh

def Calc_BF2D_Weights(LFunc, PointsRZ, Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(PointsRZ,np.ndarray) and PointsRZ.ndim==2, "Arg Points must be a (2,N) np.ndarray !"
    NFunc = len(LFunc)
    Wgh = np.zeros((PointsRZ.shape[1], NFunc))
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](PointsRZ)
    return Wgh


def BF2D_get_Op(BF2, Points, Deriv=0, indFin=None, Test=True):            # To be updated to take into account only fraction of BF
    """ Return the operator to compute the desired quantity on NP points (to be multiplied by Coefs) Y = Op(Coefs) for NF basis functions
    Input :
        BF2         A BF2D instance
        Points      A (3,NP) np.ndarray indicating (X,Y,Z) cylindrical coordinates of points at which to evaluate desired quantity (automatically converted to (R,Z) coordinates)
        Deriv       A flag indicating the desired quantity in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Lapl','D2N2-Lapl']
        Test        A bool Flag indicating whether inputs shall be tested for conformity
    Output :
        A           The operator itself as a 2D or 3D numpy array
        m           Flag indicating the kind of operation necessary
            = 0         Y = A.dot(C)                            (Y = AC, with A.shape=(NP,NF))
            = 1         Y = C.dot(A.dot(C))                     (Y = tCAC, with A.shape=(NF,NP,NF))
            = 2         Y = sum( C.dot(A[:].dot(C)) )           (e.g.: Y = tCArC + tCAzC with Ar.shape==Az.shape=(NF,NP,NF), you can compute a scalar product with each component if necessary)
    """
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert isinstance(Points,np.ndarray) and Points.ndim==2 and Points.shape[0] in [2,3], "Arg Points must be a (2-3,NP) np.ndarray !"
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D2','D2N2'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D2','D2N2'] !"
        assert indFin is None or (isinstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"

    if indFin is None:
        indFin = np.arange(0,BF2.NFunc)
    if indFin.dtype.name=='bool':
        indFin = indFin.nonzero()[0]

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])
    NF = indFin.size
    NP = Points.shape[1]
    if Points.shape[0]==3:
        RZ = np.array([np.hypot(Points[0,:],Points[1,:]),Points[2,:]])
    else:
        RZ = np.copy(Points)

    QuadR, QuadZ = BF2._get_quadPoints()
    QuadR, QuadZ = QuadR[:,indFin], QuadZ[:,indFin]
    if Deriv=='D0':
        m = 0
        A = np.zeros((NP,NF))
        for ii in range(0,NF):
            A[:,ii] = BF2._LFunc[indFin[ii]](RZ)
        return A, m
    elif Deriv=='D0N2':                 # Update in progress from here...
        m = 1
        A = np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            Yii = BF2._LFunc[ii](RZ)
            A[ii,:,ii] = Yii**2
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                A[ii,:,Ind[jj]] = Yii*BF2._LFunc[Ind[jj]](RZ)
        return A, m
    elif Deriv=='D1' and BF2.Deg>=1:
        m = 2
        Ar, Az = np.zeros((NP,NF)), np.zeros((NP,NF))
        for ii in range(0,NF):
            Ar[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Az[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
        return (Ar,Az), m
    elif Deriv=='D1N2' and BF2.Deg>=1:
        m = 2
        AR, AZ = np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            rii, zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Rii, Zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
            AR[ii,:,ii] = (Rii*zii)**2
            AZ[ii,:,ii] = (rii*Zii)**2
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                AR[ii,:,Ind[jj]] = Rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=1)[0][0](RZ[0,:]) * zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=0)[0][0](RZ[1,:])
                AZ[ii,:,Ind[jj]] = rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=0)[0][0](RZ[0,:]) * Zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=1)[0][0](RZ[1,:])
        return (AR,AZ), m
    elif Deriv=='D2' and BF2.Deg>=2:
        m = 2
        Arr, Azz, Arz = np.zeros((NP,NF)), np.zeros((NP,NF)), np.zeros((NP,NF))
        for ii in range(0,NF):
            Arr[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=2)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Azz[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=2)[0][0](RZ[1,:])
            Arz[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
        return (Arr,Azz,Arz), m
    elif Deriv=='D2N2' and BF2.Deg>=2:
        m = 2
        ARR, AZZ, ARZ, Arrzz = np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            rii, zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Rii, Zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
            RRii, ZZii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=2)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=2)[0][0](RZ[1,:])
            ARR[ii,:,ii] = (RRii*zii)**2
            AZZ[ii,:,ii] = (rii*ZZii)**2
            ARZ[ii,:,ii] = (Rii*Zii)**2
            Arrzz[ii,:,ii] = RRii*ZZii
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                ARR[ii,:,Ind[jj]] = RRii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=2)[0][0](RZ[0,:]) * zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=0)[0][0](RZ[1,:])
                AZZ[ii,:,Ind[jj]] = rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=0)[0][0](RZ[0,:]) * ZZii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=2)[0][0](RZ[1,:])
                ARZ[ii,:,Ind[jj]] = Rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=1)[0][0](RZ[0,:]) * Zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=1)[0][0](RZ[1,:])
        return (ARR,AZZ,ARZ, Arrzz), m


def BF2D_get_TotFunc(BF2, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert type(Deriv) in [int,str], "Arg Deriv must be a int or a str !"
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==BF2.NFunc), "Arg Coefs must be a float or a np.ndarray !"

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])

    if type(Coefs) is float:
        Coefs = Coefs.np.ones((BF2.NFunc,))

    NF = BF2.NFunc
    QuadR, QuadZ = BF2._get_quadPoints()
    if Deriv in [0,'D0']:
        LF = BF2._LFunc
        return lambda Points, Coefs=Coefs, LF=LF, NF=NF: np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
    elif Deriv=='D0N2':
        LF = BF2._LFunc
        return lambda Points, Coefs=Coefs, LF=LF, NF=NF: np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)**2

    elif Dbis==1:
        LDR = []
        LDZ = []
        for ii in range(0,NF):
            LDR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
            LDZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
        if Deriv=='D1':
            def FTot(Points, Coefs=Coefs, DVect=DVect, NF=NF, LDR=LDR, LDZ=LDZ):
                DVect = DVect(Points)
                Theta = np.arctan2(Points[1,:],Points[0,:])
                eR = np.array([np.cos(Theta),np.sin(Theta),np.zeros((Theta.size,))])
                DVect = np.array([np.sum(DVect*eR,axis=0), DVect[2,:]])
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)*DVect[0,:]
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)*DVect[1,:]
                return ValR+ValZ
        elif Deriv=='D1N2':
            def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return ValR**2+ValZ**2
        elif Deriv=='D1FI':
            LF = BF2._LFunc
            def FTot(Points, Coefs=Coefs, NF=NF, LF=LF, LDR=LDR, LDZ=LDZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                Val = np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return (ValR**2+ValZ**2)/Val

    elif Dbis==2:
        LDRR, LDZZ = [], []
        for ii in range(0,NF):
            LDRR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=2, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
            LDZZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=2, Test=Test)[0](PointsRZ[1,:]))
        if Deriv=='D2-Lapl':
            def FTot(Points, Coefs=Coefs, NF=NF, LDRR=LDRR, LDZZ=LDZZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return ValR+ValZ
        elif Deriv=='D2N2-Lapl':
            def FTot(Points, Coefs=Coefs, NF=NF, LDRR=LDRR, LDZZ=LDZZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return (ValR+ValZ)**2
        else:
            LDR, LDZ, LDRZ= [], [], []
            for ii in range(0,NF):
                LDR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
                LDZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
                LDRZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
            if Deriv=='D2-Gauss':
                def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ, LDRR=LDRR, LDZZ=LDZZ, LDRZ=LDRZ):
                    ValRR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValRZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDRZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    return (ValRR*ValZZ-ValRZ**2)/(1. + ValR**2 + ValZ**2)**2
            elif Deriv=='D2-Mean':
                def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ, LDRR=LDRR, LDZZ=LDZZ, LDRZ=LDRZ):
                    ValRR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValRZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDRZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    return ((1.+ValR**2)*ValZZ - 2.*ValR*ValZ*ValRZ + (1.+ValZ**2)*ValRR)/(2.*(1. + ValR**2 + ValZ**2)**(1.5))

    return FTot


def Calc_BF2D_Val(LFunc, Points, Coef=1., Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(Points,np.ndarray) and (Points.shape[0]==2 or Points.shape[0]==3), "Arg Points must be a (2,N) or (3,N) np.ndarray !"
        assert (type(Coef) is float and Coef==1.) or (isinstance(Coef,np.ndarray) and Coef.shape[0]==len(LFunc)), "Arg Coef must be a (BF2.NFunc,1) np.ndarray !"

    if Points.shape[0]==3:
        R = np.sqrt(np.sum(Points[0:2,:]**2,axis=0,keepdims=False))
        Points = np.array([[R],[Points[2,:]]])

    NFunc = len(LFunc)
    Val = np.zeros((Points.shape[1],))
    Coef = Coef*np.ones((NFunc,1))
    for ii in range(0,NFunc):
        Val = Val + Coef[ii]*LFunc[ii](Points)

    return Val









def Calc_Integ_BF2(BF2, Coefs=1., Mode='Vol', Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert Mode=='Vol' or Mode=='Surf', "Arg Mode must be 'Vol' or 'Surf' !"
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==BF2.NFunc), "Arg Coefs must be a (BF2.NFunc,) np.ndarray !"
        assert BF2.Deg <= 3, "Arg BF2 should not have Degree > 3 !"

    if type(Coefs) is float:
        Coefs = Coefs*np.ones((BF2.NFunc,))
    if Mode=='Surf':
        if BF2.Deg==0:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum(Coefs.flatten()*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==1:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum(Coefs.flatten()*0.25*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()
            IntR1 = (QR[1,:]-QR[0,:])**2/(3.*(QR[2,:]-QR[0,:]))
            IntR21 = (QR[2,:]**2 -2.*QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[0,:]*(QR[1,:]-QR[2,:]))/(6*(QR[2,:]-QR[0,:]))
            IntR22 = (-2.*QR[2,:]**2+QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[3,:]*(QR[2,:]-QR[1,:]))/(6.*(QR[3,:]-QR[1,:]))
            IntR3 = (QR[3,:]-QR[2,:])**2/(3.*(QR[3,:]-QR[1,:]))
            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            Int = np.sum(Coefs.flatten() * (IntR1+IntR21+IntR22+IntR3) * (IntZ1+IntZ21+IntZ22+IntZ3))
        elif BF2.Deg==3:
            print "NOT CODED YET !"

    else:
        if BF2.Deg==0:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum( Coefs.flatten() * 2.*np.pi * 0.5*(Supp[1,:]**2-Supp[0,:]**2)*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==1:
            Supp = BF2.get_Func_SuppRZ()
            QuadR, QuadZ = BF2._get_quadPoints()
            Int = np.sum( Coefs.flatten() * 2.*np.pi * 0.5*(QuadR[2,:]**2-QuadR[0,:]**2 + QuadR[1,:]*(QuadR[2,:]-QuadR[0,:]))*(Supp[3,:]-Supp[2,:])/6.)
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()
            IntR1 = (3.*QR[1,:]**3+QR[0,:]**3 -5.*QR[0,:]*QR[1,:]**2+QR[0,:]**2*QR[1,:])/(12.*(QR[2,:]-QR[0,:]))
            IntR21 = (QR[2,:]**3 -3.*QR[1,:]**3+QR[1,:]**2*QR[2,:]+QR[1,:]*QR[2,:]**2 -2.*QR[0,:]*QR[2,:]**2 -2.*QR[0,:]*QR[1,:]*QR[2,:] +4.*QR[0,:]*QR[1,:]**2)/(12.*(QR[2,:]-QR[0,:]))
            IntR22 = ( -3.*QR[2,:]**3+QR[1,:]**3+QR[1,:]*QR[2,:]**2+QR[1,:]**2*QR[2,:]+4.*QR[2,:]**2*QR[3,:]-2.*QR[1,:]*QR[2,:]*QR[3,:]-2.*QR[1,:]**2*QR[3,:] )/(12.*(QR[3,:]-QR[1,:]))
            IntR3 = ( QR[3,:]**3 +3.*QR[2,:]**3 -5.*QR[2,:]**2*QR[3,:]+QR[2,:]*QR[3,:]**2)/(12.*(QR[3,:]-QR[1,:]))
            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            Int = np.sum(Coefs.flatten() * 2.*np.pi * (IntR1+IntR21+IntR22+IntR3) * (IntZ1+IntZ21+IntZ22+IntZ3))
        elif BF2.Deg==3:
            print "NOT CODED YET !"

    return Int







def Calc_IntOp_BSpline2D(BF2, Deriv=0, Mode='Vol', Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert Mode in ['Surf','Vol'], "Arg Mode must be in ['Surf','Vol'] !"
        assert (type(Deriv) is int and Deriv<=BF2.Deg) or (type(Deriv) is str and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2','D1FI'] and int(Deriv[1])<=BF2.Deg), "Arg Deriv must be a int or a str !"

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])

    if Deriv=='D0':
        m = 0
        if BF2.Deg==0:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = (Supp[1,:]-Supp[0,:]) * (Supp[3,:]-Supp[2,:])
            else:
                A = 0.5*(Supp[1,:]**2-Supp[0,:]**2) * (Supp[3,:]-Supp[2,:])
        elif BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = 0.25*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:])
            else:
                QuadR, QuadZ = BF2._get_quadPoints()
                A = 0.5*(QuadR[2,:]**2-QuadR[0,:]**2 + QuadR[1,:]*(QuadR[2,:]-QuadR[0,:])) * (Supp[3,:]-Supp[2,:])/6.
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()

            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            IntZ = IntZ1+IntZ21+IntZ22+IntZ3

            if Mode=='Surf':
                IntR1 = (QR[1,:]-QR[0,:])**2/(3.*(QR[2,:]-QR[0,:]))
                IntR21 = (QR[2,:]**2 -2.*QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[0,:]*(QR[1,:]-QR[2,:]))/(6*(QR[2,:]-QR[0,:]))
                IntR22 = (-2.*QR[2,:]**2+QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[3,:]*(QR[2,:]-QR[1,:]))/(6.*(QR[3,:]-QR[1,:]))
                IntR3 = (QR[3,:]-QR[2,:])**2/(3.*(QR[3,:]-QR[1,:]))
            else:
                IntR1 = (3.*QR[1,:]**3+QR[0,:]**3 -5.*QR[0,:]*QR[1,:]**2+QR[0,:]**2*QR[1,:])/(12.*(QR[2,:]-QR[0,:]))
                IntR21 = (QR[2,:]**3 -3.*QR[1,:]**3+QR[1,:]**2*QR[2,:]+QR[1,:]*QR[2,:]**2 -2.*QR[0,:]*QR[2,:]**2 -2.*QR[0,:]*QR[1,:]*QR[2,:] +4.*QR[0,:]*QR[1,:]**2)/(12.*(QR[2,:]-QR[0,:]))
                IntR22 = ( -3.*QR[2,:]**3+QR[1,:]**3+QR[1,:]*QR[2,:]**2+QR[1,:]**2*QR[2,:]+4.*QR[2,:]**2*QR[3,:]-2.*QR[1,:]*QR[2,:]*QR[3,:]-2.*QR[1,:]**2*QR[3,:] )/(12.*(QR[3,:]-QR[1,:]))
                IntR3 = ( QR[3,:]**3 +3.*QR[2,:]**3 -5.*QR[2,:]**2*QR[3,:]+QR[2,:]*QR[3,:]**2)/(12.*(QR[3,:]-QR[1,:]))
            IntR = IntR1+IntR21+IntR22+IntR3
            A = IntR*IntZ
        elif BF2.Deg==3:
            print "NOT CODED YET !"
            A = 0

    elif Deriv=='D0N2':
        m = 1
        if BF2.Deg==0:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = scpsp.diags([(Supp[1,:]-Supp[0,:]) * (Supp[3,:]-Supp[2,:])],[0],shape=None,format=SpaFormat)
            else:
                A = scpsp.diags([0.5*(Supp[1,:]**2-Supp[0,:]**2) * (Supp[3,:]-Supp[2,:])],[0],shape=None,format=SpaFormat)
        elif BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            if Mode=='Surf':
                d0R, d0Z = (Supp[1,:]-Supp[0,:])/3., (Supp[3,:]-Supp[2,:])/3.
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intR = (kR[1]-kR[0])/6.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZ = (kZ[1]-kZ[0])/6.
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
            else:
                d0R, d0Z = (QR[2,:]**2-QR[0,:]**2 + 2.*QR[1,:]*(QR[2,:]-QR[0,:]))/12., (Supp[3,:]-Supp[2,:])/3.
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intR = (kR[1]**2-kR[0]**2)/12.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZ = (kZ[1]-kZ[0])/6.
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
            A = scpsp.vstack(LL,format='csr')
        elif BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intR = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intR = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZ = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZ = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZ = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZ = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
                A = scpsp.vstack(LL,format='csr')

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intR = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR = intR1A + intR1B + intR2A + intR2B

                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intR = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intR = intR1A + intR1B + intR2A + intR2B
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZ = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZ = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZ = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZ = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
                A = scpsp.vstack(LL,format='csr')

    elif Deriv=='D1N2':
        m = 2
        if BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []

            if Mode=='Surf':
                d0R, d0Z = (Supp[1,:]-Supp[0,:])/3., (Supp[3,:]-Supp[2,:])/3.
                d0DR, d0DZ = (QR[2,:]-QR[0,:])/((QR[2,:]-QR[1,:])*(QR[1,:]-QR[0,:])), (QZ[2,:]-QZ[0,:])/((QZ[2,:]-QZ[1,:])*(QZ[1,:]-QZ[0,:]))
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDR = d0DR[ii]
                            intRDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intRDR = -1./(kR[1]-kR[0])
                            intRDZ = (kR[1]-kR[0])/6.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZDR = (kZ[1]-kZ[0])/6.
                            intZDZ = -1./(kZ[1]-kZ[0])
                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))
            else:
                d0R, d0Z = (QR[2,:]**2-QR[0,:]**2 + 2.*QR[1,:]*(QR[2,:]-QR[0,:]))/12., (Supp[3,:]-Supp[2,:])/3.
                d0DR, d0DZ = 0.5*(QR[1,:]+QR[0,:])/(QR[1,:]-QR[0,:]) + 0.5*(QR[2,:]+QR[1,:])/(QR[2,:]-QR[1,:]), (QZ[2,:]-QZ[0,:])/((QZ[2,:]-QZ[1,:])*(QZ[1,:]-QZ[0,:]))
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDR = d0DR[ii]
                            intRDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intRDR = -0.5*(kR[1]+kR[0])/(kR[1]-kR[0])
                            intRDZ = (kR[1]**2-kR[0]**2)/12.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZDR = (kZ[1]-kZ[0])/6.
                            intZDZ = -1./(kZ[1]-kZ[0])
                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))
            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)

        elif BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []
            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DR = 4.*(QR[1,:]-QR[0,:])/(3.*(QR[2,:]-QR[0,:])**2) + 4.*( QR[0,:]**2 + QR[1,:]**2 + QR[2,:]**2 + (QR[2,:]-2.*QR[3,:])*QR[1,:] - QR[2,:]*QR[3,:] + QR[3,:]**2 + (-QR[1,:]-2.*QR[2,:]+QR[3,:])*QR[0,:] )*(QR[2,:]-QR[1,:])/(3.*(QR[2,:]-QR[0,:])**2*(QR[3,:]-QR[1,:])**2) + 4.*(QR[3,:]-QR[2,:])/(3.*(QR[3,:]-QR[1,:])**2)
                d0DZ = 4.*(QZ[1,:]-QZ[0,:])/(3.*(QZ[2,:]-QZ[0,:])**2) + 4.*( QZ[0,:]**2 + QZ[1,:]**2 + QZ[2,:]**2 + (QZ[2,:]-2.*QZ[3,:])*QZ[1,:] - QZ[2,:]*QZ[3,:] + QZ[3,:]**2 + (-QZ[1,:]-2.*QZ[2,:]+QZ[3,:])*QZ[0,:] )*(QZ[2,:]-QZ[1,:])/(3.*(QZ[2,:]-QZ[0,:])**2*(QZ[3,:]-QZ[1,:])**2) + 4.*(QZ[3,:]-QZ[2,:])/(3.*(QZ[3,:]-QZ[1,:])**2)

                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDZ = d0R[ii]
                            intRDR = d0DR[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDZ = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDR = 2.*(QR[0,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intRDZ = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                                intRDR = 2.*(2.*QR[0,Ind[jj]]-QR[0,ii]-2.*QR[1,ii]+QR[2,ii])*(QR[1,ii]-QR[0,ii])/(3.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) + 2.*(QR[0,ii]-2.*QR[1,ii]-QR[2,ii]+2.*QR[3,ii])*(QR[2,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDZ = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                                intRDR = 2.*(QR[2,ii]-QR[3,ii])/(3.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intRDZ = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                                intRDR = 2.*(2.*QR[0,ii]-QR[1,ii]-2.*QR[2,ii]+QR[3,ii])*(QR[2,ii]-QR[1,ii])/(3.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) + 2.*(QR[1,ii]-2.*QR[2,ii]-QR[3,ii]+2.*QR[3,Ind[jj]])*(QR[3,ii]-QR[2,ii])/(3.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDZ = 2.*(QZ[0,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[1,Ind[jj]])*(QZ[2,ii]-QZ[0,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,Ind[jj]]-QZ[0,ii]-2.*QZ[1,ii]+QZ[2,ii])*(QZ[1,ii]-QZ[0,ii])/(3.*(QZ[1,ii]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) + 2.*(QZ[0,ii]-2.*QZ[1,ii]-QZ[2,ii]+2.*QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDZ = 2.*(QZ[2,ii]-QZ[3,ii])/(3.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[2,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,ii]-QZ[1,ii]-2.*QZ[2,ii]+QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2) + 2.*(QZ[1,ii]-2.*QZ[2,ii]-QZ[3,ii]+2.*QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[2,ii])/(3.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."

                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DR = (QR[1,:]-QR[0,:])*(QR[0,:]+3.*QR[1,:])/(3.*(QR[2,:]-QR[0,:])**2) + ( (3.*QR[1,:]+QR[2,:])/(QR[2,:]-QR[0,:])**2 + (QR[1,:]+3.*QR[2,:])/(QR[3,:]-QR[1,:])**2 - 2.*(QR[1,:]+QR[2,:])/((QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:])) )*(QR[2,:]-QR[1,:])/3. + (QR[3,:]-QR[2,:])*(QR[3,:]+3.*QR[2,:])/(3.*(QR[3,:]-QR[1,:])**2)
                d0DZ = 4.*(QZ[1,:]-QZ[0,:])/(3.*(QZ[2,:]-QZ[0,:])**2) + 4.*( QZ[0,:]**2 + QZ[1,:]**2 + QZ[2,:]**2 + (QZ[2,:]-2.*QZ[3,:])*QZ[1,:] - QZ[2,:]*QZ[3,:] + QZ[3,:]**2 + (-QZ[1,:]-2.*QZ[2,:]+QZ[3,:])*QZ[0,:] )*(QZ[2,:]-QZ[1,:])/(3.*(QZ[2,:]-QZ[0,:])**2*(QZ[3,:]-QZ[1,:])**2) + 4.*(QZ[3,:]-QZ[2,:])/(3.*(QZ[3,:]-QZ[1,:])**2)

                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDZ = d0R[ii]
                            intRDR = d0DR[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDZ = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intRDR = (QR[0,ii]+QR[1,ii])*(QR[0,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intRDZ = intR1A + intR1B + intR2A + intR2B
                                intRDR = ( -QR[0,ii]**2 + (QR[0,ii]+3.*QR[1,ii])*QR[0,Ind[jj]] + (-3.*QR[1,ii]+QR[2,ii])*QR[1,ii] +(-2.*QR[1,ii]+QR[2,ii])*QR[0,ii] )*(QR[1,ii]-QR[0,ii])/(3.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) + ( -3.*QR[1,ii]**2 + (QR[1,ii]+QR[2,ii])*QR[0,ii] + (-QR[2,ii]+QR[3,ii])*QR[2,ii] + (-2.*QR[2,ii]+3.*QR[3,ii])*QR[1,ii] )*(QR[2,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDZ = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                                intRDR = (QR[2,ii]+QR[3,ii])*(QR[2,ii]-QR[3,ii])/(3.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intRDZ = intR1A + intR1B + intR2A + intR2B
                                intRDR = ( -QR[1,ii]**2 + (QR[1,ii]+3.*QR[2,ii])*QR[0,ii] + (-3.*QR[2,ii]+QR[3,ii])*QR[2,ii] +(-2.*QR[2,ii]+QR[3,ii])*QR[1,ii] )*(QR[2,ii]-QR[1,ii])/(3.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) + ( -3.*QR[2,ii]**2 + (QR[2,ii]+QR[3,ii])*QR[1,ii] + (-QR[3,ii]+QR[3,Ind[jj]])*QR[3,ii] + (-2.*QR[3,ii]+3.*QR[3,Ind[jj]])*QR[2,ii] )*(QR[3,ii]-QR[2,ii])/(3.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDZ = 2.*(QZ[0,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[1,Ind[jj]])*(QZ[2,ii]-QZ[0,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,Ind[jj]]-QZ[0,ii]-2.*QZ[1,ii]+QZ[2,ii])*(QZ[1,ii]-QZ[0,ii])/(3.*(QZ[1,ii]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) + 2.*(QZ[0,ii]-2.*QZ[1,ii]-QZ[2,ii]+2.*QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDZ = 2.*(QZ[2,ii]-QZ[3,ii])/(3.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[2,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,ii]-QZ[1,ii]-2.*QZ[2,ii]+QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2) + 2.*(QZ[1,ii]-2.*QZ[2,ii]-QZ[3,ii]+2.*QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[2,ii])/(3.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."

                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)


    elif Deriv=='D1FI':
        m = 3
        if BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []




    elif 'D2N2' in Deriv:
        m = 2
        if BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []

            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DDR = 4./((QR[1,:]-QR[0,:])*(QR[2,:]-QR[0,:])**2) + 4.*(QR[3,:]+QR[2,:]-QR[1,:]-QR[0,:])**2/((QR[2,:]-QR[1,:])*(QR[3,:]-QR[1,:])**2*(QR[2,:]-QR[0,:])**2) + 4./((QR[3,:]-QR[2,:])*(QR[3,:]-QR[1,:])**2)
                d0DDZ = 4./((QZ[1,:]-QZ[0,:])*(QZ[2,:]-QZ[0,:])**2) + 4.*(QZ[3,:]+QZ[2,:]-QZ[1,:]-QZ[0,:])**2/((QZ[2,:]-QZ[1,:])*(QZ[3,:]-QZ[1,:])**2*(QZ[2,:]-QZ[0,:])**2) + 4./((QZ[3,:]-QZ[2,:])*(QZ[3,:]-QZ[1,:])**2)
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDDR = d0DDR[ii]
                            intRDDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDDR = 4./((kR[1]-kR[0])*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDDZ = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intRDDR = -4.*(QR[2,ii]+QR[1,ii]-QR[1,Ind[jj]]-QR[0,Ind[jj]])/((QR[2,Ind[jj]]-QR[1,Ind[jj]])*(QR[2,Ind[jj]]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) - 4.*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[3,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])**2)
                                intRDDZ = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDDR = 4./((kR[1]-kR[0])*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                                intRDDZ = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intRDDR = -4.*(QR[3,Ind[jj]]+QR[2,Ind[jj]]-QR[2,ii]-QR[1,ii])/((QR[3,ii]-QR[2,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2) - 4.*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2)
                                intRDDZ = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common knots..."

                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDDR = d0Z[ii]
                            intZDDZ = d0DDZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDDZ = -4.*(QZ[2,ii]+QZ[1,ii]-QZ[1,Ind[jj]]-QZ[0,Ind[jj]])/((QZ[2,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDDZ = -4.*(QZ[3,Ind[jj]]+QZ[2,Ind[jj]]-QZ[2,ii]-QZ[1,ii])/((QZ[3,ii]-QZ[2,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common knots..."

                        llR[0,Ind[jj]] = intRDDR*intZDDR
                        llZ[0,Ind[jj]] = intRDDZ*intZDDZ
                    llR[0,ii] = d0DDR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DDZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DDR = 2.*(QR[1,:]+QR[0,:])/((QR[1,:]-QR[0,:])*(QR[2,:]-QR[0,:])**2) + 2.*(QR[2,:]+QR[1,:])*(QR[3,:]+QR[2,:]-QR[1,:]-QR[0,:])**2/((QR[2,:]-QR[1,:])*(QR[3,:]-QR[1,:])**2*(QR[2,:]-QR[0,:])**2) + 2.*(QR[3,:]+QR[2,:])/((QR[3,:]-QR[2,:])*(QR[3,:]-QR[1,:])**2)
                d0DDZ = 4./((QZ[1,:]-QZ[0,:])*(QZ[2,:]-QZ[0,:])**2) + 4.*(QZ[3,:]+QZ[2,:]-QZ[1,:]-QZ[0,:])**2/((QZ[2,:]-QZ[1,:])*(QZ[3,:]-QZ[1,:])**2*(QZ[2,:]-QZ[0,:])**2) + 4./((QZ[3,:]-QZ[2,:])*(QZ[3,:]-QZ[1,:])**2)
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    Ind = np.asarray([int(xxx) for xxx in Ind], dtype=int)
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDDR = d0DDR[ii]
                            intRDDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDDR = 2.*(kR[1]+kR[0])/((kR[1]-kR[0])*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDDZ = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intRDDZ = intR1A + intR1B + intR2A + intR2B
                                intRDDR = -2.*(QR[1,ii]+QR[0,ii])*(QR[2,ii]+QR[1,ii]-QR[0,ii]-QR[0,Ind[jj]])/((QR[1,ii]-QR[0,ii])*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) - 2.*(QR[2,ii]+QR[1,ii])*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[3,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDDR = 2.*(kR[1]+kR[0])/((kR[1]-kR[0])*(QR[2,Ind[jj]]-QR[0,Ind[jj]])*(QR[3,ii]-QR[1,ii]))
                                intRDDZ = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intRDDZ = intR1A + intR1B + intR2A + intR2B
                                intRDDR = -2.*(QR[2,ii]+QR[1,ii])*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) - 2.*(QR[3,ii]+QR[2,ii])*(QR[3,Ind[jj]]+QR[3,ii]-QR[2,ii]-QR[1,ii])/((QR[3,ii]-QR[2,ii])*(QR[3,Ind[jj]]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common knots..."

                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDDR = d0Z[ii]
                            intZDDZ = d0DDZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDDZ = -4.*(QZ[2,ii]+QZ[1,ii]-QZ[1,Ind[jj]]-QZ[0,Ind[jj]])/((QZ[2,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDDZ = -4.*(QZ[3,Ind[jj]]+QZ[2,Ind[jj]]-QZ[2,ii]-QZ[1,ii])/((QZ[3,ii]-QZ[2,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common knots..."

                        llR[0,Ind[jj]] = intRDDR*intZDDR
                        llZ[0,Ind[jj]] = intRDDZ*intZDDZ
                    llR[0,ii] = d0DDR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DDZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)

        if not Sparse:
            if m in [0,1]:
                A = A.toarray()
            elif m==2:
                A = (A[0].toarray(),A[1].toarray())

    return A, m



















def Calc_BF2D_DerivFunc(BF2, Deriv, Test=True):
    if Test:
        assert isinstance(BF2, BF2D), "Arg BF2 must be a MeshBase2D instance !"
        assert type(Deriv) is int, "Arg Deriv must be a int !"

    KnR, KnZ = BF2._get_quadPoints()
    if Deriv==1:
        LFuncR, LFuncZ = [], []
        for ii in range(0,BF2.NFunc):
            LFuncR.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=0, Test=False)[0](X[1,:])*BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=Deriv, Test=False)[0](X[0,:]))
            LFuncZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=0, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=Deriv, Test=False)[0](X[1,:]))
        return LFuncR, LFuncZ
    elif Deriv==2:
        # Formulas for Gauss and Mean curvature were found on http://en.wikipedia.org/wiki/Differential_geometry_of_surfaces
        DRR, DRZ, DZZ = [], [], []
        for ii in range(0,BF2.NFunc):
            DRR.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=Deriv, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=0, Test=False)[0](X[1,:]))
            DRZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=1, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=1, Test=False)[0](X[1,:]))
            DZZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=0, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=Deriv, Test=False)[0](X[1,:]))
        return DRR, DRZ, DZZ




def Get_MinMax(BF2, Coefs=1., Ratio=0.05, SubP=0.004, SubP1=0.015, SubP2=0.001, TwoSteps=False, SubMode='abs', Deriv='D0', Margin=0.2, Test=True):
    assert Ratio is None or (type(Ratio) is float and Ratio>0 and Ratio<1), "Arg Ratio must be None or a float in ]0;1[ !"
    if TwoSteps:
        X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP1, SubMode='abs', Test=Test)
    else:
        X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode='abs', Test=Test)
    nx, ny = X.size, Y.size
    dS = np.mean(np.diff(X))*np.mean(np.diff(Y))
    Xplot, Yplot = np.tile(X,(ny,1)), np.tile(Y,(nx,1)).T
    Points = np.array([Xplot.flatten(), Yplot.flatten()])
    Vals = BF2.get_TotVal(Points, Deriv=Deriv, Coefs=Coefs, Test=Test)

    def getminmaxRatioFloat(valsmax, valsmin, indmax, indmin, Ratio, Ptsmin, Ptsmax):
        DV = valsmax[indmax]-valsmin[indmin]
        imin, imax = valsmin<=valsmin[indmin]+Ratio*DV, valsmax>=valsmax[indmax]-Ratio*DV
        PMin = np.array([np.sum(Ptsmin[0,imin]*valsmin[imin])/np.sum(valsmin[imin]), np.sum(Ptsmin[1,imin]*valsmin[imin])/np.sum(valsmin[imin])])
        PMax = np.array([np.sum(Ptsmax[0,imax]*valsmax[imax])/np.sum(valsmax[imax]), np.sum(Ptsmax[1,imax]*valsmax[imax])/np.sum(valsmax[imax])])
        VMin, VMax = np.mean(valsmin[imin]), np.mean(valsmax[imax])
        return PMin, PMax, VMin, VMax

    def get_XYgridFine(DXmin, DYmin, DXmax, DYmax, Margin, subp):
        DXmin, DYmin = [DXmin[0]-Margin*(DXmin[1]-DXmin[0]), DXmin[1]+Margin*(DXmin[1]-DXmin[0])], [DYmin[0]-Margin*(DYmin[1]-DYmin[0]), DYmin[1]+Margin*(DYmin[1]-DYmin[0])]
        DXmax, DYmax = [DXmax[0]-Margin*(DXmax[1]-DXmax[0]), DXmax[1]+Margin*(DXmax[1]-DXmax[0])], [DYmax[0]-Margin*(DYmax[1]-DYmax[0]), DYmax[1]+Margin*(DYmax[1]-DYmax[0])]
        Nxmin, Nymin = (DXmin[1]-DXmin[0])/subp, (DYmin[1]-DYmin[0])/subp
        Nxmax, Nymax = (DXmax[1]-DXmax[0])/subp, (DYmax[1]-DYmax[0])/subp
        Xmin, Ymin = np.linspace(DXmin[0],DXmin[1], Nxmin), np.linspace(DYmin[0],DYmin[1], Nymin)
        Xmax, Ymax = np.linspace(DXmax[0],DXmax[1], Nxmax), np.linspace(DYmax[0],DYmax[1], Nymax)
        Ptsmin = np.array([np.tile(Xmin,(Nymin,1)).flatten(), np.tile(Ymin,(Nxmin,1)).T.flatten()])
        Ptsmax = np.array([np.tile(Xmax,(Nymax,1)).flatten(), np.tile(Ymax,(Nxmax,1)).T.flatten()])
        return Ptsmin, Ptsmax

    def get_minmaxFine(vals, indmaxi, indmini, coefs, Points=Points, ratio=0.02, SubP2=SubP2, Ratio=Ratio, Test=Test):
        DV = vals[indmaxi]-vals[indmini]
        imin, imax = vals<=vals[indmini]+ratio*DV, vals>=vals[indmaxi]-ratio*DV
        xminmin, xminmax = np.min(Points[0,imin]), np.max(Points[0,imin])
        xmaxmin, xmaxmax = np.min(Points[0,imax]), np.max(Points[0,imax])
        yminmin, yminmax = np.min(Points[1,imin]), np.max(Points[1,imin])
        ymaxmin, ymaxmax = np.min(Points[1,imax]), np.max(Points[1,imax])
        Ptsmin, Ptsmax = get_XYgridFine([xminmin,xminmax], [yminmin,yminmax], [xmaxmin, xmaxmax], [ymaxmin, ymaxmax], Margin, SubP2)
        valsmin, valsmax = BF2.get_TotVal(Ptsmin, Deriv=Deriv, Coefs=coefs, Test=Test), BF2.get_TotVal(Ptsmax, Deriv=Deriv, Coefs=coefs, Test=Test)
        indmin, indmax = np.nanargmin(valsmin), np.nanargmax(valsmax)
        if Ratio is None:
            return Ptsmin[:,indmin], Ptsmax[:,indmax], valsmin[indmin], valsmax[indmax]
        else:
            return getminmaxRatioFloat(valsmax, valsmin, indmax, indmin, Ratio, Ptsmin, Ptsmax)

    if not hasattr(Coefs,'__getitem__') or Coefs.ndim==1:
        indmin, indmax = np.nanargmin(Vals), np.nanargmax(Vals)
        if TwoSteps:
            PMin, PMax, VMin, VMax = get_minmaxFine(Vals, indmax, indmin, Coefs, Points=Points, ratio=0.02, SubP2=SubP2, Ratio=Ratio, Test=Test)
        else:
            if Ratio is None:
                PMin, PMax = Points[:,indmin], Points[:,indmax]
                VMin, VMax = Vals[indmin], Vals[indmax]
            else:
                PMin, PMax, VMin, VMax = getminmaxRatioFloat(Vals, Vals, indmax, indmin, Ratio, Points, Points)
        Surf = (Vals >= Vals(np.nanargmax(Vals))*0.5).sum()*dS
    else:
        indmin, indmax = np.nanargmin(Vals,axis=1), np.nanargmax(Vals,axis=1)
        if TwoSteps:
            ratio = 0.02 if Ratio is None else Ratio+0.02
            mmin, mmax = np.nanmin(Vals,axis=1).max(), np.nanmax(Vals,axis=1).min()
            DV = np.max(np.nanmax(Vals,axis=1)-np.nanmin(Vals,axis=1))
            assert mmin+ratio*DV <= mmax-ratio*DV, "Profile changes too much !"
            imin, imax = np.any(Vals<=mmin+ratio*DV,axis=0), np.any(Vals>=mmax-ratio*DV,axis=0)
            DXmin, DYmin = [Points[0,imin].min(), Points[0,imin].max()], [Points[1,imin].min(), Points[1,imin].max()]
            DXmax, DYmax = [Points[0,imax].min(), Points[0,imax].max()], [Points[1,imax].min(), Points[1,imax].max()]
            Ptsmin, Ptsmax = get_XYgridFine(DXmin, DYmin, DXmax, DYmax, Margin, SubP2)
            Valsmin, Valsmax = BF2.get_TotVal(Ptsmin, Deriv=Deriv, Coefs=Coefs, Test=Test), BF2.get_TotVal(Ptsmax, Deriv=Deriv, Coefs=Coefs, Test=Test)
        else:
            Ptsmin, Ptsmax = Points, Points
            Valsmin, Valsmax = Vals, Vals
        indmin, indmax = np.nanargmin(Valsmin,axis=1), np.nanargmax(Valsmax,axis=1)
        if Ratio is None:
            PMin, PMax = Ptsmin[:,indmin].T, Ptsmax[:,indmax].T
            VMin, VMax = np.nanmin(Valsmin,axis=1), np.nanmax(Valsmax,axis=1)
        else:
            Nt = Coefs.shape[0]
            PMin, PMax = np.empty((Nt,2)), np.empty((Nt,2))
            VMin, VMax = np.empty((Nt,)), np.empty((Nt,))
            for ii in range(0,Nt):
                PMin[ii,:], PMax[ii,:], VMin[ii], VMax[ii] = getminmaxRatioFloat(Valsmax[ii,:], Valsmin[ii,:], indmax[ii], indmin[ii], Ratio, Ptsmin, Ptsmax)
        Surf = np.sum(Vals >= np.tile(np.nanmax(Vals,axis=1)*0.5,(Vals.shape[1],1)).T,axis=1)*dS

    return PMin, PMax, VMin, VMax, Surf








def Calc_GetRoots(BF2, Deriv=0., Coefs=1., Test=True):
    if Test:
        assert isinstance(BF2, BF2D), "Arg BF2 must be a BF2D instance !"
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'], "Arg Deriv must be in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'] !"
    if type(Deriv) is str:
        intDeriv = int(Deriv[1])
    else:
        intDeriv = Deriv
    assert BF2.Deg > 0 and intDeriv<BF2.Deg, "Cannot find roots for Deg=0 and Deriv=Deg (root-finding only for continuous functions)"
    NCents = BF2.Mesh.NCents
    NKnots = BF2.Mesh.NKnots
    NbF = BF2.NFunc
    if type(Coefs) in [int,float]:
        Coefs = float(Coefs)*np.ones((NbF,))
    Inds = np.zeros((NCents,),dtype=bool)
    Pts = np.nan*np.zeros((2,NCents))
    Shape = [['',[]] for ii in range(0,NCents)]
    if intDeriv==0:
        if BF2.Deg==1:
            for ii in range(0,NCents):
                ind = BF2._Cents_Funcind[:,ii]
                C4 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]
                if not (np.all(C4<0) or np.all(C4>0)) and C4.size==4:
                    Inds[ii] = True
                    A = C4[0]+C4[3] - C4[1]+C4[2]
                    B = C4[1]*Kts[1,1] - C4[3]*Kts[1,0] - C4[0]*Kts[1,1] + C4[2]*Kts[1,0]
                    C = C4[1]*Kts[0,0] - C4[3]*Kts[0,0] - C4[0]*Kts[0,1] + C4[2]*Kts[0,1]
                    D = Kts[0,1]*(C4[0]*Kts[1,1]-C4[2]*Kts[1,0]) - Kts[0,0]*(C4[1]*Kts[1,1]-C4[3]*Kts[1,0])
                    if A==0.:
                        if C==0.:
                            if B==0.:
                                Shape[ii] = ['all',[]] if D==0. else ['',[]] # Else not possible
                            else:
                                Shape[ii] = ['x',[-D/B]]
                        else:
                            Shape[ii] = ['yx',[-B/C,-D/C]]
                    else:
                        if -C/A>Kts[0,1] or -C/A<Kts[0,0]:
                            Shape[ii] = ['y/x',[(B*C-D)/A,C/A,-B/C]]
                        else:
                            print "        Singularity"
        elif BF2.Deg==2:
            for ii in range(0,NCents):
                ind = BF2._Cents_Funcind[:,ii]
                C9 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]
                if not (np.all(C9<0) or np.all(C9>0)) and C9.size==9:
                    Inds[ii] = True
                    print "        Not finished yet !"




    if intDeriv==1:
        if Deriv[1]=='D1N2':
            if BF2.Deg==2:
                for ii in range(0,NCents):
                    ind = BF2._Cents_Funcind[:,ii]
                    C9 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                    Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]

                    #A0, B0 =
                    #A1, B1 =
                    #A2, B2 =
                    #alpha0, beta0, gam0 =
                    #alpha1, beta1, gam1 =
                    #alpha2, beta2, gam2 =
    return Inds, Pts, Shape






############################################
#####     Plotting functions
############################################



def Plot_BF2D(BF2, Coef=1., ax='None',SubP=0.1, SubMode='Rel', Name='Tot', TotDict=Tot2Dict_Def):
    assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
    assert ax=='None' or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
    assert type(Name) is str, "Arg Name must be a str !"

    assert type(TotDict) is dict, "Arg TotDict must be a dict !"

    X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
    nx, ny = X.size, Y.size
    Xplot, Yplot = np.dot(np.ones((ny,1)),X.reshape((1,nx))), np.dot(Y.reshape((ny,1)),np.ones((1,nx)))
    Points = np.concatenate((Xplot.reshape((1,nx*ny)), Yplot.reshape((1,nx*ny))),axis=0)
    Z = Calc_BF2D_Val(BF2.LFunc, Points, Coef=Coef, Test=True)
    Zplot= Z.reshape((ny,nx))

    if ax=='None':
        ax = tfd.Plot_BSpline_DefAxes('2D')
    ax.plot_surface(Xplot,Yplot,Zplot, label=Name, **TotDict)

    ax.figure.canvas.draw()
    return ax




def Plot_BF2D_BFuncMesh(BF2, ind, Coef=1., ax1='None', ax2='None',SubP=0.25, SubMode='Rel', Name='', TotDict=Tot2Dict_Def):
    assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
    assert type(ind) is int, "Arg ind must be a int !"
    assert ax1=='None' or isinstance(ax1,plt.Axes), "Arg ax1 must be a plt.Axes instance !"
    assert ax2=='None' or isinstance(ax2,plt.Axes), "Arg ax2 must be a plt.Axes instance !"
    assert type(Name) is str, "Arg Name must be a str !"
    assert type(TotDict) is dict, "Arg TotDict must be a dict !"

    X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
    nx, ny = X.size, Y.size
    Xplot, Yplot = np.dot(np.ones((ny,1)),X.reshape((1,nx))), np.dot(Y.reshape((ny,1)),np.ones((1,nx)))
    Points = np.concatenate((Xplot.reshape((1,nx*ny)), Yplot.reshape((1,nx*ny))),axis=0)
    Z = Calc_BF2D_Val([BF2.LFunc[ind]], Points, Coef=Coef, Test=True)
    Zplot= Z.reshape((ny,nx))

    if ax1=='None' or ax2=='None':
        ax1, ax2 = tfd.Plot_BF2D_BFuncMesh_DefAxes()

    BF2.Mesh.plot(ax=ax1)
    BF2.Mesh.plot_Cents(ax=ax1,Ind=BF2.Func_Centsind[:,ind], Knots=False)
    BF2.Mesh.plot_Knots(ax=ax1,Ind=BF2.Func_Knotsind[:,ind], Cents=False)

    ax2.plot_surface(Xplot,Yplot,Zplot, label=Name, **TotDict)
    ax1.figure.canvas.draw()
    return ax1, ax2


def Plot_BFunc_SuppMax_PolProj(BF2, ind, ax='None', Supp=True,PMax=True,SuppDict=SuppDict_Def, PMaxDict=PMaxDict_Def):
    assert type(ind) is int, "Arg ind must be a int !"
    assert type(Supp) is bool, "Arg Supp must be a bool !"
    assert type(PMax) is bool, "Arg Supp must be a bool !"
    assert type(SuppDict) is dict, "Arg SuppDict must be a dict !"
    assert type(PMaxDict) is dict, "Arg SuppDict must be a dict !"
    assert isinstance(ax,plt.Axes) or ax=='None', "Arg ax must be a plt.Axes instance !"


    if ax=='None':
        ax = tfd.Plot_BFunc_SuppMax_PolProj_DefAxes()

    if Supp:
        RZsupp = BF2.get_Func_SuppRZ()[:,ind]
        verts = [(RZsupp[0], RZsupp[2]), # left, bottom
                (RZsupp[1], RZsupp[2]), # left, top
                (RZsupp[1], RZsupp[3]), # right, top
                (RZsupp[0], RZsupp[3]), # right, bottom
                (RZsupp[0], RZsupp[2])]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        patch = patches.PathPatch(Path(verts, codes), **SuppDict)
        ax.add_patch(patch)
    if PMax:
        PMax = BF2.Func_PMax[:,ind]
        ax.plot(PMax[0],PMax[1], **PMaxDict)

    return ax



"""
###############################################################################
###############################################################################
                        Testing ground
###############################################################################
"""
"""
Deg = 2
KnotsMult1 = np.array([0.,1.,2.,3.,4.,5., 5., 5.])
KnotsMult2 = np.array([0.,0.,1.,2.,3.,4.,5.])
BS1 = BSpline(Deg,KnotsMult1)[0]
BS2 = BSpline(Deg,KnotsMult2)[0]
#ax = Plot_BSpline1D(KnotsMult1,BS1,ax='None',SubP=0.05,SubMode='Rel')
ax1, ax2 = Plot_BSpline2D(KnotsMult1, KnotsMult2, BS1, BS2, ax1=None, ax2='None',SubP=0.05,SubMode='Rel')
"""

