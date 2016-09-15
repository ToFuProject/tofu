# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
import scipy.sparse as scpsp
import scipy.interpolate as scpinterp


# ToFu-specific
import tofu.defaults as TFD





"""
###############################################################################
###############################################################################
                        B-spline definitions
###############################################################################
"""

# Direct definitions of b-splines and their derivatives for degrees 0-3

def _Nj0(x,xa,xb):
    def Nfunc(x):
        a = np.zeros((x.size,))
        a[(x >= xa) & (x < xb)] = 1.
        return a
    return Nfunc

def _Nj1(x, x0,x1,x2):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x2)
        xx, aa = x[Ind], A[Ind]
        ind = (xx>=x0)&(xx<x1)
        aa[ind] = (xx[ind]-x0)/(x1-x0)
        ind = (xx>=x1)&(xx<x2)
        aa[ind] = (x2-xx[ind])/(x2-x1)
        A[Ind] = aa
        return A
    return Nfunc

def _Nj1D1(x, x0,x1,x2):
    def Nfunc(x):
        a = np.zeros((x.size,))
        ind = (x>=x0)&(x<x1)
        a[ind] = 1./(x1-x0)
        ind = (x>=x1)&(x<x2)
        a[ind] = -1./(x2-x1)
        return a
    return Nfunc

def _Nj2(x, x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x3)
        xx, aa = x[Ind], A[Ind]
        ind = (xx>=x0)&(xx<x1)
        aa[ind] = (xx[ind]-x0)**2/((x1-x0)*(x2-x0))
        ind = (xx>=x1)&(xx<x2)
        aa[ind] = (-(x3+x2-x1-x0)*xx[ind]**2 + 2.*(x3*x2-x1*x0)*xx[ind] - x0*x2*(x3-x1)-x1*x3*(x2-x0))/((x2-x1)*(x2-x0)*(x3-x1))    # = (x2-x[ind])*(x[ind]-x0)/((x2-x1)*(x2-x0)) + (x[ind]-x1)*(x3-x[ind])/((x2-x1)*(x3-x1))
        ind = (xx>=x2)&(xx<x3)
        aa[ind] = (x3-xx[ind])**2/((x3-x2)*(x3-x1))
        A[Ind] = aa
        return A
    return Nfunc

def _Nj2D1(x, x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x3)
        xx, aa = x[Ind], A[Ind]
        ind = (xx>=x0)&(xx<x1)
        aa[ind] = 2.*(xx[ind]-x0)/((x1-x0)*(x2-x0))
        ind = (xx>=x1)&(xx<x2)
        aa[ind] = (-2.*(x3+x2-x1-x0)*xx[ind] + 2.*(x3*x2-x1*x0))/((x2-x1)*(x2-x0)*(x3-x1))
        ind = (xx>=x2)&(xx<x3)
        aa[ind] = -2.*(x3-xx[ind])/((x3-x2)*(x3-x1))
        A[Ind] = aa
        return A
    return Nfunc

def _Nj2D2(x, x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x3)
        xx, aa = x[Ind], A[Ind]
        ind = (xx>=x0)&(xx<x1)
        aa[ind] = 2./((x1-x0)*(x2-x0))
        ind = (xx>=x1)&(xx<x2)
        aa[ind] = -2.*(x3+x2-x1-x0)/((x2-x1)*(x2-x0)*(x3-x1))
        ind = (xx>=x2)&(xx<x3)
        aa[ind] = 2./((x3-x2)*(x3-x1))
        A[Ind] = aa
        return A
    return Nfunc

def _Nj3(x, x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x4)
        if np.any(Ind):
            xx, aa = x[Ind], A[Ind]
            A10, A20, A21, A30, A31, A32, A41, A42, A43 = x1-x0, x2-x0, x2-x1, x3-x0, x3-x1, x3-x2, x4-x1, x4-x2, x4-x3
            B1, B2, B3 = A31*A41, A20*A41, A30*A20
            D1, D2, D3 = A41*A42, A30*A42, A30*A31
            ind = (xx>=x0)&(xx<x1)
            aa[ind] = (xx[ind]-x0)**3/(A10*B3)
            ind = (xx>=x1)&(xx<x2)
            # = ((xx[ind]-x0)**2*(x2-xx[ind]))/((x3-x0)*(x2-x0)*(x2-x1)) + ((xx[ind]-x0)*(xx[ind]-x1)*(x3-xx[ind]))/((x3-x0)*(x3-x1)*(x2-x1)) + ((xx[ind]-x1)**2*(x4-xx[ind]))/((x4-x1)*(x3-x1)*(x2-x1))
            aa[ind] = (-(B1+B2+B3)*xx[ind]**3 + (B1*(2.*x0+x2)+B2*(x0+x1+x3)+B3*(2.*x1+x4))*xx[ind]**2 - (B1*x0*(x0+2.*x2)+B2*(x0*x1+x0*x3+x1*x3)+B3*x1*(2.*x4+x1))*xx[ind] + (B1*x2*x0**2+B2*x0*x1*x3+B3*x4*x1**2) ) / (B1*B3*A21)
            ind = (xx>=x2)&(xx<x3)
            # = ((xx[ind]-x0)*(x3-xx[ind])**2) / ((x3-x0)*(x3-x1)*(x3-x2)) + ((x3-xx[ind])*(xx[ind]-x1)*(x4-xx[ind])) / ((x3-x2)*(x3-x1)*(x4-x1)) + ((xx[ind]-x2)*(x4-xx[ind])**2) / ((x3-x2)*(x4-x2)*(x4-x1))
            aa[ind] = ( (D1+D2+D3)*xx[ind]**3 - (D1*(2.*x3+x0)+D2*(x1+x3+x4)+D3*(x2+2.*x4))*xx[ind]**2 + (D1*x3*(2.*x0+x3)+D2*(x1*x3+x1*x4+x3*x4)+D3*x4*(2.*x2+x4))*xx[ind] - (D1*x0*x3**2+D2*x1*x3*x4+D3*x2*x4**2) ) / (D1*A32*D3)
            ind = (xx>=x3)&(xx<x4)
            aa[ind] = (x4-xx[ind])**3/(D1*A43)
            A[Ind] = aa
        return A
    return Nfunc

def _Nj3D1(x, x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x4)
        if np.any(Ind):
            xx, aa = x[Ind], A[Ind]
            A10, A20, A21, A30, A31, A32, A41, A42, A43 = x1-x0, x2-x0, x2-x1, x3-x0, x3-x1, x3-x2, x4-x1, x4-x2, x4-x3
            B1, B2, B3 = A31*A41, A20*A41, A30*A20
            D1, D2, D3 = A41*A42, A30*A42, A30*A31
            ind = (xx>=x0)&(xx<x1)
            aa[ind] = 3.*(xx[ind]-x0)**2/(A10*B3)
            ind = (xx>=x1)&(xx<x2)
            aa[ind] = (-3.*(B1+B2+B3)*xx[ind]**2 + 2.*(B1*(2.*x0+x2)+B2*(x0+x1+x3)+B3*(2.*x1+x4))*xx[ind] - (B1*x0*(x0+2.*x2)+B2*(x0*x1+x0*x3+x1*x3)+B3*x1*(2.*x4+x1)) ) / (B1*B3*A21)
            ind = (xx>=x2)&(xx<x3)
            aa[ind] = ( 3.*(D1+D2+D3)*xx[ind]**2 - 2.*(D1*(2.*x3+x0)+D2*(x1+x3+x4)+D3*(x2+2.*x4))*xx[ind] + (D1*x3*(2.*x0+x3)+D2*(x1*x3+x1*x4+x3*x4)+D3*x4*(2.*x2+x4)) ) / (D1*A32*D3)
            ind = (xx>=x3)&(xx<x4)
            aa[ind] = -3.*(x4-xx[ind])**2/(D1*A43)
            A[Ind] = aa
        return A
    return Nfunc

def _Nj3D2(x, x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x4)
        if np.any(Ind):
            xx, aa = x[Ind], A[Ind]
            A10, A20, A21, A30, A31, A32, A41, A42, A43 = x1-x0, x2-x0, x2-x1, x3-x0, x3-x1, x3-x2, x4-x1, x4-x2, x4-x3
            B1, B2, B3 = A31*A41, A20*A41, A30*A20
            D1, D2, D3 = A41*A42, A30*A42, A30*A31
            ind = (xx>=x0)&(xx<x1)
            aa[ind] = 6.*(xx[ind]-x0)/(A10*B3)
            ind = (xx>=x1)&(xx<x2)
            aa[ind] = (-6.*(B1+B2+B3)*xx[ind] + 2.*(B1*(2.*x0+x2)+B2*(x0+x1+x3)+B3*(2.*x1+x4)) ) / (B1*B3*A21)
            ind = (xx>=x2)&(xx<x3)
            aa[ind] = ( 6.*(D1+D2+D3)*xx[ind] - 2.*(D1*(2.*x3+x0)+D2*(x1+x3+x4)+D3*(x2+2.*x4)) ) / (D1*A32*D3)
            ind = (xx>=x3)&(xx<x4)
            aa[ind] = 6.*(x4-xx[ind])/(D1*A43)
            A[Ind] = aa
        return A
    return Nfunc

def _Nj3D3(x, x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros((x.size,))
        Ind = (x>=x0)&(x<x4)
        if np.any(Ind):
            xx, aa = x[Ind], A[Ind]
            A10, A20, A21, A30, A31, A32, A41, A42, A43 = x1-x0, x2-x0, x2-x1, x3-x0, x3-x1, x3-x2, x4-x1, x4-x2, x4-x3
            B1, B2, B3 = A31*A41, A20*A41, A30*A20
            D1, D2, D3 = A41*A42, A30*A42, A30*A31
            ind = (xx>=x0)&(xx<x1)
            aa[ind] = 6./(A10*B3)
            ind = (xx>=x1)&(xx<x2)
            aa[ind] = -6.*(B1+B2+B3) / (B1*B3*A21)
            ind = (xx>=x2)&(xx<x3)
            aa[ind] = 6.*(D1+D2+D3) / (D1*A32*D3)
            ind = (xx>=x3)&(xx<x4)
            aa[ind] = -6./(D1*A43)
            A[Ind] = aa
        return A
    return Nfunc


# Set of basis functions in 1D

def BSpline_LFunc(Deg, Knots, Deriv=0, Test=True):
    if Test:
        assert type(Deg) is int and Deg in [0,1,2,3] and type(Deriv) is int and Deg>=Deriv, "Arg Deg and Deriv must be int and Deg >= Deriv !"
        assert isinstance(Knots,np.ndarray) and Knots.ndim==1 and np.all(Knots==np.unique(Knots)) and Knots.size>Deg+1 , "Arg Knots must be a 1-dim np.ndarray of unique increasing floats with Knots.size>Deg+1 !"
    NKnots = Knots.size
    NCents = NKnots-1
    NbF = NKnots-1-Deg
    LFunc = []
    if Deg==0:
        Func_Knotsind = np.array([np.arange(0,NKnots-1),np.arange(1,NKnots)])
        Func_Centsind = np.arange(0,NCents).reshape((1,NbF))
        Knots_Funcind = np.array([np.concatenate(([np.nan],np.arange(0,NbF))),np.concatenate((np.arange(0,NbF),[np.nan]))])
        Cents_Funcind = np.arange(0,NbF).reshape((1,NCents))
        MaxPos = np.mean(np.array([Knots[:-1],Knots[1:]]),axis=0)
        for ii in range(0,NbF):
            LFunc.append(_Nj0(0,Knots[ii],Knots[ii+1]))
    elif Deg==1:
        Func_Knotsind = np.array([np.arange(0,NKnots-2),np.arange(1,NKnots-1),np.arange(2,NKnots)])
        Func_Centsind = np.array([np.arange(0,NCents-1),np.arange(1,NCents)])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan],np.arange(0,NbF))),np.concatenate(([np.nan],np.arange(0,NbF),[np.nan])),np.concatenate((np.arange(0,NbF),[np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan],np.arange(0,NbF))),np.concatenate((np.arange(0,NbF),[np.nan]))])
        MaxPos = np.copy(Knots[1:-1])
        if Deriv==0:
            for ii in range(0,NbF):
                LFunc.append(_Nj1(0,Knots[ii],Knots[ii+1],Knots[ii+2]))
        elif Deriv==1:
            for ii in range(0,NbF):
                LFunc.append(_Nj1D1(0,Knots[ii],Knots[ii+1],Knots[ii+2]))
    elif Deg==2:
        Func_Knotsind = np.array([np.arange(0,NKnots-3),np.arange(1,NKnots-2),np.arange(2,NKnots-1),np.arange(3,NKnots)])
        Func_Centsind = np.array([np.arange(0,NCents-2),np.arange(1,NCents-1),np.arange(2,NCents)])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan],np.arange(0,NbF))),np.concatenate(([np.nan,np.nan],np.arange(0,NbF),[np.nan])),np.concatenate(([np.nan],np.arange(0,NbF),[np.nan,np.nan])),np.concatenate((np.arange(0,NbF),[np.nan,np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan,np.nan],np.arange(0,NbF))),np.concatenate(([np.nan],np.arange(0,NbF),[np.nan])),np.concatenate((np.arange(0,NbF),[np.nan,np.nan]))])
        MaxPos = np.mean(np.array([Knots[1:-2],Knots[2:-1]]),axis=0)
        if Deriv==0:
            for ii in range(0,NbF):
                LFunc.append(_Nj2(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]))
        elif Deriv==1:
            for ii in range(0,NbF):
                LFunc.append(_Nj2D1(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]))
        elif Deriv==2:
            for ii in range(0,NbF):
                LFunc.append(_Nj2D2(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]))
    elif Deg==3:
        Func_Knotsind = np.array([np.arange(0,NKnots-4),np.arange(1,NKnots-3),np.arange(2,NKnots-2),np.arange(3,NKnots-1),np.arange(4,NKnots)])
        Func_Centsind = np.array([np.arange(0,NCents-3),np.arange(1,NCents-2),np.arange(2,NCents-1),np.arange(3,NCents)])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan,np.nan],np.arange(0,NbF))),np.concatenate(([np.nan,np.nan,np.nan],np.arange(0,NbF),[np.nan])),np.concatenate(([np.nan,np.nan],np.arange(0,NbF),[np.nan,np.nan])),np.concatenate(([np.nan],np.arange(0,NbF),[np.nan,np.nan,np.nan])),np.concatenate((np.arange(0,NbF),[np.nan,np.nan,np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan],np.arange(0,NbF))),np.concatenate(([np.nan,np.nan],np.arange(0,NbF),[np.nan])),np.concatenate(([np.nan],np.arange(0,NbF),[np.nan,np.nan])),np.concatenate((np.arange(0,NbF),[np.nan,np.nan,np.nan]))])
        MaxPos = np.copy(Knots[2:-2])
        if Deriv==0:
            for ii in range(0,NbF):
                LFunc.append(_Nj3(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]))
        elif Deriv==1:
            for ii in range(0,NbF):
                LFunc.append(_Nj3D1(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]))
        elif Deriv==2:
            for ii in range(0,NbF):
                LFunc.append(_Nj3D2(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]))
        elif Deriv==3:
            for ii in range(0,NbF):
                LFunc.append(_Nj3D3(0,Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]))
    return LFunc, Func_Knotsind, Func_Centsind, Knots_Funcind, Cents_Funcind, MaxPos

def BSpline_TotFunc(Deg, Knots, Deriv=0, Coefs=1., thr=1.e-8, thrmode='rel', Test=True):
    if Test:
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D0ME','D1N2','D1FI','D2N2','D3N2'], "Arg Deriv must be in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI','D2N2','D3N2'] !"
        assert type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim==1 and Coefs.size==Knots.size-1-Deg), "Arg Coefs must be a flot or a 1D np.ndarray !"
    if type(Deriv) is str:
        intDeriv = int(Deriv[1])
    else:
        intDeriv = Deriv
    LF = BSpline_LFunc(Deg, Knots, Deriv=intDeriv, Test=Test)[0]
    if type(Coefs) in [int,float]:
        Coefs = float(Coefs)*np.ones((len(LF),))
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        Func = lambda x,Coefs=Coefs,LF=LF: np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)
    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        Func = lambda x,Coefs=Coefs,LF=LF: np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)**2
    elif Deriv=='D1FI':
        lf = BSpline_LFunc(Deg, Knots, Deriv=0, Test=Test)[0]
        Func = lambda x,Coefs=Coefs,LF=LF: np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)**2/np.sum(np.concatenate(tuple([Coefs[ii]*lf[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)
    elif Deriv=='D0ME':
        ext = [Knots[0],Knots[-1]]
        def Func(x,coefs=Coefs, LF=LF, ext=ext, thr=thr, thrmode=thrmode):
            g = np.sum([coefs[jj]*LF[jj](x).reshape((1,x.size)) for jj in range(0,NF)],axis=0)
            Int = np.nanmean(g)*(ext[1]-ext[0])  # To be finished !!! replace with the real integral of the function to make it a distribution function !!!
            g = g / Int
            if thr is not None:
                thr = thr if thrmode=='abs' else np.nanmax(g)*thr
                g[g<thr] = thr
            return -g*np.log(g)
    return Func

"""
def BSpline_TotVal(Deg, Knots, Deriv=0, Coefs=1., Test=True):
    if Test:
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI','D2N2','D3N2']
    if type(Deriv) is str:
        intDeriv = int(Deriv[1])
    else:
        intDeriv = Deriv
    LF = BSpline_LFunc(Deg, Knots, Deriv=intDeriv, Test=Test)[0]
    if type(Coefs) in [int,float]:
        Coefs = float(Coefs)*np.ones((len(LF),))
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        Val = np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)
    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        Val = np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)**2
    elif Deriv=='D1FI':
        lf = BSpline_LFunc(Deg, Knots, Deriv=0, Test=Test)[0]
        Val = np.sum(np.concatenate(tuple([Coefs[ii]*LF[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)**2/np.sum(np.concatenate(tuple([Coefs[ii]*lf[ii](x).reshape((1,x.size)) for ii in range(0,len(LF))]),axis=0),axis=0, keepdims=False)
    return Val
"""


def BSpline_TotRoot(Deg, Knots, Deriv=0, Coefs=1., Test=True):  # To be finished, useful later for advanced post-treatment
    if Test:
        assert type(Deg) is int and Deg in [0,1,2,3], "Arg Deg must be int in [0,1,2,3] !"
        assert isinstance(Knots,np.ndarray) and Knots.ndim==1 and np.all(Knots==np.unique(Knots)) and Knots.size>Deg+1 , "Arg Knots must be a 1-dim np.ndarray of unique increasing floats with Knots.size>Deg+1 !"
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'], "Arg Deriv must be in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'] !"
    if type(Deriv) is str:
        intDeriv = int(Deriv[1])
    else:
        intDeriv = Deriv
    assert Deg > 0 and intDeriv<Deg, "Cannot find roots for Deg=0 and Deriv=Deg (root-finding only for continuous functions)"
    NKnots = Knots.size
    NbF = NKnots-1-Deg
    if type(Coefs) in [int,float]:
        Coefs = float(Coefs)*np.ones((NbF,))
    Int = np.array([Knots[Deg:(-1-Deg)],Knots[(Deg+1):-Deg]])
    if intDeriv==0:
        if Deg==1:
            cc = np.array([Coefs[:-Deg],Coefs[Deg:]])
            indNeg = (cc[0,:]*cc[1,:]<=0) & ~(cc[1,:]==0)
            sols = (cc[1,indNeg]*Int[0,indNeg] - cc[0,indNeg]*Int[1,indNeg])/(cc[1,indNeg]-cc[0,indNeg])
        elif Deg==2:
            cc = np.array([Coefs[:-Deg],Coefs[1:-1],Coefs[Deg:]])
            kk = np.array([Knots[1:-4],Knots[2:-3],Knots[3:-2],Knots[4:-1]])
            A = cc[2,:]*(kk[2,:]-kk[0,:]) - cc[1,:]*(kk[3,:]+kk[2,:]-kk[1,:]-kk[0,:]) + cc[0,:]*(kk[3,:]-kk[1,:])
            B = 2.*cc[1,:]*(kk[2,:]*kk[3,:]-kk[0,:]*kk[1,:]) - 2.*cc[2,:]*(kk[2,:]-kk[0,:])*kk[1,:] - 2.*cc[0,:]*(kk[3,:]-kk[1,:])*kk[2,:]
            C = cc[2,:]*(kk[2,:]-kk[0,:])*kk[1,:]**2 + cc[0,:]*(kk[3,:]-kk[1,:])*kk[2,:]**2 - cc[1,:]*(kk[1,:]*kk[3,:]*(kk[2,:]-kk[0,:]) + kk[0,:]*kk[2,:]*(kk[3,:]-kk[1,:]))
            Delta = B**2-4.*A*C
            ind = Delta >= 0
            sols = np.array([(-B[ind]+np.sqrt(Delta[ind]))/(2.*A[ind]), (-B[ind]-np.sqrt(Delta[ind]))/(2.*A[ind])])
            ind = (sols>=Int[0,ind]) & (sols<Int[1,ind])
            sols = np.unique(sols[:,ind].flatten())
    elif intDeriv==1:
        if Deg==2:
            Eps = np.diff(Int,axis=0).flatten()/10000.
            cc = np.array([Coefs[:-Deg],Coefs[1:-1],Coefs[Deg:]])
            kk = np.array([Knots[1:-4],Knots[2:-3],Knots[3:-2],Knots[4:-1]])
            A = cc[0,:]*kk[2,:]*(kk[3,:]-kk[1,:])-cc[1,:]*(kk[3,:]*kk[2,:]-kk[1,:]*kk[0,:])+cc[2,:]*kk[1,:]*(kk[2,:]-kk[0,:])
            B = cc[0,:]*(kk[3,:]-kk[1,:])-cc[1,:]*(kk[3,:]+kk[2,:]-kk[1,:]-kk[0,:])+cc[2,:]*(kk[2,:]-kk[0,:])
            sols = A/B
            indok = (sols>=Int[0,:]-Eps) & (sols<Int[1,:]+Eps)
            sols = sols[indok]
            solbin = np.round(sols/Eps.min())
            solbin, indun =  np.unique(solbin,return_index=True)
            sols = sols[indun]
    return sols




























def Calc_IntOp_Elementary(Deg1, Deg2, BF1, BF2, K1, K2, D1, D2, Test=True):
    F1 = BSplineDeriv(Deg1, K1, Deriv=D1, Test=False)
    F2 = BSplineDeriv(Deg2, K2, Deriv=D2, Test=False)

def Calc_IntOp_BSpline(Knots, Deg, Deriv=0, Sparse=TFD.L1IntOpSpa, SpaFormat=TFD.L1IntOpSpaFormat, Test=True):
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





############################################
#####     Computing functions
############################################


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


def BF2D_get_TotFunc(BF2, Deriv=0, DVect=TFD.BF2_DVect_DefR, Coefs=1., Test=True):
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







def Calc_IntOp_BSpline2D(BF2, Deriv=0, Mode='Vol', Sparse=TFD.L1IntOpSpa, SpaFormat=TFD.L1IntOpSpaFormat, Test=True):
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






