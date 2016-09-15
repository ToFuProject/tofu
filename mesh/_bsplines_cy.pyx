# cython: profile=True
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
cimport numpy as cnp
import math
import scipy.sparse as scpsp
import scipy.interpolate as scpinterp

# ToFu-specific


ctypedef cnp.float_t DTYPE_t


"""
###############################################################################
###############################################################################
                        B-spline definitions
###############################################################################
"""

# Scipy definition


def get_scpKnotsCoefs_From_KnotsDeg(knts, Deg):
    knts2 = [knts[0]]*(1+Deg) + knts[1:-1].tolist() + [knts[-1]]*(1+Deg)
    coefs = [0.]*Deg + [1.] + [0.]*(len(knts2)-Deg-1)
    return knts2, coefs

def _NjD_scp(kntsf, coefsf, int Deg=1, int der=0):
    NFunc = lambda x, Deg=Deg, kntsf=kntsf, coefsf=coefsf, der=der: scpinterp.splev(x, (kntsf, coefsf, Deg), der=der, ext=1)
    return NFunc




# Direct definitions of b-splines and their derivatives for degrees 0-3

def _Nj0(xa, xb):
    def Nfunc(x):
        a = np.zeros(np.shape(x))
        a[(x >= xa) & (x < xb)] = 1.
        return a
    return Nfunc

def _Nj1(x0, x1, x2):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
        Ind = (x>=x0)&(x<x2)
        xx, aa = x[Ind], A[Ind]
        ind = (xx>=x0)&(xx<x1)
        aa[ind] = (xx[ind]-x0)/(x1-x0)
        ind = (xx>=x1)&(xx<x2)
        aa[ind] = (x2-xx[ind])/(x2-x1)
        A[Ind] = aa
        return A
    return Nfunc

def _Nj1D1(x0, x1, x2):
    def Nfunc(x):
        a = np.zeros(np.shape(x))
        ind = (x>=x0)&(x<x1)
        a[ind] = 1./(x1-x0)
        ind = (x>=x1)&(x<x2)
        a[ind] = -1./(x2-x1)
        return a
    return Nfunc

def _Nj2(x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj2D1(x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj2D2(x0,x1,x2,x3):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj3(x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj3D1(x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj3D2(x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def _Nj3D3(x0,x1,x2,x3,x4):
    def Nfunc(x):
        A = np.zeros(np.shape(x))
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

def BSpline_LFunc(int Deg, cnp.ndarray[cnp.float64_t, ndim=1] Knots, int Deriv=0, Mode='scp', Test=True):
    """
    Return a list of 1D b-spline of the required degree, defined using the explicitly provided knots (of single multiplicity)

    Inputs:
    -------
        Deg             int         Degree of the b-splines
        Knots           np.ndarray  Array of unique and sorted (increasing) knots
        Deriv           int         Degree of the derivative to be computed (<=Deg)
        Test            bool        Flag indicating whether the inputs should be tested for conformity

    Outputs:    (if NK denotes the number of knots, NC=NK-1 the number of centers and NF=NK-1-Deg the number of functions)
    --------
        LFunc           list        List of b-spline functions, sorted from the first to the last knot
        Func_Knotsind   np.ndarray  Indexes of the mesh knots on which each function is defined, shape = (2+Deg,NF)
        Func_Centsind   np.ndarray  Indexes of the mesh centers on which each function is defined, shape = (1+Deg,NF)
        Knots_Funcind   np.ndarray  Indexes of the functions that live on each mesh knot, shape = (2+Deg,NK)
        Cents_Funcind   np.ndarray  Indexes of the functions that live on each mesh center, shape = (1+Deg,NC)
        MaxPos          np.ndarray  Positions of the maximum of each function, shape = (NF,)
    
    """
    if Test:
        assert Deg in [0,1,2,3] and Deg>=Deriv, "Arg Deg and Deriv must be int and Deg >= Deriv !"
        assert np.all(Knots==np.unique(Knots)) and Knots.size>Deg+1 , "Arg Knots must be a 1-dim np.ndarray of unique increasing floats with Knots.size>Deg+1 !"
    cdef list LFunc
    cdef int NKnots = Knots.size
    cdef int NbF = NKnots-1-Deg, NCents = NKnots-1
    cdef Py_ssize_t ii

    if Deg==0:
        temp = np.arange(0,NbF)
        Func_Knotsind = np.array([temp,temp+1])
        Func_Centsind = temp.reshape((1,NbF))
        Knots_Funcind = np.array([np.concatenate(([np.nan],temp)),np.concatenate((temp,[np.nan]))])
        Cents_Funcind = np.copy(Func_Centsind)
        MaxPos = 0.5*(Knots[:-1]+Knots[1:])
        if not Mode=='scp':
            LFunc = [_Nj0(Knots[ii],Knots[ii+1]) for ii in range(0,NCents)]
    elif Deg==1:
        temp = np.arange(0,NbF)
        Func_Knotsind = np.array([temp,temp+1,temp+2])
        Func_Centsind = np.array([temp,temp+1])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan],temp)),np.concatenate(([np.nan],temp,[np.nan])),np.concatenate((temp,[np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan],temp)),np.concatenate((temp,[np.nan]))])
        MaxPos = Knots[1:-1]
        if not Mode=='scp':
            if Deriv==0:
                LFunc = [_Nj1(Knots[ii],Knots[ii+1],Knots[ii+2]) for ii in range(0,NbF)]
            elif Deriv==1:
                LFunc = [_Nj1D1(Knots[ii],Knots[ii+1],Knots[ii+2]) for ii in range(0,NbF)]
    elif Deg==2:
        temp = np.arange(0,NbF)
        Func_Knotsind = np.array([temp,temp+1,temp+2,temp+3])
        Func_Centsind = np.array([temp,temp+1,temp+2])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan],temp)),np.concatenate(([np.nan,np.nan],temp,[np.nan])),np.concatenate(([np.nan],temp,[np.nan,np.nan])),np.concatenate((temp,[np.nan,np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan,np.nan],temp)),np.concatenate(([np.nan],temp,[np.nan])),np.concatenate((temp,[np.nan,np.nan]))])
        MaxPos = 0.5*(Knots[1:-2]+Knots[2:-1])
        if not Mode=='scp':
            if Deriv==0:
                LFunc = [_Nj2(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]) for ii in range(0,NbF)]
            elif Deriv==1:
                LFunc = [_Nj2D1(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]) for ii in range(0,NbF)]
            elif Deriv==2:
                LFunc = [_Nj2D2(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3]) for ii in range(0,NbF)]
    elif Deg==3:
        temp = np.arange(0,NbF)
        Func_Knotsind = np.array([temp,temp+1,temp+2,temp+3,temp+4])
        Func_Centsind = np.array([temp,temp+1,temp+2,temp+3])
        Knots_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan,np.nan],temp)),np.concatenate(([np.nan,np.nan,np.nan],temp,[np.nan])),np.concatenate(([np.nan,np.nan],temp,[np.nan,np.nan])),
                                  np.concatenate(([np.nan],temp,[np.nan,np.nan,np.nan])),np.concatenate((temp,[np.nan,np.nan,np.nan,np.nan]))])
        Cents_Funcind = np.array([np.concatenate(([np.nan,np.nan,np.nan],temp)),np.concatenate(([np.nan,np.nan],temp,[np.nan])),np.concatenate(([np.nan],temp,[np.nan,np.nan])),np.concatenate((temp,[np.nan,np.nan,np.nan]))])
        MaxPos = Knots[2:-2]
        if not Mode=='scp':
            if Deriv==0:
                LFunc = [_Nj3(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]) for ii in range(0,NbF)]
            elif Deriv==1:
                LFunc = [_Nj3D1(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]) for ii in range(0,NbF)]
            elif Deriv==2:
                LFunc = [_Nj3D2(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]) for ii in range(0,NbF)]
            elif Deriv==3:
                LFunc = [_Nj3D3(Knots[ii],Knots[ii+1],Knots[ii+2],Knots[ii+3],Knots[ii+4]) for ii in range(0,NbF)]

    scp_Lkntsf, scp_Lcoeff = None, None
    if Mode=='scp':
        scp_Lkntsf, scp_Lcoeff = zip(*[get_scpKnotsCoefs_From_KnotsDeg(Knots[ii:ii+Deg+2], Deg) for ii in range(0,NbF)])
        LFunc = [_NjD_scp(scp_Lkntsf[ii], scp_Lcoeff[ii], Deg=Deg, der=Deriv) for ii in range(0,NbF)]

    return LFunc, Func_Knotsind, Func_Centsind, Knots_Funcind, Cents_Funcind, MaxPos, scp_Lkntsf, scp_Lcoeff




# Total of the functions in 1D

def BSpline_TotFunc(int Deg, cnp.ndarray[cnp.float64_t, ndim=1] Knots, Deriv=0, Coefs=1., thr=1.e-8, thrmode='rel', Abs=True, Mode='scp', Test=True):
    if Test:
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D0ME','D1N2','D1FI','D2N2','D3N2'], "Arg Deriv must be in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI','D2N2','D3N2'] !"
        assert type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim in [1,2] and Knots.size-1-Deg in Coefs.shape), "Arg Coefs must be a int / float or a np.ndarray !"
        intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv
        assert Deg>=intDeriv, "Arg Deriv cannot be larger that Deg !"
        assert Knots.size-1>Deg, "There should be enough knots to allow at least one b-spline !"

    # Preparing input
    cdef list Func = []
    cdef int NF = Knots.size-1-Deg
    Coefs = float(Coefs)*np.ones((NF,)) if type(Coefs) in [int,float] else Coefs
    Coefs = Coefs.T if Coefs.ndim==2 and Coefs.shape[0]==Knots.size-1-Deg else Coefs
    Coefs = Coefs.reshape((1,Coefs.size)) if Coefs.ndim==1 else Coefs
    cdef int Nt = Coefs.shape[0]
    cdef Py_ssize_t ii, jj

    intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv
    LF = BSpline_LFunc(Deg, Knots, Deriv=intDeriv, Mode=Mode, Test=Test)[0]

    # Computing
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        for ii in range(0,Nt):
            Func.append(lambda x,coefs=Coefs[ii,:],LF=LF: np.sum([coefs[jj]*LF[jj](x) for jj in range(0,NF)],axis=0))

    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        for ii in range(0,Nt):
            Func.append(lambda x,coefs=Coefs[ii,:],LF=LF: np.sum([coefs[jj]*LF[jj](x) for jj in range(0,NF)],axis=0)**2)

    elif Deriv=='D1FI':
        lf = BSpline_LFunc(Deg, Knots, Deriv=0, Mode=Mode, Test=Test)[0]
        Span = [Knots[0],Knots[-1]]
        for ii in range(0,Nt):
            def ff(x, coefs=Coefs[ii,:], LF=LF, lf=lf, thr=thr, thrmode=thrmode, Span=Span, Abs=Abs):
                grad2 = np.sum([coefs[jj]*LF[jj](x) for jj in range(0,NF)],axis=0)**2
                g = np.sum([coefs[jj]*lf[jj](x) for jj in range(0,NF)],axis=0)
                g = np.abs(g) if Abs else g
                if thr is not None:
                    thr = thr if thrmode=='abs' else np.nanmax(g)*thr
                    ind = (x>=Span[0]) & (x<=Span[1])
                    g[ind & (g<thr)] = thr
                return grad2/g
            Func.append(ff)

    elif Deriv=='D0ME':
        Span = [Knots[0],Knots[-1]]
        for ii in range(0,Nt):
            def ff(x,coefs=Coefs[ii,:], LF=LF, thr=thr, thrmode=thrmode, Span=Span, Abs=Abs):
                g = np.sum([coefs[jj]*LF[jj](x) for jj in range(0,NF)],axis=0) 
                Int = np.nanmean(g)*(Span[1]-Span[0])  # To be finished !!! replace with the real integral of the function to make it a distribution function !!!
                g = np.abs(g/Int) if Abs else g/Int
                if thr is not None:
                    thr = thr if thrmode=='abs' else np.nanmax(g)*thr
                    ind = (x>=Span[0]) & (x<=Span[1])
                    g[ind & (g<thr)] = thr
                return -g*np.log(g)
            Func.append(ff)

    if Nt==1:
        return Func[0]
    else:
        return Func


def BSpline_TotRoot(int Deg, cnp.ndarray[cnp.float64_t, ndim=1] Knots, Deriv=0, Coefs=1., Test=True):  # To be finished, useful later for advanced post-treatment
    if Test:
        assert Deg in [0,1,2,3], "Arg Deg must be int in [0,1,2,3] !"
        assert Knots.ndim==1 and np.all(Knots==np.unique(Knots)) and Knots.size>Deg+1 , "Arg Knots must be a 1-dim np.ndarray of unique increasing floats with Knots.size>Deg+1 !"
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



############################################
#####     Getting weigths
############################################



def Calc_BF1D_Weights(list LFunc, cnp.ndarray[cnp.float64_t, ndim=1] Pts):
    cdef int NFunc = len(LFunc)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Wgh = np.zeros((Pts.size, NFunc))
    cdef Py_ssize_t ii
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](Pts)
    return Wgh

def Calc_BF2D_Weights(list LFunc, cnp.ndarray[cnp.float64_t, ndim=2] Pts, Test=True):
    if Test:
        assert Pts.shape[0]==2, "Arg Points must be a (2,N) np.ndarray (cartesian coordinates in cross-section) !"
    cdef int NFunc = len(LFunc)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Wgh = np.zeros((Pts.shape[1], NFunc))
    cdef Py_ssize_t ii
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](Pts)
    return Wgh






############################################
#####     Gauss-Legendre
############################################




cdef _get_Quad_GaussLegendre_1D(cnp.ndarray[cnp.float64_t, ndim=1] Knots, int N):
    cdef int NI = Knots.size-1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] zeros = 0.5*(Knots[:-1]+Knots[1:])
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A = 0.5*(Knots[1:]-Knots[:-1])
    if N==1:
        pts = zeros.reshape((1,NI))
        w = 2.*np.ones((N,NI))
    elif N==2:
        pts = np.array([-A/math.sqrt(3) + zeros, A/math.sqrt(3) + zeros])
        w = np.ones((N,NI))
    elif N==3:
        pts = np.array([-A/math.sqrt(3./5.) + zeros, zeros, A/math.sqrt(3./5.) + zeros])
        w = np.ones((N,NI))*5./9.
        w[1,:] = w[1,:]*8./5.
    elif N==4:
        p1 = math.sqrt(3./7. - 2./7.*math.sqrt(6./5.))
        p2 = math.sqrt(3./7. + 2./7.*math.sqrt(6./5.))
        pts = np.array([-p2*A + zeros, -p1*A + zeros, p1*A + zeros, p2*A + zeros])
        w = np.ones((N,NI))*(18.-math.sqrt(30.))/36.
        w[1:3,:] = (18.+math.sqrt(30.))/36.
    elif N==5:
        p1 = math.sqrt(5. - 2.*math.sqrt(10./7.))/3.
        p2 = math.sqrt(5. + 2.*math.sqrt(10./7.))/3.
        pts = np.array([-p2*A + zeros, -p1*A + zeros, zeros, p1*A + zeros, p2*A + zeros])
        w0 = 0.
        w1 = (322.+13.*math.sqrt(70))/900.
        w2 = (322.-13.*math.sqrt(70))/900.
        w = np.tile([w2,w1,w0,w1,w2],(NI,1)).T
    else:
        assert N<=5, "Arg N must be <= 5 !"
    return pts, w, A


def _get_NptsGaussFromDegDeriv(Deg, Deriv, intDeriv, Mode='Surf'):
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        N = int(math.ceil((Deg-intDeriv+1.)/2.)) if Mode=='Surf' else int(math.ceil((Deg-intDeriv+2)/2.))
    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        N = int(Deg-intDeriv+1) if Mode=='Surf' else int(Deg-intDeriv+2)
    elif Deriv=='D0ME':
        N = int(math.ceil((Deg-intDeriv+2.)/2.)) if Mode=='Surf' else int(math.ceil((Deg-intDeriv+3.)/2.))
    elif Deriv=='D1FI':
        N = int(Deg-intDeriv+2) if Mode=='Surf' else int(Deg-intDeriv+3)
    return N


def get_IntQuadPts(cnp.ndarray[cnp.float64_t, ndim=1] Knots, Deg, Deriv, intDeriv, Mode='Surf', N=None):
    N = _get_NptsGaussFromDegDeriv(Deg, Deriv, intDeriv, Mode=Mode) if N is None else N
    pts, w, A = _get_Quad_GaussLegendre_1D(Knots, N)
    return pts, w, A, N



cdef _get_TermsDeg3_1D(cnp.ndarray[cnp.float64_t, ndim=1] Knots):

    x0, x1, x2, x3, x4 = Knots[:-4], Knots[1:-3], Knots[2:-2], Knots[3:-1], Knots[4:]
    A1 = -( (x3-x1)*(x4-x1) + (x2-x0)*(x4-x1) + (x3-x0)*(x2-x0) )
    B1 = (x2+2.*x0)*(x3-x1)*(x4-x1) + (x3+x0+x1)*(x2-x0)*(x4-x1) + (x4+2.*x1)*(x3-x0)*(x2-x0)
    C1 = -( x0*(2.*x2+x0)*(x3-x1)*(x4-x1) + (x3*(x0+x1)+x0*x1)*(x2-x0)*(x4-x1) + x1*(2.*x4+x1)*(x3-x0)*(x2-x0) )
    D1 = x2*x0**2*(x3-x1)*(x4-x1) + x0*x1*x3*(x2-x0)*(x4-x1) + x4*x1**2*(x3-x0)*(x2-x0)
    
    Den1 = (x4-x1)*(x3-x1)*(x3-x0)*(x2-x1)*(x2-x0)
    Den1m = (x4-x1)*(x3-x1)*(x3-x0)*(x2-x0)     # Denominator minus the intregation interval (x2-x1)

    A2 = (x1-x3)*(x0-x3) + (x2-x4)*(x0-x3) + (x1-x4)*(x2-x4)
    B2 = -( (x2+2.*x4)*(x1-x3)*(x0-x3) + (x1+x4+x3)*(x2-x4)*(x0-x3) + (x0+2.*x3)*(x1-x4)*(x2-x4) )
    C2 = x4*(2.*x2+x4)*(x1-x3)*(x0-x3) + (x1*(x4+x3)+x4*x3)*(x2-x4)*(x0-x3) + x3*(2.*x0+x3)*(x1-x4)*(x2-x4)
    D2 = -( x2*x4**2*(x1-x3)*(x0-x3) + x4*x3*x1*(x2-x4)*(x0-x3) + x0*x3**2*(x1-x4)*(x2-x4) )

    Den2 = (x4-x2)*(x4-x1)*(x3-x2)*(x3-x1)*(x3-x0)
    Den2m = (x4-x2)*(x4-x1)*(x3-x1)*(x3-x0)     # Denominator minus the intregation interval (x3-x2)

    return x0, x1, x2, x3, x4, A1, B1, C1, D1, Den1, Den1m, A2, B2, C2, D2, Den2, Den2m


cdef _get_TermsDeg3_1D_A12(x0,x1,x2,x3,x4):
    A1 = -( (x3-x1)*(x4-x1) + (x2-x0)*(x4-x1) + (x3-x0)*(x2-x0) )
    A2 = (x1-x3)*(x0-x3) + (x2-x4)*(x0-x3) + (x1-x4)*(x2-x4)
    return A1, A2

cdef _get_TermsDeg3_1D_B12(x0,x1,x2,x3,x4):
    B1 = (x2+2.*x0)*(x3-x1)*(x4-x1) + (x3+x0+x1)*(x2-x0)*(x4-x1) + (x4+2.*x1)*(x3-x0)*(x2-x0)
    B2 = -( (x2+2.*x4)*(x1-x3)*(x0-x3) + (x1+x4+x3)*(x2-x4)*(x0-x3) + (x0+2.*x3)*(x1-x4)*(x2-x4) )
    return B1, B2


##################################################
#####     Getting integral operator (linear)
##################################################



def Calc_1D_IntVal_Quad(Coefs=1., cnp.ndarray[cnp.float64_t, ndim=1] Knots=np.linspace(0.,1000,1001), int Deg=2, Deriv=0, str Mode='Vol', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None,
                        thr=1.e-8, thrmode='rel', Abs=True, N=None, Test=True):
    if Test:
        assert type(Coefs) is np.ndarray and Coefs.ndim==2, "Arg Coefs must be 2D array of coefficients !"
        assert Deg>=0 and Deg<=3, "Arg Deg must be a positive int !"
        if type(Deriv) is int:
            assert Deriv<=Deg, "Arg Deriv must be smaller than Deg !"
        elif type(Deriv) is str:
                assert int(Deriv[1])<=Deg and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2'], "Arg Deriv must be smaller than Deg and in ['D0','D1','D2','D3','D0N2','D1N2'] !"
        assert Mode in ['Vol','Surf'], "Arg Mode must be in ['Vol','Surf'] !"
        assert LFunc is None or (type(LFunc) is list and all([hasattr(ff,'__call__') for ff in LFunc])), "Arg LFunc must be a list of callables !"
        assert type(LFunc_Mode) is str, "Arg LFunc_Mode must be a str !" 
        assert quad_pts is None or type(quad_pts) is np.ndarray, "Arg quad_pts must be a np.ndarray !"
        assert quad_w is None or (type(quad_w) is np.ndarray and quad_w.shape==quad_pts.shape), "Arg quad_w must be a np.ndarray of same shape as quad_pts !"
        assert quad_aa is None or (type(quad_aa) is np.ndarray and quad_aa.shape==(quad_w.shape[1],)), "Arg quad_aa must be a np.ndarray of shape (quad_w.shape[1],) !"

    cdef int intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv
    cdef int NKnots = Knots.size
    cdef int NFunc = NKnots-Deg-1
    cdef Py_ssize_t ii, jj, Nt = Coefs.shape[0]
    cdef list Int = [0 for jj in range(0,Nt)]

    LFunc = BSpline_LFunc(Deg, Knots, Deriv=intDeriv, Mode=LFunc_Mode, Test=True)[0] if LFunc is None else LFunc
    assert len(LFunc)==NFunc, "Arg LFunc does not have the proper number of functions !"
    if quad_pts is None:
        quad_pts, quad_w, quad_aa, N = get_IntQuadPts(Knots, Deg, Deriv, intDeriv, Mode=Mode, N=N)

    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        if Mode=='Surf':
            for jj in range(0,Nt):
                Int[jj] = np.sum([Coefs[jj,ii]*np.sum(quad_aa*np.sum(quad_w*LFunc[ii](quad_pts),axis=0)) for ii in range(0,NFunc)])
        elif Mode=='Vol':
            for jj in range(0,Nt):
                Int[jj] = np.sum([Coefs[jj,ii]*np.sum(quad_aa*np.sum(quad_w*quad_pts*LFunc[ii](quad_pts),axis=0)) for ii in range(0,NFunc)])

    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        if Mode=='Surf':
            for jj in range(0,Nt):
                Int[jj] = np.sum(quad_aa*np.sum(quad_w*np.sum([LFunc[ii](quad_pts) for ii in range(0,NFunc)], axis=0)**2, axis=0))
        elif Mode=='Vol':
            for jj in range(0,Nt):
                Int[jj] = np.sum(quad_aa*np.sum(quad_w*quad_pts*np.sum([LFunc[ii](quad_pts) for ii in range(0,NFunc)], axis=0)**2, axis=0))

    elif Deriv=='D1FI':
        lf = BSpline_LFunc(Deg, Knots, Deriv=0, Mode=Mode, Test=Test)[0]
        Span = [Knots[0],Knots[-1]]
        for jj in range(0,Nt):
            grad2 = np.sum([Coefs[jj,ii]*LFunc[ii](quad_pts) for ii in range(0,NFunc)],axis=0)**2
            g = np.sum([Coefs[jj,ii]*lf[ii](quad_pts) for ii in range(0,NFunc)],axis=0)
            g = np.abs(g) if Abs else g
            if thr is not None:
                thr = thr if thrmode=='abs' else np.nanmax(g)*thr
                ind = (quad_pts>=Span[0]) & (quad_pts<=Span[1])
                g[ind & (g<thr)] = thr
            Int[jj] = np.sum(quad_aa*np.sum(quad_w*grad2/g, axis=0)) if Mode=='Surf' else np.sum(quad_aa*np.sum(quad_w*quad_pts*grad2/g, axis=0))

    elif Deriv=='D0ME':
        Span = [Knots[0],Knots[-1]]
        for jj in range(0,Nt):
            g = np.sum([Coefs[jj,ii]*LFunc[ii](quad_pts) for ii in range(0,NFunc)],axis=0)
            g = np.abs(g) if Abs else g
            if thr is not None:
                thr = thr if thrmode=='abs' else np.nanmax(g)*thr
                ind = (quad_pts>=Span[0]) & (quad_pts<=Span[1])
                g[ind & (g<thr)] = thr
            Int[jj] = np.sum(quad_aa*np.sum(quad_w*g*np.log(g), axis=0)) if Mode=='Surf' else np.sum(quad_aa*np.sum(quad_w*quad_pts*g*np.log(g), axis=0))

    if Nt==1:
        Int = Int[0]
    return Int



def Calc_1D_IntVal_Quad_Fast1t_D1FI(cnp.ndarray[cnp.float64_t, ndim=1] Coefs, int NF, list LFunc_quadpts, list LFuncD1_quadpts, indSpan, 
                                    cnp.ndarray[cnp.float64_t, ndim=1] quad_pts, cnp.ndarray[cnp.float64_t, ndim=1] quad_w, cnp.ndarray[cnp.float64_t, ndim=1] quad_aa, str Mode='Vol', thr=1.e-8, thrmode='rel', Abs=True): 
    cdef Py_ssize_t ii
    cdef cnp.ndarray[cnp.float64_t, ndim=2] grad2 = np.sum([Coefs[ii]*LFuncD1_quadpts[ii] for ii in range(0,NF)],axis=0)**2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] g = np.sum([Coefs[ii]*LFunc_quadpts[ii] for ii in range(0,NF)],axis=0)
    if Abs:
        g = np.abs(g)
    if thr is not None:
        thr = thr if thrmode=='abs' else np.nanmax(g)*thr
        g[indSpan & (g<thr)] = thr
    if Mode=='Surf':
        return np.sum(quad_aa*np.sum(quad_w*grad2/g, axis=0))
    else:
        return np.sum(quad_aa*np.sum(quad_w*quad_pts*grad2/g, axis=0))




def Calc_1D_LinIntOp(cnp.ndarray[cnp.float64_t, ndim=1] Knots=np.linspace(0.,1000,1001), int Deg=2, Deriv=0, Method=None, str Mode='Vol', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None,
                     Sparse=True, SpaFormat=None, N=None, Test=True):
    if Test:
        assert Deg>=0 and Deg<=3, "Arg Deg must be a positive int !"
        if type(Deriv) is int:
            assert Deriv<=Deg, "Arg Deriv must be smaller than Deg !"
        elif type(Deriv) is str:
                assert int(Deriv[1])<=Deg and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2'], "Arg Deriv must be smaller than Deg and in ['D0','D1','D2','D3','D0N2','D1N2'] !"
        assert Mode in ['Vol','Surf'], "Arg Mode must be in ['Vol','Surf'] !"
        assert Method is None or Method in ['quad','exact'], "Arg Method must be None or in ['quad','exact'] !"
        assert LFunc is None or (type(LFunc) is list and all([hasattr(ff,'__call__') for ff in LFunc])), "Arg LFunc must be a list of callables !"
        assert type(LFunc_Mode) is str, "Arg LFunc_Mode must be a str !" 
        assert quad_pts is None or type(quad_pts) is np.ndarray, "Arg quad_pts must be a np.ndarray !"
        assert quad_w is None or (type(quad_w) is np.ndarray and quad_w.shape==quad_pts.shape), "Arg quad_w must be a np.ndarray of same shape as quad_pts !"
        assert quad_aa is None or (type(quad_aa) is np.ndarray and quad_aa.shape==(quad_w.shape[1],)), "Arg quad_aa must be a np.ndarray of shape (quad_w.shape[1],) !"
        assert type(Sparse) is bool, "Arg Sparse must be a bool !"
        assert SpaFormat is None or (type(SpaFormat) is str and SpaFormat in ['dia','bsr','coo','csc','csr']), "Arg SpaFormat must be None or in ['dia','bsr','coo','csc','csr'] !"

    cdef int intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv
    cdef int NKnots = Knots.size
    cdef int NFunc = NKnots-Deg-1
    if LFunc is not None:
        assert len(LFunc)==NFunc, "Arg LFunc does not have the proper number of functions !"

    if Method is None:
        Lcases = [Deriv in [0,1,2,3,'D0','D1','D2','D3'], Deriv=='D0N2' and Deg<=2, Deriv=='D1N2' and Deg<=2]
        if any(Lcases):
            Method = 'exact'
        else:
            Method = 'quad'

    m = 0 if Deriv in [0,1,2,3,'D0','D1','D2','D3'] else 1
    if Method=='quad':
        A = _Calc_1D_IntOp_Quad(Knots, Deg, Deriv, intDeriv, NKnots, NFunc, Mode=Mode, LFunc=LFunc, LFunc_Mode=LFunc_Mode, quad_pts=quad_pts, quad_w=quad_w, quad_aa=quad_aa, N=N)
    elif Method=='exact':
        A = _Calc_1D_IntOp_Exact(Knots, Deg, Deriv, intDeriv, NKnots, NFunc, LFunc=LFunc, Mode=Mode)
    if m==1:
        if not Sparse:
            A = A.toarray()
        elif SpaFormat is not None and not SpaFormat=='dia':
            A = eval('scpsp.'+SpaFormat+'_matrix(A)')
    return A, m


cdef _Calc_1D_IntOp_Quad(cnp.ndarray[cnp.float64_t, ndim=1] Knots, int Deg, Deriv, intDeriv, NKnots, NFunc, Mode='Vol', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None, N=None):

    cdef Py_ssize_t ii

    LFunc = BSpline_LFunc(Deg, Knots, Deriv=intDeriv, Mode=LFunc_Mode, Test=True)[0] if LFunc is None else LFunc
    if quad_pts is None:
        quad_pts, quad_w, quad_aa, N = get_IntQuadPts(Knots, Deg, Deriv, intDeriv, Mode=Mode, N=N)
    
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        if Mode=='Surf':
            A = np.asarray([np.sum(quad_aa*np.sum(quad_w*LFunc[ii](quad_pts),axis=0)) for ii in range(0,NFunc)])
        else:
            A = np.asarray([np.sum(quad_aa*np.sum(quad_w*quad_pts*LFunc[ii](quad_pts),axis=0)) for ii in range(0,NFunc)])

    elif Deriv in ['D0N2','D1N2','D2N2','D3N2']:
        Ld, Lo = [], []
        if Mode=='Surf':
            d0 = np.asarray([np.sum(quad_aa*np.sum(quad_w*(LFunc[ii](quad_pts))**2,axis=0)) for ii in range(0,NFunc)])
            Ld, Lo = Ld+[d0], Lo+[0]
            if Deg>=1:
                d1 = np.asarray([np.sum(quad_aa[1:-1]*np.sum(quad_w[:,1:-1]*LFunc[ii](quad_pts[:,1:-1])*LFunc[ii+1](quad_pts[:,1:-1]),axis=0)) for ii in range(0,NFunc-1)])
                Ld, Lo = Ld+[d1,d1], Lo+[-1,1]
                if Deg>=2:
                    d2 = np.asarray([np.sum(quad_aa[2:-2]*np.sum(quad_w[:,2:-2]*LFunc[ii](quad_pts[:,2:-2])*LFunc[ii+2](quad_pts[:,2:-2]),axis=0)) for ii in range(0,NFunc-2)])
                    Ld, Lo = Ld+[d2,d2], Lo+[-2,2]
                    if Deg>=3:
                        d3 = np.asarray([np.sum(quad_aa[3:-3]*np.sum(quad_w[:,3:-3]*LFunc[ii](quad_pts[:,3:-3])*LFunc[ii+3](quad_pts[:,3:-3]),axis=0)) for ii in range(0,NFunc-3)])
                        Ld, Lo = Ld+[d3,d3], Lo+[-3,3] 
        else:
            d0 = np.asarray([np.sum(quad_aa*np.sum(quad_w*quad_pts*(LFunc[ii](quad_pts))**2,axis=0)) for ii in range(0,NFunc)])
            Ld, Lo = Ld+[d0], Lo+[0]
            if Deg>=1:
                d1 = np.asarray([np.sum(quad_aa[1:-1]*np.sum(quad_w[:,1:-1]*quad_pts[:,1:-1]*LFunc[ii](quad_pts[:,1:-1])*LFunc[ii+1](quad_pts[:,1:-1]),axis=0)) for ii in range(0,NFunc-1)])
                Ld, Lo = Ld+[d1,d1], Lo+[-1,1]
                if Deg>=2:
                    d2 = np.asarray([np.sum(quad_aa[2:-2]*np.sum(quad_w[:,2:-2]*quad_pts[:,2:-2]*LFunc[ii](quad_pts[:,2:-2])*LFunc[ii+2](quad_pts[:,2:-2]),axis=0)) for ii in range(0,NFunc-2)])
                    Ld, Lo = Ld+[d2,d2], Lo+[-2,2]
                    if Deg>=3:
                        d3 = np.asarray([np.sum(quad_aa[3:-3]*np.sum(quad_w[:,3:-3]*quad_pts[:,3:-3]*LFunc[ii](quad_pts[:,3:-3])*LFunc[ii+3](quad_pts[:,3:-3]),axis=0)) for ii in range(0,NFunc-3)])
                        Ld, Lo = Ld+[d3,d3], Lo+[-3,3]

        A = scpsp.diags(Ld, Lo, shape=(NFunc,NFunc))
       
    return A


cdef _Calc_1D_IntOp_Exact(cnp.ndarray[cnp.float64_t, ndim=1] Knots, int Deg, Deriv, intDeriv, NKnots, NFunc, LFunc=None, Mode='Vol'):

    if Deg==0:
        x0, x1 = Knots[:-1], Knots[1:]
    elif Deg==1:
        x0, x1, x2 = Knots[:-2], Knots[1:-1], Knots[2:]
        if Deriv in ['D0N2','D1N2','D2N2','D3N2']:
            x1d, x2d = Knots[1:-2], Knots[2:-1]
    elif Deg==2:
        x0, x1, x2, x3 = Knots[:-3], Knots[1:-2], Knots[2:-1], Knots[3:]
        if Deriv in ['D0N2','D1N2','D2N2','D3N2']:
            x0d1, x1d1, x2d1, x3d1, x4d1 = Knots[:-4], Knots[1:-3], Knots[2:-2], Knots[3:-1], Knots[4:]
            x1d2, x2d2, x3d2, x4d2 = Knots[1:-4], Knots[2:-3], Knots[3:-2], Knots[4:-1]
    elif Deg==3:
        x0, x1, x2, x3, x4, A1, B1, C1, D1, Den1, Den1m, A2, B2, C2, D2, Den2, Den2m = _get_TermsDeg3_1D(Knots)
        if Deriv in ['D0N2','D1N2','D2N2','D3N2']:
            x0d1, x1d1, x2d1, x3d1, x4d1, x5d1 = Knots[:-5], Knots[1:-4], Knots[2:-3], Knots[3:-2], Knots[4:-1], Knots[5:]
            A1d1, A2d1 = _get_TermsDeg3_1D_A12(x0d1,x1d1,x2d1,x3d1,x4d1)
            B1d1, B2d1 = _get_TermsDeg3_1D_B12(x0d1,x1d1,x2d1,x3d1,x4d1)
            A1_1d1, A2_1d1 = _get_TermsDeg3_1D_A12(x1d1,x2d1,x3d1,x4d1,x5d1)
            B1_1d1, B2_1d1 = _get_TermsDeg3_1D_B12(x1d1,x2d1,x3d1,x4d1,x5d1)

            x0d2, x1d2, x2d2, x3d2, x4d2, x5d2, x6d2 = Knots[:-6], Knots[1:-5], Knots[2:-4], Knots[3:-3], Knots[4:-2], Knots[5:-1], Knots[6:]
            A2d2 = _get_TermsDeg3_1D_A12(x0d2,x1d2,x2d2,x3d2,x4d2)[1]
            B2d2 = _get_TermsDeg3_1D_B12(x0d2,x1d2,x2d2,x3d2,x4d2)[1]
            A1_2d2 = _get_TermsDeg3_1D_A12(x2d2,x3d2,x4d2,x5d2,x6d2)[0]
            B1_2d2 = _get_TermsDeg3_1D_B12(x2d2,x3d2,x4d2,x5d2,x6d2)[0]

            x1d3, x2d3, x3d3, x4d3, x5d3, x6d3 = Knots[1:-6], Knots[2:-5], Knots[3:-4], Knots[4:-3], Knots[5:-2], Knots[6:-1]

    
    # Integral of the function or of its derivatives
    if Deriv in [0,1,2,3,'D0','D1','D2','D3']:
        if intDeriv==0:
            if Deg==0:
                x0, x1 = Knots[:-1], Knots[1:]
                A = x1-x0 if Mode=='Surf' else (x1**2-x0**2)/2.

            elif Deg==1:
                A = (x2-x0)/2. if Mode=='Surf' else (x2**2 + x1*(x2-x0) - x0**2)/6.

            elif Deg==2:
                if Mode=='Surf':
                    I1 = (x1 - x0)**2 / (3.*(x2-x0))
                    I2 = ( x2**2 + x1*x2 - 2.*x1**2 + 3.*x0*(x1-x2) ) / (6.*(x2-x0))  +  ( -2.*x2**2 + x1*x2 + x1**2 + 3.*x3*(x2-x1) ) / (6.*(x3-x1))
                    I3 = (x3 - x2)**2 / (3.*(x3-x1))
                else:
                    I1 = (3.*x1+x0)*(x1-x0)**2 / (12.*(x2-x0))
                    I2 = (x2-x1)*(x2**2+2.*x1*x2+3.*x1**2 - 2.*x0*(x2+2.*x1)) / (12.*(x2-x0)) - (x2-x1)*(x1**2+2.*x1*x2+3.*x2**2 - 2.*x3*(2.*x2+x1)) / (12.*(x3-x1))
                    I3 = (3.*x2+x3)*(x3-x2)**2 / (12.*(x3-x1))
                A = I1 + I2 + I3

            elif Deg==3:
                if Mode=='Surf':
                    I1 = ( (x1**4-x0**4)/4. - 3.*x0*(x1**3-x0**3)/3. + 3.*x0**2*(x1**2-x0**2)/2. - (x1-x0)*x0**3 ) / ((x3-x0)*(x2-x0)*(x1-x0))
                    #I2 = (A1*(x2**4-x1**4)/4. + B1*(x2**3-x1**3)/3. + C1*(x2**2-x1**2)/2. + D1*(x2-x1)) / Den1
                    #I3 = (A2*(x3**4-x2**4)/4. + B2*(x3**3-x2**3)/3. + C2*(x3**2-x2**2)/2. + D2*(x3-x2)) / Den2
                    I2 = (A1*(x2**3+x2**2*x1+x2*x1**2+x1**3)/4. + B1*(x2**2+x2*x1+x1**2)/3. + C1*(x2+x1)/2. + D1) / Den1m
                    I3 = (A2*(x3**3+x3**2*x2+x3*x2**2+x2**3)/4. + B2*(x3**2+x3*x2+x2**2)/3. + C2*(x3+x2)/2. + D2) / Den2m
                    I4 = ( -(x4**4-x3**4)/4. + 3.*x4*(x4**3-x3**3)/3. - 3.*x4**2*(x4**2-x3**2)/2. + (x4-x3)*x4**3 ) / ((x4-x3)*(x4-x2)*(x4-x1))
                else:
                    I1 = ( 12.*(x1**5-x0**5) - 45.*x0*(x1**4-x0**4) + 60.*x0**2*(x1**3-x0**3) - 30.*x0**3*(x1**2-x0**2) ) / (60.*(x3-x0)*(x2-x0)*(x1-x0))
                    #I2 = ( 12.*A1*(x2**5-x1**5) + 15.*B1*(x2**4-x1**4) + 20.*C1*(x2**3-x1**3) + 30.*D1*(x2**2-x1**2) ) / (60.*Den1)
                    #I3 = ( 12.*A2*(x3**5-x2**5) + 15.*B2*(x3**4-x2**4) + 20.*C2*(x3**3-x2**3) + 30.*D2*(x3**2-x2**2) ) / (60.*Den2)
                    I2 = ( 12.*A1*(x2**4+x2**3*x1+x2**2*x1**2+x2*x1**3+x1**4) + 15.*B1*(x2**3+x2**2*x1+x2*x1**2+x1**3) + 20.*C1*(x2**2+x2*x1+x1**2) + 30.*D1*(x2+x1) ) / (60.*Den1m)
                    I3 = ( 12.*A2*(x3**4+x3**3*x2+x3**2*x2**2+x3*x2**3+x2**4) + 15.*B2*(x3**3+x3**2*x2+x3*x2**2+x2**3) + 20.*C2*(x3**2+x3*x2+x2**2) + 30.*D2*(x3+x2) ) / (60.*Den2m)
                    I4 = -( 12.*(x4**5-x3**5) - 45.*x4*(x4**4-x3**4) + 60.*x4**2*(x4**3-x3**3) - 30.*x4**3*(x4**2-x3**2) ) / (60.*(x4-x3)*(x4-x2)*(x4-x1))
                A = I1 + I2 + I3 + I4

        elif intDeriv==1:
            if Deg==1:
                if Mode=='Surf':
                    A = np.zeros((NFunc,))
                else:
                    A = -(x2-x0)/2.

            elif Deg==2:
                if Mode=='Surf':
                    I1 = (x1-x0)/(x2-x0)
                    I2 = ( x3*x2**2 - x2**3 + x2**2*x1 + x2**2*x0 - 2.*x2*x1*x0 + x3*x1**2 + x2*x1**2 - x1**3 - 2.*x3*x2*x1 + x1**2*x0 ) / ((x2-x1)*(x2-x0)*(x3-x1))
                    I3 = -(x3-x2)/(x3-x1)
                else:
                    I1 = (2.*x1**2 - x1*x0 - x0**2) / (3.*(x2-x0))
                    I2 = (-x2**2 - x2*x1 - 4.*x1**2 + 3.*x0*x2 + 3.*x0*x1) / (6.*(x2-x0))   +   (-4.*x2**2 - x2*x1 - x1**2 + 3.*x3*x2 + 3.*x3*x1) / (6.*(x3-x1))
                    I3 = (-x3**2 - x3*x2 + 2.*x2**2) / (3.*(x3-x1))
                A = I1 + I2 + I3

            elif Deg==3:
                if Mode=='Surf':
                    I1 = ((x1**3-x0**3) - 3.*x0*(x1**2-x0**2) + 3.*x0**2*(x1-x0)) / ((x3-x0)*(x2-x0)*(x1-x0))
                    #I2 = ( A1*(x2**3-x1**3) + B1*(x2**2-x1**2) + C1*(x2-x1) ) / Den1
                    #I3 = ( A2*(x3**3-x2**3) + B2*(x3**2-x2**2) + C2*(x3-x2) ) / Den2
                    I2 = ( A1*(x2**2+x2*x1+x1**2) + B1*(x2+x1) + C1 ) / Den1m
                    I3 = ( A2*(x3**2+x3*x2+x2**2) + B2*(x3+x2) + C2 ) / Den2m
                    I4 = ( -(x4**3-x3**3) + 3.*x4*(x4**2-x3**2) - 3.*x4**2*(x4-x3)) / ((x4-x3)*(x4-x2)*(x4-x1))
                else:
                    I1 = ( 3.*x1**3 - 5.*x1**2*x0 + x1*x0**2 + x0**3 ) / (4.*(x3-x0)*(x2-x0))
                    #I2 = ( 9.*A1*(x2**4-x1**4) + 8.*B1*(x2**3-x1**3) + 6.*C1*(x2**2-x1**2) ) / (12.*Den1)
                    #I3 = ( 9.*A2*(x3**4-x2**4) + 8.*B2*(x3**3-x2**3) + 6.*C2*(x3**2-x2**2) ) / (12.*Den2)
                    I2 = ( 9.*A1*(x2**3+x2**2*x1+x2*x1**2+x1**3) + 8.*B1*(x2**2+x2*x1+x1**2) + 6.*C1*(x2+x1) ) / (12.*Den1m)
                    I3 = ( 9.*A2*(x3**3+x3**2*x2+x2*x3**2+x2**3) + 8.*B2*(x3**2+x3*x2+x2**2) + 6.*C2*(x3+x2) ) / (12.*Den2m)
                    I4 = ( -x4**3 + 5.*x4*x3**2 - x4**2*x3 - 3.*x3**3 ) / (4.*(x4-x2)*(x4-x1))
                A = I1 + I2 + I3 + I4

        elif intDeriv==2:
            if Deg==2:
                if Mode=='Surf':
                    I1 = 2./(x2-x0)
                    I2 = -2.*(x3+x2-x1-x0) / ((x2-x0)*(x3-x1))
                    I3 = 2./(x3-x1)
                else:
                    I1 = (x1+x0) / (x2-x0)
                    I2 = - (x2+x1) / (x2-x0) - (x2+x1) / (x3-x1)
                    I3 = (x3+x2) / (x3-x1)
                A = I1 + I2 + I3

            elif Deg==3:
                if Mode=='Surf':
                    I1 = 3.*(x1-x0) / ((x3-x0)*(x2-x0))
                    #I2 = ( 3.*A1*(x2**2-x1**2) + 2.*B1*(x2-x1) ) / Den1
                    #I3 = ( 3.*A2*(x3**2-x2**2) + 2.*B2*(x3-x2) ) / Den2
                    I2 = ( 3.*A1*(x2+x1) + 2.*B1 ) / Den1m
                    I3 = ( 3.*A2*(x3+x2) + 2.*B2 ) / Den2m
                    I4 = 3.*(x4-x3) / ((x4-x2)*(x4-x1))
                else:
                    I1 = ( 2.*x1**2 - x1*x0 - x0**2 ) / ((x3-x0)*(x2-x0))
                    #I2 = ( 2.*A1*(x2**3-x1**3) + B1*(x2**2-x1**2) ) / Den1
                    #I3 = ( 2.*A2*(x3**3-x2**3) + B2*(x3**2-x2**2) ) / Den2
                    I2 = ( 2.*A1*(x2**2+x2*x1+x1**2) + B1*(x2+x1) ) / Den1m
                    I3 = ( 2.*A2*(x3**2+x3*x2+x2**2) + B2*(x3+x2) ) / Den2m
                    I4 = ( x4**2 + x4*x3 - 2.*x3**2 ) / ((x4-x2)*(x4-x1))
                A = I1 + I2 + I3 + I4

        elif intDeriv==3:
            if Deg==3:
                if Mode=='Surf':
                    I1 = 6./((x3-x0)*(x2-x0))
                    #I2 = 6.*A1*(x2-x1) / Den1
                    #I3 = 6.*A2*(x3-x2) / Den2
                    I2 = 6.*A1 / Den1m
                    I3 = 6.*A2 / Den2m
                    I4 = -6/((x4-x2)*(x4-x1))
                else:
                    I1 = 3.*(x1+x0) / ((x3-x0)*(x2-x0))
                    #I2 = 3.*A1*(x2**2-x1**2) / Den1
                    #I3 = 3.*A2*(x3**2-x2**2) / Den2
                    I2 = 3.*A1*(x2+x1) / Den1m
                    I3 = 3.*A2*(x3+x2) / Den2m
                    I4 = -3.*(x4+x3) / ((x4-x2)*(x4-x1))
                A = I1 + I2 + I3 + I4



    # Integral of the squared norm of the function or of its derivatives
    elif Deriv == 'D0N2':
        if Deg==0:
            if Mode=='Surf':
                d0 = x1-x0
            else:
                d0 = (x1**2-x0**2)/2.
            A = scpsp.diags([d0], [0], shape=(NFunc,NFunc))

        elif Deg==1:
            if Mode=='Surf':
                d0 = (x2-x0)/3.
                d1 = (x2d-x1d)/6.

            else:
                I1 = (3.*x1**3 - 5.*x1**2*x0 + x1*x0**2 + x0**3) / (12.*(x1-x0))
                I2 = ( x2**3 + x2**2*x1 - 5.*x2*x1**2 + 3.*x1**3 ) / (12.*(x2-x1))
                d0 = I1 + I2
                d1 = (x2d**2-x1d**2)/12.
            A = scpsp.diags([d0,d1,d1], [0,-1,1], shape=(NFunc,NFunc))

        elif Deg==2:
            if Mode=='Surf':
                I1 = (x1-x0)**3 / (5.*(x2-x0)**2)
                I2_1 = (x2-x1)*( x2**2 + 3.*x2*x1 + 6.*x1**2 - 5.*x0*(x2+3.*x1) + 10.*x0**2 ) / (30.*(x2-x0)**2)
                I2_2 = (x2-x1)*( -3.*x2**2 - 4.*x2*x1 - 3.*x1**2 + 5.*(x3+x0)*(x2+x1) - 10.*x3*x0 ) / (30.*(x2-x0)*(x3-x1))
                I2_3 = (x2-x1)*( 6.*x2**2 + 3.*x2*x1 + x1**2 - 5.*x3*(3.*x2+x1) + 10.*x3**2 ) / (30.*(x3-x1)**2)
                I2 = I2_1 + I2_2 + I2_3
                I3 = (x3-x2)**3 / (5.*(x3-x1)**2)
                d0 = I1 + I2 + I3
                
                Id1_1 = (3.*x2d1+2.*x1d1-5.*x0d1)*(x2d1-x1d1)**2 / (60.*(x3d1-x1d1)*(x2d1-x0d1))  +  (5.*x3d1-4.*x2d1-x1d1)*(x2d1-x1d1)**2 / (20.*(x3d1-x1d1)**2)
                Id1_2 = (4.*x2d1+x3d1-5.*x1d1)*(x3d1-x2d1)**2 / (20.*(x3d1-x1d1)**2)  +  (5.*x4d1-2.*x3d1-3.*x2d1)*(x3d1-x2d1)**2 / (60.*(x4d1-x2d1)*(x3d1-x1d1))
                d1 = Id1_1 + Id1_2
                d2 = (x3d2-x2d2)**3 / (30.*(x4d2-x2d2)*(x3d2-x1d2))
            else:
                I1 = (5.*x1+x0)*(x1-x0)**3 / (30.*(x2-x0)**2)
                I2_1 = (x2-x1) * ( x2**3 + 3.*x2**2*x1 + 6.*x2*x1**2 + 10.*x1**3  -  4.*x0*(x2**2+3.*x2*x1+6.*x1**2)  +  5.*x0**2*(x2+3.*x1) ) / (60.*(x2-x0)**2)
                I2_2 = (x2-x1) * ( -2.*x2**3 - 3.*x2**2*x1 - 3.*x2*x1**2 - 2.*x1**3  +  (x3+x0)*(3.*x2**2+4.*x2*x1+3.*x1**2)  -  5.*x3*x0*(x2+x1) ) / (30.*(x2-x0)*(x3-x1))
                I2_3 = (x2-x1) * ( 10.*x2**3 + 6.*x2**2*x1 + 3.*x2*x1**2 + x1**3  -  4.*x3*(6.*x2**2+3.*x2*x1+x1**2)  +  5.*x3**2*(3.*x2+x1) ) / (60.*(x3-x1)**2)
                I2 = I2_1 + I2_2 + I2_3
                I3 = (x3+5.*x2)*(x3-x2)**2 / (30.*(x3-x1)**2)
                d0 = I1 + I2 + I3

                Id1_1 = (x2d1-x1d1)**2 * ( 2.*x2d1**2+2.*x2d1*x1d1+x1d1**2 - x0d1*(3.*x2d1+2.*x1d1) ) / (60.*(x3d1-x1d1)*(x2d1-x0d1)) \
                        + (x2d1-x1d1)**2 * ( -10.*x2d1**2-4.*x2d1*x1d1-x1d1**2 + 3.*x3d1*(4.*x2d1+x1d1) ) / (60.*(x3d1-x1d1)**2)
                Id1_2 = (x3d1-x2d1)**2 * ( x3d1**2+4.*x3d1*x2d1+10.*x2d1**2 - 3.*x1d1*(x3d1+4.*x2d1) ) / (60.*(x3d1-x1d1)**2) \
                        + (x3d1-x2d1)**2 * ( -x3d1**2-2.*x3d1*x2d1-2.*x2d1**2 + x4d1*(2.*x3d1+3.*x2d1) ) / (60.*(x4d1-x2d1)*(x3d1-x1d1))
                d1 = Id1_1 + Id1_2
                d2 = (x3d2+x2d2)*(x3d2-x2d2)**3 / (60.*(x4d2-x2d2)*(x3d2-x1d2))

            A = scpsp.diags([d0,d1,d1,d2,d2], [0,-1,1,-2,2], shape=(NFunc,NFunc))

        elif Deg==3:
            assert not Deg==3, "Exact expressions have not been coded for D0N2 of Deg=3 yet (in process, to be finished) !"
            if Mode=='Surf':
                I1 = (x1-x0)**5 / (7.*(x3-x0)**2*(x2-x0)**2)     # Ok
                I2 = (_D0N2_Deg3_Surf(x2, A1,B1,C1,D1) - _D0N2_Deg3_Surf(x1, A1,B1,C1,D1)) / (210.*Den1**2)      # Wrong or not accurate enough => Needs simplifying ?
                I3 = (_D0N2_Deg3_Surf(x3, A2,B2,C2,D2) - _D0N2_Deg3_Surf(x2, A2,B2,C2,D2)) / (210.*Den2**2)      # Wrong or not accurate enough => Needs simplifying ?
                I4 = (x4-x3)**5 / (7.*(x4-x2)**2*(x4-x1)**2)     # Ok
                d0 = I1 + I2 + I3 + I4
                
                Id1_1 = 0
                Id1_2 = 0
                Id1_3 = 0
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 0
                Id2_2 = 0
                d2 = Id2_1 + Id2_2
        
                d3 = 0

            else:
                I1 = (x1-x0)**5 / (7.*(x3-x0)**2*(x2-x0)**2)
                I2 = 0
                I3 = 0
                I4 = (x4-x3)**5 / (7.*(x4-x2)**2*(x4-x1)**2)
                d0 = I1 + I2 + I3 + I4
                
                Id1_1 = 0
                Id1_2 = 0
                Id1_3 = 0
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 0
                Id2_2 = 0
                d2 = Id2_1 + Id2_2
        
                d3 = 0
                
            A = scpsp.diags([d0,d1,d1,d2,d2,d3,d3], [0,-1,1,-2,2,-3,3], shape=(NFunc,NFunc))


    elif Deriv == 'D1N2':
        if Deg==1:
            if Mode=='Surf':
                d0 = 1./(x1-x0) + 1./(x2-x1)
                d1 = -1./(x2d-x1d)
            else:
                d0 = (x1+x0)/(2.*(x1-x0)) + (x2+x1)/(2.*(x2-x1))
                d1 = -(x2d+x1d)/(2.*(x2d-x1d))
            A = scpsp.diags([d0,d1,d1], [0,-1,1], shape=(NFunc,NFunc))

        elif Deg==2:
            if Mode=='Surf':
                I1 = 4.*(x1-x0) / (3.*(x2-x0)**2)
                I2 = 4.*(x2-x1) * ( x2**2 + x2*x1 + x1**2 + x3**2 + x3*x0 + x0**2 - x3*(x2+2.*x1) - x0*(2.*x2+x1) ) / (3.*(x3-x1)**2*(x2-x0)**2)
                I3 = 4.*(x3-x2) / (3.*(x3-x1)**2)
                d0 = I1 + I2 + I3

                Id1_1 = 2.*(x2d1-x1d1)*( x3d1 - 2.*x2d1 - x1d1 + 2.*x0d1 ) / (3.*(x3d1-x1d1)**2*(x2d1-x0d1))
                Id1_2 = 2.*(x3d1-x2d1)*( -2.*x4d1 + x3d1 + 2.*x2d1 - x1d1 ) / (3.*(x4d1-x2d1)*(x3d1-x1d1)**2)
                d1 = Id1_1 + Id1_2
                d2 = -2.*(x3d2-x2d2) / (3.*(x4d2-x2d2)*(x3d2-x1d2))
            else:
                I1 = (3.*x1+x0)*(x1-x0) / (3.*(x2-x0)**2)
                I2 = (x2-x1) * ( 3.*(x2+x1)*(x2**2+x1**2) + x3**2*(x2+3.*x1) + x0**2*(3.*x2+x1) - 2.*x3*(x2**2+2.*x1*x2+3.*x1**2) -2.*x0*(3.*x2**2+2.*x2*x1+x1**2) + 2.*x3*x0*(x2+x1) ) / (3.*(x3-x1)**2*(x2-x0)**2)
                I3 = (x3+3.*x2)*(x3-x2) / (3.*(x3-x1)**2)
                d0 = I1 + I2 + I3

                Id1_1 = (x2d1-x1d1) * ( -(3.*x2d1**2 + 2.*x2d1*x1d1 + x1d1**2) + x3d1*(x2d1+x1d1) + x0d1*(3.*x2d1+x1d1) ) / (3.*(x3d1-x1d1)**2*(x2d1-x0d1))
                Id1_2 = (x3d1-x2d1) * ( x3d1**2 + 2.*x3d1*x2d1 + 3.*x2d1**2 - x4d1*(x3d1+3.*x2d1) - x1d1*(x3d1+x2d1) ) / (3.*(x4d1-x2d1)*(x3d1-x1d1)**2)
                d1 = Id1_1 + Id1_2
                d2 = -(x3d2+x2d2)*(x3d2-x2d2) / (3.*(x4d2-x2d2)*(x3d2-x1d2))
            A = scpsp.diags([d0,d1,d1,d2,d2], [0,-1,1,-2,2], shape=(NFunc,NFunc))

        elif Deg==3:
            assert not Deg==3, "Exact expressions have not been coded for D1N2 of Deg=3 yet (in process, to be finished) !"
            if Mode=='Surf':
                I1 = 9.*(x1-x0)**3 / (5.*(x3-x0)**2*(x2-x0)**2)  # Ok
                I2 = (_D1N2_Deg3_Surf(x2, A1, B1, C1) - _D1N2_Deg3_Surf(x1, A1, B1, C1)) / (15.*Den1**2)  # Ok but simpify ?
                I3 = (_D1N2_Deg3_Surf(x3, A2, B2, C2) - _D1N2_Deg3_Surf(x2, A2, B2, C2)) / (15.*Den2**2)  # Ok but simpify ?
                I4 = 9.*(x4-x3)**3 / (5.*(x4-x2)**2*(x4-x1)**2)  # Ok
                d0 = I1 + I2 + I3 + I4

                Id1_1 = 0
                Id1_2 = 0
                Id1_3 = 0
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 0
                Id2_2 = 0
                d2 = Id2_1 + Id2_2

                d3 = -3.*(x4d3-x3d3)**3 / (10.*(x6d3-x3d3)*(x5d3-x3d3)*(x4d3-x2d3)*(x4d3-x1d3))  # Ok

            else:
                I1 = 0
                I2 = 0
                I3 = 0
                I4 = 0
                d0 = I1 + I2 + I3 + I4

                Id1_1 = 0
                Id1_2 = 0
                Id1_3 = 0
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 0
                Id2_2 = 0
                d2 = Id2_1 + Id2_2

                d3 = 0

            A = scpsp.diags([d0,d1,d1,d2,d2,d3,d3], [0,-1,1,-2,2,-3,3], shape=(NFunc,NFunc))

    elif Deriv == 'D2N2':
        if Deg==2:
            if Mode=='Surf':
                I1 = 4. / ((x2-x0)**2*(x1-x0))
                I2 = 4.*(x3+x2-x1-x0)**2 / ((x3-x1)**2*(x2-x1)*(x2-x0)**2)
                I3 = 4. / ((x3-x2)*(x3-x1)**2)
                d0 = I1 + I2 + I3

                Id1_1 = -4.*(x3d1+x2d1-x1d1-x0d1) / ((x3d1-x1d1)**2*(x2d1-x1d1)*(x2d1-x0d1))
                Id1_2 = -4.*(x4d1+x3d1-x2d1-x1d1) / ((x4d1-x2d1)*(x3d1-x2d1)*(x3d1-x1d1)**2)
                d1 = Id1_1 + Id1_2

                d2 = 4. / ((x4d2-x2d2)*(x3d2-x2d2)*(x3d2-x1d2))
            else:
                I1 = 2.*(x1+x0) / ((x2-x0)**2*(x1-x0))
                I2 = 2.*(x3+x2-x1-x0)**2*(x2+x1) / ((x3-x1)**2*(x2-x1)*(x2-x0)**2)
                I3 = 2.*(x3+x2) / ((x3-x2)*(x3-x1)**2)
                d0 = I1 + I2 + I3

                Id1_1 = -2.*(x3d1+x2d1-x1d1-x0d1)*(x2d1+x1d1) / ((x3d1-x1d1)**2*(x2d1-x1d1)*(x2d1-x0d1))
                Id1_2 = -2.*(x4d1+x3d1-x2d1-x1d1)*(x3d1+x2d1) / ((x4d1-x2d1)*(x3d1-x2d1)*(x3d1-x1d1)**2)
                d1 = Id1_1 + Id1_2

                d2 = 2.*(x3d2+x2d2) / ((x4d2-x2d2)*(x3d2-x2d2)*(x3d2-x1d2))
            A = scpsp.diags([d0,d1,d1,d2,d2], [0,-1,1,-2,2], shape=(NFunc,NFunc))

        elif Deg==3:
            if Mode=='Surf':
                I1 = 12.*(x1-x0) / ((x3-x0)**2*(x2-x0)**2)
                I2 = (12.*A1**2*(x2**2+x2*x1+x1**2) + 12.*A1*B1*(x2+x1) + 4.*B1**2) / (Den1m**2*(x2-x1))
                I3 = (12.*A2**2*(x3**2+x3*x2+x2**2) + 12.*A2*B2*(x3+x2) + 4.*B2**2) / (Den2m**2*(x3-x2))
                I4 = 12.*(x4-x3) / ((x4-x2)**2*(x4-x1)**2)
                d0 = I1 + I2 + I3 + I4

                Id1_1 = 6.*( A1d1*(2.*x2d1+x1d1) + B1d1 ) / ((x4d1-x1d1)**2*(x3d1-x1d1)**2*(x3d1-x0d1)*(x2d1-x0d1))
                Id1_2 = 2.*( 6.*A2d1*A1_1d1*(x3d1**2+x3d1*x2d1+x2d1**2) + 3.*(A1_1d1*B2d1 + A2d1*B1_1d1)*(x3d1+x2d1) + 2.*B2d1*B1_1d1 ) / ((x5d1-x2d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2*(x3d1-x2d1)*(x3d1-x1d1)**2*(x3d1-x0d1))
                Id1_3 = 6.*( A2_1d1*(x4d1+2.*x3d1) + B2_1d1 ) / ((x5d1-x3d1)*(x5d1-x2d1)*(x4d1-x3d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2)
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 6.*( A2d2*(2.*x3d2 + x2d2) + B2d2 ) / ((x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2)*(x3d2-x1d2)*(x3d2-x0d2))
                Id2_2 = 6.*( A1_2d2*(x4d2+2.*x3d2) + B1_2d2 ) / ((x6d2-x3d2)*(x5d2-x3d2)*(x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2))
                d2 = Id2_1 + Id2_2

                d3 = 6.*(x4d3-x3d3) / ((x6d3-x3d3)*(x5d3-x3d3)*(x4d3-x2d3)*(x4d3-x1d3))

            else:
                I1 = 3.*(3.*x1+x0)*(x1-x0) / ((x3-x0)**2*(x2-x0)**2)
                I2 = ( 9.*A1**2*(x2**3+x2**2*x1+x2*x1**2+x1**3) + 8.*A1*B1*(x2**2+x2*x1+x1**2) + 2.*B1**2*(x2+x1) ) / ((x4-x1)**2*(x3-x1)**2*(x3-x0)**2*(x2-x1)*(x2-x0)**2)
                I3 = ( 9.*A2**2*(x3**3+x3**2*x2+x3*x2**2+x2**3) + 8.*A2*B2*(x3**2+x3*x2+x2**2) + 2.*B2**2*(x3+x2) ) / ((x4-x2)**2*(x4-x1)**2*(x3-x2)*(x3-x1)**2*(x3-x0)**2)
                I4 = 3.*(x4+3.*x3)*(x4-x3) / ((x4-x2)**2*(x4-x1)**2)
                d0 = I1 + I2 + I3 + I4

                Id1_1 = ( 3.*A1d1*(3.*x2d1**2+2.*x2d1*x1d1+x1d1**2) + 2.*B1d1*(2.*x2d1+x1d1) ) / ((x4d1-x1d1)**2*(x3d1-x1d1)**2*(x3d1-x0d1)*(x2d1-x0d1))
                Id1_2 = ( 9.*A2d1*A1_1d1*(x3d1**3+x3d1**2*x2d1+x3d1*x2d1**2+x2d1**3) + 4.*(A1_1d1*B2d1 + A2d1*B1_1d1)*(x3d1**2+x3d1*x2d1+x2d1**2) + 2.*B2d1*B1_1d1*(x3d1+x2d1) ) / ((x5d1-x2d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2*(x3d1-x2d1)*(x3d1-x1d1)**2*(x3d1-x0d1))
                Id1_3 = ( 3.*A2_1d1*(x4d1**2+2.*x4d1*x3d1+3.*x3d1**2) + 2.*B2_1d1*(x4d1+2.*x3d1) ) / ((x5d1-x3d1)*(x5d1-x2d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2)
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = ( 3.*A2d2*(3.*x3d2**2+2.*x3d2*x2d2+x2d2**2) + 2.*B2d2*(2.*x3d2+x2d2) ) / ((x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2)*(x3d2-x1d2)*(x3d2-x0d2))
                Id2_2 = ( 3.*A1_2d2*(x4d2**2+2.*x4d2*x3d2+3.*x3d2**2) + 2.*B1_2d2*(x4d2+2.*x3d2) ) / ((x6d2-x3d2)*(x5d2-x3d2)*(x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2))
                d2 = Id2_1 + Id2_2

                d3 = 3.*(x4d3**2-x3d3**2) / ((x6d3-x3d3)*(x5d3-x3d3)*(x4d3-x2d3)*(x4d3-x1d3))

            A = scpsp.diags([d0,d1,d1,d2,d2,d3,d3], [0,-1,1,-2,2,-3,3], shape=(NFunc,NFunc))
        
    elif Deriv == 'D3N2':
        if Deg==3:
            if Mode=='Surf':
                I1 = 36. / ((x3-x0)**2*(x2-x0)**2*(x1-x0))
                I2 = 36.*A1**2 / ((x4-x1)**2*(x3-x1)**2*(x3-x0)**2*(x2-x1)*(x2-x0)**2)
                I3 = 36.*A2**2 / ((x4-x2)**2*(x4-x1)**2*(x3-x2)*(x3-x1)**2*(x3-x0)**2)
                I4 = 36./((x4-x3)*(x4-x2)**2*(x4-x1)**2)
                d0 = I1 + I2 + I3 + I4

                Id1_1 = 36.*A1d1 / ((x4d1-x1d1)**2 *(x3d1-x1d1)**2*(x3d1-x0d1)*(x2d1-x1d1)**2*(x2d1-x0d1))
                Id1_2 = 36.*A2d1*A1_1d1 / ((x5d1-x2d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2*(x3d1-x2d1)*(x3d1-x1d1)**2*(x3d1-x0d1))
                Id1_3 = -36.*A2_1d1 / ((x5d1-x3d1)*(x5d1-x2d1)*(x4d1-x3d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2)
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 36.*A2d2 / ((x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2)*(x3d2-x2d2)**2*(x3d2-x1d2)*(x3d2-x0d2))
                Id2_2 = -36.*A1_2d2 / ((x6d2-x3d2)*(x5d2-x3d2)*(x5d2-x2d2)*(x4d2-x3d2)**2*(x4d2-x2d2)**2*(x4d2-x1d2))
                d2 = Id2_1 + Id2_2

                d3 = -36. / ((x6d3-x3d3)*(x5d3-x3d3)*(x4d3-x3d3)**2*(x4d3-x2d3)*(x4d3-x1d3))

            else:
                I1 = 18.*(x1+x0) / ((x3-x0)**2*(x2-x0)**2*(x1-x0))
                I2 = 18.*A1**2*(x2+x1) / ((x4-x1)**2*(x3-x1)**2*(x3-x0)**2*(x2-x1)*(x2-x0)**2)
                I3 = 18.*A2**2*(x3+x2) / ((x4-x2)**2*(x4-x1)**2*(x3-x2)*(x3-x1)**2*(x3-x0)**2)
                I4 = 18.*(x4+x3) /((x4-x3)*(x4-x2)**2*(x4-x1)**2)
                d0 = I1 + I2 + I3 + I4

                Id1_1 = 18.*A1d1*(x2d1+x1d1) / ((x4d1-x1d1)**2 *(x3d1-x1d1)**2*(x3d1-x0d1)*(x2d1-x1d1)**2*(x2d1-x0d1))
                Id1_2 = 18.*A2d1*A1_1d1*(x3d1+x2d1) / ((x5d1-x2d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2*(x3d1-x2d1)*(x3d1-x1d1)**2*(x3d1-x0d1))
                Id1_3 = -18.*A2_1d1*(x4d1+x3d1) / ((x5d1-x3d1)*(x5d1-x2d1)*(x4d1-x3d1)*(x4d1-x2d1)**2*(x4d1-x1d1)**2)
                d1 = Id1_1 + Id1_2 + Id1_3

                Id2_1 = 18.*A2d2*(x3d2+x2d2) / ((x5d2-x2d2)*(x4d2-x2d2)**2*(x4d2-x1d2)*(x3d2-x2d2)**2*(x3d2-x1d2)*(x3d2-x0d2))
                Id2_2 = -18.*A1_2d2*(x4d2+x3d2) / ((x6d2-x3d2)*(x5d2-x3d2)*(x5d2-x2d2)*(x4d2-x3d2)**2*(x4d2-x2d2)**2*(x4d2-x1d2))
                d2 = Id2_1 + Id2_2

                d3 = -18.*(x4d3+x3d3) / ((x6d3-x3d3)*(x5d3-x3d3)*(x4d3-x3d3)**2*(x4d3-x2d3)*(x4d3-x1d3))
                
            A = scpsp.diags([d0,d1,d1,d2,d2,d3,d3], [0,-1,1,-2,2,-3,3], shape=(NFunc,NFunc))



    return A




cdef _D0N2_Deg3_Surf(x, A, B, C, D):
    return 30.*A**2*x**7 + 70.*A*B*x**6 + 42.*(2.*A*C+B**2)*x**5 + 105.*(A*D+B*C)*x**4 + 70.*(2.*B*D+C**2)*x**3 + 210.*C*D*x**2 + 210.*D**2*x

cdef _D1N2_Deg3_Surf(x, A, B, C):
    return 27.*A**2*x**5 + 45.*A*B*x**4 + 10.*(2.*B**2+3.*A*C)*x**3 + 30.*B*C*x**2 + 15.*C**2*x











"""
def Calc_IntOp_BSpline(cnp.ndarray[cnp.float64_t, ndim=1] Knots, int Deg, Deriv=0, Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
    if Test:
        assert isinstance(Knots,np.ndarray) and Knots.ndim==1, "Arg Knots must be a (N,) np.ndarray instance !"
        assert type(Deg) is int and Deg>=0 and Deg<=3, "Arg Deg must be a positive int !"
        if type(Deriv) is int:
            assert Deriv<=Deg, "Arg Deriv must be smller than Deg !"
        elif type(Deriv) is str:
                assert int(Deriv[1])<=Deg and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2'], "Arg Deriv must be smaller than Deg and in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2'] !"

    intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv

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










