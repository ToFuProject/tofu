# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
import scipy.interpolate as scpinterp
import warnings

# ToFu-specific
import tofu.helper as tfh
from tofu.geom import _GG as TFGG    # For Remap_2DFromFlat() only => make local version to get rid of dependency ?





"""
###############################################################################
###############################################################################
                        Eq2D
###############################################################################
"""


############################################
#####     Computing functions for Eq2D
############################################



def _correctRef(Ref, RefV, Nt, PtsCross, MagAx):
    if Ref in ['surf','vol','rho_p','rho_t','tf']:
        # Linear interpolation to 0.
        for ii in range(0,Nt):
            r = np.hypot(PtsCross[0,:]-MagAx[ii,0], PtsCross[1,:]-MagAx[ii,1])
            indnan = np.isnan(RefV[ii,:])
            rmin = np.nanmin(r[~indnan])
            if np.any(indnan & (r<=rmin)):
                warnings.warn("Some points close to the Magnetic Axis could be extrapolated to 0 because Ref = "+Ref)
                rminarg = ((r==rmin) & (~indnan)).nonzero()[0]
                refmin = RefV[ii,rminarg]
                RefV[ii,indnan] = refmin*r[indnan]/rmin
    return RefV



def _interp_Quant(Tab_t, Tab_Pts, Tab_vPts, Pts, LQuant='rho_p', indt=None, t=None, deg=3, Test=True):
    if Test:
        assert np.asarray(Pts).ndim==2 and 2 in np.asarray(Pts).shape and np.asarray(Pts).dtype.name=='float64', "Arg Pts must be an iterable with 2D coordinates of points !"
        assert type(Tab_vPts) is list, "Arg Tab_vPts must be a list !"
        assert (type(LQuant) is str and len(Tab_vPts)==1) or (type(LQuant) is list and len(LQuant)==len(Tab_vPts) and all([type(qq) is str for qq in LQuant])), "Arg LQuant must be a str or a list of str !"
        assert indt is None or type(indt) is int or hasattr(indt,'__iter__'), "Arg indt must be a int (index) or an iterable of such !"
        assert t is None or type(t) is float or hasattr(t,'__iter__'), "Arg t must be a float (time) or an iterable of such"

    # Pre-formatting inputs
    Pts = np.asarray(Pts)
    Pts = Pts if Pts.shape[0]==2 else Pts.T
    ind = tfh.get_indt(Tab_t=Tab_t, indt=indt, t=t, defind='all', out=int, Test=Test)

    # Compute
    NP = Pts.shape[1]
    LQuant = [LQuant] if type(LQuant) is str else LQuant
    LQ = dict([(ss,[]) for ss in LQuant])
    LQin = [qq[ind,:] if not qq is None else qq for qq in Tab_vPts]
    for jj in range(0,len(LQin)):
        if not LQin[jj] is None:
            qin, X0, X1, nx0, nx1 = TFGG.Remap_2DFromFlat(np.ascontiguousarray(Tab_Pts), [qq for qq in LQin[jj]], epsx0=None, epsx1=None)
            qq = np.nan*np.ones((len(ind),NP))
            for ii in range(0,len(ind)):
                fq = scpinterp.RectBivariateSpline(X0, X1, qin[ii], bbox=[None, None, None, None], kx=deg, ky=deg, s=0)
                qq[ii,:] = fq.ev(Pts[0,:],Pts[1,:])
        else:
            qq  =None
        LQ[LQuant[jj]] = qq
    if len(LQuant)==1:
        LQ = LQ[LQuant[0]]
    return LQ




def _get_rgrad(Pts, Tab_Pts, Tab_vPts, Tab_t, indt=None, t=None, Test=True):
    if Test:
        assert type(Pts) is np.ndarray and Pts.ndim==2 and Pts.shape[0]==2 and Pts.dtype.name=='float64', "Arg Pts must be a (2,NP) np.ndarray of floats !"
        assert type(Tab_Pts) is np.ndarray and Tab_Pts.ndim==2 and Tab_Pts.shape[0]==2 and Tab_Pts.dtype.name=='float64', "Arg Tab_Pts must be a (2,N) np.ndarray of floats !"
        assert type(Tab_vPts) is np.ndarray and Tab_vPts.ndim==2 and Tab_vPts.shape[1]== Tab_Pts.shape[1] and Tab_vPts.dtype.name=='float64', "Arg vPts must be a (Nt,N) np.ndarray of floats !"
        assert indt is None or type(indt) is int or hasattr(indt,'__iter__'), "Arg indt must be a int (index) or an iterable of such !"
        assert t is None or type(t) is float or hasattr(t,'__iter__'), "Arg t must be a float (time) or an iterable of such"

    ind = tfh.get_indt(Tab_t=Tab_t, indt=indt, t=t, defind='all', out=int, Test=Test)
    rgrad = []
    vpts, X0, X1, nx0, nx1 = TFGG.Remap_2DFromFlat(np.ascontiguousarray(Tab_Pts), [qq for qq in Tab_vPts], epsx0=None, epsx1=None)
    for ii in ind:
        rg = scpinterp.RectBivariateSpline(X0, X1, vpts[ii], bbox=[None, None, None, None], kx=3, ky=3, s=0)
        gradx = rg.ev(Pts[0,:], Pts[1,:], dx=1, dy=0)
        grady = rg.ev(Pts[0,:], Pts[1,:], dx=0, dy=1)
        norm = np.hypot(gradx,grady)
        grad = np.array([gradx/norm, grady/norm])
        rgrad.append(grad)
    if len(ind)==1:
        rgrad = rgrad[0]
    return rgrad









