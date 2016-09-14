# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 18:12:05 2014

@author: didiervezinet
"""

# Module used for Getting geometry
import numpy as np


# --------- Moving average ---------------------------

def DuFromAB(A,B):
    D = A
    u = (B-A)/np.linalg.norm(B-A)
    return D, u

def RZ2XYZ_1D(P,phi):
    return np.array([P[0]*np.cos(phi),P[0]*np.sin(phi),P[1]])

def RZ2XYZ_2D(P,phi):
    return np.array([P[0,:]*np.cos(phi),P[0,:]*np.sin(phi),P[1,:]])

def PolyFromLine(A,B,P,theta,phi, LTor, L2):
    A = RZ2XYZ_1D(A,phi)
    B = RZ2XYZ_1D(B,phi)
    P = RZ2XYZ_1D(P,phi)
    D, u = DuFromAB(A,B)
    nP = np.array([np.cos(theta),np.sin(theta)])
    nP = RZ2XYZ_1D(nP,phi)
    nP = nP/np.linalg.norm(nP)
    e1 = np.array([-np.sin(phi), np.cos(phi), 0.])
    Poly = RectFromPlaneLine(D, u, P, nP, e1, LTor, L2, Test=True)
    return Poly

def RectFromPlaneLine(D, u, P, nP, e1, L1, L2, Test=True):
    if Test:
        assert all([isinstance(vv,np.ndarray) and vv.shape==(3,) for vv in [D,u,P,nP,e1]]), "Arg [D,u,nP,e1] must be (3,) np.ndarrays !"
        assert all([type(ll) is float for ll in [L1,L2]]), "Args [LTor,L2] must be floats !"

    # Normalise
    u = u/np.linalg.norm(u)
    nP = nP/np.linalg.norm(nP)
    e1 = e1/np.linalg.norm(e1)

    # Check e1 perp. to nP and renorm
    e1 = e1 - np.sum(e1*nP)*nP
    e1 = e1/np.linalg.norm(e1)

    # Find intersection point (i.e. center of rectangle)
    assert not np.sum(u*nP)==0., "Line and Plane are parallel !"
    k = np.sum((P-D)*nP)/np.sum(u*nP)
    M = D + k*u

    Poly = RectFromPlaneCenter(M, nP, e1, L1, L2, Test=True)
    return Poly, M


def RectFromPlaneCenter(P, nP, e1, L1, L2, Test=True):
    if Test:
        assert all([isinstance(vv,np.ndarray) and vv.shape==(3,) for vv in [P,nP,e1]]), "Arg [D,u,nP,e1] must be (3,) np.ndarrays !"
        assert all([type(ll) is float for ll in [L1,L2]]), "Args [LTor,L2] must be floats !"
    # Normalise
    nP = nP/np.linalg.norm(nP)
    e1 = e1/np.linalg.norm(e1)

    # Check e1 perp. to nP and renorm
    e1 = e1 - np.sum(e1*nP)*nP
    e1 = e1/np.linalg.norm(e1)

    # Get e2
    e2 = np.cross(nP,e1)
    e2 = e2/np.linalg.norm(e2)

    # Deduce Poly coordinates
    Poly0 = P[0] + 0.5*L1*e1[0]*np.array([-1.,1.,1.,-1.,-1.]) + 0.5*L2*e2[0]*np.array([-1.,-1.,1.,1.,-1.])
    Poly1 = P[1] + 0.5*L1*e1[1]*np.array([-1.,1.,1.,-1.,-1.]) + 0.5*L2*e2[1]*np.array([-1.,-1.,1.,1.,-1.])
    Poly2 = P[2] + 0.5*L1*e1[2]*np.array([-1.,1.,1.,-1.,-1.]) + 0.5*L2*e2[2]*np.array([-1.,-1.,1.,1.,-1.])
    return np.array([Poly0,Poly1,Poly2])







###############################################################################
###############################################################################
#                        New and useful
###############################################################################


def get_indt(Tab_t=None, indt=None, t=None, defind=0, out=bool, Test=True):
    """
    Return an array of indices in bool or int form matching the input time point or time index
    """
    if Test:
        assert Tab_t is None or hasattr(Tab_t,'__iter__') and np.asarray(Tab_t).ndim==1, "Arg Tab_t must be a 1-dim iterable !"
        assert indt is None or type(indt) in [int,np.int64] or (hasattr(indt,'__iter__') and np.asarray(indt).ndim==1), "Arg indt must be an index or an iterable of indices !"
        assert t is None or type(t) in [float,np.float64] or (hasattr(t,'__iter__') and np.asarray(t).ndim==1), "Arg t must be an index or an iterable of time points !"
        assert type(defind) is int or defind in ['all'], "Arg defind must be an index or in ['all'] !"
    # Prepare input
    Nt = len(Tab_t)

    # Compute
    if indt is None and t is None:
        if type(defind) is int:
            ind = np.zeros((Nt,),dtype=bool)
            ind[defind] = True
        else:
            ind = np.ones((Nt,),dtype=bool)
    elif indt is not None:
        indt = int(indt) if not hasattr(indt,'__iter__') else [int(ii) for ii in indt]
        ind = np.zeros((Nt,),dtype=bool)
        ind[indt] = True
    elif t is not None:
        t = [t] if not hasattr(indt,'__iter__') else t
        indt = [int(np.nanargmin(np.abs(Tab_t-tt))) for tt in t]
        ind = np.zeros((Nt,),dtype=bool)
        ind[indt] = True
    if out==int:
        ind = ind.nonzero()[0]
    return ind





















