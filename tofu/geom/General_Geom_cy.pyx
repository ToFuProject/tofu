# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:49:34 2014

@author: didiervezinet
"""
import numpy as np
cimport numpy as np
import math
#from matplotlib.path import Path
import Polygon as plg
import scipy.integrate as scpinteg
import warnings
cimport cython
import datetime as dtm









#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
"""
                                    General
"""
#######################################################################################################
#######################################################################################################
#######################################################################################################



"""
###############################################################################
###############################################################################
                Types, Abs and Sign and Polygons handling
###############################################################################
"""



ctypedef np.float_t DTYPE_t


# Defining C-functions with smll overhead for loac calls
#@cython.profile(False)
cdef inline DTYPE_t Cabs(DTYPE_t x):
    return x if x >= 0.0 else -x

#@cython.profile(False)
cdef inline double Csign(double x):
    return 1. if x >= 0.0 else -1.


def MakeClockwise(np.ndarray[DTYPE_t,ndim=2] Poly):
    Clock = np.sum((Poly[0,1:]-Poly[0,:-1])*(Poly[1,1:]+Poly[1,:-1]))
    if Clock<0:
        Poly = Poly[:,-1::-1]
    return Poly

def PolyTestClockwise(np.ndarray[DTYPE_t,ndim=2] Poly):
    """ Assuming 2D closed Poly ! """
    return np.sum(Poly[0,:-1]*Poly[1,1:]-Poly[0,1:]*Poly[1,:-1]) < 0.



def PolyOrder(Poly, str order='C', Clock=True, close=True, str layout='(cc,N)', Test=True):
    """
    Return a polygon Poly as a np.ndarray formatted according to parameters

    Call:
    -----
        Poly = GG.PolyOrder(Poly, order='C', Clock=True, close=True, layout='(cc,N)')

    Inputs:
    -------
        Poly    np.ndarray or list or tuple     Input polygon under from of (cc,N) or (N,cc) np.ndarray (where cc = 2 or 3, the number of coordinates and N points), or list or tuple of vertices
        order   str                             Flag indicating whether the output np.ndarray shall be C-contiguous ('C') or Fortran-contiguous ('F')
        Clock   bool                            For 2-dimensional arrays only, flag indicating whether the output array shall represent a clockwise polygon (True) or anti-clockwise (False), or should be left unchanged (None)
        close   bool                            For 2-dimensional arrays only, flag indicating whether the output array shall be closed (True, i.e.: last point==first point), or not closed (False)
        layout  str                             Flag indicating whether the output np.ndarray shall be of shape '(cc,N)' or '(N,cc)'
        Test    bool                            Flag indicating whether the inputs should be tested for conformity, default: True

    Outputs:
    --------
        Poly    np.ndarray                      Output formatted polygon

    """
    Poly = np.asarray(Poly)
    if Test:
        assert Poly.ndim==2 and (2 in Poly.shape or 3 in Poly.shape), "Arg Poly must contain the 2D or 3D coordinates of points of a polygon !"
        assert order in ['C','F'], "Arg order must be in ['C','F'] !"
        assert type(Clock) is bool, "Arg Clock must be a bool !"
        assert type(close) is bool, "Arg close must be a bool !"
        assert layout in ['(cc,N)','(N,cc)'], "Arg layout must be in ['(cc,N)','(N,cc)'] !"

    Poly = Poly if (Poly.shape[0] in [2,3] and Poly.shape[1]>=3) else Poly.T
    Poly = Poly if np.all(Poly[:,0]==Poly[:,-1]) else np.concatenate((Poly,Poly[:,0:1]),axis=1)
    if Poly.shape[0]==2 and not Clock is None:
        clockwise = PolyTestClockwise(Poly)
        Poly = Poly if clockwise==Clock else Poly[:,::-1]
    Poly = Poly if close else Poly[:,:-1]
    Poly = Poly if layout=='(cc,N)' else Poly.T
    Poly = np.ascontiguousarray(Poly) if order=='C' else np.asfortranarray(Poly)
    return Poly


def Calc_BaryNorm_3DPoly_1D(np.ndarray[DTYPE_t,ndim=2] Poly):
    """ Computes the barycenter (Points) of a planar polygon and a normalised vector perpendicular to its plane

    D. VEZINET, Aug. 2014
    Inputs :
        Poly        A (3,N) np.ndarray (with 1st point = last point) representing a 3D polygon in (X,Y,Z) coordinates
    Outputs :
        BaryP       A (3,) np.ndarray in (X,Y,Z) coordinates representing the barycenter of Poly
        nIn         A (3,) np.ndarray in (X,Y,Z) coordinates representing a normalised vector perpendicular to the plane of Poly
    """
    assert Poly.shape[0]==3 and np.all(Poly[:,0:1]==Poly[:,-1:]), "Arg Poly should be a (3,N) ndarray with beginning=end !"
    BaryP = np.sum(Poly[:,0:Poly.shape[1]-1],axis=1,keepdims=False)/(Poly.shape[1]-1)
    un = [Poly[:,0]-BaryP, Poly[:,1]-BaryP, Poly[:,2]-BaryP]
    LL = [np.cross(un[0],un[1]),np.cross(un[0],un[2]),np.cross(un[1],un[2])]
    Ll = [ll[0]**2+ll[1]**2+ll[2]**2 for ll in LL]
    nIn = LL[Ll.index(max(Ll))]
    nIn = nIn/math.sqrt(nIn[0]**2+nIn[1]**2+nIn[2]**2)
    return BaryP, nIn


def Calc_2DPolyFrom3D_1D(np.ndarray[DTYPE_t,ndim=2] Poly, P=None, en=None, e1=None, e2=None, Test=True):
    cdef np.ndarray[DTYPE_t,ndim=2] Poly2
    cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] Pb, enb, e1b, e2b
    cdef list LL, Ll
    cdef int NPoly = Poly.shape[1]
    if Test:
        assert Poly.shape[0]==3 and np.all(Poly[:,0]==Poly[:,-1]), "Arg Poly should be a (3,N) ndarray with beginning=end !"
        assert all([vv is None or (isinstance(vv,np.ndarray) and vv.shape==(3,)) for vv in [P,en,e1,e2]]), "Args [P,en,e1,e2] must be a (3,) np.ndarrays !"
    if en is None:
        Vect = np.diff(Poly,axis=1)
        LL = [np.cross(Vect[:,0],Vect[:,1]),np.cross(Vect[:,0],Vect[:,2]),np.cross(Vect[:,1],Vect[:,2])]
        Ll = [ll[0]**2+ll[1]**2+ll[2]**2 for ll in LL]
        enb = LL[Ll.index(max(Ll))]
    else:
        assert en.shape==(3,), "Arg en must be a (3,) np.ndarray !"
        enb = en
    enb = enb/math.sqrt(enb[0]**2+enb[1]**2+enb[2]**2)
    if P is None:
        Pb = np.sum(Poly[:,0]*enb)*enb
    else:
        assert P.shape==(3,) and all([np.sum((Poly[:,ii]-P)*enb)<1e-14 for ii in range(0,NPoly)]), "Arg P must be a (3,) np.ndarray and "+str([np.sum((Poly[:,ii]-P)*enb) for ii in range(0,NPoly)])
        Pb = P
    if e1 is None:
        e1b,e2b = Calc_DefaultCheck_e1e2_PLane_1D(Pb,enb)
    else:
        assert e1.shape==e2.shape==(3,) and np.linalg.norm(e1)==1. and np.linalg.norm(e2)==1. and np.sum(e1*enb)<1.e-14 and np.sum(e2*enb)<1.e-14 and np.sum(e1*e2)<1.e-14, "Args e1 and e2 must be normalised np.ndarray perp. !"
        e1b, e2b = e1, e2
    Poly = Poly - np.tile(Pb,(NPoly,1)).T
    Poly2 = np.array([Poly[0,:]*e1b[0]+Poly[1,:]*e1b[1]+Poly[2,:]*e1b[2], Poly[0,:]*e2b[0]+Poly[1,:]*e2b[1]+Poly[2,:]*e2b[2]])
    return Poly2, Pb, enb, e1b, e2b


def Calc_2DPolyFrom3D_2D(Poly,P=None,en=None,e1=None,e2=None,Test=True):   # Not used ?
    if Test:
        assert isinstance(Poly,np.ndarray) and Poly.shape[0]==3 and np.all(Poly[:,0:1]==Poly[:,-1:]), "Arg Poly should be a (3,N) ndarray with beginning=end !"
        assert en is None or (isinstance(en,np.ndarray) and en.shape==(3,1)), "Arg en must be a (3,1) np.ndarray !"
        assert e1 is None or (isinstance(e1,np.ndarray) and e2.shape==(3,1)), "Arg e1 must be a (3,1) np.ndarray !"
        assert e2 is None or (isinstance(e2,np.ndarray) and e2.shape==(3,1)), "Arg e2 must be a (3,1) np.ndarray !"
        assert P is None or (isinstance(P,np.ndarray) and P.shape==(3,1)), "Arg P must be a (3,1) np.ndarray !"
    if en is None:
        Vect = np.diff(Poly,axis=1)
        en = np.concatenate((np.cross(Vect[:,0],Vect[:,1]).reshape((3,1)),np.cross(Vect[:,0],Vect[:,2]).reshape((3,1)),np.cross(Vect[:,1],Vect[:,2]).reshape((3,1))),axis=1)
        narg = np.argmax(np.sum(en**2,axis=0))
        en = en[:,narg:narg+1]
    en = en/np.sqrt(en[0,0]**2+en[1,0]**2+en[2,0]**2)
    if P is None:
        P = np.sum(Poly[:,0:1]*en)*en
    else:
        assert all([np.sum((Poly[:,ii:ii+1]-P)*en)<1e-15 for ii in range(0,Poly.shape[1])]), str([np.sum((Poly[:,ii:ii+1]-P)*en) for ii in range(0,Poly.shape[1])])
    e1,e2 = Calc_DefaultCheck_e1e2_PLanes_2D(P,en,e1,e2)
    NPoly = Poly.shape[1]
    Poly = Poly - np.dot(P,np.ones((1,NPoly)))
    Poly2 = np.array([np.sum(Poly*np.dot(e1,np.ones((1,NPoly))),axis=0,keepdims=False), np.sum(Poly*np.dot(e2,np.ones((1,NPoly))),axis=0,keepdims=False)])
    return Poly2, P, en, e1, e2

def Calc_3DPolyfrom2D_1D(Poly,P,en,e1,e2,Test=True):
    if Test:
        assert isinstance(Poly,np.ndarray) and Poly.shape[0]==2 and np.all(Poly[:,0:1]==Poly[:,-1:]), "Arg Poly should be a (2,N) ndarray with beginning=end !"
        assert all([isinstance(vv,np.ndarray) and vv.shape==(3,) for vv in [P,en,e1,e2]]), "Args [P,en,e1,e2] must be (3,) np.ndarrays !"
    return np.array([P[0]+e1[0]*Poly[0,:]+e2[0]*Poly[1,:], P[1]+e1[1]*Poly[0,:]+e2[1]*Poly[1,:], P[2]+e1[2]*Poly[0,:]+e2[2]*Poly[1,:]])


def Calc_DefaultCheck_e1e2_PLane_1D(DTYPE_t[::1] P, DTYPE_t[::1] nP, CrossNTHR=0.01, EPS=1.e-10):    # Used
    cdef DTYPE_t CrossNorm, Sum
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] e1b = np.zeros((3,)), e2b = np.zeros((3,))
    cdef DTYPE_t RPL = math.sqrt(P[0]**2+P[1]**2)
    if RPL==0.:
        if nP[2]==0.:
            e1b[0], e1b[1], e1b[2] = nP[1],-nP[0],0.
        else:
            e1b[0], e1b[1], e1b[2] = nP[2],0.,-nP[0]
    else:
        thetaPL = math.atan2(P[1],P[0])
        e1b[0], e1b[1], e1b[2] = -math.sin(thetaPL),math.cos(thetaPL),0.
    CrossNorm = math.sqrt( (e1b[1]*nP[2]-e1b[2]*nP[1])**2 + (e1b[2]*nP[0]-e1b[0]*nP[2])**2 + (e1b[0]*nP[1]-e1b[1]*nP[0])**2 )
    if not RPL==0.:
        if CrossNorm < CrossNTHR:
            e1b[0], e1b[1], e1b[2] = math.cos(thetaPL), math.sin(thetaPL), 0.
        Sum = nP[0]*e1b[0]+nP[1]*e1b[1]+nP[2]*e1b[2]
        e1b[0], e1b[1], e1b[2] = e1b[0]-Sum*nP[0], e1b[1]-Sum*nP[1], e1b[2]-Sum*nP[2]
        Sum = math.sqrt(e1b[0]**2+e1b[1]**2+e1b[2]**2)
    e1b[0], e1b[1], e1b[2] = e1b[0]/Sum, e1b[1]/Sum, e1b[2]/Sum
    e2b[0], e2b[1], e2b[2] = nP[1]*e1b[2]-nP[2]*e1b[1], nP[2]*e1b[0]-nP[0]*e1b[2], nP[0]*e1b[1]-nP[1]*e1b[0]
    Sum = math.sqrt(e2b[0]**2+e2b[1]**2+e2b[2]**2)
    e2b[0], e2b[1], e2b[2] = e2b[0]/Sum, e2b[1]/Sum, e2b[2]/Sum
    return e1b, e2b

def Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs, e1s=None, e2s=None, CrossNTHR=0.01, EPS=1.e-10):
    NPlans = nPs.shape[1]
    if e1s is None:
        e1s = np.nan*np.ones((3,NPlans))
        RPL = np.sqrt(Ps[0,:]**2 + Ps[1,:]**2)
        ind0 = RPL==0.
        if np.any(ind0):
            ind3 = nPs[2,:]==0.
            e1s[:,ind0 & ind3] = np.array([nPs[1,ind0 & ind3],-nPs[0,ind0 & ind3],np.zeros((np.sum(ind0 & ind3)))])
            e1s[:,ind0 & ~ind3] = np.array([nPs[2,ind0 & ~ind3],np.zeros((np.sum(ind0 & ~ind3))),-nPs[0,ind0 & ~ind3]])
        thetaPL = np.arccos(Ps[0,~ind0]/RPL[~ind0])
        thetaPL[Ps[1,~ind0]<0] = -thetaPL[Ps[1,~ind0]<0]
        e1s[:,~ind0] = np.array([-np.sin(thetaPL),np.cos(thetaPL),np.zeros((np.sum(~ind0),))])
        CrossNorm = np.array([e1s[1,:]*nPs[2,:] - e1s[2,:]*nPs[1,:], e1s[2,:]*nPs[0,:] - e1s[0,:]*nPs[2,:], e1s[0,:]*nPs[1,:] - e1s[1,:]*nPs[0,:]])
        CrossNorm = np.sqrt(np.sum(CrossNorm**2,axis=0))
        ind = CrossNorm < CrossNTHR
        if np.any(ind):
            ind = ~ind0 & ind
            e1s[:,ind] = np.array([np.cos(thetaPL), np.sin(thetaPL), np.zeros((np.sum(ind),))])
        e1s[:,~ind0] = e1s[:,~ind0] - np.dot(np.ones((3,1)),np.sum(nPs[:,~ind0]*e1s[:,~ind0],axis=0,keepdims=True))*nPs[:,~ind0]
        e1s = e1s/np.dot(np.ones((3,1)),np.sqrt(np.sum(e1s*e1s,axis=0,keepdims=True)))
    else:
        assert isinstance(e1s,np.ndarray) and e1s.shape==(3,NPlans) and np.all(np.abs(np.sqrt(np.sum(e1s*e1s,axis=0))-1) < EPS) and np.all(np.sum(nPs*e1s,axis=0)<EPS), "Arg e1s should be a (3,N) ndarray, normalised and perpendicular to nP !"
    if e2s is None:
        e2s = np.array([nPs[1,:]*e1s[2,:] - nPs[2,:]*e1s[1,:], nPs[2,:]*e1s[0,:] - nPs[0,:]*e1s[2,:], nPs[0,:]*e1s[1,:] - nPs[1,:]*e1s[0,:]])
        e2s = e2s/np.dot(np.ones((3,1)),np.sqrt(np.sum(e2s**2,axis=0,keepdims=True)) )
    else:
        assert isinstance(e2s,np.ndarray) and e2s.shape==(3,NPlans) and np.all(np.abs(np.sqrt(np.sum(e2s*e2s,axis=0))-1) < EPS) and np.all(np.sum(nPs*e2s,axis=0)<EPS) and np.all(np.sum(e2s*e1s,axis=0)<EPS), "Arg e2s should be normalised, perp. to nP and to e1 !"
    return e1s, e2s


def Calc_PolyProjPlane(Poly,P,nP,e1P=None,e2P=None,Test=True):
# Return Projection of Poly on plane (P,nP) and (optionally) the components (X1,X2) along (e1P,e2P)
    if Test:
        assert isinstance(Poly,np.ndarray) and Poly.shape[0]==3, "Arg Poly must be (3,N) ndarray !"
        assert isinstance(P,np.ndarray) and P.shape==(3,1), "Arg P must be (3,1) ndarray !"
        assert isinstance(nP,np.ndarray) and nP.shape==(3,1), "Arg nP must be (3,1) ndarray !"
        assert e1P is None or (isinstance(e1P,np.ndarray) and e1P.shape==(3,1)), "Arg e1P must be (3,1) ndarray !"
        assert e2P is None or (isinstance(e2P,np.ndarray) and e2P.shape==(3,1)), "Arg e2P must be (3,1) ndarray !"

    Pmult = np.dot(P,np.ones((1,Poly.shape[1])))
    PolySca = np.sum((Poly-Pmult)*np.dot(nP,np.ones((1,Poly.shape[1]))),axis=0,keepdims=True)
    PolyProj = Poly - np.dot(nP,PolySca)
    if not (e1P is None or e2P is None):
        assert np.dot(nP.T,e1P) == np.dot(nP.T,e2P) == 0, "Args e1P and e2P should be orthogonal to nP !"
        PolyX1 = np.sum((PolyProj-Pmult)*np.dot(e1P,np.ones((1,Poly.shape[1]))),axis=0,keepdims=True)
        PolyX2 = np.sum((PolyProj-Pmult)*np.dot(e2P,np.ones((1,Poly.shape[1]))),axis=0,keepdims=True)
    else:
        PolyX1 = PolyX2 = []

    return PolyProj, PolyX1, PolyX2


def Calc_PolysProjPlanePoint_Fast(list Polys, DTYPE_t[::1] A, DTYPE_t[::1] P, DTYPE_t[::1] nP, DTYPE_t[::1] e1P, DTYPE_t[::1] e2P):
    # Returns homothetic (with center A) projections of Poly on planes (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and planes, but a uniqu polygon list common to all of them
    # !!! Only use with a list of Polygons !!!
    # Returns only the 2D projected polygons !!!

    cdef int NList = len(Polys)
    cdef list NPperPoly = [int(Polys[ii].shape[1]) for ii in range(0,NList)]
    cdef list IndPoly = np.cumsum(NPperPoly,dtype=int).tolist()
    cdef np.ndarray[DTYPE_t, ndim=2,mode='c'] Polysbis = np.concatenate(tuple(Polys),axis=1)
    cdef int NPoly = int(Polysbis.shape[1])

    cdef list AM = [Polysbis[0,:]-A[0], Polysbis[1,:]-A[1], Polysbis[2,:]-A[2]]
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] ScaAMn = AM[0]*nP[0] + AM[1]*nP[1] + AM[2]*nP[2]
    cdef DTYPE_t APnP = (P[0]-A[0])*nP[0] + (P[1]-A[1])*nP[1] + (P[2]-A[2])*nP[2]
    assert not np.any(ScaAMn*APnP<0), "Inconsistent points found in Calc_PolysProjPlanePoint_Fast !"

    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] k = np.zeros((NPoly,))
    ind = np.abs(ScaAMn) > 1.e-14
    k[ind] = APnP/ScaAMn[ind]

    cdef list PolyProj = [A[0]+k*AM[0], A[1]+k*AM[1], A[2]+k*AM[2]]
    cdef np.ndarray[DTYPE_t,ndim=2,mode='c'] PolyX12 = np.array([(PolyProj[0]-P[0])*e1P[0]+(PolyProj[1]-P[1])*e1P[1]+(PolyProj[2]-P[2])*e1P[2], (PolyProj[0]-P[0])*e2P[0]+(PolyProj[1]-P[1])*e2P[1]+(PolyProj[2]-P[2])*e2P[2]])

    if NList>1:
        return [PolyX12[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
    else:
        return PolyX12


def Calc_PolysProjPlanePoint(Polys,A,P,nP,e1P=None,e2P=None,Test=True):
    # Returns homothetic (with center A) projections of Poly on plane (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and planes, but a unique polygon list common to all of them
    if Test:
        assert type(Polys) is list and all([isinstance(Polys[ii],np.ndarray) and Polys[ii].shape[0]==3 for ii in range(0,len(Polys))]) or (isinstance(Polys,np.ndarray) and Polys.shape[0]==3), "Arg Polys must be (3,N) ndarray or list of (3,N) ndarrays !"
        assert all([isinstance(vv,np.ndarray) and vv.shape==(3,) for vv in [A,P,nP]]), "Args [A,P,nP] must be (3,) ndarrays !"
        assert all([ee is None or (isinstance(ee,np.ndarray) and ee.shape==(3,)) for ee in [e1P,e2P]]), "Args [e1P,e2P] must be (3,) np.ndarrays !"
    if type(Polys) is list:
        NList = len(Polys)
        NPperPoly = np.array([Polys[ii].shape[1] for ii in range(0,NList)])
        IndPoly = np.cumsum(NPperPoly)
        Polys = np.concatenate(tuple(Polys),axis=1)
    else:
        NList = 1
    NPoly = Polys.shape[1]
    AM = np.array([Polys[0,:]-A[0], Polys[1,:]-A[1], Polys[2,:]-A[2]])
    ScaAMn = AM[0,:]*nP[0] + AM[1,:]*nP[1] + AM[2,:]*nP[2]
    Scasign = ScaAMn*np.sum((P-A)*nP)
    indneg = Scasign<0.
    if np.any(indneg):
        warnings.warn('Inconsistent points were found in Calc_PolysProjPlanePoint !')
        Ratio = 0.05
        AM[:,indneg] = AM[:,indneg] - (1.+Ratio)*nP.reshape((3,1))*ScaAMn[indneg]
        ScaAMn[indneg] = -Ratio*ScaAMn[indneg]
    k = np.zeros((NPoly,))
    ind = np.abs(ScaAMn) > 1.e-14
    k[ind] = np.sum((P-A)*nP)/ScaAMn[ind]
    PolyProj = np.array([A[0]+k*AM[0,:], A[1]+k*AM[1,:], A[2]+k*AM[2,:]])
    if not (e1P is None or e2P is None) and NList>1:
        Ptemp0, Ptemp1, Ptemp2 = PolyProj[0,:]-P[0], PolyProj[1,:]-P[1], PolyProj[2,:]-P[2]
        PolyX12 = np.array([Ptemp0*e1P[0]+Ptemp1*e1P[1]+Ptemp2*e1P[2], Ptemp0*e2P[0]+Ptemp1*e2P[1]+Ptemp2*e2P[2]])
        PolyProj = [PolyProj[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
        PolyX12 = [PolyX12[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
    elif not (e1P is None or e2P is None):
        Ptemp0, Ptemp1, Ptemp2 = PolyProj[0,:]-P[0], PolyProj[1,:]-P[1], PolyProj[2,:]-P[2]
        PolyX12 = np.array([Ptemp0*e1P[0]+Ptemp1*e1P[1]+Ptemp2*e1P[2], Ptemp0*e2P[0]+Ptemp1*e2P[1]+Ptemp2*e2P[2]])
    elif (e1P is None or e2P is None) and NList>1:
        PolyX12 = []
        PolyProj = [PolyProj[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
    else:
        PolyX12 = []
    return PolyProj, PolyX12


def Calc_PolysProjPlanesPoint(Polys,A,Ps,nPs,e1P=None,e2P=None,Test=True):   # Used
    """ Returns homothetic (with center A) projections of Poly on planes (P,nP)s, and optionnaly their components (X1,X2) along (e1P,e2P) """
    # Arbitrary Points and planes, but a uniqu polygon list common to all of them
    cdef Py_ssize_t ii, jj
    cdef np.ndarray[DTYPE_t,ndim=2] LPolytemp
    if Test:
        assert (type(Polys) is list and all([isinstance(pp,np.ndarray) and pp.shape[0]==3 for pp in Polys])) or isinstance(Polys,np.ndarray) and Polys.shape[0]==3, "Arg Polys must be (3,N) ndarray or a list of such !"
        assert isinstance(A,np.ndarray) and A.shape==(3,1), "Arg A must be (3,1) ndarray !"
        assert isinstance(Ps,np.ndarray) and Ps.shape[0]==3, "Arg Ps must be (3,M) ndarray !"
        assert isinstance(nPs,np.ndarray) and nPs.shape==Ps.shape, "Arg nPs must be (3,M) ndarray !"
        assert all([ee is None or (isinstance(ee,np.ndarray) and ee.shape==Ps.shape) for ee in [e1P,e2P]]), "Args [e1P,e2P] must be (3,M) ndarrays !"

    if type(Polys) is list:
        NList = len(Polys)
        NPperPoly = np.array([Polys[ii].shape[1] for ii in range(0,NList)])
        IndPoly = np.cumsum(NPperPoly)
        Polys = np.concatenate(tuple(Polys),axis=1)
    else:
        NPperPoly = Polys.shape[1]
        IndPoly = NPperPoly
        NList = 1

    NPoly, NPlans = Polys.shape[1], Ps.shape[1]
    Abis = np.swapaxes(np.resize(A.T,(NPoly,NPlans,3)),0,1)
    Polybis = np.resize(Polys.T,(NPlans,NPoly,3))
    AMbis = Polybis - Abis

    nPsbis = np.swapaxes(np.resize(nPs.T,(NPoly,NPlans,3)),0,1)
    Psbis = np.swapaxes(np.resize(Ps.T,(NPoly,NPlans,3)),0,1)
    ScaAMn = np.sum(AMbis*nPsbis,axis=2,keepdims=True)
    Scasign = ScaAMn*np.sum((Psbis-Abis)*nPsbis,axis=2,keepdims=True)
    indneg = Scasign<0
    if np.any(indneg):
        warnings.warn('Inconsistent points were found in Calc_PolysProjPlanesPoint !')
        assert not np.any(indneg), "Inconsistent points !"
        indPoly = np.any(indneg,axis=0).flatten()
        for ii in range(0,NList):
            if np.any(indPoly[(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]]):
                for jj in range(0,NPlans):
                    LPolytemp = Polys[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]]
                    LPolytemp = Calc_PolyLimByPlane_1D(LPolytemp, A.flatten(), nPs[:,jj], ~indPoly[(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]], EPS=0.2,Test=False)
                    #LPolytemp = Calc_PolyLimByPlane_2D(LPolytemp,A,nPs[:,jj:jj+1], ~indPoly[(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]], EPS=0.2,Test=False)
                    try:
                        Polybis[jj,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii],:] = LPolytemp.T
                    except ValueError:
                        Polybis[jj,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii],:] = LPolytemp.T
        AMbis = Polybis - Abis
        ScaAMn = np.sum(AMbis*nPsbis,axis=2,keepdims=True)
        Scasign = ScaAMn*np.sum((Psbis-Abis)*nPsbis,axis=2,keepdims=True)
        indneg = Scasign<0.
        assert np.all(~indneg), "Re-arrangement of inconsistent points was a failure..."

    ScaAPn = np.sum((Psbis-Abis)*nPsbis,axis=2,keepdims=True)
    del Polys, A, Ps, nPs
    k = np.zeros((NPlans,NPoly,1))
    ind = np.abs(ScaAMn) > 1.e-14
    k[ind] = ScaAPn[ind]/ScaAMn[ind]

    PolyProjii = Abis + np.resize(k.T,(3,NPoly,NPlans)).T*AMbis
    del Abis, AMbis, nPsbis, ScaAMn, ScaAPn
    PolyProjX, PolyProjY, PolyProjZ = PolyProjii[:,:,0], PolyProjii[:,:,1], PolyProjii[:,:,2]

    if not (e1P is None or e2P is None):
        e1bis = np.swapaxes(np.resize(e1P.T,(NPoly,NPlans,3)),0,1)
        e2bis = np.swapaxes(np.resize(e2P.T,(NPoly,NPlans,3)),0,1)
        PolyProjX1 = np.sum((PolyProjii - Psbis)*e1bis,axis=2,keepdims=False)
        PolyProjX2 = np.sum((PolyProjii - Psbis)*e2bis,axis=2,keepdims=False)
    else:
        PolyProjX1 = PolyProjX2 = []

    if NList > 1:
        PolyProjX = [PolyProjX[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
        PolyProjY = [PolyProjY[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
        PolyProjZ = [PolyProjZ[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
        if not (e1P is None or e2P is None):
            PolyProjX1 = [PolyProjX1[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]
            PolyProjX2 = [PolyProjX2[:,(IndPoly[ii]-NPperPoly[ii]):IndPoly[ii]] for ii in range(0,NList)]

    return PolyProjX, PolyProjY, PolyProjZ, PolyProjX1, PolyProjX2


def Calc_PolyLimByPlane_1D(Poly,P,nP,indpos, EPS=0.,Test=True): 
    """ Return a new 3D Poly (! Convex !) corresponding to the truncation of a 3D Poly by a plane """
    if Test:
        assert isinstance(Poly,np.ndarray) and Poly.shape[0]==3, "Arg Poly must be a (3,N) np.ndarray !"
        assert all([isinstance(vv,np.ndarray) and vv.shape==(3,) for vv in [P,nP]]), "Args [P,nP,indpos] must be (3,) np.ndarrays !"
        assert isinstance(indpos,np.ndarray) and indpos.shape==(Poly.shape[1],) and indpos.dtype.name=='bool', "Arg indpos should be a 1d np.ndarray(dtype='bool') !"

    Polybis, PPP, en, e1, e2 = Calc_2DPolyFrom3D_1D(Poly,Poly[:,0],Test=False)
    Lu = np.array([en[1]*nP[2]-en[2]*nP[1],en[2]*nP[0]-en[0]*nP[2],en[0]*nP[1]-en[1]*nP[0]])
    Lu = Lu/math.sqrt(Lu[0]**2+Lu[1]**2+Lu[2]**2)
    e2 = np.array([en[1]*Lu[2]-en[2]*Lu[1],en[2]*Lu[0]-en[0]*Lu[2],en[0]*Lu[1]-en[1]*Lu[0]])
    e2 = e2/math.sqrt(e2[0]**2+e2[1]**2+e2[2]**2)
    Polybis, PPP, en, Lu, e2 = Calc_2DPolyFrom3D_1D(Poly,Poly[:,0],en,Lu,e2,Test=False)
    ke2Lu = np.sum((Poly[:,0]-P)*nP)

    E2 = Polybis[1,indpos]
    indmaxe2 = np.argmax(np.abs(E2))
    E2 = E2[indmaxe2]
    mine1, maxe1 = np.min(Polybis[0,indpos]), np.max(Polybis[0,indpos])
    Sign = np.sign(E2-ke2Lu)
    EPS = EPS*np.abs(E2-ke2Lu)
    P2 = plg.Polygon(np.array([[mine1, maxe1, maxe1, mine1],[ke2Lu+Sign*EPS, ke2Lu+Sign*EPS, E2, E2]]).T)
    Polybis = P2 & plg.Polygon(Polybis.T)
    Polybis = np.array(Polybis[0]).T
    Polybis = np.concatenate((Polybis,Polybis[:,0:1]),axis=1)
    if not Polybis.shape[1]==Poly.shape[1]:
        if Polybis.shape[1] > Poly.shape[1]:
            Vect = np.sum(np.diff(Polybis,axis=1)**2,axis=0)
            if np.any(Vect==0.):
                indout = Vect==0
                Polybis = np.delete(Polybis,indout.nonzero()[0],axis=1)
            elif Polybis.shape[1] == Poly.shape[1]+1:
                indout = np.sum(Vect**2,axis=0).argmin()
                Polybis = np.delete(Polybis,indout,axis=1)
            assert Polybis.shape[1]==Poly.shape[1], "Polybis and Poly have different sizes !"
    Poly = Calc_3DPolyfrom2D_1D(Polybis,Poly[:,0],en,Lu,e2,Test=False)
    return Poly




def Calc_PolyInterLPoly2D(LPoly,Test=True):
    # Returns Polygon that is the intersection of a list of polygons
    if Test:
        assert type(LPoly) is list, "Arg LPoly should be a list of Polygons !"
        assert all([(isinstance(P,np.ndarray) and P.shape[0]==2) for P in LPoly]), "LPoly elements should be (2,N) ndarrays !"
        assert all([np.all(P[:,0]==P[:,-1]) for P in LPoly]), "LPoly elements should be closed polygons !"
    PolyRef = plg.Polygon(LPoly[0].T)
    for ii in range(1,len(LPoly)):
        PolyRef = PolyRef & plg.Polygon(LPoly[ii].T)
        if PolyRef.nPoints()==0:
            return np.empty((3,0))
    PolyRef = np.array(PolyRef[0]).T
    PolyRef = np.concatenate((PolyRef,PolyRef[:,0:1]),axis=1)
    return PolyRef



"""
###############################################################################
###############################################################################
                Intersections, segments, planes, lines
###############################################################################
"""



def Calc_Inter2Segm_2D(AB,CD):
    assert isinstance(AB,np.ndarray) and AB.shape==(2,2), "Arg AB should be a (2,2) ndarray !"
    assert isinstance(CD,np.ndarray) and CD.shape==(2,2), "Arg AB should be a (2,2) ndarray !"

    AmB, CmD = AB[:,0:1]-AB[:,1:2], CD[:,0:1]-CD[:,1:2]
    AmBn, CmDn = np.linalg.norm(AmB), np.linalg.norm(CmD)
    M = np.nan*np.ones((2,1))
    Eps = min([AmBn,CmDn])*1e-8
    S = []
    if AmBn==0. and CmDn==0. and np.all(AB[:,0]==CD[:,0]):
        M = AB[:,0:1]
    elif AmBn==0. and not CmDn==0.:
        if np.cross(CD,AB[:,0:1]-CD[:,0:1])==0.:
            sca = np.dot((AB[:,0:1]-CD[:,0:1]),CD)
            if sca >= 0 and sca < 1:
                M = AB[:,0:1]
    elif not AmBn==0. and CmDn==0.:
        if np.cross(AB,AB[:,0:1]-CD[:,0:1])==0.:
            sca = np.dot((CD[:,0:1]-AB[:,0:1]),AB)
            if sca >= 0 and sca < 1:
                M = CD[:,0:1]
    else:
        if np.abs(AmB[0,0])<Eps and np.abs(CmD[0,0])<Eps and np.abs(AB[0,0]-CD[0,0])<Eps:
            S = [AB[:,0:1]]
            return S
        elif np.abs(AmB[0,0])<Eps and not np.abs(CmD[0,0])<Eps:
            y = (CD[1,1]-CD[1,0])*(AB[0,0]-CD[0,0])/(CD[0,1]-CD[0,0]) + CD[1,0]
            M = np.array([[AB[0,0]],[y]])
        elif not np.abs(AmB[0,0])<Eps and np.abs(CmD[0,0])<Eps:
            y = (AB[1,1]-AB[1,0])*(CD[0,0]-AB[0,0])/(AB[0,1]-AB[0,0]) + AB[1,0]
            M = np.array([[CD[0,0]],[y]])
        elif not np.abs(AmB[0,0])<Eps and not np.abs(CmD[0,0])<Eps:
            bb = (AB[1,1]-AB[1,0])*AB[0,0]/(AB[0,1]-AB[0,0]) - (CD[1,1]-CD[1,0])*CD[0,0]/(CD[0,1]-CD[0,0]) + CD[1,0]-AB[1,0]
            if np.cross(AmB.flatten(),CmD.flatten())==0. and bb==0.:  # Parallel
                scaAC = np.dot((CD[:,0:1]-AB[:,0:1]).flatten(),-AmB.flatten())
                scaAD = np.dot((CD[:,1:2]-AB[:,0:1]).flatten(),-AmB.flatten())
                scaBC = np.dot((CD[:,0:1]-AB[:,1:2]).flatten(),AmB.flatten())
                scaBD = np.dot((CD[:,1:2]-AB[:,1:2]).flatten(),AmB.flatten())
                if scaAC*scaAD <= 0.:
                    S = [AB[:,0:1]]
                    return S
                elif scaBC*scaBD <= 0. and max([scaBC,scaBD])>0:
                    if min([scaAC,scaAD])<=0.:
                        S = [AB[:,0:1]]
                        return S
                    else:
                        if scaBC > scaBD:
                            S = [CD[:,0:1]]
                            return S
                        else:
                            S = [CD[:,1:2]]
                            return S
            else:
                aa = (AB[1,1]-AB[1,0])/(AB[0,1]-AB[0,0]) - (CD[1,1]-CD[1,0])/(CD[0,1]-CD[0,0])
                if not aa == 0.:
                    x = bb/aa
                    y = (AB[1,1]-AB[1,0])*(x-AB[0,0])/(AB[0,1]-AB[0,0]) + AB[1,0]
                    M = np.array([[x],[y]])
    if not np.all(np.isnan(M)):
        t = np.dot((M-AB[:,0:1]).flatten(),-AmB.flatten())/AmBn**2
        k = np.dot((M-CD[:,0:1]).flatten(),-CmD.flatten())/CmDn**2
        if t >= 0 and t < 1 and k >= 0 and k < 1:
            S = [M]
    return S


def Calc_InterLinePlane(D,nL,P,nP, Test=True):
    if Test:
        assert all(isinstance(x,np.ndarray) and x.shape[0]==3 for x in [D,nL,P,nP]), "Args should be (3,N) ndarray instances !"
        assert D.shape[1]==nL.shape[1], "Args D and nL should have sime size"
        assert P.shape[1]==nP.shape[1]==1, "Args P and nP should have shape[1]=1 !"

    k = np.nan*np.ones((D.shape[1],))
    PDnP = np.sum((D-np.dot(P,np.ones((1,D.shape[1]))))*np.dot(nP,np.ones((1,D.shape[1]))),axis=0)
    nLnP = np.sum(nL*np.dot(nP,np.ones((1,D.shape[1]))),axis=0)
    indnan = np.logical_and(nLnP==0., np.logical_not(PDnP==0.))
    ind0 = PDnP==0
    k[ind0] = 0
    k[~indnan & ~ind0] = -PDnP[~indnan & ~ind0]/nLnP[~indnan & ~ind0]
    Points = D + nL*np.dot(np.ones((3,1)),k.reshape((1,len(k))))
    return Points


def Calc_2DMeshPlane(P,nP,Theta=np.linspace(0,2*np.pi-2*np.pi/20,20),ds=0.005,Rad=0.1,e1=None,e2=None):
    assert isinstance(P,np.ndarray) and P.shape==(3,1), "Arg P should be a (3,1) ndarray"
    assert isinstance(nP,np.ndarray) and nP.shape==(3,1), "Arg nP should be a (3,1) ndarray"
    assert isinstance(Theta,np.ndarray) and (Theta.ndim==1 or (Theta.ndim==2 and min(Theta.shape==1))), "Arg Theta should be a ndarray"
    assert type(ds) is float, "Arg ds should be a float !"
    assert e1 is None or (isinstance(e1,np.ndarray) and e1.shape==(3,1)), "Arg e1 should be a (3,1) ndarray"
    assert e2 is None or (isinstance(e2,np.ndarray) and e2.shape==(3,1)), "Arg e1 should be a (3,1) ndarray"

    nP = nP/np.linalg.norm(nP)
    if e1 is None:
        RP = np.sqrt(np.sum(P[0:2,0]**2))
        thetaP = np.arccos(P[0,0]/RP)
        if P[1,0]<0:
            thetaP = -thetaP
        e1 = np.array([[-np.sin(thetaP)],[np.sin(thetaP)],[0.]])
        if np.linalg.norm(np.cross(nP.flatten(),e1.flatten())) < 0.01:
            e1 = np.array([[np.cos(thetaP)],[np.sin(thetaP)],[0.]])
    else:
        assert np.linalg.norm(e1) > 0 and np.sum(nP*e1)[0]<1e-15, "Arg e1 should be perpendicular to nP !"
    e1 = e1 - np.dot(nP,e1)*nP
    e1 = e1/np.linalg.norm(e1)

    if e2 is None:
        e2 = np.cross(nP.flatten(),e1.flatten())
    else:
        assert np.linalg.norm(e1) > 0 and np.sum(nP*e2)[0]<1e-15 and np.sum(e1*e2)[0]<1e-15, "Arg e2 should be perpendicular to nP !"
    e2 = e2 - np.sum(nP*e2)[0]*nP - np.sum(e1*e2)[0]*e1
    e2 = e2/np.linalg.norm(e2)

    NP = np.ceil(Rad/ds)
    dr = np.linspace(ds,Rad,NP).reshape((1,NP))
    Theta = Theta.flatten()
    Points = P
    for i in range(len(Theta)):
        PP = np.dot(P,np.ones((1,len(dr)))) + np.dot(e1,np.cos(Theta[i])*dr) + np.dot(e2,np.sin(Theta[i])*dr)
        Points = np.concatenate((Points,PP),axis=1)
    return Points, e1, e2, Theta, ds, Rad







"""
###############################################################################
###############################################################################
                Coordinates handling
###############################################################################
"""

def CoordShift(Points, In='(X,Y,Z)', Out='(R,Z)', CrossRef=0.):
    """ Check the shape of an array of points coordinates and/or converts from 2D to 3D, 3D to 2D, cylindrical to cartesian... (CrossRef is an angle (Tor) or a distance (X for Lin))"""
    assert all([ff in ['(X,Y,Z)','(R,phi,Z)','(R,Z)','(X,Y)','(Y,Z)'] for ff in [In,Out]]), "Arg In and Out (coordinate format) must be in ['(X,Y,Z)','(R,phi,Z)','(R,Z)','(X,Y)','(Y,Z)'] !"
    assert isinstance(Points,np.ndarray) and Points.ndim==2 and Points.shape[0] in (2,3), "Points must be a 2D np.ndarray !"
    NP = Points.shape[1]
    if Points.shape[0]==2:
        assert In in ['(R,Z)','(X,Y)','(Y,Z)'], "Inconsistent input !"
    if Out==In:
        return Points

    if Out=='(R,phi,Z)':
        assert In in ['(R,Z)','(X,Y,Z)','(X,Y)'], "Not compatible !"
        if In=='(R,Z)':
            return np.array([Points[0,:], CrossRef*np.ones((NP,)), Points[1,:]])
        elif In=='(X,Y,Z)':
            R = np.hypot(Points[0,:],Points[1,:])
            Theta = np.arctan2(Points[1,:],Points[0,:])
            return np.array([R,Theta,Points[2,:]])
        elif In=='(X,Y)':
            R = np.hypot(Points[0,:],Points[1,:])
            Theta = np.arctan2(Points[1,:],Points[0,:])
            return np.array([R,Theta,CrossRef*np.ones((NP,))])
    elif Out=='(X,Y,Z)':
        assert In in ['(R,Z)','(X,Y)','(Y,Z)','(R,phi,Z)'], "Not compatible !"
        if In=='(R,Z)':
            return np.array([Points[0,:]*np.cos(CrossRef), Points[0,:]*np.sin(CrossRef), Points[1,:]])
        elif In=='(X,Y)':
            return np.array([Points[0,:],Points[1,:],CrossRef*np.ones((NP,))])
        elif In=='(Y,Z)':
            return np.array([CrossRef*np.ones((NP,)), Points[0,:], Points[1,:]])
        elif In=='(R,phi,Z)':
            return np.array([Points[0,:]*np.cos(Points[1,:]), Points[0,:]*np.sin(Points[1,:]), Points[2,:]])
    elif Out=='(Y,Z)':
        assert In in ['(X,Y,Z)','(X,Y)'], "Not compatible !"
        if In=='(X,Y,Z)':
            return Points[1:,:]
        elif In=='(X,Y)':
            return np.array([Points[1,:], CrossRef*np.ones((NP,))])
    elif Out=='(R,Z)':
        assert In in ['(R,phi,Z)','(X,Y,Z)'], "Not compatible !"
        if In=='(R,phi,Z)':
            return Points[0::2,:]
        elif In=='(X,Y,Z)':
            return np.array([np.hypot(Points[0,:],Points[1,:]), Points[2,:]])
        elif In=='(X,Y)':
            return np.array([np.hypot(Points[0,:],Points[1,:]), CrossRef*np.ones((NP,))])
    elif Out=='(X,Y)':
        assert In in ['(X,Y,Z)','(Y,Z)','(R,phi,Z)'], "Not compatible !"
        if In=='(X,Y,Z)':
            return Points[:-1,:]
        elif In=='(Y,Z)':
            return np.array([CrossRef*np.ones((NP,)), Points[0,:]])
        elif '(R,phi,Z)':
            X = Points[0,:]*np.cos(Points[1,:])
            Y = Points[0,:]*np.sin(Points[1,:])
            return np.array([X,Y])



            
def CoordShift_1D(DTYPE_t [::1] Point, str In='(X,Y,Z)', str Out='(R,Z)', DTYPE_t CrossRef=0.):
    """ Check the shape of an array of Point coordinates and/or converts from 2D to 3D, 3D to 2D, cylindrical to cartesian... """
    assert all([ff in ['(X,Y,Z)','(R,phi,Z)','(R,Z)','(Y,Z)'] for ff in [In,Out]]), "Arg In and Out (coordinate format) must be in ['(X,Y,Z)','(R,Theta,Z)','(R,Z)'] !"
    assert Point.size in [2,3], "Arg Poly must be a (2,) or (3,) np.ndarray !"
    if Point.shape[0]==2:
        In = '(R,Z)'
    if Out==In:
        return Point
    if Out=='(R,phi,Z)':
        if In=='(R,Z)':
            return np.array([Point[0], CrossRef, Point[1]])
        elif In=='(X,Y,Z)':
            R = np.hypot(Point[0],Point[1])
            Theta = np.arctan2(Point[1],Point[0])
            return np.array([R,Theta,Point[2]])
    elif Out=='(X,Y,Z)':
        if In=='(R,Z)':
            return np.array([Point[0]*np.cos(CrossRef), Point[0]*np.sin(CrossRef), Point[1]])
        elif In=='(Y,Z)':
            return np.array([CrossRef, Point[0], Point[1]])
        elif In=='(R,phi,Z)':
            return np.array([Point[0]*np.cos(Point[1]), Point[0]*np.sin(Point[1]), Point[2]])
    elif Out=='(Y,Z)':
        if In=='(X,Y,Z)':
            return Point[1:]
    elif Out=='(R,Z)':
        if In=='(R,phi,Z)':
            return Point[0::2]
        elif In=='(X,Y,Z)':
            return np.array([np.hypot(Point[0],Point[1]), Point[2]])


# For sinograms
def ConvertImpact_Theta2Xi(theta, pP, pN, sort=True):
    if hasattr(theta,'__getitem__'):
        pP, pN, theta = np.asarray(pP), np.asarray(pN), np.asarray(theta)
        assert pP.shape==pN.shape==theta.shape, "Args pP, pN and theta must have same shape !"
        pPbis, pNbis = np.copy(pP), np.copy(pN)
        xi = theta - np.pi/2.
        ind = xi < 0
        pPbis[ind], pNbis[ind], xi[ind] = -pN[ind], -pP[ind], xi[ind]+np.pi
        if sort:
            ind = np.argsort(xi)
            xi, pP, pN = xi[ind], pPbis[ind], pNbis[ind]
        return xi, pP, pN
    else:
        assert not (hasattr(pP,'__getitem__') or hasattr(pN,'__getitem__')), "Args pP, pN and theta must have same shape !"
        xi = theta - np.pi/2.
        if xi < 0.:
            return xi+np.pi, -pN, -pP
        else:
            return xi, pP, pN




def Unique_Eps(DTYPE_t[::1] X, float epsx):
    cdef list Xbis = []
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] A
    cdef Py_ssize_t ii
    for ii in range(0,X.size):
        if not any([Cabs(xx-X[ii])<epsx for xx in Xbis]):
            Xbis.append(X[ii])
    A = np.unique(Xbis)
    return A


def Remap_2DFromFlat(DTYPE_t[:,::1] Pts, list LQuant, epsx0=None, epsx1=None):
    cdef Py_ssize_t NP = Pts.shape[1]
    cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] X0, X1
    X0 = np.unique(Pts[0,:]) if epsx0 is None else Unique_Eps(Pts[0,:], epsx0)
    X1 = np.unique(Pts[1,:]) if epsx1 is None else Unique_Eps(Pts[1,:], epsx1)
    cdef int nx0 = X0.size, nx1 = X1.size
    cdef list LQbis
    LQbis = _Remap_2DFromFlat_bis(Pts, LQuant, X0, X1, nx0, nx1, NP) if (epsx0 is None and epsx1 is None) else Remap_2DFromFlat_bis_eps(Pts, LQuant, X0, X1, nx0, nx1, NP, epsx0, epsx1)
    return LQbis, X0, X1, nx0, nx1



def _Remap_2DFromFlat_bis(DTYPE_t[:,::1] Pts, list LQuant, DTYPE_t[::1] X0, DTYPE_t[::1] X1, int nx0, int nx1, int NP):
    cdef Py_ssize_t ii, jj, indi, indj
    cdef Py_ssize_t NQ = len(LQuant)
    assert all([LQuant[ii].size==NP for ii in range(0,NQ)]), "All provided quantities must be same size as number of points !"
    cdef list LQbis = [np.nan*np.ones((nx0,nx1)) for ii in range(0,NQ)]
    for ii in xrange(0,NP):
        indi = np.searchsorted(X0,Pts[0,ii])
        indj = np.searchsorted(X1,Pts[1,ii])
        for jj in range(0,NQ):
            LQbis[jj][indi,indj] = LQuant[jj][ii]
    return LQbis

def Remap_2DFromFlat_bis_eps(DTYPE_t[:,::1] Pts, list LQuant, np.ndarray[DTYPE_t, ndim=1,mode='c'] X0, np.ndarray[DTYPE_t, ndim=1,mode='c'] X1, int nx0, int nx1, int NP, float epsx0, float epsx1):
    cdef Py_ssize_t ii, jj, indi, indj
    cdef Py_ssize_t NQ = len(LQuant)
    assert all([LQuant[ii].size==NP for ii in range(0,NQ)]), "All provided quantities must be same size as numer of points !"
    cdef list LQbis = [np.nan*np.ones((nx0,nx1)) for ii in range(0,NQ)]
    for ii in xrange(0,NP):
        indi = (np.abs(X0-Pts[0,ii])<epsx0).nonzero()[0][0]
        indj = (np.abs(X1-Pts[1,ii])<epsx1).nonzero()[0][0]
        for jj in range(0,NQ):
            LQbis[jj][indi,indj] = LQuant[jj][ii]
    return LQbis




"""
###############################################################################
###############################################################################
                    Integral functions
###############################################################################
"""

########### 2D integrals ################

def dblquad_custom(func, a, b, gfun, hfun, args=(), epsabs=0., epsrel=1e-4, limit=1000, pointsx1=None, pointsx2=None):
    def _infunc(x,func,gfun,hfun,more_args):
        a = gfun(x)
        b = hfun(x)
        myargs = (x,) + more_args
        return scpinteg.quad(func,a,b,args=myargs, epsabs=0., epsrel=epsrel, limit=limit, points=pointsx2)[0]

    return scpinteg.quad(_infunc,a,b,(func,gfun,hfun,args),epsabs=epsabs,epsrel=epsrel, limit=limit, points=pointsx1)

def dblsimps_custom(z,x=None,y=None, dx=1,dy=1, axis=-1,even='avg'):
    assert isinstance(z,np.ndarray) and (x is None or z.shape[0]==len(x)) and (y is None or z.shape[1]==len(y)), "Arg z should be a (Nx,Ny) ndarray !"
    if not np.any(z>0.):
        return 0.
    return scpinteg.simps(scpinteg.simps(z,x=y,dx=dy,axis=axis,even=even),x=x,dx=dx,axis=axis,even=even)

def dbltrapz_custom(z,x=None,y=None, dx=1,dy=1, axis=-1):
    assert isinstance(z,np.ndarray) and (x is None or z.shape[0]==len(x)) and (y is None or z.shape[1]==len(y)), "Arg z should be a (Nx,Ny) ndarray !"
    if not np.any(z>0.):
        return 0.
    return scpinteg.trapz(scpinteg.trapz(z,x=y,dx=dy,axis=axis),x=x,dx=dx,axis=axis)

def dblnptrapz_custom(z,x=None,y=None, dx=1,dy=1, axis=-1):
    assert isinstance(z,np.ndarray) and (x is None or z.shape[0]==len(x)) and (y is None or z.shape[1]==len(y)), "Arg z should be a (Nx,Ny) ndarray !"
    if not np.any(z>0.):
        return 0.
    return np.trapz(np.trapz(z,x=y,dx=dy,axis=axis),x=x,dx=dx,axis=axis)


########### 3D integrals ################


def tplquad_custom(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=0., epsrel=1e-4, limit=1000, pointsx1=None, pointsx2=None, pointsx3=None):
    def _infunc2(y,x,func,qfun,rfun,more_args):
        a2 = qfun(x,y)
        b2 = rfun(x,y)
        myargs = (y,x) + more_args
        return scpinteg.quad(func,a2,b2, args=myargs, epsabs=epsabs,epsrel=epsrel, limit=limit, points=pointsx3)[0]
    return dblquad_custom(_infunc2,a,b,gfun,hfun,(func,qfun,rfun,args),epsabs=epsabs,epsrel=epsrel, limit=limit, pointsx1=pointsx1, pointsx2=pointsx2)


def tplsimps_custom(Val,x=None,y=None,z=None, dx=1,dy=1,dz=1,axis=-1,even='avg'):
    assert isinstance(Val,np.ndarray) and (x is None or Val.shape[0]==len(x)) and (y is None or Val.shape[1]==len(y)) and (z is None or Val.shape[2]==len(z)), "Arg Val should be a (Nx,Ny,Nz) ndarray !"
    if not np.any(np.abs(Val)>0.):
        return 0.
    intz = scpinteg.simps(Val,x=z,dx=dz,axis=-1,even=even)
    intyz = scpinteg.simps(intz,x=y,dx=dy,axis=-1,even=even)
    return scpinteg.simps(intyz,x=x,dx=dx,axis=-1,even=even)


def tpltrapz_custom(Val,x=None,y=None,z=None, dx=1,dy=1,dz=1,axis=-1):
    assert isinstance(Val,np.ndarray) and (x is None or Val.shape[0]==len(x)) and (y is None or Val.shape[1]==len(y)) and (z is None or Val.shape[2]==len(z)), "Arg Val should be a (Nx,Ny,Nz) ndarray !"
    if not np.any(np.abs(Val)>0.):
        return 0.
    intz = scpinteg.trapz(Val,x=z,dx=dz,axis=-1)
    intyz = scpinteg.trapz(intz,x=y,dx=dy,axis=-1)
    return scpinteg.trapz(intyz,x=x,dx=dx,axis=-1)

def tplnptrapz_custom(Val,x=None,y=None,z=None, dx=1,dy=1,dz=1,axis=-1):
    assert isinstance(Val,np.ndarray) and (x is None or Val.shape[0]==len(x)) and (y is None or Val.shape[1]==len(y)) and (z is None or Val.shape[2]==len(z)), "Arg Val should be a (Nx,Ny,Nz) ndarray !"
    if not np.any(np.abs(Val)>0.):
        return 0.
    intz = np.trapz(Val,x=z,dx=dz,axis=-1)
    intyz = np.trapz(intz,x=y,dx=dy,axis=-1)
    return np.trapz(intyz,x=x,dx=dx,axis=-1)









    
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
"""
                          Toroidal geometry, Projections and LOS
"""
#######################################################################################################
#######################################################################################################
#######################################################################################################


def Calc_VolBaryV_CylPol(np.ndarray[DTYPE_t, ndim=2,mode='c'] Poly):
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] Ri0=Poly[0,:-1], Ri1=Poly[0,1:], Zi0=Poly[1,:-1], Zi1=Poly[1,1:]
    cdef DTYPE_t V = np.sum((Ri0*Zi1 - Zi0*Ri1)*(Ri0+Ri1))/6.
    cdef DTYPE_t BV0 = np.sum((Ri0*Zi1 - Zi0*Ri1) * 0.5 * (Ri1**2 + Ri1*Ri0 + Ri0**2)) / (6.*V)
    cdef DTYPE_t BV1 = -np.sum((Ri1**2*Zi0*(2.*Zi1+Zi0) + 2.*Ri0*Ri1*(Zi0**2-Zi1**2) - Ri0**2*Zi1*(Zi1+2.*Zi0))/4.)/(6.*V)
    return V, np.array([BV0,BV1])



def Calc_PolProj_LOS_cy(np.ndarray[DTYPE_t, ndim=2,mode='c'] D, np.ndarray[DTYPE_t, ndim=2,mode='c'] Du, kmax=np.inf):
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] uN = np.sqrt(Du[0,:]*Du[0,:]+Du[1,:]*Du[1,:]+Du[2,:]*Du[2,:])
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] RD = np.sqrt(D[0,:]*D[0,:] + D[1,:]*D[1,:])
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] uParN = np.sqrt(Du[0,:]*Du[0,:]+Du[1,:]*Du[1,:])
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] sca = D[0,:]*Du[0,:]+D[1,:]*Du[1,:]
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] cos = sca/(RD*uParN)
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] PolProjAng = np.arccos(uParN/uN)
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] k1
    cdef np.ndarray ind0, ind1, ind
    cdef int N = D.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] PRMin = np.nan*np.ones((3,N))
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] RMin = np.nan*np.ones((N,))
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] kRMin = np.nan*np.ones((N,))
    
    if type(kmax) is float:
        kmax = kmax*np.ones((N,))
    if D.shape[1]==1:
        kmax = np.array([kmax])

    ind0 = np.logical_or(uParN==0., -RD*cos*uParN<0)
    PRMin[:,ind0] = D[:,ind0]
    RMin[ind0] = RD[ind0]
    kRMin[ind0] = 0.

    ind1 = np.logical_and(np.abs(uParN)>0., -RD*cos/uParN>kmax)
    PRMin[:,ind1] = D[:,ind1] + kmax[ind1]*Du[:,ind1]
    RMin[ind1] = np.sqrt(PRMin[0,ind1]*PRMin[0,ind1] + PRMin[1,ind1]*PRMin[1,ind1])
    kRMin[ind1] = kmax[ind1]

    ind = ~np.logical_or(ind0,ind1)
    k1 = -RD[ind]*cos[ind]/uParN[ind]
    PRMin[:,ind] = D[:,ind] + (np.ones((3,1)).dot(k1.reshape((1,np.sum(ind)))))*Du[:,ind]
    RMin[ind] = RD[ind]*np.sqrt(1-cos[ind]*cos[ind])
    kRMin[ind] = k1
    return PRMin, RMin, kRMin, PolProjAng


cdef Calc_Intersect_LineCone(DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t du0, DTYPE_t du1, DTYPE_t du2, DTYPE_t C0, DTYPE_t C1, DTYPE_t v0, DTYPE_t v1):
    cdef DTYPE_t uParN = (du0*du0 + du1*du1)**0.5
    cdef DTYPE_t vN = (v0*v0 + v1*v1)**0.5
    cdef DTYPE_t duN = (du0*du0 + du1*du1 + du2*du2)**0.5
    cdef DTYPE_t eps = 1e-10
    cdef DTYPE_t R, sin, thetau, theta, rapu, rapv, a, scal, b, c, Delta, x, y, z
    cdef DTYPE_t S10=0., S11=0., S12=0., S20=0., S21=0., S22=0.
    cdef int NS = 0

    if Cabs(v1/vN)<eps and Cabs(du2/duN) < eps:
        if D2==C1: 
            NS = 1
            S10, S11, S12 = D0,D1,D2
        #else:  # Impossible
    elif Cabs(v1/vN)<eps and not Cabs(du2/duN) < eps:
        NS = 1
        S10, S11, S12 = D0+du0*(C1-D2)/du2, D1+du1*(C1-D2)/du2, C1
    elif not Cabs(v1/vN)<eps and Cabs(du2/duN) < eps:
        R = (D2-C1)*v0/v1+C0
        if R==0. and Cabs(D0*du1 - D1*du0) < eps:
            NS = 1
            S10, S11, S12 = 0., 0., D2
        elif not R==0.:
            sin = (D0*du1-D1*du0)/(R*uParN)
            if sin == 0.:
                NS = 2
                S10, S11, S12 = R*du0/uParN, R*du1/uParN, D2
                S20, S21, S22 = -R*du0/uParN, -R*du1/uParN, D2
            else:
                if du1/duN < eps:
                    thetau = math.acos(du0/uParN)
                else:
                    thetau = Csign(du1)*math.acos(du0/uParN)
                if Cabs(sin) == 1.:
                    NS = 1
                    theta = thetau-math.asin(sin)
                    S10, S11, S12 = R*math.cos(theta), R*math.sin(theta), D2
                elif Cabs(sin) < 1.:
                    NS = 2
                    theta = thetau-math.asin(sin)
                    S10, S11, S12 = R*math.cos(theta), R*math.sin(theta), D2
                    theta = thetau-(np.pi-math.asin(sin))
                    S20, S21, S22 = R*math.cos(theta), R*math.sin(theta), D2
    else:
        rapu = uParN/du2
        rapv = v0/v1
        a = rapu*rapu-rapv*rapv
        scal = D0*du0+D1*du1
        b = 2.*(scal/du2-C0*rapv-D2*rapu*rapu + C1*rapv*rapv)
        c = D0*D0 + D1*D1 - C0*C0 + D2*D2*rapu*rapu - 2.*D2*scal/du2 - C1*C1*rapv*rapv + 2*C0*C1*rapv
        Delta = b*b-4.*a*c
        if a == 0. and not b==0.:
            NS = 1
            z = -c/b
            x = D0 + (z-D2)*du0/du2
            y = D1 + (z-D2)*du1/du2
            S10, S11, S12 = x, y, z
        elif not a==0.:
            if Delta == 0.:
                NS = 1
                z = -b/(2.*a)
                x = D0 + (z-D2)*du0/du2
                y = D1 + (z-D2)*du1/du2
                S10, S11, S12 = x, y, z
            elif Delta > 0.:
                NS = 2
                z = (-b+Delta**0.5)/(2.*a)
                x = D0 + (z-D2)*du0/du2
                y = D1 + (z-D2)*du1/du2
                S10, S11, S12 = x, y, z
                z = (-b-Delta**0.5)/(2.*a)
                x = D0 + (z-D2)*du0/du2
                y = D1 + (z-D2)*du1/du2
                S20, S21, S22 = x, y, z
    return ((S10,S11,S12),(S20,S21,S22)), NS


cdef Calc_InOut_LOS(DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t du0, DTYPE_t du1, DTYPE_t du2, DTYPE_t[:,::1] TPoly, DTYPE_t[:,::1] Vin):
    cdef Py_ssize_t N = TPoly.shape[1]-1
    cdef tuple Si
    cdef Py_ssize_t ii, jj
    cdef DTYPE_t R, SCA, sca, VN2
    cdef DTYPE_t Vect0, Vect1, TPoly0, TPoly1
    cdef list SIn = [], SOut = []
    cdef DTYPE_t Sij0, Sij1, Sij2
    cdef int NIn = 0, NOut = 0
    cdef int Ni
    for ii in xrange(0,N):
        TPoly0, TPoly1 = TPoly[0,ii],TPoly[1,ii]
        Vect0, Vect1 = TPoly[0,ii+1]-TPoly0, TPoly[1,ii+1]-TPoly1
        Si, Ni = Calc_Intersect_LineCone(D0,D1,D2, du0,du1,du2, TPoly0, TPoly1, Vect0,Vect1)
        if Ni>0:
            VN2 = Vect0*Vect0+Vect1*Vect1
            for jj in range(0,Ni):
                Sij0, Sij1, Sij2 = Si[jj]
                R = (Sij0*Sij0+Sij1*Sij1)**0.5
                SCA = du0*(Sij0-D0)+du1*(Sij1-D1)+du2*(Sij2-D2)
                sca = Vect0*(R-TPoly0) + Vect1*(Sij2-TPoly1)
                if SCA >= 0 and (sca >= 0 and sca < VN2 and Cabs((Vect0*(Sij2-TPoly1)-Vect1*(R-TPoly0)))<1.e-6*VN2**0.5): # Must be on the good side of D, aligned with TPoly[:,ii] and in the good interval
                    if du0*Sij0*Vin[0,ii]/R + du1*Sij1*Vin[0,ii]/R + du2*Vin[1,ii] > 0:
                        SIn.append([Sij0, Sij1, Sij2])
                        NIn += 1
                    else:
                        SOut.append([Sij0, Sij1, Sij2])
                        NOut += 1
    return SIn, SOut, NIn, NOut


cdef Calc_InOut_LOS_Lin(DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t du0, DTYPE_t du1, DTYPE_t du2, DTYPE_t[:,::1] TPoly, DTYPE_t[:,::1] Vin, list DLong):
    cdef Py_ssize_t N = TPoly.shape[1]-1
    cdef Py_ssize_t ii
    cdef DTYPE_t P0,P1,P2, Vect0,Vect1,Vect2, v0,v1,v2, Li2, M0, M1, M2
    cdef DTYPE_t k, scauv, scadp, scaPM, scaIn
    cdef list SIn = [], SOut = []
    cdef int NIn = 0, NOut = 0

    # For cylinder
    for ii in xrange(0,N):
        P0, P1, P2 = DLong[0], TPoly[0,ii], TPoly[1,ii]
        Vect0, Vect1, Vect2 = 0., TPoly[0,ii+1]-P1, TPoly[1,ii+1]-P2
        Li2 = Vect0**2 + Vect1**2 + Vect2**2
        v0, v1, v2 = 0., Vin[0,ii], Vin[1,ii]
        
        scauv = du0*v0 + du1*v1 + du2*v2
        scadp = (P0-D0)*v0 + (P1-D1)*v1 + (P2-D2)*v2
        if scauv==0. and scadp==0.: # Case when line parallel to and included in plane
            k = 0.
        elif not scauv==0.: # Case with standard intersection
            k = scadp / scauv
        if k>=0.:   # Only if on good side of D
            M0, M1, M2 = D0+k*du0, D1+k*du1, D2+k*du2
            scaPM = (M0-P0)*Vect0 + (M1-P1)*Vect1 + (M2-P2)*Vect2
            if scaPM>=0 and scaPM<Li2 and M0>=DLong[0] and M0<=DLong[1]: # Only if in good interval of Poly and of DLong
                scaIn = (M0-D0)*v0 + (M1-D1)*v1 + (M2-D2)*v2
                if scaIn>=0:
                    SIn.append([M0,M1,M2])
                    NIn += 1
                else:
                    SOut.append([M0,M1,M2])
                    NOut += 1

    # For two faces
    P0, P1, P2 = DLong[0], TPoly[0,0], TPoly[1,0]
    v0, v1, v2 = 1.,0.,0.
    scauv = du0*v0 + du1*v1 + du2*v2
    scadp = (P0-D0)*v0 + (P1-D1)*v1 + (P2-D2)*v2
    if scauv==0. and scadp==0.: # Case when line parallel to and included in plane
        k = 0.
    elif not scauv==0.: # Case with standard intersection
        k = scadp / scauv
    if k>=0.:   # Only if on good side of D
        M0, M1, M2 = D0+k*du0, D1+k*du1, D2+k*du2
        if plg.Polygon(TPoly.T).isInside(M1,M2):
            scaIn = (M0-D0)*v0 + (M1-D1)*v1 + (M2-D2)*v2
            if scaIn>=0:
                SIn.append([M0,M1,M2])
                NIn += 1
            else:
                SOut.append([M0,M1,M2])
                NOut += 1

    P0, P1, P2 = DLong[1], TPoly[0,0], TPoly[1,0]
    v0, v1, v2 = -1.,0.,0.
    scauv = du0*v0 + du1*v1 + du2*v2
    scadp = (P0-D0)*v0 + (P1-D1)*v1 + (P2-D2)*v2
    if scauv==0. and scadp==0.: # Case when line parallel to and included in plane
        k = 0.
    elif not scauv==0.: # Case with standard intersection
        k = scadp / scauv
    if k>=0.:   # Only if on good side of D
        M0, M1, M2 = D0+k*du0, D1+k*du1, D2+k*du2
        if plg.Polygon(TPoly.T).isInside(M1,M2):
            scaIn = (M0-D0)*v0 + (M1-D1)*v1 + (M2-D2)*v2
            if scaIn>=0:
                SIn.append([M0,M1,M2])
                NIn += 1
            else:
                SOut.append([M0,M1,M2])
                NOut += 1

    return SIn, SOut, NIn, NOut


cdef Calc_InOut_LOS_ForbidArea_1D(DTYPE_t D0, DTYPE_t D1, DTYPE_t RMin, DTYPE_t Margin=0.1):
    cdef DTYPE_t Ang = math.atan2(D1,D0)
    # Add a little bit of margin to the major radius, just in case
    cdef DTYPE_t R = math.sqrt(D0**2+D1**2)*(1+Margin*RMin)
    cdef DTYPE_t L = math.sqrt(R**2-RMin**2)
    # Compute new (X,Y) coordinates with the margin
    cdef DTYPE_t X = R*math.cos(Ang), Y = R*math.sin(Ang)
    # Compute coordinates of the 2 points where the tangents touch the inner circle
    cdef DTYPE_t S1X = (RMin**2*X+RMin*Y*L)/R**2, S1Y = (RMin**2*Y-RMin*X*L)/R**2
    cdef DTYPE_t S2X = (RMin**2*X-RMin*Y*L)/R**2, S2Y = (RMin**2*Y+RMin*X*L)/R**2
    # Compute the vectors perpendicular to the 3 planes
    cdef DTYPE_t V0X = -X, V0Y = -Y
    cdef DTYPE_t V1X = -S1X, V1Y = -S1Y
    cdef DTYPE_t V2X = -S2X, V2Y = -S2Y
    # Check good orientation of vectors 1 and 2 (should be towards center)
    #if -S1X*V1X-S1Y*V1Y<0:
    #    V1X = -V1X
    #    V1Y = -V1Y
    #if -S2X*V2X-S2Y*V2Y<0:
    #    V2X = -V2X
    #    V2Y = -V2Y
    return S1X,S1Y, S2X,S2Y, V0X,V0Y, V1X,V1Y, V2X,V2Y

def Calc_InOut_LOS_ForbidArea_2D(np.ndarray[DTYPE_t, ndim=2] D, DTYPE_t RMin, DTYPE_t Margin=0.1):
    cdef np.ndarray[DTYPE_t, ndim=1] Ang = np.arctan2(D[1,:],D[0,:])
    # Add a little bit of margin to the major radius, just in case
    cdef np.ndarray[DTYPE_t, ndim=1] R = np.hypot(D[0,:],D[1,:])*(1+Margin*RMin)
    cdef np.ndarray[DTYPE_t, ndim=1] L = np.sqrt(R**2-RMin**2)
    # Compute new (X,Y) coordinates with the margin
    cdef np.ndarray[DTYPE_t, ndim=1] X = R*np.cos(Ang), Y = R*np.sin(Ang)
    # Compute coordinates of the 2 points where the tangents touch the inner circle
    cdef np.ndarray[DTYPE_t, ndim=2] S1 = np.array([(RMin**2*X+RMin*Y*L)/R**2, (RMin**2*Y-RMin*X*L)/R**2])
    cdef np.ndarray[DTYPE_t, ndim=2] S2 = np.array([(RMin**2*X-RMin*Y*L)/R**2, (RMin**2*Y+RMin*X*L)/R**2])
    # Compute the vectors perpendicular to the 3 planes
    cdef np.ndarray[DTYPE_t, ndim=2] V0 = -np.array([X, Y])
    cdef np.ndarray[DTYPE_t, ndim=2] V1 = -S1
    cdef np.ndarray[DTYPE_t, ndim=2] V2 = -S2
    # Check good orientation of vectors 1 and 2 (should be towards center)
    #ind = -np.sum(S1*V1,axis=0)<0
    #V1[:,ind] = -V1[:,ind]
    #ind = -np.sum(S2*V2,axis=0)<0
    #V2[:,ind] = -V2[:,ind]
    return S1, S2, V0, V1, V2


cdef Calc_InOut_LOS_ForbidArea_Check_1D(DTYPE_t PX, DTYPE_t PY, DTYPE_t S1X,DTYPE_t S1Y, DTYPE_t S2X,DTYPE_t S2Y, DTYPE_t V0X,DTYPE_t V0Y, DTYPE_t V1X,DTYPE_t V1Y, DTYPE_t V2X,DTYPE_t V2Y):
    """ Check if a point is potentially visible (i.e.: not in the forbidden area) """
    ind = (PX-S1X)*V0X+(PY-S1Y)*V0Y > 0 and (PX-S1X)*V1X+(PY-S1Y)*V1Y > 0 and (PX-S2X)*V2X+(PY-S2Y)*V2Y > 0
    return not ind

def Calc_InOut_LOS_ForbidArea_Check_2D(np.ndarray[DTYPE_t, ndim=2] P, np.ndarray[DTYPE_t, ndim=1] S1, np.ndarray[DTYPE_t, ndim=1] S2, np.ndarray[DTYPE_t, ndim=1] V0, np.ndarray[DTYPE_t, ndim=1]  V1, np.ndarray[DTYPE_t, ndim=1] V2):
    """ Check if any number of points are potentially visible (i.e.: not in the forbidden area) """
    ind0 = (P[0,:]-S1[0])*V0[0]+(P[1,:]-S1[1])*V0[1] > 0
    ind1 = (P[0,:]-S1[0])*V1[0]+(P[1,:]-S1[1])*V1[1] > 0
    ind2 = (P[0,:]-S2[0])*V2[0]+(P[1,:]-S2[1])*V2[1] > 0
    # Indices of the points lying in the forbidden area
    ind = ind0 & ind1 & ind2
    return ~ind


def Calc_InOut_LOS_PIO(np.ndarray[DTYPE_t, ndim=2] D, np.ndarray[DTYPE_t, ndim=2] du, np.ndarray[DTYPE_t, ndim=2] TPoly, np.ndarray[DTYPE_t, ndim=2] Vin, Forbid=True, DTYPE_t Margin=0.1):
    cdef Py_ssize_t NIn, NOut, ii, ind
    cdef Py_ssize_t NL = D.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] SIn = np.nan*np.ones((3,NL)), SOut = np.nan*np.ones((3,NL))
    cdef list Sin, Sout
    cdef list kin, kout
    cdef DTYPE_t koutfl
    # If Forbid, prepare the forbidden area parameters
    if Forbid:
        RMin = 0.95*min(np.nanmin(TPoly[0,:]), np.nanmin(np.hypot(D[0,:],D[1,:])))
        S1, S2, V0, V1, V2 = Calc_InOut_LOS_ForbidArea_2D(D, RMin, Margin=Margin)
    for ii in xrange(0,NL):
        Sin, Sout, NIn, NOut = Calc_InOut_LOS(D[0,ii],D[1,ii],D[2,ii], du[0,ii],du[1,ii],du[2,ii], TPoly, Vin)
        if NOut>0:
            kout = [(sout[0]-D[0,ii])*du[0,ii] + (sout[1]-D[1,ii])*du[1,ii] + (sout[2]-D[2,ii])*du[2,ii] for sout in Sout]
            if any([kk>0 for kk in kout]):
                koutfl = min([kk if kk>0 else np.inf for kk in kout])
                SOut[0,ii], SOut[1,ii], SOut[2,ii] = Sout[kout.index(koutfl)]
        else:
            koutfl = np.inf
        if NIn>0:
            kin = [(sin[0]-D[0,ii])*du[0,ii] + (sin[1]-D[1,ii])*du[1,ii] + (sin[2]-D[2,ii])*du[2,ii] for sin in Sin]
            if any([kk>0 and kk<koutfl for kk in kin]):
                ind = kin.index(min([kk if kk>0 and kk<koutfl else np.inf for kk in kin]))
                SIn[0,ii], SIn[1,ii], SIn[2,ii] = Sin[ind]
        # If Forbid, check whether each SIn and SOut lies in the forbidden area and suppress them if so
        if Forbid:
            indF = Calc_InOut_LOS_ForbidArea_Check_2D(SIn, S1[:,ii], S2[:,ii], V0[:,ii], V1[:,ii], V2[:,ii])
            SIn[:,~indF] = np.nan
            indF = Calc_InOut_LOS_ForbidArea_Check_2D(SOut, S1[:,ii], S2[:,ii], V0[:,ii], V1[:,ii], V2[:,ii])
            SOut[:,~indF] = np.nan
    return SIn, SOut

def Calc_InOut_LOS_PIO_Lin(np.ndarray[DTYPE_t, ndim=2] D, np.ndarray[DTYPE_t, ndim=2] du, np.ndarray[DTYPE_t, ndim=2] TPoly, np.ndarray[DTYPE_t, ndim=2] Vin, list DLong):
    cdef Py_ssize_t NIn, NOut, ii, ind
    cdef Py_ssize_t NL = D.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] SIn = np.nan*np.ones((3,NL)), SOut = np.nan*np.ones((3,NL))
    cdef list Sin, Sout
    cdef list kin, kout
    cdef DTYPE_t koutfl

    for ii in xrange(0,NL):
        Sin, Sout, NIn, NOut = Calc_InOut_LOS_Lin(D[0,ii],D[1,ii],D[2,ii], du[0,ii],du[1,ii],du[2,ii], TPoly, Vin, DLong)
        if NOut>0:
            kout = [(sout[0]-D[0,ii])*du[0,ii] + (sout[1]-D[1,ii])*du[1,ii] + (sout[2]-D[2,ii])*du[2,ii] for sout in Sout]
            if any([kk>0 for kk in kout]):
                koutfl = min([kk if kk>0 else np.inf for kk in kout])
                SOut[0,ii], SOut[1,ii], SOut[2,ii] = Sout[kout.index(koutfl)]
        else:
            koutfl = np.inf
        if NIn>0:
            kin = [(sin[0]-D[0,ii])*du[0,ii] + (sin[1]-D[1,ii])*du[1,ii] + (sin[2]-D[2,ii])*du[2,ii] for sin in Sin]
            if any([kk>0 and kk<koutfl for kk in kin]):
                ind = kin.index(min([kk if kk>0 and kk<koutfl else np.inf for kk in kin]))
                SIn[0,ii], SIn[1,ii], SIn[2,ii] = Sin[ind]
    return SIn, SOut

def Calc_InOut_LOS_Colis_1D(DTYPE_t BS0, DTYPE_t BS1, DTYPE_t BS2, DTYPE_t Pp0, DTYPE_t Pp1, DTYPE_t Pp2, np.ndarray[DTYPE_t, ndim=2] TPoly, np.ndarray[DTYPE_t, ndim=2] Vin, list DLong=[], str VType='Tor', Forbid=True, DTYPE_t Margin=0.1):
    cdef int NIn, NOut
    cdef Py_ssize_t ii
    cdef list       Sin, Sout
    cdef list       sout
    cdef DTYPE_t    BP0 = Pp0-BS0, BP1 = Pp1-BS1, BP2 = Pp2-BS2
    cdef DTYPE_t    kP2 = np.sqrt(BP0**2 + BP1**2 + BP2**2)   # np.sqrt ?
    cdef DTYPE_t    du0 = BP0/kP2, du1 = BP1/kP2, du2 = BP2/kP2
    cdef DTYPE_t    kP = BP0*du0 + BP1*du1 + BP2*du2
    Sin, Sout, NIn, NOut = Calc_InOut_LOS(BS0,BS1,BS2, du0, du1, du2, TPoly, Vin) if VType=='Tor' else Calc_InOut_LOS_Lin(BS0,BS1,BS2, du0, du1, du2, TPoly, Vin, DLong)
    if NOut==0:
        return False
    else:
        for sout in Sout:
            if (sout[0]-BS0)*du0 + (sout[1]-BS1)*du1 + (sout[2]-BS2)*du2 < kP:
                return False
    if VType=='Tor' and Forbid:
        # Check if Pp is on the good side of the Torus
        RMin = 0.95*min(np.nanmin(TPoly[0,:]),math.sqrt(BS0**2+BS1**2))
        S1X,S1Y, S2X,S2Y, V0X,V0Y, V1X,V1Y, V2X,V2Y = Calc_InOut_LOS_ForbidArea_1D(BS0,BS1,RMin,Margin=Margin)
        ind = Calc_InOut_LOS_ForbidArea_Check_1D(Pp0,Pp1, S1X,S1Y, S2X,S2Y, V0X,V0Y, V1X,V1Y, V2X,V2Y)
        if not ind:
            return False
    return True


def Calc_InOut_LOS_Colis(DTYPE_t[::1] BaryS, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=2] TPoly, np.ndarray[DTYPE_t, ndim=2] Vin, Forbid=True, DTYPE_t Margin=0.1):
    cdef int NP = Points.shape[1]
    cdef DTYPE_t BS0 = BaryS[0], BS1 = BaryS[1], BS2 = BaryS[2]
    cdef Py_ssize_t ii
    cdef list   Sin, Sout
    cdef list  sout
    cdef list   ind = [1 for ii in range(0,NP)]
    cdef np.ndarray[DTYPE_t, ndim=1]  kP2 = np.sqrt((Points[0,:]-BS0)**2 + (Points[1,:]-BS1)**2 + (Points[2,:]-BS2)**2)      # Power function or simple product ?
    cdef tuple   du = ((Points[0,:]-BS0)/kP2, (Points[1,:]-BS1)/kP2, (Points[2,:]-BS2)/kP2)
    cdef DTYPE_t du0, du1, du2, kk
    cdef DTYPE_t[::1]  kP = (Points[0,:]-BS0)*du[0] + (Points[1,:]-BS1)*du[1] + (Points[2,:]-BS2)*du[2]
    
    if Forbid:
        # Check which points are on the good side of the Torus
        RMin = 0.95*min(np.nanmin(TPoly[0,:]),np.hypot(BaryS[0],BaryS[1]))
        S1X,S1Y, S2X,S2Y, V0X,V0Y, V1X,V1Y, V2X,V2Y = Calc_InOut_LOS_ForbidArea_1D(BaryS[0], BaryS[1], RMin, Margin=Margin)
        indF = Calc_InOut_LOS_ForbidArea_Check_2D(Points,
                                                  np.array([S1X,S1Y]),np.array([S2X,S2Y]),
                                                  np.array([V0X,V0Y]),np.array([V1X,V1Y]),
                                                  np.array([V2X,V2Y]))
    else:
        indF = np.ones((NP,),dtype=bool)

    for ii in xrange(0,NP):
        if not indF[ii]:
            ind[ii] = 0
        else:
            du0, du1, du2 = du[0][ii],du[1][ii],du[2][ii]
            Sin, Sout, NIn, NOut = Calc_InOut_LOS(BS0,BS1,BS2, du0, du1, du2, TPoly, Vin)
            if NOut==0:
                ind[ii] = 0
            else:
                for sout in Sout:
                    kk = (sout[0]-BS0)*du0 + (sout[1]-BS1)*du1 + (sout[2]-BS2)*du2
                    if kk>0 and kk<kP[ii]:
                        ind[ii] = 0
                        break
    return np.asarray(ind,dtype=bool)

def Calc_InOut_LOS_Colis_Lin(DTYPE_t[::1] BaryS, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=2] VPoly, np.ndarray[DTYPE_t, ndim=2] Vin, list DLong):
    cdef int NP = Points.shape[1]
    cdef DTYPE_t BS0 = BaryS[0], BS1 = BaryS[1], BS2 = BaryS[2]
    cdef Py_ssize_t ii
    cdef list   Sin, Sout
    cdef list  sout
    cdef list   ind = [1 for ii in range(0,NP)]
    cdef np.ndarray[DTYPE_t, ndim=1]  kP2 = np.sqrt((Points[0,:]-BS0)**2 + (Points[1,:]-BS1)**2 + (Points[2,:]-BS2)**2)      # Power function or simple product ?
    cdef tuple   du = ((Points[0,:]-BS0)/kP2, (Points[1,:]-BS1)/kP2, (Points[2,:]-BS2)/kP2)
    cdef DTYPE_t du0, du1, du2, kk
    cdef DTYPE_t[::1]  kP = (Points[0,:]-BS0)*du[0] + (Points[1,:]-BS1)*du[1] + (Points[2,:]-BS2)*du[2]

    for ii in xrange(0,NP):
        du0, du1, du2 = du[0][ii],du[1][ii],du[2][ii]
        Sin, Sout, NIn, NOut = Calc_InOut_LOS_Lin(BS0,BS1,BS2, du0, du1, du2, VPoly, Vin, DLong)
        if NOut==0:
            ind[ii] = 0
        else:
            for sout in Sout:
                kk = (sout[0]-BS0)*du0 + (sout[1]-BS1)*du1 + (sout[2]-BS2)*du2
                if kk>0 and kk<kP[ii]:
                    ind[ii] = 0
                    break
    return np.asarray(ind,dtype=bool)




"""
###############################################################################
###############################################################################
                    Sinogram specific
###############################################################################
"""


cdef findRootkPMin_ImpactLine(DTYPE_t uParN, DTYPE_t uN, DTYPE_t Sca, DTYPE_t RZ0, DTYPE_t RZ1, DTYPE_t ScaP, DTYPE_t DParN, DTYPE_t kOut, DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t u0, DTYPE_t u1, DTYPE_t u2, str Mode='LOS'):
    cdef DTYPE_t a4 = (uParN*uN*uN)**2, a3 = 2*( (Sca-RZ1*u2)*(uParN*uN)**2 + ScaP*uN**4 )
    cdef DTYPE_t a2 = (uParN*(Sca-RZ1*u2))**2 + 4.*ScaP*(Sca-RZ1*u2)*uN**2 + (DParN*uN*uN)**2 - (RZ0*uParN*uParN)**2
    cdef DTYPE_t a1 = 2*( ScaP*(Sca-RZ1*u2)**2 + (Sca-RZ1*u2)*(DParN*uN)**2 - ScaP*(RZ0*uParN)**2 )
    cdef DTYPE_t a0 = ((Sca-RZ1*u2)*DParN)**2 - (RZ0*ScaP)**2
    cdef np.ndarray roo = np.roots(np.array([a4,a3,a2,a1,a0]))
    cdef list KK = list(np.real(roo[np.isreal(roo)]))   # There might be several solutions
    cdef list Pk, Pk2D, rk
    cdef DTYPE_t kk, kPMin
    if Mode=='LOS':                     # Take solution on physical LOS
        if any([kk>=0 and kk<=kOut for kk in KK]):
            KK = [kk for kk in KK if kk>=0 and kk<=kOut]
            Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
            Pk2D = [(math.sqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
            rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
            kPMin = KK[rk.index(min(rk))]
        else:
            kPMin = min([Cabs(kk) for kk in KK])  # Else, take the one closest to D
    else:
        Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
        Pk2D = [(math.sqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
        rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
        kPMin = KK[rk.index(min(rk))]
    return kPMin

cdef Calc_Impact_Line_1D_Fast(DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t u0, DTYPE_t u1, DTYPE_t u2, DTYPE_t RZ0, DTYPE_t RZ1, str Mode='LOS', DTYPE_t kOut=np.inf):
    cdef DTYPE_t    uN = math.sqrt(u0**2+u1**2+u2**2), uParN = math.sqrt(u0**2+u1**2), DParN = math.sqrt(D0**2+D1**2)
    cdef DTYPE_t    Sca = u0*D0+u1*D1+u2*D2, ScaP = u0*D0+u1*D1
    cdef DTYPE_t    kPMin
    if uParN == 0.:
        kPMin = (RZ1-D2)/u2
    else:
        kPMin = findRootkPMin_ImpactLine(uParN, uN, Sca, RZ0, RZ1, ScaP, DParN, kOut, D0, D1, D2, u0, u1, u2, Mode=Mode)
    cdef DTYPE_t    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef DTYPE_t    PMin2norm = math.sqrt(PMin0**2+PMin1**2)
    cdef DTYPE_t    PMin2D0 = PMin2norm, PMin2D1 = PMin2
    cdef DTYPE_t    RMin = math.sqrt((PMin2D0-RZ0)**2+(PMin2D1-RZ1)**2)
    cdef DTYPE_t    eTheta0 = -PMin1/PMin2norm, eTheta1 = PMin0/PMin2norm, eTheta2 = 0.
    cdef DTYPE_t    vP0 = PMin2D0-RZ0, vP1 = PMin2D1-RZ1
    cdef DTYPE_t    Theta = math.atan2(vP1,vP0)
    cdef DTYPE_t    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef DTYPE_t    er2D0 = math.cos(ImpTheta), er2D1 = math.sin(ImpTheta)
    cdef DTYPE_t    p = vP0*er2D0 + vP1*er2D1
    cdef DTYPE_t    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef DTYPE_t    phi = math.asin(-uN0*eTheta0 -uN1*eTheta1 -uN2*eTheta2)
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p, ImpTheta, phi

cdef Calc_Impact_Line_1D_Fast_Lin(DTYPE_t D0, DTYPE_t D1, DTYPE_t D2, DTYPE_t u0, DTYPE_t u1, DTYPE_t u2, DTYPE_t RZ0, DTYPE_t RZ1, str Mode='LOS', DTYPE_t kOut=np.inf):
    cdef DTYPE_t    kPMin = (RZ0-D1)*u1 + (RZ1-D2)*u2
    kPMin = kOut if Mode=='LOS' and kPMin > kOut else kPMin
    cdef DTYPE_t    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef DTYPE_t    RMin = math.sqrt((PMin1-RZ0)**2+(PMin2-RZ1)**2)
    cdef DTYPE_t    vP0 = PMin1-RZ0, vP1 = PMin2-RZ1
    cdef DTYPE_t    Theta = math.atan2(vP1,vP0)
    cdef DTYPE_t    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef DTYPE_t    er2D0 = math.cos(ImpTheta), er2D1 = math.sin(ImpTheta)
    cdef DTYPE_t    p = vP0*er2D0 + vP1*er2D1
    cdef DTYPE_t    uN = math.sqrt(u0**2+u1**2+u2**2)
    cdef DTYPE_t    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef DTYPE_t    eTheta0 = 0., eTheta1 = 0., eTheta2 = 1.
    cdef DTYPE_t    phi = math.asin(-uN0*eTheta0 -uN1*eTheta1 -uN2*eTheta2)
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p, ImpTheta, phi



def Calc_Impact_Line(DTYPE_t[::1] D, DTYPE_t[::1] u, DTYPE_t[::1] RZ, str Mode='LOS', kOut=np.inf):
    cdef tuple PMin0
    cdef DTYPE_t kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = Calc_Impact_Line_1D_Fast(D[0],D[1],D[2],u[0],u[1],u[2],RZ[0],RZ[1], Mode=Mode, kOut=kOut)
    return np.array(PMin0), kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0

def Calc_Impact_Line_Lin(DTYPE_t[::1] D, DTYPE_t[::1] u, DTYPE_t[::1] RZ, str Mode='LOS', kOut=np.inf):
    cdef tuple PMin0
    cdef DTYPE_t kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = Calc_Impact_Line_1D_Fast_Lin(D[0],D[1],D[2],u[0],u[1],u[2],RZ[0],RZ[1], Mode=Mode, kOut=kOut)
    return np.array(PMin0), kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0

def Calc_Impact_LineMulti(DTYPE_t[:,::1] D, DTYPE_t[:,::1] u, DTYPE_t[::1] RZ, DTYPE_t[::1] kOut, str Mode='LOS'):
    cdef Py_ssize_t ii
    cdef list PMin = [], kPMin = [], RMin = [], Theta = [], p = [], ImpTheta = [], phi = []
    cdef tuple PMin0
    cdef DTYPE_t kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0
    for ii in xrange(0,D.shape[1]):
        PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = Calc_Impact_Line_1D_Fast(D[0,ii],D[1,ii],D[2,ii],u[0,ii],u[1,ii],u[2,ii],RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
        PMin.append(PMin0), kPMin.append(kPMin0), RMin.append(RMin0), Theta.append(Theta0), p.append(p0), ImpTheta.append(ImpTheta0), phi.append(phi0)
    return np.asarray(PMin).T, np.asarray(kPMin), np.asarray(RMin), np.asarray(Theta), np.asarray(p), np.asarray(ImpTheta), np.asarray(phi)

def Calc_Impact_LineMulti_Lin(DTYPE_t[:,::1] D, DTYPE_t[:,::1] u, DTYPE_t[::1] RZ, DTYPE_t[::1] kOut, str Mode='LOS'):
    cdef Py_ssize_t ii
    cdef list PMin = [], kPMin = [], RMin = [], Theta = [], p = [], ImpTheta = [], phi = []
    cdef tuple PMin0
    cdef DTYPE_t kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0
    for ii in xrange(0,D.shape[1]):
        PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = Calc_Impact_Line_1D_Fast_Lin(D[0,ii],D[1,ii],D[2,ii],u[0,ii],u[1,ii],u[2,ii],RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
        PMin.append(PMin0), kPMin.append(kPMin0), RMin.append(RMin0), Theta.append(Theta0), p.append(p0), ImpTheta.append(ImpTheta0), phi.append(phi0)
    return np.asarray(PMin).T, np.asarray(kPMin), np.asarray(RMin), np.asarray(Theta), np.asarray(p), np.asarray(ImpTheta), np.asarray(phi)


def Calc_ImpactEnv(np.ndarray[DTYPE_t,ndim=1] RZ, np.ndarray[DTYPE_t,ndim=2] Poly, int NP=50, Test=True):
    """ Computes impact parameters of a Tor enveloppe (Tor is a closed 2D polygon)

    D. VEZINET, Aug. 2014
    Inputs :
        RZ          A (2,1) np.ndarray indicating the impact point
        Poly        A (2,N) np.ndarray (ideally with 1st point = last point, but optionnal) representing the 2D polygon to be used
        NP          An integer (default = 50) indicating the number of points used for discretising theta between 0 and pi
    Outputs :
        theta
    """
    if Test:
        assert RZ.size==2, 'Arg RZ should be a (2,) np.ndarray !'
        assert Poly.shape[0]==2, 'Arg Poly should be a (2,N) np.ndarray !'
    cdef int NPoly = Poly.shape[1]
    EnvTheta = np.linspace(0.,np.pi,NP,endpoint=True).reshape((NP,1))
    Vect = np.concatenate((np.cos(EnvTheta),np.sin(EnvTheta)),axis=1)
    Vectbis = np.swapaxes(np.resize(Vect,(NPoly,NP,2)),0,1)

    RZPoly = Poly - np.tile(RZ,(NPoly,1)).T
    RZPoly = np.resize(RZPoly.T,(NP,NPoly,2))
    Sca = np.sum(Vectbis*RZPoly,axis=2)
    return EnvTheta.flatten(), np.array([np.max(Sca,axis=1).T, np.min(Sca,axis=1).T])







#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
"""
                          Solid angle and vector for Detect+Apert and Detect+Lens
"""
#######################################################################################################
#######################################################################################################
#######################################################################################################




"""
###############################################################################
###############################################################################
                    Detect + Apert
###############################################################################
"""

cdef IsOnGoodSide_AllFast1P(DTYPE_t Pt0, DTYPE_t Pt1, DTYPE_t Pt2, DTYPE_t[:,::1] Barys, DTYPE_t[:,::1] ns, int NPolys):
    cdef Py_ssize_t ii
    for ii in range(0,NPolys):
        if (Pt0-Barys[0,ii])*ns[0,ii] + (Pt1-Barys[1,ii])*ns[1,ii] + (Pt2-Barys[2,ii])*ns[2,ii] <= 0.:
            return False
    return True




@cython.boundscheck(False)
@cython.wraparound(False)
cdef Calc_SAngVect_1Point1Poly(DTYPE_t P0, DTYPE_t P1, DTYPE_t P2, DTYPE_t[::1] Poly0, DTYPE_t[::1] Poly1, DTYPE_t[::1] Poly2, DTYPE_t G0, DTYPE_t G1, DTYPE_t G2):
    # Return solid angle subtended by Poly from Point, as well as normalised vector from Points to center of mass of Poly
    cdef Py_ssize_t NPoly = len(Poly0), ii = 0
    cdef DTYPE_t PG0 = G0-P0, PG1 = G1-P1, PG2 = G2-P2
    cdef DTYPE_t PGn = math.sqrt(PG0*PG0+PG1*PG1+PG2*PG2)
    cdef DTYPE_t PP10, PP11, PP12, PP20, PP21, PP22, PP1n, PP2n, ATan1, ATan2
    cdef DTYPE_t SAngs=0., V0=PG0/PGn, V1=PG1/PGn, V2=PG2/PGn 
    for ii in xrange(0,NPoly-1):
        PP10, PP11, PP12 = Poly0[ii]-P0, Poly1[ii]-P1, Poly2[ii]-P2
        PP20, PP21, PP22 = Poly0[ii+1]-P0, Poly1[ii+1]-P1, Poly2[ii+1]-P2
        PP1n = math.sqrt(PP10*PP10+PP11*PP11+PP12*PP12)
        PP2n = math.sqrt(PP20*PP20+PP21*PP21+PP22*PP22)
        ATan1 = Cabs(PG0*(PP11*PP22-PP12*PP21) + PG1*(PP12*PP20-PP10*PP22) + PG2*(PP10*PP21-PP11*PP20))
        ATan2 = PGn*PP1n*PP2n + (PG0*PP10+PG1*PP11+PG2*PP12)*PP2n + (PG0*PP20+PG1*PP21+PG2*PP22)*PP1n + (PP10*PP20+PP11*PP21+PP12*PP22)*PGn
        sa = math.atan2(ATan1, ATan2)
        if sa>=0:
            SAngs = SAngs + sa
        else:
            SAngs = SAngs + sa + np.pi
    return 2.*SAngs, V0,V1,V2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Calc_SAngVect_1Point1Poly_NoVect(DTYPE_t P0, DTYPE_t P1, DTYPE_t P2, DTYPE_t[::1] Poly0, DTYPE_t[::1] Poly1, DTYPE_t[::1] Poly2, DTYPE_t G0, DTYPE_t G1, DTYPE_t G2):
    """ Return solid angle subtended by (closed) Poly from Point """
    cdef Py_ssize_t NPoly = len(Poly0), ii = 0
    cdef DTYPE_t PG0 = G0-P0, PG1 = G1-P1, PG2 = G2-P2
    cdef DTYPE_t PGn = math.sqrt(PG0*PG0+PG1*PG1+PG2*PG2)
    cdef DTYPE_t PP10, PP11, PP12, PP20, PP21, PP22, PP1n, PP2n, ATan1, ATan2
    cdef DTYPE_t SAngs = 0. 
    for ii in xrange(0,NPoly-1):
        PP10, PP11, PP12 = Poly0[ii]-P0, Poly1[ii]-P1, Poly2[ii]-P2
        PP20, PP21, PP22 = Poly0[ii+1]-P0, Poly1[ii+1]-P1, Poly2[ii+1]-P2
        PP1n = math.sqrt(PP10*PP10+PP11*PP11+PP12*PP12)
        PP2n = math.sqrt(PP20*PP20+PP21*PP21+PP22*PP22)
        ATan1 = Cabs(PG0*(PP11*PP22-PP12*PP21) + PG1*(PP12*PP20-PP10*PP22) + PG2*(PP10*PP21-PP11*PP20))
        ATan2 = PGn*PP1n*PP2n + (PG0*PP10+PG1*PP11+PG2*PP12)*PP2n + (PG0*PP20+PG1*PP21+PG2*PP22)*PP1n + (PP10*PP20+PP11*PP21+PP12*PP22)*PGn
        sa = math.atan2(ATan1, ATan2)
        SAngs = SAngs + sa if sa>=0 else SAngs + sa + math.pi
    return 2.*SAngs



@cython.boundscheck(False)
@cython.wraparound(False)
cdef Calc_SAngVect_1Point1Poly_FromList(DTYPE_t P0, DTYPE_t P1, DTYPE_t P2, list Poly, DTYPE_t G0, DTYPE_t G1, DTYPE_t G2):
    # Return solid angle subtended by Poly from Point, as well as normalised vector from Points to center of mass of Poly
    cdef Py_ssize_t NPoly = len(Poly), ii = 0
    cdef DTYPE_t PG0 = G0-P0, PG1 = G1-P1, PG2 = G2-P2
    cdef DTYPE_t PGn = math.sqrt(PG0*PG0+PG1*PG1+PG2*PG2)
    cdef DTYPE_t PP10, PP11, PP12, PP20, PP21, PP22, PP1n, PP2n, ATan1, ATan2
    cdef DTYPE_t SAngs=0., V0=PG0/PGn, V1=PG1/PGn, V2=PG2/PGn 
    for ii in xrange(0,NPoly-1):
        PP10, PP11, PP12 = Poly[ii][0]-P0, Poly[ii][1]-P1, Poly[ii][2]-P2
        PP20, PP21, PP22 = Poly[ii+1][0]-P0, Poly[ii+1][1]-P1, Poly[ii+1][2]-P2
        PP1n = math.sqrt(PP10*PP10+PP11*PP11+PP12*PP12)
        PP2n = math.sqrt(PP20*PP20+PP21*PP21+PP22*PP22)
        ATan1 = Cabs(PG0*(PP11*PP22-PP12*PP21) + PG1*(PP12*PP20-PP10*PP22) + PG2*(PP10*PP21-PP11*PP20))
        ATan2 = PGn*PP1n*PP2n + (PG0*PP10+PG1*PP11+PG2*PP12)*PP2n + (PG0*PP20+PG1*PP21+PG2*PP22)*PP1n + (PP10*PP20+PP11*PP21+PP12*PP22)*PGn
        sa = math.atan2(ATan1, ATan2)
        if sa>=0:
            SAngs = SAngs + sa
        else:
            SAngs = SAngs + sa + np.pi
    return 2.*SAngs, V0,V1,V2

#@cython.wraparound(False)      # Faster without...
#@cython.boundscheck(False)
cdef Calc_SAngVect_LPolys1Point(list LPolys, DTYPE_t[::1] Point, DTYPE_t[::1] P, DTYPE_t[::1] nP, DTYPE_t[::1] e1P, DTYPE_t[::1] e2P):
    # Returns homothetic (with centers As) projections of Poly on plane (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and plane, but a unique plane and polygon list common to all of them
    #t1, t2, t3, t4, t5, t6 = 0.,0.,0.,0.,0.,0.    # DB
    #tt = dtm.datetime.now()                        # DB
    cdef DTYPE_t Pt0=Point[0],Pt1=Point[1],Pt2=Point[2], P0=P[0],P1=P[1],P2=P[2], nP0=nP[0],nP1=nP[1],nP2=nP[2], e10=e1P[0],e11=e1P[1],e12=e1P[2], e20=e2P[0],e21=e2P[1],e22=e2P[2]
    cdef Py_ssize_t NList = len(LPolys)
    cdef Py_ssize_t ii
    cdef DTYPE_t SAng = 0
    cdef DTYPE_t ScaAPn
    cdef np.ndarray[Py_ssize_t, ndim=1] ind
    cdef np.ndarray[DTYPE_t, ndim=1] ScaAMn, k
    cdef np.ndarray[DTYPE_t, ndim=2] PProj, PolyInt
    cdef tuple Barys
    cdef list Vect = [np.nan,np.nan,np.nan]
    cdef list AM, PRef, temp
    #t1 = (dtm.datetime.now()-tt).total_seconds()      # DB
    #tt = dtm.datetime.now()                        # DB
    for ii in range(0,NList):
        AM = [LPolys[ii][0,:]-Pt0, LPolys[ii][1,:]-Pt1, LPolys[ii][2,:]-Pt2]
        ScaAMn = AM[0]*nP0 + AM[1]*nP1 + AM[2]*nP2
        ScaAPn = (P0-Pt0)*nP0 + (P1-Pt1)*nP1 + (P2-Pt2)*nP2
        if np.any(ScaAMn*ScaAPn<0):
            return SAng, np.nan*np.ones((3,)), True
        k = np.zeros((LPolys[ii].shape[1],))
        ind =(np.abs(ScaAMn)-1.e-14 > 0).nonzero()[0]
        k[ind] = ScaAPn/ScaAMn[ind]
        PProj = np.array([(Pt0+k*AM[0]-P0)*e10+(Pt1+k*AM[1]-P1)*e11+(Pt2+k*AM[2]-P2)*e12, (Pt0+k*AM[0]-P0)*e20+(Pt1+k*AM[1]-P1)*e21+(Pt2+k*AM[2]-P2)*e22]).T
        if ii==0:
            PRef = [plg.Polygon(PProj)]
        else:
            PRef[0] = PRef[0] & plg.Polygon(PProj)
    #t2 = (dtm.datetime.now()-tt).total_seconds()        # DB
    if PRef[0].area()>1e-12:
        #tt = dtm.datetime.now()                        # DB
        PolyInt = np.array(PRef[0][0]+[PRef[0][0][0]])
        temp = [PolyInt[:,0],PolyInt[:,1]]
        Barys = PRef[0].center()
        SAng, Vect[0], Vect[1], Vect[2] = Calc_SAngVect_1Point1Poly(Pt0,Pt1,Pt2, P0+e10*temp[0]+e20*temp[1], P1+e11*temp[0]+e21*temp[1], P2+e12*temp[0]+e22*temp[1], P0+e10*Barys[0]+e20*Barys[1], P1+e11*Barys[0]+e21*Barys[1],P2+e12*Barys[0]+e22*Barys[1])
        #t3 = (dtm.datetime.now()-tt).total_seconds()        # DB
    #print t1, t2, t3                                      # DB
    return SAng, np.array(Vect), False


cdef Calc_SAngVect_LPolys1Point_NoVect(list LPolys, DTYPE_t[::1] Point, DTYPE_t[::1] P, DTYPE_t[::1] nP, DTYPE_t[::1] e1P, DTYPE_t[::1] e2P):
    # Returns homothetic (with centers As) projections of Poly on plane (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and plane, but a unique plane and polygon list common to all of them
    #t1, t2, t3, t4, t5, t6 = 0.,0.,0.,0.,0.,0.    # DB
    #tt = dtm.datetime.now()                        # DB
    cdef DTYPE_t Pt0=Point[0],Pt1=Point[1],Pt2=Point[2], P0=P[0],P1=P[1],P2=P[2], nP0=nP[0],nP1=nP[1],nP2=nP[2], e10=e1P[0],e11=e1P[1],e12=e1P[2], e20=e2P[0],e21=e2P[1],e22=e2P[2]
    cdef Py_ssize_t NList = len(LPolys)
    cdef Py_ssize_t ii
    cdef DTYPE_t SAng = 0
    cdef DTYPE_t ScaAPn
    cdef np.ndarray[Py_ssize_t, ndim=1] ind
    cdef np.ndarray[DTYPE_t, ndim=1] ScaAMn, k
    cdef np.ndarray[DTYPE_t, ndim=2] PProj, PolyInt
    cdef tuple Barys
    cdef list AM, PRef, temp
    #t1 = (dtm.datetime.now()-tt).total_seconds()      # DB
    #tt = dtm.datetime.now()                        # DB
    for ii in range(0,NList):
        AM = [LPolys[ii][0,:]-Pt0, LPolys[ii][1,:]-Pt1, LPolys[ii][2,:]-Pt2]
        ScaAMn = AM[0]*nP0 + AM[1]*nP1 + AM[2]*nP2
        ScaAPn = (P0-Pt0)*nP0 + (P1-Pt1)*nP1 + (P2-Pt2)*nP2
        if np.any(ScaAMn*ScaAPn<0):
            return SAng, np.nan*np.ones((3,)), True
        k = np.zeros((LPolys[ii].shape[1],))
        ind =(np.abs(ScaAMn)-1.e-14 > 0).nonzero()[0]
        k[ind] = ScaAPn/ScaAMn[ind]
        PProj = np.array([(Pt0+k*AM[0]-P0)*e10+(Pt1+k*AM[1]-P1)*e11+(Pt2+k*AM[2]-P2)*e12, (Pt0+k*AM[0]-P0)*e20+(Pt1+k*AM[1]-P1)*e21+(Pt2+k*AM[2]-P2)*e22]).T
        if ii==0:
            PRef = [plg.Polygon(PProj)]
        else:
            PRef[0] = PRef[0] & plg.Polygon(PProj)
    #t2 = (dtm.datetime.now()-tt).total_seconds()        # DB
    if PRef[0].area()>1e-12:
        #tt = dtm.datetime.now()                        # DB
        PolyInt = np.array(PRef[0][0]+[PRef[0][0][0]])
        temp = [PolyInt[:,0],PolyInt[:,1]]
        Barys = PRef[0].center()
        SAng = Calc_SAngVect_1Point1Poly_NoVect(Pt0,Pt1,Pt2, P0+e10*temp[0]+e20*temp[1], P1+e11*temp[0]+e21*temp[1], P2+e12*temp[0]+e22*temp[1], P0+e10*Barys[0]+e20*Barys[1], P1+e11*Barys[0]+e21*Barys[1],P2+e12*Barys[0]+e22*Barys[1])
        #t3 = (dtm.datetime.now()-tt).total_seconds()        # DB
    #print t1, t2, t3                                      # DB
    return SAng, False



def Calc_SAngVect_LPolys1Point_Flex(list LPolys, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt, np.ndarray[DTYPE_t, ndim=1,mode='c'] P, np.ndarray[DTYPE_t, ndim=1,mode='c'] nP, np.ndarray[DTYPE_t, ndim=1,mode='c'] e1, np.ndarray[DTYPE_t, ndim=1,mode='c'] e2, VectReturn=True):
    cdef DTYPE_t SAng = 0., norm
    cdef np.ndarray[DTYPE_t, ndim=1] Vect
    if VectReturn:
        SAng, Vect, ind = Calc_SAngVect_LPolys1Point(LPolys, Pt, P, nP, e1, e2)
        if ind:
            norm = math.sqrt((Pt[0]-P[0])**2 + (Pt[1]-P[1])**2 + (Pt[2]-P[2])**2)
            nP = (Pt-P)/norm
            e1P, e2P = Calc_DefaultCheck_e1e2_PLane_1D(P, nP)
            SAng, Vect, ind = Calc_SAngVect_LPolys1Point(LPolys, Pt, P, nP, e1, e2)
            assert not ind, "A point resists particular treatment !"
        return SAng, Vect
    else:
        SAng, ind = Calc_SAngVect_LPolys1Point_NoVect(LPolys, Pt, P, nP, e1, e2)
        if ind:
            norm = math.sqrt((Pt[0]-P[0])**2 + (Pt[1]-P[1])**2 + (Pt[2]-P[2])**2)
            nP = (Pt-P)/norm
            e1P, e2P = Calc_DefaultCheck_e1e2_PLane_1D(P, nP)
            SAng, ind = Calc_SAngVect_LPolys1Point_NoVect(LPolys, Pt, P, nP, e1, e2)
            assert not ind, "A point resists particular treatment !"
    return SAng



@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_SAngVect_LPolysPoints(list LPolys, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=1,mode='c'] P, np.ndarray[DTYPE_t, ndim=1,mode='c'] nP, np.ndarray[DTYPE_t, ndim=1,mode='c'] e1P, np.ndarray[DTYPE_t, ndim=1,mode='c'] e2P):
    # Returns homothetic (with centers As) projections of Poly on plane (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and plane, but a unique plane and polygon list common to all of them
    #t1, t2, t3, t4, t5, t6 = 0.,0.,0.,0.,0.,0.    # DB
    #tt = dtm.datetime.now()                        # DB
    cdef Py_ssize_t NP = Points.shape[1], NList = len(LPolys)
    cdef Py_ssize_t ii, jj, indOKii
    cdef DTYPE_t P0=P[0], P1=P[1], P2=P[2], e10=e1P[0], e11=e1P[1], e12=e1P[2], e20=e2P[0], e21=e2P[1], e22=e2P[2]
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c']  SAng = np.zeros((NP,))
    cdef np.ndarray[DTYPE_t, ndim=2]  Vect = np.nan*np.ones((3,NP))
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] NPperPoly = np.array([LPolys[ii].shape[1] for ii in range(0,NList)],dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=2]  Polys = np.concatenate(tuple(LPolys),axis=1)
    cdef Py_ssize_t                   NPoly = Polys.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3]  Asbis = np.tile(Points.reshape((3,1,NP)),(1,NPoly,1)).T
    cdef np.ndarray[DTYPE_t, ndim=3]  AM = np.tile(Polys.T.reshape(1,NPoly,3),(NP,1,1)) - Asbis
    cdef np.ndarray[DTYPE_t, ndim=3]  Pbis = np.tile(P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  nPbis = np.tile(nP.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  ScaAMn = np.sum(AM*nPbis,axis=2,keepdims=True)
    cdef np.ndarray[DTYPE_t, ndim=3]  ScaAPn = np.sum((Pbis-Asbis)*nPbis,axis=2,keepdims=True)
    cdef np.ndarray indneg = np.any(ScaAMn*ScaAPn<0,axis=1).flatten()
    cdef Py_ssize_t[::1]              IndOK = (~indneg).nonzero()[0]
    cdef np.ndarray[DTYPE_t, ndim=3]  k = np.zeros((NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  e1bis = np.tile(e1P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  e2bis = np.tile(e2P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=2]  PolyProjX1, PolyProjX2
    cdef DTYPE_t Bs0, Bs1
    cdef list Polyint, PX, PolyInt
    cdef list IndPoly = [(np.sum(NPperPoly[0:jj+1])-NPperPoly[jj], np.sum(NPperPoly[0:jj+1])-1) for jj in range(0,NList)]
    #t1 = (dtm.datetime.now()-tt).total_seconds()      # DB
    #tt = dtm.datetime.now()         # DB
    # Get the intersection of projected polygons for each point
    ind = np.abs(ScaAMn)-1.e-14 > 0.
    k[ind] = ScaAPn[ind]/ScaAMn[ind]
    k[indneg,:,:] = np.nan
    cdef np.ndarray[DTYPE_t, ndim=3] PDiff = Asbis + (np.tile(k.T,(3,1,1)).T)*AM - Pbis # = PolyProj - Pbis
    PolyProjX1 = np.sum(PDiff*e1bis,axis=2,keepdims=False)
    PolyProjX2 = np.sum(PDiff*e2bis,axis=2,keepdims=False)
    PX = [np.array([PolyProjX1[:,IndPoly[jj][0]:IndPoly[jj][1]+1],PolyProjX2[:,IndPoly[jj][0]:IndPoly[jj][1]+1]]).T for jj in range(0,NList)]
    #t2 = (dtm.datetime.now() - tt).total_seconds()    # DB
    # Compute solid angle for each point with non-zero intersection
    #tt3 = dtm.datetime.now()         # DB
    for ii in xrange(0,IndOK.size):
        #tt = dtm.datetime.now()         # DB
        indOKii = IndOK[ii]
        Polyint = [plg.Polygon(PX[0][:,indOKii,:])]
        for jj in range(1,NList):
            Polyint[0] = Polyint[0] & plg.Polygon(PX[jj][:,indOKii,:])
        #t4 += (dtm.datetime.now()-tt).total_seconds()   # DB
        if Polyint[0].area()>1e-12:
            #tt = dtm.datetime.now()    # DB
            Bs0, Bs1 = Polyint[0].center()
            PolyInt = [(P0+e10*pp[0]+e20*pp[1],P1+e11*pp[0]+e21*pp[1],P2+e12*pp[0]+e22*pp[1]) for pp in Polyint[0][0]+[Polyint[0][0][0]]]
            #t5 += (dtm.datetime.now()-tt).total_seconds()   # DB
            #tt = dtm.datetime.now()    # DB 
            SAng[indOKii], Vect[0,indOKii], Vect[1,indOKii], Vect[2,indOKii] = Calc_SAngVect_1Point1Poly_FromList(Points[0,indOKii],Points[1,indOKii],Points[2,indOKii], PolyInt, P0+e10*Bs0+e20*Bs1, P1+e11*Bs0+e21*Bs1, P2+e12*Bs0+e22*Bs1)
            #t6 += (dtm.datetime.now()-tt).total_seconds()   # DB
    #t3 = (dtm.datetime.now() - tt3).total_seconds()    # DB
    #print t1, t2, t3, t4, t5, t6 # DB
    return SAng, Vect, indneg


def Calc_SAngVect_LPolysPoints_Flex(list LPolys, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=1, mode='c'] P, np.ndarray[DTYPE_t, ndim=1, mode='c'] nP, np.ndarray[DTYPE_t, ndim=1, mode='c'] e1P, np.ndarray[DTYPE_t, ndim=1, mode='c'] e2P):
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] SAng
    cdef np.ndarray[DTYPE_t, ndim=2] Vect
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] Indneg
    cdef DTYPE_t[::1] Pt = np.zeros((3,))
    cdef DTYPE_t[::1] e1, e2
    cdef DTYPE_t nPnorm
    cdef Py_ssize_t ii
    SAng, Vect, indneg = Calc_SAngVect_LPolysPoints(LPolys, Points, P, nP, e1P, e2P)
    if np.any(indneg):
        print "            Nb. of points needing particular treatment : ", np.sum(indneg)
        Indneg = indneg.nonzero()[0]
        nPs = np.array([Points[0,indneg]-P[0], Points[1,indneg]-P[1], Points[2,indneg]-P[2]])
        nPs = nPs/np.sqrt(np.sum(nPs**2,axis=0))
        e1Ps, e2Ps = Calc_DefaultCheck_e1e2_PLanes_2D(P.reshape((3,1)).dot(np.ones((1,Indneg.size))), nPs, e1s=None, e2s=None)
        for ii in range(0,Indneg.size):
            Pt[0], Pt[1], Pt[2] = Points[0,Indneg[ii]], Points[1,Indneg[ii]], Points[2,Indneg[ii]]
            nPnorm = math.sqrt((Pt[0]-P[0])**2 + (Pt[1]-P[1])**2 + (Pt[2]-P[2])**2)
            nP = np.array([(Pt[0]-P[0])/nPnorm, (Pt[1]-P[1])/nPnorm, (Pt[2]-P[2])/nPnorm])
            e1, e2 = Calc_DefaultCheck_e1e2_PLane_1D(P, nP)
            SAng[Indneg[ii]], Vect[:,Indneg[ii]], indneg[Indneg[ii]] = Calc_SAngVect_LPolys1Point(LPolys, Pt, P, nP, e1, e2)
        assert np.all(~indneg), "Some points ("+str(np.sum(indneg))+") resist particular treatment : "+str(Points[:,indneg.nonzero()[0][0]])+" with P = "+str(P)
    return SAng, Vect
           

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_IsVis_LPolysPoints(list LPolys, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=1,mode='c'] P, np.ndarray[DTYPE_t, ndim=1,mode='c'] nP, np.ndarray[DTYPE_t, ndim=1,mode='c'] e1P, np.ndarray[DTYPE_t, ndim=1,mode='c'] e2P):
    # Returns homothetic (with centers As) projections of Poly on plane (P,nP), and optionnaly their components (X1,X2) along (e1P,e2P)
    # Arbitrary Points and plane, but a unique plane and polygon list common to all of them
    #t1, t2, t3, t4, t5 = 0.,0.,0.,0.,0.    # DB
    #tt = dtm.datetime.now()                        # DB
    cdef Py_ssize_t NP = Points.shape[1], NList = len(LPolys)
    cdef Py_ssize_t ii, jj, indOKii
    cdef np.ndarray[np.int_t, ndim=1, mode='c'] NPperPoly = np.array([LPolys[ii].shape[1] for ii in range(0,NList)],dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=2]  Polys = np.concatenate(tuple(LPolys),axis=1)
    cdef Py_ssize_t                   NPoly = Polys.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3]  Asbis = np.tile(Points.reshape((3,1,NP)),(1,NPoly,1)).T
    cdef np.ndarray[DTYPE_t, ndim=3]  AM = np.tile(Polys.T.reshape(1,NPoly,3),(NP,1,1)) - Asbis
    cdef np.ndarray[DTYPE_t, ndim=3]  Pbis = np.tile(P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  nPbis = np.tile(nP.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  ScaAMn = np.sum(AM*nPbis,axis=2,keepdims=True)
    cdef np.ndarray[DTYPE_t, ndim=3]  ScaAPn = np.sum((Pbis-Asbis)*nPbis,axis=2,keepdims=True)
    cdef np.ndarray                   indneg = np.any(ScaAMn*ScaAPn<0,axis=1).flatten()
    cdef Py_ssize_t[::1]              IndOK = (~indneg).nonzero()[0]
    cdef np.int_t[::1]                Ind = np.zeros((NP,),dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=3]  k = np.zeros((NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  e1bis = np.tile(e1P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=3]  e2bis = np.tile(e2P.reshape(1,1,3),(NP,NPoly,1))
    cdef np.ndarray[DTYPE_t, ndim=2]  PolyProjX1, PolyProjX2
    cdef list Polyint, PX
    cdef list IndPoly = [(np.sum(NPperPoly[0:jj+1])-NPperPoly[jj], np.sum(NPperPoly[0:jj+1])-1) for jj in range(0,NList)]
    #t1 = (dtm.datetime.now()-tt).total_seconds()      # DB
    #tt = dtm.datetime.now()         # DB
    # Get the intersection of projected polygons for each point
    ind = np.abs(ScaAMn)-1.e-14 > 0.
    k[ind] = ScaAPn[ind]/ScaAMn[ind]
    k[indneg,:,:] = np.nan
    cdef np.ndarray[DTYPE_t, ndim=3] PDiff = Asbis + (np.tile(k.T,(3,1,1)).T)*AM - Pbis # = PolyProj - Pbis
    PolyProjX1 = np.sum(PDiff*e1bis,axis=2,keepdims=False)
    PolyProjX2 = np.sum(PDiff*e2bis,axis=2,keepdims=False)
    PX = [np.array([PolyProjX1[:,IndPoly[jj][0]:IndPoly[jj][1]+1],PolyProjX2[:,IndPoly[jj][0]:IndPoly[jj][1]+1]]).T for jj in range(0,NList)]
    #t2 = (dtm.datetime.now() - tt).total_seconds()    # DB
    # Compute solid angle for each point with non-zero intersection
    #tt3 = dtm.datetime.now()         # DB
    for ii in xrange(0,IndOK.size):
        #tt = dtm.datetime.now()         # DB
        indOKii = IndOK[ii]
        Polyint = [plg.Polygon(PX[0][:,indOKii,:])]
        for jj in range(1,NList):
            Polyint[0] = Polyint[0] & plg.Polygon(PX[jj][:,indOKii,:])
        #t4 += (dtm.datetime.now()-tt).total_seconds()   # DB
        if Polyint[0].area()>1e-12:
            Ind[indOKii] = 1
    #t3 = (dtm.datetime.now() - tt3).total_seconds()    # DB
    #print t1, t2, t3, t4    # DB
    return np.asarray(Ind), indneg


def Calc_IsVis_LPolysPoints_Flex(list LPolys, np.ndarray[DTYPE_t, ndim=2] Points, np.ndarray[DTYPE_t, ndim=1] P, np.ndarray[DTYPE_t, ndim=1] nP, np.ndarray[DTYPE_t, ndim=1] e1P, np.ndarray[DTYPE_t, ndim=1] e2P):
    """ Return a bool np.ndarray with True if a point is visible through all input Polygons """
    cdef np.ndarray[DTYPE_t, ndim=2] nPs, e1Ps, e2Ps
    cdef np.ndarray[np.int_t, ndim=1] Indneg
    cdef np.ndarray[np.int_t, ndim=1] Ind
    cdef Py_ssize_t ii
    Ind, indneg = Calc_IsVis_LPolysPoints(LPolys, Points, P, nP, e1P, e2P)
    if np.any(indneg):
        Indneg = indneg.nonzero()[0]
        nPs = np.array([Points[0,indneg]-P[0], Points[1,indneg]-P[1], Points[2,indneg]-P[2]])
        nPs = nPs/np.sqrt(np.sum(nPs**2,axis=0))
        e1Ps, e2Ps = Calc_DefaultCheck_e1e2_PLanes_2D(P.reshape((3,1)).dot(np.ones((1,Indneg.size))), nPs, e1s=None, e2s=None)
        for ii in range(0,Indneg.size):
            Ind[Indneg[ii]] = Calc_IsVis_LPolysPoints(LPolys, Points[:,Indneg[ii]:Indneg[ii]+1], P, nPs[:,ii], e1Ps[:,ii], e2Ps[:,ii])
        assert not np.any(np.isnan(Ind)), "Some points resist particular treatment !"
    return Ind.astype(bool)



def FuncSAquad_Apert(list LPolys, nPtemp, e1bis, e2bis, Ps, e1, e2, LBaryS, LnIn, nPolys, PBary, VPoly, VVin, list DLong=[], str VType='Tor', Colis=True):
    Ps0, Ps1, Ps2 = Ps
    e10, e11, e12 = e1
    e20, e21, e22 = e2
    if Colis:
        PBary0, PBary1, PBary2 = PBary
        def FuncSA(double x2, double x1):
            cdef DTYPE_t Pt0 = Ps0+x1*e10+x2*e20, Pt1 = Ps1+x1*e11+x2*e21, Pt2 = Ps2+x1*e12+x2*e22
            if IsOnGoodSide_AllFast1P(Pt0,Pt1,Pt2, LBaryS,LnIn,nPolys) and Calc_InOut_LOS_Colis_1D(PBary0,PBary1,PBary2, Pt0,Pt1,Pt2, VPoly, VVin, DLong=DLong, VType=VType):
                return Calc_SAngVect_LPolys1Point_Flex(LPolys, np.array([Pt0,Pt1,Pt2]), PBary, nPtemp, e1bis, e2bis, VectReturn=False )
            else:
                return 0.
    else:
        def FuncSA(double x2, double x1):
            cdef DTYPE_t Pt0 = Ps0+x1*e10+x2*e20, Pt1 = Ps1+x1*e11+x2*e21, Pt2 = Ps2+x1*e12+x2*e22
            if IsOnGoodSide_AllFast1P(Pt0,Pt1,Pt2, LBaryS,LnIn,nPolys):
                return Calc_SAngVect_LPolys1Point_Flex(LPolys, np.array([Pt0,Pt1,Pt2]), PBary, nPtemp, e1bis, e2bis, VectReturn=False )
            else:
                return 0.
    return FuncSA






"""
###############################################################################
###############################################################################
                    Detect + Lens
###############################################################################
"""


def LensDetect_isInside(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, 
        np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt0, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt1, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt2, DTYPE_t tanthetmax):
    """ Return an array of bool indicating whether the provided points lie inside the viewing cone of the Detect+Lens system """
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] din = (Pt0-O0)*nIn0 + (Pt1-O1)*nIn1 + (Pt2-O2)*nIn2
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] dinTip = (Pt0-Tip0)*nIn0 + (Pt1-Tip1)*nIn1 + (Pt2-Tip2)*nIn2
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] r = np.sqrt((Pt0-Tip0-dinTip*nIn0)**2 + (Pt1-Tip1-dinTip*nIn1)**2 + (Pt2-Tip2-dinTip*nIn2)**2)
    return (r/dinTip <= tanthetmax) & (din>0.)

cdef _LensDetect_isInside_1Pt(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, DTYPE_t Pt0, DTYPE_t Pt1, DTYPE_t Pt2, DTYPE_t tanthetmax):
    """ Return a bool indicating whether the provided point lie inside the viewing cone of the Detect+Lens system, faster for numerical integration using quad """
    cdef DTYPE_t din = (Pt0-O0)*nIn0 + (Pt1-O1)*nIn1 + (Pt2-O2)*nIn2
    cdef DTYPE_t dinTip = (Pt0-Tip0)*nIn0 + (Pt1-Tip1)*nIn1 + (Pt2-Tip2)*nIn2
    cdef DTYPE_t r = math.sqrt((Pt0-Tip0-dinTip*nIn0)**2 + (Pt1-Tip1-dinTip*nIn1)**2 + (Pt2-Tip2-dinTip*nIn2)**2)
    return r/dinTip <= tanthetmax and din>0.


# Used directly in TFG.Lens() and in GG.Calc_SAngVect_LPolysPoints_Lens()
def _Lens_get_CircleInFocPlaneFromPts(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, DTYPE_t Rad, DTYPE_t F1, 
        np.ndarray[DTYPE_t, ndim=1, mode='c'] Pts0, np.ndarray[DTYPE_t, ndim=1, mode='c'] Pts1, np.ndarray[DTYPE_t, ndim=1, mode='c'] Pts2, DTYPE_t F2=np.inf):
    """
    Return the center and radius of the image of the spherical Lens projected in its focal plane as seen from arbitrary points, only usable with F2=inf so far
    
    Inputs:
    -------
        O       iterable    Contains the 3D cartesian coordinates of the center of the Lens
        nIn     iterable    Contains the 3D cartesian coordinates of the normalized vector indicating the axis of the Lens, directing towards the inside of the vessel
        Rad     float       Radius of the Lens
        F1      float       Focal distance (positive) on the side of the detector (i.e.: opposite side of the plasma and points of interest)
        Pts     iterable    Contains the 3D cartesian coordinates of all the N points for which the computation shall be done
        F2      float       inf

    Outputs:
    --------
        Cs      np.ndarray  (3,N) array of the 3D cartesian coordiantes of the centers of the circular images of the spherical Lens in its focal plane as seen from the N Pts
        RadIm   np.ndarray  (N,) array of the radius of each circular image of the Lens seen from the N points
        din     np.ndarray  (N,) array of the distance from the Lens center O of the N points along the axis (O,nIn)
        r       np.ndarray  (N,) array of the (absolute, positive) distance to the axis (O,nIn) of the N Pts
        rIm     np.ndarray  (N,) array of the (absolute, positive) distance to the axis (O,nIn) of the N centers Cs of the Lens images
        nperp   np.ndarray  (3,N) array of the 3D cartesian coordinates of the normalized vector perpendicular to the axis (O,nIn) and passing throught the N Pts
    """
    assert F2==np.inf, "Only coded for F2==inf !"

    # Computing the algorithmic distance between the Lens center O and each point along the Lens axis directed by nIn
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] din = (Pts0-O0)*nIn0 + (Pts1-O1)*nIn1 + (Pts2-O2)*nIn2
    # Deducing the normalised vector perpendicular to the Lens axis, and the distance r to the axis
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] nperp0 = Pts0-O0-din*nIn0, nperp1 = Pts1-O1-din*nIn1, nperp2 = Pts2-O2-din*nIn2
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] r = np.sqrt(nperp0**2+nperp1**2+nperp2**2)
    indokr, indokd = r>0., din>0.
    nperp0[indokr], nperp1[indokr], nperp2[indokr] = nperp0[indokr]/r[indokr], nperp1[indokr]/r[indokr], nperp2[indokr]/r[indokr]

    # Computing the distance rIm between the Lens axis and the center of the Lens image in the focal plane from Pts
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] rIm = r * F1
    rIm[indokd] = rIm[indokd] / din[indokd]
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] Cs0 = O0-F1*nIn0-nperp0*rIm, Cs1 = O1-F1*nIn1-nperp1*rIm, Cs2 = O2-F1*nIn2-nperp2*rIm

    # Computing the width the Lens image in the focal plane
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] OC = np.sqrt((O0-Cs0)**2+(O1-Cs1)**2+(O2-Cs2)**2)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] OA = np.sqrt((O0-Pts0)**2+(O1-Pts1)**2+(O2-Pts2)**2)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] RadIm = Rad * OC/OA

    return Cs0,Cs1,Cs2, RadIm, din, r, rIm, nperp0,nperp1,nperp2


cdef _Lens_get_CircleInFocPlaneFrom1Pt(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, DTYPE_t Rad, DTYPE_t F1, DTYPE_t Pt0, DTYPE_t Pt1, DTYPE_t Pt2):
    """
    Identical to _Lens_get_CircleInFocPlaneFromPts, but for 1 Point only, fast version used for numerical integration using quad 
    """
    # Computing the algorithmic distance between the Lens center O and each point along the Lens axis directed by nIn
    cdef DTYPE_t din = (Pt0-O0)*nIn0 + (Pt1-O1)*nIn1 + (Pt2-O2)*nIn2
    # Deducing the normalised vector perpendicular to the Lens axis, and the distance r to the axis
    cdef DTYPE_t nperp0 = Pt0-O0-din*nIn0, nperp1 = Pt1-O1-din*nIn1, nperp2 = Pt2-O2-din*nIn2
    cdef DTYPE_t r = math.sqrt(nperp0**2+nperp1**2+nperp2**2)
    
    cdef DTYPE_t Cs0 = np.nan, Cs1 = np.nan, Cs2 = np.nan
    cdef DTYPE_t RadIm = np.nan, rIm = np.nan
    cdef DTYPE_t OC = np.nan, OA = np.nan
    if din > 0:
        rIm = r * F1 / din
        if not r==0.:
            nperp0, nperp1, nperp2 = nperp0/r, nperp1/r, nperp2/r
        Cs0, Cs1, Cs2 = O0-F1*nIn0-nperp0*rIm, O1-F1*nIn1-nperp1*rIm, O2-F1*nIn2-nperp2*rIm
        # Compute the width the Lens image in the focal plane
        OC = math.sqrt((O0-Cs0)**2+(O1-Cs1)**2+(O2-Cs2)**2)
        OA = math.sqrt((O0-Pt0)**2+(O1-Pt1)**2+(O2-Pt2)**2)
        RadIm = Rad * OC/OA

    return Cs0,Cs1,Cs2 , RadIm, din, r, rIm, nperp0,nperp1,nperp2


cdef Calc_SAngVect_LPolys1Point_Lens(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, DTYPE_t Pt0, DTYPE_t Pt1, DTYPE_t Pt2, DTYPE_t RadL, DTYPE_t RadD, DTYPE_t F1, DTYPE_t tanthetmax, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL0, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL1, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL2, DTYPE_t[::1] thet=np.linspace(0.,np.pi,100)):
    """
    Return the solid angle and vector associated to one point in the plasma and corresponding to the intersection between the image of a spherical lens and a circular detector placed in its focal plane with the same axis
    """

    cdef DTYPE_t SAng = 0.
    cdef DTYPE_t Vect0 = np.nan, Vect1 = np.nan, Vect2 = np.nan
    cdef DTYPE_t C0, C1, C2, RadIm, d, r, rIm, nperp0, nperp1, nperp2
    cdef DTYPE_t Ref0, Ref1, Ref2
    cdef DTYPE_t Rbis, e20, e21, e22
    cdef DTYPE_t[::1] polyL0, polyL1, polyL2, poly0, poly1
    cdef DTYPE_t BaryS0, BaryS1
    cdef DTYPE_t norm
    cdef DTYPE_t AreaLim = math.pi*(min(RadL,RadD)/10000.)**2

    if _LensDetect_isInside_1Pt(O0,O1,O2, Tip0,Tip1,Tip2, nIn0,nIn1,nIn2, Pt0,Pt1,Pt2, tanthetmax):     # Check point has non-zero Solid Angle (is inside the viewing cone)
        C0,C1,C2, RadIm, d, r, rIm, nperp0,nperp1,nperp2 = _Lens_get_CircleInFocPlaneFrom1Pt(O0,O1,O2, nIn0,nIn1,nIn2, RadL, F1, Pt0,Pt1,Pt2)
        
        if rIm < RadIm+RadD: # Make sure the intersection exists

            if rIm+RadIm <= RadD: # Case with image of Lens included in Detector
                Ref0, Ref1, Ref2 = O0, O1, O2
                SAng = Calc_SAngVect_1Point1Poly_NoVect(Pt0,Pt1,Pt2, PolyL0,PolyL1,PolyL2, Ref0,Ref1,Ref2)

            elif RadIm > rIm+RadD: # Case with Detector included in image of Lens
                Ref0, Ref1, Ref2 = O0+nperp0*r, O1+nperp1*r, O2+nperp2*r # -nIn ?
                Rbis = RadD * d / F1
                e20, e21, e22 = nIn1*nperp2-nIn2*nperp1, nIn2*nperp0-nIn0*nperp2, nIn0*nperp1-nIn1*nperp0
                polyL0 = Ref0 + Rbis*np.cos(thet)*nperp0 + Rbis*np.sin(thet)*e20 
                polyL1 = Ref1 + Rbis*np.cos(thet)*nperp1 + Rbis*np.sin(thet)*e21
                polyL2 = Ref2 + Rbis*np.cos(thet)*nperp2 + Rbis*np.sin(thet)*e22
                SAng = Calc_SAngVect_1Point1Poly_NoVect(Pt0,Pt1,Pt2, polyL0,polyL1,polyL2, Ref0,Ref1,Ref2)
            
            else:
                Rbis = RadD * d / F1
                e20, e21, e22 = nIn1*nperp2-nIn2*nperp1, nIn2*nperp0-nIn0*nperp2, nIn0*nperp1-nIn1*nperp0
                polyL0 = (PolyL0-O0)*nperp0 + (PolyL1-O1)*nperp1 + (PolyL2-O2)*nperp2
                polyL1 = (PolyL0-O0)*e20 + (PolyL1-O1)*e21 + (PolyL2-O2)*e22
                poly0 = r + Rbis*np.cos(thet)
                poly1 = Rbis*np.sin(thet)
                
                PolyInt = plg.Polygon(np.array([polyL0,polyL1]).T) & plg.Polygon(np.array([poly0,poly1]).T)
                if PolyInt.area() > AreaLim:
                    BaryS0, BaryS1 = PolyInt.center()
                    Ref0, Ref1, Ref2 = O0+BaryS0*nperp0+BaryS1*e20, O1+BaryS0*nperp1+BaryS1*e21, O2+BaryS0*nperp2+BaryS1*e22
                    PolyInt0, PolyInt1 = np.array(PolyInt[0]).T
                    PolyLL0 = O0 + PolyInt0*nperp0+PolyInt1*e20
                    PolyLL1 = O1 + PolyInt0*nperp1+PolyInt1*e21
                    PolyLL2 = O2 + PolyInt0*nperp2+PolyInt1*e22
                    SAng = Calc_SAngVect_1Point1Poly_NoVect(Pt0,Pt1,Pt2, PolyLL0,PolyLL1,PolyLL2, Ref0,Ref1,Ref2)

            Vect0, Vect1, Vect2 = Ref0-Pt0, Ref1-Pt1, Ref2-Pt2
            norm = math.sqrt(Vect0**2+Vect1**2+Vect2**2)
            Vect0, Vect1, Vect2 = Vect0/norm, Vect1/norm, Vect2/norm

    return SAng, [Vect0,Vect1,Vect2]



cdef Calc_SAngVect_LPolysPoints_Lens(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt0, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt1, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt2, DTYPE_t RadL, DTYPE_t RadD, DTYPE_t F1, DTYPE_t tanthetmax, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL0, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL1, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL2, DTYPE_t[::1] thet=np.linspace(0.,np.pi,100)):
    """
    Return the solid angle and vector associated to one point in the plasma and corresponding to the intersection between the image of a spherical lens and a circular detector placed in its focal plane with the same axis
    """
    cdef int NP = int(Pt0.size), NThet = int(thet.size)
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] SAng = np.zeros((NP,))
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] Vect0 = np.nan*np.ones((NP,)), Vect1 = np.nan*np.ones((NP,)), Vect2 = np.nan*np.ones((NP,))
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] C0, C1, C2, RadIm, d, r, rIm, nperp0, nperp1, nperp2
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] nIn = np.array([nIn0, nIn1, nIn2])
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] e1, e2, norm
    cdef DTYPE_t Ref0, Ref1, Ref2
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] Rbis
    cdef DTYPE_t[::1] polyL0, polyL1, polyL2, poly0, poly1
    cdef DTYPE_t BaryS0, BaryS1
    cdef DTYPE_t AreaLim = math.pi*(min(RadL,RadD)/10000.)**2

    ind = LensDetect_isInside(O0,O1,O2, Tip0,Tip1,Tip2, nIn0,nIn1,nIn2, Pt0,Pt1,Pt2, tanthetmax)
    if np.any(ind):     # Check point has non-zero Solid Angle (is inside the viewing cone)
        Ind = ind.nonzero()[0]

        C0,C1,C2, RadIm, d, r, rIm, nperp0,nperp1,nperp2 = _Lens_get_CircleInFocPlaneFromPts(O0,O1,O2, nIn0,nIn1,nIn2, RadL, F1, Pt0[ind],Pt1[ind],Pt2[ind])
        e1 = np.array([nperp0[0],nperp1[0],nperp2[0]])
        e1 = e1/np.linalg.norm(e1)
        e2 = np.array([nIn1*nperp2[0]-nIn2*nperp1[0], nIn2*nperp0[0]-nIn0*nperp2[0], nIn0*nperp1[0]-nIn1*nperp0[0]])
        e2 = e2/np.linalg.norm(e2)

        indrIm = rIm < RadIm+RadD
        if np.any(indrIm): # Make sure the intersection exists
            
            indrIm1 = rIm+RadIm <= RadD          # Case with image of Lens included in Detector
            indrIm2 = RadIm >= rIm+RadD          # Case with Detector included in image of Lens
            indrIm3 = ~(indrIm1 | indrIm2)       # The rest

            if np.any(indrIm1):
                RefP = np.array([O0, O1, O2])
                SAng[Ind[indrIm1]] = Calc_SAngVect_LPolysPoints([np.array([PolyL0,PolyL1,PolyL2])], np.array([Pt0[ind][indrIm1],Pt1[ind][indrIm1],Pt2[ind][indrIm1]]), RefP, nIn, e1, e2)[0]
                Vect0[Ind[indrIm1]], Vect1[Ind[indrIm1]], Vect2[Ind[indrIm1]] = RefP[0]-Pt0[ind][indrIm1], RefP[1]-Pt1[ind][indrIm1], RefP[2]-Pt2[ind][indrIm1]
            
            if np.any(indrIm2):
                RefP =  np.array([O0+nperp0[indrIm2]*r[indrIm2], O1+nperp1[indrIm2]*r[indrIm2], O2+nperp2[indrIm2]*r[indrIm2]]) # -nIn ?
                Rbis = RadD * d[indrIm2] / F1
                IndrIm2 = indrIm2.nonzero()[0]
                for ii in range(0,indrIm2.sum()):
                    e20, e21, e22 = nIn1*nperp2[IndrIm2[ii]]-nIn2*nperp1[IndrIm2[ii]], nIn2*nperp0[IndrIm2[ii]]-nIn0*nperp2[IndrIm2[ii]], nIn0*nperp1[IndrIm2[ii]]-nIn1*nperp0[IndrIm2[ii]]
                    polyL0 = RefP[0,ii] + Rbis[ii]*np.cos(thet)*nperp0[IndrIm2[ii]] + Rbis*np.sin(thet)*e20
                    polyL1 = RefP[1,ii] + Rbis[ii]*np.cos(thet)*nperp1[IndrIm2[ii]] + Rbis*np.sin(thet)*e21
                    polyL2 = RefP[2,ii] + Rbis[ii]*np.cos(thet)*nperp2[IndrIm2[ii]] + Rbis*np.sin(thet)*e22
                    SAng[Ind[IndrIm2[ii]]] = Calc_SAngVect_1Point1Poly_NoVect(Pt0[ind][IndrIm2[ii]],Pt1[ind][IndrIm2[ii]],Pt2[ind][IndrIm2[ii]], polyL0,polyL1,polyL2, RefP[0,ii],RefP[1,ii],RefP[2,ii])
                    Vect0[Ind[IndrIm2[ii]]], Vect1[Ind[IndrIm2[ii]]], Vect2[Ind[IndrIm2[ii]]] = RefP[0,ii]-Pt0[ind][IndrIm2[ii]], RefP[1,ii]-Pt1[ind][IndrIm2[ii]], RefP[2,ii]-Pt2[ind][IndrIm2[ii]]

            if np.any(indrIm3):
                Rbis = RadD * d[indrIm3] / F1
                IndrIm3 = indrIm3.nonzero()[0]
                for ii in range(0,indrIm3.sum()):
                    e20, e21, e22 = nIn1*nperp2[IndrIm3[ii]]-nIn2*nperp1[IndrIm3[ii]], nIn2*nperp0[IndrIm3[ii]]-nIn0*nperp2[IndrIm3[ii]], nIn0*nperp1[IndrIm3[ii]]-nIn1*nperp0[IndrIm3[ii]]
                    polyL0 = (PolyL0-O0)*nperp0[IndrIm3[ii]] + (PolyL1-O1)*nperp1[IndrIm3[ii]] + (PolyL2-O2)*nperp2[IndrIm3[ii]]
                    polyL1 = (PolyL0-O0)*e20 + (PolyL1-O1)*e21 + (PolyL2-O2)*e22
                    poly0 = r[IndrIm3[ii]] + Rbis[ii]*np.cos(thet)
                    poly1 = Rbis[ii]*np.sin(thet)
                    PolyInt = plg.Polygon(np.array([polyL0,polyL1]).T) & plg.Polygon(np.array([poly0,poly1]).T)
                    if PolyInt.area() > AreaLim:
                        BaryS0, BaryS1 = PolyInt.center()
                        Ref0, Ref1, Ref2 = O0+BaryS0*nperp0[IndrIm3[ii]]+BaryS1*e20, O1+BaryS0*nperp1[IndrIm3[ii]]+BaryS1*e21, O2+BaryS0*nperp2[IndrIm3[ii]]+BaryS1*e22
                        PolyInt0, PolyInt1 = np.array(PolyInt[0]).T
                        PolyLL0 = O0 + PolyInt0*nperp0[IndrIm3[ii]]+PolyInt1*e20
                        PolyLL1 = O1 + PolyInt0*nperp1[IndrIm3[ii]]+PolyInt1*e21
                        PolyLL2 = O2 + PolyInt0*nperp2[IndrIm3[ii]]+PolyInt1*e22
                        SAng[Ind[IndrIm3[ii]]] = Calc_SAngVect_1Point1Poly_NoVect(Pt0[ind][IndrIm3[ii]],Pt1[ind][IndrIm3[ii]],Pt2[ind][IndrIm3[ii]], PolyLL0,PolyLL1,PolyLL2, Ref0,Ref1,Ref2)
                        Vect0[Ind[IndrIm3[ii]]], Vect1[Ind[IndrIm3[ii]]], Vect2[Ind[IndrIm3[ii]]] = Ref0-Pt0[ind][IndrIm3[ii]], Ref1-Pt1[ind][IndrIm3[ii]], Ref2-Pt2[ind][IndrIm3[ii]]

            norm = np.sqrt(Vect0**2+Vect1**2+Vect2**2)
            Vect0, Vect1, Vect2 = Vect0/norm, Vect1/norm, Vect2/norm

    return SAng, np.array([Vect0,Vect1,Vect2])



def Calc_SAngVect_LPolysPoints_Flex_Lens(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt0, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt1, np.ndarray[DTYPE_t, ndim=1,mode='c'] Pt2, DTYPE_t RadL, DTYPE_t RadD, DTYPE_t F1, DTYPE_t tanthetmax, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL0, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL1, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL2, DTYPE_t[::1] thet=np.linspace(0.,2.*np.pi,100), VectReturn=True):
    cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] SAng = np.zeros((Pt0.size,))
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Vect
    if VectReturn:
        SAng, Vect = Calc_SAngVect_LPolysPoints_Lens(O0, O1, O2, Tip0, Tip1, Tip2, nIn0, nIn1, nIn2, Pt0, Pt1, Pt2, RadL, RadD, F1, tanthetmax, PolyL0, PolyL1, PolyL2, thet=thet)
        return SAng, Vect
    else:
        SAng = Calc_SAngVect_LPolysPoints_Lens(O0, O1, O2, Tip0, Tip1, Tip2, nIn0, nIn1, nIn2, Pt0, Pt1, Pt2, RadL, RadD, F1, tanthetmax, PolyL0, PolyL1, PolyL2, thet=thet)[0]
        return SAng



def Calc_SAngVect_LPolys1Point_Flex_Lens(DTYPE_t O0, DTYPE_t O1, DTYPE_t O2, DTYPE_t Tip0, DTYPE_t Tip1, DTYPE_t Tip2, DTYPE_t nIn0, DTYPE_t nIn1, DTYPE_t nIn2, DTYPE_t Pt0, DTYPE_t Pt1, DTYPE_t Pt2, DTYPE_t RadL, DTYPE_t RadD, DTYPE_t F1, DTYPE_t tanthetmax, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL0, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL1, np.ndarray[DTYPE_t, ndim=1,mode='c'] PolyL2, DTYPE_t[::1] thet=np.linspace(0.,2.*np.pi,100), VectReturn=True):
    cdef DTYPE_t SAng = 0.
    cdef list Vect
    if VectReturn:
        SAng, Vect = Calc_SAngVect_LPolys1Point_Lens(O0, O1, O2, Tip0, Tip1, Tip2, nIn0, nIn1, nIn2, Pt0, Pt1, Pt2, RadL, RadD, F1, tanthetmax, PolyL0, PolyL1, PolyL2, thet=thet)
        return SAng, Vect
    else:
        SAng = Calc_SAngVect_LPolys1Point_Lens(O0, O1, O2, Tip0, Tip1, Tip2, nIn0, nIn1, nIn2, Pt0, Pt1, Pt2, RadL, RadD, F1, tanthetmax, PolyL0, PolyL1, PolyL2, thet=thet)[0]
        return SAng



def FuncSAquad_Lens(O, Tip, Ps, e1, e2, LBaryS, LnIn, int nPolys, DTYPE_t RadL, DTYPE_t RadD, DTYPE_t F1, DTYPE_t tanthetmax, PolyL, VPoly, VVin, list DLong=[], str VType='Tor', thet=np.linspace(0.,2.*np.pi,100), VectReturn=False, Colis=True):
    """ Return the adequate function for computing the solid angle of a Detect+Lens system, with or without colisions """
    O0, O1, O2 = O
    Tip0, Tip1, Tip2 = Tip
    Ps0, Ps1, Ps2 = Ps
    e10, e11, e12 = e1
    e20, e21, e22 = e2
    nIn0,nIn1,nIn2 = LnIn[:,1]
    PolyL0,PolyL1,PolyL2 = PolyL

    if Colis:
        def FuncSA(double x2, double x1):
            cdef DTYPE_t Pt0 = Ps0+x1*e10+x2*e20, Pt1 = Ps1+x1*e11+x2*e21, Pt2 = Ps2+x1*e12+x2*e22
            if IsOnGoodSide_AllFast1P(Pt0,Pt1,Pt2, LBaryS,LnIn,nPolys) and Calc_InOut_LOS_Colis_1D(O0,O1,O2, Pt0,Pt1,Pt2, VPoly, VVin, DLong=DLong, VType=VType):
                return Calc_SAngVect_LPolys1Point_Flex_Lens(O0,O1,O2, Tip0,Tip1,Tip2, nIn0,nIn1,nIn2, Pt0,Pt1,Pt2, RadL, RadD, F1, tanthetmax, PolyL0,PolyL1,PolyL2, thet=thet, VectReturn=False)
            else:
                return 0.
    else:
        def FuncSA(double x2, double x1):
            cdef DTYPE_t Pt0 = Ps0+x1*e10+x2*e20, Pt1 = Ps1+x1*e11+x2*e21, Pt2 = Ps2+x1*e12+x2*e22
            if IsOnGoodSide_AllFast1P(Pt0,Pt1,Pt2, LBaryS,LnIn,nPolys):
                return Calc_SAngVect_LPolys1Point_Flex_Lens(O0,O1,O2, Tip0,Tip1,Tip2, nIn0,nIn1,nIn2, Pt0,Pt1,Pt2, RadL, RadD, F1, tanthetmax, PolyL0,PolyL1,PolyL2, thet=thet, VectReturn=False)
            else:
                return 0.
    return FuncSA




