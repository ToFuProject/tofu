
# cimport
cimport cython
cimport numpy as cnp
from cpython cimport bool
from libc.math cimport sqrt as Csqrt, ceil as Cceil, abs as Cabs
from libc.math cimport floor as Cfloor, round as Cround, log2 as Clog2
from libc.math cimport cos as Ccos, acos as Cacos, sin as Csin, asin as Casin
from libc.math cimport atan2 as Catan2, pi as Cpi

# import
import sys
import numpy as np
import scipy.integrate as scpintg
from matplotlib.path import Path

if sys.version[0]=='3':
    from inspect import signature as insp
elif sys.version[0]=='2':
    from inspect import getargspec as insp



__all__ = ['CoordShift',
           'Poly_isClockwise', 'Poly_Order', 'Poly_VolAngTor',
           'Sino_ImpactEnv', 'ConvertImpact_Theta2Xi',
           '_Ves_isInside',
           '_Ves_mesh_dlfromL_cython', 
           '_Ves_meshCross_FromD', '_Ves_meshCross_FromInd', '_Ves_Smesh_Cross',
           '_Ves_Vmesh_Tor_SubFromD_cython', '_Ves_Vmesh_Tor_SubFromInd_cython',
           '_Ves_Vmesh_Lin_SubFromD_cython', '_Ves_Vmesh_Lin_SubFromInd_cython',
           '_Ves_Smesh_Tor_SubFromD_cython', '_Ves_Smesh_Tor_SubFromInd_cython',
           '_Ves_Smesh_TorStruct_SubFromD_cython', '_Ves_Smesh_TorStruct_SubFromInd_cython',
           '_Ves_Smesh_Lin_SubFromD_cython', '_Ves_Smesh_Lin_SubFromInd_cython',
           'LOS_Calc_PInOut_VesStruct',
           'check_ff', 'LOS_get_sample', 'LOS_calc_signal',
           'LOS_sino']






########################################################
########################################################
#       Coordinates handling
########################################################

def CoordShift(Pts, In='(X,Y,Z)', Out='(R,Z)', CrossRef=None):
    """ Check the shape of an array of points coordinates and/or converts from 2D to 3D, 3D to 2D, cylindrical to cartesian... (CrossRef is an angle (Tor) or a distance (X for Lin))"""
    assert all([type(ff) is str and ',' in ff for ff in [In,Out]]), "Arg In and Out (coordinate format) must be comma-separated  !"
    assert type(Pts) is np.ndarray and Pts.ndim in [1,2] and Pts.shape[0] in (2,3), "Points must be a 1D or 2D np.ndarray of 2 or 3 coordinates !"
    assert CrossRef is None or type(CrossRef) in [int,float,np.int64,np.float64], "Arg CrossRef must be a float !"
    
    # Pre-format inputs
    In, Out = In.lower(), Out.lower()

    # Get order
    Ins = In.replace('(','').replace(')','').split(',')
    Outs = Out.replace('(','').replace(')','').split(',')
    assert all([ss in ['x','y','z','r','phi'] for ss in Ins]), "Non-valid In !"
    assert all([ss in ['x','y','z','r','phi'] for ss in Outs]), "Non-valid Out !"
    InT = 'cyl' if any([ss in Ins for ss in ['r','phi']]) else 'cart'
    OutT = 'cyl' if any([ss in Outs for ss in ['r','phi']]) else 'cart'

    ndim = Pts.ndim
    if ndim==1:
        Pts = np.copy(Pts.reshape((Pts.shape[0],1)))

    # Compute
    if InT==OutT:
        assert all([ss in Ins for ss in Outs])
        pts = []
        for ii in Outs:
            if ii=='phi':
                pts.append(np.arctan2(np.sin(Pts[Ins.index(ii),:]),np.cos(Pts[Ins.index(ii),:])))
            else:
                pts.append(Pts[Ins.index(ii),:])
    elif InT=='cart':
        pts = []
        for ii in Outs:
            if ii=='r':
                assert all([ss in Ins for ss in ['x','y']])
                pts.append(np.hypot(Pts[Ins.index('x'),:],Pts[Ins.index('y'),:]))
            elif ii=='z':
                assert 'z' in Ins                
                pts.append(Pts[Ins.index('z'),:])
            elif ii=='phi':
                if all([ss in Ins for ss in ['x','y']]):
                    pts.append(np.arctan2(Pts[Ins.index('y'),:],Pts[Ins.index('x'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no phi value available !")
    else:
        pts = []
        for ii in Outs:
            if ii=='x':
                if all([ss in Ins for ss in ['r','phi']]):
                    pts.append(Pts[Ins.index('r'),:]*np.cos(Pts[Ins.index('phi'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no x value available !")
            elif ii=='y':
                assert all([ss in Ins for ss in ['r','phi']])
                pts.append(Pts[Ins.index('r'),:]*np.sin(Pts[Ins.index('phi'),:]))
            elif ii=='z':
                assert 'z' in Ins
                pts.append(Pts[Ins.index('z'),:])

    # Format output
    pts = np.vstack(pts)
    if ndim==1:
        pts = pts.flatten()
    return pts







"""
########################################################
########################################################
########################################################
#                  General Geometry
########################################################
########################################################
########################################################
"""

########################################################
########################################################
#       Polygons
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def Poly_isClockwise(cnp.ndarray[double,ndim=2] Poly):
    """ Assuming 2D closed Poly ! """
    cdef int ii, NP=Poly.shape[1]
    cdef double Sum=0.
    for ii in range(0,NP-1):
        Sum += Poly[0,ii]*Poly[1,ii+1]-Poly[0,ii+1]*Poly[1,ii]
    return Sum < 0.


def Poly_Order(cnp.ndarray[double,ndim=2] Poly, str order='C', Clock=False, close=True, str layout='(cc,N)', str layout_in=None, Test=True):
    """
    Return a polygon Poly as a np.ndarray formatted according to parameters

    Parameters 
    ----------
        Poly    np.ndarray or list or tuple     Input polygon under from of (cc,N) or (N,cc) np.ndarray (where cc = 2 or 3, the number of coordinates and N points), or list or tuple of vertices
        order   str                             Flag indicating whether the output np.ndarray shall be C-contiguous ('C') or Fortran-contiguous ('F')
        Clock   bool                            For 2-dimensional arrays only, flag indicating whether the output array shall represent a clockwise polygon (True) or anti-clockwise (False), or should be left unchanged (None)
        close   bool                            For 2-dimensional arrays only, flag indicating whether the output array shall be closed (True, i.e.: last point==first point), or not closed (False)
        layout  str                             Flag indicating whether the output np.ndarray shall be of shape '(cc,N)' or '(N,cc)'
        Test    bool                            Flag indicating whether the inputs should be tested for conformity, default: True

    Returns
    -------
        poly    np.ndarray                      Output formatted polygon

    """
    if Test:
        assert (2 in np.shape(Poly) or 3 in np.shape(Poly)) and max(np.shape(Poly))>=3, "Arg Poly must contain the 2D or 3D coordinates of at least 3 points (polygon) !"
        assert order.lower() in ['c','f'], "Arg order must be in ['c','f'] !"
        assert type(Clock) is bool, "Arg Clock must be a bool !"
        assert type(close) is bool, "Arg close must be a bool !"
        assert layout.lower() in ['(cc,n)','(n,cc)'], "Arg layout must be in ['(cc,n)','(n,cc)'] !"
        assert layout_in is None or layout_in.lower() in ['(cc,n)','(n,cc)'], "Arg layout_in must be in ['(cc,n)','(n,cc)'] !"
    
    if np.shape(Poly)==(3,3):
        assert not layout_in is None, "Could not resolve the input layout of Poly because shape==(3,3), specify !"
        poly = np.array(Poly).T if layout_in.lower()=='(n,cc)' else np.array(Poly)
    else:
        poly = np.array(Poly).T if min(np.shape(Poly))==Poly.shape[1] else np.array(Poly)
    if not np.allclose(poly[:,0],poly[:,-1], atol=1.e-9):
        poly = np.concatenate((poly,poly[:,0:1]),axis=1)
    if poly.shape[0]==2 and not Clock is None:
        if not Clock==Poly_isClockwise(poly):
            poly = poly[:,::-1]
    if not close:
        poly = poly[:,:-1]
    if layout.lower()=='(n,cc)':
        poly = poly.T 
    poly = np.ascontiguousarray(poly) if order.lower()=='c' else np.asfortranarray(poly)
    return poly




def Poly_VolAngTor(cnp.ndarray[double,ndim=2,mode='c'] Poly):
    cdef cnp.ndarray[double,ndim=1] Ri0=Poly[0,:-1], Ri1=Poly[0,1:], Zi0=Poly[1,:-1], Zi1=Poly[1,1:]
    cdef double V = np.sum((Ri0*Zi1 - Zi0*Ri1)*(Ri0+Ri1))/6.
    cdef double BV0 = np.sum((Ri0*Zi1 - Zi0*Ri1) * 0.5 * (Ri1**2 + Ri1*Ri0 + Ri0**2)) / (6.*V)
    cdef double BV1 = -np.sum((Ri1**2*Zi0*(2.*Zi1+Zi0) + 2.*Ri0*Ri1*(Zi0**2-Zi1**2) - Ri0**2*Zi1*(Zi1+2.*Zi0))/4.)/(6.*V)
    return V, np.array([BV0,BV1])






"""
###############################################################################
###############################################################################
                    Sinogram specific
###############################################################################
"""


def Sino_ImpactEnv(cnp.ndarray[double,ndim=1] RZ, cnp.ndarray[double,ndim=2] Poly, int NP=50, Test=True):
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



"""
########################################################
########################################################
########################################################
#                       Ves-specific
########################################################
########################################################
########################################################
"""




########################################################
########################################################
#       isInside
########################################################

def _Ves_isInside(Pts, VPoly, Lim=None, VType='Tor', In='(X,Y,Z)', Test=True):
    if Test:
        assert type(Pts) is np.ndarray and Pts.ndim in [1,2], "Arg Pts must be a 1D or 2D np.ndarray !"
        assert type(VPoly) is np.ndarray and VPoly.ndim==2 and VPoly.shape[0]==2, "Arg VPoly must be a (2,N) np.ndarray !"
        assert Lim is None or (hasattr(Lim,'__iter__') and len(Lim)==2) or (hasattr(Lim,'__iter__') and all([hasattr(ll,'__iter__') and len(ll)==2 for ll in Lim])), "Arg Lim must be a len()==2 iterable or a list of such !"
        assert type(VType) is str and VType.lower() in ['tor','lin'], "Arg VType must be a str in ['Tor','Lin'] !"

    path = Path(VPoly.T)
    if VType.lower()=='tor':
        if Lim is None:
            pts = CoordShift(Pts, In=In, Out='(R,Z)')
            ind = Path(VPoly.T).contains_points(pts.T, transform=None, radius=0.0)
        else:
            pts = CoordShift(Pts, In=In, Out='(R,Z,Phi)')
            ind0 = Path(VPoly.T).contains_points(pts[:2,:].T, transform=None, radius=0.0)
            if hasattr(Lim[0],'__iter__'):
                ind = np.zeros((len(Lim),Pts.shape[1]),dtype=bool)
                for ii in range(0,len(Lim)):
                    lim = [Catan2(Csin(Lim[ii][0]),Ccos(Lim[ii][0])), Catan2(Csin(Lim[ii][1]),Ccos(Lim[ii][1]))]
                    if lim[0]<lim[1]:
                        ind[ii,:] = ind0 & (pts[2,:]>=lim[0]) & (pts[2,:]<=lim[1])
                    else:
                        ind[ii,:] = ind0 & ((pts[2,:]>=lim[0]) | (pts[2,:]<=lim[1]))
            else:
                Lim = [Catan2(Csin(Lim[0]),Ccos(Lim[0])), Catan2(Csin(Lim[1]),Ccos(Lim[1]))]
                if Lim[0]<Lim[1]:
                    ind = ind0 & (pts[2,:]>=Lim[0]) & (pts[2,:]<=Lim[1])
                else:
                    ind = ind0 & ((pts[2,:]>=Lim[0]) | (pts[2,:]<=Lim[1]))
    else:
        pts = CoordShift(Pts, In=In, Out='(X,Y,Z)')
        ind0 = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
        if hasattr(Lim[0],'__iter__'):
            ind = np.zeros((len(Lim),Pts.shape[1]),dtype=bool)
            for ii in range(0,len(Lim)):
                ind[ii,:] = ind0 & (pts[0,:]>=Lim[ii][0]) & (pts[0,:]<=Lim[ii][1])
        else:
            ind = ind0 & (pts[0,:]>=Lim[0]) & (pts[0,:]<=Lim[1])
    return ind





########################################################
########################################################
#       Meshing - Common - Linear
########################################################


# Preliminary function to get optimal resolution from input resolution
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_mesh_dlfromL_cython(double[::1] LMinMax, double dL, DL=None, Lim=True, dLMode='abs', double margin=1.e-9):
    """ Get the actual reolution from the desired resolution and MinMax and limits """
    # Get the number of mesh elements in LMinMax
    cdef double N
    if dLMode.lower()=='abs':
        N = Cceil((LMinMax[1]-LMinMax[0])/dL)
    else:
        N = Cceil(1./dL)
    # Derive the real (effective) resolution
    cdef double dLr = (LMinMax[1]-LMinMax[0])/N
    # Get desired limits if any
    cdef double[::1] DLc, L
    cdef long [::1] indL
    #cdef cnp.ndarray[double,ndim=1] indL, L
    cdef double abs0, abs1, A
    cdef int nL0, nL1, Nind, ii, jj
    cdef list dl
    if DL is None:
        DLc = LMinMax
    else:
        dl = list(DL)
        if dl[0] is None:
            dl[0] = LMinMax[0]
        if dl[1] is None:
            dl[1] = LMinMax[1]
        if Lim and dl[0]<=LMinMax[0]:
            dl[0] = LMinMax[0]
        if Lim and dl[1]>=LMinMax[1]:
            dl[1] = LMinMax[1]
        DLc = np.array(dl)       
 
    # Get the extreme indices of the mesh elements that really need to be created within those limits
    abs0 = Cabs(DLc[0]-LMinMax[0])
    if abs0-dLr*Cfloor(abs0/dLr)<margin*dLr:
        nL0 = int(Cround((DLc[0]-LMinMax[0])/dLr))
    else:
        nL0 = int(Cfloor((DLc[0]-LMinMax[0])/dLr))
    abs1 = Cabs(DLc[1]-LMinMax[0])
    if abs1-dLr*Cfloor(abs1/dLr)<margin*dLr:
        nL1 = int(Cround((DLc[1]-LMinMax[0])/dLr)-1)
    else:
        nL1 = int(Cfloor((DLc[1]-LMinMax[0])/dLr))
    # Get the corresponding indices
    Nind = nL1+1-nL0
    indL = np.empty((Nind,),dtype=int)#np.linspace(nL0,nL1,Nind,endpoint=True)
    L = np.empty((Nind,))
    for ii in range(0,Nind):
        jj = nL0+ii
        indL[ii] = jj
        L[ii] = LMinMax[0] + (0.5 + (<double>jj))*dLr
    return np.asarray(L), dLr, np.asarray(indL), <long>N


########################################################
########################################################
#       Meshing - Common - Polygon face
########################################################

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_meshCross_FromD(double[::1] MinMax1, double[::1] MinMax2, double d1, double d2, D1=None, D2=None,str dSMode='abs', VPoly=None, double margin=1.e-9):
    cdef double[::1] X1, X2
    cdef double dX1, dX2
    cdef long[::1] ind1, ind2
    cdef int N1, N2, n1, n2, ii, jj, nn
    cdef cnp.ndarray[double,ndim=2] Pts
    cdef cnp.ndarray[double,ndim=1] dS
    cdef cnp.ndarray[long,ndim=1] ind

    X1, d1r, ind1, N1 = _Ves_mesh_dlfromL_cython(MinMax1, d1, D1, Lim=True, dLMode=dSMode, margin=margin)
    X2, d2r, ind2, N2 = _Ves_mesh_dlfromL_cython(MinMax2, d2, D2, Lim=True, dLMode=dSMode, margin=margin)
    n1, n2 = len(X1), len(X2)

    Pts = np.empty((2,n1*n2))
    dS = d1r*d2r*np.ones((n1*n2,))
    ind = np.empty((n1*n2,),dtype=int)
    for ii in range(0,n2):
        for jj in range(0,n1):
            nn = jj+n1*ii
            Pts[0,nn] = X1[jj]
            Pts[1,nn] = X2[ii]
            ind[nn] = ind1[jj] + n1*ind2[ii]
    if VPoly is not None:
        iin = Path(VPoly.T).contains_points(Pts.T, transform=None, radius=0.0)
        if np.sum(iin)==1:
            Pts, dS, ind = Pts[:,iin].reshape((2,1)), dS[iin], ind[iin]
        else:
            Pts, dS, ind = Pts[:,iin], dS[iin], ind[iin]
    return Pts, dS, ind, d1r, d2r    


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_meshCross_FromInd(double[::1] MinMax1, double[::1] MinMax2, double d1, double d2, long[::1] ind, str dSMode='abs', double margin=1.e-9):
    cdef double[::1] X1, X2
    cdef double dX1, dX2
    cdef long[::1] bla
    cdef int N1, N2, NP=ind.size, ii, i1, i2
    cdef cnp.ndarray[double,ndim=2] Pts
    cdef cnp.ndarray[double,ndim=1] dS

    X1, d1r, bla, N1 = _Ves_mesh_dlfromL_cython(MinMax1, d1, None, Lim=True, dLMode=dSMode, margin=margin)
    X2, d2r, bla, N2 = _Ves_mesh_dlfromL_cython(MinMax2, d2, None, Lim=True, dLMode=dSMode, margin=margin)

    Pts = np.empty((2,NP))
    dS = d1r*d2r*np.ones((NP,))
    for ii in range(0,NP):
        i2 = ind[ii] // N1
        i1 = ind[ii]-i2*N1
        Pts[0,ii] = X1[i1]
        Pts[1,ii] = X2[i2]
    return Pts, dS, d1r, d2r





@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Cross(double[:,::1] VPoly, double dL, str dLMode='abs', D1=None, D2=None, double margin=1.e-9, double DIn=0., VIn=None):
    cdef int ii, jj, nn=0, NP=VPoly.shape[1]
    cdef double[::1] LMinMax, L
    cdef double v0, v1, dlr
    cdef long[::1] indL
    cdef cnp.ndarray[long,ndim=1] N, ind
    cdef cnp.ndarray[double,ndim=1] dLr, Rref
    cdef cnp.ndarray[double,ndim=2] PtsCross
    cdef list LPtsCross=[], LdLr=[], Lind=[], LRref=[], VPolybis=[]
    
    LMinMax = np.array([0.,1.],dtype=float)
    N = np.empty((NP-1,),dtype=int)
    if DIn==0.:
        for ii in range(0,NP-1):
            v0, v1 = VPoly[0,ii+1]-VPoly[0,ii], VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0**2 + v1**2)
            L, dlr, indL, N[ii] = _Ves_mesh_dlfromL_cython(LMinMax, dL, dLMode=dLMode, DL=None, Lim=True, margin=margin)
            VPolybis.append((VPoly[0,ii],VPoly[1,ii]))
            v0, v1 = v0/LMinMax[1], v1/LMinMax[1]
            for jj in range(0,N[ii]):
                LdLr.append(dlr)
                LRref.append(VPoly[0,ii] + L[jj]*v0)
                LPtsCross.append((VPoly[0,ii] + L[jj]*v0, VPoly[1,ii] + L[jj]*v1))
                Lind.append(nn)
                nn += 1
                VPolybis.append((VPoly[0,ii] + jj*dlr*v0, VPoly[1,ii] + jj*dlr*v1))
        VPolybis.append((VPoly[0,0],VPoly[1,0]))
    else:
        for ii in range(0,NP-1):
            v0, v1 = VPoly[0,ii+1]-VPoly[0,ii], VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0**2 + v1**2)
            L, dlr, indL, N[ii] = _Ves_mesh_dlfromL_cython(LMinMax, dL, dLMode=dLMode, DL=None, Lim=True, margin=margin)
            VPolybis.append((VPoly[0,ii],VPoly[1,ii]))
            v0, v1 = v0/LMinMax[1], v1/LMinMax[1]
            for jj in range(0,N[ii]):
                LdLr.append(dlr)
                LRref.append(VPoly[0,ii] + L[jj]*v0)
                LPtsCross.append((VPoly[0,ii] + L[jj]*v0 + DIn*VIn[0,ii], VPoly[1,ii] + L[jj]*v1 + DIn*VIn[1,ii]))
                Lind.append(nn)
                nn += 1
                VPolybis.append((VPoly[0,ii] + jj*dlr*v0, VPoly[1,ii] + jj*dlr*v1))
        VPolybis.append((VPoly[0,0],VPoly[1,0]))
    
    PtsCross, dLr, ind, Rref = np.array(LPtsCross).T, np.array(LdLr), np.array(Lind,dtype=int), np.array(LRref)
    if D1 is not None:
        indin = (PtsCross[0,:]>=D1[0]) & (PtsCross[0,:]<=D1[1])
        PtsCross = PtsCross[:,indin]
        dLr, ind = dLr[indin], ind[indin]
    if D2 is not None:
        indin = (PtsCross[1,:]>=D2[0]) & (PtsCross[1,:]<=D2[1])
        PtsCross = PtsCross[:,indin]
        dLr, ind = dLr[indin], ind[indin]   
    
    return PtsCross, dLr, ind, N, Rref, np.array(VPolybis).T
























########################################################
########################################################
#       Meshing - Volume - Tor
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromD_cython(double dR, double dZ, double dRPhi, 
                                   double[::1] RMinMax, double[::1] ZMinMax,
                                   DR=None, DZ=None, DPhi=None, VPoly=None,
                                   str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    
    cdef double[::1] R0, R, Z, dRPhir, dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj, nPhi0, nPhi1, zz, NP, NRPhi_int, Rratio
    cdef cnp.ndarray[double,ndim=2] Pts, indI
    cdef cnp.ndarray[double,ndim=1] iii, dV, ind
    
    # Get the actual R and Z resolutions and mesh elements
    R0, dRr0, indR0, NR0 = _Ves_mesh_dlfromL_cython(RMinMax, dR, None, Lim=True, margin=margin)
    R, dRr, indR, NR = _Ves_mesh_dlfromL_cython(RMinMax, dR, DR, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, DZ, Lim=True, margin=margin)
    Rn = len(R)
    Zn = len(Z)
    
    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = -Cpi, Cpi
    else:
        DPhi0, DPhi1 = Catan2(Csin(DPhi[0]),Ccos(DPhi[0])), Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    
    dRPhir, dPhir = np.empty((Rn,)), np.empty((Rn,))
    Phin = np.empty((Rn,),dtype=int)
    NRPhi = np.empty((Rn,))
    NRPhi0 = np.zeros((Rn,),dtype=int)
    nRPhi0, indR0ii = 0, 0
    NP, NPhimax = 0, 0
    Rratio = int(Cceil(R[Rn-1]/R[0]))
    for ii in range(0,Rn):
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R !)
        NRPhi[ii] = Cceil(2.*Cpi*R[ii]/dRPhi)
        NRPhi_int = int(NRPhi[ii])
        dPhir[ii] = 2.*Cpi/NRPhi[ii]
        dRPhir[ii] = dPhir[ii]*R[ii]
        # Get index and cumulated indices from background
        for jj in range(indR0ii,NR0):
            if R0[jj]==R[ii]:
                indR0ii = jj
                break
            else:
                nRPhi0 += <long>Cceil(2.*Cpi*R0[jj]/dRPhi)
                NRPhi0[ii] = nRPhi0*NZ
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to be created within those limits
        abs0 = Cabs(DPhi0+Cpi)
        if abs0-dPhir[ii]*Cfloor(abs0/dPhir[ii])<margin*dPhir[ii]:
            nPhi0 = int(Cround((DPhi0+Cpi)/dPhir[ii]))
        else:
            nPhi0 = int(Cfloor((DPhi0+Cpi)/dPhir[ii]))
        abs1 = Cabs(DPhi1+Cpi)
        if abs1-dPhir[ii]*Cfloor(abs1/dPhir[ii])<margin*dPhir[ii]:
            nPhi1 = int(Cround((DPhi1+Cpi)/dPhir[ii])-1)
        else:
            nPhi1 = int(Cfloor((DPhi1+Cpi)/dPhir[ii]))
            
        if DPhi0<DPhi1:
            #indI.append(list(range(nPhi0,nPhi1+1)))
            Phin[ii] = nPhi1+1-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*Rratio+1))
            for jj in range(0,Phin[ii]):
                indI[ii,jj] = <double>( nPhi0+jj )
        else:
            #indI.append(list(range(nPhi0,NRPhi_int)+list(range(0,nPhi1+1))))
            Phin[ii] = nPhi1+1+NRPhi_int-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*Rratio+1))
            for jj in range(0,NRPhi_int-nPhi0):
                indI[ii,jj] = <double>( nPhi0+jj )
            for jj in range(NRPhi_int-nPhi0,Phin[ii]):
                indI[ii,jj] = <double>( jj- (NRPhi_int-nPhi0) )
        NP += Zn*Phin[ii]   
    
    Pts = np.empty((3,NP))
    ind = np.empty((NP,))
    dV = np.empty((NP,))
    # Compute Pts, dV and ind
    # This triple loop is the longest part, it takes ~90% of the CPU time
    NP = 0
    if Out.lower()=='(x,y,z)':
        for ii in range(0,Rn):
            iii = np.sort(indI[ii,~np.isnan(indI[ii,:])]) # To make sure the indices are in increasing order
            for zz in range(0,Zn):
                for jj in range(0,Phin[ii]):
                    indiijj = iii[jj]
                    phi = -Cpi + (0.5+indiijj)*dPhir[ii]
                    Pts[0,NP] = R[ii]*Ccos(phi)
                    Pts[1,NP] = R[ii]*Csin(phi)
                    Pts[2,NP] = Z[zz]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = dRr*dZr*dRPhir[ii]
                    NP += 1
    else:
        for ii in range(0,Rn):
            iii = np.sort(indI[ii,~np.isnan(indI[ii,:])])
            #assert iii.size==Phin[ii] and np.all(np.unique(iii)==iii)
            for zz in range(0,Zn):
                for jj in range(0,Phin[ii]):
                    indiijj = iii[jj] #indI[ii,iii[jj]]
                    Pts[0,NP] = R[ii]
                    Pts[1,NP] = Z[zz]
                    Pts[2,NP] = -Cpi + (0.5+indiijj)*dPhir[ii]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = dRr*dZr*dRPhir[ii]
                    NP += 1

    if VPoly is not None:
        if Out.lower()=='(x,y,z)':
            R = np.hypot(Pts[0,:],Pts[1,:])
            indin = Path(VPoly.T).contains_points(np.array([R,Pts[2,:]]).T, transform=None, radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(R)
        else:
            indin = Path(VPoly.T).contains_points(Pts[:-1,:].T, transform=None, radius=0.0)        
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(Pts[0,:])
        if not np.all(Ru==R):
            dRPhir = np.array([dRPhir[ii] for ii in range(0,len(R)) if R[ii] in Ru])
    return Pts, dV, ind.astype(int), dRr, dZr, np.asarray(dRPhir)




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromInd_cython(double dR, double dZ, double dRPhi, 
                                     double[::1] RMinMax, double[::1] ZMinMax, long[::1] ind, 
                                     str Out='(X,Y,Z)', double margin=1.e-9):
    """ Return the desired submesh indicated by the (numerical) indices, for the desired resolution (dR,dZ,dRphi) """
    cdef double[::1] R, Z, dRPhirRef, dPhir, Ru, dRPhir
    cdef double dRr, dZr, phi
    cdef long[::1] indR, indZ, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiR, iiZ, iiphi
    cdef double[:,::1] Phi
    cdef cnp.ndarray[double,ndim=2] Pts=np.empty((3,NP))
    cdef cnp.ndarray[double,ndim=1] dV=np.empty((NP,))
    
    # Get the actual R and Z resolutions and mesh elements
    R, dRr, indR, NR = _Ves_mesh_dlfromL_cython(RMinMax, dR, None, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, None, Lim=True, margin=margin)
    Rn, Zn = len(R), len(Z)
    
    # Number of Phi per R
    dRPhirRef, dPhir = np.empty((NR,)), np.empty((NR,))
    Ru, dRPhir = np.zeros((NR,)), np.nan*np.ones((NR,))
    NRPhi, NRPhi0 = np.empty((NR,),dtype=int), np.empty((NR+1,),dtype=int)
    Rratio = int(Cceil(R[NR-1]/R[0]))
    for ii in range(0,NR):
        NRPhi[ii] = <long>(Cceil(2.*Cpi*R[ii]/dRPhi))
        dRPhirRef[ii] = 2.*Cpi*R[ii]/<double>(NRPhi[ii])
        dPhir[ii] = 2.*Cpi/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((NR,NRPhi[ii]*Rratio+1))
        else:
            NRPhi0[ii] = NRPhi0[ii-1] + NRPhi[ii-1]*NZ
        for jj in range(0,NRPhi[ii]):
            Phi[ii,jj] = -Cpi + (0.5+<double>jj)*dPhir[ii]
            
    if Out.lower()=='(x,y,z)':
        for ii in range(0,NP):
            for jj in range(0,NR+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiR = jj-1
            iiZ = (ind[ii] - NRPhi0[iiR])//NRPhi[iiR]
            iiphi = ind[ii] - NRPhi0[iiR] - iiZ*NRPhi[iiR]
            phi = Phi[iiR,iiphi]
            Pts[0,ii] = R[iiR]*Ccos(phi)
            Pts[1,ii] = R[iiR]*Csin(phi)
            Pts[2,ii] = Z[iiZ]
            dV[ii] = dRr*dZr*dRPhirRef[iiR]
            if Ru[iiR]==0.:
                dRPhir[iiR] = dRPhirRef[iiR]
                Ru[iiR] = 1.
    else: 
        for ii in range(0,NP):
            for jj in range(0,NR+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiR = jj-1
            iiZ = (ind[ii] - NRPhi0[iiR])//NRPhi[iiR]
            iiphi = ind[ii] - NRPhi0[iiR] - iiZ*NRPhi[iiR]
            Pts[0,ii] = R[iiR]
            Pts[1,ii] = Z[iiZ]
            Pts[2,ii] = Phi[iiR,iiphi]
            dV[ii] = dRr*dZr*dRPhirRef[iiR]
            if Ru[iiR]==0.:
                dRPhir[iiR] = dRPhirRef[iiR]
                Ru[iiR] = 1.
    
    return Pts, dV, dRr, dZr, np.asarray(dRPhir)[~np.isnan(dRPhir)]



########################################################
########################################################
#       Meshing - Volume - Lin
########################################################


def _Ves_Vmesh_Lin_SubFromD_cython(double dX, double dY, double dZ, 
                                   double[::1] XMinMax, double[::1] YMinMax, double[::1] ZMinMax,
                                   DX=None, DY=None, DZ=None, VPoly=None,
                                   double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dY,dZ) "
    
    cdef double[::1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef cnp.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef cnp.ndarray[double,ndim=2] Pts
    cdef cnp.ndarray[long,ndim=1] ind
    
    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, indX, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DX, Lim=True, margin=margin)
    Y, dYr, indY, NY = _Ves_mesh_dlfromL_cython(YMinMax, dY, DY, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, DZ, Lim=True, margin=margin)
    Xn, Yn, Zn = len(X), len(Y), len(Z)
        
    Pts = np.array([np.tile(X,(Yn*Zn,1)).flatten(), np.tile(np.repeat(Y,Xn),(Zn,1)).flatten(), np.repeat(Z,Xn*Yn)])
    ind = np.repeat(NX*NY*indZ,Xn*Yn) + np.tile(np.repeat(NX*indY,Xn),(Zn,1)).flatten() + np.tile(indX,(Yn*Zn,1)).flatten()
    dV = dXr*dYr*dZr
    
    if VPoly is not None:
        indin = Path(VPoly.T).contains_points(Pts[1:,:].T, transform=None, radius=0.0)
        Pts, ind = Pts[:,indin], ind[indin]
    
    return Pts, dV, ind.astype(int), dXr, dYr, dZr
    

def _Ves_Vmesh_Lin_SubFromInd_cython(double dX, double dY, double dZ, 
                                     double[::1] XMinMax, double[::1] YMinMax, double[::1] ZMinMax,
                                     cnp.ndarray[long,ndim=1] ind, double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dY,dZ) "
    
    cdef cnp.ndarray[double,ndim=1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef long[::1] bla
    cdef cnp.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef cnp.ndarray[double,ndim=2] Pts
    
    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, bla, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, None, Lim=True, margin=margin)
    Y, dYr, bla, NY = _Ves_mesh_dlfromL_cython(YMinMax, dY, None, Lim=True, margin=margin)
    Z, dZr, bla, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, None, Lim=True, margin=margin)
    
    indZ = ind // (NX*NY)
    indY = (ind - NX*NY*indZ) // NX
    indX = ind - NX*NY*indZ - NX*indY
    Pts = np.array([X[indX.astype(int)], Y[indY.astype(int)], Z[indZ.astype(int)]])
    dV = dXr*dYr*dZr
    
    return Pts, dV, dXr, dYr, dZr







########################################################
########################################################
#       Meshing - Surface - Tor
########################################################

def _getBoundsInter2AngSeg(bool Full, double Phi0, double Phi1, double DPhi0, double DPhi1):
    """ Return Inter=True if an intersection exist (all angles in radians in [-pi;pi])

    If Inter, return Bounds, a list of tuples indicating the segments defining the intersection, with
    The intervals are ordered from lowest index to highest index (with respect to [Phi0,Phi1])
    """
    if Full:
        Bounds = [[DPhi0,DPhi1]] if DPhi0<=DPhi1 else [[-Cpi,DPhi1],[DPhi0,Cpi]]
        Inter = True
        Faces = [None, None]

    else:
        Inter, Bounds, Faces = False, None, [False,False]
        if Phi0<=Phi1:
            if DPhi0<=DPhi1:
                if DPhi0<=Phi1 and DPhi1>=Phi0:
                    Inter = True
                    Bounds = [[None,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
            else:
                if DPhi0<=Phi1 or DPhi1>=Phi0:
                    Inter = True
                    if DPhi0<=Phi1 and DPhi1>=Phi0:
                        Bounds = [[Phi0,DPhi1],[DPhi0,Phi1]]
                        Faces = [True,True] 
                    else:
                        Bounds = [[None,None]]
                        if DPhi0<=Phi1:
                            Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                            Bounds[0][1] = Phi1
                            Faces[0] = DPhi0<=Phi0
                            Faces[1] = True
                        else:
                            Bounds[0][0] = Phi0
                            Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                            Faces[0] = True
                            Faces[1] = DPhi1>=Phi1
        else:
            if DPhi0<=DPhi1:
                if DPhi0<=Phi1 or DPhi1>=Phi0:
                    Inter = True
                    if DPhi0<=Phi1 and DPhi1>=Phi0:
                        Bounds = [[Phi0,DPhi1],[DPhi0,Phi1]]
                        Faces = [True,True]
                    else:
                        Bounds = [[None,None]]
                        if DPhi0<=Phi1:
                            Bounds[0][0] = DPhi0
                            Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                            Faces[1] = DPhi1>=Phi1
                        else:
                            Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                            Bounds[0][1] = DPhi1
                            Faces[0] = DPhi0<=Phi0
            else:
                Inter = True
                if DPhi0>=Phi0 and DPhi1>=Phi0:
                    Bounds = [[Phi0,DPhi1],[DPhi0,Cpi],[-Cpi,Phi1]]
                    Faces = [True,True]
                elif DPhi0<=Phi1 and DPhi1<=Phi1:
                    Bounds = [[Phi0,Cpi],[-Cpi,DPhi1],[DPhi0,Phi1]]
                    Faces = [True,True]
                else:
                    Bounds = [[None,Cpi],[-Cpi,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[1][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
    return Inter, Bounds, Faces




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Tor_SubFromD_cython(double dL, double dRPhi, 
                                   double[:,::1] VPoly,
                                   DR=None, DZ=None, DPhi=None,
                                   double DIn=0., VIn=None, PhiMinMax=None,
                                   str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double[::1] R, Z, dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1, DDPhi, DPhiMinMax
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0, Indin
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj0=0, jj, nPhi0, nPhi1, zz, NP, NRPhi_int, Rratio, Ln
    cdef cnp.ndarray[double,ndim=2] Pts, indI, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] R0, dS, ind, dLr, Rref, dRPhir, iii
    cdef cnp.ndarray[long,ndim=1] indL, NL, indok
    
    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))]
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else Catan2(Csin(DPhi[0]),Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0], PhiMinMax[1], DPhi0, DPhi1)

    if Inter:

        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += 2.*Cpi
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += 2.*Cpi

        # Get the actual R and Z resolutions and mesh elements
        PtsCross, dLr, indL, NL, Rref, VPbis = _Ves_Smesh_Cross(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
        R0 = np.copy(Rref)
        NR0 = R0.size
        indin = np.ones((PtsCross.shape[1],),dtype=bool)
        if DR is not None:
            indin = indin & (R0>=DR[0]) & (R0<=DR[1])
        if DZ is not None:
            indin = indin & (PtsCross[1,:]>=DZ[0]) & (PtsCross[1,:]<=DZ[1])
        PtsCross, dLr, indL, Rref = PtsCross[:,indin], dLr[indin], indL[indin], Rref[indin]
        Ln = indin.sum()
        Indin = indin.nonzero()[0]

        dRPhir, dPhir = np.empty((Ln,)), np.empty((Ln,))
        Phin = np.zeros((Ln,),dtype=int)
        NRPhi = np.empty((Ln,))
        NRPhi0 = np.zeros((Ln,),dtype=int)
        nRPhi0, indR0ii = 0, 0
        NP, NPhimax = 0, 0
        Rratio = int(Cceil(np.max(Rref)/np.min(Rref)))
        indBounds = np.empty((2,nBounds),dtype=int)
        for ii in range(0,Ln):
            # Get the actual RPhi resolution and Phi mesh elements (! depends on R !)
            NRPhi[ii] = Cceil(DPhiMinMax*Rref[ii]/dRPhi)
            NRPhi_int = int(NRPhi[ii])
            dPhir[ii] = DPhiMinMax/NRPhi[ii]
            dRPhir[ii] = dPhir[ii]*Rref[ii]
            # Get index and cumulated indices from background
            for jj0 in range(indR0ii,NR0):
                if jj0==Indin[ii]:
                    indR0ii = jj0
                    break
                else:
                    nRPhi0 += <long>Cceil(DPhiMinMax*R0[jj0]/dRPhi)
                    NRPhi0[ii] = nRPhi0
            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to be created within those limits
            for kk in range(0,nBounds):
                abs0 = BC[kk][0]-PhiMinMax[0]                    
                if abs0-dPhir[ii]*Cfloor(abs0/dPhir[ii])<margin*dPhir[ii]:
                    nPhi0 = int(Cround(abs0/dPhir[ii]))
                else:
                    nPhi0 = int(Cfloor(abs0/dPhir[ii]))
                abs1 = BC[kk][1]-PhiMinMax[0]
                if abs1-dPhir[ii]*Cfloor(abs1/dPhir[ii])<margin*dPhir[ii]:
                    nPhi1 = int(Cround(abs1/dPhir[ii])-1)
                else:
                    nPhi1 = int(Cfloor(abs1/dPhir[ii]))
                indBounds[0,kk] = nPhi0
                indBounds[1,kk] = nPhi1
                Phin[ii] += nPhi1+1-nPhi0                    

            if ii==0:
                indI = np.nan*np.ones((Ln,Phin[ii]*Rratio+1))
            jj = 0
            for kk in range(0,nBounds):
                for kkb in range(indBounds[0,kk],indBounds[1,kk]+1):
                    indI[ii,jj] = <double>( kkb )
                    jj += 1
            NP += Phin[ii] 
        
        # Finish counting to get total number of points
        if jj0<=NR0-1:
            for jj0 in range(indR0ii,NR0):
                nRPhi0 += <long>Cceil(DPhiMinMax*R0[jj0]/dRPhi)
            
        # Compute Pts, dV and ind
        Pts = np.nan*np.ones((3,NP))
        ind = np.nan*np.ones((NP,))
        dS = np.nan*np.ones((NP,))
        # This triple loop is the longest part, it takes ~90% of the CPU time
        NP = 0
        if Out.lower()=='(x,y,z)':
            for ii in range(0,Ln):
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])]) # Some rare cases with doubles have to be eliminated
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    phi = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    Pts[0,NP] = PtsCross[0,ii]*Ccos(phi)
                    Pts[1,NP] = PtsCross[0,ii]*Csin(phi)
                    Pts[2,NP] = PtsCross[1,ii]
                    ind[NP] = NRPhi0[ii] + indiijj
                    dS[NP] = dLr[ii]*dRPhir[ii]
                    NP += 1
        else:
            for ii in range(0,Ln):
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])])
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    Pts[0,NP] = PtsCross[0,ii]
                    Pts[1,NP] = PtsCross[1,ii]
                    Pts[2,NP] = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    ind[NP] = NRPhi0[ii] + indiijj
                    dS[NP] = dLr[ii]*dRPhir[ii]
                    NP += 1
        indok = (~np.isnan(ind)).nonzero()[0]
        ind = ind[indok]
        dS = dS[indok]
        if len(indok)==1:
            Pts = Pts[:,indok].reshape((3,1)) 
        else:
            Pts = Pts[:,indok]
    else:
        Pts, dS, ind, NL, Rref, dRPhir, nRPhi0 = np.ones((3,0)), np.ones((0,)), np.ones((0,)), np.nan*np.ones((VPoly.shape[1]-1,)), np.ones((0,)), np.ones((0,)), 0
    return Pts, dS, ind.astype(int), NL, dLr, Rref, dRPhir, nRPhi0, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Tor_SubFromInd_cython(double dL, double dRPhi, 
                                     double[:,::1] VPoly, long[::1] ind, 
                                     double DIn=0., VIn=None, PhiMinMax=None,
                                     str Out='(X,Y,Z)', double margin=1.e-9):
    """ Return the desired submesh indicated by the (numerical) indices, for the desired resolution (dR,dZ,dRphi) """
    cdef double[::1] dRPhirRef, dPhir
    cdef long[::1] indL, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiL, iiphi, Ln, nn=0, kk=0, nRPhi0
    cdef double[:,::1] Phi
    cdef cnp.ndarray[double,ndim=2] Pts=np.empty((3,NP)), indI, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] R0, dS=np.empty((NP,)), dLr, dRPhir, Rref
    cdef cnp.ndarray[long,ndim=1] NL

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))]
        if PhiMinMax[1]>=PhiMinMax[0]:
            DPhiMinMax = PhiMinMax[1]-PhiMinMax[0]
        else:
            DPhiMinMax = 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
    
    
    # Get the actual R and Z resolutions and mesh elements
    PtsCross, dLrRef, indL, NL, RrefRef, VPbis = _Ves_Smesh_Cross(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
    Ln = dLrRef.size
    # Number of Phi per R
    dRPhirRef, dPhir, dRPhir = np.empty((Ln,)), np.empty((Ln,)), -np.ones((Ln,))
    dLr, Rref = -np.ones((Ln,)), -np.ones((Ln,))
    NRPhi, NRPhi0 = np.empty((Ln,),dtype=int), np.empty((Ln,),dtype=int)
    Rratio = int(Cceil(np.max(RrefRef)/np.min(RrefRef)))
    for ii in range(0,Ln):
        NRPhi[ii] = <long>(Cceil(DPhiMinMax*RrefRef[ii]/dRPhi))
        dRPhirRef[ii] = DPhiMinMax*RrefRef[ii]/<double>(NRPhi[ii])
        dPhir[ii] = DPhiMinMax/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((Ln,NRPhi[ii]*Rratio+1))
        else:
            NRPhi0[ii] = NRPhi0[ii-1] + NRPhi[ii-1]
        for jj in range(0,NRPhi[ii]):
            Phi[ii,jj] = PhiMinMax[0] + (0.5+<double>jj)*dPhir[ii]
    nRPhi0 = NRPhi0[Ln-1]+NRPhi[Ln-1]
            
    if Out.lower()=='(x,y,z)':
        for ii in range(0,NP):
            for jj in range(0,Ln+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiL = jj-1
            iiphi = ind[ii] - NRPhi0[iiL]
            Pts[0,ii] = PtsCross[0,iiL]*Ccos(Phi[iiL,iiphi])
            Pts[1,ii] = PtsCross[0,iiL]*Csin(Phi[iiL,iiphi])
            Pts[2,ii] = PtsCross[1,iiL]
            dS[ii] = dLrRef[iiL]*dRPhirRef[iiL]
            if dRPhir[iiL]==-1.:
                dRPhir[iiL] = dRPhirRef[iiL]
                dLr[iiL] = dLrRef[iiL]
                Rref[iiL] = RrefRef[iiL]
            
    else: 
        for ii in range(0,NP):
            for jj in range(0,Ln+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiL = jj-1
            iiphi = ind[ii] - NRPhi0[iiL]
            Pts[0,ii] = PtsCross[0,iiL]
            Pts[1,ii] = PtsCross[1,iiL]
            Pts[2,ii] = Phi[iiL,iiphi]
            dS[ii] = dLrRef[iiL]*dRPhirRef[iiL]
            if dRPhir[iiL]==-1.:
                dRPhir[iiL] = dRPhirRef[iiL]
                dLr[iiL] = dLrRef[iiL]
                Rref[iiL] = RrefRef[iiL]
    return Pts, dS, NL, dLr[dLr>-0.5], Rref[Rref>-0.5], dRPhir[dRPhir>-0.5], <long>nRPhi0, VPbis




########################################################
########################################################
#       Meshing - Surface - TorStruct
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromD_cython(double[::1] PhiMinMax, double dL, double dRPhi, 
                                         double[:,::1] VPoly,
                                         DR=None, DZ=None, DPhi=None,
                                         double DIn=0., VIn=None,
                                         str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double Dphi, dR0r=0., dZ0r=0.
    cdef int NR0=0, NZ0=0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
    cdef cnp.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef cnp.ndarray[long,ndim=1] indR0, indZ0, iind, iindF, indM, NL, ind
    cdef cnp.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[] 

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = np.array([-Cpi,Cpi])
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else Catan2(Csin(DPhi[0]),Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0], PhiMinMax[1], DPhi0, DPhi1)

    if Inter:
        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += 2.*Cpi
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += 2.*Cpi

        Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0. # Required distance effective at max R

        # Get the mesh for the faces
        if any(Faces) :
            R0, dR0r, indR0, NR0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=DR, Lim=True, margin=margin)
            Z0, dZ0r, indZ0, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=DZ, Lim=True, margin=margin)
            R0n, Z0n = len(R0), len(Z0)
            ptsrz = np.array([np.tile(R0,Z0n),np.repeat(Z0,R0n)])
            iind = NR0*np.repeat(indZ0,R0n) + np.tile(indR0,Z0n)
            indin = Path(VPoly.T).contains_points(ptsrz.T, transform=None, radius=0.0)
            if np.any(indin):
                ptsrz = ptsrz[:,indin] if indin.sum()>1 else ptsrz[:,indin].reshape((2,1))
                iindF = iind[indin]
                dsF = dR0r*dZ0r*np.ones((indin.sum(),))

        # First face
        if Faces[0]:
            if Out.lower()=='(x,y,z)':
                pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[0]+Dphi), ptsrz[0,:]*Csin(phiMinMax[0]+Dphi), ptsrz[1,:]])
            else:
                pts = np.array([ptsrz[0,:],ptsrz[1,:],(phiMinMax[0]+Dphi)*np.ones((indin.sum(),))])
            LPts.append( pts )
            Lind.append( iindF )
            LdS.append( dsF )

        # Main body
        PtsM, dSM, indM, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                                                               DR=DR, DZ=DZ, DPhi=[DPhi0,DPhi1],
                                                                                               DIn=DIn, VIn=VIn, PhiMinMax=phiMinMax,
                                                                                               Out=Out, margin=margin)

        if PtsM.shape[1]>=1:
            if PtsM.shape[1]==1:
                LPts.append(PtsM.reshape((3,1)))
            else:
                LPts.append(PtsM)
            Lind.append( indM + NR0*NZ0 )
            LdS.append( dSM )

        # Second face
        if Faces[1]:
            if Out.lower()=='(x,y,z)':
                pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[1]-Dphi), ptsrz[0,:]*Csin(phiMinMax[1]-Dphi), ptsrz[1,:]])
            else:
                pts = np.array([ptsrz[0,:],ptsrz[1,:],(phiMinMax[1]-Dphi)*np.ones((indin.sum(),))])
            LPts.append( pts )
            Lind.append( iindF + NR0*NZ0 + nRPhi0 )
            LdS.append( dsF )

        # Aggregate
        if len(LPts)==1:
            Pts = LPts[0]
            ind = Lind[0]
            dS = LdS[0]
        else:
            Pts = np.concatenate(tuple(LPts),axis=1)
            ind = np.concatenate(tuple(Lind)).astype(int)
            dS = np.concatenate(tuple(LdS))

    else:
        Pts, dS, ind, NL, Rref = np.ones((3,0)), np.ones((0,)), np.ones((0,),dtype=int), np.ones((0,),dtype=int), np.nan*np.ones((VPoly.shape[1]-1,))
        dLr, dR0r, dZ0r, dRPhir, VPbis = np.ones((0,)), 0., 0., np.ones((0,)), np.asarray(VPoly)

    return Pts, dS, ind, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromInd_cython(double[::1] PhiMinMax, double dL, double dRPhi, 
                                           double[:,::1] VPoly, cnp.ndarray[long,ndim=1] ind,
                                           double DIn=0., VIn=None,
                                           str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double Dphi, dR0r, dZ0r
    cdef int NR0, NZ0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
    cdef cnp.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef cnp.ndarray[long,ndim=1] bla, indR0, indZ0, iind, iindF, indM, NL
    cdef cnp.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[] 

    # Pre-format input
    Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0. # Required distance effective at max R

    # Get the basic meshes for the faces
    R0, dR0r, bla, NR0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, bla, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=None, Lim=True, margin=margin)
    
    PtsM, dSM, indM, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly, 
                                                                                           DR=None, DZ=None, DPhi=None,
                                                                                           DIn=DIn, VIn=VIn, PhiMinMax=phiMinMax,
                                                                                           Out=Out, margin=margin)
    # First face
    ii = (ind<NR0*NZ0).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NR0
        indR0 = (ind[ii]-indZ0*NR0)
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[0]+Dphi), R0[indR0]*Csin(phiMinMax[0]+Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0], (phiMinMax[0]+Dphi)*np.ones((nii,))])
        pts = pts if nii>1 else pts.reshape((3,1))
        LPts.append( pts )
        LdS.append( dR0r*dZ0r*np.ones((nii,)) )
 
    # Main body
    ii = (ind>=NR0*NZ0) & (ind<NR0*NZ0+PtsM.shape[1])
    nii = len(ii)
    if nii>0:
        pts = PtsM[:,ind[ii]-NR0*NZ0] if nii>1 else PtsM[:,ind[ii]-NR0*NZ0].reshape((3,1))
        LPts.append( PtsM[:,ind[ii]-NR0*NZ0] )
        LdS.append( dSM[ind[ii]-NR0*NZ0] )
        
    # Second face
    ii = (ind >= NR0*NZ0+PtsM.shape[1] ).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-(NR0*NZ0+PtsM.shape[1])) // NR0
        indR0 = ind[ii]-(NR0*NZ0+PtsM.shape[1]) - indZ0*NR0
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[1]-Dphi), R0[indR0]*Csin(phiMinMax[1]-Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0], (phiMinMax[1]-Dphi)*np.ones((nii,))])
        pts = pts if nii>1 else pts.reshape((3,1))
        LPts.append( pts )
        LdS.append( dR0r*dZ0r*np.ones((nii,)) )
        
    # Aggregate
    if len(LPts)==1:
        Pts = LPts[0]
        dS = LdS[0]
    elif len(LPts)>1:
        Pts = np.concatenate(tuple(LPts),axis=1)
        dS = np.concatenate(tuple(LdS))
        
    return Pts, dS, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis






########################################################
########################################################
#       Meshing - Surface - Lin
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef _check_DLvsLMinMax(double[::1] LMinMax, DL=None):
    Inter = 1
    if DL is not None:
        assert len(DL)==2 and DL[0]<DL[1]
        assert LMinMax[0]<LMinMax[1]
        DL = list(DL)
        if DL[0]>LMinMax[1] or DL[1]<LMinMax[0]:
            Inter = 0
        else:
            if DL[0]<=LMinMax[0]:
                DL[0] = None
            if DL[1]>=LMinMax[1]:
                DL[1] = None
    return Inter, DL



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Lin_SubFromD_cython(double[::1] XMinMax, double dL, double dX, 
                                   double[:,::1] VPoly,
                                   DX=None, DY=None, DZ=None,
                                   double DIn=0., VIn=None, double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dL) "
    cdef cnp.ndarray[double,ndim=1] X, Y0, Z0
    cdef double dXr, dY0r, dZ0r
    cdef int NY0, NZ0, Y0n, Z0n, NX, Xn, Ln, NR0, Inter=1
    cdef cnp.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] dS, dLr, Rref
    cdef cnp.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ind
    
    # Preformat
    # Adjust limits
    InterX, DX = _check_DLvsLMinMax(XMinMax,DX)
    InterY, DY = _check_DLvsLMinMax(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]),DY)
    InterZ, DZ = _check_DLvsLMinMax(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]),DZ)

    if InterX==1 and InterY==1 and InterZ==1:

        # Get the mesh for the faces
        Y0, dY0r, indY0, NY0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=DY, Lim=True, margin=margin)
        Z0, dZ0r, indZ0, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=DZ, Lim=True, margin=margin)
        Y0n, Z0n = len(Y0), len(Z0)

        # Get the actual R and Z resolutions and mesh elements
        X, dXr, indX, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DL=DX, Lim=True, margin=margin)
        Xn = len(X)
        PtsCross, dLr, indL, NL, Rref, VPbis = _Ves_Smesh_Cross(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
        NR0 = Rref.size
        indin = np.ones((PtsCross.shape[1],),dtype=bool)
        if DY is not None:
            if DY[0] is not None:
                indin = indin & (PtsCross[0,:]>=DY[0])
            if DY[1] is not None:
                indin = indin & (PtsCross[0,:]<=DY[1])
        if DZ is not None:
            if DZ[0] is not None:
                indin = indin & (PtsCross[1,:]>=DZ[0])
            if DZ[1] is not None:
                indin = indin & (PtsCross[1,:]<=DZ[1])
        PtsCross, dLr, indL, Rref = PtsCross[:,indin], dLr[indin], indL[indin], Rref[indin]
        Ln = indin.sum()
        # Agregating
        Pts = np.array([np.repeat(X,Ln), np.tile(PtsCross[0,:],Xn), np.tile(PtsCross[1,:],Xn)])
        ind = NY0*NZ0 + np.repeat(indX*NR0,Ln) + np.tile(indL,Xn)
        dS = np.tile(dLr*dXr,Xn)
        if DX is None or DX[0] is None:
            pts = np.array([(XMinMax[0]+DIn)*np.ones((Y0n*Z0n,)), np.tile(Y0,Z0n), np.repeat(Z0,Y0n)])
            iind = NY0*np.repeat(indZ0,Y0n) + np.tile(indY0,Z0n)
            indin = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
            if np.any(indin):
                pts = pts[:,indin].reshape((3,1)) if indin.sum()==1 else pts[:,indin]
                Pts = np.concatenate((pts,Pts),axis=1)
                ind = np.concatenate((iind[indin], ind))
                dS = np.concatenate((dY0r*dZ0r*np.ones((indin.sum(),)),dS))
        if DX is None or DX[1] is None:
            pts = np.array([(XMinMax[1]-DIn)*np.ones((Y0n*Z0n,)), np.tile(Y0,Z0n), np.repeat(Z0,Y0n)])
            iind = NY0*NZ0 + NX*NR0 + NY0*np.repeat(indZ0,Y0n) + np.tile(indY0,Z0n)
            indin = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
            if np.any(indin):
                pts = pts[:,indin].reshape((3,1)) if indin.sum()==1 else pts[:,indin]
                Pts = np.concatenate((Pts,pts),axis=1)
                ind = np.concatenate((ind,iind[indin]))
                dS = np.concatenate((dS,dY0r*dZ0r*np.ones((indin.sum(),))))

    else:
        Pts, dS, ind, NL, dLr, Rref = np.ones((3,0)), np.ones((0,)), np.ones((0,),dtype=int), np.ones((0,),dtype=int), np.ones((0,)), np.ones((0,))
        dXr, dY0r, dZ0r, VPbis = 0., 0., 0., np.ones((3,0))

    return Pts, dS, ind, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Lin_SubFromInd_cython(double[::1] XMinMax, double dL, double dX, 
                                     double[:,::1] VPoly, cnp.ndarray[long,ndim=1] ind,
                                     double DIn=0., VIn=None, double margin=1.e-9):
    " Return the desired surfacic submesh indicated by ind, for the desired resolution (dX,dL) "
    cdef double dXr, dY0r, dZ0r
    cdef int NX, NY0, NZ0, Ln, NR0, nii
    cdef list LPts, LdS
    cdef cnp.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] X, Y0, Z0, dS, dLr, Rref
    cdef cnp.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ii
    
    # Get the mesh for the faces
    Y0, dY0r, bla, NY0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, bla, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=None, Lim=True, margin=margin)
    
    # Get the actual R and Z resolutions and mesh elements
    X, dXr, bla, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DL=None, Lim=True, margin=margin)
    PtsCross, dLr, bla, NL, Rref, VPbis = _Ves_Smesh_Cross(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
    Ln = PtsCross.shape[1] 

    LPts, LdS = [], []
    # First face
    ii = (ind<NY0*NZ0).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NY0
        indY0 = (ind[ii]-indZ0*NY0)
        if nii==1:
            LPts.append( np.array([[XMinMax[0]+DIn], [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[0]+DIn)*np.ones((nii,)), Y0[indY0], Z0[indZ0]]) )
        LdS.append( dY0r*dZ0r*np.ones((nii,)) )

    # Cylinder
    ii = ((ind>=NY0*NZ0) & (ind<NY0*NZ0+NX*Ln)).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indX = (ind[ii]-NY0*NZ0) // Ln
        indL = (ind[ii]-NY0*NZ0 - Ln*indX)
        if nii==1:
            LPts.append( np.array([[X[indX]], [PtsCross[0,indL]], [PtsCross[1,indL]]]) )
            LdS.append( np.array([dXr*dLr[indL]]) )
        else:
            LPts.append( np.array([X[indX], PtsCross[0,indL], PtsCross[1,indL]]) )
            LdS.append( dXr*dLr[indL] )

    # End face
    ii = (ind >= NY0*NZ0+NX*Ln).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-NY0*NZ0-NX*Ln) // NY0
        indY0 = ind[ii]-NY0*NZ0-NX*Ln - NY0*indZ0
        if nii==1:
            LPts.append( np.array([[XMinMax[1]-DIn], [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[1]-DIn)*np.ones((nii,)), Y0[indY0], Z0[indZ0]]) )
        LdS.append( dY0r*dZ0r*np.ones((nii,)) )

    # Format output
    if len(LPts)==1:
        Pts, dS = LPts[0], LdS[0]
    else:
        Pts = np.concatenate(tuple(LPts),axis=1)
        dS = np.concatenate(tuple(LdS))
        
    return Pts, dS, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis




"""
########################################################
########################################################
########################################################
#                       LOS-specific
########################################################
########################################################
########################################################
"""




########################################################
########################################################
#       PIn POut
########################################################


def LOS_Calc_PInOut_VesStruct(Ds, dus,
                              cnp.ndarray[double, ndim=2,mode='c'] VPoly, cnp.ndarray[double, ndim=2,mode='c'] VIn, Lim=None,
                              LSPoly=None, LSLim=None, LSVIn=None,
                              RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9, EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9,
                              VType='Tor', Test=True):
    """ Compute the entry and exit point of all provided LOS for the provided vessel polygon (toroidal or linear), also return the normal vector at impact point and the index of the impact segment

    For each LOS,

    Parameters
    ----------



    Return
    ------
    PIn :       np.ndarray
        Point of entry (if any) of the LOS into the vessel, returned in (X,Y,Z) cartesian coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    POut :      np.ndarray
        Point of exit of the LOS from the vessel, returned in (X,Y,Z) cartesian coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    VOut :      np.ndarray

    IOut :      np.ndarray

    """
    if Test:
        assert type(Ds) is np.ndarray and type(dus) is np.ndarray and Ds.ndim in [1,2] and Ds.shape==dus.shape and Ds.shape[0]==3, "Args Ds and dus must be of the same shape (3,) or (3,NL) !"
        assert VPoly.shape[0]==2 and VIn.shape[0]==2 and VIn.shape[1]==VPoly.shape[1]-1, "Args VPoly and VIn must be of the same shape (2,NS) !"
        C1 = all([pp is None for pp in [LSPoly,LSLim,LSVIn]])
        C2 = all([hasattr(pp,'__iter__') and len(pp)==len(LSPoly) for pp in [LSPoly,LSLim,LSVIn]])
        assert C1 or C2, "Args LSPoly,LSLim,LSVIn must be None or lists of same len() !"
        assert RMin is None or type(RMin) in [float,int,np.float64,np.int64], "Arg RMin must be None or a float !"
        assert type(Forbid) is bool, "Arg Forbid must be a bool !"
        assert all([type(ee) in [int,float,np.int64,np.float64] and ee<1.e-4 for ee in [EpsUz,EpsVz,EpsA,EpsB,EpsPlane]]), "Args [EpsUz,EpsVz,EpsA,EpsB] must be floats < 1.e-4 !"
        assert type(VType) is str and VType.lower() in ['tor','lin'], "Arg VType must be a str in ['Tor','Lin'] !"

    cdef int ii, jj

    v = Ds.ndim==2
    if not v:
        Ds, dus = Ds.reshape((3,1)), dus.reshape((3,1))
    NL = Ds.shape[1]
    IOut = np.zeros((3,Ds.shape[1]))
    if VType.lower()=='tor':
        if RMin is None:
            RMin = 0.95*min(np.min(VPoly[0,:]),
                            np.min(np.hypot(Ds[0,:],Ds[1,:])))
        PIn, POut, VperpIn, VperpOut, IIn, IOut[2,:] = Calc_LOS_PInOut_Tor(Ds, dus, VPoly, VIn, Lim=Lim, Forbid=Forbid, RMin=RMin,
                                                                         EpsUz=EpsUz, EpsVz=EpsVz, EpsA=EpsA, EpsB=EpsB, EpsPlane=EpsPlane)
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        kPIn = np.sqrt(np.sum((PIn-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        assert np.allclose(kPIn,np.sum((PIn-Ds)*dus,axis=0),equal_nan=True)
        if LSPoly is not None:
            Ind = np.zeros((2,NL))
            for ii in range(0,len(LSPoly)):
                if LSLim[ii] is None or not all([hasattr(ll,'__iter__') for ll in LSLim[ii]]):
                    lslim = [LSLim[ii]]
                else:
                    lslim = LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn, pOut, vperpIn, vperpOut, iIn, iOut = Calc_LOS_PInOut_Tor(Ds, dus, LSPoly[ii], LSVIn[ii], Lim=lslim[jj], Forbid=Forbid, RMin=RMin,
                                                                                  EpsUz=EpsUz, EpsVz=EpsVz, EpsA=EpsA, EpsB=EpsB, EpsPlane=EpsPlane)
                    kpin = np.sqrt(np.sum((Ds-pIn)**2,axis=0))
                    indNoNan = (~np.isnan(kpin)) & (~np.isnan(kPOut))
                    indout = np.zeros((NL,),dtype=bool)
                    indout[indNoNan] = kpin[indNoNan]<kPOut[indNoNan]
                    indout[(~np.isnan(kpin)) & np.isnan(kPOut)] = True
                    if np.any(indout):
                        kPOut[indout] = kpin[indout]
                        POut[:,indout] = pIn[:,indout]
                        VperpOut[:,indout] = vperpIn[:,indout]
                        IOut[2,indout] = iIn[indout]
                        IOut[0,indout] = 1+ii
                        IOut[1,indout] = jj
    else:
        PIn, POut, VperpIn, VperpOut, IIn, IOut[2,:] = Calc_LOS_PInOut_Lin(Ds, dus, VPoly, VIn, Lim, EpsPlane=EpsPlane)
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        kPIn = np.sqrt(np.sum((PIn-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        assert np.allclose(kPIn,np.sum((PIn-Ds)*dus,axis=0),equal_nan=True)
        if LSPoly is not None:
            Ind = np.zeros((2,NL))
            for ii in range(0,len(LSPoly)):
                lslim = [LSLim[ii]] if not all([hasattr(ll,'__iter__') for ll in LSLim[ii]]) else LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn, pOut, vperpIn, vperpOut, iIn, iOut = Calc_LOS_PInOut_Lin(Ds, dus, LSPoly[ii], LSVIn[ii], lslim[jj], EpsPlane=EpsPlane)
                    kpin = np.sqrt(np.sum((Ds-pIn)**2,axis=0))
                    indNoNan = (~np.isnan(kpin)) & (~np.isnan(kPOut))
                    indout = np.zeros((NL,),dtype=bool)
                    indout[indNoNan] = kpin[indNoNan]<kPOut[indNoNan]
                    indout[(~np.isnan(kpin)) & np.isnan(kPOut)] = True
                    if np.any(indout):
                        kPOut[indout] = kpin[indout]
                        POut[:,indout] = pIn[:,indout]
                        VperpOut[:,indout] = vperpIn[:,indout]
                        IOut[2,indout] = iIn[indout]
                        IOut[0,indout] = 1+ii
                        IOut[1,indout] = jj

    if not v:
        PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut = PIn.flatten(), POut.flatten(), kPIn[0], kPOut[0], VperpIn.flatten(), VperpOut.flatten(), IIn[0], IOut.flatten()
    return PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut







@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Tor(double [:,::1] Ds, double [:,::1] us, double [:,::1] VPoly, double [:,::1] vIn, Lim=None,
                         bool Forbid=True, RMin=None, double EpsUz=1.e-6, double EpsVz=1.e-9, double EpsA=1.e-9, double EpsB=1.e-9, double EpsPlane=1.e-9):

    cdef int ii, jj, Nl=Ds.shape[1], Ns=vIn.shape[1]
    cdef double Rmin, upscaDp, upar2, Dpar2, Crit2, kout, kin
    cdef int indin=0, indout=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0., L0=0., L1=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef int Forbidbis, Forbid0
    cdef cnp.ndarray[double,ndim=2] SIn_=np.nan*np.ones((3,Nl)), SOut_=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=2] VPerp_In=np.nan*np.ones((3,Nl)), VPerp_Out=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=1] indIn_=np.nan*np.ones((Nl,)), indOut_=np.nan*np.ones((Nl,))

    cdef double[:,::1] SIn=SIn_, SOut=SOut_, VPerpIn=VPerp_In, VPerpOut=VPerp_Out
    cdef double[::1] indIn=indIn_, indOut=indOut_
    if Lim is not None:
        L0 = Catan2(Csin(Lim[0]),Ccos(Lim[0]))
        L1 = Catan2(Csin(Lim[1]),Ccos(Lim[1]))

    ################
    # Prepare input
    if RMin is None:
        Rmin = 0.95*min(np.min(VPoly[0,:]),np.min(np.hypot(Ds[0,:],Ds[1,:])))
    else:
        Rmin = RMin

    ################
    # Compute
    if Forbid:
        Forbid0, Forbidbis = 1, 1
    else:
        Forbid0, Forbidbis = 0, 0
    for ii in range(0,Nl):
        upscaDp = us[0,ii]*Ds[0,ii] + us[1,ii]*Ds[1,ii]
        upar2 = us[0,ii]**2 + us[1,ii]**2
        Dpar2 = Ds[0,ii]**2 + Ds[1,ii]**2
        # Prepare in case Forbid is True
        if Forbid0 and not Dpar2>0:
            Forbidbis = 0
        if Forbidbis:
            # Compute coordinates of the 2 points where the tangents touch the inner circle
            L = Csqrt(Dpar2-Rmin**2)
            S1X = (Rmin**2*Ds[0,ii]+Rmin*Ds[1,ii]*L)/Dpar2
            S1Y = (Rmin**2*Ds[1,ii]-Rmin*Ds[0,ii]*L)/Dpar2
            S2X = (Rmin**2*Ds[0,ii]-Rmin*Ds[1,ii]*L)/Dpar2
            S2Y = (Rmin**2*Ds[1,ii]+Rmin*Ds[0,ii]*L)/Dpar2

        # Compute all solutions
        # Set tolerance value for us[2,ii]
        # EpsUz is the tolerated DZ across 20m (max Tokamak size)
        Crit2 = EpsUz**2*upar2/400.
        kout, kin, Done = 1.e12, 1e12, 0
        # Case with horizontal semi-line
        if us[2,ii]**2<Crit2:
            for jj in range(0,Ns):
                # Solutions exist only in the case with non-horizontal segment (i.e.: cone, not plane)
                if (VPoly[1,jj+1]-VPoly[1,jj])**2>EpsVz**2:
                    q = (Ds[2,ii]-VPoly[1,jj])/(VPoly[1,jj+1]-VPoly[1,jj])
                    # The intersection must stand on the segment
                    if q>=0 and q<1:
                        C = q**2*(VPoly[0,jj+1]-VPoly[0,jj])**2 + 2.*q*VPoly[0,jj]*(VPoly[0,jj+1]-VPoly[0,jj]) + VPoly[0,jj]**2
                        delta = upscaDp**2 - upar2*(Dpar2-C)
                        if delta>0.:
                            sqd = Csqrt(delta)
                            # The intersection must be on the semi-line (i.e.: k>=0)
                            # First solution
                            if -upscaDp - sqd >=0:
                                k = (-upscaDp - sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                    # Get the normalized perpendicular vector at intersection
                                    phi = Catan2(sol1,sol0)
                                    # Check sol inside the Lim
                                    if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                        # Get the scalar product to determine entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            indout = jj
                                            Done = 1
                                            #print(1, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            indin = jj
                                            #print(2, k)

                            # Second solution
                            if -upscaDp + sqd >=0:
                                k = (-upscaDp + sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                    # Get the normalized perpendicular vector at intersection
                                    phi = Catan2(sol1,sol0)
                                    if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                        # Get the scalar product to determine entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            indout = jj
                                            Done = 1
                                            #print(3, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            indin = jj
                                            #print(4, k)

        # More general non-horizontal semi-line case
        else:
            for jj in range(Ns):
                v0, v1 = VPoly[0,jj+1]-VPoly[0,jj], VPoly[1,jj+1]-VPoly[1,jj]
                A = v0**2 - upar2*(v1/us[2,ii])**2
                B = VPoly[0,jj]*v0 + v1*(Ds[2,ii]-VPoly[1,jj])*upar2/us[2,ii]**2 - upscaDp*v1/us[2,ii]
                C = -upar2*(Ds[2,ii]-VPoly[1,jj])**2/us[2,ii]**2 + 2.*upscaDp*(Ds[2,ii]-VPoly[1,jj])/us[2,ii] - Dpar2 + VPoly[0,jj]**2

                if A**2<EpsA**2 and B**2>EpsB**2:
                    q = -C/(2.*B)
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 1, k, kout, sca0, sca1, sca2
                                if sca0<0 and sca1<0 and sca2<0:
                                    continue
                            # Get the normalized perpendicular vector at intersection
                            phi = Catan2(sol1,sol0)
                            if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                # Get the scalar product to determine entry or exit point
                                sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                if sca<=0 and k<kout:
                                    kout = k
                                    indout = jj
                                    Done = 1
                                    #print(5, k)
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj
                                    #print(6, k)

                elif A**2>=EpsA**2 and B**2>A*C:
                    sqd = Csqrt(B**2-A*C)
                    # First solution
                    q = (-B + sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 2, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        indout = jj
                                        Done = 1
                                        #print(7, k, q, A, B, C, sqd)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(8, k, jj)

                    # Second solution
                    q = (-B - sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]

                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 3, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        indout = jj
                                        Done = 1
                                        #print(9, k, jj)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(10, k, q, A, B, C, sqd, v0, v1, jj)

        if Lim is not None:
            ephiIn0, ephiIn1 = -Csin(L0), Ccos(L0)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    # Check if in VPoly
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L0) + (Ds[1,ii]+k*us[1,ii])*Csin(L0), Ds[2,ii]+k*us[2,ii]
                    if Path(VPoly.T).contains_point([sol0,sol1], transform=None, radius=0.0):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -1
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k
                            indin = -1

            ephiIn0, ephiIn1 = Csin(L1), -Ccos(L1)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L1) + (Ds[1,ii]+k*us[1,ii])*Csin(L1), Ds[2,ii]+k*us[2,ii]
                    # Check if in VPoly
                    if Path(VPoly.T).contains_point([sol0,sol1], transform=None, radius=0.0):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -2
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k
                            indin = -2

        if Done==1:
            SOut[0,ii] = Ds[0,ii] + kout*us[0,ii]
            SOut[1,ii] = Ds[1,ii] + kout*us[1,ii]
            SOut[2,ii] = Ds[2,ii] + kout*us[2,ii]
            phi = Catan2(SOut[1,ii],SOut[0,ii])
            if indout==-1:
                VPerpOut[0,ii] = -Csin(L0)
                VPerpOut[1,ii] = Ccos(L0)
                VPerpOut[2,ii] = 0.
            elif indout==-2:
                VPerpOut[0,ii] = Csin(L1)
                VPerpOut[1,ii] = -Ccos(L1)
                VPerpOut[2,ii] = 0.
            else:
                VPerpOut[0,ii] = Ccos(phi)*vIn[0,indout]
                VPerpOut[1,ii] = Csin(phi)*vIn[0,indout]
                VPerpOut[2,ii] = vIn[1,indout]
            indOut[ii] = indout
            if kin<kout:
                SIn[0,ii] = Ds[0,ii] + kin*us[0,ii]
                SIn[1,ii] = Ds[1,ii] + kin*us[1,ii]
                SIn[2,ii] = Ds[2,ii] + kin*us[2,ii]
                phi = Catan2(SIn[1,ii],SIn[0,ii])
                if indin==-1:
                    VPerpIn[0,ii] = Csin(L0)
                    VPerpIn[1,ii] = -Ccos(L0)
                    VPerpIn[2,ii] = 0.
                elif indin==-2:
                    VPerpIn[0,ii] = -Csin(L1)
                    VPerpIn[1,ii] = Ccos(L1)
                    VPerpIn[2,ii] = 0.
                else:
                    VPerpIn[0,ii] = -Ccos(phi)*vIn[0,indin]
                    VPerpIn[1,ii] = -Csin(phi)*vIn[0,indin]
                    VPerpIn[2,ii] = -vIn[1,indin]
                indIn[ii] = indin

    return np.asarray(SIn), np.asarray(SOut), np.asarray(VPerpIn), np.asarray(VPerpOut), np.asarray(indIn), np.asarray(indOut)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Lin(double[:,::1] Ds, double [:,::1] us, double[:,::1] VPoly, double[:,::1] VIn, Lim, double EpsPlane=1.e-9):

    cdef int ii=0, jj=0, Nl=Ds.shape[1], Ns=VIn.shape[1]
    cdef double kin, kout, scauVin, q, X, sca, L0=<double>Lim[0], L1=<double>Lim[1]
    cdef int indin=0, indout=0, Done=0
    cdef cnp.ndarray[double,ndim=2] SIn_=np.nan*np.ones((3,Nl)), SOut_=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=2] VPerp_In=np.nan*np.ones((3,Nl)), VPerp_Out=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=1] indIn_=np.nan*np.ones((Nl,)), indOut_=np.nan*np.ones((Nl,))

    cdef double[:,::1] SIn=SIn_, SOut=SOut_, VPerpIn=VPerp_In, VPerpOut=VPerp_Out
    cdef double[::1] indIn=indIn_, indOut=indOut_

    for ii in range(0,Nl):

        kout, kin, Done = 1.e12, 1e12, 0
        # For cylinder
        for jj in range(0,Ns):
            scauVin = us[1,ii]*VIn[0,jj] + us[2,ii]*VIn[1,jj]
            # Only if plane not parallel to line
            if Cabs(scauVin)>EpsPlane:
                k = -((Ds[1,ii]-VPoly[0,jj])*VIn[0,jj] + (Ds[2,ii]-VPoly[1,jj])*VIn[1,jj])/scauVin
                # Only if on good side of semi-line
                if k>=0.:
                    V1, V2 = VPoly[0,jj+1]-VPoly[0,jj], VPoly[1,jj+1]-VPoly[1,jj]
                    q = ((Ds[1,ii] + k*us[1,ii]-VPoly[0,jj])*V1 + (Ds[2,ii] + k*us[2,ii]-VPoly[1,jj])*V2)/(V1**2+V2**2)
                    # Only of on the fraction of plane
                    if q>=0. and q<1.:
                        X = Ds[0,ii] + k*us[0,ii]
                        # Only if within limits
                        if X>=L0 and X<=L1:
                            sca = us[1,ii]*VIn[0,jj] + us[2,ii]*VIn[1,jj]
                            # Only if new
                            if sca<=0 and k<kout:
                                kout = k
                                indout = jj
                                Done = 1
                            elif sca>=0 and k<min(kin,kout):
                                kin = k
                                indin = jj

        # For two faces
        # Only if plane not parallel to line
        if Cabs(us[0,ii])>EpsPlane:
            # First face
            k = -(Ds[0,ii]-L0)/us[0,ii]
            # Only if on good side of semi-line
            if k>=0.:
                # Only if inside VPoly
                if Path(VPoly.T).contains_point([Ds[1,ii]+k*us[1,ii],Ds[2,ii]+k*us[2,ii]], transform=None, radius=0.0):
                    if us[0,ii]<=0 and k<kout:
                        kout = k
                        indout = -1
                        Done = 1
                    elif us[0,ii]>=0 and k<min(kin,kout):
                        kin = k
                        indin = -1
            # Second face
            k = -(Ds[0,ii]-L1)/us[0,ii]
            # Only if on good side of semi-line
            if k>=0.:
                # Only if inside VPoly
                if Path(VPoly.T).contains_point([Ds[1,ii]+k*us[1,ii],Ds[2,ii]+k*us[2,ii]], transform=None, radius=0.0):
                    if us[0,ii]>=0 and k<kout:
                        kout = k
                        indout = -2
                        Done = 1
                    elif us[0,ii]<=0 and k<min(kin,kout):
                        kin = k
                        indin = -2

        if Done==1:
            SOut[0,ii] = Ds[0,ii] + kout*us[0,ii]
            SOut[1,ii] = Ds[1,ii] + kout*us[1,ii]
            SOut[2,ii] = Ds[2,ii] + kout*us[2,ii]
            # To be finished
            # phi = Catan2(SOut[1,ii],SOut[0,ii])
            if indout==-1:
                VPerpOut[0,ii] = 1.
                VPerpOut[1,ii] = 0.
                VPerpOut[2,ii] = 0.
            elif indout==-2:
                VPerpOut[0,ii] = -1.
                VPerpOut[1,ii] = 0.
                VPerpOut[2,ii] = 0.
            else:
                VPerpOut[0,ii] = 0.
                VPerpOut[1,ii] = VIn[0,indout]
                VPerpOut[2,ii] = VIn[1,indout]
            indOut[ii] = indout
            if kin<kout:
                SIn[0,ii] = Ds[0,ii] + kin*us[0,ii]
                SIn[1,ii] = Ds[1,ii] + kin*us[1,ii]
                SIn[2,ii] = Ds[2,ii] + kin*us[2,ii]
                if indin==-1:
                    VPerpIn[0,ii] = -1.
                    VPerpIn[1,ii] = 0.
                    VPerpIn[2,ii] = 0.
                elif indin==-2:
                    VPerpIn[0,ii] = 1.
                    VPerpIn[1,ii] = 0.
                    VPerpIn[2,ii] = 0.
                else:
                    VPerpIn[0,ii] = 0.
                    VPerpIn[1,ii] = -VIn[0,indin]
                    VPerpIn[2,ii] = -VIn[1,indin]
                indIn[ii] = indin

    return np.asarray(SIn), np.asarray(SOut), np.asarray(VPerpIn), np.asarray(VPerpOut), np.asarray(indIn), np.asarray(indOut)




######################################################################
#               Sampling
######################################################################

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.profile(False)
@cython.linetrace(False)
@cython.binding(False)
def LOS_get_sample(double[:,::1] Ds, double[:,::1] us, dL,
                   double[:,::1] DLs, str dLMode='abs', str method='sum',
                   Test=True):

    """ Return the sampled line, with the specified method

    'linspace': return the N+1 edges, including the first and last point
    'sum' :     return N segments centers
    'simps':    return N+1 egdes, N even (for scipy.integrate.simps)
    'romb' :    return N+1 edges, N+1 = 2**k+1 (for scipy.integrate.romb)
    """
    if Test:
        assert Ds.shape[0]==us.shape[0]==3, "Args Ds, us - dim 0"
        assert DLs.shape[0]==2, "Arg DLs - dim 0"
        assert Ds.shape[1]==us.shape[1]==DLs.shape[1], "Args Ds, us, DLs 1"
        C0 = not hasattr(dL,'__iter__') and dL>0.
        C1 = hasattr(dL,'__iter__') and len(dL)==Ds.shape[1] and np.all(dL>0.)
        assert C0 or C1, "Arg dL must be >0. !"
        assert dLMode.lower() in ['abs','rel'], "Arg dLMode in ['abs','rel']"
        assert method.lower() in ['sum','simps','romb'], "Arg method"

    cdef unsigned int ii, jj, N, ND = Ds.shape[1]
    cdef double kkk, D0, D1, D2, u0, u1, u2, dl0, dl
    cdef cnp.ndarray[double,ndim=1] dLr = np.empty((ND,),dtype=float)
    cdef cnp.ndarray[double,ndim=1] kk
    cdef cnp.ndarray[double,ndim=2] pts
    cdef list Pts=[0 for ii in range(0,ND)], k=[0 for ii in range(0,ND)]

    dLMode = dLMode.lower()
    method = method.lower()
    # Case with unique dL
    if not hasattr(dL,'__iter__'):
        if dLMode=='rel':
            N = <long>(Cceil(1./dL))
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    kk = np.empty((N,),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk
            elif method=='simps':
                N = N if N%2==0 else N+1
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            else:
                N = 2**(<long>(Cceil(Clog2(<double>N))))
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

        else:
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    kk = np.empty((N,),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            elif method=='simps':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    N = N if N%2==0 else N+1
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            else:
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

    # Case with different resolution for each LOS
    else:
        if dLMode=='rel':
            if method=='sum':
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    kk = np.empty((N,),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk
            elif method=='simps':
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    N = N if N%2==0 else N+1
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            else:
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

        else:
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    kk = np.empty((N,),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            elif method=='simps':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    N = N if N%2==0 else N+1
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

            else:
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    dLr[ii] = dl
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    kk = np.empty((N+1,),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        kk[jj] = kkk
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    Pts[ii] = pts
                    k[ii] = kk

    return Pts, k, dLr






######################################################################
#               Signal calculation
######################################################################


cdef get_insp(ff):
    out = insp(ff)
    if sys.version[0]=='3':
        pars = out.parameters.values()
        na = np.sum([(pp.kind==pp.POSITIONAL_OR_KEYWORD
                      and pp.default is pp.empty) for pp in pars])
        kw = [pp.name for pp in pars if (pp.kind==pp.POSITIONAL_OR_KEYWORD
                                         and pp.default is not pp.empty)]
    else:
        nat, nak = len(out.args), len(out.defaults)
        na = nat-nak
        kw = [out.args[ii] for ii in range(nat-1,na-1,-1)][::-1]
    return na, kw



def check_ff(ff, t=None, Ani=None, bool Vuniq=False):
    cdef bool ani
    stre = "Input emissivity function (ff)"
    assert hasattr(ff,'__call__'), stre+" must be a callable (function) !"
    na, kw = get_insp(ff)
    assert na==1, stre+" must take only one positional argument: ff(Pts) !"
    assert 't' in kw, stre+" must have kwarg 't=None' for time vector !"
    C = type(t) in [int,float,np.int64,np.float64] or hasattr(t,'__iter__')
    assert t is None or C, "Arg t must be None, a scalar or an iterable !"
    Pts = np.array([[1,2],[3,4],[5,6]])
    NP = Pts.shape[1]
    try:
        out = ff(Pts, t=t)
    except Exception:
        Str = stre+" must take one positional arg: a (3,N) np.ndarray"
        assert False, Str
    if hasattr(t,'__iter__'):
        nt = len(t)
        Str = ("ff(Pts,t=t), where Pts is a (3,N) np.array and "
               +"t a len()=nt iterable, must return a (nt,N) np.ndarray !")
        assert type(out) is np.ndarray and out.shape==(nt,NP), Str
    else:
        Str = ("When fed a (3,N) np.array only, or if t is a scalar,"
               +" ff must return a (N,) np.ndarray !")
        assert type(out) is np.ndarray and out.shape==(NP,), Str

    ani = ('Vect' in kw) if Ani is None else Ani
    if ani:
        Str = "If Ani=True, ff must take a keyword argument 'Vect=None' !"
        assert 'Vect' in kw, Str
        Vect = np.array([1,2,3]) if Vuniq else np.ones(Pts.shape)
        try:
            out = ff(Pts, Vect=Vect, t=t)
        except Exception:
            Str = "If Ani=True, ff must handle multiple points Pts (3,N) with "
            if Vuniq:
                Str += "a unique common vector (Vect as a len()=3 iterable)"
            else:
                Str += "multiple vectors (Vect as a (3,N) np.ndarray)"
            assert False, Str
        if hasattr(t,'__iter__'):
            Str = ("If Ani=True, ff must return a (nt,N) np.ndarray when "
                   +"Pts is (3,N), Vect is provided and t is (nt,)")
            assert type(out) is np.ndarray and out.shape==(nt,NP), Str
        else:
            Str = ("If Ani=True, ff must return a (nt,N) np.ndarray when "
                   +"Pts is (3,N), Vect is provided and t is (nt,)")
            assert type(out) is np.ndarray and out.shape==(NP,), Str
    return ani



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.profile(False)
@cython.linetrace(False)
@cython.binding(False)
def LOS_calc_signal(ff, double[:,::1] Ds, double[:,::1] us, dL,
                   double[:,::1] DLs, t=None, Ani=None, dict fkwdargs={},
                   str dLMode='abs', str method='simps',
                   Test=True):

    """ Return the sampled line, with the specified method

    'linspace': return the N+1 edges, including the first and last point
    'sum' :     return N segments centers
    'simps':    return N+1 egdes, N even (for scipy.integrate.simps)
    'romb' :    return N+1 edges, N+1 = 2**k+1 (for scipy.integrate.romb)
    """
    if Test:
        assert Ds.shape[0]==us.shape[0]==3, "Args Ds, us - dim 0"
        assert DLs.shape[0]==2, "Arg DLs - dim 0"
        assert Ds.shape[1]==us.shape[1]==DLs.shape[1], "Args Ds, us, DLs 1"
        C0 = not hasattr(dL,'__iter__') and dL>0.
        C1 = hasattr(dL,'__iter__') and len(dL)==Ds.shape[1] and np.all(dL>0.)
        assert C0 or C1, "Arg dL must be >0. !"
        assert dLMode.lower() in ['abs','rel'], "Arg dLMode in ['abs','rel']"
        assert method.lower() in ['sum','simps','romb'], "Arg method"
    # Testing function
    cdef bool ani = check_ff(ff,t=t,Ani=Ani)

    cdef unsigned int nt, axm, ii, jj, N, ND = Ds.shape[1]
    cdef double kkk, D0, D1, D2, u0, u1, u2, dl0, dl
    cdef cnp.ndarray[double,ndim=2] pts
    if t is None or not hasattr(t,'__iter__'):
        nt = 1
        axm = 0
    else:
        nt = len(t)
        axm = 1
    cdef cnp.ndarray[double,ndim=2] sig = np.empty((nt,ND),dtype=float)

    dLMode = dLMode.lower()
    method = method.lower()
    # Case with unique dL
    if not hasattr(dL,'__iter__'):
        if dLMode=='rel':
            N = <long>(Cceil(1./dL))
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = np.sum(ff(pts,t=t,**fkwdargs),axis=axm)*dl

            elif method=='simps':
                N = N if N%2==0 else N+1
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.simps(ff(pts,t=t,**fkwdargs),
                                              x=None,dx=dl,axis=axm)

            else:
                N = 2**(<long>(Cceil(Clog2(<double>N))))
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.romb(ff(pts,t=t,**fkwdargs),
                                             dx=dl,axis=axm,show=False)

        else:
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = np.sum(ff(pts,t=t,**fkwdargs),axis=axm)*dl

            elif method=='simps':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    N = N if N%2==0 else N+1
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.simps(ff(pts,t=t,**fkwdargs),
                                              x=None,dx=dl,axis=axm)

            else:
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.romb(ff(pts,t=t,**fkwdargs),
                                             dx=dl,axis=axm,show=False)

    # Case with different resolution for each LOS
    else:
        if dLMode=='rel':
            if method=='sum':
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = np.sum(ff(pts,t=t,**fkwdargs),axis=axm)*dl
            elif method=='simps':
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    N = N if N%2==0 else N+1
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.simps(ff(pts,t=t,**fkwdargs),
                                              x=None,dx=dl,axis=axm)

            else:
                for ii in range(0,ND):
                    N = <long>(Cceil(1./dL[ii]))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl0 = DLs[0,ii]
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.romb(ff(pts,t=t,**fkwdargs),
                                             dx=dl,axis=axm,show=False)

        else:
            if method=='sum':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N),dtype=float)
                    for jj in range(0,N):
                        kkk = dl0 + (0.5+<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = np.sum(ff(pts,t=t,**fkwdargs),axis=axm)*dl

            elif method=='simps':
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    N = N if N%2==0 else N+1
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.simps(ff(pts,t=t,**fkwdargs),
                                              x=None,dx=dl,axis=axm)

            else:
                for ii in range(0,ND):
                    dl0 = DLs[0,ii]
                    # Compute the number of intervals to satisfy the resolution
                    N = <long>(Cceil((DLs[1,ii]-dl0)/dL[ii]))
                    N = 2**(<long>(Cceil(Clog2(<double>N))))
                    dl = (DLs[1,ii]-dl0)/<double>N
                    D0, D1, D2 = Ds[0,ii], Ds[1,ii], Ds[2,ii]
                    u0, u1, u2 = us[0,ii], us[1,ii], us[2,ii]
                    pts = np.empty((3,N+1),dtype=float)
                    for jj in range(0,N+1):
                        kkk = dl0 + (<double>jj)*dl
                        pts[0,jj] = D0 + kkk*u0
                        pts[1,jj] = D1 + kkk*u1
                        pts[2,jj] = D2 + kkk*u2
                    if ani:
                        fkwdargs['Vect'] = (-u0,-u1,-u2)
                    sig[:,ii] = scpintg.romb(ff(pts,t=t,**fkwdargs),
                                             dx=dl,axis=axm,show=False)

    if nt==1:
        return sig.ravel()
    else:
        return sig








######################################################################
#               Sinogram-specific
######################################################################


cdef LOS_sino_findRootkPMin_Tor(double uParN, double uN, double Sca, double RZ0, double RZ1, double ScaP, double DParN, double kOut, double D0, double D1, double D2, double u0, double u1, double u2, str Mode='LOS'):
    cdef double a4 = (uParN*uN*uN)**2, a3 = 2*( (Sca-RZ1*u2)*(uParN*uN)**2 + ScaP*uN**4 )
    cdef double a2 = (uParN*(Sca-RZ1*u2))**2 + 4.*ScaP*(Sca-RZ1*u2)*uN**2 + (DParN*uN*uN)**2 - (RZ0*uParN*uParN)**2
    cdef double a1 = 2*( ScaP*(Sca-RZ1*u2)**2 + (Sca-RZ1*u2)*(DParN*uN)**2 - ScaP*(RZ0*uParN)**2 )
    cdef double a0 = ((Sca-RZ1*u2)*DParN)**2 - (RZ0*ScaP)**2
    cdef cnp.ndarray roo = np.roots(np.array([a4,a3,a2,a1,a0]))
    cdef list KK = list(np.real(roo[np.isreal(roo)]))   # There might be several solutions
    cdef list Pk, Pk2D, rk
    cdef double kk, kPMin
    if Mode=='LOS':                     # Take solution on physical LOS
        if any([kk>=0 and kk<=kOut for kk in KK]):
            KK = [kk for kk in KK if kk>=0 and kk<=kOut]
            Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
            Pk2D = [(Csqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
            rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
            kPMin = KK[rk.index(min(rk))]
        else:
            kPMin = min([Cabs(kk) for kk in KK])  # Else, take the one closest to D
    else:
        Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
        Pk2D = [(Csqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
        rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
        kPMin = KK[rk.index(min(rk))]
    return kPMin



cdef LOS_sino_Tor(double D0, double D1, double D2, double u0, double u1, double u2, double RZ0, double RZ1, str Mode='LOS', double kOut=np.inf):
    cdef double    uN = Csqrt(u0**2+u1**2+u2**2), uParN = Csqrt(u0**2+u1**2), DParN = Csqrt(D0**2+D1**2)
    cdef double    Sca = u0*D0+u1*D1+u2*D2, ScaP = u0*D0+u1*D1
    cdef double    kPMin
    if uParN == 0.:
        kPMin = (RZ1-D2)/u2
    else:
        kPMin = LOS_sino_findRootkPMin_Tor(uParN, uN, Sca, RZ0, RZ1, ScaP, DParN, kOut, D0, D1, D2, u0, u1, u2, Mode=Mode)
    cdef double    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef double    PMin2norm = Csqrt(PMin0**2+PMin1**2)
    cdef double    PMin2D0 = PMin2norm, PMin2D1 = PMin2
    cdef double    RMin = Csqrt((PMin2D0-RZ0)**2+(PMin2D1-RZ1)**2)
    cdef double    eTheta0 = -PMin1/PMin2norm, eTheta1 = PMin0/PMin2norm, eTheta2 = 0.
    cdef double    vP0 = PMin2D0-RZ0, vP1 = PMin2D1-RZ1
    cdef double    Theta = Catan2(vP1,vP0)
    cdef double    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef double    er2D0 = Ccos(ImpTheta), er2D1 = Csin(ImpTheta)
    cdef double    p = vP0*er2D0 + vP1*er2D1
    cdef double    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef double    phi = Casin(-uN0*eTheta0 -uN1*eTheta1 -uN2*eTheta2)
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p, ImpTheta, phi



cdef LOS_sino_Lin(double D0, double D1, double D2, double u0, double u1, double u2, double RZ0, double RZ1, str Mode='LOS', double kOut=np.inf):
    cdef double    kPMin
    if u0**2==1.:
        kPMin = 0.
    else:
        kPMin = ( (RZ0-D1)*u1+(RZ1-D2)*u2 ) / (1-u0**2)
    kPMin = kOut if Mode=='LOS' and kPMin > kOut else kPMin
    cdef double    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef double    RMin = Csqrt((PMin1-RZ0)**2+(PMin2-RZ1)**2)
    cdef double    vP0 = PMin1-RZ0, vP1 = PMin2-RZ1
    cdef double    Theta = Catan2(vP1,vP0)
    cdef double    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef double    er2D0 = Ccos(ImpTheta), er2D1 = Csin(ImpTheta)
    cdef double    p = vP0*er2D0 + vP1*er2D1
    cdef double    uN = Csqrt(u0**2+u1**2+u2**2)
    cdef double    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef double    phi = Catan2(uN0, Csqrt(uN1**2+uN2**2))
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p, ImpTheta, phi


def LOS_sino(double[:,::1] D, double[:,::1] u, double[::1] RZ, double[::1] kOut, str Mode='LOS', str VType='Tor'):
    cdef unsigned int nL = D.shape[1], ii
    cdef tuple out
    cdef cnp.ndarray[double,ndim=2] PMin = np.empty((3,nL))
    cdef cnp.ndarray[double,ndim=1] kPMin=np.empty((nL,)), RMin=np.empty((nL,))
    cdef cnp.ndarray[double,ndim=1] Theta=np.empty((nL,)), p=np.empty((nL,))
    cdef cnp.ndarray[double,ndim=1] ImpTheta=np.empty((nL,)), phi=np.empty((nL,))
    if VType.lower()=='tor':
        for ii in range(0,nL):
            out = LOS_sino_Tor(D[0,ii],D[1,ii],D[2,ii],u[0,ii],u[1,ii],u[2,ii],
                               RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
            ((PMin[0,ii],PMin[1,ii],PMin[2,ii]),
             kPMin[ii], RMin[ii], Theta[ii], p[ii], ImpTheta[ii], phi[ii]) = out
    else:
        for ii in range(0,nL):
            out = LOS_sino_Lin(D[0,ii],D[1,ii],D[2,ii],u[0,ii],u[1,ii],u[2,ii],
                               RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
            ((PMin[0,ii],PMin[1,ii],PMin[2,ii]),
             kPMin[ii], RMin[ii], Theta[ii], p[ii], ImpTheta[ii], phi[ii]) = out
    return PMin, kPMin, RMin, Theta, p, ImpTheta, phi
