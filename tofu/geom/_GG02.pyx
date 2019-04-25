# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
cimport cython
cimport numpy as np
from cpython cimport bool
from libc.math cimport sqrt as Csqrt, ceil as Cceil, fabs as Cabs
from libc.math cimport floor as Cfloor, round as Cround, log2 as Clog2
from libc.math cimport cos as Ccos, acos as Cacos, sin as Csin, asin as Casin
from libc.math cimport atan2 as Catan2, pi as Cpi
from libc.math cimport NAN as Cnan
from libc.math cimport isnan as Cisnan
from cpython.array cimport array, clone
from cython.parallel import prange
from cython.parallel cimport parallel
from libc.stdlib cimport malloc, free, realloc

from _basic_geom_tools cimport _VSMALL, _SMALL
from _basic_geom_tools cimport is_point_in_path_vec
from _raytracing_tools cimport comp_bbox_poly_tor
from _raytracing_tools cimport comp_bbox_poly_tor_lim
from _raytracing_tools cimport raytracing_inout_struct_lin
from _raytracing_tools cimport raytracing_inout_struct_tor
from _raytracing_tools cimport raytracing_minmax_struct_lin
from _raytracing_tools cimport raytracing_minmax_struct_tor
from _sampling_tools cimport discretize_line1d_core
from _sampling_tools cimport discretize_ves_poly


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
           "comp_dist_los_circle",
           "comp_dist_los_circle_vec",
           "comp_dist_los_vpoly",
           "comp_dist_los_vpoly_vec",
           "is_close_los_vpoly_vec",
           "is_close_los_circle",
           "is_close_los_circle_vec",
           "which_los_closer_vpoly_vec",
           "which_vpoly_closer_los_vec",
           "LOS_sino_findRootkPMin_Tor",
           'Poly_isClockwise', 'Poly_Order', 'Poly_VolAngTor',
           'Sino_ImpactEnv', 'ConvertImpact_Theta2Xi',
           '_Ves_isInside',
           'discretize_line1d',
           'discretize_polygon', '_Ves_meshCross_FromInd',
           'discretize_vpoly',
           '_Ves_Vmesh_Tor_SubFromD_cython', '_Ves_Vmesh_Tor_SubFromInd_cython',
           '_Ves_Vmesh_Lin_SubFromD_cython', '_Ves_Vmesh_Lin_SubFromInd_cython',
           '_Ves_Smesh_Tor_SubFromD_cython', '_Ves_Smesh_Tor_SubFromInd_cython',
           '_Ves_Smesh_TorStruct_SubFromD_cython',
           '_Ves_Smesh_TorStruct_SubFromInd_cython',
           '_Ves_Smesh_Lin_SubFromD_cython',
           '_Ves_Smesh_Lin_SubFromInd_cython',
           'LOS_Calc_PInOut_VesStruct',
           "LOS_Calc_kMinkMax_VesStruct",
           'SLOW_LOS_Calc_PInOut_VesStruct',
           'LOS_isVis_PtFromPts_VesStruct',
           'check_ff', 'LOS_get_sample', 'LOS_calc_signal',
           'LOS_sino','integrate1d']


########################################################
########################################################
#       Coordinates handling
########################################################

def CoordShift(Pts, In='(X,Y,Z)', Out='(R,Z)', CrossRef=None):
    """ Check the shape of an array of points coordinates and/or converts from
    2D to 3D, 3D to 2D, cylindrical to cartesian...
    (CrossRef is an angle (Tor) or a distance (X for Lin))
    """
    assert all([type(ff) is str and ',' in ff for ff in [In,Out]]), (
        "Arg In and Out (coordinate format) must be comma-separated  !")
    assert type(Pts) is np.ndarray and Pts.ndim in [1,2] and \
        Pts.shape[0] in (2,3), ("Points must be a 1D or 2D np.ndarray "
                                "of 2 or 3 coordinates !")
    okTypes = [int,float,np.int64,np.float64]
    assert CrossRef is None or type(CrossRef) in okTypes, (
        "Arg CrossRef must be a float !")

    # Pre-format inputs
    In, Out = In.lower(), Out.lower()

    # Get order
    Ins = In.replace('(','').replace(')','').split(',')
    Outs = Out.replace('(','').replace(')','').split(',')
    # TODO: @DV > it looks to me that (x, r, phi) could be a valid in/out-put
    # >>>>> ajouter assert pour le moment
    assert all([ss in ['x','y','z','r','phi'] for ss in Ins]), "Non-valid In!"
    assert all([ss in ['x','y','z','r','phi'] for ss in Outs]), "Non-valid Out!"
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
                # TODO : @DV > why ? no need of transform
                # >>> ts les angles entre [-pi, pi] -> ajouter un if ?
                pts.append(np.arctan2(np.sin(Pts[Ins.index(ii),:]),
                                      np.cos(Pts[Ins.index(ii),:])))
            else:
                pts.append(Pts[Ins.index(ii),:])
    elif InT=='cart':
        pts = []
        for ii in Outs:
            if ii=='r':
                assert all([ss in Ins for ss in ['x','y']])
                pts.append(np.hypot(Pts[Ins.index('x'),:],
                                    Pts[Ins.index('y'),:]))
            elif ii=='z':
                assert 'z' in Ins
                pts.append(Pts[Ins.index('z'),:])
            elif ii=='phi':
                if all([ss in Ins for ss in ['x','y']]):
                    pts.append(np.arctan2(Pts[Ins.index('y'),:],
                                          Pts[Ins.index('x'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no phi value available !")
                # TODO: @VZ > else... ? if Outs = (x, r, phi) ?
    else:
        pts = []
        for ii in Outs:
            if ii=='x':
                if all([ss in Ins for ss in ['r','phi']]) :
                    # TODO : @VZ : and CrossRef is None ?
                    # >>> ajouter un warning si crossref a ete defini
                    pts.append(Pts[Ins.index('r'),:] *
                               np.cos(Pts[Ins.index('phi'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no x value available !")
            elif ii=='y':
                assert all([ss in Ins for ss in ['r','phi']])
                pts.append(Pts[Ins.index('r'),:] *
                           np.sin(Pts[Ins.index('phi'),:]))
            elif ii=='z':
                assert 'z' in Ins
                pts.append(Pts[Ins.index('z'),:])
                # TODO : else....?

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
def Poly_isClockwise(np.ndarray[double,ndim=2] Poly):
    """ Assuming 2D closed Poly !
    TODO @LM :
    http://www.faqs.org/faqs/graphics/algorithms-faq/
    A slightly faster method is based on the observation that it isn't
    necessary to compute the area.  Find the lowest vertex (or, if
    there is more than one vertex with the same lowest coordinate,
    the rightmost of those vertices) and then take the cross product
    of the edges fore and aft of it.  Both methods are O(n) for n vertices,
    but it does seem a waste to add up the total area when a single cross
    product (of just the right edges) suffices.  Code for this is
    available at ftp://cs.smith.edu/pub/code/polyorient.C (2K).
    """
    cdef int ii, NP=Poly.shape[1]
    cdef double Sum=0.
    for ii in range(0,NP-1):
        # Slightly faster solution: (to check)  and try above solution ?
        # TODO : @LM > Test on of this sols:
        # Sol 1 (unit test time = 2.7)
        # Sum += (Poly[0,ii+1]-Poly[0,ii])*(Poly[1,ii+1]+Poly[1,ii])
        # Sol 2 (unit test time = 1.9)
        #     p1 = [0, 0]
        # p2 = [1, 0]
        # p3 = [.5, .5]
        # p4 = [1, 1]
        # p5 = [0,1]
        # p6 = [0,0]
        # points = [p1, p2, p3, p4, p5, p6]
        # idmin = points.index(min(points)) #0.99
        # idm1 = idmin - 1
        # idp1 = idmin + 1 % 7
        # res = points[idm1][0] * (points[idmin][1] - points[idp1][1]) + \
        #   points[idmin][0] * (points[idp1][1] - points[idm1][1]) + \
        #   points[idp1][0] * (points[idm1][1] - points[idmin][1])
        # Sol DV (unit test time = 2.9)
        Sum += Poly[0,ii]*Poly[1,ii+1]-Poly[0,ii+1]*Poly[1,ii]
    return Sum < 0.


def Poly_Order(np.ndarray[double,ndim=2] Poly, str order='C', Clock=False,
               close=True, str layout='(cc,N)',
               str layout_in=None, Test=True):
    """
    Return a polygon Poly as a np.ndarray formatted according to parameters

    Parameters
    ----------
        Poly    np.ndarray or list    Input polygon under from of (cc,N) or
                or tuple              (N,cc) np.ndarray (where cc = 2 or 3, the
                                      number of coordinates and N points), or
                                      list or tuple of vertices
        order   str                   Flag indicating whether the output
                                      np.ndarray shall be C-contiguous ('C') or
                                      Fortran-contiguous ('F')
        Clock   bool                  For 2-dimensional arrays only, flag indi-
                                      cating whether the output array shall
                                      represent a clockwise polygon (True) or
                                      anti-clockwise (False), or should be left
                                      unchanged (None)
        close   bool                  For 2-dimensional arrays only, flag indi-
                                      cating whether the output array shall be
                                      closed (True, ie: last point==first point)
                                      or not closed (False)
        layout  str                   Flag indicating whether the output
                                      np.ndarray shall be of shape '(cc,N)'
                                      or '(N,cc)'
        Test    bool                  Flag indicating whether the inputs should
                                      be tested for conformity, default: True

    Returns
    -------
        poly    np.ndarray            Output formatted polygon
    """
    if Test:
        assert (2 in np.shape(Poly) or 3 in np.shape(Poly)), \
          "Arg Poly must contain the 2D or 3D coordinates of at least 3 points!"
        assert max(np.shape(Poly))>=3, ("Arg Poly must contain the 2D or 3D",
                                        " coordinates of at least 3 points!")
        assert order.lower() in ['c','f'], "Arg order must be in ['c','f']!"
        assert type(Clock) is bool, "Arg Clock must be a bool!"
        assert type(close) is bool, "Arg close must be a bool!"
        assert layout.lower() in ['(cc,n)','(n,cc)'], \
          "Arg layout must be in ['(cc,n)','(n,cc)']!"
        assert layout_in is None or layout_in.lower() in ['(cc,n)','(n,cc)'],\
          "Arg layout_in must be None or in ['(cc,n)','(n,cc)']!"

    if np.shape(Poly)==(3,3):
        assert not layout_in is None, \
          ("Could not resolve the input layout of Poly because shape==(3,3)",
           " Please specify if input is in '(cc,n)' or '(n,cc)' format!")
        poly = np.array(Poly).T if layout_in.lower()=='(n,cc)' \
           else np.array(Poly)
    else:
        poly = np.array(Poly).T if min(np.shape(Poly))==Poly.shape[1]\
           else np.array(Poly)
    if not np.allclose(poly[:,0],poly[:,-1], atol=_VSMALL):
        poly = np.concatenate((poly,poly[:,0:1]),axis=1)
    if poly.shape[0]==2 and not Clock is None:
        if not Clock==Poly_isClockwise(poly):
            poly = poly[:,::-1]
    if not close:
        poly = poly[:,:-1]
    if layout.lower()=='(n,cc)':
        poly = poly.T
        # TODO : @LM @DV > seems strange to me that we order all polys
        # in order "(cc,n)" and just last minute we look at what's actually
        # asked
        # >> ok
    poly = np.ascontiguousarray(poly) if order.lower()=='c' \
           else np.asfortranarray(poly)
    return poly


def Poly_VolAngTor(np.ndarray[double,ndim=2,mode='c'] Poly):
    # TODO : @LM > to check the formulas can they be optimized ?
    cdef np.ndarray[double,ndim=1] Ri0 = Poly[0,:-1], Ri1 = Poly[0,1:]
    cdef np.ndarray[double,ndim=1] Zi0 = Poly[1,:-1], Zi1 = Poly[1,1:]
    cdef double V   =  np.sum((Ri0*Zi1 - Zi0*Ri1) * (Ri0+Ri1)) / 6.
    cdef double BV0 =  np.sum(0.5 * (Ri0*Zi1 - Zi0*Ri1) *
                              (Ri1**2 + Ri1*Ri0 + Ri0**2)) / (6.*V)
    cdef double BV1 = -np.sum((Ri1**2*Zi0*(2.*Zi1+Zi0) +
                               2.*Ri0*Ri1*(Zi0**2-Zi1**2) -
                               Ri0**2*Zi1*(Zi1+2.*Zi0))/4.) / (6.*V)
    return V, np.array([BV0,BV1])



"""
###############################################################################
###############################################################################
                    Sinogram specific
###############################################################################
"""

def Sino_ImpactEnv(np.ndarray[double,ndim=1] RZ,
                   np.ndarray[double,ndim=2] Poly, int NP=50, Test=True):
    """ Computes impact parameters of a Tor enveloppe
    (a `Tor` is a closed 2D polygon)

    D. VEZINET, Aug. 2014
    Parameters
    ----------
    RZ :    np.ndarray
        (2,) array indicating the reference impact point
    Poly :  np.ndarray
        (2,N) array containing the coordinatesof a closed polygon
    NP :    int
        Number of indicating the number of points used for discretising theta between 0 and pi

    Returns
    -------
        theta
    """
    if Test:
        assert RZ.size==2, 'Arg RZ should be a (2,) np.ndarray!'
        assert Poly.shape[0]==2, 'Arg Poly should be a (2,N) np.ndarray!'
    cdef int NPoly = Poly.shape[1]
    # Theta sampling and unit vector
    theta = np.linspace(0.,np.pi,NP,endpoint=True)
    vect = np.array([np.cos(theta), np.sin(theta)])

    # Scalar product
    sca = np.sum(vect[:,:,np.newaxis]*(Poly-RZ[:,np.newaxis])[:,np.newaxis,:],axis=0)
    scamin = np.min(sca,axis=1)
    scamax = np.max(sca,axis=1)
    return theta, np.array([scamax, scamin])


# For sinograms
def ConvertImpact_Theta2Xi(theta, pP, pN, sort=True):
    if hasattr(theta,'__getitem__'):
        pP, pN, theta = np.asarray(pP), np.asarray(pN), np.asarray(theta)
        assert pP.shape==pN.shape==theta.shape, (
            "Args pP, pN and theta must have same shape!")
        pPbis, pNbis = np.copy(pP), np.copy(pN)
        xi = theta - np.pi/2.
        ind = xi < 0
        pPbis[ind], pNbis[ind], xi[ind] = -pN[ind], -pP[ind], xi[ind]+np.pi
        if sort:
            ind = np.argsort(xi)
            xi, pP, pN = xi[ind], pPbis[ind], pNbis[ind]
        return xi, pP, pN
    else:
        assert not (hasattr(pP,'__getitem__') or hasattr(pN,'__getitem__')), (
            "Args pP, pN and theta must have same shape!")
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

def _Ves_isInside(Pts, VPoly, Lim=None, nLim=None,
                  VType='Tor', In='(X,Y,Z)', Test=True):
    if Test:
        assert type(Pts) is np.ndarray and Pts.ndim in [1,2], "Arg Pts must be a 1D or 2D np.ndarray !"
        assert type(VPoly) is np.ndarray and VPoly.ndim==2 and VPoly.shape[0]==2, "Arg VPoly must be a (2,N) np.ndarray !"
        assert Lim is None or (hasattr(Lim,'__iter__') and len(Lim)==2) or (hasattr(Lim,'__iter__') and all([hasattr(ll,'__iter__') and len(ll)==2 for ll in Lim])), "Arg Lim must be a len()==2 iterable or a list of such !"
        assert type(VType) is str and VType.lower() in ['tor','lin'], "Arg VType must be a str in ['Tor','Lin'] !"
        assert type(nLim) in [int,np.int64] and nLim>=0

    path = Path(VPoly.T)
    if VType.lower()=='tor':
        if Lim is None or nLim==0:
            pts = CoordShift(Pts, In=In, Out='(R,Z)')
            # TODO : @LM > voir avec la fct matplotlib et est-ce que c'est possible de
            # recoder pour faire plus rapide
            ind = Path(VPoly.T).contains_points(pts.T, transform=None,
                                                radius=0.0)
        else:
            try:
                pts = CoordShift(Pts, In=In, Out='(R,Z,Phi)')
            except Exception as err:
                msg = str(err)
                msg += "\n    You may have specified points in (R,Z)"
                msg += "\n    But there are toroidally limited elements !"
                msg += "\n      (i.e.: element with self.nLim>0)"
                msg += "\n    These require to know the phi of points !"
                raise Exception(msg)

            ind0 = Path(VPoly.T).contains_points(pts[:2,:].T,
                                                 transform=None, radius=0.0)
            if nLim>1:
                ind = np.zeros((nLim,Pts.shape[1]),dtype=bool)
                for ii in range(0,len(Lim)):
                    lim = [Catan2(Csin(Lim[ii][0]),Ccos(Lim[ii][0])),
                           Catan2(Csin(Lim[ii][1]),Ccos(Lim[ii][1]))]
                    if lim[0]<lim[1]:
                        ind[ii,:] = (ind0
                                     & (pts[2,:]>=lim[0])
                                     & (pts[2,:]<=lim[1]))
                    else:
                        ind[ii,:] = (ind0
                                     & ((pts[2,:]>=lim[0])
                                        | (pts[2,:]<=lim[1])))
            else:
                Lim = [Catan2(Csin(Lim[0,0]),Ccos(Lim[0,0])),
                       Catan2(Csin(Lim[0,1]),Ccos(Lim[0,1]))]
                if Lim[0]<Lim[1]:
                    ind = ind0 & (pts[2,:]>=Lim[0]) & (pts[2,:]<=Lim[1])
                else :
                    ind = ind0 & ((pts[2,:]>=Lim[0]) | (pts[2,:]<=Lim[1]))
    else:
        pts = CoordShift(Pts, In=In, Out='(X,Y,Z)')
        ind0 = Path(VPoly.T).contains_points(pts[1:,:].T,
                                             transform=None, radius=0.0)
        if nLim>1:
            ind = np.zeros((nLim,Pts.shape[1]),dtype=bool)
            for ii in range(0,nLim):
                ind[ii,:] = (ind0
                             & (pts[0,:]>=Lim[ii][0])
                             & (pts[0,:]<=Lim[ii][1]))
        else:
            ind = ind0 & (pts[0,:]>=Lim[0,0]) & (pts[0,:]<=Lim[0,1])
    return ind


# ==============================================================================
#
#                                   LINEAR MESHING
#                          i.e. Discretizing horizontal lines
#
# ==============================================================================
def discretize_line1d(double[::1] LMinMax, double dstep,
                       DL=None, bint Lim=True,
                       str mode='abs', double margin=_VSMALL):
    """
    Discretize a 1D segment LMin-LMax. If `mode` is "abs" (absolute), then the
    segment will be discretized in cells each of size `dstep`. Else, if `mode`
    is "rel" (relative), the meshing step is relative to the segments norm (ie.
    the actual discretization step will be (LMax - LMin)/dstep).
    It is possible to only one to discretize the segment on a sub-domain. If so,
    the sub-domain limits are given in DL.
    Parameters
    ==========
    LMinMax : (2)-double array
        Gives the limits LMin and LMax of the segment. LMinMax = [LMin, LMax]
    dstep: double
        Step of discretization, can be absolute (default) or relative
    DL : (optional) (2)-double array
        Sub domain of discretization. If not None and if Lim, LMinMax = DL
        (can be only on one limit and can be bigger or smaller than original).
        Actual desired limits
    Lim : (optional) bool
        Indicated if the subdomain should be taken into account
    mode : (optional) string
        If `mode` is "abs" (absolute), then the
        segment will be discretized in cells each of size `dstep`. Else,
        if "rel" (relative), the meshing step is relative to the segments norm
        (the actual discretization step will be (LMax - LMin)/dstep).
    margin : (optional) double
        Margin value for cell length
    Returns
    =======
    ldiscret: double array
        array of the discretized coordinates on the segment of desired limits
    resolution: double
        step of discretization
    lindex: int array
        array of the indices corresponding to ldiscret with respects to the
        original segment LMinMax (if no DL, from 0 to N-1)
    N : int64
        Number of points on LMinMax segment
    """
    cdef long sz_ld
    cdef long[1] N
    cdef double[2] dl_array
    cdef double[1] resolution
    cdef double* ldiscret = NULL
    cdef long* lindex = NULL
    #.. preparing inputs........................................................
    if DL is None:
        dl_array[0] = Cnan
        dl_array[1] = Cnan
    else:
        if DL[0] is None:
            dl_array[0] = Cnan
        else:
            dl_array[0] = DL[0]
        if DL[1] is None:
            dl_array[1] = Cnan
        else:
            dl_array[1] = DL[1]
    #.. calling cython function.................................................
    sz_ld = discretize_line1d_core(LMinMax, dstep, dl_array, Lim, mode, margin,
                                    &ldiscret, resolution, &lindex, N)
    #.. converting and returning................................................
    return np.asarray(<double[:sz_ld]> ldiscret), resolution[0],\
        np.asarray(<long[:sz_ld]>lindex), N[0]


# ==============================================================================
#
#                                   2D MESHING
#                           i.e. Discretizing polygons
#
# ==============================================================================
def discretize_polygon(double[::1] LMinMax1, double[::1] LMinMax2,
                       double dstep1, double dstep2,
                       D1=None, D2=None, str mode='abs',
                       double[:,::1] VPoly=None,
                       double margin=_VSMALL):
    """
    Parameters
    ==========
    LMinMax : (2)-double array
        Gives the limits LMin and LMax of the segment. LMinMax = [LMin, LMax]
    dstep: double
        Step of discretization, can be absolute (default) or relative
    DL : (optional) (2)-double array
        Sub domain of discretization. If not None and if Lim, LMinMax = DL
        (can be only on one limit and can be bigger or smaller than original).
        Actual desired limits
    Lim : (optional) bool
        Indicated if the subdomain should be taken into account
    mode : (optional) string
        If `mode` is "abs" (absolute), then the
        segment will be discretized in cells each of size `dstep`. Else,
        if "rel" (relative), the meshing step is relative to the segments norm
        (the actual discretization step will be (LMax - LMin)/dstep).
    margin : (optional) double
        Margin value for cell length
    Returns
    =======
    ldiscret: double array
        array of the discretized coordinates on the segment of desired limits
    resolution: double
        step of discretization
    lindex: int array
        array of the indices corresponding to ldiscret with respects to the
        original segment LMinMax (if no DL, from 0 to N-1)
    N : int64
        Number of points on LMinMax segment
    """
    cdef int num_pts_vpoly
    cdef int ndisc
    cdef int tot_true
    cdef long nind1
    cdef long nind2
    cdef long[1] num_cells1
    cdef long[1] num_cells2
    cdef int[1] nL0_1
    cdef int[1] nL0_2
    cdef bint* are_in_poly = NULL
    cdef long* lindex1_arr = NULL
    cdef long* lindex2_arr = NULL
    cdef double* ldiscret1_arr = NULL
    cdef double* ldiscret2_arr = NULL
    cdef double* ldiscr_tmp
    cdef long* lindex_tmp
    cdef array ldiscr
    cdef array lresol
    cdef array lindex
    cdef double[2] dl1_array
    cdef double[2] dl2_array
    cdef double[2] resolutions
    # .. Treating subdomains and Limits ........................................
    if D1 is None:
        dl1_array[0] = Cnan
        dl1_array[1] = Cnan
    else:
        if D1[0] is None:
            dl1_array[0] = Cnan
        else:
            dl1_array[0] = D1[0]
        if D1[1] is None:
            dl1_array[1] = Cnan
        else:
            dl1_array[1] = D1[1]
    if D2 is None:
        dl2_array[0] = Cnan
        dl2_array[1] = Cnan
    else:
        if D2[0] is None:
            dl2_array[0] = Cnan
        else:
            dl2_array[0] = D2[0]
        if D2[1] is None:
            dl2_array[1] = Cnan
        else:
            dl2_array[1] = D2[1]
    # .. Discretizing on the first direction ...................................
    nind1 = discretize_line1d_core(LMinMax1, dstep1, dl1_array,
                                    True, mode, margin,
                                    &ldiscret1_arr, &resolutions[0],
                                    &lindex1_arr, num_cells1)
    # .. Discretizing on the second direction ..................................
    nind2 = discretize_line1d_core(LMinMax2, dstep2, dl2_array,
                                    True, mode, margin,
                                    &ldiscret2_arr, &resolutions[1],
                                    &lindex2_arr, num_cells2)
    #....
    if VPoly is not None:
        ndisc = nind1 * nind2
        ldiscr_tmp = <double *>malloc(ndisc * 2 * sizeof(double))
        lindex_tmp = <long *>malloc(ndisc * sizeof(long))
        for ii in range(0,nind2):
            for jj in range(0,nind1):
                nn = jj + nind1 * ii
                ldiscr_tmp[nn] = ldiscret1_arr[jj]
                ldiscr_tmp[ndisc + nn] = ldiscret2_arr[ii]
                lindex_tmp[nn] = lindex1_arr[jj] + nind1 * lindex2_arr[ii]
        num_pts_vpoly = VPoly.shape[1] - 1
        are_in_poly = <bint *>malloc(ndisc * sizeof(bint))
        tot_true = is_point_in_path_vec(num_pts_vpoly,
                                        &VPoly[0][0], &VPoly[1][0],
                                        ndisc,
                                        &ldiscr_tmp[0], &ldiscr_tmp[ndisc],
                                        are_in_poly)
        ldiscr = clone(array('d'), tot_true*2, True)
        lindex = clone(array('l'), tot_true, True)
        lresol = clone(array('d'), tot_true, True)
        jj = 0
        for ii in range(ndisc):
            if are_in_poly[ii]:
                lresol[jj] = resolutions[0] * resolutions[1]
                lindex[jj] = lindex_tmp[ii]
                ldiscr[jj] = ldiscr_tmp[ii]
                ldiscr[jj + tot_true] = ldiscr_tmp[ii + ndisc]
                jj = jj + 1
        return np.asarray(ldiscr).reshape(2,tot_true), np.asarray(lresol),\
          np.asarray(lindex), resolutions[0], resolutions[1]
    else:
        ndisc = nind1 * nind2
        ldiscr = clone(array('d'), ndisc*2, True)
        lindex = clone(array('l'), ndisc, True)
        lresol = clone(array('d'), ndisc, True)
        for ii in range(nind2):
            for jj in range(nind1):
                nn = jj + nind1 * ii
                ldiscr[nn] = ldiscret1_arr[jj]
                ldiscr[ndisc + nn] = ldiscret2_arr[ii]
                lindex[nn] = lindex1_arr[jj] + nind1 * lindex2_arr[ii]
                lresol[nn] = resolutions[0] * resolutions[1]
        return np.asarray(ldiscr).reshape(2,ndisc), np.asarray(lresol),\
          np.asarray(lindex), resolutions[0], resolutions[1]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_meshCross_FromInd(double[::1] MinMax1, double[::1] MinMax2, double d1,
                           double d2, long[::1] ind, str dSMode='abs',
                           double margin=_VSMALL):
    #cdef double[::1] X1, X2
    #cdef double dX1, dX2
    cdef double d1r, d2r
    #cdef long[::1] dummy
    cdef int N1, N2
    cdef int NP = ind.size
    cdef int ii, i1, i2
    cdef np.ndarray[double,ndim=2] Pts = np.empty((2,NP))
    cdef np.ndarray[double,ndim=1] dS
    cdef long[2] num_cells
    cdef double[2] resolution
    cdef double[2] dl_array
    cdef double* X1 = NULL
    cdef double* X2 = NULL
    cdef long* dummy = NULL
    #.. preparing inputs........................................................
    dl_array[0] = Cnan
    dl_array[1] = Cnan
    #.. calling cython function.................................................
    discretize_line1d_core(MinMax1, d1, dl_array, True, dSMode, margin,
                            &X1, &resolution[0], &dummy, &num_cells[0])
    discretize_line1d_core(MinMax2, d2, dl_array, True, dSMode, margin,
                            &X2, &resolution[1], &dummy, &num_cells[1])
    d1r = resolution[0]
    d2r = resolution[1]
    N1 = num_cells[0]
    N2 = num_cells[1]
    dS = d1r*d2r*np.ones((NP,))
    for ii in range(0,NP):
        i2 = ind[ii] // N1
        i1 = ind[ii]-i2*N1
        Pts[0,ii] = X1[i1]
        Pts[1,ii] = X2[i2]
    return Pts, dS, d1r, d2r


def discretize_vpoly(double[:,::1] VPoly, double dL,
                     str mode='abs', list D1=None, list D2=None,
                     double margin=_VSMALL, double DIn=0.,
                     double[:,::1] VIn=None):
    cdef int NP=VPoly.shape[1]
    cdef int[1] sz_vb, sz_ot
    cdef double* XCross = NULL
    cdef double* YCross = NULL
    cdef double* XPolybis = NULL
    cdef double* YPolybis = NULL
    cdef double* Rref = NULL
    cdef double* resolution = NULL
    cdef long* ind = NULL
    cdef long* numcells = NULL
    cdef np.ndarray[double,ndim=2] PtsCross, VPolybis
    cdef np.ndarray[double,ndim=1] Rref_arr, resol
    cdef np.ndarray[long,ndim=1] ind_arr, N_arr
    cdef np.ndarray[long,ndim=1] ind_in

    assert (not (DIn == 0.) or VIn is not None)

    discretize_ves_poly(VPoly, dL, mode, margin, DIn, VIn,
                        &XCross, &YCross, &resolution,
                        &ind, &numcells, &Rref, &XPolybis, &YPolybis,
                        sz_vb, sz_ot, NP)
    PtsCross = np.asarray([<double[:sz_ot[0]]> XCross,
                           <double[:sz_ot[0]]> YCross])
    VPolybis = np.asarray([<double[:sz_vb[0]]> XPolybis,
                           <double[:sz_vb[0]]> YPolybis])
    resol = np.asarray(<double[:sz_ot[0]]> resolution)
    Rref_arr = np.asarray(<double[:sz_ot[0]]> Rref)
    ind_arr = np.asarray(<long[:sz_ot[0]]> ind)
    N_arr = np.asarray(<long[:NP-1]> numcells)
    if D1 is not None:
        indin = (PtsCross[0,:]>=D1[0]) & (PtsCross[0,:]<=D1[1])
        PtsCross = PtsCross[:,indin]
        resol = resol[indin]
        ind_arr = ind_arr[indin]
    if D2 is not None:
        indin = (PtsCross[1,:]>=D2[0]) & (PtsCross[1,:]<=D2[1])
        PtsCross = PtsCross[:,indin]
        resol = resol[indin]
        ind_arr = ind_arr[indin]
    return PtsCross, resol, ind_arr, N_arr, Rref_arr, VPolybis





########################################################
########################################################
#       Meshing - Volume - Tor
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromD_cython(double dR, double dZ, double dRPhi,
                                   double[::1] RMinMax, double[::1] ZMinMax,
                                   double[::1] DR=None, double[::1] DZ=None,
                                   DPhi=None, VPoly=None,
                                   str Out='(X,Y,Z)', double margin=_VSMALL):
    """Return the desired submesh indicated by the limits (DR,DZ,DPhi),
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] R0, R, Z, dRPhir, dPhir, NRPhi
    #cdef double[::1] dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj, nPhi0, nPhi1, zz
    cdef int NP, NRPhi_int, Rratio
    cdef np.ndarray[double,ndim=2] Pts, indI
    cdef np.ndarray[double,ndim=1] iii, dV, ind

    # Get the actual R and Z resolutions and mesh elements
    R0, dRr0, indR0, NR0 = discretize_line1d(RMinMax, dR, None,
                                              Lim=True, margin=margin)
    R, dRr, indR, NR = discretize_line1d(RMinMax, dR, DR, Lim=True,
                                          margin=margin)
    Z, dZr, indZ, NZ = discretize_line1d(ZMinMax, dZ, DZ, Lim=True,
                                          margin=margin)
    Rn = len(R)
    Zn = len(Z)

    # Get the limits if any (and make sure to replace them in the proper
    # quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = -Cpi, Cpi
    else:
        DPhi0 = Catan2(Csin(DPhi[0]), Ccos(DPhi[0]))
        DPhi1 = Catan2(Csin(DPhi[1]), Ccos(DPhi[1]))

    dRPhir, dPhir = np.empty((Rn,)), np.empty((Rn,))
    Phin = np.empty((Rn,),dtype=int)
    NRPhi = np.empty((Rn,))
    NRPhi0 = np.zeros((Rn,),dtype=int)
    nRPhi0, indR0ii = 0, 0
    NP, NPhimax = 0, 0
    Rratio = int(Cceil(R[Rn-1]/R[0]))

    for ii in range(0,Rn):
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
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
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        abs0 = Cabs(DPhi0+Cpi)
        if abs0-dPhir[ii]*Cfloor(abs0/dPhir[ii]) < margin*dPhir[ii]:
            nPhi0 = int(Cround((DPhi0+Cpi)/dPhir[ii]))
        else:
            nPhi0 = int(Cfloor((DPhi0+Cpi)/dPhir[ii]))
        abs1 = Cabs(DPhi1+Cpi)
        if abs1-dPhir[ii]*Cfloor(abs1/dPhir[ii]) < margin*dPhir[ii]:
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
            # To make sure the indices are in increasing order
            iii = np.sort(indI[ii,~np.isnan(indI[ii,:])])
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
            indin = Path(VPoly.T).contains_points(np.array([R,Pts[2,:]]).T,
                                                  transform=None, radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(R)
        else:
            indin = Path(VPoly.T).contains_points(Pts[:-1,:].T, transform=None,
                                                  radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(Pts[0,:])
        if not np.all(Ru==R):
            dRPhir = np.array([dRPhir[ii] for ii in range(0,len(R)) \
                               if R[ii] in Ru])
    return Pts, dV, ind.astype(int), dRr, dZr, np.asarray(dRPhir)




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromInd_cython(double dR, double dZ, double dRPhi,
                                     double[::1] RMinMax, double[::1] ZMinMax,
                                     long[::1] ind, str Out='(X,Y,Z)',
                                     double margin=_VSMALL):
    """ Return the desired submesh indicated by the (numerical) indices,
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] R, Z, dRPhirRef, dPhir, Ru, dRPhir
    cdef double dRr, dZr, phi
    cdef long[::1] indR, indZ, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiR, iiZ, iiphi
    cdef double[:,::1] Phi
    cdef np.ndarray[double,ndim=2] Pts=np.empty((3,NP))
    cdef np.ndarray[double,ndim=1] dV=np.empty((NP,))

    # Get the actual R and Z resolutions and mesh elements
    R, dRr, indR, NR = discretize_line1d(RMinMax, dR, None, Lim=True,
                                                margin=margin)
    Z, dZr, indZ, NZ = discretize_line1d(ZMinMax, dZ, None, Lim=True,
                                                margin=margin)
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
                                   double[::1] XMinMax, double[::1] YMinMax,
                                   double[::1] ZMinMax,
                                   double[::1] DX=None,
                                   double[::1] DY=None,
                                   double[::1] DZ=None, VPoly=None,
                                   double margin=_VSMALL):
    """ Return the desired submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dY,dZ)
    """
    cdef double[::1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef np.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef np.ndarray[double,ndim=2] Pts
    cdef np.ndarray[long,ndim=1] ind

    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, indX, NX = discretize_line1d(XMinMax, dX, DX, Lim=True,
                                                margin=margin)
    Y, dYr, indY, NY = discretize_line1d(YMinMax, dY, DY, Lim=True,
                                                margin=margin)
    Z, dZr, indZ, NZ = discretize_line1d(ZMinMax, dZ, DZ, Lim=True,
                                                margin=margin)
    Xn, Yn, Zn = len(X), len(Y), len(Z)

    Pts = np.array([np.tile(X,(Yn*Zn,1)).flatten(),
                    np.tile(np.repeat(Y,Xn),(Zn,1)).flatten(),
                    np.repeat(Z,Xn*Yn)])
    ind = np.repeat(NX*NY*indZ,Xn*Yn) + \
      np.tile(np.repeat(NX*indY,Xn),(Zn,1)).flatten() + \
      np.tile(indX,(Yn*Zn,1)).flatten()
    dV = dXr*dYr*dZr

    if VPoly is not None:
        indin = Path(VPoly.T).contains_points(Pts[1:,:].T, transform=None,
                                              radius=0.0)
        Pts, ind = Pts[:,indin], ind[indin]

    return Pts, dV, ind.astype(int), dXr, dYr, dZr


def _Ves_Vmesh_Lin_SubFromInd_cython(double dX, double dY, double dZ,
                                     double[::1] XMinMax, double[::1] YMinMax,
                                     double[::1] ZMinMax,
                                     np.ndarray[long,ndim=1] ind,
                                     double margin=_VSMALL):
    """ Return the desired submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dY,dZ)
    """

    cdef np.ndarray[double,ndim=1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef long[::1] bla
    cdef np.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef np.ndarray[double,ndim=2] Pts

    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, bla, NX = discretize_line1d(XMinMax, dX, None, Lim=True,
                                               margin=margin)
    Y, dYr, bla, NY = discretize_line1d(YMinMax, dY, None, Lim=True,
                                               margin=margin)
    Z, dZr, bla, NZ = discretize_line1d(ZMinMax, dZ, None, Lim=True,
                                               margin=margin)

    indZ = ind // (NX*NY)
    indY = (ind - NX*NY*indZ) // NX
    indX = ind - NX*NY*indZ - NX*indY
    Pts = np.array([X[indX.astype(int)],
                    Y[indY.astype(int)],
                    Z[indZ.astype(int)]])
    dV = dXr*dYr*dZr

    return Pts, dV, dXr, dYr, dZr



########################################################
########################################################
#       Meshing - Surface - Tor
########################################################

def _getBoundsInter2AngSeg(bool Full, double Phi0, double Phi1,
                           double DPhi0, double DPhi1):
    """ Return Inter=True if an intersection exist (all angles in radians
    in [-pi;pi])

    If Inter, return Bounds, a list of tuples indicating the segments defining
    the intersection, with
    The intervals are ordered from lowest index to highest index (with respect
    to [Phi0,Phi1])
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
                                   str Out='(X,Y,Z)', double margin=_VSMALL):
    """ Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi)
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] R, Z, dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1, DDPhi, DPhiMinMax
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0, Indin
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj0=0, jj, nPhi0, nPhi1
    cdef int zz, NP, NRPhi_int, Rratio, Ln
    cdef np.ndarray[double,ndim=2] Pts, indI, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] R0, dS, ind, dLr, Rref, dRPhir, iii
    cdef np.ndarray[long,ndim=1] indL, NL, indok

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])),
                     Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))]
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] \
          else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper
    # quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else Catan2(Csin(DPhi[0]),
                                                            Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else Catan2(Csin(DPhi[1]),
                                                            Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0],
                                                  PhiMinMax[1], DPhi0, DPhi1)

    if Inter:

        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += 2.*Cpi
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += 2.*Cpi

        # Get the actual R and Z resolutions and mesh elements
        PtsCross, dLr, indL, \
          NL, Rref, VPbis = discretize_vpoly(VPoly, dL, D1=None, D2=None,
                                             margin=margin, DIn=DIn, VIn=VIn)
        R0 = np.copy(Rref)
        NR0 = R0.size
        indin = np.ones((PtsCross.shape[1],),dtype=bool)
        if DR is not None:
            indin = indin & (R0>=DR[0]) & (R0<=DR[1])
        if DZ is not None:
            indin = indin & (PtsCross[1,:]>=DZ[0]) & (PtsCross[1,:]<=DZ[1])
        PtsCross, dLr, indL, Rref = PtsCross[:,indin], dLr[indin], \
          indL[indin], Rref[indin]
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
            # Get the actual RPhi resolution and Phi mesh elements
            # (! depends on R!)
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
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
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
                # Some rare cases with doubles have to be eliminated:
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])])
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
        Pts, dS, ind, NL, Rref, dRPhir, nRPhi0 = np.ones((3,0)), np.ones((0,)),\
          np.ones((0,)), np.nan*np.ones((VPoly.shape[1]-1,)),\
          np.ones((0,)), np.ones((0,)), 0
    return Pts, dS, ind.astype(int), NL, dLr, Rref, dRPhir, nRPhi0, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Tor_SubFromInd_cython(double dL, double dRPhi,
                                     double[:,::1] VPoly, long[::1] ind,
                                     double DIn=0., VIn=None, PhiMinMax=None,
                                     str Out='(X,Y,Z)', double margin=_VSMALL):
    """ Return the desired submesh indicated by the (numerical) indices,
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] dRPhirRef, dPhir
    cdef long[::1] indL, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiL, iiphi, Ln, nn=0, kk=0, nRPhi0
    cdef double[:,::1] Phi
    cdef np.ndarray[double,ndim=2] Pts=np.empty((3,NP)), indI, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] R0, dS=np.empty((NP,)), dLr, dRPhir, Rref
    cdef np.ndarray[long,ndim=1] NL

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]), Ccos(PhiMinMax[0])),
                     Catan2(Csin(PhiMinMax[1]), Ccos(PhiMinMax[1]))]
        if PhiMinMax[1]>=PhiMinMax[0]:
            DPhiMinMax = PhiMinMax[1]-PhiMinMax[0]
        else:
            DPhiMinMax = 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]


    # Get the actual R and Z resolutions and mesh elements
    PtsCross, dLrRef, indL,\
      NL, RrefRef, VPbis = discretize_vpoly(VPoly, dL, D1=None, D2=None,
                                            margin=margin, DIn=DIn, VIn=VIn)
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
    return Pts, dS, NL, dLr[dLr>-0.5], Rref[Rref>-0.5], \
      dRPhir[dRPhir>-0.5], <long>nRPhi0, VPbis



########################################################
########################################################
#       Meshing - Surface - TorStruct
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromD_cython(double[::1] PhiMinMax, double dL,
                                         double dRPhi,
                                         double[:,::1] VPoly,
                                         double[::1] DR=None,
                                         double[::1] DZ=None,
                                         double[::1] DPhi=None,
                                         double DIn=0., VIn=None,
                                         str Out='(X,Y,Z)',
                                         double margin=_VSMALL):
    """Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi),
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double Dphi, dR0r=0., dZ0r=0.
    cdef int NR0=0, NZ0=0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),
                                                  Ccos(PhiMinMax[0])),
                                           Catan2(Csin(PhiMinMax[1]),
                                                  Ccos(PhiMinMax[1]))])
    cdef np.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef np.ndarray[long,ndim=1] indR0, indZ0, iind, iindF, indM, NL, ind
    cdef np.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[]

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = np.array([-Cpi,Cpi])
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])),
                              Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0]\
          else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any and make sure to replace them in the proper quadrant
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None \
          else Catan2(Csin(DPhi[0]),Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None \
          else Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0],
                                                  PhiMinMax[1], DPhi0, DPhi1)

    if Inter:
        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += 2.*Cpi
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += 2.*Cpi

        # Required distance effective at max R
        Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0.

        # Get the mesh for the faces
        if any(Faces) :
            R0, dR0r, indR0,\
              NR0 = discretize_line1d(np.array([np.min(VPoly[0,:]),
                                                       np.max(VPoly[0,:])]),
                                             dL, DL=DR, Lim=True, margin=margin)
            Z0, dZ0r, indZ0,\
              NZ0 = discretize_line1d(np.array([np.min(VPoly[1,:]),
                                                       np.max(VPoly[1,:])]),
                                             dL, DL=DZ, Lim=True, margin=margin)
            R0n, Z0n = len(R0), len(Z0)
            ptsrz = np.array([np.tile(R0,Z0n),np.repeat(Z0,R0n)])
            iind = NR0*np.repeat(indZ0,R0n) + np.tile(indR0,Z0n)
            indin = Path(VPoly.T).contains_points(ptsrz.T, transform=None,
                                                  radius=0.0)
            if np.any(indin):
                ptsrz = ptsrz[:,indin] if indin.sum()>1 \
                  else ptsrz[:,indin].reshape((2,1))
                iindF = iind[indin]
                dsF = dR0r*dZ0r*np.ones((indin.sum(),))

        # First face
        if Faces[0]:
            if Out.lower()=='(x,y,z)':
                pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[0]+Dphi),
                                ptsrz[0,:]*Csin(phiMinMax[0]+Dphi),
                                ptsrz[1,:]])
            else:
                pts = np.array([ptsrz[0,:],
                                ptsrz[1,:],
                                (phiMinMax[0]+Dphi)*np.ones((indin.sum(),))])
            LPts.append( pts )
            Lind.append( iindF )
            LdS.append( dsF )

        # Main body
        PtsM, dSM,\
          indM, NL,\
          dLr, Rref,\
          dRPhir, nRPhi0,\
          VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                 DR=DR, DZ=DZ,
                                                 DPhi=[DPhi0,DPhi1],
                                                 DIn=DIn, VIn=VIn,
                                                 PhiMinMax=phiMinMax,
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
                pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[1]-Dphi),
                                ptsrz[0,:]*Csin(phiMinMax[1]-Dphi),
                                ptsrz[1,:]])
            else:
                pts = np.array([ptsrz[0,:],
                                ptsrz[1,:],
                                (phiMinMax[1]-Dphi)*np.ones((indin.sum(),))])
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
        Pts, dS, ind, NL, Rref = np.ones((3,0)), np.ones((0,)),\
          np.ones((0,),dtype=int), np.ones((0,),dtype=int),\
          np.nan*np.ones((VPoly.shape[1]-1,))
        dLr, dR0r, dZ0r, dRPhir, VPbis = np.ones((0,)), 0., 0.,\
          np.ones((0,)), np.asarray(VPoly)

    return Pts, dS, ind, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromInd_cython(double[::1] PhiMinMax, double dL,
                                           double dRPhi, double[:,::1] VPoly,
                                           np.ndarray[long,ndim=1] ind,
                                           double DIn=0., VIn=None,
                                           str Out='(X,Y,Z)',
                                           double margin=_VSMALL):
    """ Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi)
    for the desired resolution (dR,dZ,dRphi) """
    cdef double Dphi, dR0r, dZ0r
    cdef int NR0, NZ0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),
                                                  Ccos(PhiMinMax[0])),
                                           Catan2(Csin(PhiMinMax[1]),
                                                  Ccos(PhiMinMax[1]))])
    cdef np.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef np.ndarray[long,ndim=1] bla, indR0, indZ0, iind, iindF, indM, NL
    cdef np.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[]

    # Pre-format input
    # Required distance effective at max R
    Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0.

    # Get the basic meshes for the faces
    R0, dR0r, bla, NR0 = discretize_line1d(np.array([np.min(VPoly[0,:]),
                                                            np.max(VPoly[0,:])]),
                                                  dL, DL=None, Lim=True,
                                                  margin=margin)
    Z0, dZ0r, bla, NZ0 = discretize_line1d(np.array([np.min(VPoly[1,:]),
                                                            np.max(VPoly[1,:])]),
                                                  dL, DL=None, Lim=True,
                                                  margin=margin)

    PtsM, dSM, indM,\
      NL, dLr, Rref,\
      dRPhir, nRPhi0, VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                             DR=None, DZ=None,
                                                             DPhi=None,
                                                             DIn=DIn, VIn=VIn,
                                                             PhiMinMax=phiMinMax,
                                                             Out=Out, margin=margin)
    # First face
    ii = (ind<NR0*NZ0).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NR0
        indR0 = (ind[ii]-indZ0*NR0)
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[0]+Dphi),
                            R0[indR0]*Csin(phiMinMax[0]+Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0],
                            (phiMinMax[0]+Dphi)*np.ones((nii,))])
        pts = pts if nii>1 else pts.reshape((3,1))
        LPts.append( pts )
        LdS.append( dR0r*dZ0r*np.ones((nii,)) )

    # Main body
    ii = (ind>=NR0*NZ0) & (ind<NR0*NZ0+PtsM.shape[1])
    nii = len(ii)
    if nii>0:
        pts = PtsM[:,ind[ii]-NR0*NZ0] if nii>1\
          else PtsM[:,ind[ii]-NR0*NZ0].reshape((3,1))
        LPts.append( PtsM[:,ind[ii]-NR0*NZ0] )
        LdS.append( dSM[ind[ii]-NR0*NZ0] )

    # Second face
    ii = (ind >= NR0*NZ0+PtsM.shape[1] ).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-(NR0*NZ0+PtsM.shape[1])) // NR0
        indR0 = ind[ii]-(NR0*NZ0+PtsM.shape[1]) - indZ0*NR0
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[1]-Dphi),
                            R0[indR0]*Csin(phiMinMax[1]-Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0],
                            (phiMinMax[1]-Dphi)*np.ones((nii,))])
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
                                   double DIn=0., VIn=None,
                                   double margin=_VSMALL):
    """Return the desired surfacic submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dL) """
    cdef np.ndarray[double,ndim=1] X, Y0, Z0
    cdef double dXr, dY0r, dZ0r
    cdef int NY0, NZ0, Y0n, Z0n, NX, Xn, Ln, NR0, Inter=1
    cdef np.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] dS, dLr, Rref
    cdef np.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ind

    # Preformat
    # Adjust limits
    InterX, DX = _check_DLvsLMinMax(XMinMax,DX)
    InterY, DY = _check_DLvsLMinMax(np.array([np.min(VPoly[0,:]),
                                              np.max(VPoly[0,:])]), DY)
    InterZ, DZ = _check_DLvsLMinMax(np.array([np.min(VPoly[1,:]),
                                              np.max(VPoly[1,:])]), DZ)

    if InterX==1 and InterY==1 and InterZ==1:

        # Get the mesh for the faces
        Y0, dY0r,\
          indY0, NY0 = discretize_line1d(np.array([np.min(VPoly[0,:]),
                                                    np.max(VPoly[0,:])]),
                                          dL, DL=DY, Lim=True, margin=margin)
        Z0, dZ0r,\
          indZ0, NZ0 = discretize_line1d(np.array([np.min(VPoly[1,:]),
                                                    np.max(VPoly[1,:])]),
                                          dL, DL=DZ, Lim=True, margin=margin)
        Y0n, Z0n = len(Y0), len(Z0)

        # Get the actual R and Z resolutions and mesh elements
        X, dXr, indX, NX = discretize_line1d(XMinMax, dX,
                                              DL=DX,
                                              Lim=True, margin=margin)
        Xn = len(X)
        PtsCross, dLr, indL,\
          NL, Rref, VPbis = discretize_vpoly(VPoly, dL, D1=None, D2=None,
                                             margin=margin, DIn=DIn, VIn=VIn)
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
        PtsCross, dLr,\
          indL, Rref = PtsCross[:,indin], dLr[indin], indL[indin], Rref[indin]
        Ln = indin.sum()
        # Agregating
        Pts = np.array([np.repeat(X,Ln),
                        np.tile(PtsCross[0,:],Xn),
                        np.tile(PtsCross[1,:],Xn)])
        ind = NY0*NZ0 + np.repeat(indX*NR0,Ln) + np.tile(indL,Xn)
        dS = np.tile(dLr*dXr,Xn)
        if DX is None or DX[0] is None:
            pts = np.array([(XMinMax[0]+DIn)*np.ones((Y0n*Z0n,)),
                            np.tile(Y0,Z0n),
                            np.repeat(Z0,Y0n)])
            iind = NY0*np.repeat(indZ0,Y0n) + np.tile(indY0,Z0n)
            indin = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None,
                                                  radius=0.0)
            if np.any(indin):
                pts = pts[:,indin].reshape((3,1)) if indin.sum()==1\
                  else pts[:,indin]
                Pts = np.concatenate((pts,Pts),axis=1)
                ind = np.concatenate((iind[indin], ind))
                dS = np.concatenate((dY0r*dZ0r*np.ones((indin.sum(),)),dS))
        if DX is None or DX[1] is None:
            pts = np.array([(XMinMax[1]-DIn)*np.ones((Y0n*Z0n,)),
                            np.tile(Y0,Z0n),
                            np.repeat(Z0,Y0n)])
            iind = NY0*NZ0 + NX*NR0 + NY0*np.repeat(indZ0,Y0n) +\
              np.tile(indY0,Z0n)
            indin = Path(VPoly.T).contains_points(pts[1:,:].T,
                                                  transform=None,
                                                  radius=0.0)
            if np.any(indin):
                pts = pts[:,indin].reshape((3,1)) if indin.sum()==1\
                  else pts[:,indin]
                Pts = np.concatenate((Pts,pts),axis=1)
                ind = np.concatenate((ind,iind[indin]))
                dS = np.concatenate((dS,dY0r*dZ0r*np.ones((indin.sum(),))))

    else:
        Pts, dS, ind,\
          NL, dLr, Rref = np.ones((3,0)), np.ones((0,)),\
          np.ones((0,),dtype=int), np.ones((0,),dtype=int),\
          np.ones((0,)), np.ones((0,))
        dXr, dY0r, dZ0r, VPbis = 0., 0., 0., np.ones((3,0))

    return Pts, dS, ind, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Lin_SubFromInd_cython(double[::1] XMinMax, double dL, double dX,
                                     double[:,::1] VPoly, np.ndarray[long,ndim=1] ind,
                                     double DIn=0., VIn=None, double margin=_VSMALL):
    " Return the desired surfacic submesh indicated by ind, for the desired resolution (dX,dL) "
    cdef double dXr, dY0r, dZ0r
    cdef int NX, NY0, NZ0, Ln, NR0, nii
    cdef list LPts, LdS
    cdef np.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] X, Y0, Z0, dS, dLr, Rref
    cdef np.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ii

    # Get the mesh for the faces
    Y0, dY0r, bla, NY0 = discretize_line1d(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, bla, NZ0 = discretize_line1d(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=None, Lim=True, margin=margin)

    # Get the actual R and Z resolutions and mesh elements
    X, dXr, bla, NX = discretize_line1d(XMinMax, dX, DL=None, Lim=True, margin=margin)
    PtsCross, dLr, bla, NL, Rref, VPbis = discretize_vpoly(VPoly, dL,
                                                           D1=None, D2=None,
                                                           margin=margin,
                                                           DIn=DIn, VIn=VIn)
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


# =============================================================================
# = Set of functions for Ray-tracing
# =============================================================================

def LOS_Calc_PInOut_VesStruct(double[:, ::1] ray_orig,
                              double[:, ::1] ray_vdir,
                              double[:, ::1] ves_poly,
                              double[:, ::1] ves_norm,
                              long[::1] lstruct_nlim=None,
                              double[::1] ves_lims=None,
                              double[::1] lstruct_polyx=None,
                              double[::1] lstruct_polyy=None,
                              list lstruct_lims=None,
                              double[::1] lstruct_normx=None,
                              double[::1] lstruct_normy=None,
                              long[::1] lnvert=None,
                              int nstruct_tot=0,
                              int nstruct_lim=0,
                              double rmin=-1,
                              double eps_uz=_SMALL, double eps_a=_VSMALL,
                              double eps_vz=_VSMALL, double eps_b=_VSMALL,
                              double eps_plane=_VSMALL, str ves_type='Tor',
                              bint forbid=1, bint test=1, int num_threads=16):
    """
    Computes the entry and exit point of all provided LOS for the provided
    vessel polygon (toroidal or linear) with its associated structures.
    Return the normal vector at impact and the index of the impact segment

    Params
    ======
    ray_orig : (3, num_los) double array
       LOS origin points coordinates
    ray_vdir : (3, num_los) double array
       LOS normalized direction vector
    ves_poly : (2, num_vertex) double array
       Coordinates of the vertices of the Polygon defining the 2D poloidal
       cut of the Vessel
    ves_norm : (2, num_vertex-1) double array
       Normal vectors going "inwards" of the edges of the Polygon defined
       by ves_poly
    nstruct : int
       Total number of structures (counting each limited structure as one)
    ves_lims : array
       Contains the limits min and max of vessel
    lstruct_poly : list
       List of coordinates of the vertices of all structures on poloidal plane
    lstruct_lims : list
       List of limits of all structures
    lstruct_nlim : array of ints
       List of number of limits for all structures
    lstruct_norm : list
       List of coordinates of "inwards" normal vectors of the polygon of all
       the structures
    rmin : double
       Minimal radius of vessel to take into consideration
    eps_<val> : double
       Small value, acceptance of error
    vtype : string
       Type of vessel ("Tor" or "Lin")
    forbid : bool
       Should we forbid values behind vissible radius ? (see rmin)
    test : bool
       Should we run tests ?
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    Returns
    ======
    coeff_inter_in : (num_los) array
       scalars level of "in" intersection of the LOS (if k=0 at origin)
    coeff_inter_out : (num_los) array
       scalars level of "out" intersection of the LOS (if k=0 at origin)
    vperp_out : (3, num_los) array
       Coordinates of the normal vector of impact of the LOS (NaN if none)
    ind_inter_out : (3, num_los)
       Index of structure impacted by LOS: ind_inter_out[:,ind_los]=(i,j,k)
       where k is the index of edge impacted on the j-th sub structure of the
       structure number i. If the LOS impacted the vessel i=j=0
    """
    cdef int npts_poly = ves_norm.shape[1]
    cdef int num_los = ray_orig.shape[1]
    cdef int ind_struct = 0
    cdef int ii, jj, kk
    cdef int len_lim
    cdef int ind_min
    cdef int nvert
    cdef double Crit2_base = eps_uz * eps_uz /400.
    cdef double lim_min = 0.
    cdef double lim_max = 0.
    cdef double rmin2 = 0.
    cdef str error_message
    cdef bint forbidbis, forbid0
    cdef bint bool1, bool2
    cdef double *lbounds = <double *>malloc(nstruct_tot * 6 * sizeof(double))
    cdef double *langles = <double *>malloc(nstruct_tot * 2 * sizeof(double))
    cdef array vperp_out = clone(array('d'), num_los * 3, True)
    cdef array coeff_inter_in  = clone(array('d'), num_los, True)
    cdef array coeff_inter_out = clone(array('d'), num_los, True)
    cdef array ind_inter_out = clone(array('i'), num_los * 3, True)
    cdef int *llimits = NULL
    cdef long *lsz_lim = NULL
    cdef int[1] llim_ves
    cdef double[2] lbounds_ves
    cdef double[2] lim_ves

    # == Testing inputs ========================================================
    if test:
        error_message = "ray_orig and ray_vdir must have the same shape: "\
                        + "(3,) or (3,NL)!"
        assert tuple(ray_orig.shape) == tuple(ray_vdir.shape) and \
          ray_orig.shape[0] == 3, error_message
        error_message = "ves_poly and ves_norm must have the same shape (2,NS)!"
        assert ves_poly.shape[0] == 2 and ves_norm.shape[0] == 2 and \
            npts_poly == ves_poly.shape[1]-1, error_message
        bool1 = lstruct_lims is None or len(lstruct_normy) == len(lstruct_normy)
        bool2 = lstruct_normx is None or len(lstruct_polyx) == len(lstruct_polyy)
        error_message = "lstruct_poly, lstruct_lims, lstruct_norm must be None"\
                        + " or lists of same len!"
        assert bool1 and bool2, error_message
        error_message = "[eps_uz,eps_vz,eps_a,eps_b] must be floats < 1.e-4!"
        assert all([ee < 1.e-4 for ee in [eps_uz, eps_a,
                                          eps_vz, eps_b,
                                          eps_plane]]), error_message
        error_message = "ves_type must be a str in ['Tor','Lin']!"
        assert ves_type.lower() in ['tor', 'lin'], error_message
        error_message = "If you define structures you must define all the "\
                        + "structural variables: \n"\
                        + "    - lstruct_polyx, lstruct_polyy, lstruct_lims,\n"\
                        + "    - lstruct_nlim, nstruct_tot, nstruct_lim,\n"\
                        + "    - lnvert, lstruct_normx, lstruct_normy\n"
        bool1 = ((lstruct_polyx is not None)
                 or (lstruct_polyy is not None)
                 or (lstruct_normx is not None)
                 or (lstruct_normy is not None)
                 or (lstruct_nlim is not None)
                 or (lstruct_lims is not None)
                 or (lnvert is not None)
                 or (nstruct_tot > 0) or (nstruct_lim > 0))
        if bool1:
            try:
                bool1 = ((len(lstruct_polyx) > 0)
                         or (len(lstruct_polyy) > 0)
                         or (len(lstruct_normx) > 0)
                         or (len(lstruct_normy) > 0)
                         or (len(lstruct_nlim) > 0)
                         or (len(lstruct_lims) > 0)
                         or (len(lnvert) > 0)
                         or (nstruct_tot > 0)
                         or (nstruct_lim > 0))
                bool2 = ((len(lstruct_polyx) > 0)
                         and (len(lstruct_polyy) > 0)
                         and (len(lstruct_normx) > 0)
                         and (len(lstruct_normy) > 0)
                         and (len(lstruct_nlim) > 0)
                         and (len(lstruct_lims) > 0)
                         and (len(lnvert) > 0)
                         and (nstruct_tot > 0)
                         and (nstruct_lim > 0))
                assert (not bool1 or bool2), error_message
            except Exception:
                assert False, error_message
        else:
            bool2 = ((lstruct_polyx is not None)
                 and (lstruct_polyy is not None)
                 and (lstruct_normx is not None)
                 and (lstruct_normy is not None)
                 and (lstruct_nlim is not None)
                 and (lstruct_lims is not None)
                 and (lnvert is not None)
                 and (nstruct_tot > 0) and (nstruct_lim > 0))
            assert (not bool1 or bool2), error_message

    # ==========================================================================
    if ves_type.lower() == 'tor':
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None or np.size(ves_lims) == 0:
            are_limited = False
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
            llim_ves[0] = 1
        else:
            are_limited = True
            lbounds_ves[0] = Catan2(Csin(ves_lims[0]), Ccos(ves_lims[0]))
            lbounds_ves[1] = Catan2(Csin(ves_lims[1]), Ccos(ves_lims[1]))
            llim_ves[0] = 0
        # -- Toroidal case -----------------------------------------------------
        # rmin is necessary to avoid looking on the other side of the tokamak
        if rmin < 0.:
            rmin = 0.95*min(np.min(ves_poly[0, ...]),
                                np.min(np.hypot(ray_orig[0, ...],
                                                ray_orig[1, ...])))
        rmin2 = rmin*rmin
        # Variable to avoid looking "behind" blind spot of tore
        if forbid:
            forbid0, forbidbis = 1, 1
        else:
            forbid0, forbidbis = 0, 0

        # -- Computing intersection between LOS and Vessel ---------------------
        raytracing_inout_struct_tor(num_los, ray_vdir, ray_orig,
                                    coeff_inter_out, coeff_inter_in,
                                    vperp_out, lstruct_nlim, ind_inter_out,
                                    forbid0, forbidbis,
                                    rmin, rmin2, Crit2_base,
                                    npts_poly,  NULL, lbounds_ves,
                                    llim_ves, NULL, NULL,
                                    &ves_poly[0][0],
                                    &ves_poly[1][0],
                                    &ves_norm[0][0],
                                    &ves_norm[1][0],
                                    eps_uz, eps_vz, eps_a, eps_b, eps_plane,
                                    num_threads, False) # structure is in

        # -- Treating the structures (if any) ----------------------------------
        if nstruct_tot > 0:
            ind_struct = 0
            llimits = <int *>malloc(nstruct_tot * sizeof(int))
            lsz_lim = <long *>malloc(nstruct_lim * sizeof(long))
            for ii in range(nstruct_lim):
                # For fast accessing
                len_lim = lstruct_nlim[ii]
                # We get the limits if any
                if len_lim == 0:
                    lslim = [None]
                    lstruct_nlim[ii] = lstruct_nlim[ii] + 1
                elif len_lim == 1:
                    lslim = [[lstruct_lims[ii][0, 0], lstruct_lims[ii][0, 1]]]
                else:
                    lslim = lstruct_lims[ii]
                # We get the number of vertices and limits of the struct's poly
                if ii == 0:
                    lsz_lim[0] = 0
                    nvert = lnvert[0]
                    ind_min = 0
                else:
                    nvert = lnvert[ii] - lnvert[ii - 1]
                    lsz_lim[ii] = lstruct_nlim[ii-1] + lsz_lim[ii-1]
                    ind_min = lnvert[ii-1]
                # and loop over the limits (one continous structure)
                for jj in range(max(len_lim,1)):
                    # We compute the structure's bounding box:
                    if lslim[jj] is not None:
                        lim_ves[0] = lslim[jj][0]
                        lim_ves[1] = lslim[jj][1]
                        llimits[ind_struct] = 0 # False : struct is limited
                        lim_min = Catan2(Csin(lim_ves[0]), Ccos(lim_ves[0]))
                        lim_max = Catan2(Csin(lim_ves[1]), Ccos(lim_ves[1]))
                        comp_bbox_poly_tor_lim(nvert,
                                               &lstruct_polyx[ind_min],
                                               &lstruct_polyy[ind_min],
                                               &lbounds[ind_struct*6],
                                               lim_min, lim_max)
                    else:
                        llimits[ind_struct] = 1 # True : is continous
                        comp_bbox_poly_tor(nvert,
                                           &lstruct_polyx[ind_min],
                                           &lstruct_polyy[ind_min],
                                           &lbounds[ind_struct*6])
                        lim_min = 0.
                        lim_max = 0.
                    langles[ind_struct*2] = lim_min
                    langles[ind_struct*2 + 1] = lim_max
                    ind_struct = 1 + ind_struct
            # end loops over structures

            # -- Computing intersection between structures and LOS -------------
            raytracing_inout_struct_tor(num_los, ray_vdir, ray_orig,
                                        coeff_inter_out, coeff_inter_in,
                                        vperp_out, lstruct_nlim, ind_inter_out,
                                        forbid0, forbidbis,
                                        rmin, rmin2, Crit2_base,
                                        nstruct_lim,
                                        lbounds, langles, llimits,
                                        &lnvert[0], lsz_lim,
                                        &lstruct_polyx[0], &lstruct_polyy[0],
                                        &lstruct_normx[0], &lstruct_normy[0],
                                        eps_uz, eps_vz, eps_a, eps_b, eps_plane,
                                        num_threads,
                                        True) # the structure is "OUT"
            free(lsz_lim)
            free(llimits)
    else:
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None  or np.size(ves_lims) == 0:
            are_limited = False
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
        else:
            are_limited = True
            lbounds_ves[0] = ves_lims[0]
            lbounds_ves[1] = ves_lims[1]

        # -- Cylindrical case --------------------------------------------------
        raytracing_inout_struct_lin(num_los, ray_orig, ray_vdir, npts_poly,
                                    &ves_poly[0][0], &ves_poly[1][0],
                                    &ves_norm[0][0], &ves_norm[1][0],
                                    lbounds_ves[0], lbounds_ves[1],
                                    coeff_inter_in, coeff_inter_out,
                                    vperp_out, ind_inter_out, eps_plane,
                                    0, 0) # The vessel is strcuture 0,0

        # -- Treating the structures (if any) ----------------------------------
        if nstruct_tot > 0:
            ind_struct = 0
            for ii in range(nstruct_lim):
                # -- Analyzing the limits --------------------------------------
                len_lim = lstruct_nlim[ii]
                # We get the limits if any
                if len_lim == 0:
                    lslim = [None]
                    lstruct_nlim[ii] = lstruct_nlim[ii] + 1
                elif len_lim == 1:
                    lslim = [[lstruct_lims[ii][0, 0], lstruct_lims[ii][0, 1]]]
                else:
                    lslim = lstruct_lims[ii]
                if ii == 0:
                    nvert = lnvert[0]
                    ind_min = 0
                else:
                    nvert = lnvert[ii] - lnvert[ii - 1]
                    ind_min = lnvert[ii-1]
                # and loop over the limits (one continous structure)
                for jj in range(max(len_lim,1)):
                    if lslim[jj] is not None:
                        lbounds_ves[0] = lslim[jj][0]
                        lbounds_ves[1] = lslim[jj][1]
                    raytracing_inout_struct_lin(num_los, ray_orig, ray_vdir,
                                                nvert-1,
                                                &lstruct_polyx[ind_min],
                                                &lstruct_polyy[ind_min],
                                                &lstruct_normx[ind_min-ii],
                                                &lstruct_normy[ind_min-ii],
                                                lbounds_ves[0], lbounds_ves[1],
                                                coeff_inter_in, coeff_inter_out,
                                                vperp_out, ind_inter_out,
                                                eps_plane, ii+1, jj)

    free(lbounds)
    free(langles)

    return np.asarray(coeff_inter_in), np.asarray(coeff_inter_out),\
           np.transpose(np.asarray(vperp_out).reshape(num_los,3)),\
           np.transpose(np.asarray(ind_inter_out, dtype=int).reshape(num_los, 3))



# =============================================================================
# = Ray tracing when we only want kMin / kMax
# -   (useful when working with flux surfaces)
# =============================================================================
def LOS_Calc_kMinkMax_VesStruct(double[:, ::1] ray_orig,
                                double[:, ::1] ray_vdir,
                                double[:, :, ::1] ves_poly,
                                double[:, :, ::1] ves_norm,
                                int num_surf,
                                double[::1] ves_lims=None,
                                long[::1] lnvert=None,
                                double rmin=-1,
                                double eps_uz=_SMALL, double eps_a=_VSMALL,
                                double eps_vz=_VSMALL, double eps_b=_VSMALL,
                                double eps_plane=_VSMALL, str ves_type='Tor',
                                bint forbid=1, bint test=1, int num_threads=16):
    """
    Computes the entry and exit point of all provided LOS for the provided
    polygons (toroidal or linear)  of IN structures (non-solid, or `empty`
    inside for the LOS).
    Attention: the surfaces can be limited, but they all have to have the
    same limits defined by (ves_lims)
    Return the set of kmin / kmax for each In struct and for each LOS

    Params
    ======
    ray_orig : (3, num_los) double array
       LOS origin points coordinates
    ray_vdir : (3, num_los) double array
       LOS normalized direction vector
    num_surf : int
       number of surfaxes, aka 'in' structures or 'vessels'
    ves_poly : (num_surf, 2, num_vertex) double array
       Coordinates of the vertices of the Polygon defining the 2D poloidal
       cut of the `in` structures
    ves_norm : (num_surf, 2, num_vertex-1) double array
       Normal vectors going "inwards" of the edges of the Polygon defined
       by ves_poly
    ves_lims : array
       Contains the limits min and max of vessel
    rmin : double
       Minimal radius of vessel to take into consideration
    eps<val> : double
       Small value, acceptance of error
    vtype : string
       Type of vessel ("Tor" or "Lin")
    forbid : bool
       Should we forbid values behind vissible radius ? (see rmin)
    test : bool
       Should we run tests ?
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    Return
    ======
    coeff_inter_in : (num_surf, num_los) array
       scalars level of "in" intersection of the LOS (if k=0 at origin) for
       each surface
       [kmin(surf0, los0), kmin(surf0, los1), ..., kmin(surf1, los0),....]
    coeff_inter_out : (num_surf, num_los) array
       scalars level of "out" intersection of the LOS (if k=0 at origin) for
       each surface
       [kmax(surf0, los0), kmax(surf0, los1), ..., kmax(surf1, los0),....]
    """
    cdef int npts_poly
    cdef int num_los = ray_orig.shape[1]
    cdef int ind_struct = 0
    cdef int ind_surf
    cdef int len_lim
    cdef int ind_min
    cdef double Crit2_base = eps_uz * eps_uz /400.
    cdef double lim_min = 0.
    cdef double lim_max = 0.
    cdef double rmin2 = 0.
    cdef str error_message
    cdef bint forbidbis, forbid0
    cdef bint bool1, bool2
    cdef array coeff_inter_in  = clone(array('d'), num_los * num_surf, True)
    cdef array coeff_inter_out = clone(array('d'), num_los * num_surf, True)
    cdef int *llimits = NULL
    cdef long *lsz_lim = NULL
    cdef bint are_limited
    cdef double[2] lbounds_ves
    cdef double[2] lim_ves

    # == Testing inputs ========================================================
    if test:
        error_message = "ray_orig and ray_vdir must have the same shape: "\
                        + "(3,) or (3,NL)!"
        assert tuple(ray_orig.shape) == tuple(ray_vdir.shape) and \
          ray_orig.shape[0] == 3, error_message
        error_message = "[eps_uz,eps_vz,eps_a,eps_b] must be floats < 1.e-4!"
        assert all([ee < 1.e-4 for ee in [eps_uz, eps_a,
                                          eps_vz, eps_b,
                                          eps_plane]]), error_message
        error_message = "ves_type must be a str in ['Tor','Lin']!"
        assert ves_type.lower() in ['tor', 'lin'], error_message

    # ==========================================================================
    if ves_type.lower() == 'tor':
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None  or np.size(ves_lims) == 0:
            are_limited = False
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
        else:
            are_limited = True
            lbounds_ves[0] = Catan2(Csin(ves_lims[0]), Ccos(ves_lims[0]))
            lbounds_ves[1] = Catan2(Csin(ves_lims[1]), Ccos(ves_lims[1]))
        # -- Toroidal case -----------------------------------------------------
        for ind_surf in range(num_surf):
            # rmin is necessary to avoid looking on the other side of the tok
            if rmin < 0.:
                rmin = 0.95*min(np.min(ves_poly[ind_surf, 0, ...]),
                                    np.min(np.hypot(ray_orig[0, ...],
                                                    ray_orig[1, ...])))
            rmin2 = rmin*rmin
            # Variable to avoid looking "behind" blind spot of tore
            if forbid:
                forbid0, forbidbis = 1, 1
            else:
                forbid0, forbidbis = 0, 0
            # Getting size of poly
            npts_poly = lnvert[ind_surf]
            # -- Computing intersection between LOS and Vessel -----------------
            raytracing_minmax_struct_tor(num_los, ray_vdir, ray_orig,
                                         &coeff_inter_out.data.as_doubles[ind_surf*num_los],
                                         &coeff_inter_in.data.as_doubles[ind_surf*num_los],
                                         forbid0, forbidbis,
                                         rmin, rmin2, Crit2_base,
                                         npts_poly, lbounds_ves,
                                         are_limited,
                                         &ves_poly[ind_surf][0][0],
                                         &ves_poly[ind_surf][1][0],
                                         &ves_norm[ind_surf][0][0],
                                         &ves_norm[ind_surf][1][0],
                                         eps_uz, eps_vz, eps_a,
                                         eps_b, eps_plane,
                                         num_threads)
    else:
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None  or np.size(ves_lims) == 0:
            are_limited = False
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
        else:
            are_limited = True
            lbounds_ves[0] = ves_lims[0]
            lbounds_ves[1] = ves_lims[1]

        # -- Cylindrical case --------------------------------------------------
        for ind_surf in range(num_surf):
            # Getting size of poly
            npts_poly = lnvert[ind_surf]
            raytracing_minmax_struct_lin(num_los, ray_orig, ray_vdir, npts_poly,
                                         &ves_poly[ind_surf][0][0],
                                         &ves_poly[ind_surf][1][0],
                                         &ves_norm[ind_surf][0][0],
                                         &ves_norm[ind_surf][1][0],
                                         lbounds_ves[0], lbounds_ves[1],
                                         &coeff_inter_out.data.as_doubles[ind_surf*num_los],
                                         &coeff_inter_in.data.as_doubles[ind_surf*num_los],
                                         eps_plane)

    return np.asarray(coeff_inter_in), np.asarray(coeff_inter_out)


def LOS_isVis_PtFromPts_VesStruct(double pt0, double pt1, double pt2,
                                  np.ndarray[double, ndim=1,mode='c'] k,
                                  np.ndarray[double, ndim=2,mode='c'] pts,
                                  np.ndarray[double, ndim=2,mode='c'] VPoly,
                                  np.ndarray[double, ndim=2,mode='c'] VIn,
                                  Lim=None, LSPoly=None, LSLim=None, LSVIn=None,
                                  RMin=None, Forbid=True, EpsUz=_SMALL,
                                  EpsVz=_VSMALL, EpsA=_VSMALL, EpsB=_VSMALL,
                                  EpsPlane=_VSMALL, VType='Tor', Test=True):
    """ Return an array of bool indices indicating whether each point in pts is
    visible from Pt considering vignetting
    """
    if Test:
        C0 = (VPoly.shape[0]==2 and VIn.shape[0]==2
              and VIn.shape[1]==VPoly.shape[1]-1)
        msg = "Args VPoly and VIn must be of the same shape (2,NS)!"
        assert C0, msg
        C0 = all([pp is None for pp in [LSPoly,LSLim,LSVIn]])
        C1 = all([hasattr(pp,'__iter__') and len(pp)==len(LSPoly)
                  for pp in [LSPoly,LSLim,LSVIn]])
        msg = "Args LSPoly,LSLim,LSVIn must be None or lists of same len()!"
        assert C0 or C1, msg
        C0 = RMin is None or type(RMin) in [float,int,np.float64,np.int64]
        assert msg, "Arg RMin must be None or a float!"
        assert type(Forbid) is bool, "Arg Forbid must be a bool!"
        C0 = all([type(ee) in [int,float,np.int64,np.float64] and ee<1.e-4
                  for ee in [EpsUz,EpsVz,EpsA,EpsB,EpsPlane]])
        assert C0, "Args [EpsUz,EpsVz,EpsA,EpsB] must be floats < 1.e-4!"
        C0 = type(VType) is str and VType.lower() in ['tor','lin']
        assert C0, "Arg VType must be a str in ['Tor','Lin']!"

    cdef int ii, jj, npts=pts.shape[1]
    cdef np.ndarray[double, ndim=2, mode='c'] Ds, dus
    Ds = np.tile(np.r_[pt0,pt1,pt2], (npts,1)).T
    dus = (pts-Ds)/k

    if VType.lower()=='tor':
        # RMin is necessary to avoid looking on the other side of the tokamak
        if RMin is None:
            RMin = 0.95*min(np.min(VPoly[0,:]),
                            np.min(np.hypot(Ds[0,:],Ds[1,:])))

        # Main function to compute intersections with Vessel
        POut = Calc_LOS_PInOut_Tor(Ds, dus, VPoly, VIn, Lim=Lim, Forbid=Forbid,
                                   RMin=RMin, EpsUz=EpsUz, EpsVz=EpsVz,
                                   EpsA=EpsA, EpsB=EpsB, EpsPlane=EpsPlane)[1]

        # k = coordinate (in m) along the line from D
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        # Structural optimzation : do everything in one big for loop and only
        # keep the relevant points (to save memory)
        if LSPoly is not None:
            for ii in range(0,len(LSPoly)):
                C0 = not all([hasattr(ll,'__iter__') for ll in LSLim[ii]])
                if LSLim[ii] is None or C0:
                    lslim = [LSLim[ii]]
                else:
                    lslim = LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn = Calc_LOS_PInOut_Tor(Ds, dus, LSPoly[ii], LSVIn[ii],
                                              Lim=lslim[jj], Forbid=Forbid,
                                              RMin=RMin, EpsUz=EpsUz,
                                              EpsVz=EpsVz, EpsA=EpsA, EpsB=EpsB,
                                              EpsPlane=EpsPlane)[0]
                    kpin = np.sqrt(np.sum((Ds-pIn)**2,axis=0))
                    indNoNan = (~np.isnan(kpin)) & (~np.isnan(kPOut))
                    indout = np.zeros((npts,),dtype=bool)
                    indout[indNoNan] = kpin[indNoNan]<kPOut[indNoNan]
                    indout[(~np.isnan(kpin)) & np.isnan(kPOut)] = True
                    if np.any(indout):
                        kPOut[indout] = kpin[indout]
    else:
        POut = Calc_LOS_PInOut_Lin(Ds, dus, VPoly, VIn, Lim, EpsPlane=EpsPlane)[1]
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        if LSPoly is not None:
            for ii in range(0,len(LSPoly)):
                C0 = not all([hasattr(ll,'__iter__') for ll in LSLim[ii]])
                lslim = [LSLim[ii]] if C0 else LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn = Calc_LOS_PInOut_Lin(Ds, dus, LSPoly[ii], LSVIn[ii],
                                              lslim[jj], EpsPlane=EpsPlane)[0]
                    kpin = np.sqrt(np.sum((Ds-pIn)**2,axis=0))
                    indNoNan = (~np.isnan(kpin)) & (~np.isnan(kPOut))
                    indout = np.zeros((npts,),dtype=bool)
                    indout[indNoNan] = kpin[indNoNan]<kPOut[indNoNan]
                    indout[(~np.isnan(kpin)) & np.isnan(kPOut)] = True
                    if np.any(indout):
                        kPOut[indout] = kpin[indout]

    ind = np.zeros((npts,),dtype=bool)
    indok = (~np.isnan(k)) & (~np.isnan(kPOut))
    ind[indok] = k[indok]<kPOut[indok]
    return ind






######################################################################
#               Sampling
######################################################################


# .................................. TODO .................................
# optimize this algorithm..................................................
# .................................. TODO .................................
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
    # NE PAS RENVOYER LES POINTS..........................................
    # k = [liste de k pour LOS_0, liste de k pour LOS_1, ...]
    # + un autre tab d'indices : [indice du dernier de LOS_0, indice du dernier de LOS_1, ...]
    # tq : on peut utiliser split avec k
    #.....................................................................
    """ Return the sampled line, with the specified method

    #.. n'existe plus : 'linspace': return the N+1 edges, including the first and last point
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
        assert C0 or C1, "Arg dL must be >0.!"
        assert dLMode.lower() in ['abs','rel'], "Arg dLMode in ['abs','rel']"
        assert method.lower() in ['sum','simps','romb'], "Arg method"

    cdef unsigned int ii, jj, N, ND = Ds.shape[1]
    cdef double kkk, D0, D1, D2, u0, u1, u2, dl0, dl
    cdef np.ndarray[double,ndim=1] dLr = np.empty((ND,),dtype=float)
    cdef np.ndarray[double,ndim=1] kk
    cdef np.ndarray[double,ndim=2] pts
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
    assert hasattr(ff,'__call__'), stre+" must be a callable (function)!"
    na, kw = get_insp(ff)
    assert na==1, stre+" must take only one positional argument: ff(Pts)!"
    assert 't' in kw, stre+" must have kwarg 't=None' for time vector!"
    C = type(t) in [int,float,np.int64,np.float64] or hasattr(t,'__iter__')
    assert t is None or C, "Arg t must be None, a scalar or an iterable!"
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
               +"t a len()=nt iterable, must return a (nt,N) np.ndarray!")
        assert type(out) is np.ndarray and out.shape==(nt,NP), Str
    else:
        Str = ("When fed a (3,N) np.array only, or if t is a scalar,"
               +" ff must return a (N,) np.ndarray!")
        assert type(out) is np.ndarray and out.shape==(NP,), Str

    ani = ('Vect' in kw) if Ani is None else Ani
    if ani:
        Str = "If Ani=True, ff must take a keyword argument 'Vect=None'!"
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



def integrate1d(y, double dx, t=None, str method='sum'):
    """ Generic integration method ['sum','simps','romb']

        Not used internally
        Useful when the sampling points need to be interpolated via equilibrium
    """
    cdef unsigned int nt, axm
    if t is None or not hasattr(t,'__iter__'):
        nt = 1
        axm = 0
    else:
        nt = len(t)
        axm = 1
    ind = np.isnan(y)
    if np.any(ind):
        y = y.copy()
        y[ind] = 0.

    cdef np.ndarray[double,ndim=1] s = np.empty((nt,),dtype=float)

    if method=='sum':
        s = np.sum(y, axis=axm)*dx
    elif method=='simps':
        s = scpintg.simps(y, x=None, dx=dx, axis=axm)
    elif method=='romb':
        s = scpintg.romb(y, dx=dx, axis=axm, show=False)
    else:
        raise Exception("Arg method must be in ['sum','simps','romb']")
    return s




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
        assert C0 or C1, "Arg dL must be >0.!"
        assert dLMode.lower() in ['abs','rel'], "Arg dLMode in ['abs','rel']"
        assert method.lower() in ['sum','simps','romb'], "Arg method"
    # Testing function
    cdef bool ani = check_ff(ff,t=t,Ani=Ani)

    cdef unsigned int nt, axm, ii, jj, N, ND = Ds.shape[1]
    cdef double kkk, D0, D1, D2, u0, u1, u2, dl0, dl
    cdef np.ndarray[double,ndim=2] pts
    if t is None or not hasattr(t,'__iter__'):
        nt = 1
        axm = 0
    else:
        nt = len(t)
        axm = 1
    cdef np.ndarray[double,ndim=2] sig = np.empty((nt,ND),dtype=float)

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


def LOS_sino_findRootkPMin_Tor(double uParN, double uN, double Sca, double RZ0,
                                double RZ1, double ScaP, double DParN,
                                double kOut, double D0, double D1, double D2,
                                double u0, double u1, double u2, str Mode='LOS'):
    """
    Rendre "vectoriel" sur LOS et sur les cercles (deux boucles "for")
    Intersection ligne et cercle
    double uParN : composante de u parallel au plan (x,y)
        double uN : uz
        double Sca : ??? produit scalaire ... ?
        double RZ0 : Grand rayon du cercle
        double RZ1 : Z
        => cercle est centré au point (0, 0, RZ1) et rayon RZ0
        double ScaP : .... ?
        double DParN : D origine de LOS.... ? N => norme de la composante du vecteur OD
        double kOut : kmax où on peut trouver un résultat
        double D0, double D1, double D2 : composantes de D (origine LOS)
        double u0, double u1, double u2 : composantes de U (direction LOS)
        str Mode='LOS' : si LOS pas de sol après kmax)
    ::: Faire une fonction double mais qui renvoit QUE un tableau de bool avec true si
    la distance est plus petite qu'un certain eps, false sinon.
    TODO: ........... @LM
    """
    cdef double a4 = (uParN*uN*uN)**2, a3 = 2*( (Sca-RZ1*u2)*(uParN*uN)**2 + ScaP*uN**4 )
    cdef double a2 = (uParN*(Sca-RZ1*u2))**2 + 4.*ScaP*(Sca-RZ1*u2)*uN**2 + (DParN*uN*uN)**2 - (RZ0*uParN*uParN)**2
    cdef double a1 = 2*( ScaP*(Sca-RZ1*u2)**2 + (Sca-RZ1*u2)*(DParN*uN)**2 - ScaP*(RZ0*uParN)**2 )
    cdef double a0 = ((Sca-RZ1*u2)*DParN)**2 - (RZ0*ScaP)**2
    cdef np.ndarray roo = np.roots(np.array([a4,a3,a2,a1,a0]))
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
    return kPMin # + distance au cercle



cdef LOS_sino_Tor(double D0, double D1, double D2, double u0, double u1,
                  double u2, double RZ0, double RZ1, str Mode='LOS', double kOut=np.inf):

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



cdef inline void NEW_LOS_sino_Tor(double orig0, double orig1, double orig2,
                                  double dirv0, double dirv1, double dirv2,
                                  double circ_radius, double circ_normz,
                                  double[9] results,
                                  bint is_LOS_Mode=False,
                                  double kOut=np.inf) :
    cdef double[3] dirv, orig
    cdef double[2] res
    cdef double normu, normu_sqr
    cdef double kPMin

    normu_sqr = dirv0 * dirv0 + dirv1 * dirv1 + dirv2 * dirv2
    normu = Csqrt(normu_sqr)
    dirv[0] = dirv0
    dirv[2] = dirv2
    dirv[1] = dirv1
    orig[0] = orig0
    orig[1] = orig1
    orig[2] = orig2

    if dirv0 == 0. and dirv1 == 0.:
        kPMin = (circ_normz-orig2)/dirv2
    else:
        dist_los_circle_core(dirv, orig,
                             circ_radius, circ_normz,
                             normu_sqr, res)
        kPMin = res[0]
        if is_LOS_Mode and kPMin > kOut:
            kPMin = kOut
 
    # Computing the point's coordinates.........................................
    cdef double PMin0 = orig0 + kPMin * dirv0
    cdef double PMin1 = orig1 + kPMin * dirv1
    cdef double PMin2 = orig2 + kPMin * dirv2
    cdef double PMin2norm = Csqrt(PMin0**2+PMin1**2)
    cdef double RMin = Csqrt((PMin2norm - circ_radius)**2
                             + (PMin2   - circ_normz)**2)
    cdef double vP0 = PMin2norm - circ_radius
    cdef double vP1 = PMin2     - circ_normz
    cdef double Theta = Catan2(vP1, vP0)
    cdef double ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef double er2D0 = Ccos(ImpTheta)
    cdef double er2D1 = Csin(ImpTheta)
    cdef double p0 = vP0*er2D0 + vP1*er2D1
    cdef double eTheta0 = -PMin1 / PMin2norm
    cdef double eTheta1 =  PMin0 / PMin2norm
    cdef double normu0 = dirv0/normu
    cdef double normu1 = dirv1/normu
    cdef double phi = Casin(-normu0 * eTheta0 - normu1 * eTheta1)
    # Filling the results ......................................................
    results[0] = PMin0
    results[1] = PMin1
    results[2] = PMin2
    results[3] = kPMin
    results[4] = RMin
    results[5] = Theta
    results[6] = p0
    results[7] = ImpTheta
    results[8] = phi
    return

cdef inline void NEW_los_sino_tor_vec(int num_los,
                                      double[:,::1] origins,
                                      double[:,::1] directions,
                                      double circ_radius,
                                      double circ_normz,
                                      double[:,::1] los_closest_coords,
                                      double[::1] los_closest_coeffs,
                                      double[::1] circle_closest_rmin,
                                      double[::1] circle_closest_theta,
                                      double[::1] circle_closest_p,
                                      double[::1] circle_closest_imptheta,
                                      double[::1] circle_closest_phi,
                                      bint is_LOS_Mode=False,
                                      double[::1] kOut=None) nogil:
    cdef int ind_los
    cdef double* dirv
    cdef double* orig
    cdef double* res
    cdef double normu, normu_sq
    cdef double kPMin, PMin2norm, vP0, vP1, Theta
    cdef double eTheta0
    cdef double eTheta1
    cdef double normu0
    cdef double normu1
    cdef double distance
    cdef double PMin0, PMin1, PMin2

    with nogil, parallel():
        dirv = <double*>malloc(3*sizeof(double))
        orig = <double*>malloc(3*sizeof(double))
        res = <double*>malloc(2*sizeof(double))
        for ind_los in prange(num_los):
            dirv[0] = directions[0, ind_los]
            dirv[1] = directions[1, ind_los]
            dirv[2] = directions[2, ind_los]
            orig[0] = origins[0, ind_los]
            orig[1] = origins[1, ind_los]
            orig[2] = origins[2, ind_los]
            normu_sq = dirv[0] * dirv[0] + dirv[1] * dirv[1] + dirv[2] * dirv[2]
            normu = Csqrt(normu_sq)
            # Computing coeff of closest on line................................
            if dirv[0] == 0. and dirv[1] == 0.:
                kPMin = (circ_normz-orig[2])/dirv[2]
            else:
                dist_los_circle_core(dirv, orig, circ_radius,
                                     circ_normz, normu_sq, res)
                kPMin = res[0]
                distance = res[1]
            if is_LOS_Mode and kOut is not None and kPMin > kOut[ind_los]:
                kPMin = kOut[ind_los]
            los_closest_coeffs[ind_los] = kPMin

            # Computing the info of the closest point on LOS & Circle...........
            PMin0 = orig[0] + kPMin * dirv[0]
            PMin1 = orig[1] + kPMin * dirv[1]
            PMin2 = orig[2] + kPMin * dirv[2]
            los_closest_coords[0, ind_los] = PMin0
            los_closest_coords[1, ind_los] = PMin1
            los_closest_coords[2, ind_los] = PMin2
            # Computing RMin:
            PMin2norm = Csqrt(PMin0**2+PMin1**2)
            circle_closest_rmin[ind_los] = Csqrt((PMin2norm - circ_radius)**2
                                        + (PMin2   - circ_normz)**2)
            # Theta and ImpTheta:
            vP0 = PMin2norm - circ_radius
            vP1 = PMin2     - circ_normz
            Theta = Catan2(vP1, vP0)
            circle_closest_theta[ind_los] = Theta
            if Theta < 0:
                Theta = Theta + Cpi
            circle_closest_imptheta[ind_los] = Theta
            circle_closest_p[ind_los] = vP0 * Ccos(Theta) + vP1 * Csin(Theta)
            # Phi:
            eTheta0 = - PMin1 / PMin2norm
            eTheta1 =   PMin0 / PMin2norm
            normu0 = dirv[0]/normu
            normu1 = dirv[1]/normu
            circle_closest_phi[ind_los] = Casin(-normu0 * eTheta0 - normu1 * eTheta1)
    return



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
    cdef double    p0 = vP0*er2D0 + vP1*er2D1
    cdef double    uN = Csqrt(u0**2+u1**2+u2**2)
    cdef double    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef double    phi = Catan2(uN0, Csqrt(uN1**2+uN2**2))
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p0, ImpTheta, phi


def LOS_sino(double[:,::1] D, double[:,::1] u, double[::1] RZ, double[::1] kOut,
             str Mode='LOS', str VType='Tor', bint try_new_algo=True):
    cdef unsigned int nL = D.shape[1], ii
    cdef tuple out
    cdef np.ndarray[double,ndim=2] PMin = np.empty((3,nL))
    cdef np.ndarray[double,ndim=1] kPMin=np.empty((nL,)), RMin=np.empty((nL,))
    cdef np.ndarray[double,ndim=1] Theta=np.empty((nL,)), p=np.empty((nL,))
    cdef np.ndarray[double,ndim=1] ImpTheta=np.empty((nL,)), phi=np.empty((nL,))
    cdef double[9] results
    cdef bint is_LOS_Mode
    if VType.lower()=='tor':
        if not try_new_algo:
            for ii in range(0,nL):
                out = LOS_sino_Tor(D[0,ii],D[1,ii],D[2,ii],
                                   u[0,ii],u[1,ii],u[2,ii],
                                   RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
                ((PMin[0,ii],PMin[1,ii],PMin[2,ii]),
                 kPMin[ii], RMin[ii], Theta[ii],
                 p[ii], ImpTheta[ii], phi[ii]) = out
        else:
            is_LOS_Mode = Mode.lower() == 'los'
            NEW_los_sino_tor_vec(nL, D, u, RZ[0], RZ[1],
                                 PMin, kPMin, RMin, Theta, p,
                                 ImpTheta, phi,
                                 is_LOS_Mode=is_LOS_Mode,
                                 kOut=kOut)
    else:
        for ii in range(0,nL):
            out = LOS_sino_Lin(D[0,ii],D[1,ii],D[2,ii],u[0,ii],u[1,ii],u[2,ii],
                               RZ[0],RZ[1], Mode=Mode, kOut=kOut[ii])
            ((PMin[0,ii],PMin[1,ii],PMin[2,ii]),
             kPMin[ii], RMin[ii], Theta[ii], p[ii], ImpTheta[ii], phi[ii]) = out
    return PMin, kPMin, RMin, Theta, p, ImpTheta, phi








########################################################
########################################################
########################################################
#                   Solid Angle
########################################################
########################################################
########################################################


######################################################
######################################################
#               Dust
######################################################
######################################################

def Dust_calc_SolidAngle(pos, r, pts,
                         approx=True, out_coefonly=False,
                         VType='Tor', VPoly=None, VIn=None, VLim=None,
                         LSPoly=None, LSLim=None, LSVIn=None, Forbid=True,
                         Test=True):
    """ Compute the solid angle of a moving particle of varying radius as seen
    from any number of pixed points

    Can be done w/o the approximation that r<<d
    If Ves (and optionally LSPoly) are provided, takes into account vignetting
    """
    cdef block = VPoly is not None
    cdef float pir2
    cdef int ii, jj, nptsok, nt=pos.shape[1], npts=pts.shape[1]
    cdef np.ndarray[double, ndim=2, mode='c'] sang=np.zeros((nt,npts))

    if block:
        ind = ~_Ves_isInside(pts, VPoly, Lim=VLim, VType=VType,
                             In='(X,Y,Z)', Test=Test)
        if LSPoly is not None:
            for ii in range(0,len(LSPoly)):
                ind = ind & _Ves_isInside(pts, LSPoly[ii], Lim=LSLim[ii],
                                          VType=VType, In='(X,Y,Z)', Test=Test)
        ind = (~ind).nonzero()[0]
        ptstemp = np.ascontiguousarray(pts[:,ind])
        nptsok = ind.size

        if approx and out_coefonly:
            for ii in range(0,nt):
                k = np.sqrt((pos[0,ii]-ptstemp[0,:])**2
                            + (pos[1,ii]-ptstemp[1,:])**2
                            + (pos[2,ii]-ptstemp[2,:])**2)

                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii], k, ptstemp,
                                                    VPoly, VIn, Lim=VLim,
                                                    LSPoly=LSPoly, LSLim=LSLim,
                                                    LSVIn=LSVIn, Forbid=Forbid,
                                                    VType=VType, Test=Test)
                for jj in range(0,nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = Cpi/k[jj]**2
        elif approx:
            for ii in range(0,nt):
                k = np.sqrt((pos[0,ii]-ptstemp[0,:])**2
                            + (pos[1,ii]-ptstemp[1,:])**2
                            + (pos[2,ii]-ptstemp[2,:])**2)

                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii], k, ptstemp,
                                                    VPoly, VIn, Lim=VLim,
                                                    LSPoly=LSPoly, LSLim=LSLim,
                                                    LSVIn=LSVIn, Forbid=Forbid,
                                                    VType=VType, Test=Test)
                pir2 = Cpi*r[ii]**2
                for jj in range(0,nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = pir2/k[jj]**2
        else:
            pir2 = 2*Cpi
            for ii in range(0,nt):
                k = np.sqrt((pos[0,ii]-ptstemp[0,:])**2
                            + (pos[1,ii]-ptstemp[1,:])**2
                            + (pos[2,ii]-ptstemp[2,:])**2)

                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii], k, ptstemp,
                                                    VPoly, VIn, Lim=VLim,
                                                    LSPoly=LSPoly, LSLim=LSLim,
                                                    LSVIn=LSVIn, Forbid=Forbid,
                                                    VType=VType, Test=Test)
                for jj in range(0,nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = pir2*(1-Csqrt(1-r[ii]**2/k[jj]**2))

    else:
        if approx and out_coefonly:
            for ii in range(0,nt):
                for jj in range(0,npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[1,ii]-pts[1,jj])**2
                            + (pos[2,ii]-pts[2,jj])**2)
                    sang[ii,jj] = Cpi/dij2
        elif approx:
            for ii in range(0,nt):
                pir2 = Cpi*r[ii]**2
                for jj in range(0,npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2)
                    sang[ii,jj] = pir2/dij2
        else:
            pir2 = 2*Cpi
            for ii in range(0,nt):
                for jj in range(0,npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2)
                    sang[ii,jj] = pir2*(1-Csqrt(1-r[ii]**2/dij2))
    return sang

# ==============================================================================
#
#                       VECTOR CALCULUS HELPERS
#
# ==============================================================================
cdef inline void compute_cross_prod(const double[3] vec_a,
                                    const double[3] vec_b,
                                    double[3] res) nogil:
    res[0] = vec_a[1]*vec_b[2] - vec_a[2]*vec_b[1]
    res[1] = vec_a[2]*vec_b[0] - vec_a[0]*vec_b[2]
    res[2] = vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]
    return

cdef inline double compute_dot_prod(const double[3] vec_a,
                                    const double[3] vec_b) nogil:
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]


cdef inline double compute_g(double s, double m2b2, double rm0sqr,
                             double m0sqr, double b1sqr) nogil:
    return s + m2b2 - rm0sqr*s / Csqrt(m0sqr*s*s + b1sqr)

cdef inline double compute_bisect(double m2b2, double rm0sqr,
                                  double m0sqr, double b1sqr,
                                  double smin, double smax) nogil:
    cdef int maxIterations = 10000
    cdef double root = 0.
    root = compute_find(m2b2, rm0sqr, m0sqr, b1sqr,
                smin, smax, -1.0, 1.0, maxIterations, root)
    gmin = compute_g(root, m2b2, rm0sqr, m0sqr, b1sqr)
    return root

cdef inline double compute_find(double m2b2, double rm0sqr,
                                double m0sqr, double b1sqr,
                                double t0, double t1, double f0, double f1,
                                int maxIterations, double root) nogil:
    cdef double fm, product
    if (t0 < t1):
        # Test the endpoints to see whether F(t) is zero.
        if f0 == 0.:
            root = t0
            return root
        if f1 == 0.:
            root = t1
            return root
        if f0*f1 > 0.:
            # It is not known whether the interval bounds a root.
            return root
        for i in range(2, maxIterations+1):
            root = (0.5) * (t0 + t1)
            if (root == t0 or root == t1):
                # The numbers t0 and t1 are consecutive floating-point
                # numbers.
                break
            fm = compute_g(root, m2b2, rm0sqr, m0sqr, b1sqr)
            product = fm * f0
            if (product < 0.):
                t1 = root
                f1 = fm
            elif (product > 0.):
                t0 = root
                f0 = fm
            else:
                break
        return root
    else:
        return root


# ==============================================================================
#
#                       DISTANCE CIRCLE - LOS
#
# ==============================================================================
def comp_dist_los_circle(np.ndarray[double,ndim=1,mode='c'] ray_vdir,
                         np.ndarray[double,ndim=1,mode='c'] ray_orig,
                         double radius, double circ_z, double norm_dir=-1.0):
    """
    This function computes the intersection of a Ray (or Line Of Sight)
    and a circle in 3D. It returns `kmin` and `dist`. Where `kmin` is the
    coefficient such that the ray of origin O = [ori1, ori2, ori3]
    and of directional vector D = [dir1, dir2, dir3] is closest to the circle
     of radius `radius` and centered `(0, 0, circ_z)` at the point
    P = O + kmin * D.
    And `distance` the distance between the two closest points (line closest
    and circle closest)
    The variable `norm_dir` is the squared norm of the direction of the ray.
    Params
    =====
    ray_vdir: (3) double array
        ray's director vector V such that P \in Ray iff P(t) = O + t*V
    ray_orig : (3) double array
        ray's origin coordinates O such that P \in Ray iff P(t) = O + t*V
    radius : double
        radius r of horizontal circle centered in (0,0,circ_z)
    circ_z : double
        3rd coordinate of horizontal circle centered in (0,0,circ_z) of radius r
    norm_dir : double (optional)
        If for computation reasons it makes sense, you can pass the norm of the
        director vector
    Returns
    =======
    result : double (2) array
       - result[0] will contain the k coefficient to find the line point closest
       closest point
       - result[1] will contain the DISTANCE from line closest point to circle
       to the circle
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `dist_los_circle_core`
    """
    cdef double[2] res
    dist_los_circle_core(<double*>ray_vdir.data,
                         <double*>ray_orig.data,
                         radius, circ_z, norm_dir, res)
    return np.asarray(res)

cdef inline void dist_los_circle_core(const double[3] direct,
                                      const double[3] origin,
                                      const double radius, const double circ_z,
                                      double norm_dir,
                                      double[2] result) nogil:
    """
    This function computes the intersection of a Ray (or Line Of Sight)
    and a horizontal circle in 3. It returns `kmin` the coefficient such that
    the ray of origin O = [ori1, ori2, ori3] and of directional vector
    D = [dir1, dir2, dir3] is closest to the circle of radius `radius`,
    center `(0, 0, circ_z)` and of normal (0,0,1) at the point P = O + kmin * D.
    And `distance` the distance between the two closest points
    The variable `norm_dir` is the norm of the direction of the ray.
    if you haven't normalized the ray (and for optimization reasons you dont
    want to, you can pass norm_dir = -1
    ---
    Source: https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
    The line is P(t) = B+t*M.  The circle is |X-C| = r with Dot(N,X-C)=0.

    Params
    ======
    direct : double (3) array
       directional vector of the ray
    origin : double (3) array
       origin of the array (in 3d)
    radius : double
       radius of the circle
    circ_z : double
       3rd coordinate of the center of the circle
       ie. the circle center is (0,0, circ_z)
    norm_dir : double (3) array
       normal of the direction of the vector (for computation performance)
    result : double (2) array
       - result[0] will contain the k coefficient to find the line point closest
       closest point
       - result[1] will contain the DISTANCE from line closest point to circle
       to the circle
    """
    cdef int numRoots, i
    cdef double zero = 0., m0sqr, m0, rm0
    cdef double lambd, m2b2, b1sqr, b1, r0sqr, twoThirds, sHat, gHat, cutoff, s
    cdef double[3] D
    cdef double[3] MxN
    cdef double[3] DxN
    cdef double[3] NxDelta
    cdef double[3] circle_normal
    cdef double[3] roots
    cdef double[3] diff
    cdef double[3] direction
    cdef double[3] line_closest
    cdef double[3] circle_center
    cdef double[3] circle_closest
    cdef double tmin
    cdef double distance
    cdef double inv_norm_dir

    if norm_dir < 0:
        norm_dir = Csqrt(compute_dot_prod(direct, direct))
    inv_norm_dir = 1./ norm_dir
    # .. initialization .....
    for i in range(3):
        circle_center[i] = 0.
        circle_normal[i] = 0.
        roots[i] = 0.
        # we normalize direction
        direction[i] = direct[i] * inv_norm_dir
    circle_normal[2] = 1
    circle_center[2] = circ_z

    D[0] = origin[0]
    D[1] = origin[1]
    D[2] = origin[2] - circ_z
    compute_cross_prod(direction, circle_normal, MxN)
    compute_cross_prod(D, circle_normal, DxN)
    m0sqr = compute_dot_prod(MxN, MxN)

    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        numRoots = 0

        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*direction =
        # (0,b1',b2').
        m0 = Csqrt(m0sqr)
        rm0 = radius * m0
        lambd = -compute_dot_prod(MxN, DxN) / m0sqr
        for i in range(3):
            D[i] += lambd * direction[i]
            DxN[i] += lambd * MxN[i]
        m2b2 = compute_dot_prod(direction, D)
        b1sqr = compute_dot_prod(DxN, DxN)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = Csqrt(b1sqr)
            rm0sqr = radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = Csqrt((rm0sqr * b1sqr)**twoThirds - b1sqr) / m0
                gHat = rm0sqr * sHat / Csqrt(m0sqr * sHat * sHat + b1sqr)
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                else:
                    s = zero
                roots[numRoots] = s
                numRoots += 1
        else:
            # The new line origin is B' = (0,0,b2').
            if (m2b2 < zero):
                s = -m2b2 + rm0
                roots[numRoots] = s
                numRoots += 1

            elif (m2b2 > zero):
                s = -m2b2 - rm0
                roots[numRoots] = s
                numRoots += 1
            else:
                s = -m2b2 + rm0
                roots[numRoots] = s
                numRoots += 1
                s = -m2b2 - rm0
                roots[numRoots] = s
                numRoots += 1
        # Checking which one is the closest solution............................
        tmin = roots[0] + lambd
        for i in range(1,numRoots):
            t = roots[i] + lambd
            if (t>0 and t<tmin):
                tmin = t
        if tmin < 0:
            tmin = 0.
        # Now that we know the closest point on the line we can compute the
        # closest point on the circle and compute the distance
        line_closest[0] = origin[0] + tmin * direction[0]
        line_closest[1] = origin[1] + tmin * direction[1]
        line_closest[2] = origin[2] + tmin * direction[2]
        compute_cross_prod(circle_normal, line_closest, NxDelta)
        if not (Cabs(NxDelta[0]) <= _VSMALL
                and Cabs(NxDelta[1]) <= _VSMALL
                and Cabs(NxDelta[2]) <= _VSMALL):
            norm_ppar = Csqrt(line_closest[0]*line_closest[0]
                              + line_closest[1]*line_closest[1])
            circle_closest[0] = radius * line_closest[0] / norm_ppar
            circle_closest[1] = radius * line_closest[1] / norm_ppar
            circle_closest[2] = circle_center[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
        else:
            diff[0] = line_closest[0] - radius
            diff[1] = line_closest[1]
            diff[2] = line_closest[2] - circle_center[2]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
    else:
        # The line direction and the plane normal are parallel.
        # There is only one solution the intersection between line and plane
        if not (Cabs(DxN[0]) <= _VSMALL
                and Cabs(DxN[1]) <= _VSMALL
                and Cabs(DxN[2]) <= _VSMALL):
            # The line is A+t*N but with A != C.
            t = -compute_dot_prod(direction, D)
            # We compute line closest
            line_closest[0] = origin[0] + t * direction[0]
            line_closest[1] = origin[1] + t * direction[1]
            line_closest[2] = origin[2] + t * direction[2]
            # We compute cirlce closest
            for i in range(3):
                diff[i] = line_closest[i] - circle_center[i]
            distance = radius / Csqrt(compute_dot_prod(diff, diff))
            circle_closest[0] = line_closest[0] * distance
            circle_closest[1] = line_closest[1] * distance
            circle_closest[2] = circ_z + (line_closest[2] - circ_z) * distance
            if t < 0:
                # fi t is negative, we take origin as closest point
                line_closest[0] = origin[0]
                line_closest[1] = origin[1]
                line_closest[2] = origin[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = t
            result[1] = distance
        else:
            # The line direction and the normal vector are on the same line
            # so C is the closest point for the circle and the distance is
            # the radius unless the ray's origin is after the circle center
            if (origin[2] * direction[2] <= circle_center[2] * direction[2]) :
                t = Cabs(circle_center[2] - origin[2])
                result[0] = t
                result[1] = radius
            else:
                t = Cabs(circle_center[2] - origin[2])
                result[0] = 0
                result[1] = Csqrt(radius*radius + t*t)
    result[0] = result[0] * inv_norm_dir
    return


def comp_dist_los_circle_vec(int nlos, int ncircles,
                             np.ndarray[double,ndim=2,mode='c'] dirs,
                             np.ndarray[double,ndim=2,mode='c'] oris,
                             np.ndarray[double,ndim=1,mode='c'] circle_radius,
                             np.ndarray[double,ndim=1,mode='c'] circle_z,
                             np.ndarray[double,ndim=1,mode='c'] norm_dir = None):
    """
    This function computes the intersection of a Ray (or Line Of Sight)
    and a circle in 3D. It returns `kmin`, the coefficient such that the
    ray of origin O = [ori1, ori2, ori3] and of directional vector
    D = [dir1, dir2, dir3] is closest to the circle of radius `radius`
    and centered `(0, 0, circ_z)` at the point P = O + kmin * D.
    The variable `norm_dir` is the squared norm of the direction of the ray.
    This is the vectorial version, we expect the directions and origins to be:
    dirs = [[dir1_los1, dir2_los1, dir3_los1], [dir1_los2,...]
    oris = [[ori1_los1, ori2_los1, ori3_los1], [ori1_los2,...]
    Returns
    =======
    res : (2, nlos, ncircles)
        res = [res_k, res_d] where res_k is a (nlos, ncircles) numpy array
        with the k coefficients for each LOS where the minimum distance
        to each circle is reached
        is met for each circle, and res_d is a (nlos, ncircles) numpy array
        with the distance between each LOS to each circle
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `dist_los_circle_core`
    """
    cdef array kmin_tab = clone(array('d'), nlos*ncircles, True)
    cdef array dist_tab = clone(array('d'), nlos*ncircles, True)

    if norm_dir is None:
        norm_dir = -np.ones(nlos)
    comp_dist_los_circle_vec_core(nlos, ncircles,
                                  <double*>dirs.data,
                                  <double*>oris.data,
                                  <double*>circle_radius.data,
                                  <double*>circle_z.data,
                                  <double*>norm_dir.data,
                                  kmin_tab, dist_tab)
    return np.asarray(kmin_tab).reshape(nlos, ncircles), \
        np.asarray(dist_tab).reshape(nlos, ncircles)

cdef inline void comp_dist_los_circle_vec_core(int num_los, int num_cir,
                                               double* los_directions,
                                               double* los_origins,
                                               double* circle_radius,
                                               double* circle_z,
                                               double* norm_dir_tab,
                                               double[::1] res_k,
                                               double[::1] res_dist) nogil:
    """ This function computes the intersection of a Ray (or Line Of Sight)
    # and a circle in 3D. It returns `kmin`, the coefficient such that the
    # ray of origin O = [ori1, ori2, ori3] and of directional vector
    # D = [dir1, dir2, dir3] is closest to the circle of radius `radius`
    # and centered `(0, 0, circ_z)` at the point P = O + kmin * D.
    # The variable `norm_dir` is the squared norm of the direction of the ray.
    # This is the vectorial version, we expect the directions and origins to be:
    # dirs = [dir1_los1, dir2_los1, dir3_los1, dir1_los2,...]
    # oris = [ori1_los1, ori2_los1, ori3_los1, ori1_los2,...]
    # res = [kmin(los1, cir1), kmin(los1, cir2),...]
    # ---
    # This is the PYTHON function, use only if you need this computation from
    # Python, if you need it from Cython, use `dist_los_circle_core`
    """
    cdef int i, ind_los, ind_cir
    cdef double* loc_res
    cdef double* dirv
    cdef double* orig
    cdef double radius, circ_z, norm_dir
    with nogil, parallel():
        dirv = <double*>malloc(3*sizeof(double))
        orig = <double*>malloc(3*sizeof(double))
        loc_res = <double*>malloc(2*sizeof(double))
        for ind_los in prange(num_los):
            for i in range(3):
                dirv[i] = los_directions[ind_los * 3 + i]
                orig[i] = los_origins[ind_los * 3 + i]
            norm_dir = norm_dir_tab[ind_los]
            if norm_dir < 0.:
                norm_dir = Csqrt(compute_dot_prod(dirv, dirv))
            for ind_cir in range(num_cir):
                radius = circle_radius[ind_cir]
                circ_z = circle_z[ind_cir]
                dist_los_circle_core(dirv, orig, radius, circ_z,
                                     norm_dir, loc_res)
                res_k[ind_los * num_cir + ind_cir] = loc_res[0]
                res_dist[ind_los * num_cir + ind_cir] = loc_res[1]
        free(dirv)
        free(orig)
        free(loc_res)
    return

# ==============================================================================
#
#                       TEST CLOSENESS CIRCLE - LOS
#
# ==============================================================================

def is_close_los_circle(np.ndarray[double,ndim=1,mode='c'] ray_vdir,
                        np.ndarray[double,ndim=1,mode='c'] ray_orig,
                        double radius, double circ_z, double eps,
                        double norm_dir=-1.0):
    """
    This function checks if at maximum a LOS is at a distance epsilon
    form a cirlce
    The result is True when distance < epsilon
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `is_los_close_circle_core`
    """
    return is_close_los_circle_core(<double*>ray_vdir.data,
                                    <double*>ray_orig.data,
                                    radius, circ_z, norm_dir, eps)


cdef inline bint is_close_los_circle_core(const double[3] direct,
                                          const double[3] origin,
                                          double radius, double circ_z,
                                          double norm_dir, double eps) nogil:
    # Source: https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
    # The line is P(t) = B+t*M.  The circle is |X-C| = r with Dot(N,X-C)=0.
    cdef int numRoots, i
    cdef double zero = 0., m0sqr, m0, rm0
    cdef double lambd, m2b2, b1sqr, b1, r0sqr, twoThirds, sHat, gHat, cutoff, s
    cdef double[3] D
    cdef double[3] MxN
    cdef double[3] DxN
    cdef double[3] NxDelta
    cdef double[3] circle_normal
    cdef double[3] roots
    cdef double[3] diff
    cdef double[3] circle_center
    cdef double[3] circle_closest
    cdef double[3] line_closest
    cdef double[3] direction
    cdef double tmin
    cdef double distance
    cdef double inv_norm_dir
    cdef bint are_close

    # .. initialization .....
    if norm_dir < 0:
        norm_dir = Csqrt(compute_dot_prod(direct, direct))
    inv_norm_dir = 1./ norm_dir
    # .. initialization .....
    for i in range(3):
        circle_center[i] = 0.
        circle_normal[i] = 0.
        roots[i] = 0.
        # we normalize direction
        direction[i] = direct[i] * inv_norm_dir
    circle_normal[2] = 1
    circle_center[2] = circ_z

    D[0] = origin[0]
    D[1] = origin[1]
    D[2] = origin[2] - circ_z
    compute_cross_prod(direction, circle_normal, MxN)
    compute_cross_prod(D, circle_normal, DxN)
    m0sqr = compute_dot_prod(MxN, MxN)

    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        numRoots = 0
        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*direction =
        # (0,b1',b2').
        m0 = Csqrt(m0sqr)
        rm0 = radius * m0
        lambd = -compute_dot_prod(MxN, DxN) / m0sqr
        for i in range(3):
            D[i] += lambd * direction[i]
            DxN[i] += lambd * MxN[i]
        m2b2 = compute_dot_prod(direction, D)
        b1sqr = compute_dot_prod(DxN, DxN)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = Csqrt(b1sqr)
            rm0sqr = radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = Csqrt((rm0sqr * b1sqr)**twoThirds - b1sqr) / m0
                gHat = rm0sqr * sHat / Csqrt(m0sqr * sHat * sHat + b1sqr)
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                else:
                    s = zero
                roots[numRoots] = s
                numRoots += 1
        else:
            # The new line origin is B' = (0,0,b2').
            if (m2b2 < zero):
                s = -m2b2 + rm0
                roots[numRoots] = s
                numRoots += 1
            elif (m2b2 > zero):
                s = -m2b2 - rm0
                roots[numRoots] = s
                numRoots += 1
            else:
                s = -m2b2 + rm0
                roots[numRoots] = s
                numRoots += 1
                s = -m2b2 - rm0
                roots[numRoots] = s
                numRoots += 1
        # Checking which one is the closest solution............................
        tmin = roots[0] + lambd
        for i in range(1,numRoots):
            t = roots[i] + lambd
            if (t>0 and t<tmin):
                tmin = t
        if tmin < 0:
            tmin = 0.
        # Now that we know the closest point on the line we can compute the
        # closest point on the circle and compute the distance
        line_closest[0] = origin[0] + tmin * direction[0]
        line_closest[1] = origin[1] + tmin * direction[1]
        line_closest[2] = origin[2] + tmin * direction[2]
        compute_cross_prod(circle_normal, line_closest, NxDelta)
        if not (Cabs(NxDelta[0]) <= _VSMALL
                and Cabs(NxDelta[1]) <= _VSMALL
                and Cabs(NxDelta[2]) <= _VSMALL):
            norm_ppar = Csqrt(line_closest[0]*line_closest[0]
                              + line_closest[1]*line_closest[1])
            circle_closest[0] = radius * line_closest[0] / norm_ppar
            circle_closest[1] = radius * line_closest[1] / norm_ppar
            circle_closest[2] = circle_center[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            are_close = distance < eps
            return are_close
        else:
            diff[0] = line_closest[0] + radius
            diff[1] = line_closest[1]
            diff[2] = line_closest[2] - circle_center[2]
            distance = Csqrt(compute_dot_prod(diff, diff))
            are_close = distance < eps
            return are_close
    else:
        # The line direction and the plane normal are parallel.
        # There is only one solution the intersection between line and plane
        if not (Cabs(DxN[0]) <= _VSMALL
                and Cabs(DxN[1]) <= _VSMALL
                and Cabs(DxN[2]) <= _VSMALL):
            # The line is A+t*N but with A != C.
            t = -compute_dot_prod(direction, D)
            # We compute line closest
            line_closest[0] = origin[0] + t * direction[0]
            line_closest[1] = origin[1] + t * direction[1]
            line_closest[2] = origin[2] + t * direction[2]
            # We compute cirlce closest
            for i in range(3):
                diff[i] = line_closest[i] - circle_center[i]
            distance = radius / Csqrt(compute_dot_prod(diff, diff))
            circle_closest[0] = line_closest[0] * distance
            circle_closest[1] = line_closest[1] * distance
            circle_closest[2] = circ_z + (line_closest[2] - circ_z) * distance
            if t < 0:
                # fi t is negative, we take origin as closest point
                line_closest[0] = origin[0]
                line_closest[1] = origin[1]
                line_closest[2] = origin[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            are_close = distance < eps
            return are_close
        else:
            # The line direction and the normal vector are on the same line
            # so C is the closest point for the circle and the distance is
            # the radius unless the ray's origin is after the circle center
            if (origin[2] * direction[2] <= circle_center[2] * direction[2]) :
                are_close = radius < eps
                return are_close
            else:
                t = Cabs(circle_center[2] - origin[2])
                are_close = Csqrt(radius*radius + t*t) < eps
                return are_close


def is_close_los_circle_vec(int nlos, int ncircles, double epsilon,
                             np.ndarray[double,ndim=2,mode='c'] dirs,
                             np.ndarray[double,ndim=2,mode='c'] oris,
                             np.ndarray[double,ndim=1,mode='c'] circle_radius,
                             np.ndarray[double,ndim=1,mode='c'] circle_z,
                             np.ndarray[double,ndim=1,mode='c'] norm_dir=None):
    """
    This function checks if at maximum a LOS is at a distance epsilon
    form a cirlce. Vectorial version
    The result is True when distance < epsilon
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `is_los_close_circle_core`
    """
    cdef array res = clone(array('i'), nlos, True)

    if norm_dir is None:
        norm_dir = -np.ones(nlos)
    is_close_los_circle_vec_core(nlos, ncircles,
                                 epsilon,
                                 <double*>dirs.data,
                                 <double*>oris.data,
                                 <double*>circle_radius.data,
                                 <double*>circle_z.data,
                                 <double*>norm_dir.data,
                                 res)
    return np.asarray(res, dtype=bool).reshape(nlos, ncircles)


cdef inline void is_close_los_circle_vec_core(int num_los, int num_cir,
                                              double eps,
                                              double* los_directions,
                                              double* los_origins,
                                              double* circle_radius,
                                              double* circle_z,
                                              double* norm_dir_tab,
                                              int[::1] res) nogil:
    """
    This function computes the intersection of a Ray (or Line Of Sight)
    and a circle in 3D. It returns `kmin`, the coefficient such that the
    ray of origin O = [ori1, ori2, ori3] and of directional vector
    D = [dir1, dir2, dir3] is closest to the circle of radius `radius`
    and centered `(0, 0, circ_z)` at the point P = O + kmin * D.
    The variable `norm_dir` is the squared norm of the direction of the ray.
    This is the vectorial version, we expect the directions and origins to be:
    dirs = [dir1_los1, dir2_los1, dir3_los1, dir1_los2,...]
    oris = [ori1_los1, ori2_los1, ori3_los1, ori1_los2,...]
    res = [kmin(los1, cir1), kmin(los1, cir2),...]
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `dist_los_circle_core`
    """
    cdef int i, ind_los, ind_cir
    cdef double* dirv
    cdef double* orig
    cdef double radius, circ_z, norm_dir
    with nogil, parallel():
        dirv = <double*>malloc(3*sizeof(double))
        orig = <double*>malloc(3*sizeof(double))
        for ind_los in prange(num_los):
            for i in range(3):
                dirv[i] = los_directions[ind_los * 3 + i]
                orig[i] = los_origins[ind_los * 3 + i]
            norm_dir = norm_dir_tab[ind_los]
            if norm_dir < 0.:
                norm_dir = Csqrt(compute_dot_prod(dirv, dirv))
            for ind_cir in range(num_cir):
                radius = circle_radius[ind_cir]
                circ_z = circle_z[ind_cir]
                res[ind_los * num_cir
                    + ind_cir] = is_close_los_circle_core(dirv, orig, radius,
                                                          circ_z, norm_dir, eps)
        free(dirv)
        free(orig)
    return

# ==============================================================================
#
#                       DISTANCE BETWEEN LOS AND EXT-POLY
#
# ==============================================================================

def comp_dist_los_vpoly(double[:, ::1] ray_orig,
                        double[:, ::1] ray_vdir,
                        double[:, ::1] ves_poly,
                        double eps_uz=_SMALL, double eps_a=_VSMALL,
                        double eps_vz=_VSMALL, double eps_b=_VSMALL,
                        double eps_plane=_VSMALL, str ves_type='Tor',
                        int num_threads=16):
    """
    This function computes the distance (and the associated k) between num_los
    Rays (or LOS) and an `IN` structure (a polygon extruded around the axis
    (0,0,1), eg. a flux surface).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        ray_orig : (3, num_los) double array
           LOS origin points coordinates
        ray_vdir : (3, num_los) double array
           LOS normalized direction vector
        ves_poly : (2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the Vessel
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        kmin_vpoly : (num_los) double array
            Of the form [k_0, k_1, ..., k_n], where k_i is the coefficient
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (num_los) double array
            `distance[i]` is the distance from P_i to the extruded polygon.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `simple_dist_los_vpoly_core`
    """
    cdef int npts_poly = ves_poly.shape[1]
    cdef int num_los = ray_orig.shape[1]
    cdef int ii, ind_vert, ind_los
    cdef double* res_loc = NULL
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef array dist_vpoly = clone(array('d'), num_los, True)
    cdef array kmin_vpoly = clone(array('d'), num_los, True)
    cdef double[::1] dist_view, kmin_view
    dist_view = dist_vpoly
    kmin_view = kmin_vpoly
    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
        # We use local arrays for each thread so...
        loc_org   = <double *> malloc(sizeof(double) * 3)
        loc_dir   = <double *> malloc(sizeof(double) * 3)
        res_loc = <double *> malloc(2*sizeof(double))
        # == The parallelization over the LOS ==================================
        for ind_los in prange(num_los, schedule='dynamic'):
            loc_org[0] = ray_orig[0, ind_los]
            loc_org[1] = ray_orig[1, ind_los]
            loc_org[2] = ray_orig[2, ind_los]
            loc_dir[0] = ray_vdir[0, ind_los]
            loc_dir[1] = ray_vdir[1, ind_los]
            loc_dir[2] = ray_vdir[2, ind_los]
            # -- Computing values that depend on the LOS/ray -------------------
            upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
            upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
            dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
            invuz = 1./loc_dir[2]
            crit2 = upar2*crit2_base
            simple_dist_los_vpoly_core(loc_org, loc_dir,
                                     &ves_poly[0][0],
                                     &ves_poly[1][0],
                                     npts_poly, upscaDp,
                                     upar2, dpar2,
                                     invuz, crit2,
                                     eps_uz, eps_vz,
                                     eps_a, eps_b,
                                     res_loc)
            dist_view[ind_los] = res_loc[1]
            kmin_view[ind_los] = res_loc[0]
        free(loc_org)
        free(loc_dir)
        free(res_loc)
    return np.asarray(kmin_vpoly), np.asarray(dist_vpoly)


def comp_dist_los_vpoly_vec(int nvpoly, int nlos,
                            np.ndarray[double,ndim=2,mode='c'] ray_orig,
                            np.ndarray[double,ndim=2,mode='c'] ray_vdir,
                            np.ndarray[double,ndim=3,mode='c'] ves_poly,
                            double eps_uz=_SMALL, double eps_a=_VSMALL,
                            double eps_vz=_VSMALL, double eps_b=_VSMALL,
                            double eps_plane=_VSMALL, str ves_type='Tor',
                            str algo_type='simple', int num_threads=16):
    """
    This function computes the distance (and the associated k) between num_los
    Rays (or LOS) and several `IN` structures (polygons extruded around the axis
    (0,0,1), eg. flux surfaces).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, num_los) double array
           LOS origin points coordinates
        ray_vdir : (3, num_los) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces
           WARNING : we suppose all poly are nested in each other,
                     from inner to outer
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        kmin_vpoly : (npoly, num_los) double array
            Of the form [k_00, k_01, ..., k_0n, k_10, k_11, ..., k_1n, ...]
            where k_ij is the coefficient for the j-th flux surface
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (npoly, num_los) double array
            `distance[i * num_poly + j]` is the distance from P_i to the i-th
            extruded poly.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `comp_dist_los_vpoly_vec_core`
    """
    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    cdef array kmin_tab = clone(array('d'), nvpoly*nlos, True)
    cdef array dist_tab = clone(array('d'), nvpoly*nlos, True)
    comp_dist_los_vpoly_vec_core(nvpoly, nlos,
                                 <double*>ray_orig.data,
                                 <double*>ray_vdir.data,
                                 ves_poly,
                                 eps_uz, eps_a,
                                 eps_vz, eps_b,
                                 eps_plane,
                                 ves_type,
                                 algo_type,
                                 kmin_tab, dist_tab,
                                 num_threads)
    return np.asarray(kmin_tab).reshape(nlos, nvpoly),\
        np.asarray(dist_tab).reshape(nlos, nvpoly)


cdef inline void comp_dist_los_vpoly_vec_core(int num_poly, int nlos,
                                              double* ray_orig,
                                              double* ray_vdir,
                                              double[:,:,::1] ves_poly,
                                              double eps_uz,
                                              double eps_a,
                                              double eps_vz,
                                              double eps_b,
                                              double eps_plane,
                                              str ves_type,
                                              str algo_type,
                                              double[::1] res_k,
                                              double[::1] res_dist,
                                              int num_threads=16):
    """
    This function computes the distance (and the associated k) between nlos
    Rays (or LOS) and several `IN` structures (polygons extruded around the axis
    (0,0,1), eg. flux surfaces).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        num_poly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces.
           WARNING : we suppose all poly are nested in each other,
                     from inner to outer
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        kmin_vpoly : (npoly, nlos) double array
            Of the form [k_00, k_01, ..., k_0n, k_10, k_11, ..., k_1n, ...]
            where k_ij is the coefficient for the j-th flux surface
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (npoly, nlos) double array
            `distance[j, i]` is the distance from P_i to the i-th extruded poly.
    ---
    This is the CYTHON function, use only if you need this computation from
    Cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, ind_pol2
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double* lpolyx
    cdef double* lpolyy
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.

    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    # == Defining parallel part ================================================
    with nogil, parallel():
        # We use local arrays for each thread so...
        loc_dir = <double*>malloc(3*sizeof(double))
        loc_org = <double*>malloc(3*sizeof(double))
        loc_res = <double*>malloc(2*sizeof(double))
        # == The parallelization over the LOS ==================================
        for ind_los in prange(nlos, schedule='dynamic'):
            for i in range(3):
                loc_dir[i] = ray_vdir[ind_los * 3 + i]
                loc_org[i] = ray_orig[ind_los * 3 + i]
            # -- Computing values that depend on the LOS/ray -------------------
            upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
            upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
            dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
            invuz = 1./loc_dir[2]
            crit2 = upar2*crit2_base
            # -- Looping over each flux surface---------------------------------
            for ind_pol in range(num_poly):
                npts_poly = ves_poly[ind_pol].shape[1]
                simple_dist_los_vpoly_core(loc_org, loc_dir,
                                           &ves_poly[ind_pol][0][0],
                                           &ves_poly[ind_pol][1][0],
                                           npts_poly, upscaDp,
                                           upar2, dpar2,
                                           invuz, crit2,
                                           eps_uz, eps_vz,
                                           eps_a, eps_b,
                                           loc_res)
                res_k[ind_los * num_poly + ind_pol] = loc_res[0]
                res_dist[ind_los * num_poly + ind_pol] = loc_res[1]
                if not loc_res[1] == loc_res[1] : #is nan
                    for ind_pol2 in range(ind_pol, num_poly):
                        res_k[ind_los * num_poly + ind_pol2] = Cnan
                        res_dist[ind_los * num_poly + ind_pol2] = Cnan
                    continue
        free(loc_dir)
        free(loc_org)
        free(loc_res)
    return


cdef inline void simple_dist_los_vpoly_core(const double[3] ray_orig,
                                            const double[3] ray_vdir,
                                            const double* lpolyx,
                                            const double* lpolyy,
                                            const int nvert,
                                            const double upscaDp,
                                            const double upar2,
                                            const double dpar2,
                                            const double invuz,
                                            const double crit2,
                                            const double eps_uz,
                                            const double eps_vz,
                                            const double eps_a,
                                            const double eps_b,
                                            double* res_final) nogil:
    """
    This function computes the distance (and the associated k) between a Ray
    (or Line Of Sight) and an `IN` structure (a polygon extruded around the axis
    (0,0,1), eg. a flux surface).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        ray_orig : (3) double array
           LOS origin point coordinates, noted often : `u`
        ray_vdir : (3) double array
           LOS normalized direction vector, noted often : `D`
        lpolyx : (num_vertex) double array
           1st coordinates of the vertices of the Polygon defining the poloidal
           cut of the Vessel
        lpolyy : (num_vertex) double array
           2nd coordinates of the vertices of the Polygon defining the poloidal
           cut of the Vessel
        nvert : integer
           number of vertices describing the polygon
        upscaDp : double
           if u = [ux, uy, uz] is the direction of the ray, and D=[dx, dy, dz]
           its origin, then upscaDp = ux*dx + uy*dy
        upar2 : double
           if u = [ux, uy, uz] is the direction of the ray, and D=[dx, dy, dz]
           its origin, then upar2 = ux*ux + uy*uy
        dpar2 : double
           if u = [ux, uy, uz] is the direction of the ray, and D=[dx, dy, dz]
           its origin, then dpar2 = dx*dx + dy*dy
        invuz : double
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        kmin_vpoly : (num_los) double array
            Of the form [k_0, k_1, ..., k_n], where k_i is the coefficient
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (num_los) double array
            `distance[i]` is the distance from P_i to the extruded polygon.
             if the i-th LOS intersects the poly, then distance[i] = Cnan
    ---
    This is the Cython version, only accessible from cython. If you need
    to use it from Python please use: comp_dist_los_vpoly
    """
    cdef int jj
    cdef int indin=0
    cdef int indout=0
    cdef double norm_dir2, norm_dir2_ori
    cdef double radius_z
    cdef double q, coeff, sqd, k
    cdef double v0, v1, val_a, val_b
    cdef double[2] res_a
    cdef double[2] res_b
    cdef double[3] circle_tangent
    cdef double rdotvec
    res_final[0] = 1000000000
    res_final[1] = 1000000000

    # == Compute all solutions =================================================
    # Set tolerance value for ray_vdir[2,ii]
    # eps_uz is the tolerated DZ across 20m (max Tokamak size)
    norm_dir2 = Csqrt(compute_dot_prod(ray_vdir, ray_vdir))
    norm_dir2_ori = norm_dir2
    for jj in range(3):
        ray_vdir[jj] = ray_vdir[jj] / norm_dir2
    norm_dir2 = 1.
    if ray_vdir[2] * ray_vdir[2] < crit2:
        # -- Case with horizontal semi-line ------------------------------------
        for jj in range(nvert-1):
            if (lpolyy[jj+1] - lpolyy[jj])**2 > eps_vz * eps_vz:
                # If segment AB is NOT horizontal, then we can compute distance
                # between LOS and cone.
                # First we compute the "circle" on the cone that lives on the
                # same plane as the line
                q = (ray_orig[2] - lpolyy[jj]) / (lpolyy[jj+1] - lpolyy[jj])
                if q < 0. :
                    # Then we only need to compute distance to circle C_A
                    dist_los_circle_core(ray_vdir, ray_orig,
                                        lpolyx[jj], lpolyy[jj],
                                        norm_dir2, res_a)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_a)
                else:
                    # The we need to compute the radius (the height is Z_D)
                    # of the circle in the same plane as the LOS and compute the
                    # distance between the LOS and circle.
                    radius_z = q * (lpolyx[jj+1] - lpolyx[jj]) + lpolyx[jj]
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         radius_z, ray_orig[2],
                                         norm_dir2, res_a)
                    if res_a[1] < _VSMALL:
                        # The line is either tangent or intersects the frustum
                        # we need to make the difference
                        k = res_a[0]
                        # we compute the ray from circle center to P
                        circle_tangent[0] = -ray_orig[0] - k * ray_vdir[0]
                        circle_tangent[1] = -ray_orig[1] - k * ray_vdir[1]
                        circle_tangent[2] = 0. # the line is horizontal
                        rdotvec = compute_dot_prod(circle_tangent, ray_vdir)
                        if Cabs(rdotvec) > _VSMALL:
                            # There is an intersection, distance = Cnan
                            res_final[1] = Cnan # distance
                            res_final[0] = Cnan # k
                            # no need to continue
                            return
                if (res_final[1] > res_a[1]
                    or (res_final[1] == res_a[1] and res_final[0] > res_a[0])):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance
            else:
                # -- case with horizontal cone (aka cone is a plane annulus) ---
                # Then the shortest distance is the distance to the
                # outline circles
                # computing distance to cricle C_A of radius R_A and height Z_A
                dist_los_circle_core(ray_vdir, ray_orig,
                                     lpolyx[jj], lpolyy[jj],
                                     norm_dir2, res_a)
                if res_a[1] < _VSMALL:
                    # The line is either tangent or intersects the frustum
                    # we need to make the difference
                    k = res_a[0]
                    # we compute the ray from circle center to P
                    circle_tangent[0] = -ray_orig[0] - k * ray_vdir[0]
                    circle_tangent[1] = -ray_orig[1] - k * ray_vdir[1]
                    circle_tangent[2] = 0. # the ray is horizontal
                    rdotvec = compute_dot_prod(circle_tangent, ray_vdir)
                    if Cabs(rdotvec) > _VSMALL:
                        # There is an intersection, distance = Cnan
                        res_final[1] = Cnan # distance
                        res_final[0] = Cnan # k
                        # no need to continue
                        return
                dist_los_circle_core(ray_vdir, ray_orig,
                                     lpolyx[jj+1], lpolyy[jj+1],
                                     norm_dir2, res_b)
                if res_b[1] < _VSMALL:
                    # The line is either tangent or intersects the frustum
                    # we need to make the difference
                    k = res_b[0]
                    # we compute the ray from circle center to P
                    circle_tangent[0] = -ray_orig[0] - k * ray_vdir[0]
                    circle_tangent[1] = -ray_orig[1] - k * ray_vdir[1]
                    circle_tangent[2] = 0. # the ray is horizontal
                    rdotvec = compute_dot_prod(circle_tangent, ray_vdir)
                    if Cabs(rdotvec) > _VSMALL:
                        # There is an intersection, distance = Cnan
                        res_final[1] = Cnan # distance
                        res_final[0] = Cnan # k
                        # no need to continue
                        return
                # The result is the one associated to the shortest distance
                if (res_final[1] > res_a[1] or
                    (res_final[1] == res_a[1] and res_final[0] > res_a[0])):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance
                if (res_final[1] > res_b[1] or
                    (res_final[1] == res_b[1] and res_final[0] > res_b[0])):
                    res_final[0] = res_b[0] # k
                    res_final[1] = res_b[1] # distance
    else:
        # == More general non-horizontal semi-line case ========================
        for jj in range(nvert-1):
            v0 = lpolyx[jj+1]-lpolyx[jj]
            v1 = lpolyy[jj+1]-lpolyy[jj]
            val_a = v0 * v0 - upar2 * v1 * invuz * v1 * invuz
            val_b = lpolyx[jj] * v0 + v1 * (ray_orig[2] - lpolyy[jj]) * upar2 *\
                    invuz * invuz - upscaDp * v1 * invuz
            coeff = - upar2 * (ray_orig[2] - lpolyy[jj])**2 * invuz * invuz +\
                    2. * upscaDp * (ray_orig[2]-lpolyy[jj]) * invuz -\
                    dpar2 + lpolyx[jj] * lpolyx[jj]
            if (val_a * val_a < eps_a * eps_a):
                if (val_b * val_b < eps_b * eps_b):
                    # let's see if C is 0 or not
                    if coeff * coeff < eps_a * eps_a :
                        # then LOS included in cone and then we can choose point
                        # such that q = 0,  k = (z_A - z_D) / uz
                        res_a[0] = (lpolyy[jj] - ray_orig[2]) * invuz
                        res_a[1] = 0 # distance = 0 since LOS in cone
                else: # (val_b * val_b > eps_b * eps_b):
                    q = -coeff / (2. * val_b)
                    if q < 0. :
                        # Then we only need to compute distance to circle C_A
                        dist_los_circle_core(ray_vdir, ray_orig,
                                            lpolyx[jj], lpolyy[jj],
                                            norm_dir2, res_a)
                    elif q > 1:
                        # Then we only need to compute distance to circle C_B
                        dist_los_circle_core(ray_vdir, ray_orig,
                                            lpolyx[jj+1], lpolyy[jj+1],
                                             norm_dir2, res_a)
                    else :
                        k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                        if k >= 0.:
                            # Then there is an intersection
                            res_final[0] = Cnan
                            res_final[1] = Cnan
                            return # no need to move forward
                        else:
                            # The closest point on the line is the LOS origin
                            res_a[0] = 0
                            res_a[1] = -k * Csqrt(norm_dir2)
                if (res_final[1] > res_a[1]
                    or (res_final[1] == res_a[1] and res_final[0] > res_a[0])):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance
            elif (val_b * val_b >= val_a * coeff):
                sqd = Csqrt(val_b * val_b - val_a * coeff)
                # First solution
                q = (-val_b + sqd) / val_a
                if q < 0:
                    # Then we only need to compute distance to circle C_A
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         lpolyx[jj], lpolyy[jj],
                                         norm_dir2, res_a)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_a)
                else :
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k >= 0.:
                        # There is an intersection
                        res_final[0] = Cnan
                        res_final[1] = Cnan
                        return # no need to continue
                    else:
                        # The closest point on the LOS is its origin
                        res_a[0] = 0
                        res_a[1] = -k * Csqrt(norm_dir2)
                if (res_final[1] > res_a[1]
                    or (res_final[1] == res_a[1] and res_final[0] > res_a[0])):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance
                # Second solution
                q = (-val_b - sqd) / val_a
                if q < 0:
                    # Then we only need to compute distance to circle C_A
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         lpolyx[jj], lpolyy[jj],
                                         norm_dir2, res_b)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_b)
                else:
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k>=0.:
                        # there is an intersection
                        res_final[0] = Cnan
                        res_final[1] = Cnan
                        return # no need to continue
                    else:
                        # The closest point on the LOS is its origin
                        res_b[0] = 0
                        res_b[1] = -k * Csqrt(norm_dir2)
                if (res_final[1] > res_b[1]
                    or (res_final[1] == res_b[1] and res_final[0] > res_b[0])):
                    res_final[0] = res_b[0]
                    res_final[1] = res_b[1]
    res_final[0] = res_final[0] / norm_dir2_ori
    return


# ==============================================================================
#
#                         ARE LOS AND EXT-POLY CLOSE
#
# ==============================================================================

def is_close_los_vpoly_vec(int nvpoly, int nlos,
                           np.ndarray[double,ndim=2,mode='c'] ray_orig,
                           np.ndarray[double,ndim=2,mode='c'] ray_vdir,
                           np.ndarray[double,ndim=3,mode='c'] ves_poly,
                           double epsilon,
                           double eps_uz=_SMALL, double eps_a=_VSMALL,
                           double eps_vz=_VSMALL, double eps_b=_VSMALL,
                           double eps_plane=_VSMALL, str ves_type='Tor',
                           str algo_type='simple', int num_threads=16):
    """
    This function tests if the distance between num_los Rays (or LOS) and
    several `IN` structures (polygons extruded around the axis (0,0,1),
    eg. flux surfaces) is smaller than `epsilon`.
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, num_los) double array
           LOS origin points coordinates
        ray_vdir : (3, num_los) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        epsilon : double
           Value for testing if distance < epsilon
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        are_close : (npoly * num_los) bool array
            `are_close[i * num_poly + j]` indicates if distance between i-th LOS
            and j-th poly are closer than epsilon. (True if distance<epsilon)
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `is_close_los_vpoly_vec_core`
    """
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    cdef array are_close = clone(array('i'), nvpoly*nlos, True)
    is_close_los_vpoly_vec_core(nvpoly, nlos,
                                <double*>ray_orig.data,
                                <double*>ray_vdir.data,
                                ves_poly,
                                eps_uz, eps_a,
                                eps_vz, eps_b,
                                eps_plane,
                                ves_type,
                                algo_type,
                                epsilon,
                                are_close,
                                num_threads)
    return np.asarray(are_close, dtype=bool).reshape(nlos, nvpoly)


cdef inline void is_close_los_vpoly_vec_core(int num_poly, int nlos,
                                             double* ray_orig,
                                             double* ray_vdir,
                                             double[:,:,::1] ves_poly,
                                             double eps_uz,
                                             double eps_a,
                                             double eps_vz,
                                             double eps_b,
                                             double eps_plane,
                                             str ves_type,
                                             str algo_type,
                                             double epsilon,
                                             int[::1] are_close,
                                             int num_threads=16):
    """
    This function computes the distance (and the associated k) between nlos
    Rays (or LOS) and several `IN` structures (polygons extruded around the axis
    (0,0,1), eg. flux surfaces).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        num_poly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces.
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        epsilon : double
           Value for testing if distance < epsilon
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        are_close : (npoly * num_los) bool array
            `are_close[i * num_poly + j]` indicates if distance between i-th LOS
            and j-th poly are closer than epsilon. (True if distance<epsilon)
    ---
    This is the CYTHON function, use only if you need this computation from
    Cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, ind_pol2
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double* lpolyx
    cdef double* lpolyy
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.

    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    # == Defining parallel part ================================================
    with nogil, parallel():
        # We use local arrays for each thread so...
        loc_dir = <double*>malloc(3*sizeof(double))
        loc_org = <double*>malloc(3*sizeof(double))
        loc_res = <double*>malloc(2*sizeof(double))
        # == The parallelization over the LOS ==================================
        for ind_los in prange(nlos, schedule='dynamic'):
            for i in range(3):
                loc_dir[i] = ray_vdir[ind_los * 3 + i]
                loc_org[i] = ray_orig[ind_los * 3 + i]
            # -- Computing values that depend on the LOS/ray -------------------
            upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
            upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
            dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
            invuz = 1./loc_dir[2]
            crit2 = upar2*crit2_base
            # -- Looping over each flux surface---------------------------------
            for ind_pol in range(num_poly):
                npts_poly = ves_poly[ind_pol].shape[1]
                simple_dist_los_vpoly_core(loc_org, loc_dir,
                                           &ves_poly[ind_pol][0][0],
                                           &ves_poly[ind_pol][1][0],
                                           npts_poly, upscaDp,
                                           upar2, dpar2,
                                           invuz, crit2,
                                           eps_uz, eps_vz,
                                           eps_a, eps_b,
                                           loc_res)
                if loc_res[1] < epsilon:
                    are_close[ind_los * num_poly + ind_pol] = 1
                elif loc_res[1] == loc_res[1]: # is nan
                    continue
        free(loc_dir)
        free(loc_org)
        free(loc_res)
    return



# ==============================================================================
#
#                         WHICH LOS/VPOLY IS CLOSER
#
# ==============================================================================

def which_los_closer_vpoly_vec(int nvpoly, int nlos,
                               np.ndarray[double,ndim=2,mode='c'] ray_orig,
                               np.ndarray[double,ndim=2,mode='c'] ray_vdir,
                               np.ndarray[double,ndim=3,mode='c'] ves_poly,
                               double eps_uz=_SMALL, double eps_a=_VSMALL,
                               double eps_vz=_VSMALL, double eps_b=_VSMALL,
                               double eps_plane=_VSMALL, str ves_type='Tor',
                               str algo_type='simple', int num_threads=16):
    """
    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, num_los) double array
           LOS origin points coordinates
        ray_vdir : (3, num_los) double array
           LOS direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        ind_close_los : (npoly) int array
            Of the form [ind_0, ind_1, ..., ind_(npoly-1)]
            where ind_i is the coefficient for the i-th flux surface
            such that the ind_i-th ray (LOS) is closest to the extruded polygon
            among all other LOS without going over it.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `which_los_closer_vpoly_vec_core`
    """
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    cdef array ind_close_tab = clone(array('i'), nvpoly, True)
    which_los_closer_vpoly_vec_core(nvpoly, nlos,
                                    <double*>ray_orig.data,
                                    <double*>ray_vdir.data,
                                    ves_poly,
                                    eps_uz, eps_a,
                                    eps_vz, eps_b,
                                    eps_plane,
                                    ves_type,
                                    algo_type,
                                    ind_close_tab,
                                    num_threads)
    return np.asarray(ind_close_tab)


cdef inline void which_los_closer_vpoly_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 str ves_type,
                                                 str algo_type,
                                                 int[::1] ind_close_tab,
                                                 int num_threads=16):
    """
    Params
    ======
        num_poly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces.
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        ind_close_tab : (npoly) int array
            Of the form [ind_0, ind_1, ..., ind_(npoly-1)]
            where ind_i is the coefficient for the i-th flux surface
            such that the ind_i-th ray (LOS) is closest to the extruded polygon
            among all other LOS without going over it.
    ---
    This is the CYTHON function, use only if you need this computation from
    Cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, ind_pol2, indloc
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double loc_dist
    cdef double* lpolyx
    cdef double* lpolyy
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef array kmin_tab = clone(array('d'), num_poly*nlos, True)
    cdef array dist_tab = clone(array('d'), num_poly*nlos, True)

    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    for indloc in range(num_poly):
        ind_close_tab[indloc] = -1
    comp_dist_los_vpoly_vec_core(num_poly, nlos,
                                 ray_orig,
                                 ray_vdir,
                                 ves_poly,
                                 eps_uz, eps_a,
                                 eps_vz, eps_b,
                                 eps_plane,
                                 ves_type,
                                 algo_type,
                                 kmin_tab, dist_tab,
                                 num_threads)

    # We use local arrays for each thread so...
    for ind_pol in range(num_poly):
        loc_dist = 100000000.
        for ind_los in range(nlos):
            if (dist_tab[ind_los*num_poly + ind_pol] < loc_dist):
                ind_close_tab[ind_pol] = ind_los
                loc_dist = dist_tab[ind_los*num_poly + ind_pol]

    return




def which_vpoly_closer_los_vec(int nvpoly, int nlos,
                               np.ndarray[double,ndim=2,mode='c'] ray_orig,
                               np.ndarray[double,ndim=2,mode='c'] ray_vdir,
                               np.ndarray[double,ndim=3,mode='c'] ves_poly,
                               double eps_uz=_SMALL, double eps_a=_VSMALL,
                               double eps_vz=_VSMALL, double eps_b=_VSMALL,
                               double eps_plane=_VSMALL, str ves_type='Tor',
                               str algo_type='simple', int num_threads=16):
    """
    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, num_los) double array
           LOS origin points coordinates
        ray_vdir : (3, num_los) double array
           LOS direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        ind_close_los : (nlos) int array
            Of the form [ind_0, ind_1, ..., ind_(nlos-1)]
            where ind_i is the coefficient for the i-th LOS (ray)
            such that the ind_i-th poly (flux surface) is closest to the LOS
            among all other poly without going over it.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from Cython, use `which_vpoly_closer_los_vec_core`
    """
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    cdef array ind_close_tab = clone(array('i'), nlos, True)
    which_vpoly_closer_los_vec_core(nvpoly, nlos,
                                    <double*>ray_orig.data,
                                    <double*>ray_vdir.data,
                                    ves_poly,
                                    eps_uz, eps_a,
                                    eps_vz, eps_b,
                                    eps_plane,
                                    ves_type,
                                    algo_type,
                                    ind_close_tab,
                                    num_threads)
    return np.asarray(ind_close_tab)


cdef inline void which_vpoly_closer_los_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 str ves_type,
                                                 str algo_type,
                                                 int[::1] ind_close_tab,
                                                 int num_threads=16):
    """
    Params
    ======
        num_poly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
           LOS normalized direction vector
        ves_poly : (num_pol, 2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the different IN surfaces.
           WARNING : we suppose all poly are nested in each other,
                     and the first one is the smallest one
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        ind_close_los : (nlos) int array
            Of the form [ind_0, ind_1, ..., ind_(nlos-1)]
            where ind_i is the coefficient for the i-th LOS (ray)
            such that the ind_i-th poly (flux surface) is closest to the LOS
            among all other poly without going over it.
    ---
    This is the CYTHON function, use only if you need this computation from
    Cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, ind_pol2, indloc
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double* lpolyx
    cdef double* lpolyy
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.

    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    from warnings import warn
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    # initialization ...............................................
    for indloc in range(nlos):
        ind_close_tab[indloc] = num_poly-1

    # == Defining parallel part ================================================
    with nogil, parallel():
        # We use local arrays for each thread so...
        loc_dir = <double*>malloc(3*sizeof(double))
        loc_org = <double*>malloc(3*sizeof(double))
        loc_res = <double*>malloc(2*sizeof(double))
        # == The parallelization over the LOS ==================================
        for ind_los in prange(nlos, schedule='dynamic'):
            for i in range(3):
                loc_dir[i] = ray_vdir[ind_los * 3 + i]
                loc_org[i] = ray_orig[ind_los * 3 + i]
            # -- Computing values that depend on the LOS/ray -------------------
            upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
            upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
            dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
            invuz = 1./loc_dir[2]
            crit2 = upar2*crit2_base
            # -- Looping over each flux surface---------------------------------
            for ind_pol in range(num_poly):
                npts_poly = ves_poly[ind_pol].shape[1]
                simple_dist_los_vpoly_core(loc_org, loc_dir,
                                           &ves_poly[ind_pol][0][0],
                                           &ves_poly[ind_pol][1][0],
                                           npts_poly, upscaDp,
                                           upar2, dpar2,
                                           invuz, crit2,
                                           eps_uz, eps_vz,
                                           eps_a, eps_b,
                                           loc_res)
                # filling the array when nan found .............................
                if not loc_res[1] == loc_res[1]:
                    #the closer poly is the one just before
                    ind_close_tab[ind_los] = ind_pol-1
                    continue
        free(loc_dir)
        free(loc_org)
        free(loc_res)
    return



"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++ OLD FUNCTIONS CEMETRY ++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


# deprecated version !!!!!!!!!!!!! TO ERASE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
def SLOW_LOS_Calc_PInOut_VesStruct(Ds, dus,
                              np.ndarray[double, ndim=2,mode='c'] VPoly,
                              np.ndarray[double, ndim=2,mode='c'] VIn,
                              Lim=None, nLim=None,
                              LSPoly=None, LSLim=None, lSnLim=None, LSVIn=None,
                              RMin=None, Forbid=True,
                              EpsUz=_SMALL, EpsVz=_VSMALL, EpsA=_VSMALL,
                              EpsB=_VSMALL, EpsPlane=_VSMALL,
                              VType='Tor', Test=True):

    from warnings import warn
    warn("THIS IS THE OLD VERSION OF THIS FUNCTION, PLEASE USE THE NEW ONE",
         DeprecationWarning, stacklevel=2)
    warn("THIS IS THE OLD VERSION OF THIS FUNCTION, PLEASE USE THE NEW ONE",
         Warning)
    """ Compute the entry and exit point of all provided LOS for the provided
    vessel polygon (toroidal or linear), also return the normal vector at
    impact point and the index of the impact segment

    For each LOS,

    Parameters
    ----------



    Return
    ------
    PIn :       np.ndarray
        Point of entry (if any) of the LOS into the vessel, returned in (X,Y,Z)
        cartesian coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    POut :      np.ndarray
        Point of exit of the LOS from the vessel, returned in (X,Y,Z) cartesian
        coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    VOut :      np.ndarray

    IOut :      np.ndarray

    """
    if Test:
        assert type(Ds) is np.ndarray and type(dus) is np.ndarray and \
            Ds.ndim in [1,2] and Ds.shape==dus.shape and \
            Ds.shape[0]==3, (
                "Args Ds and dus must be of the same shape (3,) or (3,NL)!")
        assert VPoly.shape[0]==2 and VIn.shape[0]==2 and \
            VIn.shape[1]==VPoly.shape[1]-1, (
                "Args VPoly and VIn must be of the same shape (2,NS)!")
        C1 = all([pp is None for pp in [LSPoly,LSLim,LSVIn]])
        C2 = all([hasattr(pp,'__iter__') and len(pp)==len(LSPoly) for pp
                  in [LSPoly,LSLim,LSVIn]])
        assert C1 or C2, "Args LSPoly,LSLim,LSVIn must be None or lists of same len()!"
        assert RMin is None or type(RMin) in [float,int,np.float64,np.int64], (
            "Arg RMin must be None or a float!")
        assert type(Forbid) is bool, "Arg Forbid must be a bool!"
        assert all([type(ee) in [int,float,np.int64,np.float64] and ee<1.e-4
                    for ee in [EpsUz,EpsVz,EpsA,EpsB,EpsPlane]]), \
                        "Args [EpsUz,EpsVz,EpsA,EpsB] must be floats < 1.e-4!"
        assert type(VType) is str and VType.lower() in ['tor','lin'], (
            "Arg VType must be a str in ['Tor','Lin']!")

    cdef int ii, jj

    print("\n ---- > Using the WRONG one !!!!!!!\n")
    if nLim==0:
        Lim = None
    elif nLim==1:
        Lim = [Lim[0,0],Lim[0,1]]
    if lSnLim is not None:
        for ii in range(0,len(lSnLim)):
            if lSnLim[ii]==0:
                LSLim[ii] = None
            elif lSnLim[ii]==1:
                LSLim[ii] = [LSLim[ii][0,0],LSLim[ii][0,1]]

    v = Ds.ndim==2
    if not v:
        Ds, dus = Ds.reshape((3,1)), dus.reshape((3,1))
    NL = Ds.shape[1]
    IOut = np.zeros((3,Ds.shape[1]))
    if VType.lower()=='tor':
        # RMin is necessary to avoid looking on the other side of the tokamak
        if RMin is None:
            RMin = 0.95*min(np.min(VPoly[0,:]),
                            np.min(np.hypot(Ds[0,:],Ds[1,:])))

        # Main function to compute intersections with Vessel
        PIn, POut, \
            VperpIn, VperpOut, \
            IIn, IOut[2,:] = Calc_LOS_PInOut_Tor(Ds, dus, VPoly, VIn, Lim=Lim,
                                                 Forbid=Forbid, RMin=RMin,
                                                 EpsUz=EpsUz, EpsVz=EpsVz,
                                                 EpsA=EpsA, EpsB=EpsB,
                                                 EpsPlane=EpsPlane)

        # k = coordinate (in m) along the line from D
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        kPIn = np.sqrt(np.sum((PIn-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        assert np.allclose(kPIn,np.sum((PIn-Ds)*dus,axis=0),equal_nan=True)

        # If there are Struct, call the same function
        # Structural optimzation : do everything in one big for loop and only
        # keep the relevant points (to save memory)
        if LSPoly is not None:
            Ind = np.zeros((2,NL))
            for ii in range(0,len(LSPoly)):
                if LSLim[ii] is None or not all([hasattr(ll,'__iter__') for ll in LSLim[ii]]):
                    lslim = [LSLim[ii]]
                else:
                    lslim = LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn, pOut,\
                        vperpIn, vperpOut,\
                        iIn, iOut = Calc_LOS_PInOut_Tor(Ds, dus, LSPoly[ii],
                                                        LSVIn[ii], Lim=lslim[jj],
                                                        Forbid=Forbid, RMin=RMin,
                                                        EpsUz=EpsUz, EpsVz=EpsVz,
                                                        EpsA=EpsA, EpsB=EpsB,
                                                        EpsPlane=EpsPlane)
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
        PIn, POut, \
            VperpIn, VperpOut, \
            IIn, IOut[2,:] = Calc_LOS_PInOut_Lin(Ds, dus, VPoly, VIn,
                                                 Lim, EpsPlane=EpsPlane)
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        kPIn = np.sqrt(np.sum((PIn-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        assert np.allclose(kPIn,np.sum((PIn-Ds)*dus,axis=0),equal_nan=True)
        if LSPoly is not None:
            Ind = np.zeros((2,NL))
            for ii in range(0,len(LSPoly)):
                lslim = [LSLim[ii]] if not all([hasattr(ll,'__iter__')
                                                for ll in LSLim[ii]]) \
                                                    else LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn, pOut, \
                        vperpIn, vperpOut, \
                        iIn, iOut = Calc_LOS_PInOut_Lin(Ds, dus, LSPoly[ii],
                                                        LSVIn[ii], lslim[jj],
                                                        EpsPlane=EpsPlane)
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
        PIn, POut, \
            kPIn, kPOut, \
            VperpIn, VperpOut, \
            IIn, IOut = PIn.flatten(), POut.flatten(), kPIn[0], kPOut[0], \
                        VperpIn.flatten(), VperpOut.flatten(), IIn[0], \
                        IOut.flatten()
    return PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Lin(double[:,::1] Ds, double [:,::1] us, double[:,::1] VPoly, double[:,::1] VIn, Lim, double EpsPlane=1.e-9):

    from warnings import warn
    warn("THIS IS THE OLD VERSION OF THIS FUNCTION, PLEASE USE THE NEW ONE",
         DeprecationWarning, stacklevel=2)

    cdef int ii=0, jj=0, Nl=Ds.shape[1], Ns=VIn.shape[1]
    cdef double kin, kout, scauVin, q, X, sca, L0=<double>Lim[0], L1=<double>Lim[1]
    cdef int indin=0, indout=0, Done=0
    cdef np.ndarray[double,ndim=2] SIn_=np.nan*np.ones((3,Nl)), SOut_=np.nan*np.ones((3,Nl))
    cdef np.ndarray[double,ndim=2] VPerp_In=np.nan*np.ones((3,Nl)), VPerp_Out=np.nan*np.ones((3,Nl))
    cdef np.ndarray[double,ndim=1] indIn_=np.nan*np.ones((Nl,)), indOut_=np.nan*np.ones((Nl,))

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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Tor(double [:,::1] Ds, double [:,::1] us, double [:,::1] VPoly, double [:,::1] vIn, Lim=None,
                         bool Forbid=True, RMin=None, double EpsUz=1.e-6, double EpsVz=1.e-9, double EpsA=1.e-9, double EpsB=1.e-9, double EpsPlane=1.e-9):
    from warnings import warn
    warn("THIS IS THE OLD VERSION OF THIS FUNCTION, PLEASE USE THE NEW ONE",
         DeprecationWarning, stacklevel=2)

    cdef int ii, jj, Nl=Ds.shape[1], Ns=vIn.shape[1]
    cdef double Rmin, upscaDp, upar2, Dpar2, Crit2, kout, kin
    cdef int indin=0, indout=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0., L0=0., L1=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef int Forbidbis, Forbid0
    cdef np.ndarray[double,ndim=2] SIn_=np.nan*np.ones((3,Nl)), SOut_=np.nan*np.ones((3,Nl))
    cdef np.ndarray[double,ndim=2] VPerp_In=np.nan*np.ones((3,Nl)), VPerp_Out=np.nan*np.ones((3,Nl))
    cdef np.ndarray[double,ndim=1] indIn_=np.nan*np.ones((Nl,)), indOut_=np.nan*np.ones((Nl,))

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
