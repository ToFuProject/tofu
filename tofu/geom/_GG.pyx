# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#
# -- Python libraries imports --------------------------------------------------
from warnings import warn
import numpy as np
import scipy.integrate as scpintg
from matplotlib.path import Path

# -- cython libraries imports --------------------------------------------------
from cpython cimport bool
from cpython.array cimport array, clone
from cython.parallel import prange
from cython.parallel cimport parallel
# -- C libraries imports -------------------------------------------------------
cimport cython
cimport numpy as np
from libc.math cimport sqrt as c_sqrt, ceil as c_ceil, fabs as c_abs
from libc.math cimport floor as c_floor, round as c_round
from libc.math cimport cos as c_cos, sin as c_sin, asin as c_asin
from libc.math cimport atan2 as c_atan2, pi as c_pi
from libc.math cimport NAN as C_NAN
from libc.math cimport INFINITY as C_INF
from libc.stdlib cimport malloc, free
# -- ToFu library imports ------------------------------------------------------
from ._basic_geom_tools cimport _VSMALL, _SMALL
from ._basic_geom_tools cimport _TWOPI
from . cimport _basic_geom_tools as _bgt
from . cimport _raytracing_tools as _rt
from . cimport _distance_tools as _dt
from . cimport _sampling_tools as _st
from . cimport _vignetting_tools as _vt
from . import _openmp_tools as _ompt

# == Exports ===================================================================
__all__ = ['coord_shift',
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
           'Poly_isClockwise', 'Poly_VolAngTor',
           'poly_area', "poly_area_and_barycenter",
           'Sino_ImpactEnv', 'ConvertImpact_Theta2Xi',
           '_Ves_isInside',
           'discretize_line1d',
           'discretize_segment2d', '_Ves_meshCross_FromInd',
           'discretize_vpoly',
           '_Ves_Vmesh_Tor_SubFromD_cython',
           '_Ves_Vmesh_Tor_SubFromInd_cython',
           '_Ves_Vmesh_Tor_SubFromD_cython_old',
           '_Ves_Vmesh_Tor_SubFromInd_cython_old',
           '_Ves_Vmesh_Lin_SubFromD_cython', '_Ves_Vmesh_Lin_SubFromInd_cython',
           '_Ves_Smesh_Tor_SubFromD_cython', '_Ves_Smesh_Tor_SubFromInd_cython',
           '_Ves_Smesh_TorStruct_SubFromD_cython',
           '_Ves_Smesh_TorStruct_SubFromInd_cython',
           '_Ves_Smesh_Lin_SubFromD_cython',
           '_Ves_Smesh_Lin_SubFromInd_cython',
           'LOS_Calc_PInOut_VesStruct',
           "LOS_Calc_kMinkMax_VesStruct",
           "LOS_isVis_PtFromPts_VesStruct",
           "LOS_areVis_PtsFromPts_VesStruct",
           'LOS_get_sample', 'LOS_calc_signal',
           'LOS_sino', 'integrate1d',
           "triangulate_by_earclipping",
           "vignetting",
           "Dust_calc_SolidAngle"]


########################################################
########################################################
#       Coordinates handling
########################################################

def coord_shift(points, in_format='(X,Y,Z)',
                out_format='(R,Z)',
                cross_format=None):
    """ Check the shape of an array of points coordinates and/or converts from
    2D to 3D, 3D to 2D, cylindrical to cartesian...
    (CrossRef is an angle (Tor) or a distance (X for Lin))
    """
    cdef str str_ii
    cdef long ncoords = points.shape[0]
    cdef long npts
    assert all([type(ff) is str and ',' in ff
                for ff in [in_format, out_format]]), (
                        "Arg In and Out (coordinate format) "
                        + "must be comma-separated  !")
    assert type(points) is np.ndarray, "Points must be a np.ndarray "
    assert points.ndim in [1, 2], "Points must be a 1D or 2D np.ndarray"
    assert ncoords in (2,3), ("Points must be a 1D or 2D np.ndarray "
                              "of 2 or 3 coordinates !")
    ok_types = [int,float,np.int64,np.float64]
    assert cross_format is None or type(cross_format) in ok_types, (
        "Arg CrossRef must be a float !")

    # Pre-format inputs
    in_format, out_format = in_format.lower(), out_format.lower()

    # Get order
    ins = in_format.replace('(', '').replace(')', '').split(',')
    outs = out_format.replace('(', '').replace(')', '').split(',')

    assert all([ss in ['x','y','z','r','phi'] for ss in ins]), "Non-valid In!"
    assert all([ss in ['x','y','z','r','phi'] for ss in outs]), "Non-valid Out!"
    in_type = 'cyl' if any([ss in ins for ss in ['r','phi']]) else 'cart'
    out_type = 'cyl' if any([ss in outs for ss in ['r','phi']]) else 'cart'

    ndim = points.ndim
    if ndim==1:
        points = np.copy(points.reshape((ncoords, 1)))

    # Compute
    if in_type==out_type:
        assert all([ss in ins for ss in outs])
        pts = []
        for str_ii in outs:
            if str_ii=='phi':
                pts.append(np.arctan2(np.sin(points[ins.index(str_ii), :]),
                                      np.cos(points[ins.index(str_ii), :])))
            else:
                pts.append(points[ins.index(str_ii), :])
    elif in_type=='cart':
        pts = []
        for str_ii in outs:
            if str_ii=='r':
                assert all([ss in ins for ss in ['x','y']])
                pts.append(_bgt.compute_hypot(points[ins.index('x'), :],
                                              points[ins.index('y'), :]))
            elif str_ii=='z':
                assert 'z' in ins
                pts.append(points[ins.index('z'), :])
            elif str_ii=='phi':
                if all([ss in ins for ss in ['x','y']]):
                    pts.append(np.arctan2(points[ins.index('y'), :],
                                          points[ins.index('x'), :]))
                elif cross_format is not None:
                    npts = points.shape[1]
                    pts.append(cross_format * np.ones((npts,)))
                else:
                    raise Exception("There is no phi value available !")
    else:
        pts = []
        for str_ii in outs:
            if str_ii=='x':
                if all([ss in ins for ss in ['r','phi']]) :
                    pts.append(points[ins.index('r'), :] *
                               np.cos(points[ins.index('phi'), :]))
                elif cross_format is not None:
                    npts = points.shape[1]
                    pts.append(cross_format * np.ones((npts,)))
                else:
                    raise Exception("There is no x value available !")
            elif str_ii=='y':
                assert all([ss in ins for ss in ['r','phi']])
                pts.append(points[ins.index('r'), :] *
                           np.sin(points[ins.index('phi'), :]))
            elif str_ii=='z':
                assert 'z' in ins
                pts.append(points[ins.index('z'), :])

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


def Poly_isClockwise(np.ndarray[double,ndim=2] Poly):
    """ Assuming 2D closed Poly !
    http://www.faqs.org/faqs/graphics/algorithms-faq/
    Find the lowest vertex (or, if there is more than one vertex with
    the same lowest coordinate, the rightmost of those vertices) and then
    take the cross product of the edges before and after it.
    Both methods are O(n) for n vertices, but it does seem a waste to add up
    the total area when a single cross product (of just the right edges)
    suffices.  Code for this is available at
    ftp://cs.smith.edu/pub/code/polyorient.C (2K).
    """
    cdef double res
    cdef double[:,::1] mv_poly = np.ascontiguousarray(Poly)
    cdef int npts = mv_poly.shape[1]
    cdef int ndim = mv_poly.shape[0]
    cdef double[::1] mvx
    cdef double[::1] mvy
    cdef int idmin
    cdef int idm1
    cdef int idp1
    cdef str err_msg = ""
    # Checking that Poly wasn't given in the shape (npts, ndim)
    if ndim > npts:
        mv_poly = np.ascontiguousarray(Poly.T)
        npts = mv_poly.shape[1]
        ndim = mv_poly.shape[0]
    mvx = mv_poly[0,:]
    mvy = mv_poly[1,:]
    # Getting index of lower right corner and its neighbors
    idmin = _bgt.find_ind_lowerright_corner(mvx, mvy, npts)
    idm1 = idmin - 1
    idp1 = (idmin + 1) % npts
    if idmin == 0 :
        idm1 = npts - 2
    # Computing area of lower right triangle
    res = mvx[idm1]  * (mvy[idmin] - mvy[idp1]) + \
          mvx[idmin] * (mvy[idp1]  - mvy[idm1]) + \
          mvx[idp1]  * (mvy[idm1]  - mvy[idmin])
    if abs(res) < _VSMALL:
        err_msg += ("In Poly_isClockwise : \n"
                    + "   Found lowest right point at index : "
                    + str(idmin)
                    + ", of coordinates :" + str(mvx[idmin])
                    + ", " + str(mvy[idmin]) + ".\n"
                    + "   The two neighboring points are : "
                    + str(idm1) + " and " + str(idp1) + ".")
        raise Exception(err_msg) # not working
    return res < 0.


def format_poly(np.ndarray[double,ndim=2] poly, str order='C', Clock=False,
               close=True, Test=True):
    """
    Return a polygon poly as a np.ndarray formatted according to parameters

    Parameters
    ----------
        poly    np.ndarray or list    Input np.ndarray of shape (cc,N)
                or tuple              (where cc = 2 or 3, the number of
                                      coordinates and N points), or
                                      list or tuple of vertices of a polygon
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
        Test    bool                  Flag indicating whether the inputs should
                                      be tested for conformity, default: True

    Returns
    -------
        poly    np.ndarray            Output formatted polygon
    """
    cdef int ndim = poly.shape[0]
    cdef int npts = poly.shape[1]

    if Test:
        assert (ndim == 2 or ndim == 3), \
            ("Arg poly must contain the 2D or 3D coordinates of N points."
             " And be shaped in the form (dim, N).")
        assert poly.shape[1]>=3, ("Arg poly must contain the 2D or 3D",
                                  " coordinates of at least 3 points!")
        assert order.lower() in ['c','f'], "Arg order must be in ['c','f']!"
        assert type(Clock) is bool, "Arg Clock must be a bool!"
        assert type(close) is bool, "Arg close must be a bool!"

    # we close the poly if not closed
    if not np.allclose(poly[:,0], poly[:,npts-1], atol=_VSMALL):
        poly = np.concatenate((poly,poly[:,0:1]),axis=1)
        npts += 1 # we added a point

    # verifying that poly is (counter)clockwise
    if ndim==2 and not Clock is None:
        try:
            if not Clock==Poly_isClockwise(poly):
                poly = poly[:,::-1]
        except Exception as excp:
            raise excp

    # if we didn't want it close, we take out last point
    if not close:
        poly = poly[:,:npts-1]

    # taking care of continuity
    poly = np.ascontiguousarray(poly) if order.lower()=='c' \
           else np.asfortranarray(poly)

    return poly


def Poly_VolAngTor(np.ndarray[double,ndim=2,mode='c'] Poly):
    cdef int npts = Poly.shape[1]
    cdef np.ndarray[double,ndim=1] Ri0 = Poly[0,:npts-1], Ri1 = Poly[0,1:]
    cdef np.ndarray[double,ndim=1] Zi0 = Poly[1,:npts-1], Zi1 = Poly[1,1:]
    cdef double V   =  np.sum((Ri0*Zi1 - Zi0*Ri1) * (Ri0+Ri1)) / 6.
    cdef double BV0 =  np.sum(0.5 * (Ri0*Zi1 - Zi0*Ri1) *
                              (Ri1**2 + Ri1*Ri0 + Ri0**2)) / (6.*V)
    cdef double BV1 = -np.sum((Ri1**2*Zi0*(2.*Zi1+Zi0) +
                               2.*Ri0*Ri1*(Zi0**2-Zi1**2) -
                               Ri0**2*Zi1*(Zi1+2.*Zi0))/4.) / (6.*V)
    return V, np.array([BV0,BV1])


def poly_area(double[:,::1] poly, int npts):
    cdef int ii
    cdef double area = 0.
    # # 2 A(P) = sum_{i=1}^{n} ( x_i  (y_{i+1} - y_{i-1}) )
    for ii in range(1,npts):
        area += poly[0,ii] * (poly[1,ii+1] - poly[1,ii-1])
    area += poly[0,0] * (poly[1,1] - poly[1,npts-1])
    return area/2

def poly_area_and_barycenter(double[:,::1] poly, int npts):
    cdef int ii
    cdef double a2
    cdef double area
    cdef double inva6
    cdef np.ndarray[double,ndim=1] cg = np.zeros((2,))
    cdef double[::1] cg_mv = cg
    cdef double[2] p1, p2
    for ii in range(npts):
        p1[0] = poly[0,ii]
        p1[1] = poly[1,ii]
        p2[0] = poly[0,ii+1]
        p2[1] = poly[1,ii+1]
        a2 = p1[0]*p2[1] - p2[0]*p1[1]
        cg_mv[0] += (p1[0] + p2[0])*a2
        cg_mv[1] += (p1[1] + p2[1])*a2
    area = poly_area(poly, npts)
    inva6 = 1. / area / 6.
    cg[0] = cg[0] * inva6
    cg[1] = cg[1] * inva6
    return cg, area

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

def _Ves_isInside(double[:, ::1] pts, double[:, ::1] ves_poly,
                  double[:, ::1] ves_lims=None, int nlim=0,
                  str ves_type='Tor', str in_format='(X,Y,Z)', bint test=True):
    """
    Checks if points Pts are in vessel VPoly.
    VPoly should be CLOSED
    """
    cdef str err_msg
    cdef str ves_type_low = ves_type.lower()
    cdef str in_form_low = in_format.lower()
    cdef bint is_cartesian
    cdef bint is_toroidal = ves_type_low == 'tor'
    cdef int[3] order
    cdef np.ndarray[int,ndim=1] is_inside
    cdef list in_letters = in_form_low.replace('(',
                                               '').replace(')','').split(',')
    # preparing format of coordinates:
    is_cartesian = all([ss in ['x','y','z'] for ss in in_letters])
    if is_cartesian:
        order[0] = in_letters.index('x')
        order[1] = in_letters.index('y')
        order[2] = in_letters.index('z')
    else:
        order[0] = in_letters.index('r')
        order[1] = in_letters.index('z')
        order[2] = in_letters.index('phi')
    # --------------------------------------------------------------------------
    if test:
        err_msg = "Arg VPoly must be a (2,N) np.ndarray !"
        assert ves_poly.shape[0]==2, err_msg
        err_msg = "Arg ves_type must be a str in ['Tor','Lin'] !"
        assert ves_type_low in ['tor','lin'], err_msg
        assert nlim>=0, "nlim should be integer >= 0"
        err_msg = "No valid format, you gave =" + in_format
        assert (is_cartesian or
                all([ss in ['r', 'z', 'phi'] for ss in in_letters])), err_msg
        if is_toroidal and ves_lims is not None:
            assert is_cartesian or 'phi' in in_letters, err_msg
    # --------------------------------------------------------------------------
    is_inside = np.zeros(max(nlim,1)*pts.shape[1],dtype=np.int32)
    _rt.is_inside_vessel(pts, ves_poly, ves_lims, nlim, is_toroidal,
                         is_cartesian, order, is_inside)
    if nlim == 0 or nlim==1:
        return is_inside.astype(bool)
    return is_inside.astype(bool).reshape(nlim, pts.shape[1])

# ==============================================================================
#
#                                 LINEAR MESHING
#                       i.e. Discretizing horizontal lines
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
    cdef int mode_num
    cdef str err_mess
    cdef str mode_low = mode.lower()
    cdef long sz_ld
    cdef long[1] N
    cdef long* lindex = NULL
    cdef double* ldiscret = NULL
    cdef double[2] dl_array
    cdef double[1] resolution
    # ...
    mode_num = _st.get_nb_dmode(mode_low)
    # .. Testing ...............................................................
    err_mess = "Mode has to be 'abs' (absolute) or 'rel' (relative)"
    assert mode_num >= 0, err_mess
    # .. preparing inputs.......................................................
    _st.cythonize_subdomain_dl(DL, dl_array) # dl_array is initialized
    #.. calling cython function.................................................
    sz_ld = _st.discretize_line1d_core(&LMinMax[0], dstep, dl_array, Lim,
                                       mode_num, margin, &ldiscret, resolution,
                                       &lindex, N)
    #.. converting and returning................................................
    ld_arr = np.copy(np.asarray(<double[:sz_ld]> ldiscret))
    li_arr = np.copy(np.asarray(<long[:sz_ld]>lindex)).astype(int)
    free(ldiscret)
    free(lindex)
    return ld_arr, resolution[0], li_arr, N[0]


# ==============================================================================
#
#                                   2D MESHING
#                           i.e. Discretizing polygons
#
# ==============================================================================
def discretize_segment2d(double[::1] LMinMax1, double[::1] LMinMax2,
                         double dstep1, double dstep2,
                         D1=None,
                         D2=None,
                         str mode='abs',
                         double[:,::1] VPoly=None,
                         double margin=_VSMALL):
    """
    Discretizes a 2D segment where the 1st coordinates are defined in LMinMax1
    and the second ones in LMinMax2. The refinement in x is defined by dstep1,
    and dstep2 defines the resolution on y.
    Optionnally you can give a VPoly to which the discretized in which the
    segment has to be included.
    This function is basically a generalization of discretize_line1d.
    Parameters
    ==========
    LMinMax1 : (2)-double array
        Gives the limits LMin and LMax of the x coordinates of the segment.
        LMinMax1 = [xmin, xmax]
    LMinMax2 : (2)-double array
        Gives the limits LMin and LMax of the y coordinates of the segment.
        LMinMax2 = [ymin, ymax]
    dstep1 or dstep2 : double
        Step of discretization, can be absolute (default) or relative
    D1 or D2 : (optional) (2)-double array
        Sub domain of discretization. If not None and if Lim, LMinMax = DL
        (can be only on one limit and can be bigger or smaller than original).
        Actual desired limits
    mode : (optional) string
        If `mode` is "abs" (absolute), then the
        segment will be discretized in cells each of size `dstep`. Else,
        if "rel" (relative), the meshing step is relative to the segments norm
        (the actual discretization step will be (LMax - LMin)/dstep).
    margin : (optional) double
        Margin value for cell length
    VPoly : (optional) 2d double array
        If present, we check that the discretized segment is included in the
        path defined by VPoly.
    Returns
    =======
    ldiscret: double 2d array
        array of the discretized coordinates on the segment of desired limits
    resolution: double array
        step of discretization on 2d (typically resolution-on-x*resolution-on-y)
    lindex: int array
        array of the indices corresponding to ldiscret with respects to the
        original segment LMinMax (if no DL, from 0 to N-1)
    resol1 : double
        Smallest resolution on x
    resol2 : double
        Smallest resolution on y
    """
    cdef int nn
    cdef int ndisc
    cdef int ii, jj
    cdef int tot_true
    cdef int mode_num
    cdef int num_pts_vpoly
    cdef str err_mess
    cdef str mode_low = mode.lower()
    cdef long nind1
    cdef long nind2
    cdef long[1] ncells1
    cdef long[1] ncells2
    cdef double[2] dl1_array
    cdef double[2] dl2_array
    cdef double[2] resolutions
    cdef long[:] lindex_view
    cdef double[:] lresol_view
    cdef double[:,:] ldiscr_view
    cdef int* are_in_poly = NULL
    cdef long* lindex1_arr = NULL
    cdef long* lindex2_arr = NULL
    cdef long* lindex_tmp  = NULL
    cdef double* ldiscr_tmp = NULL
    cdef double* ldiscret1_arr = NULL
    cdef double* ldiscret2_arr = NULL
    cdef np.ndarray[double,ndim=2] ldiscr
    cdef np.ndarray[double,ndim=1] lresol
    cdef np.ndarray[long,ndim=1] lindex
    # ...
    mode_num = _st.get_nb_dmode(mode_low)
    # .. Testing ...............................................................
    err_mess = "Mode has to be 'abs' (absolute) or 'rel' (relative)"
    assert mode_num >= 0, err_mess
    # .. Treating subdomains and Limits ........................................
    _st.cythonize_subdomain_dl(D1, dl1_array)
    _st.cythonize_subdomain_dl(D2, dl2_array)
    # .. Discretizing on the first direction ...................................
    nind1 = _st.discretize_line1d_core(&LMinMax1[0], dstep1, dl1_array,
                                       True, mode_num, margin,
                                       &ldiscret1_arr, &resolutions[0],
                                       &lindex1_arr, ncells1)
    # .. Discretizing on the second direction ..................................
    nind2 = _st.discretize_line1d_core(&LMinMax2[0], dstep2, dl2_array,
                                       True, mode_num, margin,
                                       &ldiscret2_arr, &resolutions[1],
                                       &lindex2_arr, ncells2)
    #....
    if VPoly is not None:
        ndisc = nind1 * nind2
        ldiscr_tmp = <double *>malloc(ndisc * 2 * sizeof(double))
        lindex_tmp = <long *>malloc(ndisc * sizeof(long))
        for ii in range(nind2):
            for jj in range(nind1):
                nn = jj + nind1 * ii
                ldiscr_tmp[nn] = ldiscret1_arr[jj]
                ldiscr_tmp[ndisc + nn] = ldiscret2_arr[ii]
                lindex_tmp[nn] = lindex1_arr[jj] + nind1 * lindex2_arr[ii]
        num_pts_vpoly = VPoly.shape[1] - 1
        are_in_poly = <int *>malloc(ndisc * sizeof(int))
        tot_true = _bgt.is_point_in_path_vec(num_pts_vpoly,
                                             &VPoly[0][0], &VPoly[1][0],
                                             ndisc,
                                             &ldiscr_tmp[0], &ldiscr_tmp[ndisc],
                                             are_in_poly)
        ldiscr = np.empty((2, tot_true), dtype=float)
        lresol = np.empty((tot_true,), dtype=float)
        lindex = np.empty((tot_true,), dtype=int)
        ldiscr_view = ldiscr
        lindex_view = lindex
        lresol_view = lresol
        jj = 0
        for ii in range(ndisc):
            if are_in_poly[ii]:
                lresol_view[jj] = resolutions[0] * resolutions[1]
                lindex_view[jj] = lindex_tmp[ii]
                ldiscr_view[0,jj] = ldiscr_tmp[ii]
                ldiscr_view[1,jj] = ldiscr_tmp[ii + ndisc]
                jj = jj + 1
        free(ldiscr_tmp)
        free(lindex_tmp)
        free(are_in_poly)
        free(ldiscret1_arr)
        free(ldiscret2_arr)
        free(lindex1_arr)
        free(lindex2_arr)
        return ldiscr, lresol,\
            lindex, resolutions[0], resolutions[1]
    else:
        ndisc = nind1 * nind2
        ldiscr = np.empty((2, ndisc), dtype=float)
        lresol = np.empty((ndisc,), dtype=float)
        lindex = np.empty((ndisc,), dtype=int)
        ldiscr_view = ldiscr
        lindex_view = lindex
        lresol_view = lresol
        for ii in range(nind2):
            for jj in range(nind1):
                nn = jj + nind1 * ii
                ldiscr_view[0,nn] = ldiscret1_arr[jj]
                ldiscr_view[1,nn] = ldiscret2_arr[ii]
                lindex_view[nn] = lindex1_arr[jj] + nind1 * lindex2_arr[ii]
                lresol_view[nn] = resolutions[0] * resolutions[1]
        free(ldiscret1_arr)
        free(ldiscret2_arr)
        free(lindex1_arr)
        free(lindex2_arr)
        return ldiscr, lresol,\
            lindex, resolutions[0], resolutions[1]


def _Ves_meshCross_FromInd(double[::1] MinMax1,
                           double[::1] MinMax2,
                           double d1,
                           double d2,
                           long[::1] ind,
                           str dSMode='abs',
                           double margin=_VSMALL):

    cdef int ii
    cdef int i1, i2
    cdef int ncells1
    cdef int mode_num
    cdef int num_pts = ind.size
    cdef long[2] ncells
    cdef long* dummy = NULL
    cdef double d1r, d2r
    cdef double[2] resolution
    cdef double[2] dl_array
    cdef double* X1 = NULL
    cdef double* X2 = NULL
    cdef str mode_low = dSMode.lower()
    cdef np.ndarray[double,ndim=2] pts_disc = np.empty((2, num_pts))
    cdef np.ndarray[double,ndim=1] dS
    # ...
    mode_num = _st.get_nb_dmode(mode_low)
    # .. Testing ...............................................................
    err_mess = "Mode has to be 'abs' (absolute) or 'rel' (relative)"
    assert mode_num >= 0, err_mess
    #.. preparing inputs........................................................
    dl_array[0] = C_NAN
    dl_array[1] = C_NAN
    #.. calling cython function.................................................
    _st.discretize_line1d_core(&MinMax1[0], d1, dl_array, True, mode_num,
                               margin, &X1, &resolution[0], &dummy,
                               &ncells[0])
    _st.discretize_line1d_core(&MinMax2[0], d2, dl_array, True, mode_num,
                               margin, &X2, &resolution[1], &dummy,
                               &ncells[1])
    d1r = resolution[0]
    d2r = resolution[1]
    ncells1 = ncells[0]
    # creating array of resolutions/surfaces
    dS = d1r*d2r*np.ones((num_pts,))
    for ii in range(num_pts):
        i2 = ind[ii] // ncells1
        i1 = ind[ii] - i2 * ncells1
        pts_disc[0, ii] = X1[i1]
        pts_disc[1, ii] = X2[i2]
    # freeing arrays allocated in discretize_line1d_core
    free(X1)
    free(X2)
    free(dummy)
    return pts_disc, dS, d1r, d2r


def discretize_vpoly(double[:,::1] VPoly, double dL,
                     str mode='abs',
                     list D1=None,
                     list D2=None,
                     double margin=_VSMALL, double DIn=0.,
                     double[:,::1] VIn=None):
    """
    Discretizes a VPoly (2D polygon of a cross section in (R,Z) coordinates).
    The refinement on R and Z is defined dL.
    Optionnally you can give a coefficient (DIn) and the normal vectors going
    inwards the VPoly (VIn), and the result will be slightly shifted inwards
    (or outwards if DIn<0) of DIn*VIn.
    Parameters
    ==========
    VPoly : (2, num_vertex) double array
       Coordinates of the vertices of the Polygon defining the 2D poloidal
       cut of the Vessel
    dL : double
        Step of discretization, can be absolute (default) or relative
    D1 or D2 : (optional) (2)-double array
        Sub domain of discretization.
        (can be only on one limit and can be bigger or smaller than original).
        Actual desired limits
    mode : (optional) string
        If `mode` is "abs" (absolute), then the
        segment will be discretized in cells each of size `dstep`. Else,
        if "rel" (relative), the meshing step is relative to the segments norm
    margin : (optional) double
        Margin value for cell length
    Returns
    =======
    return PtsCross, resol, ind_arr, N_arr, Rref_arr, VPolybis
    PtsCross: double 2d array
        array of the discretized coordinates of the VPoly
    resol: double 1d array
        step of discretization on 2d
    ind_arr: int  1d array
        array of the indices corresponding to ldiscret with respects to the
        original VPoly
    N_arr : int 1d array
        number of cells on each segment of the VPoly
    Rref_arr : double 1d array
        reference Radius coordinates, not shifted even if DIn <> 0.
        If DIn == 0, then Rref_arr = PtsCross[0, ...]
    VPolybis :

    """
    cdef int npts_disc=VPoly.shape[1]
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
    cdef np.ndarray[double,ndim=1] PtsCrossX, PtsCrossY
    cdef np.ndarray[double,ndim=1] VPolybisX, VPolybisY
    cdef np.ndarray[double,ndim=1] Rref_arr, resol
    cdef np.ndarray[long,ndim=1] ind_arr, N_arr
    cdef np.ndarray[np.npy_bool,ndim=1,cast=True] indin
    cdef str mode_low = mode.lower()
    cdef int mode_num = _st.get_nb_dmode(mode_low)
    # .. Testing ...............................................................
    assert mode_num >= 0, "Mode has to be 'abs' (absolute) or 'rel' (relative)"
    assert (not (DIn == 0.) or VIn is not None)
    # .. preparing inputs.......................................................
    _st.discretize_vpoly_core(VPoly, dL, mode_num, margin, DIn, VIn,
                            &XCross, &YCross, &resolution,
                            &ind, &numcells, &Rref, &XPolybis, &YPolybis,
                            &sz_vb[0], &sz_ot[0], npts_disc)
    assert not ((XCross == NULL) or (YCross == NULL)
                or (XPolybis == NULL) or (YPolybis == NULL))
    PtsCrossX = np.array(<double[:sz_ot[0]]> XCross)
    PtsCrossY = np.array(<double[:sz_ot[0]]> YCross)
    PtsCross = np.array([PtsCrossX,PtsCrossY])
    VPolybisX = np.array(<double[:sz_vb[0]]> XPolybis)
    VPolybisY = np.array(<double[:sz_vb[0]]> YPolybis)
    VPolybis = np.array([VPolybisX, VPolybisY])
    resol = np.array(<double[:sz_ot[0]]> resolution)
    Rref_arr = np.array(<double[:sz_ot[0]]> Rref)
    ind_arr = np.array(<long[:sz_ot[0]]> ind)
    N_arr = np.array(<long[:npts_disc-1]> numcells)
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
    free(XCross)
    free(YCross)
    free(XPolybis)
    free(YPolybis)
    free(resolution)
    free(Rref)
    free(ind)
    free(numcells)
    return PtsCross, resol, ind_arr, N_arr, Rref_arr, VPolybis


# ==============================================================================
#
#                     3D MESHING in TOROIDAL configurations
#                           i.e. Discretizing Volumes
#
# ==============================================================================
def _Ves_Vmesh_Tor_SubFromD_cython(double rstep, double zstep, double phistep,
                                   double[::1] RMinMax, double[::1] ZMinMax,
                                   list DR=None, list DZ=None, DPhi=None,
                                   double[:,::1] limit_vpoly=None,
                                   str out_format='(X,Y,Z)',
                                   double margin=_VSMALL,
                                   int num_threads=48):
    """Returns the desired submesh indicated by the limits (DR,DZ,DPhi),
    for the desired resolution (rstep,zstep,dRphi).

    Parameters
    ----------
        rstep (double): refinement along radius `r`
        zstep (double): refinement along height `z`
        phistep (double): refinement along toroidal direction `phi`
        RMinMax: array specifying the limits min and max in `r`
        ZMinMax: array specifying the limits min and max in `z`
        DR: array specifying the actual sub-volume limits to get in `r`
        DZ: array specifying the actual sub-volume limits to get in `z`
        DPhi: array specifying the actual sub-volume limits to get in `phi`
        limit_vpoly: array-like defining the `(R,Z)` coordinates of the poloidal
            cut of the limiting flux surface
        out_format(string): either "(X,Y,Z)" or "(R,Z,Phi)" for cartesian or
            polar coordinates
        margin(double): tolerance error.
            Defaults to |_VSMALL|
    Returns
    ------
       pts: (3, npts) array in out_format (cartesian or polar) of
            discretized volume
       res3d: (npts) resolution on each point
       ind:   (npts) indices to reconstruct the points in 3D (useful only if
              limit_vpoly is not none)
       reso_r: (double) resolution on r (constant)
       reso_z: (double) resolution on z (constant)
       reso_phi : (sz_r) array, resolution R*dPhi, phi resolution on each R
       sz_r : (int) number of points in r
       sz_z : (int) number of points in z
    """
    cdef int jj
    cdef int npts_disc = 0
    cdef int sz_r, sz_z
    cdef int r_ratio
    cdef int ind_loc_r0
    cdef int npts_vpoly
    cdef int[1] max_sz_phi
    cdef str out_low = out_format.lower()
    cdef bint is_cart = out_low == '(x,y,z)'
    cdef double min_phi, max_phi
    cdef double min_phi_pi
    cdef double max_phi_pi
    cdef double abs0, abs1
    cdef double reso_r_z
    cdef double twopi_over_dphi
    cdef long[1] ncells_r0, ncells_r, ncells_z
    cdef long[::1] ind_mv
    cdef double[2] limits_dl
    cdef double[1] reso_r0, reso_r, reso_z
    cdef double[::1] dv_mv
    cdef double[::1] reso_phi_mv
    cdef double[:, ::1] poly_mv
    cdef double[:, ::1] pts_mv
    cdef long[:, ::1] indi_mv
    cdef long[:, :, ::1] lnp
    cdef long*  ncells_rphi  = NULL
    cdef long*  tot_nc_plane = NULL
    cdef long*  lindex   = NULL
    cdef long*  lindex_z = NULL
    cdef long*  sz_phi = NULL
    cdef double* disc_r0 = NULL
    cdef double* disc_r  = NULL
    cdef double* disc_z  = NULL
    cdef double* step_rphi = NULL
    cdef long[::1] first_ind_mv
    cdef np.ndarray[long, ndim=2] indI
    cdef np.ndarray[long, ndim=1] ind
    cdef np.ndarray[double,ndim=1] reso_phi
    cdef np.ndarray[double,ndim=2] pts
    cdef np.ndarray[double,ndim=1] res3d
    #
    # Get the actual R and Z resolutions and mesh elements
    # .. First we discretize R without limits ..................................
    _st.cythonize_subdomain_dl(None, limits_dl) # no limits
    _ = _st.discretize_line1d_core(&RMinMax[0], rstep, limits_dl,
                                   True, 0, # discretize in absolute mode
                                   margin, &disc_r0, reso_r0, &lindex,
                                   ncells_r0)
    free(lindex) # getting rid of things we dont need
    lindex = NULL
    # .. Now the actual R limited  .............................................
    _st.cythonize_subdomain_dl(DR, limits_dl)
    sz_r = _st.discretize_line1d_core(&RMinMax[0], rstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_r, reso_r, &lindex,
                                      ncells_r)
    free(lindex) # getting rid of things we dont need
    # .. Now Z .................................................................
    _st.cythonize_subdomain_dl(DZ, limits_dl)
    sz_z = _st.discretize_line1d_core(&ZMinMax[0], zstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_z, reso_z, &lindex_z,
                                      ncells_z)
    # .. Preparing for phi: get the limits if any and make sure to replace them
    # .. in the proper quadrants ...............................................
    if DPhi is None:
        min_phi = -c_pi
        max_phi = c_pi
    else:
        min_phi = DPhi[0] # to avoid conversions
        min_phi = c_atan2(c_sin(min_phi), c_cos(min_phi))
        max_phi = DPhi[1] # to avoid conversions
        max_phi = c_atan2(c_sin(max_phi), c_cos(max_phi))
    # .. Initialization ........................................................
    sz_phi = <long*>malloc(sz_r*sizeof(long))
    tot_nc_plane = <long*>malloc(sz_r*sizeof(long))
    ncells_rphi  = <long*>malloc(sz_r*sizeof(long))
    step_rphi    = <double*>malloc(sz_r*sizeof(double))
    reso_phi = np.empty((sz_r,)) # we create the numpy array
    reso_phi_mv = reso_phi # and its associated memoryview
    r_ratio = <int>(c_ceil(disc_r[sz_r - 1] / disc_r[0]))
    twopi_over_dphi = _TWOPI / phistep
    ind_loc_r0 = 0
    ncells_rphi0 = 0
    min_phi_pi = min_phi + c_pi
    max_phi_pi = max_phi + c_pi
    abs0 = c_abs(min_phi_pi)
    abs1 = c_abs(max_phi_pi)
    # ... doing 0 loop before
    if min_phi < max_phi:
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
        ncells_rphi[0] = <int>c_ceil(twopi_over_dphi * disc_r[0])
        loc_nc_rphi = ncells_rphi[0]
        step_rphi[0] = _TWOPI / ncells_rphi[0]
        inv_drphi = 1. / step_rphi[0]
        reso_phi_mv[0] = step_rphi[0] * disc_r[0]
        tot_nc_plane[0] = 0 # initialization
        # Get index and cumulated indices from background
        for jj in range(ind_loc_r0, ncells_r0[0]):
            if disc_r0[jj]==disc_r[0]:
                ind_loc_r0 = jj
                break
            else:
                ncells_rphi0 += <long>c_ceil(twopi_over_dphi * disc_r0[jj])
                tot_nc_plane[0] = ncells_rphi0 * ncells_z[0]
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        if abs0 - step_rphi[0]*c_floor(abs0 * inv_drphi) < margin*step_rphi[0]:
            nphi0 = int(c_round((min_phi + c_pi) * inv_drphi))
        else:
            nphi0 = int(c_floor((min_phi +c_pi) * inv_drphi))
        if abs1-step_rphi[0]*c_floor(abs1 * inv_drphi) < margin*step_rphi[0]:
            nphi1 = int(c_round((max_phi+c_pi) * inv_drphi)-1)
        else:
            nphi1 = int(c_floor((max_phi+c_pi) * inv_drphi))
        sz_phi[0] = nphi1 + 1 - nphi0
        max_sz_phi[0] = sz_phi[0]
        indI = -np.ones((sz_r, sz_phi[0] * r_ratio + 1), dtype=int)
        indi_mv = indI
        for jj in range(sz_phi[0]):
            indi_mv[0, jj] = nphi0 + jj
        npts_disc += sz_z * sz_phi[0]
    else:
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
        ncells_rphi[0] = <int>c_ceil(twopi_over_dphi * disc_r[0])
        loc_nc_rphi = ncells_rphi[0]
        step_rphi[0] = _TWOPI / ncells_rphi[0]
        inv_drphi = 1. / step_rphi[0]
        reso_phi_mv[0] = step_rphi[0] * disc_r[0]
        tot_nc_plane[0] = 0 # initialization
        # Get index and cumulated indices from background
        for jj in range(ind_loc_r0, ncells_r0[0]):
            if disc_r0[jj]==disc_r[0]:
                ind_loc_r0 = jj
                break
            else:
                ncells_rphi0 += <long>c_ceil(twopi_over_dphi * disc_r0[jj])
                tot_nc_plane[0] = ncells_rphi0 * ncells_z[0]
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        if abs0 - step_rphi[0]*c_floor(abs0 * inv_drphi) < margin*step_rphi[0]:
            nphi0 = int(c_round((min_phi + c_pi) * inv_drphi))
        else:
            nphi0 = int(c_floor((min_phi + c_pi) * inv_drphi))
        if abs1-step_rphi[0]*c_floor(abs1 * inv_drphi) < margin*step_rphi[0]:
            nphi1 = int(c_round((max_phi+c_pi) * inv_drphi)-1)
        else:
            nphi1 = int(c_floor((max_phi+c_pi) * inv_drphi))
        sz_phi[0] = nphi1+1+loc_nc_rphi-nphi0
        max_sz_phi[0] = sz_phi[0]
        indI = -np.ones((sz_r, sz_phi[0] * r_ratio + 1), dtype=int)
        indi_mv = indI
        for jj in range(loc_nc_rphi - nphi0):
            indi_mv[0, jj] = nphi0 + jj
        for jj in range(loc_nc_rphi - nphi0, sz_phi[0]):
            indi_mv[0, jj] = jj - (loc_nc_rphi - nphi0)
        npts_disc += sz_z * sz_phi[0]
    # ... doing the others .....................................................
    npts_disc += _st.vmesh_disc_phi(sz_r, sz_z, ncells_rphi, phistep,
                             ncells_rphi0,
                             disc_r, disc_r0, step_rphi,
                             reso_phi_mv, tot_nc_plane, ind_loc_r0,
                             ncells_r0[0], ncells_z[0], &max_sz_phi[0],
                             min_phi, max_phi, sz_phi, indi_mv,
                             margin, num_threads)
    # ... vignetting ...........................................................
    is_in_vignette = np.ones((sz_r, sz_z), dtype=int) # by default yes

    if limit_vpoly is not None:
        npts_vpoly = limit_vpoly.shape[1] - 1

        # we make sure it is closed
        if not(abs(limit_vpoly[0, 0] - limit_vpoly[0, npts_vpoly]) < _VSMALL
                and abs(limit_vpoly[1, 0]
                        - limit_vpoly[1, npts_vpoly]) < _VSMALL):
            poly_mv = np.concatenate((limit_vpoly, limit_vpoly[:, 0:1]), axis=1)
            npts_vpoly += 1
        else:
            poly_mv = limit_vpoly

        _ = _vt.are_in_vignette(sz_r, sz_z,
                                poly_mv, npts_vpoly,
                                disc_r, disc_z,
                                is_in_vignette)

    # Preparing an array of indices to associate (r, z, phi) => npts_disc
    lnp = np.empty((sz_r, sz_z, max_sz_phi[0]), dtype=int)
    new_np = _st.vmesh_get_index_arrays(lnp, is_in_vignette,
                                        sz_r, sz_z, sz_phi)
    if limit_vpoly == None:
        assert npts_disc == new_np, f"No matching {npts_disc} vs {new_np}"
    else:
        npts_disc = new_np

    pts = np.empty((3,npts_disc))
    ind = np.empty((npts_disc,), dtype=int)
    res3d  = np.empty((npts_disc,))
    pts_mv = pts
    ind_mv = ind
    dv_mv  = res3d
    reso_r_z = reso_r[0]*reso_z[0]

    indI = np.sort(indI, axis=1)
    indi_mv = indI
    first_ind_mv = np.argmax(indI > -1, axis=1).astype(int)

    _st.vmesh_assemble_arrays(first_ind_mv, indi_mv,
                          is_in_vignette,
                          is_cart, sz_r,
                          sz_z, lindex_z,
                          ncells_rphi, tot_nc_plane,
                          reso_r_z, step_rphi,
                          disc_r, disc_z, lnp, sz_phi,
                          dv_mv, reso_phi_mv, pts_mv, ind_mv,
                          num_threads)
    free(disc_r)
    free(disc_z)
    free(disc_r0)
    free(sz_phi)
    free(lindex_z)
    free(step_rphi)
    free(ncells_rphi)
    free(tot_nc_plane)
    return pts, res3d, ind, reso_r[0], reso_z[0], reso_phi, sz_r, sz_z


def _Ves_Vmesh_Tor_SubFromInd_cython(double rstep, double zstep, double phistep,
                                     double[::1] RMinMax, double[::1] ZMinMax,
                                     long[::1] ind, str Out='(X,Y,Z)',
                                     double margin=_VSMALL, int num_threads=48):
    """
    Return the desired submesh indicated by the (numerical) indices,
    for the desired resolution (rstep, zstep, phistep)
    """
    cdef str out_low = Out.lower()
    cdef bint is_cart = out_low == '(x,y,z)'
    cdef int sz_r, sz_z
    cdef long npts_disc=len(ind)
    cdef double twopi_over_dphi
    cdef double[::1] dRPhirRef
    cdef int[::1] Ru
    cdef np.ndarray[double,ndim=2] pts=np.empty((3,npts_disc))
    cdef np.ndarray[double,ndim=1] res3d=np.empty((npts_disc,))
    cdef long[1]   ncells_r, ncells_z
    cdef double[1] reso_r, reso_z
    cdef double[2] limits_dl
    cdef int* ncells_rphi = NULL
    cdef long* lindex = NULL
    cdef long* tot_nc_plane = NULL
    cdef double* disc_r = NULL
    cdef double* disc_z = NULL
    cdef double** phi_tab = NULL

    # Get the actual R and Z resolutions and mesh elements
    # .. We discretize R .......................................................
    _st.cythonize_subdomain_dl(None, limits_dl)
    sz_r = _st.discretize_line1d_core(&RMinMax[0], rstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_r, reso_r, &lindex,
                                      ncells_r)
    free(lindex) # getting rid of things we dont need
    lindex = NULL
    # .. We discretize Z .......................................................
    sz_z = _st.discretize_line1d_core(&ZMinMax[0], zstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_z, reso_z, &lindex,
                                      ncells_z)
    free(lindex)
    # Number of Phi per R
    dRPhirRef =  np.empty((sz_r,))
    Ru = np.zeros((sz_r,), dtype=np.dtype("i"))
    dRPhir = np.nan*np.ones((sz_r,))
    tot_nc_plane = <long*> malloc((sz_r + 1) * sizeof(long))
    # .. Initialization ........................................................
    ncells_rphi  = <int*>malloc(sz_r * sizeof(int))
    phi_tab = <double**>malloc(sizeof(double*))
    phi_tab[0] = NULL
    reso_r_z = reso_r[0]*reso_z[0]
    twopi_over_dphi = _TWOPI / phistep
    # .. Discretizing Phi (with respect to the corresponding radius R) .........
    _st.vmesh_ind_init_tabs(ncells_rphi, disc_r, sz_r, sz_z,
                            twopi_over_dphi, dRPhirRef,
                            tot_nc_plane, &phi_tab[0],
                            num_threads)
    # .. Computing the points coordinates ......................................
    if is_cart:
        _st.vmesh_ind_cart_loop(npts_disc, sz_r, ind, tot_nc_plane,
                                ncells_rphi, phi_tab[0], disc_r, disc_z,
                                pts, res3d, reso_r_z, dRPhirRef, Ru,
                                dRPhir, num_threads)
    else:
        _st.vmesh_ind_polr_loop(npts_disc, sz_r, ind, tot_nc_plane,
                                ncells_rphi, phi_tab[0], disc_r, disc_z,
                                pts, res3d, reso_r_z, dRPhirRef, Ru,
                                dRPhir, num_threads)
    free(ncells_rphi)
    free(tot_nc_plane)
    if not phi_tab[0] == NULL:
        free(phi_tab[0])
    if not phi_tab == NULL:
        free(phi_tab)
    return (pts, res3d, reso_r[0], reso_z[0],
            np.asarray(dRPhir)[~np.isnan(dRPhir)])


# ==============================================================================
#
#                      3D MESHING in LINEAR configurations
#                           i.e. Discretizing Volumes
#
# ==============================================================================

def _Ves_Vmesh_Lin_SubFromD_cython(double dX, double dY, double dZ,
                                   double[::1] XMinMax, double[::1] YMinMax,
                                   double[::1] ZMinMax,
                                   list DX=None, list DY=None, list DZ=None,
                                   limit_vpoly=None,
                                   double margin=_VSMALL):
    """ Return the desired submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dY,dZ)
    """
    cdef double[::1] X, Y, Z
    cdef double dXr, dYr, reso_z, res3d
    cdef np.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, Xn, Yn, Zn
    cdef np.ndarray[double,ndim=2] pts
    cdef np.ndarray[long,ndim=1] ind

    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, indX, NX = discretize_line1d(XMinMax, dX, DX, Lim=True,
                                                margin=margin)
    Y, dYr, indY, NY = discretize_line1d(YMinMax, dY, DY, Lim=True,
                                                margin=margin)
    Z, reso_z, indZ, _ = discretize_line1d(ZMinMax, dZ, DZ, Lim=True,
                                                margin=margin)
    Xn, Yn, Zn = len(X), len(Y), len(Z)

    pts = np.array([np.tile(X,(Yn*Zn,1)).flatten(),
                    np.tile(np.repeat(Y,Xn),(Zn,1)).flatten(),
                    np.repeat(Z,Xn*Yn)])
    ind = np.repeat(NX*NY*indZ,Xn*Yn) + \
      np.tile(np.repeat(NX*indY,Xn),(Zn,1)).flatten() + \
      np.tile(indX,(Yn*Zn,1)).flatten()
    res3d = dXr*dYr*reso_z

    if limit_vpoly is not None:
        indin = Path(limit_vpoly.T).contains_points(pts[1:,:].T,
                                                    transform=None,
                                                    radius=0.0)
        pts, ind = pts[:,indin], ind[indin]

    return pts, res3d, ind.astype(int), dXr, dYr, reso_z


def _Ves_Vmesh_Lin_SubFromInd_cython(double dX, double dY, double dZ,
                                     double[::1] XMinMax, double[::1] YMinMax,
                                     double[::1] ZMinMax,
                                     np.ndarray[long,ndim=1] ind,
                                     double margin=_VSMALL):
    """ Return the desired submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dY,dZ)
    """

    cdef np.ndarray[double,ndim=1] X, Y, Z
    cdef double dXr, dYr, reso_z, res3d
    cdef np.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY
    cdef np.ndarray[double,ndim=2] pts

    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, _, NX = discretize_line1d(XMinMax, dX, None, Lim=True,
                                               margin=margin)
    Y, dYr, _, NY = discretize_line1d(YMinMax, dY, None, Lim=True,
                                               margin=margin)
    Z, reso_z, _, _ = discretize_line1d(ZMinMax, dZ, None, Lim=True,
                                               margin=margin)

    indZ = ind // (NX*NY)
    indY = (ind - NX*NY*indZ) // NX
    indX = ind - NX*NY*indZ - NX*indY
    pts = np.array([X[indX.astype(int)],
                    Y[indY.astype(int)],
                    Z[indZ.astype(int)]])
    res3d = dXr*dYr*reso_z

    return pts, res3d, dXr, dYr, reso_z



########################################################
########################################################
#       Meshing - Surface - Tor
########################################################

def _getBoundsinter2AngSeg(bool Full, double Phi0, double Phi1,
                           double DPhi0, double DPhi1):
    """ Return inter=True if an intersection exist (all angles in radians
    in [-pi;pi])

    If inter, return Bounds, a list of tuples indicating the segments defining
    the intersection, with
    The intervals are ordered from lowest index to highest index (with respect
    to [Phi0,Phi1])
    """
    if Full:
        Bounds = [[DPhi0,DPhi1]] if DPhi0<=DPhi1 else [[-c_pi,DPhi1],[DPhi0,c_pi]]
        inter = True
        Faces = [None, None]

    else:
        inter, Bounds, Faces = False, None, [False,False]
        if Phi0<=Phi1:
            if DPhi0<=DPhi1:
                if DPhi0<=Phi1 and DPhi1>=Phi0:
                    inter = True
                    Bounds = [[None,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
            else:
                if DPhi0<=Phi1 or DPhi1>=Phi0:
                    inter = True
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
                    inter = True
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
                inter = True
                if DPhi0>=Phi0 and DPhi1>=Phi0:
                    Bounds = [[Phi0,DPhi1],[DPhi0,c_pi],[-c_pi,Phi1]]
                    Faces = [True,True]
                elif DPhi0<=Phi1 and DPhi1<=Phi1:
                    Bounds = [[Phi0,c_pi],[-c_pi,DPhi1],[DPhi0,Phi1]]
                    Faces = [True,True]
                else:
                    Bounds = [[None,c_pi],[-c_pi,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[1][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
    return inter, Bounds, Faces


def _Ves_Smesh_Tor_SubFromD_cython(double dL, double dRPhi,
                                   double[:,::1] VPoly,
                                   DR=None,
                                   DZ=None,
                                   DPhi=None,
                                   double DIn=0., VIn=None, PhiMinMax=None,
                                   str Out='(X,Y,Z)', double margin=_VSMALL):
    """ Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi)
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double DPhi0, DPhi1, DDPhi, DPhiMinMax
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0, Indin
    cdef int NR0, nRPhi0, indR0ii, ii, jj0=0, jj, nphi0, nphi1
    cdef int npts_disc, radius_ratio, Ln
    cdef np.ndarray[double,ndim=2] pts, indI, ptsCross, VPbis
    cdef np.ndarray[double,ndim=1] R0, dS, ind, dLr, Rref, dRPhir, iii
    cdef np.ndarray[long,ndim=1] indL, NL, indok

    # To avoid warnings:
    indI = np.empty((1,1))
    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-c_pi,c_pi]
        DPhiMinMax = _TWOPI
        Full = True
    else:
        PhiMinMax = [c_atan2(c_sin(PhiMinMax[0]),c_cos(PhiMinMax[0])),
                     c_atan2(c_sin(PhiMinMax[1]),c_cos(PhiMinMax[1]))]
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] \
          else _TWOPI + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper
    # quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else c_atan2(c_sin(DPhi[0]),
                                                            c_cos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else c_atan2(c_sin(DPhi[1]),
                                                            c_cos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else _TWOPI+DPhi1-DPhi0

    inter, Bounds, _ = _getBoundsinter2AngSeg(Full, PhiMinMax[0],
                                                  PhiMinMax[1], DPhi0, DPhi1)

    if inter:

        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += _TWOPI
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += _TWOPI

        # Get the actual R and Z resolutions and mesh elements
        ptsCross, dLr, indL, \
          NL, Rref, VPbis = discretize_vpoly(VPoly, dL, D1=None, D2=None,
                                             margin=margin, DIn=DIn, VIn=VIn)
        R0 = np.copy(Rref)
        NR0 = R0.size
        indin = np.ones((ptsCross.shape[1],),dtype=bool)
        if DR is not None:
            if DR[0] is not None:
                indin = indin & (R0 >= DR[0])
            if DR[1] is not None:
                indin = indin & (R0 <= DR[1])
        if DZ is not None:
            if DZ[0] is not None:
                indin = indin & (ptsCross[1,:] >= DZ[0])
            if DZ[1] is not None:
                indin = indin & (ptsCross[1,:] <= DZ[1])

        ptsCross = ptsCross[:,indin]
        dLr = dLr[indin]
        indL = indL[indin]
        Rref = Rref[indin]

        Ln = indin.sum()
        Indin = indin.nonzero()[0].astype(int)

        dRPhir, dPhir = np.empty((Ln,)), np.empty((Ln,))
        Phin = np.zeros((Ln,),dtype=int)
        NRPhi = np.empty((Ln,))
        NRPhi0 = np.zeros((Ln,),dtype=int)
        nRPhi0, indR0ii = 0, 0
        npts_disc = 0
        radius_ratio = int(c_ceil(np.max(Rref)/np.min(Rref)))
        indBounds = np.empty((2,nBounds),dtype=int)
        for ii in range(0,Ln):
            # Get the actual RPhi resolution and Phi mesh elements
            # (! depends on R!)
            NRPhi[ii] = c_ceil(DPhiMinMax*Rref[ii]/dRPhi)
            NRPhi_int = int(NRPhi[ii])
            dPhir[ii] = DPhiMinMax/NRPhi[ii]
            dRPhir[ii] = dPhir[ii]*Rref[ii]
            # Get index and cumulated indices from background
            for jj0 in range(indR0ii,NR0):
                if jj0==Indin[ii]:
                    indR0ii = jj0
                    break
                else:
                    nRPhi0 += <long>c_ceil(DPhiMinMax*R0[jj0]/dRPhi)
                    NRPhi0[ii] = nRPhi0
            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
            for kk in range(0,nBounds):
                abs0 = BC[kk][0]-PhiMinMax[0]
                if abs0-dPhir[ii]*c_floor(abs0/dPhir[ii])<margin*dPhir[ii]:
                    nphi0 = int(c_round(abs0/dPhir[ii]))
                else:
                    nphi0 = int(c_floor(abs0/dPhir[ii]))
                abs1 = BC[kk][1]-PhiMinMax[0]
                if abs1-dPhir[ii]*c_floor(abs1/dPhir[ii])<margin*dPhir[ii]:
                    nphi1 = int(c_round(abs1/dPhir[ii])-1)
                else:
                    nphi1 = int(c_floor(abs1/dPhir[ii]))
                indBounds[0,kk] = nphi0
                indBounds[1,kk] = nphi1
                Phin[ii] += nphi1+1-nphi0

            if ii==0:
                indI = np.nan*np.ones((Ln,Phin[ii]*radius_ratio+1))
            jj = 0
            for kk in range(0,nBounds):
                for kkb in range(indBounds[0,kk],indBounds[1,kk]+1):
                    indI[ii,jj] = <double>( kkb )
                    jj += 1
            npts_disc += Phin[ii]

        # Finish counting to get total number of points
        if jj0<=NR0-1:
            for jj0 in range(indR0ii,NR0):
                nRPhi0 += <long>c_ceil(DPhiMinMax*R0[jj0]/dRPhi)

        # Compute pts, res3d and ind
        pts = np.nan*np.ones((3,npts_disc))
        ind = np.nan*np.ones((npts_disc,))
        dS = np.nan*np.ones((npts_disc,))
        # This triple loop is the longest part, it takes ~90% of the CPU time
        npts_disc = 0
        if Out.lower()=='(x,y,z)':
            for ii in range(0,Ln):
                # Some rare cases with doubles have to be eliminated:
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])])
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    phi = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    pts[0,npts_disc] = ptsCross[0,ii]*c_cos(phi)
                    pts[1,npts_disc] = ptsCross[0,ii]*c_sin(phi)
                    pts[2,npts_disc] = ptsCross[1,ii]
                    ind[npts_disc] = NRPhi0[ii] + indiijj
                    dS[npts_disc] = dLr[ii]*dRPhir[ii]
                    npts_disc += 1
        else:
            for ii in range(0,Ln):
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])])
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    pts[0,npts_disc] = ptsCross[0,ii]
                    pts[1,npts_disc] = ptsCross[1,ii]
                    pts[2,npts_disc] = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    ind[npts_disc] = NRPhi0[ii] + indiijj
                    dS[npts_disc] = dLr[ii]*dRPhir[ii]
                    npts_disc += 1
        indok = (~np.isnan(ind)).nonzero()[0].astype(int)
        ind = ind[indok]
        dS = dS[indok]
        if len(indok)==1:
            pts = pts[:,indok].reshape((3,1))
        else:
            pts = pts[:,indok]
    else:
        pts, dS, ind, NL, Rref, dRPhir, nRPhi0 = np.ones((3,0)), np.ones((0,)),\
          np.ones((0,)), np.nan*np.ones((VPoly.shape[1]-1,)),\
          np.ones((0,)), np.ones((0,)), 0
    return np.ascontiguousarray(pts), dS, ind.astype(int), NL, dLr, Rref, dRPhir, nRPhi0, VPbis


def _Ves_Smesh_Tor_SubFromInd_cython(double dL, double dRPhi,
                                     double[:,::1] VPoly, long[::1] ind,
                                     double DIn=0., VIn=None, PhiMinMax=None,
                                     str Out='(X,Y,Z)', double margin=_VSMALL):
    """ Return the desired submesh indicated by the (numerical) indices,
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] dRPhirRef, dPhir
    cdef long[::1] indL, NRPhi0, NRPhi
    cdef long  NP=len(ind), radius_ratio
    cdef int ii=0, jj=0, iiL, iiphi, Ln, nRPhi0
    cdef double[:,::1] Phi
    cdef np.ndarray[double,ndim=2] pts=np.empty((3,NP)), ptsCross, VPbis
    cdef np.ndarray[double,ndim=1] dS=np.empty((NP,)), dLr, dRPhir, Rref
    cdef np.ndarray[long,ndim=1] NL

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-c_pi,c_pi]
        DPhiMinMax = _TWOPI
    else:
        PhiMinMax = [c_atan2(c_sin(PhiMinMax[0]), c_cos(PhiMinMax[0])),
                     c_atan2(c_sin(PhiMinMax[1]), c_cos(PhiMinMax[1]))]
        if PhiMinMax[1]>=PhiMinMax[0]:
            DPhiMinMax = PhiMinMax[1]-PhiMinMax[0]
        else:
            DPhiMinMax = _TWOPI + PhiMinMax[1] - PhiMinMax[0]


    # Get the actual R and Z resolutions and mesh elements
    ptsCross, dLrRef, indL,\
      NL, RrefRef, VPbis = discretize_vpoly(VPoly, dL, D1=None, D2=None,
                                            margin=margin, DIn=DIn, VIn=VIn)
    Ln = dLrRef.size
    # Number of Phi per R
    dRPhirRef, dPhir, dRPhir = np.empty((Ln,)), np.empty((Ln,)), -np.ones((Ln,))
    dLr, Rref = -np.ones((Ln,)), -np.ones((Ln,))
    NRPhi, NRPhi0 = np.empty((Ln,),dtype=int), np.empty((Ln,),dtype=int)
    radius_ratio = int(c_ceil(np.max(RrefRef)/np.min(RrefRef)))
    for ii in range(0,Ln):
        NRPhi[ii] = <long>(c_ceil(DPhiMinMax*RrefRef[ii]/dRPhi))
        dRPhirRef[ii] = DPhiMinMax*RrefRef[ii]/<double>(NRPhi[ii])
        dPhir[ii] = DPhiMinMax/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((Ln,NRPhi[ii]*radius_ratio+1))
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
            pts[0,ii] = ptsCross[0,iiL]*c_cos(Phi[iiL,iiphi])
            pts[1,ii] = ptsCross[0,iiL]*c_sin(Phi[iiL,iiphi])
            pts[2,ii] = ptsCross[1,iiL]
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
            pts[0,ii] = ptsCross[0,iiL]
            pts[1,ii] = ptsCross[1,iiL]
            pts[2,ii] = Phi[iiL,iiphi]
            dS[ii] = dLrRef[iiL]*dRPhirRef[iiL]
            if dRPhir[iiL]==-1.:
                dRPhir[iiL] = dRPhirRef[iiL]
                dLr[iiL] = dLrRef[iiL]
                Rref[iiL] = RrefRef[iiL]
    return pts, dS, NL, dLr[dLr>-0.5], Rref[Rref>-0.5], \
      dRPhir[dRPhir>-0.5], <long>nRPhi0, VPbis



########################################################
########################################################
#       Meshing - Surface - TorStruct
########################################################

def _Ves_Smesh_TorStruct_SubFromD_cython(double[::1] PhiMinMax, double dL,
                                         double dRPhi,
                                         double[:,::1] VPoly,
                                         list DR=None,
                                         list DZ=None,
                                         double[::1] DPhi=None,
                                         double DIn=0., VIn=None,
                                         str Out='(X,Y,Z)',
                                         double margin=_VSMALL):
    """Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi),
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double Dphi, dR0r=0., dZ0r=0.
    cdef int NR0=0, NZ0=0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([c_atan2(c_sin(PhiMinMax[0]),
                                                  c_cos(PhiMinMax[0])),
                                           c_atan2(c_sin(PhiMinMax[1]),
                                                  c_cos(PhiMinMax[1]))])
    cdef np.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef np.ndarray[long,ndim=1] indR0, indZ0, iind, iindF, indM, NL, ind
    cdef np.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[]

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = np.array([-c_pi,c_pi])
        DPhiMinMax = _TWOPI
        Full = True
    else:
        PhiMinMax = np.array([c_atan2(c_sin(PhiMinMax[0]),c_cos(PhiMinMax[0])),
                              c_atan2(c_sin(PhiMinMax[1]),c_cos(PhiMinMax[1]))])
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0]\
          else _TWOPI + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any and make sure to replace them in the proper quadrant
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None \
          else c_atan2(c_sin(DPhi[0]),c_cos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None \
          else c_atan2(c_sin(DPhi[1]),c_cos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else _TWOPI+DPhi1-DPhi0

    inter, Bounds, Faces = _getBoundsinter2AngSeg(Full, PhiMinMax[0],
                                                  PhiMinMax[1], DPhi0, DPhi1)

    if inter:
        BC = list(Bounds)
        nBounds = len(Bounds)
        for ii in range(0,nBounds):
            if BC[ii][0]<PhiMinMax[0]:
                BC[ii][0] += _TWOPI
            if BC[ii][1]<=PhiMinMax[0]:
                BC[ii][1] += _TWOPI

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
                pts = np.array([ptsrz[0,:]*c_cos(phiMinMax[0]+Dphi),
                                ptsrz[0,:]*c_sin(phiMinMax[0]+Dphi),
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
                                                 DPhi=[DPhi0, DPhi1],
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
                pts = np.array([ptsrz[0,:]*c_cos(phiMinMax[1]-Dphi),
                                ptsrz[0,:]*c_sin(phiMinMax[1]-Dphi),
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
    cdef double[::1] phiMinMax = np.array([c_atan2(c_sin(PhiMinMax[0]),
                                                  c_cos(PhiMinMax[0])),
                                           c_atan2(c_sin(PhiMinMax[1]),
                                                  c_cos(PhiMinMax[1]))])
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
            pts = np.array([R0[indR0]*c_cos(phiMinMax[0]+Dphi),
                            R0[indR0]*c_sin(phiMinMax[0]+Dphi), Z0[indZ0]])
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
            pts = np.array([R0[indR0]*c_cos(phiMinMax[1]-Dphi),
                            R0[indR0]*c_sin(phiMinMax[1]-Dphi), Z0[indZ0]])
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

cdef inline int _check_DLvsLMinMax(double[::1] LMinMax,
                                   list DL=None):
    cdef int inter = 1
    cdef bint dl0_is_not_none
    cdef bint dl1_is_not_none
    if DL is not None:
        dl0_is_not_none = DL[0] is not None
        dl1_is_not_none = DL[1] is not None
        if len(DL) != 2 or LMinMax[0]>=LMinMax[1]:
            assert(False)
        if dl0_is_not_none and dl1_is_not_none and DL[0] >= DL[1]:
            assert(False)
        if dl0_is_not_none and DL[0]>LMinMax[1]:
            inter = 0
        elif dl1_is_not_none and DL[1]<LMinMax[0]:
            inter = 0
        else:
            if dl0_is_not_none and DL[0]<=LMinMax[0]:
                DL[0] = None
            if dl1_is_not_none and DL[1]>=LMinMax[1]:
                DL[1] = None
    return inter

def _Ves_Smesh_Lin_SubFromD_cython(double[::1] XMinMax, double dL, double dX,
                                   double[:,::1] VPoly,
                                   list DX=None,
                                   list DY=None,
                                   list DZ=None,
                                   double DIn=0., VIn=None,
                                   double margin=_VSMALL):
    """Return the desired surfacic submesh indicated by the limits (DX,DY,DZ),
    for the desired resolution (dX,dL) """
    cdef np.ndarray[double,ndim=1] X, Y0, Z0
    cdef double dXr, dY0r, dZ0r
    cdef int NY0, NZ0, Y0n, Z0n, NX, Xn, Ln, NR0, inter=1
    cdef np.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] dS, dLr, Rref
    cdef np.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ind

    # Preformat
    # Adjust limits
    interX = _check_DLvsLMinMax(XMinMax, DX)
    interY = _check_DLvsLMinMax(np.array([np.min(VPoly[0,:]),
                                          np.max(VPoly[0,:])]), DY)
    interZ = _check_DLvsLMinMax(np.array([np.min(VPoly[1,:]),
                                              np.max(VPoly[1,:])]), DZ)

    if interX==1 and interY==1 and interZ==1:

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


def _Ves_Smesh_Lin_SubFromInd_cython(double[::1] XMinMax, double dL, double dX,
                                     double[:,::1] VPoly,
                                     np.ndarray[long,ndim=1] ind,
                                     double DIn=0., VIn=None,
                                     double margin=_VSMALL):
    """ Return the desired surfacic submesh indicated by ind,
    for the desired resolution (dX,dL) """
    cdef double dXr, dY0r, dZ0r
    cdef int NX, NY0, NZ0, Ln, NR0, nii
    cdef list LPts, LdS
    cdef np.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef np.ndarray[double,ndim=1] X, Y0, Z0, dS, dLr, Rref
    cdef np.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ii

    # Get the mesh for the faces
    Y0, dY0r, _, NY0 = discretize_line1d(np.array([np.min(VPoly[0, :]),
                                                   np.max(VPoly[0,:])]),
                                         dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, _, NZ0 = discretize_line1d(np.array([np.min(VPoly[1, :]),
                                                   np.max(VPoly[1,:])]),
                                         dL, DL=None, Lim=True, margin=margin)

    # Get the actual R and Z resolutions and mesh elements
    X, dXr, _, NX = discretize_line1d(XMinMax, dX,
                                      DL=None, Lim=True, margin=margin)
    PtsCross, dLr, _, NL, Rref, VPbis = discretize_vpoly(VPoly, dL,
                                                           D1=None, D2=None,
                                                           margin=margin,
                                                           DIn=DIn, VIn=VIn)
    Ln = PtsCross.shape[1]

    LPts, LdS = [], []
    # First face
    ii = (ind<NY0*NZ0).nonzero()[0].astype(int)
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NY0
        indY0 = (ind[ii]-indZ0*NY0)
        if nii==1:
            LPts.append( np.array([[XMinMax[0] + DIn],
                                   [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[0] + DIn)*np.ones((nii,)),
                                   Y0[indY0], Z0[indZ0]]) )
        LdS.append( dY0r*dZ0r*np.ones((nii,)) )

    # Cylinder
    ii = ((ind>=NY0*NZ0) & (ind<NY0*NZ0+NX*Ln)).nonzero()[0].astype(int)
    nii = len(ii)
    if nii>0:
        indX = (ind[ii]-NY0*NZ0) // Ln
        indL = (ind[ii]-NY0*NZ0 - Ln*indX)
        if nii==1:
            LPts.append( np.array([[X[indX]],
                                   [PtsCross[0,indL]], [PtsCross[1,indL]]]) )
            LdS.append( np.array([dXr*dLr[indL]]) )
        else:
            LPts.append( np.array([X[indX],
                                   PtsCross[0,indL], PtsCross[1,indL]]) )
            LdS.append( dXr*dLr[indL] )

    # End face
    ii = (ind >= NY0*NZ0+NX*Ln).nonzero()[0].astype(int)
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-NY0*NZ0-NX*Ln) // NY0
        indY0 = ind[ii]-NY0*NZ0-NX*Ln - NY0*indZ0
        if nii==1:
            LPts.append( np.array([[XMinMax[1] - DIn],
                                   [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[1] - DIn)*np.ones((nii,)),
                                   Y0[indY0], Z0[indZ0]]) )
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

    Parameters
    ----------
    ray_orig : (3, nlos) double array
       LOS origin points coordinates
    ray_vdir : (3, nlos) double array
       LOS normalized direction vector
    ves_poly : (2, num_vertex) double array
       Coordinates of the vertices of the Polygon defining the 2D poloidal
       cut of the Vessel
    ves_norm : (2, num_vertex-1) double array
       Normal vectors going "inwards" of the edges of the Polygon defined
       by ves_poly
    nstruct_tot : int
       Total number of structures (counting each limited structure as one)
    ves_lims : array
       Contains the limits min and max of vessel
    lstruct_polyx : array
       List of x coordinates of the vertices of all structures on poloidal plane
       If no structures : None
    lstruct_polyy : array
       List of y coordinates of the vertices of all structures on poloidal plane
       If no structures : None
    lstruct_lims : array
       List of limits of all structures
       If no structures : None
    lstruct_nlim : array of ints
       List of number of limits for all structures
       If no structures : None
    lstruct_normx : array
       List of x coordinates of "inwards" normal vectors of the polygon of all
       the structures
       If no structures : None
    lstruct_normy : array
       List of y coordinates of "inwards" normal vectors of the polygon of all
       the structures
       If no structures : None
    rmin : double
       Minimal radius of vessel to take into consideration
    eps_<val> : double
       Small value, acceptance of error
    ves_type : string
       Type of vessel ("Tor" or "Lin")
    forbid : bool
       Should we forbid values behind visible radius ? (see rmin)
    test : bool
       Should we run tests ?
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.

    Returns
    -------
    coeff_inter_in : (nlos) array
       scalars level of "in" intersection of the LOS (if k=0 at origin)
    coeff_inter_out : (nlos) array
       scalars level of "out" intersection of the LOS (if k=0 at origin)
    vperp_out : (3, nlos) array
       Coordinates of the normal vector of impact of the LOS (NaN if none)
    ind_inter_out : (3, nlos)
       Index of structure impacted by LOS: ind_inter_out[:,ind_los]=(i,j,k)
       where k is the index of edge impacted on the j-th sub structure of the
       structure number i. If the LOS impacted the vessel i=j=0
    """
    cdef str vt_lower = ves_type.lower()
    cdef str error_message
    cdef int sz_ves_lims
    cdef int nlos = ray_orig.shape[1]
    cdef int npts_poly = ves_norm.shape[1]
    cdef bint bool1, bool2
    cdef double min_poly_r
    cdef array vperp_out = clone(array('d'), nlos * 3, True)
    cdef array coeff_inter_in  = clone(array('d'), nlos, True)
    cdef array coeff_inter_out = clone(array('d'), nlos, True)
    cdef array ind_inter_out = clone(array('i'), nlos * 3, True)
    cdef double[::1] lstruct_lims_np
    # == Testing inputs ========================================================
    if test:
        error_message = "ray_orig and ray_vdir must have the same shape: "\
                        + "(3,) or (3,NL)!"
        assert tuple(ray_orig.shape) == tuple(ray_vdir.shape) and \
          ray_orig.shape[0] == 3, error_message
        error_message = "ves_poly and ves_norm must have the same shape (2,NS)!"
        assert ves_poly.shape[0] == 2 and ves_norm.shape[0] == 2, error_message
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
        assert vt_lower in ['tor', 'lin'], error_message
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
    if ves_lims is not None:
        sz_ves_lims = np.size(ves_lims)
    else:
        sz_ves_lims = 0
    min_poly_r = _bgt.comp_min(ves_poly[0, ...], npts_poly-1)

    lstruct_lims_np = flatten_lstruct_lims(lstruct_lims)
    _rt.compute_inout_tot(nlos, npts_poly,
                          ray_orig, ray_vdir,
                          ves_poly, ves_norm,
                          lstruct_nlim, ves_lims,
                          lstruct_polyx, lstruct_polyy,
                          lstruct_lims_np, lstruct_normx,
                          lstruct_normy, lnvert,
                          nstruct_tot, nstruct_lim,
                          sz_ves_lims, min_poly_r, rmin,
                          eps_uz, eps_a, eps_vz, eps_b,
                          eps_plane, vt_lower=='tor',
                          forbid, num_threads,
                          coeff_inter_out, coeff_inter_in, vperp_out,
                          ind_inter_out)
    return np.asarray(coeff_inter_in), np.asarray(coeff_inter_out),\
           np.transpose(np.asarray(vperp_out).reshape(nlos,3)),\
           np.transpose(np.asarray(ind_inter_out,
                                   dtype=int).reshape(nlos, 3))


# =============================================================================
# = Ray tracing when we only want kMin / kMax
# -   (useful when working with flux surfaces)
# =============================================================================
def LOS_Calc_kMinkMax_VesStruct(double[:, ::1] ray_orig,
                                double[:, ::1] ray_vdir,
                                list ves_poly,
                                list ves_norm,
                                int num_surf,
                                long[::1] lnvert,
                                double[::1] ves_lims=None,
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
    ray_orig : (3, nlos) double array
       LOS origin points coordinates
    ray_vdir : (3, nlos) double array
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
       Should we forbid values behind visible radius ? (see rmin)
    test : bool
       Should we run tests ?
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    Return
    ======
    coeff_inter_in : (num_surf, nlos) array
       scalars level of "in" intersection of the LOS (if k=0 at origin) for
       each surface
       [kmin(surf0, los0), kmin(surf0, los1), ..., kmin(surf1, los0),....]
    coeff_inter_out : (num_surf, nlos) array
       scalars level of "out" intersection of the LOS (if k=0 at origin) for
       each surface
       [kmax(surf0, los0), kmax(surf0, los1), ..., kmax(surf1, los0),....]
    """
    cdef int npts_poly
    cdef int nlos = ray_orig.shape[1]
    cdef int ind_struct = 0
    cdef int ind_surf
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef double lim_min = 0.
    cdef double lim_max = 0.
    cdef double rmin2 = 0.
    cdef str error_message
    cdef bint forbidbis, forbid0
    cdef bint bool1, bool2
    cdef array coeff_inter_in  = clone(array('d'), nlos * num_surf, True)
    cdef array coeff_inter_out = clone(array('d'), nlos * num_surf, True)
    cdef int *llimits = NULL
    cdef long *lsz_lim = NULL
    cdef bint are_limited
    cdef double[2] lbounds_ves
    cdef double[2] lim_ves
    cdef double[:,::1] tmp_poly
    cdef double[:,::1] tmp_norm
    cdef double* ptr_coeff_in
    cdef double* ptr_coeff_out

    # initializations ...
    ptr_coeff_in = coeff_inter_in.data.as_doubles
    ptr_coeff_out = coeff_inter_out.data.as_doubles

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
            lbounds_ves[0] = c_atan2(c_sin(ves_lims[0]), c_cos(ves_lims[0]))
            lbounds_ves[1] = c_atan2(c_sin(ves_lims[1]), c_cos(ves_lims[1]))
        # -- Toroidal case -----------------------------------------------------
        for ind_surf in range(num_surf):
            # rmin is necessary to avoid looking on the other side of the tok
            if rmin < 0.:
                rmin = 0.95*min(np.min(ves_poly[ind_surf][0, ...]),
                                _bgt.comp_min_hypot(ray_orig[0, ...],
                                                    ray_orig[1, ...],
                                                    nlos))
            rmin2 = rmin*rmin
            # Variable to avoid looking "behind" blind spot of tore
            if forbid:
                forbid0, forbidbis = 1, 1
            else:
                forbid0, forbidbis = 0, 0
            # Getting size of poly
            npts_poly = lnvert[ind_surf]
            tmp_poly = ves_poly[ind_surf]
            tmp_norm = ves_norm[ind_surf]
            # -- Computing intersection between LOS and Vessel -----------------
            _rt.raytracing_minmax_struct_tor(nlos, ray_vdir, ray_orig,
                                             &ptr_coeff_out[ind_surf*nlos],
                                             &ptr_coeff_in[ind_surf*nlos],
                                             forbid0, forbidbis,
                                             rmin, rmin2, crit2_base,
                                             npts_poly, lbounds_ves,
                                             are_limited,
                                             &tmp_poly[0][0],
                                             &tmp_poly[1][0],
                                             &tmp_norm[0][0],
                                             &tmp_norm[1][0],
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
            tmp_poly = ves_poly[ind_surf]
            tmp_norm = ves_norm[ind_surf]
            _rt.raytracing_minmax_struct_lin(nlos, ray_orig, ray_vdir,
                                             npts_poly,
                                             &tmp_poly[0][0],
                                             &tmp_poly[1][0],
                                             &tmp_norm[0][0],
                                             &tmp_norm[1][0],
                                             lbounds_ves[0], lbounds_ves[1],
                                             &ptr_coeff_out[ind_surf*nlos],
                                             &ptr_coeff_in[ind_surf*nlos],
                                             eps_plane)

    return np.asarray(coeff_inter_in), np.asarray(coeff_inter_out)


def flatten_lstruct_lims(list lstruct_lims) -> double[::1]:
    """
    utilitary function to flatten lstruct_lims
    """
    cdef list flat_list
    cdef double[::1] lstruct_lims_np
    if lstruct_lims is None or np.size(lstruct_lims) == 0:
        lstruct_lims_np = np.array([C_NAN])
    else:
        flat_list = []
        for ele in lstruct_lims:
            if isinstance(ele, (list, np.ndarray)) and np.size(ele) > 1:
                for elele in ele:
                    if type(elele) is list:
                        flat_list += elele
                    else:
                        flat_list += elele.flatten().tolist()
            else:
                flat_list += [C_NAN]
        lstruct_lims_np = np.array(flat_list)

    return lstruct_lims_np


# =============================================================================
# = Tools to know if one or multiple points are visible from other points
# =============================================================================
def LOS_areVis_PtsFromPts_VesStruct(np.ndarray[double, ndim=2,mode='c'] pts1,
                                    np.ndarray[double, ndim=2,mode='c'] pts2,
                                    double[:, ::1] ves_poly=None,
                                    double[:, ::1] ves_norm=None,
                                    double[:, ::1] dist=None,
                                    double[::1] ves_lims=None,
                                    long[::1] lstruct_nlim=None,
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
                                    double eps_plane=_VSMALL,
                                    str ves_type='tor',
                                    bint forbid=True,
                                    bint test=True,
                                    int num_threads=16):
    """
    Return an array of booleans indicating whether each point in pts1 can see
    each point in pts2 considering vignetting a given
    configuration.
        pts1 : (3, npts1) cartesian coordinates of viewing points
        pts2 : (3, npts2) cartesian coordinates of points to check if viewable
        dist : optional argument : distance between the points pts1, pts2
        ves_* : vessel descriptors (poly, norm, limits)
        lstruct_* : config's structure descriptors (poly, limits, norms,
                    number of structures, ...)
        eps_* : values of precision in each direction
        forbid : boolean if true forbids checking "behind" the tokamak
        test : boolean check if input is valid or not
        num_threads : number of threads for parallelization
    Output:
        are_seen: (npts1, npts2) array of ints indicating if viewing points pts1
                  can see the other points pts2.
                  are_seen[i,j] = 1 if pts1[i] sees point pts2[j]
                                  0 else
    """
    cdef str msg
    cdef int npts1=pts1.shape[1]
    cdef int npts2=pts2.shape[1]
    cdef bint bool1, bool2
    cdef np.ndarray[long, ndim=2, mode='c'] are_seen = np.empty((npts1, npts2),
                                                                dtype=int)
    cdef double[::1] lstruct_lims_np
    # == Testing inputs ========================================================
    if test:
        msg = "ves_poly and ves_norm are not optional arguments"
        assert ves_poly is not None and ves_norm is not None, msg
        bool1 = (ves_poly.shape[0]==2 and ves_norm.shape[0]==2
              and ves_norm.shape[1]==ves_poly.shape[1]-1)
        msg = "Args ves_poly and ves_norm must be of the same shape (2,NS)!"
        assert bool1, msg
        bool1 = lstruct_lims is None or len(lstruct_normy) == len(lstruct_normx)
        bool2 = lstruct_normx is None or len(lstruct_polyx) == len(lstruct_polyy)
        msg = "Args lstruct_polyx, lstruct_polyy, lstruct_lims, lstruct_normx,"\
              + " lstruct_normy, must be None or lists of same len()!"
        assert bool1 and bool2, msg
        msg = "[eps_uz,eps_vz,eps_a,eps_b] must be floats < 1.e-4!"
        assert all([ee < 1.e-4 for ee in [eps_uz, eps_a,
                                          eps_vz, eps_b,
                                          eps_plane]]), msg
        msg = "ves_type must be a str in ['Tor','Lin']!"
        assert ves_type.lower() in ['tor', 'lin'], msg

    lstruct_lims_np = flatten_lstruct_lims(lstruct_lims)
    _rt.are_visible_vec_vec(pts1, npts1,
                            pts2, npts2,
                            ves_poly, ves_norm,
                            are_seen, dist, ves_lims,
                            lstruct_nlim,
                            lstruct_polyx, lstruct_polyy,
                            lstruct_lims_np,
                            lstruct_normx, lstruct_normy,
                            lnvert, nstruct_tot, nstruct_lim,
                            rmin, eps_uz, eps_a, eps_vz, eps_b,
                            eps_plane, ves_type.lower()=='tor',
                            forbid, num_threads)
    return are_seen


def LOS_isVis_PtFromPts_VesStruct(double pt0, double pt1, double pt2,
                                  np.ndarray[double, ndim=2,mode='c'] pts,
                                  double[::1] dist=None,
                                  double[:, ::1] ves_poly=None,
                                  double[:, ::1] ves_norm=None,
                                  double[::1] ves_lims=None,
                                  long[::1] lstruct_nlim=None,
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
                                  bint forbid=True,
                                  bint test=True,
                                  int num_threads=16):
    """
    Return an array of booleans indicating whether each point in pts is
    visible from the point P = [pt0, pt1, pt2] considering vignetting a given
    configuration.
        pt0 : x - coordinate of the viewing point P
        pt1 : y - coordinate of the viewing point P
        pt2 : z - coordinate of the viewing point P
        pts : (3, npts) cartesian coordinates of points to check if viewable
        dist : optional argument : distance between points and P
        ves_* : vessel descriptors (poly, norm, limits)
        lstruct_* : config's structure descriptors (poly, limits, norms,
                    number of structures, ...)
        eps_* : values of precision in each direction
        forbid : boolean if true forbids checking "behind" the tokamak
        test : boolean check if input is valid or not
        num_threads : number of threads for parallelization
    Output:
        is_seen: (npts1) array of ints indicating if viewing point P
                 can see the other points pts.
                 is_seen[i] = 1 if P sees point pts[i]
                              0 else
    """
    cdef str msg
    cdef int npts = pts.shape[1]
    cdef bint bool1, bool2
    cdef double[::1] lstruct_lims_np
    cdef np.ndarray[long, ndim=1, mode='c'] is_seen
    # == Testing inputs ========================================================
    if test:
        msg = "ves_poly and ves_norm are not optional arguments"
        assert ves_poly is not None and ves_norm is not None, msg
        bool1 = (ves_poly.shape[0]==2 and ves_norm.shape[0]==2
              and ves_norm.shape[1]==ves_poly.shape[1]-1)
        msg = "Args ves_poly and ves_norm must be of the same shape (2,NS)!"
        assert bool1, msg
        bool1 = lstruct_lims is None or len(lstruct_normy) == len(lstruct_normx)
        bool2 = lstruct_normx is None or len(lstruct_polyx) == len(lstruct_polyy)
        msg = "Args lstruct_polyx, lstruct_polyy, lstruct_lims, lstruct_normx,"\
              + " lstruct_normy, must be None or lists of same len()!"
        assert bool1 and bool2, msg
        msg = "[eps_uz,eps_vz,eps_a,eps_b] must be floats < 1.e-4!"
        assert all([ee < 1.e-4 for ee in [eps_uz, eps_a,
                                          eps_vz, eps_b,
                                          eps_plane]]), msg
        msg = "ves_type must be a str in ['Tor','Lin']!"
        assert ves_type.lower() in ['tor', 'lin'], msg
    # ...
    lstruct_lims_np = flatten_lstruct_lims(lstruct_lims)
    is_seen = np.empty((npts), dtype=int)
    _rt.is_visible_pt_vec(pt0, pt1, pt2,
                          pts, npts,
                          ves_poly, ves_norm,
                          &is_seen[0], dist, ves_lims,
                          lstruct_nlim,
                          lstruct_polyx, lstruct_polyy,
                          lstruct_lims_np,
                          lstruct_normx, lstruct_normy,
                          lnvert, nstruct_tot, nstruct_lim,
                          rmin, eps_uz, eps_a, eps_vz, eps_b,
                          eps_plane, ves_type.lower()=='tor',
                          forbid, num_threads)
    return is_seen


# ==============================================================================
#
#                                 VIGNETTING
#
# ==============================================================================
def triangulate_by_earclipping(np.ndarray[double,ndim=2] poly):
    cdef int nvert = poly.shape[1]
    cdef np.ndarray[long,ndim=1] ltri = np.empty((nvert-2)*3, dtype=int)
    cdef double* diff = NULL
    cdef bint* lref = NULL
    # Initialization ...........................................................
    diff = <double*>malloc(3*nvert*sizeof(double))
    lref = <bint*>malloc(nvert*sizeof(bint))
    _vt.compute_diff3d(&poly[0,0], nvert, diff)
    _vt.are_points_reflex(nvert, diff, lref)
    # Calling core function.....................................................
    _vt.earclipping_poly(&poly[0,0], &ltri[0], diff, lref, nvert)
    free(diff)
    free(lref)
    return ltri


def vignetting(double[:, ::1] ray_orig, double[:, ::1] ray_vdir,
               list vignett_poly, long[::1] lnvert, int num_threads=16):
    """
    ray_orig : (3, nlos) double array
       LOS origin points coordinates
    ray_vdir : (3, nlos) double array
       LOS normalized direction vector
    vignett_poly : (num_vign, 3, num_vertex) double list of arrays
       Coordinates of the vertices of the Polygon defining the 3D vignett.
       POLY CLOSED
    lnvert : (num_vign) long array
       Number of vertices for each vignett (without counting the rebound)
    Returns
    ======
    goes_through: (num_vign, nlos) bool array
       Indicates for each vignett if each LOS wents through or not
    """
    cdef int ii
    cdef int nvign, nlos
    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] goes_through
    cdef long** ltri = NULL
    cdef double* lbounds = NULL
    cdef double** data = NULL
    cdef bint* bool_res = NULL
    cdef np.ndarray[double, ndim=2, mode="c"] temp
    # -- Initialization --------------------------------------------------------
    nvign = len(vignett_poly)
    nlos = ray_orig.shape[1]
    goes_through = np.empty((nlos*nvign),dtype=bool)
    # re writting vignett_poly to C type:
    data = <double **> malloc(nvign*sizeof(double *))
    for ii in range(nvign):
        temp = vignett_poly[ii]
        data[ii] = &temp[0,0]
    # -- Preparation -----------------------------------------------------------
    lbounds = <double*>malloc(sizeof(double) * 6 * nvign)
    _rt.compute_3d_bboxes(data, &lnvert[0], nvign, &lbounds[0],
                          num_threads=num_threads)
    ltri = <long**>malloc(sizeof(long*)*nvign)
    _vt.triangulate_polys(data, &lnvert[0], nvign, ltri,
                          num_threads)
    # -- We call core function -------------------------------------------------
    bool_res = <bint*>malloc(nlos*nvign*sizeof(bint))
    _vt.vignetting_core(ray_orig, ray_vdir, data, &lnvert[0], &lbounds[0],
                        ltri, nvign, nlos, &bool_res[0],num_threads)
    for ii in range(nlos*nvign):
        goes_through[ii] = bool_res[ii]
    # -- Cleaning up -----------------------------------------------------------
    free(bool_res)
    free(lbounds)
    # We have to free each array for each vignett:
    for ii in range(nvign):
        free(ltri[ii])
    free(ltri) # and now we can free the main pointer
    free(data)
    return goes_through


# ==============================================================================
#
#                                  LOS SAMPLING
#
# ==============================================================================
def LOS_get_sample(int nlos, dL, double[:,::1] los_lims, str dmethod='abs',
                   str method='sum', bint Test=True, int num_threads=16):
    """
    Return the sampled line, with the specified method
    -   'sum' :     return N segments centers
    -   'simps':    return N+1 egdes, N even (for scipy.integrate.simps)
    -   'romb' :    return N+1 edges, N+1 = 2**k+1 (for scipy.integrate.romb)
      The dmethod defines if the discretization step given is absolute ('abs')
      or relative ('rel')
    Params
    ======
    dL: double or list of doubles
        If dL is a single double: discretization step for all LOS.
        Else dL should be a list of size nlos with the discretization
        step for each nlos.
    los_lims: (2, nlos) double array
        For each nlos, it given the maximum and minimum limits of the ray
    dmethod: string
        type of discretization step: 'abs' for absolute or 'rel' for relative
    method: string
        method of quadrature on the LOS
    Test: bool
        to indicate if tests should be done or not

    How to recompute Points coordinates from results
    -------
    k, res, lind = Los_get_sample(...)
    nbrepet = np.r_[lind[0], np.diff(lind), k.size - lind[-1]]
    kus = k * np.repeat(ray_vdir, nbrepet, axis=1)
    Pts = np.repeat(ray_orig, nbrepet, axis=1) + kus
    """
    cdef str error_message
    cdef str dmode = dmethod.lower()
    cdef str imode = method.lower()
    cdef int sz1_dls, sz2_dls
    cdef int sz_coeff
    cdef int n_imode, n_dmode
    cdef bint dl_is_list
    cdef bint bool1, bool2
    cdef double[::1] dl_view
    cdef np.ndarray[double,ndim=1] dLr
    cdef np.ndarray[double,ndim=1] coeff_arr
    cdef np.ndarray[long,ndim=1] los_ind
    cdef long* tmp_arr
    cdef double* los_coeffs = NULL
    cdef double** coeff_ptr = NULL
    cdef long* los_ind_ptr = NULL
    # .. ray_orig shape needed for testing and in algo .........................
    dLr = np.zeros((nlos,), dtype=float)
    los_ind = np.zeros((nlos,), dtype=int)
    dl_is_list = hasattr(dL, '__iter__')
    # .. verifying arguments ...................................................
    if Test:
        sz1_dls = los_lims.shape[0]
        sz2_dls = los_lims.shape[1]
        assert sz1_dls == 2, "Dim 0 of arg los_lims should be 2"
        error_message = "Args los_lims should have dim 1 = nlos"
        assert nlos == sz2_dls, error_message
        bool1 = not dl_is_list and dL > 0.
        bool2 = dl_is_list and len(dL)==nlos and np.all(dL>0.)
        assert bool1 or bool2, "Arg dL must be a double or a List, and dL >0.!"
        error_message = "Argument dmethod (discretization method) should be in"\
                        +" ['abs','rel'], for absolute or relative."
        assert dmode in ['abs','rel'], error_message
        error_message = "Wrong method of integration." \
                        + " Options are: ['sum','simps','romb', 'linspace']"
        assert imode in ['sum','simps','romb','linspace'], error_message
    # Init
    coeff_ptr = <double**> malloc(sizeof(double*))
    los_ind_ptr = <long*> malloc(nlos*sizeof(long))
    coeff_ptr[0] = NULL
    # Getting number of modes:
    n_dmode = _st.get_nb_dmode(dmode)
    n_imode = _st.get_nb_imode(imode)
    # -- Core functions --------------------------------------------------------
    if not dl_is_list:
        # Case with unique discretization step dL
        sz_coeff = _st.los_get_sample_core_const_res(nlos,
                                                     &los_lims[0,0],
                                                     &los_lims[1,0],
                                                     n_dmode, n_imode,
                                                     <double>dL,
                                                     &coeff_ptr[0],
                                                     &dLr[0],
                                                     &los_ind_ptr[0],
                                                     num_threads)
    else:
        # Case with different resolution for each LOS
        dl_view=dL
        _st.los_get_sample_core_var_res(nlos,
                                        &los_lims[0,0],
                                        &los_lims[1,0],
                                        n_dmode, n_imode,
                                        &dl_view[0],
                                        &coeff_ptr[0],
                                        &dLr[0],
                                        &los_ind_ptr[0],
                                        num_threads)
        sz_coeff = los_ind_ptr[nlos-1]
    coeffs  = np.copy(np.asarray(<double[:sz_coeff]>coeff_ptr[0]))
    indices = np.copy(np.asarray(<long[:nlos]>los_ind_ptr).astype(int))
    # -- freeing -----------------------------------------------------------
    if not los_ind_ptr == NULL:
        free(los_ind_ptr)
    if not coeff_ptr == NULL:
        if not coeff_ptr[0] == NULL:
            free(coeff_ptr[0])
        free(coeff_ptr)
    return coeffs, dLr, indices[:nlos-1]



######################################################################
#               Signal calculation
######################################################################

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


def LOS_calc_signal(func, double[:,::1] ray_orig, double[:,::1] ray_vdir, res,
                    double[:,::1] lims, str dmethod='abs',
                    str method='sum', bint ani=False,
                    t=None, fkwdargs={}, str minimize='calls',
                    bint Test=True, int num_threads=16):
    """ Compute the synthetic signal, minimizing either function calls or memory
    Params
    =====
    func : python function st. func(pts, t=None, vect=None) => data
           with pts : ndarray (3, npts) - points where function is evaluated
                vect : ndarray(3, npts) - if anisotropic signal vector of emiss.
                t: ndarray(m) - times where to compute the function
           returns: data : ndarray(nt,nraf) if nt = 1, the array must be 2D
                           values of func at pts, at given time
           func is the function to be integrated along the LOS
    ray_orig: ndarray (3, nlos) LOS origins
    ray_vdir: ndarray (3, nlos) LOS directional vector
    res: double or list of doubles
        If res is a single double: discretization step for all LOS.
        Else res should be a list of size nlos with the discretization
        step for each nlos.
    lims: (2, nlos) double array
        For each nlos, it given the maximum and minimum limits of the ray
    dmethod: string
        type of discretization step: 'abs' for absolute or 'rel' for relative
    method: string
        method of quadrature on the LOS
    ani : bool
        to indicate if emission is anisotropic or not
    t : None or array-like
        times where to integrate
    minimize: string
        "calls" : we use algorithm to minimize the calls to 'func' (default)
        "memory": we use algorithm to minimize memory used
        "hybrid": a mix of both methods
    Test : bool
        we test if the inputs are giving in a proper way.
    num_threads: int
        number of threads if we want to parallelize the code.
    """
    cdef str error_message
    cdef str dmode = dmethod.lower()
    cdef str imode = method.lower()
    cdef str minim = minimize.lower()
    cdef int jjp1
    cdef int sz1_ds
    cdef int sz1_us, sz2_us
    cdef int sz1_dls, sz2_dls
    cdef int n_imode, n_dmode
    cdef int nlos
    cdef int nt=0, ii, jj
    cdef bint res_is_list
    cdef bint C0, C1
    cdef list ltime
    cdef double loc_r
    cdef long[1] nb_rows
    cdef long[::1] indbis
    cdef double[1] loc_eff_res
    cdef double[::1] reseff_mv
    cdef double[::1] res_mv
    cdef double[:,::1] val_mv
    cdef double[:,::1] pts_mv
    cdef double[:,::1] usbis_mv
    cdef double[:,::1] val_2d
    cdef np.ndarray[double,ndim=2] usbis
    cdef np.ndarray[double,ndim=2] pts
    cdef np.ndarray[double,ndim=2, mode='fortran'] sig
    cdef np.ndarray[double,ndim=1] reseff
    cdef np.ndarray[double,ndim=1] k
    cdef np.ndarray[double,ndim=1] res_arr
    cdef np.ndarray[long,ndim=1] ind
    cdef long* ind_arr = NULL
    cdef double* reseff_arr = NULL
    cdef double** coeff_ptr = NULL
    # .. ray_orig shape needed for testing and in algo .........................
    sz1_ds = ray_orig.shape[0]
    nlos = ray_orig.shape[1]
    res_is_list = hasattr(res, '__iter__')
    # .. verifying arguments ...................................................
    if Test:
        sz1_us = ray_vdir.shape[0]
        sz2_us = ray_vdir.shape[1]
        sz1_dls = lims.shape[0]
        sz2_dls = lims.shape[1]
        assert sz1_ds == 3, "Dim 0 of arg ray_orig should be 3"
        assert sz1_us == 3, "Dim 0 of arg ray_vdir should be 3"
        assert sz1_dls == 2, "Dim 0 of arg lims should be 2"
        error_message = ("Args ray_orig, ray_vdir, and lims "
                         + "should have same dimension 1")
        assert nlos == sz2_us == sz2_dls, error_message
        C0 = not res_is_list and res > 0.
        C1 = res_is_list and len(res)==nlos and np.all(res>0.)
        assert C0 or C1, "Arg res must be a double or a List, and all res >0.!"
        error_message = "Argument dmethod (discretization method) should be in"\
                        +" ['abs','rel'], for absolute or relative."
        assert dmode in ['abs','rel'], error_message
        error_message = "Wrong method of integration." \
                        + " Options are: ['sum','simps','romb']"
        assert imode in ['sum','simps','romb'], error_message
        error_message = "Wrong minimize optimization."\
                        + " Options are: ['calls','memory','hybrid']"
        assert minim in ['calls','memory','hybrid'], error_message
    # -- Preformat output signal -----------------------------------------------
    if t is None:
        if minim == 'memory':
            minim = 'calls'
            error_message = "If t is None !"
            error_message += "  => there is no point in using minimize='memory'"
            error_message += "  => switching to minimize = '%s'"%minim
            warn(error_message)
        nt = 1
        ltime = [None]
    elif not hasattr(t,'__iter__'):
        nt = 1
        ltime = [t]
    else:
        nt = len(t)
        if isinstance(t, list):
            ltime = t
        else:
            ltime = t.tolist()
    # -- Inizializations -------------------------------------------------------
    # Getting number of modes:
    n_dmode = _st.get_nb_dmode(dmode)
    n_imode = _st.get_nb_imode(imode)
    # Initialization result
    sig = np.empty((nt, nlos), dtype=float, order='F')
    # If the resolution is the same for every LOS, we create a tab
    if res_is_list :
        res_arr = np.asarray(res)
    else:
        res_arr = np.ones((nlos,), dtype=float) * res
    res_mv = res_arr
    # --------------------------------------------------------------------------
    # Minimize function calls: sample (vect), call (once) and integrate
    if minim == 'calls':
        if n_imode != 0:
            # Integration mode is Simpson or Romberg
            # Discretize all LOS
            k, reseff, ind = LOS_get_sample(nlos, res_arr, lims,
                                            dmethod=dmode, method=imode,
                                            num_threads=num_threads, Test=Test)
            nbrep = np.r_[ind[0], np.diff(ind), k.size - ind[nlos-2]]
            # get pts and values
            usbis = np.repeat(ray_vdir, nbrep, axis=1)
            pts = np.repeat(ray_orig, nbrep, axis=1) + k[None, :]*usbis
            # memory view:
            reseff_mv = reseff
            indbis = np.concatenate(([0],ind,[k.size]))
        else:
            coeff_ptr = <double**>malloc(sizeof(double*))
            coeff_ptr[0] = NULL
            reseff_arr = <double*>malloc(nlos*sizeof(double))
            ind_arr = <long*>malloc(nlos*sizeof(long))
            # .. we sample lines of sight ......................................
            _st.los_get_sample_core_var_res(nlos,
                                            &lims[0, 0],
                                            &lims[1, 0],
                                            n_dmode, n_imode,
                                            &res_arr[0],
                                            &coeff_ptr[0],
                                            &reseff_arr[0],
                                            &ind_arr[0],
                                            num_threads)
            sz_coeff = ind_arr[nlos-1]
            pts = np.empty((3,sz_coeff))
            usbis = np.empty((3,sz_coeff))
            usbis_mv = usbis
            pts_mv = pts
            _st.los_get_sample_pts(nlos,
                                   &pts_mv[0,0],
                                   &pts_mv[1,0],
                                   &pts_mv[2,0],
                                   &usbis_mv[0,0],
                                   &usbis_mv[1,0],
                                   &usbis_mv[2,0],
                                   ray_orig, ray_vdir,
                                   coeff_ptr[0],
                                   ind_arr,
                                   num_threads)
            # ..................................................................
        if ani:
            val_2d = func(pts, t=t, vect=-usbis, **fkwdargs)
        else:
            val_2d = func(pts, t=t, **fkwdargs)

        # Integrate
        if n_imode == 0:  # "sum" integration mode
            # .. integrating function ..........................................
            reseffs = np.copy(np.asarray(<double[:nlos]>reseff_arr))
            indices = np.copy(np.asarray(<long[:nlos-1]>ind_arr).astype(int))
            sig = np.asfortranarray(np.add.reduceat(val_2d,
                                                    np.r_[0, indices],
                                                    axis=-1)
                                    * reseffs[None, :])
            # Cleaning up...
            free(coeff_ptr[0])
            free(coeff_ptr)
            free(reseff_arr)
            free(ind_arr)
        elif n_imode == 1:  # "simpson" integration mode
            for ii in range(nlos):
                jj = indbis[ii]
                jjp1 = indbis[ii+1]
                val_mv = val_2d[:,jj:jjp1]
                loc_r = reseff_mv[ii]
                sig[:,ii] = scpintg.simps(val_mv,
                                             x=None, dx=loc_r, axis=-1)
        else:  # Romberg integration mode
            for ii in range(nlos):
                sig[:,ii] = scpintg.romb(val_2d[:,indbis[ii]:indbis[ii+1]],
                                         dx=reseff_mv[ii], axis=1, show=False)
    # --------------------------------------------------------------------------
    # Minimize memory use: loop everything, starting with LOS
    # then pts then time
    elif minim == 'memory':
        # loop over LOS and parallelize
        if ani:
            if n_imode == 0:  # sum integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode,
                                                                n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:, ii:ii+1],
                                                                ray_vdir[:, ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], vect=-usbis, **fkwdargs)
                        sig[jj, ii] = np.sum(val)*loc_eff_res[0]
            elif n_imode == 1:  # simpson integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode, n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:,ii:ii+1],
                                                                ray_vdir[:,ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], vect=-usbis, **fkwdargs)
                        sig[jj, ii] = scpintg.simps(val, x=None,
                                                    dx=loc_eff_res[0])
            elif n_imode == 2:  # romberg integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode, n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:,ii:ii+1],
                                                                ray_vdir[:,ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], vect=-usbis, **fkwdargs)
                        sig[jj, ii] = scpintg.romb(val, show=False,
                                                   dx=loc_eff_res[0])
        else:
            # -- not anisotropic -----------------------------------------------
            if n_imode == 0:  # "sum" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0, ii],
                                                     lims[1, ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:,ii:ii+1],
                                                     ray_vdir[:,ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], **fkwdargs)
                        sig[jj, ii] = np.sum(val)*loc_eff_res[0]
            elif n_imode == 1:  # "simpson" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0,ii],
                                                     lims[1,ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:,ii:ii+1],
                                                     ray_vdir[:,ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], **fkwdargs)
                        sig[jj, ii] = scpintg.simps(val, x=None,
                                                    dx=loc_eff_res[0])
            elif n_imode == 2:  # "romberg" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0, ii],
                                                     lims[1, ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:,ii:ii+1],
                                                     ray_vdir[:,ii:ii+1])
                    # loop over time for calling and integrating
                    for jj in range(nt):
                        val = func(pts, t=ltime[jj], **fkwdargs)
                        sig[jj, ii] = scpintg.romb(val, show=False,
                                                   dx=loc_eff_res[0])
    # --------------------------------------------------------------------------
    # HYBRID method: Minimize memory and calls (compromise): loop everything,
    # starting with LOS, call func only once for each los (treat all times)
    # loop over time for integrals
    else:
        # loop over LOS
        if ani:
            if n_imode == 0:  # sum integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode,
                                                                n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:, ii:ii+1],
                                                                ray_vdir[:, ii:ii+1])
                    val_2d = func(pts, t=t, vect=-usbis, **fkwdargs)
                    sig[:, ii] = np.sum(val_2d, axis=-1)*loc_eff_res[0]
            elif n_imode == 1:  # simpson integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode, n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:, ii:ii+1],
                                                                ray_vdir[:, ii:ii+1])
                    val = func(pts, t=t, vect=-usbis, **fkwdargs)
                    # integration
                    sig[:, ii] = scpintg.simps(val, x=None, axis=-1,
                                               dx=loc_eff_res[0])
            elif n_imode == 2:  # romberg integration mode
                for ii in range(nlos):
                    pts, usbis = _st.call_get_sample_single_ani(lims[0, ii],
                                                                lims[1, ii],
                                                                res_mv[ii],
                                                                n_dmode,
                                                                n_imode,
                                                                &loc_eff_res[0],
                                                                &nb_rows[0],
                                                                ray_orig[:, ii:ii+1],
                                                                ray_vdir[:, ii:ii+1])
                    val = func(pts, t=t, vect=-usbis, **fkwdargs)
                    sig[:, ii] = scpintg.romb(val, show=False, axis=1,
                                               dx=loc_eff_res[0])
        else:
            # -- not anisotropic -----------------------------------------------
            if n_imode == 0:  # "sum" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0, ii],
                                                     lims[1, ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:, ii:ii+1],
                                                     ray_vdir[:, ii:ii+1])
                    val_2d = func(pts, t=t, **fkwdargs)
                    sig[:, ii] = np.sum(val_2d, axis=-1)*loc_eff_res[0]
            elif n_imode == 1:  # "simpson" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0, ii],
                                                     lims[1, ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:, ii:ii+1],
                                                     ray_vdir[:, ii:ii+1])
                    val = func(pts, t=t, **fkwdargs)
                    sig[:, ii] = scpintg.simps(val, x=None, axis=-1,
                                                dx=loc_eff_res[0])
            elif n_imode == 2:  # "romberg" integration mode
                for ii in range(nlos):
                    pts = _st.call_get_sample_single(lims[0, ii],
                                                     lims[1, ii],
                                                     res_mv[ii],
                                                     n_dmode, n_imode,
                                                     &loc_eff_res[0],
                                                     &nb_rows[0],
                                                     ray_orig[:, ii:ii+1],
                                                     ray_vdir[:, ii:ii+1])
                    val = func(pts, t=t, **fkwdargs)
                    sig[:, ii] = scpintg.romb(val, show=False, axis=1,
                                              dx=loc_eff_res[0])
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
    intersection ligne et cercle
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
        if any([0 <= kk <= kOut for kk in KK]):
            KK = [kk for kk in KK if 0 <= kk <= kOut]
            Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
            Pk2D = [(c_sqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
            rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
            kPMin = KK[rk.index(min(rk))]
        else:
            kPMin = min([c_abs(kk) for kk in KK])  # Else, take the one closest to D
    else:
        Pk = [(D0+kk*u0,D1+kk*u1,D2+kk*u2) for kk in KK]
        Pk2D = [(c_sqrt(pp[0]**2+pp[1]**2), pp[2]) for pp in Pk]
        rk = [(pp[0]-RZ0)**2+(pp[1]-RZ1)**2 for pp in Pk2D]
        kPMin = KK[rk.index(min(rk))]
    return kPMin # + distance au cercle



cdef LOS_sino_Tor(double D0, double D1, double D2, double u0, double u1,
                  double u2, double RZ0, double RZ1, str Mode='LOS', double
                  kOut=C_INF):

    cdef double    uN = c_sqrt(u0**2+u1**2+u2**2), uParN = c_sqrt(u0**2+u1**2), DParN = c_sqrt(D0**2+D1**2)
    cdef double    Sca = u0*D0+u1*D1+u2*D2, ScaP = u0*D0+u1*D1
    cdef double    kPMin
    if uParN == 0.:
        kPMin = (RZ1-D2)/u2
    else:
        kPMin = LOS_sino_findRootkPMin_Tor(uParN, uN, Sca, RZ0, RZ1, ScaP, DParN, kOut, D0, D1, D2, u0, u1, u2, Mode=Mode)
    cdef double    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef double    PMin2norm = c_sqrt(PMin0**2+PMin1**2)
    cdef double    PMin2D0 = PMin2norm, PMin2D1 = PMin2
    cdef double    RMin = c_sqrt((PMin2D0-RZ0)**2+(PMin2D1-RZ1)**2)
    cdef double    eTheta0 = -PMin1/PMin2norm, eTheta1 = PMin0/PMin2norm, eTheta2 = 0.
    cdef double    vP0 = PMin2D0-RZ0, vP1 = PMin2D1-RZ1
    cdef double    Theta = c_atan2(vP1,vP0)
    cdef double    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef double    er2D0 = c_cos(ImpTheta), er2D1 = c_sin(ImpTheta)
    cdef double    p = vP0*er2D0 + vP1*er2D1
    cdef double    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef double    phi = c_asin(-uN0*eTheta0 -uN1*eTheta1 -uN2*eTheta2)
    return (PMin0,PMin1,PMin2), kPMin, RMin, Theta, p, ImpTheta, phi



cdef inline void NEW_LOS_sino_Tor(double orig0, double orig1, double orig2,
                                  double dirv0, double dirv1, double dirv2,
                                  double circ_radius, double circ_normz,
                                  double[9] results,
                                  bint is_LOS_Mode=False,
                                  double kOut=C_INF) nogil:
    cdef double[3] dirv, orig
    cdef double[2] res
    cdef double normu, normu_sqr
    cdef double kPMin

    normu_sqr = dirv0 * dirv0 + dirv1 * dirv1 + dirv2 * dirv2
    normu = c_sqrt(normu_sqr)
    dirv[0] = dirv0
    dirv[2] = dirv2
    dirv[1] = dirv1
    orig[0] = orig0
    orig[1] = orig1
    orig[2] = orig2

    if dirv0 == 0. and dirv1 == 0.:
        kPMin = (circ_normz-orig2)/dirv2
    else:
        _dt.dist_los_circle_core(dirv, orig,
                                circ_radius, circ_normz,
                                normu_sqr, res)
        kPMin = res[0]
        if is_LOS_Mode and kPMin > kOut:
            kPMin = kOut

    # Computing the point's coordinates.........................................
    cdef double PMin0 = orig0 + kPMin * dirv0
    cdef double PMin1 = orig1 + kPMin * dirv1
    cdef double PMin2 = orig2 + kPMin * dirv2
    cdef double PMin2norm = c_sqrt(PMin0**2+PMin1**2)
    cdef double RMin = c_sqrt((PMin2norm - circ_radius)**2
                             + (PMin2   - circ_normz)**2)
    cdef double vP0 = PMin2norm - circ_radius
    cdef double vP1 = PMin2     - circ_normz
    cdef double Theta = c_atan2(vP1, vP0)
    cdef double ImpTheta = Theta if Theta>=0 else Theta + c_pi
    cdef double er2D0 = c_cos(ImpTheta)
    cdef double er2D1 = c_sin(ImpTheta)
    cdef double p0 = vP0*er2D0 + vP1*er2D1
    cdef double eTheta0 = -PMin1 / PMin2norm
    cdef double eTheta1 =  PMin0 / PMin2norm
    cdef double normu0 = dirv0/normu
    cdef double normu1 = dirv1/normu
    cdef double phi = c_asin(-normu0 * eTheta0 - normu1 * eTheta1)
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

cdef inline void NEW_los_sino_tor_vec(int nlos,
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
    cdef double kPMin, PMin2norm, Theta
    cdef double vP0 = 0.0
    cdef double vP1 = 0.0
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
        for ind_los in prange(nlos):
            dirv[0] = directions[0, ind_los]
            dirv[1] = directions[1, ind_los]
            dirv[2] = directions[2, ind_los]
            orig[0] = origins[0, ind_los]
            orig[1] = origins[1, ind_los]
            orig[2] = origins[2, ind_los]
            normu_sq = dirv[0] * dirv[0] + dirv[1] * dirv[1] + dirv[2] * dirv[2]
            normu = c_sqrt(normu_sq)
            # Computing coeff of closest on line................................
            if dirv[0] == 0. and dirv[1] == 0.:
                kPMin = (circ_normz-orig[2])/dirv[2]
            else:
                _dt.dist_los_circle_core(dirv, orig, circ_radius,
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
            PMin2norm = c_sqrt(PMin0**2+PMin1**2)
            circle_closest_rmin[ind_los] = c_sqrt((PMin2norm - circ_radius)**2
                                        + (PMin2   - circ_normz)**2)
            # Theta and ImpTheta:
            vP0 = PMin2norm - circ_radius
            vP1 = PMin2     - circ_normz
            Theta = c_atan2(vP1, vP0)
            circle_closest_theta[ind_los] = Theta
            if Theta < 0:
                Theta = Theta + c_pi
            circle_closest_imptheta[ind_los] = Theta
            circle_closest_p[ind_los] = vP0 * c_cos(Theta) + vP1 * c_sin(Theta)
            # Phi:
            eTheta0 = - PMin1 / PMin2norm
            eTheta1 =   PMin0 / PMin2norm
            normu0 = dirv[0]/normu
            normu1 = dirv[1]/normu
            circle_closest_phi[ind_los] = c_asin(-normu0 * eTheta0 - normu1 * eTheta1)
        free(dirv)
        free(orig)
        free(res)
    return



cdef LOS_sino_Lin(double D0, double D1, double D2, double u0, double u1, double
                  u2, double RZ0, double RZ1, str Mode='LOS',
                  double kOut=C_INF):
    cdef double    kPMin
    if u0**2==1.:
        kPMin = 0.
    else:
        kPMin = ( (RZ0-D1)*u1+(RZ1-D2)*u2 ) / (1-u0**2)
    kPMin = kOut if Mode=='LOS' and kPMin > kOut else kPMin
    cdef double    PMin0 = D0+kPMin*u0, PMin1 = D1+kPMin*u1, PMin2 = D2+kPMin*u2
    cdef double    RMin = c_sqrt((PMin1-RZ0)**2+(PMin2-RZ1)**2)
    cdef double    vP0 = PMin1-RZ0, vP1 = PMin2-RZ1
    cdef double    Theta = c_atan2(vP1,vP0)
    cdef double    ImpTheta = Theta if Theta>=0 else Theta + np.pi
    cdef double    er2D0 = c_cos(ImpTheta), er2D1 = c_sin(ImpTheta)
    cdef double    p0 = vP0*er2D0 + vP1*er2D1
    cdef double    uN = c_sqrt(u0**2+u1**2+u2**2)
    cdef double    uN0 = u0/uN, uN1 = u1/uN, uN2 = u2/uN
    cdef double    phi = c_atan2(uN0, c_sqrt(uN1**2+uN2**2))
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
    cdef array k
    cdef np.ndarray[double, ndim=1, mode='c'] vis
    cdef double[::1] k_view
    cdef double[::1] lspolyx, lspolyy
    cdef double[::1] lsnormx, lsnormy
    if block:
        ind = ~_Ves_isInside(pts, VPoly, ves_lims=VLim, ves_type=VType,
                             in_format='(X,Y,Z)', test=Test)
        if LSPoly is not None:
            for ii in range(len(LSPoly)):
                ind = ind & _Ves_isInside(pts, LSPoly[ii], ves_lims=LSLim[ii],
                                          ves_type=VType, in_format='(X,Y,Z)',
                                          test=Test)
            lspolyx = LSPoly[0,...]
            lspolyy = LSPoly[1,...]
            lsnormx = LSVIn[0,...]
            lsnormy = LSVIn[1,...]
        else:
            lspolyx = None
            lspolyy = None
            lsnormx = None
            lsnormy = None
        ind = (~ind).nonzero()[0]
        ptstemp = np.ascontiguousarray(pts[:,ind])
        nptsok = ind.size
        k = clone(array('d'), nptsok, True)
        k_view = k
        if approx and out_coefonly:
            for ii in range(nt):
                _bgt.compute_dist_pt_vec(pos[0,ii], pos[1,ii],
                                         pos[2,ii], nptsok,
                                         ptstemp, &k_view[0])
                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii], ptstemp,
                                                    k=k_view,
                                                    ves_poly=VPoly,
                                                    ves_norm=VIn, ves_lims=VLim,
                                                    lstruct_polyx=lspolyx,
                                                    lstruct_polyy=lspolyy,
                                                    lstruct_lims=LSLim,
                                                    lstruct_normx=lsnormx,
                                                    lstruct_normy=lsnormy,
                                                    forbid=Forbid,
                                                    ves_type=VType, test=Test)
                for jj in range(nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = c_pi/k_view[jj]**2
        elif approx:
            for ii in range(nt):
                _bgt.compute_dist_pt_vec(pos[0,ii], pos[1,ii],
                                         pos[2,ii], nptsok,
                                         ptstemp, &k_view[0])
                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii], ptstemp,
                                                    ves_poly=VPoly,
                                                    k=k_view,
                                                    ves_norm=VIn, ves_lims=VLim,
                                                    lstruct_polyx=lspolyx,
                                                    lstruct_polyy=lspolyy,
                                                    lstruct_lims=LSLim,
                                                    lstruct_normx=lsnormx,
                                                    lstruct_normy=lsnormy,
                                                    forbid=Forbid,
                                                    ves_type=VType, test=Test)
                pir2 = c_pi*r[ii]**2
                for jj in range(nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = pir2/k_view[jj]**2
        else:
            pir2 = _TWOPI
            for ii in range(nt):
                _bgt.compute_dist_pt_vec(pos[0,ii], pos[1,ii],
                                         pos[2,ii], nptsok,
                                         ptstemp, &k_view[0])
                vis = LOS_isVis_PtFromPts_VesStruct(pos[0,ii], pos[1,ii],
                                                    pos[2,ii],
                                                    ptstemp,
                                                    ves_poly=VPoly,
                                                    k=k_view,
                                                    ves_norm=VIn, ves_lims=VLim,
                                                    lstruct_polyx=lspolyx,
                                                    lstruct_polyy=lspolyy,
                                                    lstruct_lims=LSLim,
                                                    lstruct_normx=lsnormx,
                                                    lstruct_normy=lsnormy,
                                                    forbid=Forbid,
                                                    ves_type=VType, test=Test)
                for jj in range(0,nptsok):
                    if vis[jj]:
                        sang[ii,ind[jj]] = pir2*(1-c_sqrt(1-r[ii]**2/k[jj]**2))

    else:
        if approx and out_coefonly:
            for ii in range(nt):
                for jj in range(npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[1,ii]-pts[1,jj])**2
                            + (pos[2,ii]-pts[2,jj])**2)
                    sang[ii,jj] = c_pi/dij2
        elif approx:
            for ii in range(nt):
                pir2 = c_pi*r[ii]**2
                for jj in range(npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2)
                    sang[ii,jj] = pir2/dij2
        else:
            pir2 = _TWOPI
            for ii in range(nt):
                for jj in range(npts):
                    dij2 = ((pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2
                            + (pos[0,ii]-pts[0,jj])**2)
                    sang[ii,jj] = pir2*(1-c_sqrt(1-r[ii]**2/dij2))
    return sang


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
    Python, if you need it from cython, use `dist_los_circle_core`
    """
    cdef double[2] res
    _dt.dist_los_circle_core(<double*>ray_vdir.data,
                            <double*>ray_orig.data,
                            radius, circ_z, norm_dir, res)
    return np.asarray(res)

def comp_dist_los_circle_vec(int nlos, int ncircles,
                             np.ndarray[double,ndim=2,mode='c'] dirs,
                             np.ndarray[double,ndim=2,mode='c'] oris,
                             np.ndarray[double,ndim=1,mode='c'] circle_radius,
                             np.ndarray[double,ndim=1,mode='c'] circle_z,
                             np.ndarray[double,ndim=1,mode='c'] norm_dir = None,
                             int num_threads=48):
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
    Python, if you need it from cython, use `dist_los_circle_core`
    """
    cdef array kmin_tab = clone(array('d'), nlos*ncircles, True)
    cdef array dist_tab = clone(array('d'), nlos*ncircles, True)

    if norm_dir is None:
        norm_dir = -np.ones(nlos)
    _dt.comp_dist_los_circle_vec_core(nlos, ncircles,
                                      <double*>dirs.data,
                                      <double*>oris.data,
                                      <double*>circle_radius.data,
                                      <double*>circle_z.data,
                                      <double*>norm_dir.data,
                                      kmin_tab, dist_tab, num_threads)
    return np.asarray(kmin_tab).reshape(nlos, ncircles), \
        np.asarray(dist_tab).reshape(nlos, ncircles)


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
    Python, if you need it from cython, use `is_los_close_circle_core`
    """
    return _dt.is_close_los_circle_core(<double*>ray_vdir.data,
                                        <double*>ray_orig.data,
                                        radius, circ_z, norm_dir, eps)

def is_close_los_circle_vec(int nlos, int ncircles, double epsilon,
                             np.ndarray[double,ndim=2,mode='c'] dirs,
                             np.ndarray[double,ndim=2,mode='c'] oris,
                             np.ndarray[double,ndim=1,mode='c'] circle_radius,
                             np.ndarray[double,ndim=1,mode='c'] circle_z,
                             np.ndarray[double,ndim=1,mode='c'] norm_dir=None,
                             int num_threads=48):
    """
    This function checks if at maximum a LOS is at a distance epsilon
    form a cirlce. Vectorial version
    The result is True when distance < epsilon
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from cython, use `is_los_close_circle_core`
    """
    cdef array res = clone(array('i'), nlos, True)

    if norm_dir is None:
        norm_dir = -np.ones(nlos)
    _dt.is_close_los_circle_vec_core(nlos, ncircles,
                                     epsilon,
                                     <double*>dirs.data,
                                     <double*>oris.data,
                                     <double*>circle_radius.data,
                                     <double*>circle_z.data,
                                     <double*>norm_dir.data,
                                     res, num_threads)
    return np.asarray(res, dtype=bool).reshape(nlos, ncircles)

# ==============================================================================
#
#                       DISTANCE BETWEEN LOS AND EXT-POLY
#
# ==============================================================================
def comp_dist_los_vpoly(double[:, ::1] ray_orig,
                        double[:, ::1] ray_vdir,
                        double[:, ::1] ves_poly,
                        double disc_step=0.1,
                        double eps_uz=_SMALL, double eps_a=_VSMALL,
                        double eps_vz=_VSMALL, double eps_b=_VSMALL,
                        double eps_plane=_VSMALL, str ves_type='Tor',
                        int num_threads=16, bint debug=False,
                        int debug_nlos=-1):
    """
    This function computes the distance (and the associated k) between nlos
    Rays (or LOS) and an `IN` structure (a polygon extruded around the axis
    (0,0,1), eg. a flux surface).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
           LOS normalized direction vector
        ves_poly : (2, num_vertex) double array
           Coordinates of the vertices of the Polygon defining the 2D poloidal
           cut of the Vessel
        eps_<val> : double
           Small value, acceptance of error
    Returns
    =======
        kmin_vpoly : (nlos) double array
            Of the form [k_0, k_1, ..., k_n], where k_i is the coefficient
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (nlos) double array
            `distance[i]` is the distance from P_i to the extruded polygon.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from cython, use `simple_dist_los_vpoly_core`
    """
    cdef int npts_poly = ves_poly.shape[1]
    cdef int nlos = ray_orig.shape[1]
    cdef int ii, ind_vert, ind_los
    cdef double* res_loc = NULL
    cdef double* loc_org = NULL
    cdef double* loc_dir = NULL
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef np.ndarray[double,ndim=1] dist = np.empty((nlos,),dtype=float)
    cdef np.ndarray[double,ndim=1] kmin = np.empty((nlos,),dtype=float)
    cdef double* list_vpoly_x = NULL
    cdef double* list_vpoly_y = NULL
    cdef int new_npts_poly
    # == Discretizing vpolys ===================================================
    _st.simple_discretize_vpoly_core(ves_poly,
                                    npts_poly,
                                    disc_step, # discretization step
                                    &list_vpoly_x,
                                    &list_vpoly_y,
                                    &new_npts_poly,
                                    0, # mode = absolute
                                    _VSMALL)
    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
        # We use local arrays for each thread so...
        loc_org   = <double *> malloc(sizeof(double) * 3)
        loc_dir   = <double *> malloc(sizeof(double) * 3)
        res_loc = <double *> malloc(2*sizeof(double))
        # == The parallelization over the LOS ==================================
        for ind_los in prange(nlos, schedule='dynamic'):
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
            _dt.simple_dist_los_vpoly_core(loc_org, loc_dir,
                                           list_vpoly_x,
                                           list_vpoly_y,
                                           new_npts_poly, upscaDp,
                                           upar2, dpar2,
                                           invuz, crit2,
                                           eps_uz, eps_vz,
                                           eps_a, eps_b,
                                           res_loc)
            kmin[ind_los] = res_loc[0]
            dist[ind_los] = res_loc[1]
        free(loc_org)
        free(loc_dir)
        free(res_loc)
    free(list_vpoly_x)
    free(list_vpoly_y)
    return kmin, dist

def comp_dist_los_vpoly_vec(int nvpoly, int nlos,
                            np.ndarray[double,ndim=2,mode='c'] ray_orig,
                            np.ndarray[double,ndim=2,mode='c'] ray_vdir,
                            np.ndarray[double,ndim=3,mode='c'] ves_poly,
                            double eps_uz=_SMALL, double eps_a=_VSMALL,
                            double eps_vz=_VSMALL, double eps_b=_VSMALL,
                            double eps_plane=_VSMALL, str ves_type='Tor',
                            str algo_type='simple', int num_threads=16):
    """
    This function computes the distance (and the associated k) between nlos
    Rays (or LOS) and several `IN` structures (polygons extruded around the axis
    (0,0,1), eg. flux surfaces).
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
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
        kmin_vpoly : (npoly, nlos) double array
            Of the form [k_00, k_01, ..., k_0n, k_10, k_11, ..., k_1n, ...]
            where k_ij is the coefficient for the j-th flux surface
            such that the i-th ray (LOS) is closest to the extruded polygon
            at the point P_i = orig[i] + kmin[i] * vdir[i]
        dist_vpoly : (npoly, nlos) double array
            `distance[i * num_poly + j]` is the distance from P_i to the i-th
            extruded poly.
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from cython, use `comp_dist_los_vpoly_vec_core`
    """
    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    cdef np.ndarray[double, ndim=1] kmin = np.empty((nvpoly*nlos,), dtype=float)
    cdef np.ndarray[double, ndim=1] dist = np.empty((nvpoly*nlos,), dtype=float)
    cdef int algo_num = 0
    cdef int ves_num = 1
    _dt.comp_dist_los_vpoly_vec_core(nvpoly, nlos,
                                    <double*>ray_orig.data,
                                    <double*>ray_vdir.data,
                                    ves_poly,
                                    eps_uz, eps_a,
                                    eps_vz, eps_b,
                                    eps_plane,
                                    ves_num,
                                    algo_num,
                                    &kmin[0], &dist[0],
                                    0.05,
                                    num_threads)
    return kmin.reshape(nlos, nvpoly),\
        dist.reshape(nlos, nvpoly)

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
    This function tests if the distance between nlos Rays (or LOS) and
    several `IN` structures (polygons extruded around the axis (0,0,1),
    eg. flux surfaces) is smaller than `epsilon`.
    For more details on the algorithm please see PDF: <name_of_pdf>.pdf #TODO

    Params
    ======
        nvpoly : int
           Number of flux surfaces
        nlos : int
           Number of LOS
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
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
        are_close : (npoly * nlos) bool array
            `are_close[i * num_poly + j]` indicates if distance between i-th LOS
            and j-th poly are closer than epsilon. (True if distance<epsilon)
    ---
    This is the PYTHON function, use only if you need this computation from
    Python, if you need it from cython, use `is_close_los_vpoly_vec_core`
    """
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)
    # ==========================================================================
    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)
    # ==========================================================================

    cdef array are_close = clone(array('i'), nvpoly*nlos, True)
    _dt.is_close_los_vpoly_vec_core(nvpoly, nlos,
                                <double*>ray_orig.data,
                                <double*>ray_vdir.data,
                                ves_poly,
                                eps_uz, eps_a,
                                eps_vz, eps_b,
                                eps_plane,
                                epsilon,
                                are_close,
                                num_threads)
    return np.asarray(are_close, dtype=bool).reshape(nlos, nvpoly)

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
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
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
    Python, if you need it from cython, use `which_los_closer_vpoly_vec_core`
    """
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)

    # ==========================================================================
    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)
    # ==========================================================================

    cdef array ind_close_tab = clone(array('i'), nvpoly, True)
    _dt.which_los_closer_vpoly_vec_core(nvpoly, nlos,
                                    <double*>ray_orig.data,
                                    <double*>ray_vdir.data,
                                    ves_poly,
                                    eps_uz, eps_a,
                                    eps_vz, eps_b,
                                    eps_plane,
                                    ind_close_tab,
                                    num_threads)
    return np.asarray(ind_close_tab)


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
        ray_orig : (3, nlos) double array
           LOS origin points coordinates
        ray_vdir : (3, nlos) double array
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
    Python, if you need it from cython, use `which_vpoly_closer_los_vec_core`
    """
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)
    # ==========================================================================
    if not algo_type.lower() == "simple" or not ves_type.lower() == "tor":
        assert False, "The function is only implemented with the simple"\
            + " algorithm and for toroidal vessels... Sorry!"
    warn("This function supposes that the polys are nested from inner to outer",
         Warning)
    # ==========================================================================

    cdef array ind_close_tab = clone(array('i'), nlos, True)
    _dt.which_vpoly_closer_los_vec_core(nvpoly, nlos,
                                    <double*>ray_orig.data,
                                    <double*>ray_vdir.data,
                                    ves_poly,
                                    eps_uz, eps_a,
                                    eps_vz, eps_b,
                                    eps_plane,
                                    ind_close_tab,
                                    num_threads)
    return np.asarray(ind_close_tab)



def _Ves_Vmesh_Tor_SubFromD_cython_old(double dR, double dZ, double dRPhi,
                                       double[::1] RMinMax, double[::1] ZMinMax,
                                       list DR=None,
                                       list DZ=None,
                                       DPhi=None, VPoly=None,
                                       str Out='(X,Y,Z)', double margin=_VSMALL):
    """
    Return the desired submesh indicated by the limits (DR,DZ,DPhi),
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] R0, R, Z, dRPhir, dPhir, NRPhi, hypot
    cdef double reso_r0, reso_r, reso_z, DPhi0, DPhi1
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj, nPhi0, nPhi1, zz
    cdef int NP, NRPhi_int, radius_ratio
    cdef np.ndarray[double,ndim=2] Pts, indI
    cdef np.ndarray[double,ndim=1] iii, dV, ind


    warn("You are using the old algorithm for meshing a volume."
         + " This algorithm is slower than the new one.", Warning)

    # Get the actual R and Z resolutions and mesh elements
    R0, reso_r0, indR0, NR0 = discretize_line1d(RMinMax, dR, None,
                                             Lim=True, margin=margin)
    R, reso_r, indR, NR = discretize_line1d(RMinMax, dR, DR, Lim=True,
                                          margin=margin)
    Z, reso_z, indZ, NZ = discretize_line1d(ZMinMax, dZ, DZ, Lim=True,
                                          margin=margin)
    Rn = len(R)
    Zn = len(Z)
    # Get the limits if any (and make sure to replace them in the proper
    # quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = -c_pi, c_pi
    else:
        DPhi0 = c_atan2(c_sin(DPhi[0]), c_cos(DPhi[0]))
        DPhi1 = c_atan2(c_sin(DPhi[1]), c_cos(DPhi[1]))
    dRPhir, dPhir = np.empty((Rn,)), np.empty((Rn,))
    Phin = np.empty((Rn,),dtype=int)
    NRPhi = np.empty((Rn,))
    NRPhi0 = np.zeros((Rn,),dtype=int)
    nRPhi0, indR0ii = 0, 0
    NP, NPhimax = 0, 0
    radius_ratio = int(c_ceil(R[Rn-1]/R[0]))
    for ii in range(0,Rn):
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
        NRPhi[ii] = c_ceil(2.*c_pi*R[ii]/dRPhi)
        NRPhi_int = int(NRPhi[ii])
        dPhir[ii] = 2.*c_pi/NRPhi[ii]
        dRPhir[ii] = dPhir[ii]*R[ii]
        # Get index and cumulated indices from background
        for jj in range(indR0ii,NR0):
            if R0[jj]==R[ii]:
                indR0ii = jj
                break
            else:
                nRPhi0 += <long>c_ceil(2.*c_pi*R0[jj]/dRPhi)
                NRPhi0[ii] = nRPhi0*NZ
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        abs0 = c_abs(DPhi0+c_pi)
        if abs0-dPhir[ii]*c_floor(abs0/dPhir[ii]) < margin*dPhir[ii]:
            nPhi0 = int(c_round((DPhi0+c_pi)/dPhir[ii]))
        else:
            nPhi0 = int(c_floor((DPhi0+c_pi)/dPhir[ii]))
        abs1 = c_abs(DPhi1+c_pi)
        if abs1-dPhir[ii]*c_floor(abs1/dPhir[ii]) < margin*dPhir[ii]:
            nPhi1 = int(c_round((DPhi1+c_pi)/dPhir[ii])-1)
        else:
            nPhi1 = int(c_floor((DPhi1+c_pi)/dPhir[ii]))

        if DPhi0<DPhi1:
            #indI.append(list(range(nPhi0,nPhi1+1)))
            Phin[ii] = nPhi1+1-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*radius_ratio+1))
            for jj in range(0,Phin[ii]):
                indI[ii,jj] = <double>( nPhi0+jj )
        else:
            #indI.append(list(range(nPhi0,NRPhi_int)+list(range(0,nPhi1+1))))
            Phin[ii] = nPhi1+1+NRPhi_int-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*radius_ratio+1))
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
                    phi = -c_pi + (0.5+indiijj)*dPhir[ii]
                    Pts[0,NP] = R[ii]*c_cos(phi)
                    Pts[1,NP] = R[ii]*c_sin(phi)
                    Pts[2,NP] = Z[zz]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = reso_r*reso_z*dRPhir[ii]
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
                    Pts[2,NP] = -c_pi + (0.5+indiijj)*dPhir[ii]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = reso_r*reso_z*dRPhir[ii]
                    NP += 1
    if VPoly is not None:
        if Out.lower()=='(x,y,z)':
            hypot = _bgt.compute_hypot(Pts[0,:],Pts[1,:])
            indin = Path(VPoly.T).contains_points(np.array([hypot,Pts[2,:]]).T,
                                                  transform=None, radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(hypot)
        else:
            indin = Path(VPoly.T).contains_points(Pts[:-1,:].T, transform=None,
                                                  radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(Pts[0,:])
        # TODO : Warning : do we need the following lines ????
        # if not np.all(Ru==R):
        #     dRPhir = np.array([dRPhir[ii] for ii in range(0,len(R)) \
        #                        if R[ii] in Ru])
    return Pts, dV, ind.astype(int), reso_r, reso_z, np.asarray(dRPhir)


def _Ves_Vmesh_Tor_SubFromInd_cython_old(double dR, double dZ, double dRPhi,
                                     double[::1] RMinMax, double[::1] ZMinMax,
                                     long[::1] ind, str Out='(X,Y,Z)',
                                     double margin=_VSMALL):
    """ Return the desired submesh indicated by the (numerical) indices,
    for the desired resolution (dR,dZ,dRphi)
    """
    cdef double[::1] R, Z, dRPhirRef, dPhir, Ru, dRPhir
    cdef double reso_r, reso_z, phi
    cdef long[::1] indR, indZ, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), radius_ratio
    cdef int ii=0, jj=0, iiR, iiZ, iiphi
    cdef double[:,::1] Phi
    cdef np.ndarray[double,ndim=2] Pts=np.empty((3,NP))
    cdef np.ndarray[double,ndim=1] dV=np.empty((NP,))

    warn("You are using the old algorithm for meshing a volume."
         + " This algorithm is slower than the new one.", Warning)

    # Get the actual R and Z resolutions and mesh elements
    R, reso_r, indR, NR = discretize_line1d(RMinMax, dR, None, Lim=True,
                                                margin=margin)
    Z, reso_z, indZ, NZ = discretize_line1d(ZMinMax, dZ, None, Lim=True,
                                                margin=margin)
    Rn, Zn = len(R), len(Z)

    # Number of Phi per R
    dRPhirRef, dPhir = np.empty((NR,)), np.empty((NR,))
    Ru, dRPhir = np.zeros((NR,)), np.nan*np.ones((NR,))
    NRPhi, NRPhi0 = np.empty((NR,),dtype=int), np.empty((NR+1,),dtype=int)
    radius_ratio = int(c_ceil(R[NR-1]/R[0]))
    for ii in range(0,NR):
        NRPhi[ii] = <long>(c_ceil(2.*c_pi*R[ii]/dRPhi))
        dRPhirRef[ii] = 2.*c_pi*R[ii]/<double>(NRPhi[ii])
        dPhir[ii] = 2.*c_pi/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((NR,NRPhi[ii]*radius_ratio+1))
        else:
            NRPhi0[ii] = NRPhi0[ii-1] + NRPhi[ii-1]*NZ
        for jj in range(0,NRPhi[ii]):
            Phi[ii,jj] = -c_pi + (0.5+<double>jj)*dPhir[ii]

    if Out.lower()=='(x,y,z)':
        for ii in range(0,NP):
            for jj in range(0,NR+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiR = jj-1
            iiZ = (ind[ii] - NRPhi0[iiR])//NRPhi[iiR]
            iiphi = ind[ii] - NRPhi0[iiR] - iiZ*NRPhi[iiR]
            phi = Phi[iiR,iiphi]
            Pts[0,ii] = R[iiR]*c_cos(phi)
            Pts[1,ii] = R[iiR]*c_sin(phi)
            Pts[2,ii] = Z[iiZ]
            dV[ii] = reso_r*reso_z*dRPhirRef[iiR]
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
            dV[ii] = reso_r*reso_z*dRPhirRef[iiR]
            if Ru[iiR]==0.:
                dRPhir[iiR] = dRPhirRef[iiR]
                Ru[iiR] = 1.
    return Pts, dV, reso_r, reso_z, np.asarray(dRPhir)[~np.isnan(dRPhir)]


# ==============================================================================
#
#                       Solid Angle Computation
#                        subtended by a sphere
#
# ==============================================================================
def compute_solid_angle_map(double[:,::1] part_coords, double[::1] part_r,
                            double rstep, double zstep, double phistep,
                            double[::1] RMinMax, double[::1] ZMinMax,
                            bint approx=True,
                            list DR=None, list DZ=None, DPhi=None,
                            double[:,::1] limit_vpoly=None,
                            bint block=False,
                            double[:, ::1] ves_poly=None,
                            double[:, ::1] ves_norm=None,
                            double[::1] ves_lims=None,
                            long[::1] lstruct_nlim=None,
                            double[::1] lstruct_polyx=None,
                            double[::1] lstruct_polyy=None,
                            list lstruct_lims=None,
                            double[::1] lstruct_normx=None,
                            double[::1] lstruct_normy=None,
                            long[::1] lnvert=None,
                            int nstruct_tot=0,
                            int nstruct_lim=0,
                            double rmin=-1, bint forbid=True,
                            double eps_uz=_SMALL, double eps_a=_VSMALL,
                            double eps_vz=_VSMALL, double eps_b=_VSMALL,
                            double eps_plane=_VSMALL, str ves_type='Tor',
                            double margin=_VSMALL, int num_threads=48,
                            bint test=True):
    """
    Computes the 2D map of the integrated solid angles subtended by each of
    the sz_p particles P of radius part_r at the position part_coords
    in the sampled volume.
    If approx, a 8th degree approximation will be used for the computation
    of the solid angle

    Parameters
    ----------
    part_coords: (3, sz_p) double memory-view
	    cartesian coordinates of P particles
    part_r: (sz_p) double memory-view
        the radii of the P particles
    rstep: double
        refinement along radius `r`
    zstep: double
        refinement along height `z`
    phistep: double
        refinement along toroidal direction `phi`
    approx: bool
        do you want to use approximation (8th order) or exact formula ?
        default: True
    RMinMax: double memory-view
        limits min and max in `r`
    ZMinMax: double memory-view
        limits min and max in `z`
    DR: double memory-view, optional
        actual sub-volume limits to get in `r`
    DZ: double memory-view, optional
        actual sub-volume limits to get in `z`
    DPhi: double memory-view, optional
        actual sub-volume limits to get in `phi`
    limit_vpoly: (3, npts) double memory-view, optional
        if we only want to discretize the volume inside a certain flux surface.
        Defines the `(R,Z)` coords of the poloidal cut of the limiting flux
        surface.
    block: bool, optional
        check if particles are viewable from viewing points or if there is a
        structural element blocking visibility (False)
    ves_poly : (2, num_vertex) double array
       Coordinates of the vertices of the Polygon defining the 2D poloidal
       cut of the Vessel
    ves_norm : (2, num_vertex-1) double array
       Normal vectors going "inwards" of the edges of the Polygon defined
       by ves_poly
    nstruct_tot : int
       Total number of structures (counting each limited structure as one)
    ves_lims : array
       Contains the limits min and max of vessel
    lstruct_polyx : array
       List of x coordinates of the vertices of all structures on poloidal plane
       If no structures : None
    lstruct_polyy : array
       List of y coordinates of the vertices of all structures on poloidal plane
       If no structures : None
    lstruct_lims : array
       List of limits of all structures
       If no structures : None
    lstruct_nlim : array of ints
       List of number of limits for all structures
       If no structures : None
    lstruct_normx : double memory-view, optional
       List of x-coordinates of "inwards" normal vectors of the polygon of all
       the structures
       If no structures : None
    lstruct_normy : double memory-view, optional
       List of y-coordinates of "inwards" normal vectors of the polygon of all
       the structures
       If no structures : None
    rmin : double, optional
       Minimal radius of vessel to take into consideration
    forbid : bool, optional
       Should we forbid values behind visible radius ? (see rmin)
    eps_<val> : double, optional
       Small value, acceptance of error
    margin: double, optional
        tolerance error. Defaults to |_VSMALL|
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    test : bool, optional
       Should we run tests? Default True

    Returns
    -------
        pts:    (2, npts) array of (R, Z) coordinates of viewing points in
                vignette where solid angle is integrated
        sa_map: (npts, sz_p) array approx solid angle integrated along phi
                integral (sa * dphi * r)
        ind:    (npts) indices to reconstruct (R,Z) map from sa_map
        rdrdz:  (npts) volume unit: dr*dz
    """
    cdef int jj
    cdef int sz_p
    cdef int sz_r
    cdef int sz_z
    cdef int npts_pol
    cdef int r_ratio
    cdef int ind_loc_r0
    cdef int npts_disc = 0
    cdef int[1] max_sz_phi
    cdef double min_phi, max_phi
    cdef double min_phi_pi
    cdef double max_phi_pi
    cdef double abs0, abs1
    cdef double reso_r_z
    cdef double twopi_over_dphi
    cdef long[1] ncells_r0, ncells_r, ncells_z
    cdef long[::1] ind_mv
    cdef long[::1] first_ind_mv
    cdef double[2] limits_dl
    cdef double[1] reso_r0, reso_r, reso_z
    cdef double[::1] reso_rdrdz_mv
    cdef double[::1] lstruct_lims_np
    cdef double[:, ::1] poly_mv
    cdef double[:, ::1] pts_mv
    cdef long[:, ::1] indi_mv
    cdef long[:, ::1] ind_rz2pol
    cdef long[:, ::1] is_in_vignette
    cdef long*  ncells_rphi  = NULL
    cdef long*  lindex   = NULL
    cdef long*  lindex_z = NULL
    cdef long*  sz_phi = NULL
    cdef double* disc_r0 = NULL
    cdef double* disc_r  = NULL
    cdef double* disc_z  = NULL
    cdef double* step_rphi = NULL
    cdef np.ndarray[long, ndim=2] indI
    cdef np.ndarray[long, ndim=1] ind
    cdef np.ndarray[double, ndim=1] reso_rdrdz
    cdef np.ndarray[double, ndim=2] pts
    cdef np.ndarray[double, ndim=2] sa_map
    #
    # == Testing inputs ========================================================
    if test:
        if block:
            msg = "ves_poly and ves_norm are not optional arguments"
            assert ves_poly is not None and ves_norm is not None, msg
            bool1 = (ves_poly.shape[0]==2 and ves_norm.shape[0]==2
                     and ves_norm.shape[1]==ves_poly.shape[1]-1)
            msg = "Args ves_poly & ves_norm must be of the same shape (2, NS)!"
            assert bool1, msg
            bool1 = (lstruct_lims is None
                     or len(lstruct_normy) == len(lstruct_normx))
            bool2 = (lstruct_normx is None
                     or len(lstruct_polyx) == len(lstruct_polyy))
            msg = "Args lstruct_polyx, lstruct_polyy, lstruct_lims,"\
                + " lstruct_normx, lstruct_normy, must be None or"\
                + " lists of same len()!"
            assert bool1 and bool2, msg
        msg = "[eps_uz,eps_vz,eps_a,eps_b] must be floats < 1.e-4!"
        assert all([ee < 1.e-4 for ee in [eps_uz, eps_a,
                                          eps_vz, eps_b,
                                          eps_plane]]), msg
        msg = "ves_type must be a str in ['Tor','Lin']!"
        assert ves_type.lower() in ['tor', 'lin'], msg
    # ...
    # .. Getting size of arrays ................................................
    sz_p = part_coords.shape[1]
    # # .. Check if points are visible ...........................................
    # Get the actual R and Z resolutions and mesh elements
    # .. First we discretize R without limits ..................................
    _st.cythonize_subdomain_dl(None, limits_dl) # no limits
    _ = _st.discretize_line1d_core(&RMinMax[0], rstep, limits_dl,
                                   True, 0, # discretize in absolute mode
                                   margin, &disc_r0, reso_r0, &lindex,
                                   ncells_r0)
    free(lindex) # getting rid of things we dont need
    lindex = NULL
    # .. Now the actual R limited  .............................................
    _st.cythonize_subdomain_dl(DR, limits_dl)
    sz_r = _st.discretize_line1d_core(&RMinMax[0], rstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_r, reso_r, &lindex,
                                      ncells_r)
    free(lindex) # getting rid of things we dont need
    # .. Now Z .................................................................
    _st.cythonize_subdomain_dl(DZ, limits_dl)
    sz_z = _st.discretize_line1d_core(&ZMinMax[0], zstep, limits_dl,
                                      True, 0, # discretize in absolute mode
                                      margin, &disc_z, reso_z, &lindex_z,
                                      ncells_z)
    # .. Preparing for phi: get the limits if any and make sure to replace them
    # .. in the proper quadrants ...............................................
    if DPhi is None:
        min_phi = -c_pi
        max_phi = c_pi
    else:
        min_phi = DPhi[0] # to avoid conversions
        min_phi = c_atan2(c_sin(min_phi), c_cos(min_phi))
        max_phi = DPhi[1] # to avoid conversions
        max_phi = c_atan2(c_sin(max_phi), c_cos(max_phi))
    # .. Initialization ........................................................
    sz_phi = <long*>malloc(sz_r*sizeof(long))
    ncells_rphi  = <long*>malloc(sz_r*sizeof(long))
    step_rphi    = <double*>malloc(sz_r*sizeof(double))
    r_ratio = <int>(c_ceil(disc_r[sz_r - 1] / disc_r[0]))
    twopi_over_dphi = _TWOPI / phistep
    ind_loc_r0 = 0
    min_phi_pi = min_phi + c_pi
    max_phi_pi = max_phi + c_pi
    abs0 = c_abs(min_phi_pi)
    abs1 = c_abs(max_phi_pi)
    # ... doing 0 loop before ..................................................
    if min_phi < max_phi:
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
        ncells_rphi[0] = <int>c_ceil(twopi_over_dphi * disc_r[0])
        loc_nc_rphi = ncells_rphi[0]
        step_rphi[0] = _TWOPI / ncells_rphi[0]
        inv_drphi = 1. / step_rphi[0]
        # Get index and cumulated indices from background
        for jj in range(ind_loc_r0, ncells_r0[0]):
            if disc_r0[jj]==disc_r[0]:
                ind_loc_r0 = jj
                break
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        if abs0 - step_rphi[0]*c_floor(abs0 * inv_drphi) < margin*step_rphi[0]:
            nphi0 = int(c_round((min_phi + c_pi) * inv_drphi))
        else:
            nphi0 = int(c_floor((min_phi +c_pi) * inv_drphi))
        if abs1-step_rphi[0]*c_floor(abs1 * inv_drphi) < margin*step_rphi[0]:
            nphi1 = int(c_round((max_phi+c_pi) * inv_drphi)-1)
        else:
            nphi1 = int(c_floor((max_phi+c_pi) * inv_drphi))
        sz_phi[0] = nphi1 + 1 - nphi0
        max_sz_phi[0] = sz_phi[0]
        ind_i = -np.ones((sz_r, sz_phi[0] * r_ratio + 1), dtype=int)
        indi_mv = ind_i
        for jj in range(sz_phi[0]):
            indi_mv[0, jj] = nphi0 + jj
        npts_disc += sz_z * sz_phi[0]
    else:
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R!)
        ncells_rphi[0] = <int>c_ceil(twopi_over_dphi * disc_r[0])
        loc_nc_rphi = ncells_rphi[0]
        step_rphi[0] = _TWOPI / ncells_rphi[0]
        inv_drphi = 1. / step_rphi[0]
        # Get index and cumulated indices from background
        for jj in range(ind_loc_r0, ncells_r0[0]):
            if disc_r0[jj]==disc_r[0]:
                ind_loc_r0 = jj
                break
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to
        # be created within those limits
        if abs0 - step_rphi[0]*c_floor(abs0 * inv_drphi) < margin*step_rphi[0]:
            nphi0 = int(c_round((min_phi + c_pi) * inv_drphi))
        else:
            nphi0 = int(c_floor((min_phi + c_pi) * inv_drphi))
        if abs1-step_rphi[0]*c_floor(abs1 * inv_drphi) < margin*step_rphi[0]:
            nphi1 = int(c_round((max_phi+c_pi) * inv_drphi)-1)
        else:
            nphi1 = int(c_floor((max_phi+c_pi) * inv_drphi))
        sz_phi[0] = nphi1+1+loc_nc_rphi-nphi0
        max_sz_phi[0] = sz_phi[0]
        ind_i = -np.ones((sz_r, sz_phi[0] * r_ratio + 1), dtype=int)
        indi_mv = ind_i
        for jj in range(loc_nc_rphi - nphi0):
            indi_mv[0, jj] = nphi0 + jj
        for jj in range(loc_nc_rphi - nphi0, sz_phi[0]):
            indi_mv[0, jj] = jj - (loc_nc_rphi - nphi0)
        npts_disc += sz_z * sz_phi[0]
    # ... doing the others .....................................................
    npts_disc += _st.sa_disc_phi(sz_r, sz_z, ncells_rphi, phistep,
                                 disc_r, disc_r0, step_rphi,
                                 ind_loc_r0,
                                 ncells_r0[0], ncells_z[0], &max_sz_phi[0],
                                 min_phi, max_phi, sz_phi, indi_mv,
                                 margin, num_threads)
    # ... vignetting ...........................................................
    is_in_vignette = np.ones((sz_r, sz_z), dtype=int) # by default yes
    if limit_vpoly is not None:
        npts_vpoly = limit_vpoly.shape[1] - 1
        # we make sure it is closed
        if not(abs(limit_vpoly[0, 0] - limit_vpoly[0, npts_vpoly]) < _VSMALL
                and abs(limit_vpoly[1, 0]
                        - limit_vpoly[1, npts_vpoly]) < _VSMALL):
            poly_mv = np.concatenate((limit_vpoly, limit_vpoly[:,0:1]), axis=1)
        else:
            poly_mv = limit_vpoly
        _ = _vt.are_in_vignette(sz_r, sz_z,
                                poly_mv, npts_vpoly,
                                disc_r, disc_z,
                                is_in_vignette)
    # .. preparing for actual discretization ...................................
    ind_rz2pol = np.empty((sz_r, sz_z), dtype=int)
    npts_pol = _st.sa_get_index_arrays(ind_rz2pol,
                                     is_in_vignette,
                                     sz_r, sz_z)
    # initializing arrays
    reso_rdrdz = np.empty((npts_pol, ))
    sa_map = np.zeros((npts_pol, sz_p))
    pts = np.empty((2, npts_pol))
    ind = -np.ones((npts_pol, ), dtype=int)
    pts_mv = pts
    ind_mv = ind
    reso_rdrdz_mv = reso_rdrdz
    reso_r_z = reso_r[0]*reso_z[0]
    ind_i = np.sort(ind_i, axis=1)
    indi_mv = ind_i
    first_ind_mv = np.argmax(ind_i > -1, axis=1).astype(int)
    # initializing utilitary arrays
    num_threads = _ompt.get_effective_num_threads(num_threads)
    lstruct_lims_np = flatten_lstruct_lims(lstruct_lims)
    # ..............
    _st.sa_assemble_arrays(block,
                           approx,
                           part_coords,
                           part_r,
                           is_in_vignette,
                           sa_map,
                           ves_poly,
                           ves_norm,
                           ves_lims,
                           lstruct_nlim,
                           lstruct_polyx,
                           lstruct_polyy,
                           lstruct_lims_np,
                           lstruct_normx,
                           lstruct_normy,
                           lnvert,
                           nstruct_tot,
                           nstruct_lim,
                           rmin,
                           eps_uz, eps_a,
                           eps_vz, eps_b,
                           eps_plane,
                           forbid,
                           first_ind_mv,
                           indi_mv,
                           sz_p, sz_r, sz_z,
                           ncells_rphi,
                           reso_r_z,
                           disc_r,
                           step_rphi,
                           disc_z,
                           ind_rz2pol,
                           sz_phi,
                           reso_rdrdz_mv,
                           pts_mv,
                           ind_mv,
                           num_threads)
    # ... freeing up memory ....................................................
    free(lindex_z)
    free(disc_r)
    free(disc_z)
    free(disc_r0)
    free(sz_phi)
    free(step_rphi)
    free(ncells_rphi)

    return pts, sa_map, ind, reso_r_z
