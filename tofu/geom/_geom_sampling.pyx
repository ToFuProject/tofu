# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#

from cpython.array cimport array, clone
from cython.parallel import prange
from cython.parallel cimport parallel
from libc.stdlib cimport malloc, free, realloc
cdef double _VSMALL = 1.e-9
cdef double _SMALL = 1.e-6


# ==============================================================================
#
#                                   LINEAR MESHING
#                               i.e. Discretizing lines
#
# ==============================================================================
def discretize_segment(double[::1] LMinMax, double dstep,
                       double[::1] DL=None, bint Lim=True,
                       str mode='abs', double margin=_VSMALL):
    """
    Discretize a segment LMin-LMax. If `mode` is "abs" (absolute), then the
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
    cdef array ldiscret
    cdef array lindex
    cdef long N, ntot
    cdef double resolution
    cdef double[2] desired_limits

    discretize_segment(LMinMax, dstep,
                       ldiscret, resolution, lindex, N, ntot,
                       DL, Lim, mode, margin)
    return np.asarray(ldiscret), dLr, np.asarray(lindex), <long>N



cdef void discretize_segment_core(double[2] LMinMax, double dstep,
                                  double* ldiscret, double resolution,
                                  int* lindex, int num_cells, int tot_ncells,
                                  double[2] DL=NULL, bint Lim=True,
                                  str mode='abs', double margin=_VSMALL):
    """
    Discretize a segment LMin-LMax. If `mode` is "abs" (absolute), then the
    segment will be discretized in cells each of size `dstep`. Else, if `mode`
    is "rel" (relative), the meshing step is relative to the segments norm (ie.
    the actual discretization step will be (LMax - LMin)/dstep).
    It is possible to only one to discretize the segment on a sub-domain. If so,
    the sub-domain limits are given in DL.
    CYTHON core
    Parameters
    ==========
    LMinMax : (2)-double array
        Gives the limits LMin and LMax of the segment. LMinMax = [LMin, LMax]
    dstep: double
        Step of discretization, can be absolute (default) or relative
    DL : (optional) (2)-double array
        Sub domain of discretization. If not None and if Lim, LMinMax = DL
        (can be only on one limit)
    Lim : (optional) bool
        Indicated if the subdomain should be taken into account
    mode : (optional) string
        If `mode` is "abs" (absolute), then the
        segment will be discretized in cells each of size `dstep`. Else,
        if "rel" (relative), the meshing step is relative to the segments norm
        (the actual discretization step will be (LMax - LMin)/dstep).
    margin : (optional) double
        Margin value for cell length
    """
    cdef int nL0, nL1, Nind, ii, jj
    cdef double abs0, abs1
    cdef double inv_reso, new_margin
    cdef double[2] desired_limits

    # .. Computing "real" discretization step, depending on `mode`..............
    if mode.lower()=='abs':
        num_cells = Cceil((LMinMax[1] - LMinMax[0])/dstep)
    else:
        num_cells = Cceil(1./dstep)
    resolution = (LMinMax[1] - LMinMax[0])/num_cells
    # .. Computing desired limits ..............................................
    if DL is None:
        desired_limits[0] = LMinMax[0]
        desired_limits[1] = LMinMax[1]
    else:
        if DL[0] is None:
            DL[0] = LMinMax[0]
        if DL[1] is None:
            DL[1] = LMinMax[1]
        if Lim and DL[0]<=LMinMax[0]:
            DL[0] = LMinMax[0]
        if Lim and DL[1]>=LMinMax[1]:
            DL[1] = LMinMax[1]
        desired_limits[0] = DL[0]
        desired_limits[1] = DL[1]
    # .. Get the extreme indices of the mesh elements that really need to be
    # created within those limits...............................................
    inv_reso = 1./resolution
    new_margin = margin*resolution
    abs0 = Cabs(desired_limits[0] - LMinMax[0])
    if abs0 - resolution * Cfloor(abs0 * inv_reso) < new_margin:
        nL0 = int(Cround((desired_limits[0] - LMinMax[0]) * inv_reso))
    else:
        nL0 = int(Cfloor((desired_limits[0] - LMinMax[0]) * inv_reso))
    abs1 = Cabs(desired_limits[1] - LMinMax[0])
    if abs1 - resolution * Cfloor(abs1 * inv_reso) < new_margin:
        nL1 = int(Cround((desired_limits[1] - LMinMax[0]) * inv_reso) - 1)
    else:
        nL1 = int(Cfloor((desired_limits[1] - LMinMax[0]) * inv_reso))
    # Get the total number of indices
    Nind = nL1 + 1 - nL0
    # .. Computing coordinates and indices .....................................
    ldiscret = clone(array('d'), Nind, True)
    lindex = clone(array('i'), Nind, True)
    for ii in range(Nind):
        jj = nL0 + ii
        lindex[ii] = jj
        ldiscret[ii] = LMinMax[0] + (0.5 + jj)*dLr
    return


# ==============================================================================
#
#                                   SAMPLING
#
# ==============================================================================
 def _Ves_Smesh_Cross(double[:,::1] VPoly, double dL, str dLMode='abs',
                      double[::1] D1=None, double[::1] D2=None,
                      double margin=_VSMALL, double DIn=0.,
                      double[:,::1] VIn=None):
     # local variables
     cdef int npts_poly = VPoly.shape[1]
     # return variables
     cdef array pts_cross  = clone(array('d'), npts_poly*2, True)
     

     return np.transpose(np.asarray(pts_cross).reshape(npts, 2)),\
       dLr, ind, N, Rref, np.array(VPolybis).T
