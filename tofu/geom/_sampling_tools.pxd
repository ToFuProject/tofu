# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for sampling and discretizating
################################################################################
cimport cython
from libc.math cimport ceil as Cceil, fabs as Cabs
from libc.math cimport floor as Cfloor, round as Cround
from libc.math cimport isnan as Cisnan
from cpython.array cimport array, clone, extend
from libc.stdlib cimport malloc, free, realloc

# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================

cdef inline long discretize_segment_core(double[::1] LMinMax, double dstep,
                                         double[2] DL, bint Lim,
                                         str mode, double margin,
                                         double** ldiscret_arr,
                                         double[1] resolution,
                                         long** lindex_arr, long[1] N):
    cdef int ii
    cdef int[1] nL0
    cdef long[1] Nind

    print("------------------- before 1")
    first_discretize_segment_core(LMinMax, dstep,
                                  resolution, N, Nind, nL0,
                                  DL, Lim, mode, margin)
    print("------------------- before 2")

    if (Nind[0] <= 0):
        assert False, "Size should be > 0"+str(Nind[0])
    else:
        print("Nind[0] :::::::::::::::::::::::::::", Nind[0])
    print("------------------- before 3")
    if (ldiscret_arr[0] == NULL):
        ldiscret_arr[0] = <double *>malloc(Nind[0] * sizeof(double))
    else:
        ldiscret_arr[0] = <double *>realloc(ldiscret_arr[0],
                                            Nind[0] * sizeof(double))
        # free(ldiscret_arr[0])
        # ldiscret_arr[0] = <double *>malloc(Nind[0] * sizeof(double))
    print("------------------- before 4")
    if (lindex_arr[0] == NULL):
        lindex_arr[0] = <long *>malloc(Nind[0] * sizeof(int))
    else:
        lindex_arr[0] = <long *>realloc(lindex_arr[0], Nind[0] * sizeof(int))
        # free(lindex_arr[0])
        # lindex_arr[0] = <long *>malloc(Nind[0] * sizeof(int))
    print("------------------- before 5")
    second_discretize_segment_core(LMinMax, ldiscret_arr[0], lindex_arr[0],
                                   nL0[0], resolution[0], Nind[0])
    print("----------outing")
    return Nind[0]


cdef inline void first_discretize_segment_core(double[::1] LMinMax,
                                               double dstep,
                                               double[1] resolution,
                                               long[1] num_cells,
                                               long[1] Nind,
                                               int[1] nL0,
                                               double[2] DL,
                                               bint Lim,
                                               str mode,
                                               double margin):
    """
    Computes the resolution, the desired limits, and the number of cells when
    discretising the segmen LMinMax with the given parameters. It doesn't do the
    actual discretization.
    For that part, please refer to: second_discretize_segment_core
    """
    cdef int nL1, ii, jj
    cdef double abs0, abs1
    cdef double inv_reso, new_margin
    cdef double[2] desired_limits

    # .. Computing "real" discretization step, depending on `mode`..............
    if mode.lower()=='abs':
        num_cells[0] = <int>Cceil((LMinMax[1] - LMinMax[0])/dstep)
    else:
        num_cells[0] = <int>Cceil(1./dstep)
    resolution[0] = (LMinMax[1] - LMinMax[0])/num_cells[0]
    # .. Computing desired limits ..............................................
    if Cisnan(DL[0]) and Cisnan(DL[1]):
        desired_limits[0] = LMinMax[0]
        desired_limits[1] = LMinMax[1]
    else:
        if Cisnan(DL[0]):
            DL[0] = LMinMax[0]
        if Cisnan(DL[1]):
            DL[1] = LMinMax[1]
        if Lim and DL[0]<=LMinMax[0]:
            DL[0] = LMinMax[0]
        if Lim and DL[1]>=LMinMax[1]:
            DL[1] = LMinMax[1]
        desired_limits[0] = DL[0]
        desired_limits[1] = DL[1]
    # .. Get the extreme indices of the mesh elements that really need to be
    # created within those limits...............................................
    inv_reso = 1./resolution[0]
    new_margin = margin*resolution[0]
    abs0 = Cabs(desired_limits[0] - LMinMax[0])
    if abs0 - resolution[0] * Cfloor(abs0 * inv_reso) < new_margin:
        nL0[0] = int(Cround((desired_limits[0] - LMinMax[0]) * inv_reso))
    else:
        nL0[0] = int(Cfloor((desired_limits[0] - LMinMax[0]) * inv_reso))
    abs1 = Cabs(desired_limits[1] - LMinMax[0])
    if abs1 - resolution[0] * Cfloor(abs1 * inv_reso) < new_margin:
        nL1 = int(Cround((desired_limits[1] - LMinMax[0]) * inv_reso) - 1)
    else:
        nL1 = int(Cfloor((desired_limits[1] - LMinMax[0]) * inv_reso))
    # Get the total number of indices
    Nind[0] = nL1 + 1 - nL0[0]
    return

cdef inline void second_discretize_segment_core(double[::1] LMinMax,
                                                double* ldiscret,
                                                long* lindex,
                                                int nL0,
                                                double resolution,
                                                long Nind):
    """
    Does the actual discretization of the segment LMinMax.
    Computes the coordinates of the cells on the discretized segment and the
    associated list of indices.
    This function need some parameters computed with the first algorithm:
    first_discretize_segment_core
    """
    cdef int ii, jj
    # .. Computing coordinates and indices .....................................
    for ii in range(Nind):
        jj = nL0 + ii
        lindex[ii] = jj
        ldiscret[ii] = LMinMax[0] + (0.5 + jj) * resolution
    return
