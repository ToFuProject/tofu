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
from libc.math cimport sqrt as Csqrt
from libc.math cimport isnan as Cisnan
from libc.math cimport NAN as Cnan
from libc.stdlib cimport malloc, free, realloc
from cpython.array cimport array, clone
from _basic_geom_tools cimport _VSMALL
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

    first_discretize_segment_core(LMinMax, dstep,
                                  resolution, N, Nind, nL0,
                                  DL, Lim, mode, margin)
    if (ldiscret_arr[0] == NULL):
        ldiscret_arr[0] = <double *>malloc(Nind[0] * sizeof(double))
    else:
        ldiscret_arr[0] = <double *>realloc(ldiscret_arr[0],
                                            Nind[0] * sizeof(double))
    if (lindex_arr[0] == NULL):
        lindex_arr[0] = <long *>malloc(Nind[0] * sizeof(long))
    else:
        lindex_arr[0] = <long *>realloc(lindex_arr[0], Nind[0] * sizeof(long))
    second_discretize_segment_core(LMinMax, ldiscret_arr[0], lindex_arr[0],
                                   nL0[0], resolution[0], Nind[0])
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

# ==============================================================================
# =  Vessel's poloidal cut discretization
# ==============================================================================

cdef inline void discretize_ves_poly(double[:, ::1] VPoly, double dstep,
                                     str mode, double margin, double DIn,
                                     double[:, ::1] VIn,
                                     double** XCross, double** YCross,
                                     double** resol,
                                     long** ind, long** numcells, double** Rref,
                                     double** XPolybis, double** YPolybis,
                                     int[1] tot_sz_vb, int[1] tot_sz_ot,
                                     int NP):
    cdef Py_ssize_t sz_vpoly_bis = 0
    cdef Py_ssize_t sz_others = 0
    cdef Py_ssize_t last_sz_vbis = 0
    cdef Py_ssize_t last_sz_othr = 0
    cdef int ii, jj
    cdef double v0, v1
    cdef double inv_norm
    cdef double[1] loc_resolu
    cdef double[2] LMinMax
    cdef double[2] dl_array
    cdef double* ldiscret = NULL
    cdef long* lindex = NULL

    #.. initialization..........................................................
    LMinMax[0] = 0.
    dl_array[0] = Cnan
    dl_array[1] = Cnan
    numcells[0] = <long*>malloc((NP-1)*sizeof(long))
    #.. Filling arrays..........................................................
    if abs(DIn) < _VSMALL:
        for ii in range(NP-1):
            v0 = VPoly[0,ii+1]-VPoly[0,ii]
            v1 = VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0*v0 + v1*v1)
            inv_norm = 1./LMinMax[1]
            discretize_segment_core(LMinMax, dstep, dl_array, True,
                                    mode, margin, &ldiscret, loc_resolu,
                                    &lindex, &numcells[0][ii])
            # .. prepaaring Poly bis array......................................
            last_sz_vbis = sz_vpoly_bis
            sz_vpoly_bis += 1 + numcells[0][ii]
            XPolybis[0] = <double*>realloc(XPolybis[0], sz_vpoly_bis*sizeof(double))
            YPolybis[0] = <double*>realloc(YPolybis[0], sz_vpoly_bis*sizeof(double))
            XPolybis[0][sz_vpoly_bis - (1 + numcells[0][ii])] = VPoly[0, ii]
            YPolybis[0][sz_vpoly_bis - (1 + numcells[0][ii])] = VPoly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += numcells[0][ii]
            resol[0]  = <double*>realloc(resol[0],  sizeof(double)*sz_others)
            Rref[0]   = <double*>realloc(Rref[0],   sizeof(double)*sz_others)
            XCross[0] = <double*>realloc(XCross[0], sizeof(double)*sz_others)
            YCross[0] = <double*>realloc(YCross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            for jj in range(numcells[0][ii]):
                ind[0][last_sz_othr + jj] = last_sz_othr + jj
                resol[0][last_sz_othr + jj] = loc_resolu[0]
                Rref[0][last_sz_othr + jj] = VPoly[0,ii] + ldiscret[jj]*v0
                XCross[0][last_sz_othr + jj] = VPoly[0,ii] + ldiscret[jj]*v0
                YCross[0][last_sz_othr + jj] = VPoly[1,ii] + ldiscret[jj]*v1
                XPolybis[0][last_sz_vbis + jj] = VPoly[0,ii] + jj*loc_resolu[0]*v0
                YPolybis[0][last_sz_vbis + jj] = VPoly[1,ii] + jj*loc_resolu[0]*v1
        # We close the polygon of VPolybis
        sz_vpoly_bis += 1
        XPolybis[0] = <double*>realloc(XPolybis[0], sz_vpoly_bis*sizeof(double))
        YPolybis[0] = <double*>realloc(YPolybis[0], sz_vpoly_bis*sizeof(double))
        XPolybis[0][sz_vpoly_bis - 1] = VPoly[0, 0]
        YPolybis[0][sz_vpoly_bis - 1] = VPoly[1, 0]
    else:
        for ii in range(NP-1):
            v0 = VPoly[0,ii+1]-VPoly[0,ii]
            v1 = VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0*v0 + v1*v1)
            inv_norm = 1./LMinMax[1]
            discretize_segment_core(LMinMax, dstep, dl_array, True,
                                    mode, margin, &ldiscret, loc_resolu,
                                    &lindex, &numcells[0][ii])
            # .. prepaaring Poly bis array......................................
            last_sz_vbis = sz_vpoly_bis
            sz_vpoly_bis += 1 + numcells[0][ii]
            XPolybis[0] = <double*>realloc(XPolybis[0], sz_vpoly_bis*sizeof(double))
            YPolybis[0] = <double*>realloc(YPolybis[0], sz_vpoly_bis*sizeof(double))
            XPolybis[0][sz_vpoly_bis - (1 + numcells[0][ii])] = VPoly[0, ii]
            YPolybis[0][sz_vpoly_bis - (1 + numcells[0][ii])] = VPoly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += numcells[0][ii]
            resol[0]  = <double*>realloc(resol[0],  sizeof(double)*sz_others)
            Rref[0]   = <double*>realloc(Rref[0],   sizeof(double)*sz_others)
            XCross[0] = <double*>realloc(XCross[0], sizeof(double)*sz_others)
            YCross[0] = <double*>realloc(YCross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            for jj in range(numcells[0][ii]):
                ind[0][last_sz_othr] = last_sz_othr
                resol[0][last_sz_othr] = loc_resolu[0]
                Rref[0][last_sz_othr] = VPoly[0,ii] + ldiscret[jj]*v0
                XCross[0][last_sz_othr] = VPoly[0,ii] + ldiscret[jj]*v0 + DIn*VIn[0,ii]
                YCross[0][last_sz_othr] = VPoly[1,ii] + ldiscret[jj]*v1 + DIn*VIn[1,ii]
                XPolybis[0][last_sz_vbis + jj] = VPoly[0,ii] + jj*loc_resolu[0]*v0
                YPolybis[0][last_sz_vbis + jj] = VPoly[1,ii] + jj*loc_resolu[0]*v1
                last_sz_othr += 1
        # We close the polygon of VPolybis
        sz_vpoly_bis += 1
        XPolybis[0] = <double*>realloc(XPolybis[0], sz_vpoly_bis*sizeof(double))
        YPolybis[0] = <double*>realloc(YPolybis[0], sz_vpoly_bis*sizeof(double))
        XPolybis[0][sz_vpoly_bis - 1] = VPoly[0, 0]
        YPolybis[0][sz_vpoly_bis - 1] = VPoly[1, 0]
    tot_sz_vb[0] = sz_vpoly_bis
    tot_sz_ot[0] = sz_others
    return
