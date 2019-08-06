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
from libc.math cimport log2 as Clog2
from libc.stdlib cimport malloc, free, realloc
from cython.parallel import prange
from cython.parallel cimport parallel
from cpython.array cimport array, clone
from _basic_geom_tools cimport _VSMALL

# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================
cdef inline long discretize_line1d_core(double* LMinMax, double dstep,
                                         double[2] DL, bint Lim,
                                         int mode, double margin,
                                         double** ldiscret_arr,
                                         double[1] resolution,
                                         long** lindex_arr, long[1] N) nogil:
    cdef int[1] nL0
    cdef long[1] Nind

    first_discretize_line1d_core(LMinMax, dstep,
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
    second_discretize_line1d_core(LMinMax, ldiscret_arr[0], lindex_arr[0],
                                   nL0[0], resolution[0], Nind[0])
    return Nind[0]

cdef inline void first_discretize_line1d_core(double* LMinMax,
                                              double dstep,
                                              double[1] resolution,
                                              long[1] num_cells,
                                              long[1] Nind,
                                              int[1] nL0,
                                              double[2] DL,
                                              bint Lim,
                                              int mode,
                                              double margin) nogil:
    """
    Computes the resolution, the desired limits, and the number of cells when
    discretising the segmen LMinMax with the given parameters. It doesn't do the
    actual discretization.
    For that part, please refer to: second_discretize_line1d_core
    """
    cdef int nL1, ii, jj
    cdef double abs0, abs1
    cdef double inv_resol, new_margin
    cdef double[2] desired_limits

    # .. Computing "real" discretization step, depending on `mode`..............
    if mode == 0: # absolute
        num_cells[0] = <int>Cceil((LMinMax[1] - LMinMax[0])/dstep)
    else: # relative
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
    inv_resol = 1./resolution[0]
    new_margin = margin*resolution[0]
    abs0 = Cabs(desired_limits[0] - LMinMax[0])
    if abs0 - resolution[0] * Cfloor(abs0 * inv_resol) < new_margin:
        nL0[0] = int(Cround((desired_limits[0] - LMinMax[0]) * inv_resol))
    else:
        nL0[0] = int(Cfloor((desired_limits[0] - LMinMax[0]) * inv_resol))
    abs1 = Cabs(desired_limits[1] - LMinMax[0])
    if abs1 - resolution[0] * Cfloor(abs1 * inv_resol) < new_margin:
        nL1 = int(Cround((desired_limits[1] - LMinMax[0]) * inv_resol) - 1)
    else:
        nL1 = int(Cfloor((desired_limits[1] - LMinMax[0]) * inv_resol))
    # Get the total number of indices
    Nind[0] = nL1 + 1 - nL0[0]
    return

cdef inline void second_discretize_line1d_core(double* LMinMax,
                                               double* ldiscret,
                                               long* lindex,
                                               int nL0,
                                               double resolution,
                                               long Nind) nogil:
    """
    Does the actual discretization of the segment LMinMax.
    Computes the coordinates of the cells on the discretized segment and the
    associated list of indices.
    This function need some parameters computed with the first algorithm:
    first_discretize_line1d_core
    """
    cdef int ii, jj
    # .. Computing coordinates and indices .....................................
    for ii in range(Nind):
        jj = nL0 + ii
        lindex[ii] = jj
        ldiscret[ii] = LMinMax[0] + (0.5 + jj) * resolution
    return


cdef inline void simple_discretize_line1d(double[2] LMinMax, double dstep,
                                          int mode, double margin,
                                          double** ldiscret_arr,
                                          double[1] resolution,
                                          long[1] N) nogil:
    """
    Similar version, more simple :
    - Not possible to define a sub set
    - Gives back a discretized line WITH the min boundary
    - WITHOUT max boundary
    """
    cdef int ii
    cdef int numcells
    cdef double resol
    cdef double first = LMinMax[0]

    if mode == 0: # absolute
        numcells = <int>Cceil((LMinMax[1] - first)/dstep)
    else: # relative
        num_cells = <int>Cceil(1./dstep)
    if num_cells < 1 :
        num_cells = 1
    resol = (LMinMax[1] - first)/numcells
    resolution[0] = resol
    N[0] = numcells
    if (ldiscret_arr[0] == NULL):
        ldiscret_arr[0] = <double *>malloc(N[0] * sizeof(double))
    else:
        ldiscret_arr[0] = <double *>realloc(ldiscret_arr[0],
							N[0] * sizeof(double))
    for ii in range(numcells):
        ldiscret_arr[0][ii] = first + resol * ii
    return

# --- Utility function for discretizing line ---
cdef inline void cythonize_subdomain_dl(DL, double[2] dl_array):
    # All functions to discretize a line need to get a subdomain of
    # discretization which can be None for both extremities or only
    # one or none. However cython doesn't work too well with parameters
    # that can be an array, a list, none, etc. So this functions will convert
    # this obscure parameter to something more 'cythonic'
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
    return


# ==============================================================================
# =  Vessel's poloidal cut discretization
# ==============================================================================

cdef inline void discretize_vpoly_core(double[:, ::1] VPoly, double dstep,
                                       int mode, double margin, double DIn,
                                       double[:, ::1] VIn,
                                       double** XCross, double** YCross,
                                       double** reso, long** ind,
                                       long** numcells, double** Rref,
                                       double** XPolybis, double** YPolybis,
                                       int[1] tot_sz_vb, int[1] tot_sz_ot,
                                       int NP) nogil:
    cdef Py_ssize_t sz_vbis = 0
    cdef Py_ssize_t sz_others = 0
    cdef Py_ssize_t last_sz_vbis = 0
    cdef Py_ssize_t last_sz_othr = 0
    cdef int ii, jj
    cdef double v0, v1
    cdef double rv0, rv1
    cdef double inv_norm
    cdef double shiftx, shifty
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
    if Cabs(DIn) < _VSMALL:
        for ii in range(NP-1):
            v0 = VPoly[0,ii+1]-VPoly[0,ii]
            v1 = VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0*v0 + v1*v1)
            inv_norm = 1./LMinMax[1]
            discretize_line1d_core(LMinMax, dstep, dl_array, True,
                                    mode, margin, &ldiscret, loc_resolu,
                                    &lindex, &numcells[0][ii])
            # .. prepaaring Poly bis array......................................
            last_sz_vbis = sz_vbis
            sz_vbis += 1 + numcells[0][ii]
            XPolybis[0] = <double*>realloc(XPolybis[0], sz_vbis*sizeof(double))
            YPolybis[0] = <double*>realloc(YPolybis[0], sz_vbis*sizeof(double))
            XPolybis[0][sz_vbis - (1 + numcells[0][ii])] = VPoly[0, ii]
            YPolybis[0][sz_vbis - (1 + numcells[0][ii])] = VPoly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += numcells[0][ii]
            reso[0] = <double*>realloc(reso[0], sizeof(double)*sz_others)
            Rref[0] = <double*>realloc(Rref[0], sizeof(double)*sz_others)
            XCross[0] = <double*>realloc(XCross[0], sizeof(double)*sz_others)
            YCross[0] = <double*>realloc(YCross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            rv0 = loc_resolu[0]*v0
            rv1 = loc_resolu[0]*v1
            for jj in range(numcells[0][ii]):
                ind[0][last_sz_othr + jj] = last_sz_othr + jj
                reso[0][last_sz_othr + jj] = loc_resolu[0]
                Rref[0][last_sz_othr + jj] = VPoly[0,ii] + ldiscret[jj] * v0
                XCross[0][last_sz_othr + jj] = VPoly[0,ii] + ldiscret[jj] * v0
                YCross[0][last_sz_othr + jj] = VPoly[1,ii] + ldiscret[jj] * v1
                XPolybis[0][last_sz_vbis + jj] = VPoly[0,ii] + jj * rv0
                YPolybis[0][last_sz_vbis + jj] = VPoly[1,ii] + jj * rv1
        # We close the polygon of VPolybis
        sz_vbis += 1
        XPolybis[0] = <double*>realloc(XPolybis[0], sz_vbis*sizeof(double))
        YPolybis[0] = <double*>realloc(YPolybis[0], sz_vbis*sizeof(double))
        XPolybis[0][sz_vbis - 1] = VPoly[0, 0]
        YPolybis[0][sz_vbis - 1] = VPoly[1, 0]
    else:
        for ii in range(NP-1):
            v0 = VPoly[0,ii+1]-VPoly[0,ii]
            v1 = VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0*v0 + v1*v1)
            inv_norm = 1./LMinMax[1]
            discretize_line1d_core(LMinMax, dstep, dl_array, True,
                                    mode, margin, &ldiscret, loc_resolu,
                                    &lindex, &numcells[0][ii])
            # .. prepaaring Poly bis array......................................
            last_sz_vbis = sz_vbis
            sz_vbis += 1 + numcells[0][ii]
            XPolybis[0] = <double*>realloc(XPolybis[0], sz_vbis*sizeof(double))
            YPolybis[0] = <double*>realloc(YPolybis[0], sz_vbis*sizeof(double))
            XPolybis[0][sz_vbis - (1 + numcells[0][ii])] = VPoly[0, ii]
            YPolybis[0][sz_vbis - (1 + numcells[0][ii])] = VPoly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += numcells[0][ii]
            reso[0]  = <double*>realloc(reso[0],  sizeof(double)*sz_others)
            Rref[0]   = <double*>realloc(Rref[0],   sizeof(double)*sz_others)
            XCross[0] = <double*>realloc(XCross[0], sizeof(double)*sz_others)
            YCross[0] = <double*>realloc(YCross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            rv0 = loc_resolu[0]*v0
            rv1 = loc_resolu[0]*v1
            shiftx = DIn*VIn[0,ii]
            shifty = DIn*VIn[1,ii]
            for jj in range(numcells[0][ii]):
                ind[0][last_sz_othr] = last_sz_othr
                reso[0][last_sz_othr] = loc_resolu[0]
                Rref[0][last_sz_othr]   = VPoly[0,ii] + ldiscret[jj]*v0
                XCross[0][last_sz_othr] = VPoly[0,ii] + ldiscret[jj]*v0 + shiftx
                YCross[0][last_sz_othr] = VPoly[1,ii] + ldiscret[jj]*v1 + shifty
                XPolybis[0][last_sz_vbis + jj] = VPoly[0,ii] + jj * rv0
                YPolybis[0][last_sz_vbis + jj] = VPoly[1,ii] + jj * rv1
                last_sz_othr += 1
        # We close the polygon of VPolybis
        sz_vbis += 1
        XPolybis[0] = <double*>realloc(XPolybis[0], sz_vbis*sizeof(double))
        YPolybis[0] = <double*>realloc(YPolybis[0], sz_vbis*sizeof(double))
        XPolybis[0][sz_vbis - 1] = VPoly[0, 0]
        YPolybis[0][sz_vbis - 1] = VPoly[1, 0]
    tot_sz_vb[0] = sz_vbis
    tot_sz_ot[0] = sz_others
    return


# ------------------------------------------------------------------------------
# - Simplified version of previous algo
# ------------------------------------------------------------------------------
cdef inline void simple_discretize_vpoly_core(double[:, ::1] VPoly,
                                              int num_pts,
                                              double dstep,
                                              double** XCross,
                                              double** YCross,
                                              int[1] new_nb_pts,
                                              int mode,
                                              double margin) nogil:
    cdef Py_ssize_t sz_others = 0
    cdef Py_ssize_t last_sz_othr = 0
    cdef int ii, jj
    cdef double v0, v1
    cdef double inv_norm
    cdef double[1] loc_resolu
    cdef double[2] LMinMax
    cdef long[1] numcells
    cdef double* ldiscret = NULL
    #.. initialization..........................................................
    LMinMax[0] = 0.
    #.. Filling arrays..........................................................
    for ii in range(num_pts-1):
        v0 = VPoly[0,ii+1]-VPoly[0,ii]
        v1 = VPoly[1,ii+1]-VPoly[1,ii]
        LMinMax[1] = Csqrt(v0*v0 + v1*v1)
        inv_norm = 1./LMinMax[1]
        simple_discretize_line1d(LMinMax, dstep, mode, margin,
                                 &ldiscret, loc_resolu, &numcells[0])
        # .. preparing other arrays ........................................
        last_sz_othr = sz_others
        sz_others += numcells[0]
        XCross[0] = <double*>realloc(XCross[0], sizeof(double)*sz_others)
        YCross[0] = <double*>realloc(YCross[0], sizeof(double)*sz_others)
        # ...
        v0 = v0 * inv_norm
        v1 = v1 * inv_norm
        for jj in range(numcells[0]):
            XCross[0][last_sz_othr + jj] = VPoly[0,ii] + ldiscret[jj] * v0
            YCross[0][last_sz_othr + jj] = VPoly[1,ii] + ldiscret[jj] * v1
    # We close the polygon of VPolybis
    new_nb_pts[0] = sz_others
    free(ldiscret)
    return


# ==============================================================================
# == LOS sampling
# ==============================================================================

# -- Quadrature Rules : Middle Rule --------------------------------------------
cdef inline void middle_rule_rel_single(int num_raf,
                                        double los_kmin,
                                        double loc_resol,
                                        double* los_coeffs) nogil:
    # Middle quadrature rule with relative resolution step
    # for a single particle
    cdef Py_ssize_t jj

    for jj in prange(num_raf):
        los_coeffs[jj] = los_kmin + (0.5 + jj)*loc_resol
    return

cdef inline void middle_rule_rel(int num_los, int num_raf,
                                 double* los_kmin,
                                 double* los_kmax,
                                 double* los_resolution,
                                 double* los_coeffs,
                                 long* los_ind,
                                 int num_threads) nogil:
    # Middle quadrature rule with relative resolution step
    # for MULTIPLE LOS
    cdef Py_ssize_t ii
    cdef int first_index
    cdef double inv_nraf
    cdef double loc_resol
    inv_nraf = 1./num_raf

    for ii in range(num_los):
        if ii == 0:
            los_ind[ii] = num_raf
        else:
            los_ind[ii] = num_raf + los_ind[ii-1]
        with nogil, parallel(num_threads=num_threads):
            loc_resol = (los_kmax[ii] - los_kmin[ii])*inv_nraf
            los_resolution[ii] = loc_resol
            first_index = ii*num_raf
            middle_rule_rel_single(num_raf, los_kmin[ii],
                                   loc_resol, &los_coeffs[first_index])
    return

cdef inline void middle_rule_abs_1_single(double inv_resol,
                                          double los_kmin,
                                          double los_kmax,
                                          double* los_resolution,
                                          long* ind_cum) nogil:
    # Middle quadrature rule with absolute resolution step
    # for one LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_2, this function computes the resolutions
    # and the right indices
    cdef long num_raf
    cdef double seg_length
    cdef double loc_resol
    # ...
    seg_length = los_kmax - los_kmin
    num_raf = <long>(Cceil(seg_length*inv_resol))
    loc_resol = seg_length / num_raf
    los_resolution[0] = loc_resol
    ind_cum[0] = num_raf
    return


cdef inline void middle_rule_abs_1(int num_los, double resol,
                                   double* los_kmin,
                                   double* los_kmax,
                                   double* los_resolution,
                                   long* ind_cum,
                                   int num_threads) nogil:
    # Middle quadrature rule with absolute resolution step
    # for SEVERAL LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_2, this function computes the resolutions
    # and the right indices
    cdef Py_ssize_t ii
    cdef double inv_resol
    # ...
    with nogil, parallel(num_threads=num_threads):
        inv_resol = 1./resol
        for ii in prange(num_los):
            middle_rule_abs_1_single(inv_resol, los_kmin[ii],
                                     los_kmax[ii],
                                     &los_resolution[ii],
                                     &ind_cum[ii])
    return

cdef inline void middle_rule_abs_2_single(long num_raf,
                                          double loc_x,
                                          double loc_resol,
                                          double* los_coeffs) nogil:
    # Middle quadrature rule with absolute resolution step
    # for one LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_2, this function computes the coeffs
    cdef Py_ssize_t jj

    for jj in prange(num_raf):
        los_coeffs[jj] = loc_x + (0.5 + jj) * loc_resol
    return

cdef inline void middle_rule_abs_2(int num_los,
                                   double* los_kmin,
                                   long* ind_cum,
                                   double* los_resolution,
                                   double* los_coeffs,
                                   int num_threads) nogil:
    # Middle quadrature rule with absolute resolution step
    # for SEVERAL LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_2, this function computes the coeffs
    cdef Py_ssize_t ii
    cdef long num_raf
    cdef long first_index
    cdef double loc_resol
    cdef double loc_x
    # filling tab......
    for ii in range(num_los):
        num_raf = ind_cum[ii]
        if ii==0:
            first_index = 0
        else:
            first_index = ind_cum[ii-1]
            ind_cum[ii] = first_index + ind_cum[ii]
        loc_resol = los_resolution[ii]
        loc_x = los_kmin[ii]
        with nogil, parallel(num_threads=num_threads):
            middle_rule_abs_2_single(num_raf, loc_x, loc_resol,
                                     &los_coeffs[first_index])
    return


cdef inline void middle_rule_abs_var_single(int num_raf,
                                            double loc_resol,
                                            double los_kmin,
                                            double* los_coeffs) nogil:
    # Middle quadrature rule with absolute variable resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf):
        los_coeffs[jj] = los_kmin + (0.5 + jj) * loc_resol
    return

cdef inline void middle_rule_abs_var(int num_los, double* resolutions,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Middle quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    cdef double seg_length
    # ...
    with nogil, parallel(num_threads=num_threads):
        for ii in range(num_los):
            seg_length = los_kmax[ii] - los_kmin[ii]
            num_raf = <int>(Cceil(seg_length/resolutions[ii]))
            loc_resol = seg_length / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                los_ind[ii] = num_raf
                los_coeffs[0] = <double*>malloc(num_raf * sizeof(double))
                first_index = 0
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                              los_ind[ii] * sizeof(double))
            middle_rule_abs_var_single(num_raf,
                                       loc_resol,
                                       los_kmin[ii],
                                       &los_coeffs[0][first_index])
    return

cdef inline void middle_rule_rel_var_single(int num_raf,
                                            double loc_resol,
                                            double los_kmin,
                                            double* los_coeffs) nogil:
    # Middle quadrature rule with relative variable resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf):
        los_coeffs[jj] = los_kmin + (0.5 + jj) * loc_resol
    return

cdef inline void middle_rule_rel_var(int num_los, double* resolutions,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Middle quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            num_raf = <int>(Cceil(1./resolutions[ii]))
            loc_resol = 1./num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf
                los_coeffs[0] = <double*>malloc(num_raf * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                              los_ind[ii] * sizeof(double))
            middle_rule_rel_var_single(num_raf, loc_resol, los_kmin[ii],
                                       &los_coeffs[0][first_index])
    return

# -- Quadrature Rules : Left Rule ----------------------------------------------
cdef inline void left_rule_rel_single(int num_raf,
                                      double inv_nraf,
                                      double loc_x,
                                      double loc_resol,
                                      double* los_coeffs) nogil:
    # Left quadrature rule with relative resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in range(num_raf + 1):
        los_coeffs[jj] = loc_x + jj * loc_resol
    return


cdef inline void left_rule_rel(int num_los, int num_raf,
                               double* los_kmin,
                               double* los_kmax,
                               double* los_resolution,
                               double* los_coeffs,
                               long* los_ind, int num_threads) nogil:
    # Left quadrature rule with relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int first_index
    cdef double inv_nraf
    cdef double loc_resol
    cdef double loc_x
    inv_nraf = 1./num_raf
    # ...
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(num_los):
            loc_x = los_kmin[ii]
            loc_resol = (los_kmax[ii] - loc_x)*inv_nraf
            los_resolution[ii] = loc_resol
            first_index = ii*(num_raf + 1)
            los_ind[ii] = first_index + num_raf + 1
            left_rule_rel_single(num_raf, inv_nraf, loc_x, loc_resol,
                                 &los_coeffs[first_index])
    return

cdef inline void simps_left_rule_abs_single(int num_raf,
                                            double loc_resol,
                                            double los_kmin,
                                            double* los_coeffs) nogil:
    # Simpson left quadrature rule with absolute resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return


cdef inline void simps_left_rule_abs(int num_los, double resol,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Simpson left quadrature rule with absolute resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    cdef double inv_resol = 1./resol
    # ...
    for ii in range(num_los):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length*inv_resol))
        if num_raf%2==1:
            num_raf = num_raf + 1
        loc_resol = seg_length / num_raf
        los_resolution[ii] = loc_resol
        if ii == 0:
            first_index = 0
            los_ind[ii] = num_raf + 1
            los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
        else:
            first_index = los_ind[ii -1]
            los_ind[ii] = num_raf +  1 + first_index
            los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                             los_ind[ii] * sizeof(double))
        simps_left_rule_abs_single(num_raf, loc_resol, los_kmin[ii],
                                   &los_coeffs[0][first_index])
    return

cdef inline void romb_left_rule_abs_single(int num_raf,
                                           double loc_resol,
                                           double los_kmin,
                                           double* los_coeffs) nogil:
    # Romboid left quadrature rule with relative resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return

cdef inline void romb_left_rule_abs(int num_los, double resol,
                                    double* los_kmin,
                                    double* los_kmax,
                                    double* los_resolution,
                                    double** los_coeffs,
                                    long* los_ind, int num_threads) nogil:
    # Romboid left quadrature rule with relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    cdef double inv_resol = 1./resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            seg_length = los_kmax[ii] - los_kmin[ii]
            num_raf = <int>(Cceil(seg_length*inv_resol))
            num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
            loc_resol = seg_length / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf + 1
                los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf +  1 + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                                 los_ind[ii] * sizeof(double))
            romb_left_rule_abs_single(num_raf, loc_resol, los_kmin[ii],
                                      &los_coeffs[0][first_index])
    return


cdef inline void simps_left_rule_rel_var_single(int num_raf,
                                                double loc_resol,
                                                double los_kmin,
                                                double* los_coeffs) nogil:
    # Simpson left quadrature rule with variable relative resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return

cdef inline void simps_left_rule_rel_var(int num_los, double* resolutions,
                                         double* los_kmin,
                                         double* los_kmax,
                                         double* los_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil:
    # Simpson left quadrature rule with variable relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            num_raf = <int>(Cceil(1./resolutions[ii]))
            if num_raf%2==1:
                num_raf = num_raf+1
            loc_resol = 1. / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf + 1
                los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf +  1 + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                                 los_ind[ii] * sizeof(double))
            simps_left_rule_rel_var_single(num_raf, loc_resol, los_kmin[ii],
                                           &los_coeffs[0][first_index])
    return



cdef inline void simps_left_rule_abs_var_single(int num_raf,
                                                double loc_resol,
                                                double los_kmin,
                                                double* los_coeffs) nogil:
    # Simpson left quadrature rule with absolute variable resolution step
    # for one LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return

cdef inline void simps_left_rule_abs_var(int num_los, double* resolutions,
                                         double* los_kmin,
                                         double* los_kmax,
                                         double* los_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil:
    # Simpson left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            seg_length = los_kmax[ii] - los_kmin[ii]
            num_raf = <int>(Cceil(seg_length/resolutions[ii]))
            if num_raf%2==1:
                num_raf = num_raf+1
            loc_resol = seg_length / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf + 1
                los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf +  1 + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                                 los_ind[ii] * sizeof(double))
            simps_left_rule_abs_var_single(num_raf, loc_resol,
                                           los_kmin[ii],
                                           &los_coeffs[0][first_index])
    return

cdef inline void romb_left_rule_rel_var_single(int num_raf,
                                               double loc_resol,
                                               double los_kmin,
                                               double* los_coeffs) nogil:
    # Romboid left quadrature rule with relative variable resolution step
    # for ONE LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return


cdef inline void romb_left_rule_rel_var(int num_los, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* los_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            num_raf = <int>(Cceil(1./resolutions[ii]))
            num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
            loc_resol = 1. / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf + 1
                los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf +  1 + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                                 los_ind[ii] * sizeof(double))
            romb_left_rule_rel_var_single(num_raf, loc_resol, los_kmin[ii],
                                          &los_coeffs[0][first_index])
    return


cdef inline void romb_left_rule_abs_var_single(int num_raf,
                                               double loc_resol,
                                               double los_kmin,
                                               double* los_coeffs) nogil:
    # Romboid left quadrature rule with absolute variable resolution step
    # for ONE LOS
    cdef Py_ssize_t jj
    # ...
    for jj in prange(num_raf + 1):
        los_coeffs[jj] = los_kmin + jj * loc_resol
    return

cdef inline void romb_left_rule_abs_var(int num_los, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* los_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    # ...
    for ii in range(num_los):
        with nogil, parallel(num_threads=num_threads):
            seg_length = los_kmax[ii] - los_kmin[ii]
            num_raf = <int>(Cceil(seg_length/resolutions[ii]))
            num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
            loc_resol = seg_length / num_raf
            los_resolution[ii] = loc_resol
            if ii == 0:
                first_index = 0
                los_ind[ii] = num_raf + 1
                los_coeffs[0] = <double*>malloc((num_raf + 1) * sizeof(double))
            else:
                first_index = los_ind[ii-1]
                los_ind[ii] = num_raf +  1 + first_index
                los_coeffs[0] = <double*>realloc(los_coeffs[0],
                                                 los_ind[ii] * sizeof(double))

            romb_left_rule_abs_var_single(num_raf, loc_resol, los_kmin[ii],
                                          &los_coeffs[0][first_index])
    return

# -- Get number of integration mode --------------------------------------------
cdef inline int get_nb_imode(str imode) :
    # gil required...........
    if imode == 'sum':
        return 0
    if imode == 'simps':
        return 1
    if imode == 'romb':
        return 2
    return -1

cdef inline int get_nb_dmode(str dmode) :
    # gil required...........
    if dmode == 'rel':
        return 1
    if dmode == 'abs':
        return 0
    return -1


# ==============================================================================
# == LOS sampling Algorithm for a SINGLE LOS
# ==============================================================================
cdef inline int LOS_get_sample_single(double los_kmin, double los_kmax,
                                      double resol, int n_dmode, int n_imode,
                                      double[1] eff_res,
                                      double** coeffs) nogil:
    """
    Sampling line of sight of origin ray_orig, direction vector ray_vdir,
    with discretization step resol, using the discretization method n_dmode,
    and the quadrature rule n_imode. los_kmin defines the first limit of the LOS
    Out parameters
    --------------
    eff_res : effective resolution used
    coeffs : 'k' coefficients on ray.
    Returns
    =======
       size of elements in coeffs[0]
    The different type of discretizations and quadratures:
    n_dmode
    =======
      - 0 : the discretization step given is absolute ('abs')
      - 1 : the discretization step given is relative ('rel')
    n_imode
    =====
      - 0 : 'sum' quadrature, using the N segment centers
      - 1 : 'simps' return N+1 egdes, N even (for scipy.integrate.simps)
      - 2 : 'romb' return N+1 edges, N+1 = 2**k+1 (for scipy.integrate.romb)
    """
    cdef int nraf
    cdef long[1] ind_cum
    cdef double invnraf
    cdef double invresol
    cdef double seg_length
    # ...
    if n_dmode == 1:
        # discretization step is relative
        nraf = <int> Cceil(1./resol)
        if n_imode==0:
            # 'sum' quad
            coeffs[0] = <double*>malloc(nraf*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)/resol
            middle_rule_rel_single(nraf, los_kmin, eff_res[0],
                                   &coeffs[0][0])
            return nraf
        elif n_imode==1:
            # 'simps' quad
            if nraf%2==1:
                nraf = nraf+1
            invnraf = 1./nraf
            coeffs[0] = <double*>malloc((nraf + 1)*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)*invnraf
            left_rule_rel_single(nraf, invnraf, los_kmin,
                                 eff_res[0], &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            nraf = 2**(<int>(Cceil(Clog2(nraf))))
            invnraf = 1./nraf
            coeffs[0] = <double*>malloc((nraf + 1)*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)*invnraf
            left_rule_rel_single(nraf, invnraf, los_kmin,
                                 eff_res[0], &coeffs[0][0])
            return nraf + 1
    else:
        # discretization step is absolute, n_dmode==0
        if n_imode==0:
            # 'sum' quad
            invresol = 1./resol
            middle_rule_abs_1_single(invresol, los_kmin, los_kmax,
                                         &eff_res[0], &ind_cum[0])
            coeffs[0] = <double*>malloc((ind_cum[0])*sizeof(double))
            middle_rule_abs_2_single(ind_cum[0], los_kmin, eff_res[0],
                                     &coeffs[0][0])
            return ind_cum[0]
        elif n_imode==1:
            # 'simps' quad
            seg_length = los_kmax - los_kmin
            nraf = <int>(Cceil(seg_length/resol))
            if nraf%2==1:
                nraf = nraf+1
            eff_res[0] = seg_length / nraf
            coeffs[0] = <double*>malloc((nraf+1)*sizeof(double))
            simps_left_rule_abs_single(nraf, eff_res[0],
                                       los_kmin, &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            seg_length = los_kmax - los_kmin
            nraf = <int>(Cceil(seg_length/resol))
            nraf = 2**(<int>(Cceil(Clog2(nraf))))
            eff_res[0] = seg_length / nraf
            coeffs[0] = <double*>malloc((nraf+1)*sizeof(double))
            romb_left_rule_abs_single(nraf, eff_res[0], los_kmin,
                                      &coeffs[0][0])
            return nraf + 1
