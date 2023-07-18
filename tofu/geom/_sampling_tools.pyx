# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
#
################################################################################
# Utility functions for sampling and discretizating
################################################################################
from libc.math cimport ceil as c_ceil, fabs as c_abs
from libc.math cimport floor as c_floor, round as c_round
from libc.math cimport sqrt as c_sqrt
from libc.math cimport pi as c_pi, cos as c_cos, sin as c_sin, atan2 as c_atan2
from libc.math cimport isnan as c_isnan
from libc.math cimport NAN as C_NAN
from libc.math cimport log2 as c_log2
from libc.stdlib cimport malloc, free, realloc
from cython.parallel import prange
from cython.parallel cimport parallel
from cython.parallel cimport threadid
from cpython.array cimport array, clone
# for utility functions:
import numpy as np
cimport numpy as cnp
cimport cython
# tofu libs
from ._basic_geom_tools cimport _VSMALL
from ._basic_geom_tools cimport _TWOPI
from . cimport _basic_geom_tools as _bgt
from . cimport _raytracing_tools as _rt

# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================
cdef inline long discretize_line1d_core(double* lminmax, double dstep,
                                        double[2] dl, bint lim,
                                        int mode, double margin,
                                        double** ldiscret_arr,
                                        double[1] resolution,
                                        long** lindex_arr, long[1] n) nogil:
    """Discretizes a 1D line defined over `[liminmax[0], lminmax[1]]` with
    a discretization step resolution (out value) computed from `dstep` which
    can be given in absolute or relative mode. It is possible to only get a
    subdomain `[dl[0], dl[1]]` of the line. `lindex_arr` indicates the indices
    of points to take into account depending on the subdomain dl. n indicates
    the number of points on the discretized subdomain."""
    cdef int[1] nL0
    cdef long[1] nind
    # ..
    first_discretize_line1d_core(lminmax, dstep,
                                 resolution, n, &nind[0], &nL0[0],
                                 dl, lim, mode, margin)
    if ldiscret_arr[0] == NULL:
        ldiscret_arr[0] = <double *>malloc(nind[0] * sizeof(double))
    else:
        ldiscret_arr[0] = <double *>realloc(ldiscret_arr[0],
                                            nind[0] * sizeof(double))
    if lindex_arr[0] == NULL:
        lindex_arr[0] = <long *>malloc(nind[0] * sizeof(long))
    else:
        lindex_arr[0] = <long *>realloc(lindex_arr[0], nind[0] * sizeof(long))
    second_discretize_line1d_core(lminmax, ldiscret_arr[0], lindex_arr[0],
                                  nL0[0], resolution[0], nind[0])
    return nind[0]


cdef inline void first_discretize_line1d_core(double* lminmax,
                                              double dstep,
                                              double[1] resolution,
                                              long[1] ncells,
                                              long[1] nind,
                                              int[1] nl0,
                                              double[2] dl,
                                              bint lim,
                                              int mode,
                                              double margin) nogil:
    """
    Computes the resolution, the desired limits, and the number of cells when
    discretising the segment lminmax with the given parameters. It doesn't do
    the actual discretization.
    For that part, please refer to: second_discretize_line1d_core
    """
    cdef int nl1
    cdef double abs0, abs1
    cdef double inv_resol, new_margin
    cdef double[2] desired_limits

    # .. Computing "real" discretization step, depending on `mode`..............
    if mode == 0: # absolute
        ncells[0] = <int>c_ceil((lminmax[1] - lminmax[0]) / dstep)
    else: # relative
        ncells[0] = <int>c_ceil(1. / dstep)
    resolution[0] = (lminmax[1] - lminmax[0]) / ncells[0]
    # .. Computing desired limits ..............................................
    if c_isnan(dl[0]) and c_isnan(dl[1]):
        desired_limits[0] = lminmax[0]
        desired_limits[1] = lminmax[1]
    else:
        if c_isnan(dl[0]):
            dl[0] = lminmax[0]
        if c_isnan(dl[1]):
            dl[1] = lminmax[1]
        if lim and dl[0]<lminmax[0]:
            dl[0] = lminmax[0]
        if lim and dl[1]>lminmax[1]:
            dl[1] = lminmax[1]
        desired_limits[0] = dl[0]
        desired_limits[1] = dl[1]
    # .. Get the extreme indices of the mesh elements that really need to be
    # created within those limits...............................................
    inv_resol = 1./resolution[0]
    new_margin = margin*resolution[0]
    abs0 = c_abs(desired_limits[0] - lminmax[0])
    if abs0 - resolution[0] * c_floor(abs0 * inv_resol + _VSMALL) < new_margin:
        nl0[0] = int(c_round((desired_limits[0] - lminmax[0]) * inv_resol))
    else:
        nl0[0] = int(c_floor((desired_limits[0] - lminmax[0]) * inv_resol))
    abs1 = c_abs(desired_limits[1] - lminmax[0])
    if abs1 - resolution[0] * c_floor(abs1 * inv_resol + _VSMALL) < new_margin:
        nl1 = int(c_round((desired_limits[1] - lminmax[0]) * inv_resol) - 1)
    else:
        nl1 = int(c_floor((desired_limits[1] - lminmax[0]) * inv_resol))
    # Get the total number of indices
    nind[0] = nl1 + 1 - nl0[0]
    return


cdef inline void second_discretize_line1d_core(double* lminmax,
                                               double* ldiscret,
                                               long* lindex,
                                               int nl0,
                                               double resolution,
                                               long nind) nogil:
    """
    Does the actual discretization of the segment lminmax.
    Computes the coordinates of the cells on the discretized segment and the
    associated list of indices.
    This function need some parameters computed with the first algorithm:
    first_discretize_line1d_core
    """
    cdef int ii, jj
    # .. Computing coordinates and indices .....................................
    for ii in range(nind):
        jj = nl0 + ii
        lindex[ii] = jj
        ldiscret[ii] = lminmax[0] + (0.5 + jj) * resolution
    return


cdef inline void simple_discretize_line1d(double[2] lminmax, double dstep,
                                          int mode, double margin,
                                          double** ldiscret_arr,
                                          double[1] resolution,
                                          long[1] n) nogil:
    """
    Similar version, more simple :
    - Not possible to define a sub set
    - Gives back a discretized line WITH the min boundary
    - WITHOUT max boundary
    """
    cdef int ii
    cdef int ncells
    cdef double resol
    cdef double first = lminmax[0]

    if mode == 0: # absolute
        ncells = <int>c_ceil((lminmax[1] - first) / dstep)
    else: # relative
        ncells = <int>c_ceil(1. / dstep)
    if ncells < 1 :
        ncells = 1
    resol = (lminmax[1] - first) / ncells
    resolution[0] = resol
    n[0] = ncells
    if ldiscret_arr[0] == NULL:
        ldiscret_arr[0] = <double *>malloc(n[0] * sizeof(double))
    else:
        ldiscret_arr[0] = <double *>realloc(ldiscret_arr[0],
                                            n[0] * sizeof(double))
    for ii in range(ncells):
        ldiscret_arr[0][ii] = first + resol * ii
    return


# --- Utility function for discretizing line ---
cdef inline void cythonize_subdomain_dl(dl, double[2] dl_array):
    # All functions to discretize a line need to get a subdomain of
    # discretization which can be None for both extremities or only
    # one or none. However cython doesn't work too well with parameters
    # that can be an array, a list, none, etc. So this functions will convert
    # this obscure parameter to something more 'cythonic'
    if dl is None:
        dl_array[0] = C_NAN
        dl_array[1] = C_NAN
    else:
        if dl[0] is None:
            dl_array[0] = C_NAN
        else:
            dl_array[0] = dl[0]
        if dl[1] is None:
            dl_array[1] = C_NAN
        else:
            dl_array[1] = dl[1]
    return


# ==============================================================================
# =  Vessel's poloidal cut discretization
# ==============================================================================
cdef inline void discretize_vpoly_core(double[:, ::1] ves_poly, double dstep,
                                       int mode, double margin, double din,
                                       double[:, ::1] ves_vin,
                                       double** xcross, double** ycross,
                                       double** reso, long** ind,
                                       long** ncells, double** rref,
                                       double** xpolybis, double** ypolybis,
                                       int[1] tot_sz_vb, int[1] tot_sz_ot,
                                       int np) nogil:
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
    cdef double[2] lminmax
    cdef double[2] dl_array
    cdef double* ldiscret = NULL
    cdef long* lindex = NULL

    #.. initialization..........................................................
    lminmax[0] = 0.
    dl_array[0] = C_NAN
    dl_array[1] = C_NAN
    ncells[0] = <long*>malloc((np-1)*sizeof(long))
    #.. Filling arrays..........................................................
    if c_abs(din) < _VSMALL:
        for ii in range(np-1):
            v0 = ves_poly[0, ii+1]-ves_poly[0, ii]
            v1 = ves_poly[1, ii+1]-ves_poly[1, ii]
            lminmax[1] = c_sqrt(v0 * v0 + v1 * v1)
            inv_norm = 1. / lminmax[1]
            discretize_line1d_core(lminmax, dstep, dl_array, True,
                                   mode, margin, &ldiscret, loc_resolu,
                                   &lindex, &ncells[0][ii])
            # .. preparing Poly bis array......................................
            last_sz_vbis = sz_vbis
            sz_vbis += 1 + ncells[0][ii]
            xpolybis[0] = <double*>realloc(xpolybis[0], sz_vbis*sizeof(double))
            ypolybis[0] = <double*>realloc(ypolybis[0], sz_vbis*sizeof(double))
            xpolybis[0][sz_vbis - (1 + ncells[0][ii])] = ves_poly[0, ii]
            ypolybis[0][sz_vbis - (1 + ncells[0][ii])] = ves_poly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += ncells[0][ii]
            reso[0] = <double*>realloc(reso[0], sizeof(double)*sz_others)
            rref[0] = <double*>realloc(rref[0], sizeof(double)*sz_others)
            xcross[0] = <double*>realloc(xcross[0], sizeof(double)*sz_others)
            ycross[0] = <double*>realloc(ycross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            rv0 = loc_resolu[0]*v0
            rv1 = loc_resolu[0]*v1
            for jj in range(ncells[0][ii]):
                ind[0][last_sz_othr + jj] = last_sz_othr + jj
                reso[0][last_sz_othr + jj] = loc_resolu[0]
                rref[0][last_sz_othr + jj] = ves_poly[0, ii] + ldiscret[jj] * v0
                xcross[0][last_sz_othr + jj] = ves_poly[0, ii] + ldiscret[jj] * v0
                ycross[0][last_sz_othr + jj] = ves_poly[1, ii] + ldiscret[jj] * v1
                xpolybis[0][last_sz_vbis + jj] = ves_poly[0, ii] + jj * rv0
                ypolybis[0][last_sz_vbis + jj] = ves_poly[1, ii] + jj * rv1
        # We close the polygon of VPolybis
        sz_vbis += 1
        xpolybis[0] = <double*>realloc(xpolybis[0], sz_vbis*sizeof(double))
        ypolybis[0] = <double*>realloc(ypolybis[0], sz_vbis*sizeof(double))
        xpolybis[0][sz_vbis - 1] = ves_poly[0, 0]
        ypolybis[0][sz_vbis - 1] = ves_poly[1, 0]
    else:
        for ii in range(np-1):
            v0 = ves_poly[0, ii+1]-ves_poly[0, ii]
            v1 = ves_poly[1, ii+1]-ves_poly[1, ii]
            lminmax[1] = c_sqrt(v0 * v0 + v1 * v1)
            inv_norm = 1. / lminmax[1]
            discretize_line1d_core(lminmax, dstep, dl_array, True,
                                   mode, margin, &ldiscret, loc_resolu,
                                   &lindex, &ncells[0][ii])
            # .. prepaaring Poly bis array......................................
            last_sz_vbis = sz_vbis
            sz_vbis += 1 + ncells[0][ii]
            xpolybis[0] = <double*>realloc(xpolybis[0], sz_vbis*sizeof(double))
            ypolybis[0] = <double*>realloc(ypolybis[0], sz_vbis*sizeof(double))
            xpolybis[0][sz_vbis - (1 + ncells[0][ii])] = ves_poly[0, ii]
            ypolybis[0][sz_vbis - (1 + ncells[0][ii])] = ves_poly[1, ii]
            # .. preparing other arrays ........................................
            last_sz_othr = sz_others
            sz_others += ncells[0][ii]
            reso[0]  = <double*>realloc(reso[0],  sizeof(double)*sz_others)
            rref[0]   = <double*>realloc(rref[0],   sizeof(double)*sz_others)
            xcross[0] = <double*>realloc(xcross[0], sizeof(double)*sz_others)
            ycross[0] = <double*>realloc(ycross[0], sizeof(double)*sz_others)
            ind[0] = <long*>realloc(ind[0], sizeof(long)*sz_others)
            # ...
            v0 = v0 * inv_norm
            v1 = v1 * inv_norm
            rv0 = loc_resolu[0]*v0
            rv1 = loc_resolu[0]*v1
            shiftx = din*ves_vin[0, ii]
            shifty = din*ves_vin[1, ii]
            for jj in range(ncells[0][ii]):
                ind[0][last_sz_othr] = last_sz_othr
                reso[0][last_sz_othr] = loc_resolu[0]
                rref[0][last_sz_othr]   = ves_poly[0, ii] + ldiscret[jj]*v0
                xcross[0][last_sz_othr] = ves_poly[0, ii] + ldiscret[jj]*v0 + shiftx
                ycross[0][last_sz_othr] = ves_poly[1, ii] + ldiscret[jj]*v1 + shifty
                xpolybis[0][last_sz_vbis + jj] = ves_poly[0, ii] + jj * rv0
                ypolybis[0][last_sz_vbis + jj] = ves_poly[1, ii] + jj * rv1
                last_sz_othr += 1
        # We close the polygon of VPolybis
        sz_vbis += 1
        xpolybis[0] = <double*>realloc(xpolybis[0], sz_vbis*sizeof(double))
        ypolybis[0] = <double*>realloc(ypolybis[0], sz_vbis*sizeof(double))
        xpolybis[0][sz_vbis - 1] = ves_poly[0, 0]
        ypolybis[0][sz_vbis - 1] = ves_poly[1, 0]
    tot_sz_vb[0] = sz_vbis
    tot_sz_ot[0] = sz_others
    return


# ------------------------------------------------------------------------------
# - Simplified version of previous algo
# ------------------------------------------------------------------------------
cdef inline void simple_discretize_vpoly_core(double[:, ::1] ves_poly,
                                              int num_pts,
                                              double dstep,
                                              double** xcross,
                                              double** ycross,
                                              int[1] new_nb_pts,
                                              int mode,
                                              double margin) nogil:
    cdef Py_ssize_t sz_others = 0
    cdef Py_ssize_t last_sz_othr = 0
    cdef int ii, jj
    cdef double v0, v1
    cdef double inv_norm
    cdef double[1] loc_resolu
    cdef double[2] lminmax
    cdef long[1] ncells
    cdef double* ldiscret = NULL
    #.. initialization..........................................................
    lminmax[0] = 0.
    #.. Filling arrays..........................................................
    for ii in range(num_pts-1):
        v0 = ves_poly[0, ii+1]-ves_poly[0, ii]
        v1 = ves_poly[1, ii+1]-ves_poly[1, ii]
        lminmax[1] = c_sqrt(v0 * v0 + v1 * v1)
        inv_norm = 1. / lminmax[1]
        simple_discretize_line1d(lminmax, dstep, mode, margin,
                                 &ldiscret, loc_resolu, &ncells[0])
        # .. preparing other arrays ........................................
        last_sz_othr = sz_others
        sz_others += ncells[0]
        xcross[0] = <double*>realloc(xcross[0], sizeof(double)*sz_others)
        ycross[0] = <double*>realloc(ycross[0], sizeof(double)*sz_others)
        # ...
        v0 = v0 * inv_norm
        v1 = v1 * inv_norm
        for jj in range(ncells[0]):
            xcross[0][last_sz_othr + jj] = ves_poly[0, ii] + ldiscret[jj] * v0
            ycross[0][last_sz_othr + jj] = ves_poly[1, ii] + ldiscret[jj] * v1
    # We close the polygon of VPolybis
    new_nb_pts[0] = sz_others
    free(ldiscret)
    return


# ==============================================================================
# == LOS sampling
# ==============================================================================

# -- Quadrature Rules : Middle Rule --------------------------------------------
cdef inline void middle_rule_single(int num_raf,
                                    double los_kmin,
                                    double loc_resol,
                                    double* los_coeffs) nogil:
    # Middle quadrature rule with relative resolution step
    # for a single particle
    cdef Py_ssize_t jj
    for jj in range(num_raf):
        los_coeffs[jj] = los_kmin + (0.5 + jj)*loc_resol
    return

cdef inline void middle_rule_rel(int nlos, int num_raf,
                                 double* los_kmin,
                                 double* los_kmax,
                                 double* eff_resolution,
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
    # doing special case ilos = 0:
    los_ind[0] = num_raf
    loc_resol = (los_kmax[0] - los_kmin[0])*inv_nraf
    eff_resolution[0] = loc_resol
    first_index = 0
    middle_rule_single(num_raf, los_kmin[0],
                       loc_resol, &los_coeffs[first_index])
    # Now for the rest:
    with nogil, parallel(num_threads=num_threads):
        for ii in range(1, nlos):
            los_ind[ii] = num_raf + los_ind[ii-1]
            loc_resol = (los_kmax[ii] - los_kmin[ii])*inv_nraf
            eff_resolution[ii] = loc_resol
            first_index = ii*num_raf
            middle_rule_single(num_raf, los_kmin[ii],
                               loc_resol, &los_coeffs[first_index])
    return

cdef inline void middle_rule_abs_s1_single(double inv_resol,
                                          double los_kmin,
                                          double los_kmax,
                                          double* eff_resolution,
                                          long* ind_cum) nogil:
    # Middle quadrature rule with absolute resolution step
    # for one LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_s2, this function computes the resolutions
    # and the right indices
    cdef int num_raf
    cdef double seg_length
    cdef double loc_resol
    # ...
    seg_length = los_kmax - los_kmin
    num_raf = <int>(c_ceil(seg_length * inv_resol))
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    ind_cum[0] = num_raf
    return


cdef inline void middle_rule_abs_s1(int nlos, double resol,
                                   double* los_kmin,
                                   double* los_kmax,
                                   double* eff_resolution,
                                   long* ind_cum,
                                   int num_threads) nogil:
    # Middle quadrature rule with absolute resolution step
    # for SEVERAL LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_s2, this function computes the resolutions
    # and the right indices
    cdef Py_ssize_t ii
    cdef double inv_resol
    # ...
    with nogil, parallel(num_threads=num_threads):
        inv_resol = 1./resol
        for ii in prange(nlos):
            middle_rule_abs_s1_single(inv_resol, los_kmin[ii],
                                     los_kmax[ii],
                                     &eff_resolution[ii],
                                     &ind_cum[ii])
    return

cdef inline void middle_rule_abs_s2(int nlos,
                                   double* los_kmin,
                                   double* eff_resolution,
                                   long* ind_cum,
                                   double* los_coeffs,
                                   int num_threads) nogil:
    # Middle quadrature rule with absolute resolution step
    # for SEVERAL LOS
    # First step of the function, this function should be called
    # before middle_rule_abs_s2, this function computes the coeffs
    cdef Py_ssize_t ii
    cdef long num_raf
    cdef long first_index
    cdef double loc_resol
    cdef double loc_x
    # Treating the first ilos seperately
    num_raf = ind_cum[0]
    first_index = 0
    loc_resol = eff_resolution[0]
    loc_x = los_kmin[0]
    middle_rule_single(num_raf, loc_x, loc_resol,
                       &los_coeffs[first_index])

    # filling tab...... CANNOT BE PARALLEL !!
    for ii in range(1, nlos):
        num_raf = ind_cum[ii]
        first_index = ind_cum[ii-1]
        ind_cum[ii] = first_index + ind_cum[ii]
        loc_resol = eff_resolution[ii]
        loc_x = los_kmin[ii]
        middle_rule_single(num_raf, loc_x, loc_resol,
                           &los_coeffs[first_index])
    return


cdef inline void middle_rule_abs_var_s1(int nlos,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* resolutions,
                                        double* eff_resolution,
                                        long* los_ind,
                                        long* los_nraf,
                                        int num_threads) nogil:
    # Middle quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    cdef double seg_length
    # Treating first ilos first ......................................
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(c_ceil(seg_length / resolutions[0]))
    loc_resol = seg_length / num_raf
    # keeping values
    los_nraf[0] = num_raf
    eff_resolution[0] = loc_resol
    los_ind[0] = num_raf
    first_index = 0
    # Now the rest ...................................................
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(c_ceil(seg_length / resolutions[ii]))
        loc_resol = seg_length / num_raf
        # keeping values
        los_nraf[ii] = num_raf
        eff_resolution[ii] = loc_resol
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf + first_index
    return


cdef inline void middle_rule_abs_var_s2(int nlos,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        double** los_coeffs,
                                        long* los_ind, long* los_nraf,
                                        int num_threads) nogil:
    # Middle quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol

    # Treating ilos= 0 first .......................................
    first_index = 0
    loc_resol = eff_resolution[0]
    num_raf = los_nraf[0]
    middle_rule_single(num_raf,
                       los_kmin[0],
                       loc_resol,
                       &los_coeffs[0][first_index])
    # ...
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(1, nlos):
            first_index = los_ind[ii-1]
            loc_resol = eff_resolution[ii]
            num_raf = los_nraf[ii]
            middle_rule_single(num_raf,
                               los_kmin[ii],
                               loc_resol,
                               &los_coeffs[0][first_index])
    return


cdef inline void middle_rule_abs_var(int nlos,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* resolutions,
                                     double* eff_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Middle quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    los_nraf = <long*> malloc(nlos * sizeof(long))
    middle_rule_abs_var_s1(nlos, los_kmin, los_kmax, resolutions,
                           eff_resolution, los_ind, &los_nraf[0],
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    middle_rule_abs_var_s2(nlos, los_kmin, los_kmax,
                           eff_resolution, los_coeffs,
                           los_ind, los_nraf, num_threads)
    # ...
    free(los_nraf)
    return


cdef inline void middle_rule_rel_var_s1(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        long* los_ind,
                                        long* los_nraf,
                                        int num_threads) nogil:
    # Middle quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # ... Treating the first los .....................................
    num_raf = <int>(c_ceil(1. / resolutions[0]))
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf
    # .. Treating the rest of los ....................................
    for ii in range(1,nlos):
        num_raf = <int>(c_ceil(1. / resolutions[ii]))
        loc_resol = (los_kmax[ii] - los_kmin[ii])/num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf + first_index
    return


cdef inline void middle_rule_rel_var_s2(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        double** los_coeffs,
                                        long* los_ind, long* los_nraf,
                                        int num_threads) nogil:
    # Middle quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # .. Treating first los .........................................
    num_raf = los_nraf[0]
    loc_resol = eff_resolution[0]
    first_index = 0
    middle_rule_single(num_raf, los_kmin[0], loc_resol,
                       &los_coeffs[0][first_index])
    # ... and the rest of los .......................................
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(1, nlos):
            num_raf = los_nraf[ii]
            loc_resol = eff_resolution[ii]
            first_index = los_ind[ii-1]
            middle_rule_single(num_raf, los_kmin[ii], loc_resol,
                               &los_coeffs[0][first_index])
    return


cdef inline void middle_rule_rel_var(int nlos, double* resolutions,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* eff_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Middle quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    middle_rule_rel_var_s1(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    middle_rule_rel_var_s2(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_coeffs,
                           los_ind,
                           los_nraf,
                           num_threads)
    free(los_nraf)
    return


# -- Quadrature Rules : Left Rule ----------------------------------------------
cdef inline void left_rule_single(int num_raf,
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


cdef inline void left_rule_rel(int nlos, int num_raf,
                               double* los_kmin,
                               double* los_kmax,
                               double* eff_resolution,
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
        for ii in prange(nlos):
            loc_x = los_kmin[ii]
            loc_resol = (los_kmax[ii] - loc_x)*inv_nraf
            eff_resolution[ii] = loc_resol
            first_index = ii*(num_raf + 1)
            los_ind[ii] = first_index + num_raf + 1
            left_rule_single(num_raf, loc_x, loc_resol,
                             &los_coeffs[first_index])
    return


cdef inline void simps_left_rule_abs_s1(int nlos, double resol,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        long* los_ind, long* los_nraf,
                                        int num_threads) nogil:
    # Simpson left quadrature rule with absolute resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    cdef double inv_resol = 1./resol
    # ... Treating the first los .......................................
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(c_ceil(seg_length * inv_resol))
    if num_raf%2==1:
        num_raf += 1
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ... Treating the rest of los .....................................
    for ii in range(1, nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(c_ceil(seg_length * inv_resol))
        if num_raf%2==1:
            num_raf += 1
        loc_resol = seg_length / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii -1]
        los_ind[ii] = num_raf +  1 + first_index
    return

cdef inline void left_rule_abs_s2(int nlos, double resol,
                                  double* los_kmin,
                                  double* los_kmax,
                                  double* eff_resolution,
                                  double** los_coeffs,
                                  long* los_ind, long* los_nraf,
                                  int num_threads) nogil:
    # Simpson left quadrature rule with absolute resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii,
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # ... Treating the first los .........................................
    num_raf = los_nraf[0]
    loc_resol = eff_resolution[0]
    first_index = 0
    left_rule_single(num_raf, los_kmin[0], loc_resol,
                     &los_coeffs[0][first_index])
    # ... Treating the rest of the los ...................................
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(1,nlos):
            num_raf = los_nraf[ii]
            loc_resol = eff_resolution[ii]
            first_index = los_ind[ii -1]
            left_rule_single(num_raf, los_kmin[ii], loc_resol,
                             &los_coeffs[0][first_index])
    return


cdef inline void simps_left_rule_abs(int nlos, double resol,
                                     double* los_kmin,
                                     double* los_kmax,
                                     double* eff_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil:
    # Simpson left quadrature rule with absolute resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    simps_left_rule_abs_s1(nlos, resol,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    los_coeffs[0][0] = -1.
    los_coeffs[0][1] = -1.
    los_coeffs[0][2] = -1.
    left_rule_abs_s2(nlos, resol,
                     los_kmin, los_kmax,
                     eff_resolution,
                     los_coeffs,
                     los_ind,
                     los_nraf,
                     num_threads)
    free(los_nraf)
    return


cdef inline void romb_left_rule_abs_s1(int nlos, double resol,
                                    double* los_kmin,
                                    double* los_kmax,
                                    double* eff_resolution,
                                    long* los_ind, long* los_nraf,
                                    int num_threads) nogil:
    # Romboid left quadrature rule with relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    cdef double inv_resol = 1./resol
    # ... Treating the first los ....................................
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(c_ceil(seg_length * inv_resol))
    num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ... Treating the rest of the los ..............................
    for ii in range(1, nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(c_ceil(seg_length * inv_resol))
        num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
        loc_resol = seg_length / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf +  1 + first_index
    return


cdef inline void romb_left_rule_abs(int nlos, double resol,
                                    double* los_kmin,
                                    double* los_kmax,
                                    double* eff_resolution,
                                    double** los_coeffs,
                                    long* los_ind, int num_threads) nogil:
    # Romboid left quadrature rule with relative resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    romb_left_rule_abs_s1(nlos, resol,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    left_rule_abs_s2(nlos, resol,
                     los_kmin, los_kmax,
                     eff_resolution,
                     los_coeffs,
                     los_ind,
                     los_nraf,
                     num_threads)
    free(los_nraf)
    return


cdef inline void simps_left_rule_rel_var_s1(int nlos, double* resolutions,
                                            double* los_kmin,
                                            double* los_kmax,
                                            double* eff_resolution,
                                            long* los_ind, long* los_nraf,
                                            int num_threads) nogil:
    # Simpson left quadrature rule with variable relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    num_raf = <int>(c_ceil(1. / resolutions[0]))
    if num_raf%2==1:
        num_raf += 1
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1, nlos):
        num_raf = <int>(c_ceil(1. / resolutions[ii]))
        if num_raf%2==1:
            num_raf += 1
        loc_resol = (los_kmax[ii] - los_kmin[ii]) / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf +  1 + first_index
    return


cdef inline void left_rule_rel_var_s2(int nlos, double* resolutions,
                                      double* los_kmin,
                                      double* los_kmax,
                                      double* eff_resolution,
                                      double** los_coeffs,
                                      long* los_ind, long* los_nraf,
                                      int num_threads) nogil:
    # Simpson left quadrature rule with variable relative resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    # .. Treating first los .........................................
    num_raf = los_nraf[0]
    loc_resol = eff_resolution[0]
    first_index = 0
    left_rule_single(num_raf, los_kmin[0], loc_resol,
                     &los_coeffs[0][first_index])
    # ... and the rest of los .......................................
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(1,nlos):
            num_raf = los_nraf[ii]
            loc_resol = eff_resolution[ii]
            first_index = los_ind[ii-1]
            left_rule_single(num_raf, los_kmin[ii], loc_resol,
                             &los_coeffs[0][first_index])
    return


cdef inline void simps_left_rule_rel_var(int nlos, double* resolutions,
                                         double* los_kmin,
                                         double* los_kmax,
                                         double* eff_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil:
    # Simpson left quadrature rule with variable relative resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    simps_left_rule_rel_var_s1(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    left_rule_rel_var_s2(nlos, resolutions,
                         los_kmin, los_kmax,
                         eff_resolution,
                         los_coeffs,
                         los_ind,
                         los_nraf,
                         num_threads)
    free(los_nraf)
    return


cdef inline void simps_left_rule_abs_var_s1(int nlos, double* resolutions,
                                         double* los_kmin,
                                         double* los_kmax,
                                         double* eff_resolution,
                                         long* los_ind, long* los_nraf,
                                         int num_threads) nogil:
    # Simpson left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(c_ceil(seg_length / resolutions[0]))
    if num_raf%2==1:
        num_raf += 1
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(c_ceil(seg_length / resolutions[ii]))
        if num_raf%2==1:
            num_raf += 1
        loc_resol = seg_length / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf +  1 + first_index
    return


cdef inline void simps_left_rule_abs_var(int nlos, double* resolutions,
                                         double* los_kmin,
                                         double* los_kmax,
                                         double* eff_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil:
    # Simpson left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    simps_left_rule_abs_var_s1(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    left_rule_rel_var_s2(nlos, resolutions,
                         los_kmin, los_kmax,
                         eff_resolution,
                         los_coeffs,
                         los_ind,
                         los_nraf,
                         num_threads)
    free(los_nraf)
    return


cdef inline void romb_left_rule_rel_var_s1(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        long* los_ind, long* los_nraf,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    num_raf = <int>(c_ceil(1. / resolutions[0]))
    num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        num_raf = <int>(c_ceil(1. / resolutions[ii]))
        num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
        loc_resol = (los_kmax[ii] - los_kmin[ii]) / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf +  1 + first_index
    return


cdef inline void romb_left_rule_rel_var(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with relative variable resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    romb_left_rule_rel_var_s1(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    left_rule_rel_var_s2(nlos, resolutions,
                         los_kmin, los_kmax,
                         eff_resolution,
                         los_coeffs,
                         los_ind,
                         los_nraf,
                         num_threads)
    free(los_nraf)
    return


cdef inline void romb_left_rule_abs_var_s1(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        long* los_ind, long* los_nraf,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef Py_ssize_t ii
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(c_ceil(seg_length / resolutions[0]))
    num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(c_ceil(seg_length / resolutions[ii]))
        num_raf = <int>(2**(<int>(c_ceil(c_log2(num_raf)))))
        loc_resol = seg_length / num_raf
        eff_resolution[ii] = loc_resol
        los_nraf[ii] = num_raf
        first_index = los_ind[ii-1]
        los_ind[ii] = num_raf +  1 + first_index
    return


cdef inline void romb_left_rule_abs_var(int nlos, double* resolutions,
                                        double* los_kmin,
                                        double* los_kmax,
                                        double* eff_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil:
    # Romboid left quadrature rule with absolute variable resolution step
    # for SEVERAL LOS
    cdef long* los_nraf
    # ...
    los_nraf = <long*> malloc(nlos * sizeof(long))
    romb_left_rule_abs_var_s1(nlos, resolutions,
                           los_kmin, los_kmax,
                           eff_resolution,
                           los_ind,
                           los_nraf,
                           num_threads)
    los_coeffs[0] = <double*>malloc(los_ind[nlos-1]*sizeof(double))
    left_rule_rel_var_s2(nlos, resolutions,
                         los_kmin, los_kmax,
                         eff_resolution,
                         los_coeffs,
                         los_ind,
                         los_nraf,
                         num_threads)
    free(los_nraf)
    return


# -- Get number of integration mode --------------------------------------------
cpdef inline int get_nb_imode(str imode):
    # gil required...........
    if imode == 'sum':
        return 0
    if imode == 'simps':
        return 1
    if imode == 'romb':
        return 2
    return -1


cpdef inline int get_nb_dmode(str dmode):
    # gil required...........
    if dmode == 'rel':
        return 1
    if dmode == 'abs':
        return 0
    return -1


# ==============================================================================
# == LOS sampling Algorithm for a SINGLE LOS
# ==============================================================================
cdef inline int los_get_sample_single(double los_kmin, double los_kmax,
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
      - 0 : 'sum' quadrature, using the n segment centers
      - 1 : 'simps' return n+1 egdes, n even (for scipy.integrate.simps)
      - 2 : 'romb' return n+1 edges, n+1 = 2**k+1 (for scipy.integrate.romb)
    """
    cdef int nraf
    cdef long[1] ind_cum
    cdef double invnraf
    cdef double invresol
    cdef double seg_length
    # ...
    if n_dmode == 1:
        # discretization step is relative
        nraf = <int> c_ceil(1. / resol)
        if n_imode==0:
            # 'sum' quad
            coeffs[0] = <double*>malloc(nraf*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)/resol
            middle_rule_single(nraf, los_kmin, eff_res[0],
                               &coeffs[0][0])
            return nraf
        elif n_imode==1:
            # 'simps' quad
            if nraf%2==1:
                nraf += 1
            invnraf = 1./nraf
            coeffs[0] = <double*>malloc((nraf + 1)*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)*invnraf
            left_rule_single(nraf, los_kmin,
                                 eff_res[0], &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            nraf = <int>(2**(<int>(c_ceil(c_log2(nraf)))))
            invnraf = 1./nraf
            coeffs[0] = <double*>malloc((nraf + 1)*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)*invnraf
            left_rule_single(nraf,  los_kmin,
                                 eff_res[0], &coeffs[0][0])
            return nraf + 1
    else:
        # discretization step is absolute, n_dmode==0
        if n_imode==0:
            # 'sum' quad
            invresol = 1./resol
            middle_rule_abs_s1_single(invresol, los_kmin, los_kmax,
                                         &eff_res[0], &ind_cum[0])
            coeffs[0] = <double*>malloc((ind_cum[0])*sizeof(double))
            middle_rule_single(ind_cum[0], los_kmin, eff_res[0],
                               &coeffs[0][0])
            return ind_cum[0]
        elif n_imode==1:
            # 'simps' quad
            seg_length = los_kmax - los_kmin
            nraf = <int>(c_ceil(seg_length / resol))
            if nraf%2==1:
                nraf += 1
            eff_res[0] = seg_length / nraf
            coeffs[0] = <double*>malloc((nraf+1)*sizeof(double))
            left_rule_single(nraf, los_kmin, eff_res[0],
                                   &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            seg_length = los_kmax - los_kmin
            nraf = <int>(c_ceil(seg_length / resol))
            nraf = <int>(2**(<int>(c_ceil(c_log2(nraf)))))
            eff_res[0] = seg_length / nraf
            coeffs[0] = <double*>malloc((nraf+1)*sizeof(double))
            left_rule_single(nraf, los_kmin, eff_res[0],
                                      &coeffs[0][0])
            return nraf + 1
    return -1


# ==============================================================================
# == Utility functions for signal computation (LOS_calc_signal)
# ==============================================================================
# -- anisotropic case ----------------------------------------------------
cdef inline call_get_sample_single_ani(double los_kmin, double los_kmax,
                                       double resol,
                                       int n_dmode, int n_imode,
                                       double[1] eff_res,
                                       long[1] nb_rows,
                                       double[:, ::1] ray_orig,
                                       double[:, ::1] ray_vdir):
    # This function doesn't compute anything new.
    # It's a utility function for LOS_calc_signal to avoid reptitions
    # It samples a LOS and recreates the points on that LOS
    # plus this is for the anisotropic version so it also compute usbis
    cdef int sz_coeff
    cdef double** los_coeffs = NULL
    cdef cnp.ndarray[double,ndim=1,mode='c'] ksbis
    cdef cnp.ndarray[double,ndim=2,mode='c'] usbis
    cdef cnp.ndarray[double,ndim=2,mode='c'] pts

    # Initialization utility array
    los_coeffs = <double**>malloc(sizeof(double*))
    los_coeffs[0] = NULL
    # Sampling
    sz_coeff = los_get_sample_single(los_kmin, los_kmax,
                                     resol,
                                     n_dmode, n_imode,
                                     &eff_res[0],
                                     &los_coeffs[0])
    nb_rows[0] = sz_coeff
    # computing points
    usbis = np.repeat(ray_vdir, sz_coeff, axis=1)
    ksbis = np.asarray(<double[:sz_coeff]>los_coeffs[0])
    pts = ray_orig + ksbis[None, :] * usbis
    # freeing memory used
    if los_coeffs != NULL:
        if los_coeffs[0] != NULL:
            free(los_coeffs[0])
        free(los_coeffs)
    return pts, usbis


# -- not anisotropic ------------------------------------------------------
cdef inline cnp.ndarray[double,ndim=2,mode='c'] call_get_sample_single(
    double los_kmin,
    double los_kmax,
    double resol,
    int n_dmode, int n_imode,
    double[1] eff_res,
    long[1] nb_rows,
    double[:, ::1] ray_orig,
    double[:, ::1] ray_vdir):
    # This function doesn't compute anything new.
    # It's a utility function for LOS_calc_signal to avoid reptitions
    # It samples a LOS and recreates the points on that LOS
    # plus this is for the anisotropic version so it also compute usbis
    cdef int sz_coeff
    cdef double** los_coeffs = NULL
    cdef cnp.ndarray[double,ndim=2,mode='c'] pts
    # Initialization utility array
    los_coeffs = <double**>malloc(sizeof(double*))
    los_coeffs[0] = NULL
    # Sampling
    sz_coeff = los_get_sample_single(los_kmin, los_kmax,
                                     resol,
                                     n_dmode, n_imode,
                                     &eff_res[0],
                                     &los_coeffs[0])
    nb_rows[0] = sz_coeff
    # computing points
    pts = ray_orig \
          + np.asarray(<double[:sz_coeff]>los_coeffs[0]) \
          * np.repeat(ray_vdir, sz_coeff, axis=1)
    if los_coeffs != NULL:
        if los_coeffs[0] != NULL:
            free(los_coeffs[0])
        free(los_coeffs)
    return pts


# -- LOS get sample utility -----------------------------------------------
cdef inline int los_get_sample_core_const_res(int nlos,
                                              double* los_lim_min,
                                              double* los_lim_max,
                                              int n_dmode, int n_imode,
                                              double val_resol,
                                              double** coeff_ptr,
                                              double* dl_r,
                                              long* los_ind,
                                              int num_threads) nogil:
    # ...
    cdef int num_cells
    cdef int ntmp
    if n_dmode==1: # relative
        #         return coeff_arr, dLr, los_ind[:nlos-1]
        num_cells = <int> c_ceil(1. / val_resol)
        if n_imode==0: # sum
            coeff_ptr[0] = <double*>malloc(sizeof(double)*num_cells*nlos)
            middle_rule_rel(nlos, num_cells, los_lim_min, los_lim_max,
                            &dl_r[0], coeff_ptr[0], los_ind,
                            num_threads=num_threads)
            return num_cells*nlos
        elif n_imode==1: #simps
            num_cells = num_cells if num_cells%2==0 else num_cells+1
            coeff_ptr[0] = <double*>malloc(sizeof(double)*(num_cells+1)*nlos)
            left_rule_rel(nlos, num_cells,
                          los_lim_min, los_lim_max, &dl_r[0],
                          coeff_ptr[0], los_ind,
                          num_threads=num_threads)
            return (num_cells+1)*nlos
        elif n_imode==2: #romb
            num_cells = <int>(2**(<int>c_ceil(c_log2(num_cells))))
            coeff_ptr[0] = <double*>malloc(sizeof(double)*(num_cells+1)*nlos)
            left_rule_rel(nlos, num_cells,
                          los_lim_min, los_lim_max,
                          &dl_r[0], coeff_ptr[0], los_ind,
                          num_threads=num_threads)
            return (num_cells+1)*nlos
    else: # absolute
        if n_imode==0: #sum
            middle_rule_abs_s1(nlos, val_resol, los_lim_min, los_lim_max,
                               &dl_r[0], los_ind,
                               num_threads=num_threads)
            ntmp = _bgt.sum_naive_int(los_ind, nlos)
            coeff_ptr[0] = <double*>malloc(sizeof(double)*ntmp)
            middle_rule_abs_s2(nlos, los_lim_min, &dl_r[0],
                               los_ind, coeff_ptr[0],
                               num_threads=num_threads)
            return ntmp
        elif n_imode==1:# simps
            simps_left_rule_abs(nlos, val_resol,
                                los_lim_min, los_lim_max,
                                &dl_r[0], coeff_ptr, los_ind,
                                num_threads=num_threads)
            return los_ind[nlos-1]
        else:# romb
            romb_left_rule_abs(nlos, val_resol,
                               los_lim_min, los_lim_max,
                               &dl_r[0], coeff_ptr, los_ind,
                               num_threads=num_threads)
            return los_ind[nlos-1]
    return -1


cdef inline void los_get_sample_core_var_res(int nlos,
                                            double* los_lim_min,
                                            double* los_lim_max,
                                            int n_dmode, int n_imode,
                                            double* resol,
                                            double** coeff_ptr,
                                            double* eff_res,
                                            long* los_ind,
                                            int num_threads) nogil:
    if n_dmode==0: #absolute
        if n_imode==0: # sum
            middle_rule_abs_var(nlos,
                                los_lim_min, los_lim_max,
                                resol, &eff_res[0],
                                coeff_ptr, los_ind,
                                num_threads=num_threads)
        elif n_imode==1:# simps
            simps_left_rule_abs_var(nlos, resol,
                                    los_lim_min, los_lim_max,
                                    &eff_res[0], coeff_ptr, los_ind,
                                    num_threads=num_threads)
        else: # romb
            romb_left_rule_abs_var(nlos, resol,
                                   los_lim_min, los_lim_max,
                                   &eff_res[0], coeff_ptr, los_ind,
                                   num_threads=num_threads)
    else: # relative
        if n_imode==0: # sum
            middle_rule_rel_var(nlos, resol,
                                los_lim_min, los_lim_max,
                                &eff_res[0], coeff_ptr, los_ind,
                                num_threads=num_threads)
        elif n_imode==1: # simps
            simps_left_rule_rel_var(nlos, resol,
                                    los_lim_min, los_lim_max,
                                    &eff_res[0], coeff_ptr, los_ind,
                                    num_threads=num_threads)
        else: # romb
            romb_left_rule_rel_var(nlos, resol,
                                   los_lim_min, los_lim_max,
                                   &eff_res[0], coeff_ptr, los_ind,
                                   num_threads=num_threads)
    return


# -- utility for calc signal ---------------------------------------------------
cdef inline void los_get_sample_pts(int nlos,
                                    double* ptx,
                                    double* pty,
                                    double* ptz,
                                    double* usx,
                                    double* usy,
                                    double* usz,
                                    double[:, ::1] ray_orig,
                                    double[:, ::1] ray_vdir,
                                    double* coeff_ptr,
                                    long* los_ind,
                                    int num_threads) nogil:
    cdef double loc_ox, loc_oy, loc_oz
    cdef double loc_vx, loc_vy, loc_vz
    cdef int ii, jj
    # Initialization
    loc_ox = ray_orig[0,0]
    loc_oy = ray_orig[1,0]
    loc_oz = ray_orig[2,0]
    loc_vx = ray_vdir[0,0]
    loc_vy = ray_vdir[1,0]
    loc_vz = ray_vdir[2,0]
    for ii in range(los_ind[0]):
        ptx[ii] = loc_ox + coeff_ptr[ii] * loc_vx
        pty[ii] = loc_oy + coeff_ptr[ii] * loc_vy
        ptz[ii] = loc_oz + coeff_ptr[ii] * loc_vz
        usx[ii] = loc_vx
        usy[ii] = loc_vy
        usz[ii] = loc_vz
    # Other lines of sights:
    for jj in range(1, nlos):
        loc_ox = ray_orig[0,jj]
        loc_oy = ray_orig[1,jj]
        loc_oz = ray_orig[2,jj]
        loc_vx = ray_vdir[0,jj]
        loc_vy = ray_vdir[1,jj]
        loc_vz = ray_vdir[2,jj]
        for ii in range(los_ind[jj-1], los_ind[jj]):
            ptx[ii] = loc_ox + coeff_ptr[ii] * loc_vx
            pty[ii] = loc_oy + coeff_ptr[ii] * loc_vy
            ptz[ii] = loc_oz + coeff_ptr[ii] * loc_vz
            usx[ii] = loc_vx
            usy[ii] = loc_vy
            usz[ii] = loc_vz
    return

# -- utility for vmesh sub from D ----------------------------------------------
cdef inline int  vmesh_disc_phi(int sz_r, int sz_z,
                                long* ncells_rphi,
                                double phistep,
                                int ncells_rphi0,
                                double* disc_r,
                                double* disc_r0,
                                double* step_rphi,
                                double[::1] reso_phi_mv,
                                long* tot_nc_plane,
                                int ind_loc_r0,
                                int ncells_r0,
                                int ncells_z,
                                int* max_sz_phi,
                                double min_phi,
                                double max_phi,
                                long* sz_phi,
                                long[:, ::1] indi_mv,
                                double margin,
                                int num_threads) nogil:
    cdef int ii, jj
    cdef int npts_disc
    cdef int loc_nc_rphi
    cdef double min_phi_pi
    cdef double max_phi_pi
    cdef double margin_step
    cdef double abs0, abs1
    cdef int nphi0, nphi1
    # .. Initialization Variables ..............................................
    npts_disc = 0
    twopi_over_dphi = _TWOPI / phistep
    min_phi_pi = min_phi + c_pi
    max_phi_pi = max_phi + c_pi
    abs0 = c_abs(min_phi_pi)
    abs1 = c_abs(max_phi_pi)
    #
    # .. Discretizing Phi (with respect to the corresponding radius R) .........
    if min_phi < max_phi:
        for ii in range(1, sz_r):
            # Get the actual RPhi resolution and Phi mesh elements
            # (depends on R!)
            ncells_rphi[ii] = <int>c_ceil(twopi_over_dphi * disc_r[ii])
            loc_nc_rphi = ncells_rphi[ii]
            step_rphi[ii] = _TWOPI / ncells_rphi[ii]
            reso_phi_mv[ii] = step_rphi[ii] * disc_r[ii]
            tot_nc_plane[ii] = 0 # initialization
            # Get index and cumulated indices from background
            for jj in range(ind_loc_r0, ncells_r0):
                if disc_r0[jj]==disc_r[ii]:
                    ind_loc_r0 = jj
                    break
                else:
                    ncells_rphi0 += <long>c_ceil(twopi_over_dphi * disc_r0[jj])
                    tot_nc_plane[ii] = ncells_rphi0 * ncells_z

            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
            margin_step = margin * step_rphi[ii]
            if abs0 - step_rphi[ii]*c_floor(abs0 / step_rphi[ii]) < margin_step:
                nphi0 = int(c_round(min_phi_pi / step_rphi[ii]))
            else:
                nphi0 = int(c_floor(min_phi_pi / step_rphi[ii]))
            if abs1-step_rphi[ii]*c_floor(abs1 / step_rphi[ii]) < margin_step:
                nphi1 = int(c_round(max_phi_pi / step_rphi[ii]) - 1)
            else:
                nphi1 = int(c_floor(max_phi_pi / step_rphi[ii]))
            sz_phi[ii] = nphi1 + 1 - nphi0
            if max_sz_phi[0] < sz_phi[ii]:
                max_sz_phi[0] = sz_phi[ii]
            with nogil, parallel(num_threads=num_threads):
                for jj in prange(sz_phi[ii]):
                    indi_mv[ii,jj] = nphi0 + jj
            npts_disc += sz_z * sz_phi[ii]
    else:
        for ii in range(1, sz_r):
            # Get the actual RPhi resolution and Phi mesh elements
            # (depends on R!)
            ncells_rphi[ii] = <int>c_ceil(twopi_over_dphi * disc_r[ii])
            loc_nc_rphi = ncells_rphi[ii]
            step_rphi[ii] = _TWOPI / ncells_rphi[ii]
            reso_phi_mv[ii] = step_rphi[ii] * disc_r[ii]
            tot_nc_plane[ii] = 0 # initialization
            # Get index and cumulated indices from background
            for jj in range(ind_loc_r0, ncells_r0):
                if disc_r0[jj]==disc_r[ii]:
                    ind_loc_r0 = jj
                    break
                else:
                    ncells_rphi0 += <long>c_ceil(twopi_over_dphi * disc_r0[jj])
                    tot_nc_plane[ii] = ncells_rphi0 * ncells_z
            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
            margin_step = margin*step_rphi[ii]
            if abs0 - step_rphi[ii]*c_floor(abs0 / step_rphi[ii]) < margin_step:
                nphi0 = int(c_round(min_phi_pi / step_rphi[ii]))
            else:
                nphi0 = int(c_floor(min_phi_pi / step_rphi[ii]))
            if abs1-step_rphi[ii]*c_floor(abs1 / step_rphi[ii]) < margin_step:
                nphi1 = int(c_round(max_phi_pi / step_rphi[ii]) - 1)
            else:
                nphi1 = int(c_floor(max_phi_pi / step_rphi[ii]))
            sz_phi[ii] = nphi1+1+loc_nc_rphi-nphi0
            if max_sz_phi[0] < sz_phi[ii]:
                max_sz_phi[0] = sz_phi[ii]
            with nogil, parallel(num_threads=num_threads):
                for jj in prange(loc_nc_rphi - nphi0):
                    indi_mv[ii, jj] = nphi0 + jj
                for jj in prange(loc_nc_rphi - nphi0, sz_phi[ii]):
                    indi_mv[ii, jj] = jj - (loc_nc_rphi - nphi0)
            npts_disc += sz_z * sz_phi[ii]

    return npts_disc


cdef inline int vmesh_get_index_arrays(long[:, :, ::1] ind_rzphi2dic,
                                       long[:, ::1] is_in_vignette,
                                       int sz_r,
                                       int sz_z,
                                       long* sz_phi) nogil:
    cdef int rr, zz, pp
    cdef int npts_disc = 0
    cdef int loc_sz_phi

    for rr in range(sz_r):
        loc_sz_phi = sz_phi[rr]
        for zz in range(sz_z):
            if is_in_vignette[rr, zz]:
                for pp in range(loc_sz_phi):
                    ind_rzphi2dic[rr, zz, pp] = npts_disc
                    npts_disc += 1
    return npts_disc


cdef inline void vmesh_assemble_arrays_cart(int rr,
                                            int sz_z,
                                            long* lindex_z,
                                            long[::1] is_in_vignette,
                                            long* ncells_rphi,
                                            long* tot_nc_plane,
                                            double reso_r_z,
                                            double* step_rphi,
                                            double* disc_r,
                                            double* disc_z,
                                            long[:, :, ::1] ind_rzphi2dic,
                                            long* sz_phi,
                                            long[::1] iii,
                                            double[::1] dv_mv,
                                            double[::1] reso_phi_mv,
                                            double[:, ::1] pts_mv,
                                            long[::1] ind_mv) nogil:
    cdef int zz
    cdef int jj
    cdef long zrphi
    cdef long indiijj
    cdef double phi
    cdef long npts_disc
    # ..
    for zz in range(sz_z):
        zrphi = lindex_z[zz] * ncells_rphi[rr]
        if is_in_vignette[zz]:
            for jj in range(sz_phi[rr]):
                npts_disc = ind_rzphi2dic[rr, zz, jj]
                indiijj = iii[jj]
                phi = -c_pi + (0.5 + indiijj) * step_rphi[rr]
                pts_mv[0, npts_disc] = disc_r[rr] * c_cos(phi)
                pts_mv[1, npts_disc] = disc_r[rr] * c_sin(phi)
                pts_mv[2, npts_disc] = disc_z[zz]
                ind_mv[npts_disc] = tot_nc_plane[rr] + zrphi + indiijj
                dv_mv[npts_disc] = reso_r_z*reso_phi_mv[rr]
    return


cdef inline void vmesh_assemble_arrays_polr(int rr,
                                            int sz_z,
                                            long* lindex_z,
                                            long[::1] is_in_vignette,
                                            long* ncells_rphi,
                                            long* tot_nc_plane,
                                            double reso_r_z,
                                            double* step_rphi,
                                            double* disc_r,
                                            double* disc_z,
                                            long[:, :, ::1] ind_rzphi2dic,
                                            long* sz_phi,
                                            long[::1] iii,
                                            double[::1] dv_mv,
                                            double[::1] reso_phi_mv,
                                            double[:, ::1] pts_mv,
                                            long[::1] ind_mv) nogil:
    cdef int zz
    cdef int jj
    cdef long npts_disc
    cdef long zrphi
    cdef long indiijj
    # ..
    for zz in range(sz_z):
        zrphi = lindex_z[zz] * ncells_rphi[rr]
        if is_in_vignette[zz]:
            for jj in range(sz_phi[rr]):
                npts_disc = ind_rzphi2dic[rr, zz, jj]
                indiijj = iii[jj]
                pts_mv[0, npts_disc] = disc_r[rr]
                pts_mv[1, npts_disc] = disc_z[zz]
                pts_mv[2, npts_disc] = -c_pi + (0.5 + indiijj) * step_rphi[rr]
                ind_mv[npts_disc] = tot_nc_plane[rr] + zrphi + indiijj
                dv_mv[npts_disc] = reso_r_z * reso_phi_mv[rr]
    return



cdef inline void vmesh_assemble_arrays(long[::1] first_ind_mv,
                                       long[:, ::1] indi_mv,
                                       long[:, ::1] is_in_vignette,
                                       bint is_cart,
                                       int sz_r,
                                       int sz_z,
                                       long* lindex_z,
                                       long* ncells_rphi,
                                       long* tot_nc_plane,
                                       double reso_r_z,
                                       double* step_rphi,
                                       double* disc_r,
                                       double* disc_z,
                                       long[:, :, ::1] ind_rzphi2dic,
                                       long* sz_phi,
                                       double[::1] dv_mv,
                                       double[::1] reso_phi_mv,
                                       double[:, ::1] pts_mv,
                                       long[::1] ind_mv,
                                       int num_threads) nogil:
    cdef int rr
    # ...
    with nogil, parallel(num_threads=num_threads):
        if is_cart:
            for rr in prange(sz_r):
                # To make sure the indices are in increasing order
                vmesh_assemble_arrays_cart(rr, sz_z, lindex_z,
                                           is_in_vignette[rr],
                                           ncells_rphi, tot_nc_plane,
                                           reso_r_z, step_rphi,
                                           disc_r, disc_z, ind_rzphi2dic,
                                           sz_phi,
                                           indi_mv[rr,first_ind_mv[rr]:],
                                           dv_mv, reso_phi_mv, pts_mv, ind_mv)
        else:
            for rr in prange(sz_r):
                vmesh_assemble_arrays_polr(rr, sz_z, lindex_z,
                                           is_in_vignette[rr],
                                           ncells_rphi, tot_nc_plane,
                                           reso_r_z, step_rphi,
                                           disc_r, disc_z, ind_rzphi2dic,
                                           sz_phi,
                                           indi_mv[rr,first_ind_mv[rr]:],
                                           dv_mv, reso_phi_mv, pts_mv, ind_mv)
    return


# -- utility for vmesh from indices --------------------------------------------
cdef inline void vmesh_ind_init_tabs(int* ncells_rphi,
                                     double* disc_r,
                                     int sz_r, int sz_z,
                                     double twopi_over_dphi,
                                     double[::1] d_r_phir_ref,
                                     long* tot_nc_plane,
                                     double** phi_tab,
                                     int num_threads) nogil:
    cdef int rr
    cdef int jj
    cdef int radius_ratio
    cdef int loc_nc_rphi
    cdef double* step_rphi = NULL
    # .. Discretizing Phi (with respect to the corresponding radius R) .........
    step_rphi    = <double*>malloc(sz_r * sizeof(double))
    radius_ratio = <int>c_ceil(disc_r[sz_r - 1] / disc_r[0])
    # we do first the step 0 to avoid an if in a for loop:
    loc_nc_rphi = <int>(c_ceil(disc_r[0] * twopi_over_dphi))
    ncells_rphi[0] = loc_nc_rphi
    step_rphi[0] = _TWOPI / loc_nc_rphi
    d_r_phir_ref[0] = disc_r[0] * step_rphi[0]
    tot_nc_plane[0] = 0
    phi_tab[0] = <double*>malloc(sz_r * (loc_nc_rphi * radius_ratio + 1)
                             * sizeof(double))
    with nogil, parallel(num_threads=num_threads):
        for jj in prange(loc_nc_rphi):
            phi_tab[0][jj * sz_r] = -c_pi + (0.5 + jj) * step_rphi[0]
    # now we do the rest of the loop
    for rr in range(1, sz_r):
        loc_nc_rphi = <int>(c_ceil(disc_r[rr] * twopi_over_dphi))
        ncells_rphi[rr] = loc_nc_rphi
        step_rphi[rr] = _TWOPI / loc_nc_rphi
        d_r_phir_ref[rr] = disc_r[rr] * step_rphi[rr]
        tot_nc_plane[rr] = tot_nc_plane[rr-1] + ncells_rphi[rr-1] * sz_z
        with nogil, parallel(num_threads=num_threads):
            for jj in range(loc_nc_rphi):
                phi_tab[0][rr + sz_r * jj] = -c_pi + (0.5 + jj) * step_rphi[rr]

    tot_nc_plane[sz_r] = tot_nc_plane[sz_r-1] + ncells_rphi[sz_r-1] * sz_z
    free(step_rphi)
    return


cdef inline void vmesh_ind_cart_loop(int np,
                                     int sz_r,
                                     long[::1] ind,
                                     long* tot_nc_plane,
                                     int* ncells_rphi,
                                     double* phi_tab,
                                     double* disc_r,
                                     double* disc_z,
                                     double[:, ::1] pts,
                                     double[::1] res3d,
                                     double reso_r_z,
                                     double[::1] d_r_phir_ref,
                                     int[::1] ru,
                                     double[::1] d_r_phir,
                                     int num_threads) nogil:
    cdef int ii
    cdef int jj
    cdef int ii_r, ii_z, iiphi
    cdef double phi
    # we compute the points coordinates from the indices values
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(np):
            jj = 0
            for jj in range(sz_r+1):
                if ind[ii]-tot_nc_plane[jj]<0:
                    break
            ii_r = jj-1
            ii_z =  (ind[ii] - tot_nc_plane[ii_r]) // ncells_rphi[ii_r]
            iiphi = ind[ii] - tot_nc_plane[ii_r] - ii_z * ncells_rphi[ii_r]
            phi = phi_tab[ii_r + sz_r * iiphi]
            pts[0, ii] = disc_r[ii_r] * c_cos(phi)
            pts[1, ii] = disc_r[ii_r] * c_sin(phi)
            pts[2, ii] = disc_z[ii_z]
            res3d[ii] = reso_r_z * d_r_phir_ref[ii_r]
            if ru[ii_r]==0:
                d_r_phir[ii_r] = d_r_phir_ref[ii_r]
                ru[ii_r] = 1
    return


cdef inline void vmesh_ind_polr_loop(int np,
                                     int sz_r,
                                     long[::1] ind,
                                     long* tot_nc_plane,
                                     int* ncells_rphi,
                                     double* phi_tab,
                                     double* disc_r,
                                     double* disc_z,
                                     double[:, ::1] pts,
                                     double[::1] res3d,
                                     double reso_r_z,
                                     double[::1] d_r_phir_ref,
                                     int[::1] ru,
                                     double[::1] d_r_phir,
                                     int num_threads) nogil:
    cdef int ii
    cdef int jj
    cdef int ii_r, ii_z, iiphi
    # we compute the points coordinates from the indices values
    with nogil, parallel(num_threads=num_threads):
        for ii in prange(np):
            for jj in range(sz_r+1):
                if ind[ii]-tot_nc_plane[jj]<0:
                    break
            ii_r = jj-1
            ii_z =  (ind[ii] - tot_nc_plane[ii_r]) // ncells_rphi[ii_r]
            iiphi = ind[ii] - tot_nc_plane[ii_r] - ii_z * ncells_rphi[ii_r]
            pts[0, ii] = disc_r[ii_r]
            pts[1, ii] = disc_z[ii_z]
            pts[2, ii] = phi_tab[ii_r + sz_r * iiphi]
            res3d[ii] = reso_r_z * d_r_phir_ref[ii_r]
            if ru[ii_r]==0:
                d_r_phir[ii_r] = d_r_phir_ref[ii_r]
                ru[ii_r] = 1
    return


# ==============================================================================
# == Solid Angle Computation
# ==============================================================================
cdef inline int sa_get_index_arrays(long[:, ::1] ind_rz2pol,
                                    long[:, ::1] is_in_vignette,
                                    int sz_r,
                                    int sz_z) nogil:
    cdef int rr, zz
    cdef int npts_pol = 0

    for rr in range(sz_r):
        for zz in range(sz_z):
            if is_in_vignette[rr, zz]:
                ind_rz2pol[rr, zz] = npts_pol
                npts_pol += 1
    return npts_pol


# -- utility for discretizing phi ----------------------------------------------
cdef inline int  sa_disc_phi(int sz_r, int sz_z,
                             long* ncells_rphi,
                             double phistep,
                             double* disc_r,
                             double* disc_r0,
                             double* step_rphi,
                             int ind_loc_r0,
                             int ncells_r0,
                             int ncells_z,
                             int* max_sz_phi,
                             double min_phi,
                             double max_phi,
                             long* sz_phi,
                             long[:, ::1] indi_mv,
                             double margin,
                             int num_threads) nogil:
    cdef int rr, jj
    cdef int npts_disc = 0
    cdef int loc_nc_rphi
    cdef double min_phi_pi
    cdef double max_phi_pi
    cdef double margin_step
    cdef double abs0, abs1
    cdef int nphi0, nphi1
    # .. Initialization Variables ..............................................
    twopi_over_dphi = _TWOPI / phistep
    min_phi_pi = min_phi + c_pi
    max_phi_pi = max_phi + c_pi
    abs0 = c_abs(min_phi_pi)
    abs1 = c_abs(max_phi_pi)
    #
    # .. Discretizing Phi (with respect to the corresponding radius R) .........
    if min_phi < max_phi:
        for rr in range(1, sz_r):
            # Get the actual RPhi resolution and Phi mesh elements
            # (depends on R!)
            ncells_rphi[rr] = <int>c_ceil(twopi_over_dphi * disc_r[rr])
            loc_nc_rphi = ncells_rphi[rr]
            step_rphi[rr] = _TWOPI / ncells_rphi[rr]
            # Get index and cumulated indices from background
            for jj in range(ind_loc_r0, ncells_r0):
                if disc_r0[jj]==disc_r[rr]:
                    ind_loc_r0 = jj
                    break

            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
            margin_step = margin * step_rphi[rr]
            if abs0 - step_rphi[rr]*c_floor(abs0 / step_rphi[rr]) < margin_step:
                nphi0 = int(c_round(min_phi_pi / step_rphi[rr]))
            else:
                nphi0 = int(c_floor(min_phi_pi / step_rphi[rr]))
            if abs1-step_rphi[rr]*c_floor(abs1 / step_rphi[rr]) < margin_step:
                nphi1 = int(c_round(max_phi_pi / step_rphi[rr]) - 1)
            else:
                nphi1 = int(c_floor(max_phi_pi / step_rphi[rr]))
            sz_phi[rr] = nphi1 + 1 - nphi0
            if max_sz_phi[0] < sz_phi[rr]:
                max_sz_phi[0] = sz_phi[rr]
            with nogil, parallel(num_threads=num_threads):
                for jj in prange(sz_phi[rr]):
                    indi_mv[rr, jj] = nphi0 + jj
            npts_disc += sz_z * sz_phi[rr]
    else:
        for rr in range(1, sz_r):
            # Get the actual RPhi resolution and Phi mesh elements
            # (depends on R!)
            ncells_rphi[rr] = <int>c_ceil(twopi_over_dphi * disc_r[rr])
            loc_nc_rphi = ncells_rphi[rr]
            step_rphi[rr] = _TWOPI / ncells_rphi[rr]
            #reso_phi_mv[rr] = step_rphi[rr] * disc_r[rr]
            # Get index and cumulated indices from background
            for jj in range(ind_loc_r0, ncells_r0):
                if disc_r0[jj]==disc_r[rr]:
                    ind_loc_r0 = jj
                    break
            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to
            # be created within those limits
            margin_step = margin*step_rphi[rr]
            if abs0 - step_rphi[rr]*c_floor(abs0 / step_rphi[rr]) < margin_step:
                nphi0 = int(c_round(min_phi_pi / step_rphi[rr]))
            else:
                nphi0 = int(c_floor(min_phi_pi / step_rphi[rr]))
            if abs1-step_rphi[rr]*c_floor(abs1 / step_rphi[rr]) < margin_step:
                nphi1 = int(c_round(max_phi_pi / step_rphi[rr]) - 1)
            else:
                nphi1 = int(c_floor(max_phi_pi / step_rphi[rr]))
            sz_phi[rr] = nphi1 + 1 + loc_nc_rphi - nphi0
            if max_sz_phi[0] < sz_phi[rr]:
                max_sz_phi[0] = sz_phi[rr]
            with nogil, parallel(num_threads=num_threads):
                for jj in prange(loc_nc_rphi - nphi0):
                    indi_mv[rr, jj] = nphi0 + jj
                for jj in prange(loc_nc_rphi - nphi0, sz_phi[rr]):
                    indi_mv[rr, jj] = jj - (loc_nc_rphi - nphi0)
            npts_disc += sz_z * sz_phi[rr]
    return npts_disc


cdef inline void sa_assemble_arrays(int block,
                                    int use_approx,
                                    double[:, ::1] part_coords,
                                    double[::1] part_rad,
                                    long[:, ::1] is_in_vignette,
                                    double[:, ::1] sa_map,
                                    double[:, ::1] ves_poly,
                                    double[:, ::1] ves_norm,
                                    double[::1] ves_lims,
                                    long[::1] lstruct_nlim,
                                    double[::1] lstruct_polyx,
                                    double[::1] lstruct_polyy,
                                    double[::1] lstruct_lims,
                                    double[::1] lstruct_normx,
                                    double[::1] lstruct_normy,
                                    long[::1] lnvert,
                                    int nstruct_tot,
                                    int nstruct_lim,
                                    double rmin,
                                    double eps_uz, double eps_a,
                                    double eps_vz, double eps_b,
                                    double eps_plane,
                                    bint forbid,
                                    long[::1] first_ind,
                                    long[:, ::1] indi_mv,
                                    int sz_p,
                                    int sz_r, int sz_z,
                                    long* ncells_rphi,
                                    double reso_r_z,
                                    double* disc_r,
                                    double* step_rphi,
                                    double* disc_z,
                                    long[:, ::1] ind_rz2pol,
                                    long* sz_phi,
                                    double[::1] reso_rdrdz,
                                    double[:, ::1] pts_mv,
                                    long[::1] ind_mv,
                                    int num_threads):
    cdef int rr
    cdef int zz
    cdef int ind_pol
    cdef int sz_ves_lims
    cdef int npts_poly
    cdef int sz_pol
    cdef cnp.ndarray[double,ndim=3] ray_orig
    cdef cnp.ndarray[double,ndim=3] ray_vdir
    cdef cnp.ndarray[double,ndim=2] vperp_out
    cdef cnp.ndarray[double,ndim=2] coeff_inter_in
    cdef cnp.ndarray[double,ndim=2] coeff_inter_out
    cdef cnp.ndarray[int,ndim=2] ind_inter_out
    cdef array ind_pol2r = clone(array('i'), ind_mv.size, True)
    cdef array ind_pol2z = clone(array('i'), ind_mv.size, True)

    sz_pol = ind_mv.size
    ind_pol = 0
    for rr in range(sz_r):
        for zz in range(sz_z):
            if is_in_vignette[rr, zz]:
                ind_pol2r[ind_pol] = rr
                ind_pol2z[ind_pol] = zz
                ind_mv[ind_pol] = rr * sz_z + zz
                loc_r = disc_r[rr]
                loc_z = disc_z[zz]
                reso_rdrdz[ind_pol] = loc_r * reso_r_z
                pts_mv[0, ind_pol] = loc_r
                pts_mv[1, ind_pol] = loc_z
                ind_pol += 1

    if block:
        # declared here so that cython can run without gil
        if ves_lims is not None:
            sz_ves_lims = np.size(ves_lims)
        else:
            sz_ves_lims = 0
        npts_poly = ves_norm.shape[1]
        ray_orig = np.zeros((num_threads, 3, sz_p))
        ray_vdir = np.zeros((num_threads, 3, sz_p))
        vperp_out = np.zeros((num_threads, 3 * sz_p))
        coeff_inter_in  = np.zeros((num_threads, sz_p))
        coeff_inter_out = np.zeros((num_threads, sz_p))
        ind_inter_out = np.zeros((num_threads, sz_p * 3), dtype=np.int32)

        if use_approx: # if block and use_approx

            assemble_block_approx(part_coords, part_rad,
                                  sa_map,
                                  ves_poly, ves_norm,
                                  ves_lims,
                                  lstruct_nlim, lstruct_polyx, lstruct_polyy,
                                  lstruct_lims, lstruct_normx, lstruct_normy,
                                  lnvert, vperp_out,
                                  coeff_inter_in, coeff_inter_out,
                                  ind_inter_out, sz_ves_lims,
                                  ray_orig, ray_vdir, npts_poly,
                                  nstruct_tot, nstruct_lim,
                                  rmin,
                                  eps_uz, eps_a,
                                  eps_vz, eps_b, eps_plane,
                                  forbid,
                                  first_ind, indi_mv,
                                  sz_p, sz_pol,
                                  ncells_rphi,
                                  disc_r, step_rphi,
                                  disc_z, ind_pol2r, ind_pol2z, sz_phi,
                                  num_threads)

        else: # if block and (not use_approx)

            assemble_block_exact(part_coords, part_rad,
                                 sa_map,
                                 ves_poly, ves_norm,
                                 ves_lims,
                                 lstruct_nlim,
                                 lstruct_polyx,
                                 lstruct_polyy,
                                 lstruct_lims,
                                 lstruct_normx,
                                 lstruct_normy,
                                 lnvert, vperp_out,
                                 coeff_inter_in, coeff_inter_out,
                                 ind_inter_out, sz_ves_lims,
                                 ray_orig, ray_vdir, npts_poly,
                                 nstruct_tot, nstruct_lim,
                                 rmin,
                                 eps_uz, eps_a,
                                 eps_vz, eps_b, eps_plane,
                                 forbid,
                                 first_ind, indi_mv,
                                 sz_p, sz_pol,
                                 ncells_rphi,
                                 disc_r, step_rphi,
                                 disc_z, ind_pol2r, ind_pol2z, sz_phi,
                                 num_threads)

    else: # if not block

        if use_approx: # if (not block) and use_approx

            assemble_unblock_approx(part_coords, part_rad,
                                    sa_map,
                                    first_ind, indi_mv,
                                    sz_p, sz_pol,
                                    ncells_rphi,
                                    disc_r, step_rphi,
                                    disc_z, ind_pol2r, ind_pol2z,
                                    sz_phi,
                                    num_threads)

        else: # if (not block) and (not use_approx)

            assemble_unblock_exact(part_coords, part_rad,
                                   sa_map,
                                   first_ind, indi_mv,
                                   sz_p, sz_pol,
                                   ncells_rphi,
                                   disc_r, step_rphi,
                                   disc_z, ind_pol2r, ind_pol2z,
                                   sz_phi,
                                   num_threads)
    return


cdef inline void assemble_block_approx(double[:, ::1] part_coords,
                                       double[::1] part_rad,
                                       double[:, ::1] sa_map,
                                       double[:, ::1] ves_poly,
                                       double[:, ::1] ves_norm,
                                       double[::1] ves_lims,
                                       long[::1] lstruct_nlim,
                                       double[::1] lstruct_polyx,
                                       double[::1] lstruct_polyy,
                                       double[::1] lstruct_lims,
                                       double[::1] lstruct_normx,
                                       double[::1] lstruct_normy,
                                       long[::1] lnvert,
                                       double[:, ::1] vperp_out,
                                       double[:, ::1] coeff_inter_in,
                                       double[:, ::1] coeff_inter_out,
                                       int[:, ::1] ind_inter_out,
                                       int sz_ves_lims,
                                       double[:, :, ::1] ray_orig,
                                       double[:, :, ::1] ray_vdir,
                                       int npts_poly,
                                       int nstruct_tot,
                                       int nstruct_lim,
                                       double rmin,
                                       double eps_uz, double eps_a,
                                       double eps_vz, double eps_b,
                                       double eps_plane,
                                       bint forbid,
                                       long[::1] first_ind_mv,
                                       long[:, ::1] indi_mv,
                                       int sz_p,
                                       int sz_pol,
                                       long* ncells_rphi,
                                       double* disc_r,
                                       double* step_rphi,
                                       double* disc_z,
                                       int[::1] ind_pol2r,
                                       int[::1] ind_pol2z,
                                       long* sz_phi,
                                       int num_threads) nogil:
    cdef int rr
    cdef int zz
    cdef int jj
    cdef int pp
    cdef int ind_pol
    cdef int loc_first_ind
    cdef int loc_size_phi
    cdef long indiijj
    cdef double vol_pi
    cdef double loc_x
    cdef double loc_y
    cdef double loc_r
    cdef double loc_z
    cdef double loc_phi
    cdef double loc_step_rphi
    cdef long* is_vis
    cdef double* dist = NULL

    cdef long thid

    with nogil, parallel(num_threads=num_threads):
        dist = <double*> malloc(sz_p * sizeof(double))
        is_vis = <long*> malloc(sz_p * sizeof(long))

        thid = threadid()

        for ind_pol in prange(sz_pol, schedule="dynamic"):
            rr = ind_pol2r[ind_pol]
            loc_r = disc_r[rr]
            vol_pi = step_rphi[rr] * loc_r * c_pi
            loc_size_phi = sz_phi[rr]
            loc_step_rphi = step_rphi[rr]
            loc_first_ind = first_ind_mv[rr]
            zz = ind_pol2z[ind_pol]
            loc_z = disc_z[zz]
            for jj in range(loc_size_phi):
                indiijj = indi_mv[rr, loc_first_ind + jj]
                loc_phi = - c_pi + (0.5 + indiijj) * loc_step_rphi
                loc_x = loc_r * c_cos(loc_phi)
                loc_y = loc_r * c_sin(loc_phi)
                # computing distance ....
                _bgt.compute_dist_pt_vec(loc_x, loc_y, loc_z,
                                         sz_p, part_coords,
                                         &dist[0])
                # checking if visible .....
                _rt.is_visible_pt_vec_core(loc_x, loc_y, loc_z,
                                           part_coords,
                                           sz_p,
                                           ves_poly, ves_norm,
                                           &is_vis[0], dist,
                                           ves_lims,
                                           lstruct_nlim,
                                           lstruct_polyx,
                                           lstruct_polyy,
                                           lstruct_lims,
                                           lstruct_normx,
                                           lstruct_normy,
                                           lnvert,
                                           vperp_out[thid],
                                           coeff_inter_in[thid],
                                           coeff_inter_out[thid],
                                           ind_inter_out[thid],
                                           sz_ves_lims,
                                           ray_orig[thid],
                                           ray_vdir[thid],
                                           npts_poly,
                                           nstruct_tot, nstruct_lim,
                                           rmin,
                                           eps_uz, eps_a,
                                           eps_vz, eps_b, eps_plane,
                                           1, # is toroidal
                                           forbid, 1)
                for pp in range(sz_p):
                    if is_vis[pp] and dist[pp] > part_rad[pp]:
                        sa_map[ind_pol,
                               pp] += sa_approx_formula(part_rad[pp],
                                                        dist[pp],
                                                        vol_pi)
        free(dist)
        free(is_vis)

    return


cdef inline void assemble_unblock_approx(double[:, ::1] part_coords,
                                         double[::1] part_rad,
                                         double[:, ::1] sa_map,
                                         long[::1] first_ind_mv,
                                         long[:, ::1] indi_mv,
                                         int sz_p,
                                         int sz_pol,
                                         long* ncells_rphi,
                                         double* disc_r,
                                         double* step_rphi,
                                         double* disc_z,
                                         int[::1] ind_pol2r,
                                         int[::1] ind_pol2z,
                                         long* sz_phi,
                                         int num_threads) nogil:
    cdef int rr
    cdef int zz
    cdef int jj
    cdef int pp
    cdef int ind_pol
    cdef int loc_first_ind
    cdef int loc_size_phi
    cdef long indiijj
    cdef double vol_pi
    cdef double loc_x
    cdef double loc_y
    cdef double loc_r
    cdef double loc_z
    cdef double loc_phi
    cdef double loc_step_rphi
    cdef double* dist = NULL

    with nogil, parallel(num_threads=num_threads):
        dist = <double*> malloc(sz_p * sizeof(double))
        for ind_pol in prange(sz_pol, schedule="guided"):
            rr = ind_pol2r[ind_pol]
            loc_r = disc_r[rr]
            vol_pi = step_rphi[rr] * loc_r * c_pi
            loc_size_phi = sz_phi[rr]
            loc_step_rphi = step_rphi[rr]
            loc_first_ind = first_ind_mv[rr]
            zz = ind_pol2z[ind_pol]
            loc_z = disc_z[zz]
            for jj in range(loc_size_phi):
                indiijj = indi_mv[rr, loc_first_ind + jj]
                loc_phi = - c_pi + (0.5 + indiijj) * loc_step_rphi
                loc_x = loc_r * c_cos(loc_phi)
                loc_y = loc_r * c_sin(loc_phi)
                # computing distance ....
                _bgt.compute_dist_pt_vec(loc_x,
                                         loc_y,
                                         loc_z,
                                         sz_p, part_coords,
                                         &dist[0])
                for pp in range(sz_p):
                    if dist[pp] > part_rad[pp]:
                        sa_map[ind_pol,
                               pp] += sa_approx_formula(part_rad[pp],
                                                        dist[pp],
                                                        vol_pi)
        free(dist)
    return


cdef inline double sa_approx_formula(double radius,
                                     double distance,
                                     double volpi,
                                     int debug=0) nogil:
    """
    Eigth degree approximation of solid angle computation subtended by a
    sphere of radius `radius` at a distance `distance`.

    Parameters
    ----------
    radius: double
        radius of the particle
    distance: double
        distance between particle and viewing point
    volpi: double
        volume unit (eg: dPhi * R * dR * dZ) times pi

    Returns
    --------
        Approximation of solid angle to the 4th order:
        \Omega * dVol = pi (r/d)^2 + pi/4 (r/d)^4  + pi/8 * (r/d)**6
                        + pi 5/64 (r/d) ** 8
    """
    cdef double r_over_d = radius / distance

    # return r_over_d ** 2 * volpi
    return (r_over_d ** 2 + r_over_d ** 4 * 0.25 + r_over_d**6 * 0.125
            + r_over_d ** 8 * 0.078125) * volpi


# -----------------------------------------------------------------------------
#                      Exact formula computation
# -----------------------------------------------------------------------------
cdef inline void assemble_block_exact(double[:, ::1] part_coords,
                                      double[::1] part_rad,
                                      double[:, ::1] sa_map,
                                      double[:, ::1] ves_poly,
                                      double[:, ::1] ves_norm,
                                      double[::1] ves_lims,
                                      long[::1] lstruct_nlim,
                                      double[::1] lstruct_polyx,
                                      double[::1] lstruct_polyy,
                                      double[::1] lstruct_lims,
                                      double[::1] lstruct_normx,
                                      double[::1] lstruct_normy,
                                      long[::1] lnvert,
                                      double[:, ::1] vperp_out,
                                      double[:, ::1] coeff_inter_in,
                                      double[:, ::1] coeff_inter_out,
                                      int[:, ::1] ind_inter_out,
                                      int sz_ves_lims,
                                      double[:, :, ::1] ray_orig,
                                      double[:, :, ::1] ray_vdir,
                                      int npts_poly,
                                      int nstruct_tot,
                                      int nstruct_lim,
                                      double rmin,
                                      double eps_uz, double eps_a,
                                      double eps_vz, double eps_b,
                                      double eps_plane,
                                      bint forbid,
                                      long[::1] first_ind_mv,
                                      long[:, ::1] indi_mv,
                                      int sz_p,
                                      int sz_pol,
                                      long* ncells_rphi,
                                      double* disc_r,
                                      double* step_rphi,
                                      double* disc_z,
                                      int[::1] ind_pol2r,
                                      int[::1] ind_pol2z,
                                      long* sz_phi,
                                      int num_threads) nogil:
    cdef int rr
    cdef int zz
    cdef int jj
    cdef int pp
    cdef int ind_pol
    cdef int loc_first_ind
    cdef int loc_size_phi
    cdef long indiijj
    cdef double vol_pi
    cdef double loc_r
    cdef double loc_z
    cdef double loc_x
    cdef double loc_y
    cdef double loc_phi
    cdef double loc_step_rphi
    cdef long* is_vis
    cdef double* dist = NULL

    cdef long thid

    with nogil, parallel(num_threads=num_threads):
        dist = <double*> malloc(sz_p * sizeof(double))
        is_vis = <long*> malloc(sz_p * sizeof(long))

        thid = threadid()

        for ind_pol in prange(sz_pol, schedule="dynamic"):
            rr = ind_pol2r[ind_pol]
            loc_r = disc_r[rr]
            vol_pi = step_rphi[rr] * loc_r * c_pi
            loc_size_phi = sz_phi[rr]
            loc_step_rphi = step_rphi[rr]
            loc_first_ind = first_ind_mv[rr]
            zz = ind_pol2z[ind_pol]
            loc_z = disc_z[zz]
            for jj in range(loc_size_phi):
                indiijj = indi_mv[rr, loc_first_ind + jj]
                loc_phi = - c_pi + (0.5 + indiijj) * loc_step_rphi
                loc_x = loc_r * c_cos(loc_phi)
                loc_y = loc_r * c_sin(loc_phi)
                # computing distance ....
                _bgt.compute_dist_pt_vec(loc_x,
                                         loc_y,
                                         loc_z,
                                         sz_p, part_coords,
                                         &dist[0])
                # checking if visible .....
                _rt.is_visible_pt_vec_core(loc_x, loc_y, loc_z,
                                           part_coords,
                                           sz_p,
                                           ves_poly, ves_norm,
                                           &is_vis[0], dist,
                                           ves_lims,
                                           lstruct_nlim,
                                           lstruct_polyx,
                                           lstruct_polyy,
                                           lstruct_lims,
                                           lstruct_normx,
                                           lstruct_normy,
                                           lnvert, vperp_out[thid],
                                           coeff_inter_in[thid],
                                           coeff_inter_out[thid],
                                           ind_inter_out[thid],
                                           sz_ves_lims,
                                           ray_orig[thid],
                                           ray_vdir[thid],
                                           npts_poly,
                                           nstruct_tot, nstruct_lim,
                                           rmin,
                                           eps_uz, eps_a,
                                           eps_vz, eps_b, eps_plane,
                                           1, # is toroidal
                                           forbid, 1)
                for pp in range(sz_p):
                    if is_vis[pp] and dist[pp] > part_rad[pp]:
                        sa_map[ind_pol,
                               pp] += sa_exact_formula(part_rad[pp],
                                                       dist[pp],
                                                       vol_pi)
        free(dist)
        free(is_vis)
    return


cdef inline void assemble_unblock_exact(double[:, ::1] part_coords,
                                        double[::1] part_rad,
                                        double[:, ::1] sa_map,
                                        long[::1] first_ind_mv,
                                        long[:, ::1] indi_mv,
                                        int sz_p,
                                        int sz_pol,
                                        long* ncells_rphi,
                                        double* disc_r,
                                        double* step_rphi,
                                        double* disc_z,
                                        int[::1] ind_pol2r,
                                        int[::1] ind_pol2z,
                                        long* sz_phi,
                                        int num_threads) nogil:
    cdef int rr
    cdef int zz
    cdef int jj
    cdef int pp
    cdef int ind_pol
    cdef int loc_first_ind
    cdef int loc_size_phi
    cdef long indiijj
    cdef double vol_pi
    cdef double loc_r
    cdef double loc_x
    cdef double loc_y
    cdef double loc_z
    cdef double loc_phi
    cdef double loc_step_rphi
    cdef double* dist = NULL

    with nogil, parallel(num_threads=num_threads):
        dist = <double*> malloc(sz_p * sizeof(double))
        for ind_pol in prange(sz_pol, schedule="guided"):
            rr = ind_pol2r[ind_pol]
            loc_r = disc_r[rr]
            vol_pi = step_rphi[rr] * loc_r * c_pi
            loc_size_phi = sz_phi[rr]
            loc_step_rphi = step_rphi[rr]
            loc_first_ind = first_ind_mv[rr]
            zz = ind_pol2z[ind_pol]
            loc_z = disc_z[zz]
            for jj in range(loc_size_phi):
                indiijj = indi_mv[rr, loc_first_ind + jj]
                loc_phi = - c_pi + (0.5 + indiijj) * loc_step_rphi
                loc_x = loc_r * c_cos(loc_phi)
                loc_y = loc_r * c_sin(loc_phi)
                # computing distance ....
                _bgt.compute_dist_pt_vec(loc_x,
                                         loc_y,
                                         loc_z,
                                         sz_p, part_coords,
                                         &dist[0])
                for pp in range(sz_p):
                    if dist[pp]  > part_rad[pp]:
                        sa_map[ind_pol,
                               pp] += sa_exact_formula(part_rad[pp],
                                                       dist[pp],
                                                       vol_pi)
        free(dist)
    return


cdef inline double sa_exact_formula(double radius,
                                    double distance,
                                    double volpi) nogil:
    """
    Solid angle computation subtended by a
    sphere of radius `radius` at a distance `distance`.

    Parameters
    ----------
    radius: double
        radius of the particle
    distance: double
        distance between particle and viewing point
    volpi: double
        volume unit (eg: dPhi * R * dR * dZ) times pi

    Returns
    --------
        Solid angle times unit volume:
        \Omega * dVol = pi (r/d)^2 + pi/4 (r/d)^4
    """
    cdef double r_over_d = radius / distance

    return 2 * volpi * (1. - c_sqrt(1. - r_over_d**2))


# ##################################################################################
# ##################################################################################
#               Solid angle of a polygon
# ##################################################################################


cdef inline double comp_sa_tri(
    double A_x,
    double A_y,
    double A_z,
    double B_x,
    double B_y,
    double B_z,
    double C_x,
    double C_y,
    double C_z,
    double pt_x,
    double pt_y,
    double pt_z,
) nogil:
    """
    Given by:
    numerator = 3 G \dot (b \cross c)
    denominator =  A B C + (A.B)C + (A.C)B + (B.C)A
    with G centroid of triangle
    """
    cdef double G_x
    cdef double G_y
    cdef double G_z
    cdef double cross_bc_x
    cdef double cross_bc_y
    cdef double cross_bc_z
    cdef double An
    cdef double Bn
    cdef double Cn
    cdef double sca_AB
    cdef double sca_AC
    cdef double sca_BC

    cdef double numerator
    cdef double denominator
    cdef double result

    # get centroid
    G_x = (A_x + B_x + C_x) / 3.
    G_y = (A_y + B_y + C_y) / 3.
    G_z = (A_z + B_z + C_z) / 3.

    # get (b cross c)
    cross_bc_x = (B_y - G_y)*(C_z - G_z) - (B_z - G_z)*(C_y - G_y)
    cross_bc_y = (B_z - G_z)*(C_x - G_x) - (B_x - G_x)*(C_z - G_z)
    cross_bc_z = (B_x - G_x)*(C_y - G_y) - (B_y - G_y)*(C_x - G_x)

    # numerator
    numerator = 3.*c_abs(
        (G_x - pt_x) * cross_bc_x
        + (G_y - pt_y) * cross_bc_y
        + (G_z - pt_z) * cross_bc_z
    )

    # norms
    An = c_sqrt((A_x - pt_x)**2 + (A_y - pt_y)**2 + (A_z - pt_z)**2)
    Bn = c_sqrt((B_x - pt_x)**2 + (B_y - pt_y)**2 + (B_z - pt_z)**2)
    Cn = c_sqrt((C_x - pt_x)**2 + (C_y - pt_y)**2 + (C_z - pt_z)**2)

    sca_AB = (A_x - pt_x)*(B_x - pt_x) + (A_y - pt_y)*(B_y - pt_y) + (A_z - pt_z)*(B_z - pt_z)
    sca_AC = (A_x - pt_x)*(C_x - pt_x) + (A_y - pt_y)*(C_y - pt_y) + (A_z - pt_z)*(C_z - pt_z)
    sca_BC = (B_x - pt_x)*(C_x - pt_x) + (B_y - pt_y)*(C_y - pt_y) + (B_z - pt_z)*(C_z - pt_z)

    denominator = An*Bn*Cn + sca_AB * Cn + sca_AC * Bn + sca_BC * An

    # handfle negative denominator
    return 2 * c_atan2(numerator, denominator)
