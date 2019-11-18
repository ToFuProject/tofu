# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for sampling and discretizating
################################################################################
from libc.math cimport ceil as Cceil, fabs as Cabs
from libc.math cimport floor as Cfloor, round as Cround
from libc.math cimport sqrt as Csqrt
from libc.math cimport isnan as Cisnan
from libc.math cimport NAN as Cnan
from libc.math cimport log2 as Clog2
from libc.stdlib cimport malloc, free, realloc
from cython.parallel import prange
from cython.parallel cimport parallel
from _basic_geom_tools cimport _VSMALL
# for utility functions:
import numpy as np
cimport numpy as cnp
# tofu libs
cimport _basic_geom_tools as _bgt


# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================
cdef inline long discretize_line1d_core(double* lminmax, double dstep,
                                        double[2] dl, bint lim,
                                        int mode, double margin,
                                        double** ldiscret_arr,
                                        double[1] resolution,
                                        long** lindex_arr, long[1] n) nogil:
    cdef int[1] nL0
    cdef long[1] nind

    first_discretize_line1d_core(lminmax, dstep,
                                 resolution, n, nind, nL0,
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
    discretising the segmen lminmax with the given parameters. It doesn't do the
    actual discretization.
    For that part, please refer to: second_discretize_line1d_core
    """
    cdef int nl1, ii, jj
    cdef double abs0, abs1
    cdef double inv_resol, new_margin
    cdef double[2] desired_limits

    # .. Computing "real" discretization step, depending on `mode`..............
    if mode == 1: # absolute
        ncells[0] = <int>Cceil((lminmax[1] - lminmax[0]) / dstep)
    else: # relative
        ncells[0] = <int>Cceil(1./dstep)
    resolution[0] = (lminmax[1] - lminmax[0]) / ncells[0]
    # .. Computing desired limits ..............................................
    if Cisnan(dl[0]) and Cisnan(dl[1]):
        desired_limits[0] = lminmax[0]
        desired_limits[1] = lminmax[1]
    else:
        if Cisnan(dl[0]):
            dl[0] = lminmax[0]
        if Cisnan(dl[1]):
            dl[1] = lminmax[1]
        if lim and dl[0]<=lminmax[0]:
            dl[0] = lminmax[0]
        if lim and dl[1]>=lminmax[1]:
            dl[1] = lminmax[1]
        desired_limits[0] = dl[0]
        desired_limits[1] = dl[1]
    # .. Get the extreme indices of the mesh elements that really need to be
    # created within those limits...............................................
    inv_resol = 1./resolution[0]
    new_margin = margin*resolution[0]
    abs0 = Cabs(desired_limits[0] - lminmax[0])
    if abs0 - resolution[0] * Cfloor(abs0 * inv_resol) < new_margin:
        nl0[0] = int(Cround((desired_limits[0] - lminmax[0]) * inv_resol))
    else:
        nl0[0] = int(Cfloor((desired_limits[0] - lminmax[0]) * inv_resol))
    abs1 = Cabs(desired_limits[1] - lminmax[0])
    if abs1 - resolution[0] * Cfloor(abs1 * inv_resol) < new_margin:
        nl1 = int(Cround((desired_limits[1] - lminmax[0]) * inv_resol) - 1)
    else:
        nl1 = int(Cfloor((desired_limits[1] - lminmax[0]) * inv_resol))
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

    if mode == 1: # absolute
        ncells = <int>Cceil((lminmax[1] - first) / dstep)
    else: # relative
        ncells = <int>Cceil(1./dstep)
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
    dl_array[0] = Cnan
    dl_array[1] = Cnan
    ncells[0] = <long*>malloc((np-1)*sizeof(long))
    #.. Filling arrays..........................................................
    if Cabs(din) < _VSMALL:
        for ii in range(np-1):
            v0 = ves_poly[0,ii+1]-ves_poly[0,ii]
            v1 = ves_poly[1,ii+1]-ves_poly[1,ii]
            lminmax[1] = Csqrt(v0 * v0 + v1 * v1)
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
                rref[0][last_sz_othr + jj] = ves_poly[0,ii] + ldiscret[jj] * v0
                xcross[0][last_sz_othr + jj] = ves_poly[0,ii] + ldiscret[jj] * v0
                ycross[0][last_sz_othr + jj] = ves_poly[1,ii] + ldiscret[jj] * v1
                xpolybis[0][last_sz_vbis + jj] = ves_poly[0,ii] + jj * rv0
                ypolybis[0][last_sz_vbis + jj] = ves_poly[1,ii] + jj * rv1
        # We close the polygon of VPolybis
        sz_vbis += 1
        xpolybis[0] = <double*>realloc(xpolybis[0], sz_vbis*sizeof(double))
        ypolybis[0] = <double*>realloc(ypolybis[0], sz_vbis*sizeof(double))
        xpolybis[0][sz_vbis - 1] = ves_poly[0, 0]
        ypolybis[0][sz_vbis - 1] = ves_poly[1, 0]
    else:
        for ii in range(np-1):
            v0 = ves_poly[0,ii+1]-ves_poly[0,ii]
            v1 = ves_poly[1,ii+1]-ves_poly[1,ii]
            lminmax[1] = Csqrt(v0 * v0 + v1 * v1)
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
            shiftx = din*ves_vin[0,ii]
            shifty = din*ves_vin[1,ii]
            for jj in range(ncells[0][ii]):
                ind[0][last_sz_othr] = last_sz_othr
                reso[0][last_sz_othr] = loc_resolu[0]
                rref[0][last_sz_othr]   = ves_poly[0,ii] + ldiscret[jj]*v0
                xcross[0][last_sz_othr] = ves_poly[0,ii] + ldiscret[jj]*v0 + shiftx
                ycross[0][last_sz_othr] = ves_poly[1,ii] + ldiscret[jj]*v1 + shifty
                xpolybis[0][last_sz_vbis + jj] = ves_poly[0,ii] + jj * rv0
                ypolybis[0][last_sz_vbis + jj] = ves_poly[1,ii] + jj * rv1
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
        v0 = ves_poly[0,ii+1]-ves_poly[0,ii]
        v1 = ves_poly[1,ii+1]-ves_poly[1,ii]
        lminmax[1] = Csqrt(v0 * v0 + v1 * v1)
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
            xcross[0][last_sz_othr + jj] = ves_poly[0,ii] + ldiscret[jj] * v0
            ycross[0][last_sz_othr + jj] = ves_poly[1,ii] + ldiscret[jj] * v1
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
        for ii in prange(1, nlos):
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
    num_raf = <int>(Cceil(seg_length*inv_resol))
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
    num_raf = <int>(Cceil(seg_length/resolutions[0]))
    loc_resol = seg_length / num_raf
    # keeping values
    los_nraf[0] = num_raf
    eff_resolution[0] = loc_resol
    los_ind[0] = num_raf
    first_index = 0
    # Now the rest ...................................................
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length/resolutions[ii]))
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

    # Treting ilos= 0 first .......................................
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
    cdef double seg_length
    cdef double loc_resol
    # ... Treating the first los .....................................
    num_raf = <int>(Cceil(1./resolutions[0]))
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf
    # .. Treating the rest of los ....................................
    for ii in range(1,nlos):
        num_raf = <int>(Cceil(1./resolutions[ii]))
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
    num_raf = <int>(Cceil(seg_length*inv_resol))
    if num_raf%2==1:
        num_raf = num_raf + 1
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ... Treating the rest of los .....................................
    for ii in range(1, nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length*inv_resol))
        if num_raf%2==1:
            num_raf = num_raf + 1
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
    cdef Py_ssize_t ii, jj
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
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    cdef double inv_resol = 1./resol
    # ... Treating the first los ....................................
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(Cceil(seg_length*inv_resol))
    num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ... Treating the rest of the los ..............................
    for ii in range(1, nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length*inv_resol))
        num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
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
    num_raf = <int>(Cceil(1./resolutions[0]))
    if num_raf%2==1:
        num_raf = num_raf+1
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1, nlos):
        num_raf = <int>(Cceil(1./resolutions[ii]))
        if num_raf%2==1:
            num_raf = num_raf+1
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
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(Cceil(seg_length/resolutions[0]))
    if num_raf%2==1:
        num_raf = num_raf+1
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length/resolutions[ii]))
        if num_raf%2==1:
            num_raf = num_raf+1
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
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double loc_resol
    num_raf = <int>(Cceil(1./resolutions[0]))
    num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
    loc_resol = (los_kmax[0] - los_kmin[0])/num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        num_raf = <int>(Cceil(1./resolutions[ii]))
        num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
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
    cdef Py_ssize_t ii, jj
    cdef int num_raf
    cdef int first_index
    cdef double seg_length
    cdef double loc_resol
    seg_length = los_kmax[0] - los_kmin[0]
    num_raf = <int>(Cceil(seg_length/resolutions[0]))
    num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
    loc_resol = seg_length / num_raf
    eff_resolution[0] = loc_resol
    los_nraf[0] = num_raf
    first_index = 0
    los_ind[0] = num_raf + 1
    # ...
    for ii in range(1,nlos):
        seg_length = los_kmax[ii] - los_kmin[ii]
        num_raf = <int>(Cceil(seg_length/resolutions[ii]))
        num_raf = 2**(<int>(Cceil(Clog2(num_raf))))
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
    return 0 # absolute


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
        nraf = <int> Cceil(1./resol)
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
                nraf = nraf+1
            invnraf = 1./nraf
            coeffs[0] = <double*>malloc((nraf + 1)*sizeof(double))
            eff_res[0] = (los_kmax - los_kmin)*invnraf
            left_rule_single(nraf, los_kmin,
                                 eff_res[0], &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            nraf = 2**(<int>(Cceil(Clog2(nraf))))
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
            nraf = <int>(Cceil(seg_length/resol))
            if nraf%2==1:
                nraf = nraf+1
            eff_res[0] = seg_length / nraf
            coeffs[0] = <double*>malloc((nraf+1)*sizeof(double))
            left_rule_single(nraf, los_kmin, eff_res[0],
                                   &coeffs[0][0])
            return nraf + 1
        elif n_imode==2:
            # 'romb' quad
            seg_length = los_kmax - los_kmin
            nraf = <int>(Cceil(seg_length/resol))
            nraf = 2**(<int>(Cceil(Clog2(nraf))))
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
                                       double[:,::1] ray_orig,
                                       double[:,::1] ray_vdir):
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
cdef inline cnp.ndarray[double,ndim=2,mode='c'] call_get_sample_single(double los_kmin, double los_kmax,
                                   double resol,
                                   int n_dmode, int n_imode,
                                   double[1] eff_res,
                                   long[1] nb_rows,
                                   double[:,::1] ray_orig,
                                   double[:,::1] ray_vdir):
    # This function doesn't compute anything new.
    # It's a utility function for LOS_calc_signal to avoid reptitions
    # It samples a LOS and recreates the points on that LOS
    # plus this is for the anisotropic version so it also compute usbis
    cdef int sz_coeff
    cdef int ii, jj
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
                                              double* dLr,
                                              long* los_ind,
                                              int num_threads) nogil:
    # ...
    cdef int N
    cdef int ntmp
    if n_dmode==1: # relative
        #         return coeff_arr, dLr, los_ind[:nlos-1]
        N = <int> Cceil(1./val_resol)
        if n_imode==0: # sum
            #coeff_arr = np.empty((N*nlos,), dtype=float)
            coeff_ptr[0] = <double*>malloc(sizeof(double)*N*nlos)
            middle_rule_rel(nlos, N, los_lim_min, los_lim_max,
                                &dLr[0], coeff_ptr[0], los_ind,
                                num_threads=num_threads)
            return N*nlos
        elif n_imode==1: #simps
            N = N if N%2==0 else N+1
            # coeff_arr = np.empty(((N+1)*nlos,), dtype=float)
            coeff_ptr[0] = <double*>malloc(sizeof(double)*(N+1)*nlos)
            left_rule_rel(nlos, N,
                          los_lim_min, los_lim_max, &dLr[0],
                          coeff_ptr[0], los_ind,
                          num_threads=num_threads)
            return (N+1)*nlos
        elif n_imode==2: #romb
            N = 2**(<int>Cceil(Clog2(N)))
            # coeff_arr = np.empty(((N+1)*nlos,), dtype=float)
            coeff_ptr[0] = <double*>malloc(sizeof(double)*(N+1)*nlos)
            left_rule_rel(nlos, N,
                          los_lim_min, los_lim_max,
                          &dLr[0], coeff_ptr[0], los_ind,
                          num_threads=num_threads)
            return (N+1)*nlos
    else: # absolute
        if n_imode==0: #sum
            middle_rule_abs_s1(nlos, val_resol, los_lim_min, los_lim_max,
                              &dLr[0], los_ind,
                              num_threads=num_threads)
            #ntmp = np.sum(los_ind)
            #coeff_arr = np.empty((ntmp,), dtype=float)
            ntmp = _bgt.sum_naive_int(los_ind, nlos)
            coeff_ptr[0] = <double*>malloc(sizeof(double)*ntmp)
            middle_rule_abs_s2(nlos, los_lim_min, &dLr[0],
                              los_ind, coeff_ptr[0],
                              num_threads=num_threads)
            return ntmp
        elif n_imode==1:# simps
            simps_left_rule_abs(nlos, val_resol,
                                los_lim_min, los_lim_max,
                                &dLr[0], coeff_ptr, los_ind,
                                num_threads=num_threads)
            return los_ind[nlos-1]
        else:# romb
            romb_left_rule_abs(nlos, val_resol,
                               los_lim_min, los_lim_max,
                               &dLr[0], coeff_ptr, los_ind,
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
                                    double[:,::1] ray_orig,
                                    double[:,::1] ray_vdir,
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
