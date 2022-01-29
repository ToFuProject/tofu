# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for Ray-tracing
################################################################################
cimport cython
from libc.math cimport sqrt as c_sqrt, fabs as c_abs
from libc.math cimport cos as c_cos, sin as c_sin
from libc.math cimport atan2 as c_atan2
from libc.math cimport NAN as C_NAN
from libc.math cimport pi as c_pi
from libc.math cimport isnan as c_isnan
from cython.parallel import prange
from cython.parallel cimport parallel
from cpython.array cimport array, clone
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
from cython cimport view
# importing ToFu functions:
from ._basic_geom_tools cimport _VSMALL
from ._basic_geom_tools cimport is_point_in_path
from ._basic_geom_tools cimport is_point_in_path_vec
from ._basic_geom_tools cimport compute_inv_and_sign
from . cimport _basic_geom_tools as _bgt

# ==============================================================================
# =  3D Bounding box (not Toroidal)
# ==============================================================================
cdef inline void compute_3d_bboxes(const double*const* vignett_poly,
                                   const long* lnvert,
                                   const int nvign,
                                   double* lbounds,
                                   const int num_threads) nogil:
    """
    Computes coordinates of bounding boxes of a list of 3d objects in a general
    space (not related to a tore).
    """
    cdef int ivign
    cdef int nvert
    # ...
    # -- Defining parallel part ------------------------------------------------
    with nogil, parallel(num_threads=num_threads):
        for ivign in prange(nvign):
            nvert = lnvert[ivign]
            comp_bbox_poly3d(nvert,
                             &vignett_poly[ivign][0*nvert],
                             &vignett_poly[ivign][1*nvert],
                             &vignett_poly[ivign][2*nvert],
                             &lbounds[ivign*6])
    return


cdef inline void comp_bbox_poly3d(const int nvert,
                                  const double* vertx,
                                  const double* verty,
                                  const double* vertz,
                                  double[6] bounds) nogil:
    """
    Computes bounding box of a 3d polygon
    Params
    =====
    nvert : integer
       Number of vertices in the poygon
    vert : double array
       Coordinates of the polygon defining the structure in the poloidal plane
       such that vert[0:3, ii] = (x_i, y_i, z_i) the coordinates of the i-th
       vertex
    bounds : (6) double array <INOUT>
       coordinates of the lowerleftback point and of the upperrightfront point
       of the bounding box of the polygon
    """
    cdef int ii
    cdef double xmax=vertx[0], xmin=vertx[0]
    cdef double ymax=verty[0], ymin=verty[0]
    cdef double zmax=vertz[0], zmin=vertz[0]
    cdef double tmp_val
    for ii in range(1, nvert):
        # x....
        tmp_val = vertx[ii]
        if tmp_val > xmax:
            xmax = tmp_val
        elif tmp_val < xmin :
            xmin = tmp_val
        # y....
        tmp_val = verty[ii]
        if tmp_val > ymax:
            ymax = tmp_val
        elif tmp_val < ymin :
            ymin = tmp_val
        # z....
        tmp_val = vertz[ii]
        if tmp_val > zmax:
            zmax = tmp_val
        elif tmp_val < zmin :
            zmin = tmp_val
    bounds[0] = xmin
    bounds[1] = ymin
    bounds[2] = zmin
    bounds[3] = xmax
    bounds[4] = ymax
    bounds[5] = zmax
    return

# ==============================================================================
# =  Computation of Bounding Boxes (in toroidal configuration)
# ==============================================================================
cdef inline void comp_bbox_poly_tor(const int nvert,
                                    const double* vertr,
                                    const double* vertz,
                                    double[6] bounds) nogil:
    """
    Computes bounding box of a toroidally continous structure defined by
    the vertices vert.
    Params
    =====
        nvert : integer
           Number of vertices in the poygon
        vert : double array
           Coordinates of the polygon defining the structure in the poloidal
           plane such that vert[0:3, ii] = (x_i, y_i) the coordinates of the
           i-th vertex
        bounds : (6) double array <INOUT>
           coordinates of the lowerleftback point and of the upperrightfront
           point of the bounding box of the structure toroidally continous on
           the tore.
    """
    cdef int ii
    cdef double rmax=vertr[0], zmin=vertz[0], zmax=vertz[0]
    cdef double tmp_val
    for ii in range(1, nvert):
        tmp_val = vertr[ii]
        if tmp_val > rmax:
            rmax = tmp_val
        tmp_val = vertz[ii]
        if tmp_val > zmax:
            zmax = tmp_val
        elif tmp_val < zmin:
            zmin = tmp_val
    bounds[0] = -rmax
    bounds[1] = -rmax
    bounds[2] = zmin
    bounds[3] = rmax
    bounds[4] = rmax
    bounds[5] = zmax
    return

cdef inline void comp_bbox_poly_tor_lim(const int nvert,
                                        const double* vertr,
                                        const double* vertz,
                                        double[6] bounds,
                                        const double lmin,
                                        const double lmax) nogil:
    """
    Computes bounding box of a toroidally limited structure defined by
    the vertices vert, and limited to the angles (lmin, lmax)
    Params
    =====
    nvert : integer
       Number of vertices in the poygon
    vert : double array
       Coordinates of the polygon defining the structure in the poloidal plane
       such that vert[0:3, ii] = (x_i, y_i) the coordinates of the i-th vertex
    bounds : (6) double array <INOUT>
       coordinates of the lowerleftback point and of the upperrightfront point
       of the bounding box of the structure toroidally limited on the tore.
    lmin : double
       minimum toroidal angle where the structure lays.
    lmax : double
       maximum toroidal angle where the structure lays.
    """
    cdef int ii
    cdef double toto=100000.
    cdef double xmin=toto, xmax=-toto
    cdef double ymin=toto, ymax=-toto
    cdef double zmin=toto, zmax=-toto
    cdef double cos_min, sin_min
    cdef double cos_max, sin_max
    cdef double half_pi = 0.5 * c_pi
    cdef double[3] temp
    cdef double[6] bounds_min
    # ...
    cos_min = c_cos(lmin)
    sin_min = c_sin(lmin)
    cos_max = c_cos(lmax)
    sin_max = c_sin(lmax)
    if (lmin >= 0.) and (lmax >= 0.):
        if lmax > half_pi and lmin < half_pi:
            comp_bbox_poly_tor(nvert, vertr, vertz, &bounds_min[0])
            if ymax < bounds_min[4]:
                ymax = bounds_min[4]
        elif lmax < half_pi and lmin > half_pi:
            comp_bbox_poly_tor(nvert, vertr, vertz, &bounds_min[0])
            if ymin > bounds_min[1]:
                ymin = bounds_min[1]
    elif (lmin <= 0 and lmax <= 0):
        if lmax < -half_pi and lmin > -half_pi:
            comp_bbox_poly_tor(nvert, vertr, vertz, &bounds_min[0])
            if ymin > bounds_min[1]:
                ymin = bounds_min[1]
        elif lmax > -half_pi and lmin < -half_pi:
            comp_bbox_poly_tor(nvert, vertr, vertz, &bounds_min[0])
            if ymax < bounds_min[4]:
                ymax = bounds_min[4]
    elif (c_abs(c_abs(lmin) - c_pi) > _VSMALL
          and c_abs(c_abs(lmax) - c_pi) > _VSMALL):
        if lmin >= 0 :
            # lmin and lmax of opposite signs, so lmax < 0. Divide and conquer:
            comp_bbox_poly_tor_lim(nvert, vertr, vertz, &bounds[0], lmin, c_pi)
            comp_bbox_poly_tor_lim(nvert, vertr, vertz, &bounds_min[0],
                                   -c_pi, lmax)
        else:
            # lmin and lmax of opposite signs, so lmax <= 0. Divide and conquer:
            comp_bbox_poly_tor_lim(nvert, vertr, vertz, &bounds[0], lmin, -0.0)
            comp_bbox_poly_tor_lim(nvert, vertr, vertz, &bounds_min[0], 0, lmax)
        # we compute the extremes of the two boxes:
        for ii in range(3):
            if bounds[ii] > bounds_min[ii]:
                bounds[ii] = bounds_min[ii]
        for ii in range(3, 6):
            if bounds[ii] < bounds_min[ii]:
                bounds[ii] = bounds_min[ii]
        return
    for ii in range(nvert):
        temp[0] = vertr[ii]
        temp[1] = vertz[ii]
        coordshift_simple1d(temp, False, 1., cos_min, sin_min)
        # initialization:
        if xmin > temp[0]:
            xmin = temp[0]
        if xmax < temp[0]:
            xmax = temp[0]
        if ymin > temp[1]:
            ymin = temp[1]
        if ymax < temp[1]:
            ymax = temp[1]
        if zmin > temp[2]:
            zmin = temp[2]
        if zmax < temp[2]:
            zmax = temp[2]
        # .....
        temp[0] = vertr[ii]
        temp[1] = vertz[ii]
        coordshift_simple1d(temp, False, 1., cos_max, sin_max)
        if xmin > temp[0]:
            xmin = temp[0]
        if xmax < temp[0]:
            xmax = temp[0]
        if ymin > temp[1]:
            ymin = temp[1]
        if ymax < temp[1]:
            ymax = temp[1]
        if zmin > temp[2]:
            zmin = temp[2]
        if zmax < temp[2]:
            zmax = temp[2]
    bounds[0] = xmin
    bounds[1] = ymin
    bounds[2] = zmin
    bounds[3] = xmax
    bounds[4] = ymax
    bounds[5] = zmax
    return

cdef inline void coordshift_simple1d(double[3] pts, bint in_is_cartesian,
                                     double CrossRef, double cos_phi,
                                     double sin_phi) nogil:
    """
    Similar to coordshift but only pas from 3D cartesian to 3D toroidal
    coordinates or vice-versa.
    """
    cdef double x, y, z
    cdef double r, p
    if in_is_cartesian:
        if CrossRef==0.:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            pts[0] = c_sqrt(x*x+y*y)
            pts[1] = z
            pts[2] = c_atan2(y,x)
        else:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            pts[0] = c_sqrt(x*x+y*y)
            pts[1] = z
            pts[2] = CrossRef
    else:
        if CrossRef==0.:
            r = pts[0]
            z = pts[1]
            p = pts[2]
            pts[0] = r*c_cos(p)
            pts[1] = r*c_sin(p)
            pts[2] = z
        else:
            r = pts[0]
            z = pts[1]
            pts[0] = r*cos_phi
            pts[1] = r*sin_phi
            pts[2] = z
    return


cdef inline void is_inside_vessel(double[:, ::1] pts, double[:, ::1] ves_poly,
                                  double[:,::1] ves_lims, int nlim,
                                  bint is_toroidal, bint in_is_cartesian,
                                  int[3] order,
                                  int[::1] is_inside) nogil:
    """
    check if points are inside given vessel
    pts : points to check if inside
    ves_poly : (2, nvert) array defining polygon of the vessel
    ves_lims : (2, nlim) limits of the vessel, None if contiguous
    nlim : number of limits
    is_toroidal : true if vessel is toroidal
    in_is_cartesian : true if pts coordinates are given in cartesian coordinates
    order : order where the coordinates are stored, for example :
              - (x,y,z) => order = (0, 1, 2)
              - (z,x,y) => order = (1, 2, 0)
              - (r, z, phi) => order = (0, 1, 2)
              - (z, r, phi) => order = (1, 0, 2)
    is_inside : result, array of bools of size npts
    """
    cdef int ii, jj
    cdef int fst
    cdef int npts = pts.shape[1]
    cdef int nvert = ves_poly.shape[1] - 1
    cdef double[3] shift_pts
    cdef double[2] lims
    cdef double rii, zii, pii
    cdef bint in_ves
    # --------------------------------------------------------------------------
    if is_toroidal:
        if ves_lims is None or nlim == 0:
            if in_is_cartesian:
                for ii in range(npts):
                    shift_pts[0] = pts[order[0], ii]
                    shift_pts[1] = pts[order[1], ii]
                    shift_pts[2] = pts[order[2], ii]
                    coordshift_simple1d(shift_pts, in_is_cartesian,
                                        -1, # we dont care about phi
                                        -1, -1) # no need for these args
                    is_inside[ii] = is_point_in_path(nvert,
                                                     &ves_poly[0][0],
                                                     &ves_poly[1][0],
                                                     shift_pts[0], shift_pts[1])
            else:
                is_point_in_path_vec(nvert, &ves_poly[0][0], &ves_poly[1][0],
                                     npts, &pts[order[0]][0], &pts[order[1]][0],
                                     &is_inside[0])
        else:
            # -- There are limits ----------------------------------------------
            if in_is_cartesian:
                for jj in range(nlim):
                    fst = jj * npts
                    lims[0] = c_atan2(c_sin(ves_lims[jj][0]),
                                     c_cos(ves_lims[jj][0]))
                    lims[1] = c_atan2(c_sin(ves_lims[jj][1]),
                                     c_cos(ves_lims[jj][1]))
                    if lims[0] < lims[1]:
                        for ii in range(npts):
                            shift_pts[0] = pts[order[0], ii]
                            shift_pts[1] = pts[order[1], ii]
                            shift_pts[2] = pts[order[2], ii]
                            coordshift_simple1d(shift_pts, in_is_cartesian,
                                                0, # we need to compute phi
                                                -1, -1) # no need for these args
                            in_ves = is_point_in_path(nvert,
                                                      &ves_poly[0][0],
                                                      &ves_poly[1][0],
                                                      shift_pts[0],
                                                      shift_pts[1])
                            is_inside[fst + ii] = (in_ves
                                                   & (shift_pts[2]>=lims[0])
                                                   & (shift_pts[2]<=lims[1]))
                    else:
                        for ii in range(npts):
                            shift_pts[0] = pts[order[0], ii]
                            shift_pts[1] = pts[order[1], ii]
                            shift_pts[2] = pts[order[2], ii]
                            coordshift_simple1d(shift_pts, in_is_cartesian,
                                                0, # we need to compute phi
                                                -1, -1) # no need for these args
                            in_ves = is_point_in_path(nvert,
                                                      &ves_poly[0][0],
                                                      &ves_poly[1][0],
                                                      shift_pts[0],
                                                      shift_pts[1])
                            is_inside[fst + ii] = (in_ves
                                                   & ((shift_pts[2]>=lims[0])
                                                      | (shift_pts[2]<=lims[1])))
            else:
                # -- in coordinates: polar -------------------------------------
                for jj in range(nlim):
                    fst = jj * npts
                    lims[0] = c_atan2(c_sin(ves_lims[jj][0]),
                                     c_cos(ves_lims[jj][0]))
                    lims[1] = c_atan2(c_sin(ves_lims[jj][1]),
                                     c_cos(ves_lims[jj][1]))
                    if lims[0] < lims[1]:
                        for ii in range(npts):
                            rii = pts[order[0],ii]
                            zii = pts[order[1],ii]
                            pii = c_atan2(c_sin(pts[order[2],ii]),
                                         c_cos(pts[order[2],ii]))
                            in_ves = is_point_in_path(nvert,
                                                      &ves_poly[0][0],
                                                      &ves_poly[1][0],
                                                      rii, zii)
                            is_inside[fst + ii] = (in_ves
                                                   & (pii>=lims[0])
                                                   & (pii<=lims[1]))
                    else:
                        for ii in range(npts):
                            rii = pts[order[0],ii]
                            zii = pts[order[1],ii]
                            pii = c_atan2(c_sin(pts[order[2],ii]),
                                         c_cos(pts[order[2],ii]))
                            in_ves = is_point_in_path(nvert,
                                                      &ves_poly[0][0],
                                                      &ves_poly[1][0],
                                                      rii, zii)
                            is_inside[fst + ii] = (in_ves
                                                   & ((pii>=lims[0])
                                                      | (pii<=lims[1])))
    else:
        # -- Cylindrical case --------------------------------------------------
        if in_is_cartesian:
            for jj in range(nlim):
                fst = jj * npts
                for ii in range(npts):
                    in_ves = is_point_in_path(nvert,
                                              &ves_poly[0][0],
                                              &ves_poly[1][0],
                                              pts[order[1],ii],
                                              pts[order[2],ii])
                    is_inside[fst + ii] = (in_ves
                                           & (pts[order[0],ii]>=ves_lims[0][0])
                                           & (pts[order[0],ii]<=ves_lims[0][1]))
        else:
            # polar coordinates
            for jj in range(nlim):
                fst = jj * npts
                for ii in range(npts):
                    shift_pts[0] = pts[order[0], ii]
                    shift_pts[1] = pts[order[1], ii]
                    shift_pts[2] = pts[order[2], ii]
                    coordshift_simple1d(shift_pts, in_is_cartesian,
                                        0, # we need to compute phi
                                        -1, -1) # no need for these args
                    in_ves = is_point_in_path(nvert,
                                              &ves_poly[0][0],
                                              &ves_poly[1][0],
                                              shift_pts[1],
                                              shift_pts[2])
                    is_inside[fst + ii] = (in_ves
                                           & (shift_pts[0]>=ves_lims[0][0])
                                           & (shift_pts[0]<=ves_lims[0][1]))
    return





# ==============================================================================
# =  Raytracing basic tools: intersection ray and axis aligned bounding box
# ==============================================================================
cdef inline bint inter_ray_aabb_box(const int[3] sign,
                                    const double[3] inv_direction,
                                    const double* bounds,
                                    const double[3] ds,
                                    const bint countin) nogil:
    """
    Computes intersection between a ray (LOS) and a axis aligned bounding
    box. It returns True if ray intersects box, else False.
    Params
    =====
       sign : (3) int array
          Sign of the direction of the ray.
          If sign[i] = 1, ray_vdir[i] < 0, else sign[i] = 0
       inv_direction : (3) double array
          Inverse on each axis of direction of LOS
       bounds : (6) double array
          [3d coords of lowerleftback point of bounding box,
           3d coords of upperrightfront point of bounding box]
       ds : (3) double array
          [3d coords of origin of ray]
    Returns
    =======
       True if ray intersects bounding box, else False
    """
    cdef double tmin, tmax, tymin, tymax
    cdef double tzmin, tzmax
    cdef int t0 = 1000000
    cdef bint res

    # computing intersection
    tmin = (bounds[sign[0]*3] - ds[0]) * inv_direction[0]
    tmax = (bounds[(1-sign[0])*3] - ds[0]) * inv_direction[0]
    tymin = (bounds[(sign[1])*3 + 1] - ds[1]) * inv_direction[1]
    tymax = (bounds[(1-sign[1])*3+1] - ds[1]) * inv_direction[1]
    if ( (tmin > tymax) or (tymin > tmax) ):
        return 0
    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax
    if not inv_direction[2] == 1./_VSMALL:
        tzmin = (bounds[(sign[2])*3+2] - ds[2]) * inv_direction[2]
        tzmax = (bounds[(1-sign[2])*3+2] - ds[2]) * inv_direction[2]
    else:
        tzmin = C_NAN
        tzmax = C_NAN
    if ( (tmin > tzmax) or (tzmin > tmax) ):
        return 0
    if (tzmin > tmin):
        tmin = tzmin
    if (tzmax < tmax):
        tmax = tzmax
    if countin and (tmin < 0.) and (tmax < 0.):
        return 0
    elif not countin and tmin < 0:
        return 0

    res = (tmin < t0) and (tmax > -t0)
    return  res


# ==============================================================================
# =  Raytracing basic tools: intersection ray and triangle (in 3d space)
# ==============================================================================
cdef inline bint inter_ray_triangle(const double[3] ray_orig,
                                    const double[3] ray_vdir,
                                    const double* vert0,
                                    const double* vert1,
                                    const double* vert2) nogil:
    cdef int ii
    cdef double det, invdet, u, v
    cdef double[3] edge1, edge2
    cdef double[3] pvec, tvec, qvec
    #...
    for ii in range(3):
        edge1[ii] = vert1[ii] - vert0[ii]
        edge2[ii] = vert2[ii] - vert0[ii]
    # begin calculating determinant  also used to calculate U parameter
    _bgt.compute_cross_prod(ray_vdir, edge2, pvec)
    # if determinant is near zero ray lies in plane of triangle
    det = _bgt.compute_dot_prod(edge1, pvec)
    if c_abs(det) < _VSMALL:
        return False
    invdet = 1./det
    # calculate distance from vert to ray origin
    for ii in range(3):
        tvec[ii] = ray_orig[ii] - vert0[ii]
    # calculate U parameter and test bounds
    u = _bgt.compute_dot_prod(tvec, pvec) * invdet
    if u < 0. or u > 1.:
        return False
    # prepare to test V parameter
    _bgt.compute_cross_prod(tvec, edge1, qvec)
    # calculate V parameter and test bounds
    v = _bgt.compute_dot_prod(ray_vdir, qvec) * invdet
    if v < 0. or u + v > 1.:
        return False
    return True


# ==============================================================================
# =  Raytracing on a Torus
# ==============================================================================
cdef inline void raytracing_inout_struct_tor(const int num_los,
                                             const double[:,::1] ray_vdir,
                                             const double[:,::1] ray_orig,
                                             double[::1] coeff_inter_out,
                                             double[::1] coeff_inter_in,
                                             double[::1] vperp_out,
                                             const long* lstruct_nlim,
                                             int[::1] ind_inter_out,
                                             const bint forbid0,
                                             const bint forbidbis_org,
                                             const double rmin,
                                             const double rmin2,
                                             const double crit2_base,
                                             const int nstruct_lim,
                                             const double* lbounds,
                                             const double* langles,
                                             const int* lis_limited,
                                             const long* lnvert,
                                             const long* lsz_lim,
                                             const double* lstruct_polyx,
                                             const double* lstruct_polyy,
                                             const double* lstruct_normx,
                                             const double* lstruct_normy,
                                             const double eps_uz,
                                             const double eps_vz,
                                             const double eps_a,
                                             const double eps_b,
                                             const double eps_plane,
                                             const int num_threads,
                                             const bint is_out_struct) nogil:
    """
    Computes the entry and exit point of all provided LOS/rays for a set of
    structures that can be of type "OUT" (is_out_struct=True) or "IN"
    (is_out_struct=False) in a TORE. An "OUT" structure cannot be penetrated
    whereas an "IN" structure can. The latter is typically a vessel and are
    toroidally continous. If a structure is limited we can determine the number
    of limits and the limits itself. For optimization reasons we will also pass
    the bounding box limits. And the information of the last intersected point,
    if any.
    This functions is only a wrapper calling `raytracing_inout_struct_tor_inomp`
    after creating a parallel region if and only if num_threads>1.

    Params
    ======
    num_los : int
       Total number of lines of sight (LOS) (aka. rays)
    ray_vdir : (3, num_los) double array
       LOS normalized direction vector
    ray_orig : (3, num_los) double array
       LOS origin points coordinates
    coeff_inter_out : (num_los) double array <INOUT>
       Coefficient of exit (kout) of the last point of intersection for each LOS
       with the global geometry (with ALL structures)
    coeff_inter_in : (num_los) double array <INOUT>
       Coefficient of entry (kin) of the last point of intersection for each LOS
       with the global geometry (with ALL structures). If intersection at origin
       k = 0, if no intersection k = NAN
    vperp_out : (3*num_los) double array <INOUT>
       Coordinates of the normal vector of impact of the LOS (0 if none). It is
       stored in the following way [v_{0,x}, v_{0,y}, v_{0,z}, ..., v_{n-1,z}]
    lstruct_nlim : array of ints
       List of number of limits for all structures
    ind_inter_out : (3 * num_los)  <INOUT>
       Index of structure impacted by LOS such that:
                ind_inter_out[ind_los*3:ind_los*3+3]=(i,j,k)
       where k is the index of edge impacted on the j-th sub structure of the
       structure number i. If the LOS impacted the vessel i=j=0
    forbid0 : bool
       Should we forbid values behind visible radius ? (see Rmin). If false,
       will test "hidden" part always, else, it will depend on the LOS and
       on forbidbis.
    forbidbis: bint
       Should we forbid values behind visible radius for each LOS ?
    rmin : double
       Minimal radius of vessel to take into consideration
    rmin2 : double
       Squared valued of the minimal radius
    crit2_base : double
       Critical value to evaluate for each LOS if horizontal or not
    nstruct_lim : int
       Number of OUT structures (not counting the limited versions).
       If not is_out_struct then length of vpoly.
    lbounds : (6 * nstruct) double array
       Coordinates of lower and upper edges of the bounding box for each
       structures (nstruct = sum_i(nstruct_lim * lsz_lim[i])
       If not is_out_struct then NULL
    langles : (2 * nstruct) double array
       Minimum and maximum angles where the structure lives. If the structure
       number 'i' is toroidally continous then langles[i:i+2] = [0, 0].
    lis_limited : (nstruct) int array
       List of bool to know if the structures (or the vessel) is limited or not.
    lnvert : (nstruct_lim) int array
       List of vertices of each polygon for each structure
       If not is_out_struct then NULL
    lsz_lim : (nstruct) int array
       List of the total number of structures before the ith structure. First
       element is always 0, else lsz_lim[i] = sum_j(lstruct_nlim[j], j=0..i-1)
       If not is_out_struct then NULL
    lstruct_polyx : (ntotnvert)
       List of "x" coordinates of the polygon's vertices of all structures on
       the poloidal plane
    lstruct_polyy : (ntotnvert)
       List of "y" coordinates of the polygon's vertices of all structures on
       the poloidal plane
    lstruct_normx : (2, num_vertex-1) double array
       List of "x" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by lstruct_poly
    lstruct_normy : (2, num_vertex-1) double array
       List of "y" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by lstruct_poly
    eps<val> : double
       Small value, acceptance of error
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    is_out_struct : bint
       Bool to determine if the structure is "OUT" or "IN". An "OUT" structure
       cannot be penetrated whereas an "IN" structure can. The latter is
       typically a vessel and are toroidally continous.
    """
    cdef int ind_los_range
    cdef int ind_los_prange

    cdef double* kpin_loc = NULL
    cdef double* kpout_loc = NULL
    cdef double* loc_dir = NULL
    cdef double* loc_org = NULL
    cdef double* loc_vp = NULL

    cdef int* ind_loc = NULL
    cdef double* invr_ray = NULL
    cdef double* last_pout = NULL
    cdef double* lim_ves = NULL
    cdef int* sign_ray = NULL

    if num_threads == 1:
        # == Allocating loop  variables ========================================

        # We use local arrays for each thread
        kpin_loc  = <double *> malloc(sizeof(double) * 1)
        kpout_loc = <double *> malloc(sizeof(double) * 1)
        loc_dir   = <double *> malloc(sizeof(double) * 3)
        loc_org   = <double *> malloc(sizeof(double) * 3)
        loc_vp    = <double *> malloc(sizeof(double) * 3)
        ind_loc   = <int *> malloc(sizeof(int) * 1)
        if is_out_struct:
            # if the structure is "out" (solid) we need more arrays
            invr_ray  = <double *> malloc(sizeof(double) * 3)
            last_pout = <double *> malloc(sizeof(double) * 3)
            lim_ves   = <double *> malloc(sizeof(double) * 2)
            sign_ray  = <int *> malloc(sizeof(int) * 3)

        # == Iterating over the LOS ============================================
        for ind_los_range in range(num_los):
            raytracing_inout_struct_tor_inomp(num_los, ray_vdir, ray_orig,
                                              coeff_inter_out,
                                              coeff_inter_in,
                                              vperp_out, lstruct_nlim,
                                              ind_inter_out,
                                              forbid0, forbidbis_org,
                                              rmin, rmin2, crit2_base,
                                              nstruct_lim,
                                              lbounds, langles,
                                              lis_limited, lnvert, lsz_lim,
                                              lstruct_polyx, lstruct_polyy,
                                              lstruct_normx, lstruct_normy,
                                              eps_uz, eps_vz, eps_a, eps_b,
                                              eps_plane,
                                              is_out_struct,
                                              ind_los_range,
                                              kpin_loc, kpout_loc,
                                              loc_dir, loc_org, loc_vp,
                                              ind_loc,
                                              invr_ray, last_pout, lim_ves,
                                              sign_ray)

        free(loc_org)
        free(loc_dir)
        free(loc_vp)
        free(kpin_loc)
        free(kpout_loc)
        free(ind_loc)
        if is_out_struct:
            free(last_pout)
            free(lim_ves)
            free(invr_ray)
            free(sign_ray)


    else:
        with nogil, parallel(num_threads=num_threads):
            # == Allocating loop  variables ====================================

            # We use local arrays for each thread
            loc_org   = <double *> malloc(sizeof(double) * 3)
            loc_dir   = <double *> malloc(sizeof(double) * 3)
            loc_vp    = <double *> malloc(sizeof(double) * 3)
            kpin_loc  = <double *> malloc(sizeof(double) * 1)
            kpout_loc = <double *> malloc(sizeof(double) * 1)
            ind_loc   = <int *> malloc(sizeof(int) * 1)
            if is_out_struct:
                # if the structure is "out" (solid) we need more arrays
                last_pout = <double *> malloc(sizeof(double) * 3)
                invr_ray  = <double *> malloc(sizeof(double) * 3)
                lim_ves   = <double *> malloc(sizeof(double) * 2)
                sign_ray  = <int *> malloc(sizeof(int) * 3)

            # == Iterating over the LOS in parallel ============================
            for ind_los_prange in prange(num_los):
                raytracing_inout_struct_tor_inomp(num_los, ray_vdir, ray_orig,
                                                  coeff_inter_out,
                                                  coeff_inter_in,
                                                  vperp_out, lstruct_nlim,
                                                  ind_inter_out,
                                                  forbid0, forbidbis_org,
                                                  rmin, rmin2, crit2_base,
                                                  nstruct_lim,
                                                  lbounds, langles,
                                                  lis_limited, lnvert, lsz_lim,
                                                  lstruct_polyx, lstruct_polyy,
                                                  lstruct_normx, lstruct_normy,
                                                  eps_uz, eps_vz, eps_a, eps_b,
                                                  eps_plane,
                                                  is_out_struct,
                                                  ind_los_prange,
                                                  kpin_loc, kpout_loc,
                                                  loc_dir, loc_org, loc_vp,
                                                  ind_loc,
                                                  invr_ray, last_pout, lim_ves,
                                                  sign_ray)

            free(loc_org)
            free(loc_dir)
            free(loc_vp)
            free(kpin_loc)
            free(kpout_loc)
            free(ind_loc)
            if is_out_struct:
                free(last_pout)
                free(lim_ves)
                free(invr_ray)
                free(sign_ray)

    return


cdef inline void raytracing_inout_struct_tor_inomp(const int num_los,
                                                   const double[:,::1] ray_vdir,
                                                   const double[:,::1] ray_orig,
                                                   double[::1] coeff_inter_out,
                                                   double[::1] coeff_inter_in,
                                                   double[::1] vperp_out,
                                                   const long* lstruct_nlim,
                                                   int[::1] ind_inter_out,
                                                   const bint forbid0,
                                                   const bint forbidbis_org,
                                                   const double rmin,
                                                   const double rmin2,
                                                   const double crit2_base,
                                                   const int nstruct_lim,
                                                   const double* lbounds,
                                                   const double* langles,
                                                   const int* lis_limited,
                                                   const long* lnvert,
                                                   const long* lsz_lim,
                                                   const double* lstruct_polyx,
                                                   const double* lstruct_polyy,
                                                   const double* lstruct_normx,
                                                   const double* lstruct_normy,
                                                   const double eps_uz,
                                                   const double eps_vz,
                                                   const double eps_a,
                                                   const double eps_b,
                                                   const double eps_plane,
                                                   const bint is_out_struct,
                                                   const int ind_los,
                                                   double* kpin_loc,
                                                   double* kpout_loc,
                                                   double* loc_dir,
                                                   double* loc_org,
                                                   double* loc_vp,
                                                   int* ind_loc,
                                                   double* invr_ray,
                                                   double* last_pout,
                                                   double* lim_ves,
                                                   int* sign_ray) nogil:

    cdef double upscaDp=0., upar2=0., dpar2=0., crit2=0., idpar2=0.
    cdef double dist = 0., s1x = 0., s1y = 0., s2x = 0., s2y = 0.
    cdef double lim_min=0., lim_max=0., invuz=0.
    cdef int totnvert=0
    cdef int nvert
    cdef int ind_struct, ind_bounds
    cdef int ii, jj
    cdef bint lim_is_none
    cdef bint found_new_kout
    cdef bint inter_bbox
    cdef bint forbidbis = forbidbis_org

    ind_struct = 0
    loc_org[0] = ray_orig[0, ind_los]
    loc_org[1] = ray_orig[1, ind_los]
    loc_org[2] = ray_orig[2, ind_los]
    loc_dir[0] = ray_vdir[0, ind_los]
    loc_dir[1] = ray_vdir[1, ind_los]
    loc_dir[2] = ray_vdir[2, ind_los]
    loc_vp[0] = 0.
    loc_vp[1] = 0.
    loc_vp[2] = 0.
    if is_out_struct:
        # if structure is of "Out" type, then we compute the last
        # point where it went out of a structure.
        ind_loc[0] = ind_inter_out[2+3*ind_los]
        kpin_loc[0] = coeff_inter_out[ind_los]
        last_pout[0] = kpin_loc[0] * loc_dir[0] + loc_org[0]
        last_pout[1] = kpin_loc[0] * loc_dir[1] + loc_org[1]
        last_pout[2] = kpin_loc[0] * loc_dir[2] + loc_org[2]
        compute_inv_and_sign(loc_dir, sign_ray, invr_ray)
    else:
        ind_loc[0] = 0
        kpin_loc[0] = 0
        kpout_loc[0] = 0

    # -- Computing values that depend on the LOS/ray ---------------------------
    upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
    upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
    dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
    idpar2 = 1./dpar2
    invuz = 1./loc_dir[2]
    crit2 = upar2*crit2_base

    # -- Prepare in case forbid is True ----------------------------------------
    if forbid0 and not dpar2>0:
        forbidbis = 0
    if forbidbis:
        # Compute coordinates of the 2 points where the tangents touch
        # the inner circle
        dist = c_sqrt(dpar2-rmin2)
        s1x = (rmin2 * loc_org[0] + rmin * loc_org[1] * dist) * idpar2
        s1y = (rmin2 * loc_org[1] - rmin * loc_org[0] * dist) * idpar2
        s2x = (rmin2 * loc_org[0] - rmin * loc_org[1] * dist) * idpar2
        s2y = (rmin2 * loc_org[1] + rmin * loc_org[0] * dist) * idpar2

    # == Case "OUT" structure ==================================================
    if is_out_struct:
        # We work on each structure
        for ii in range(nstruct_lim):
            # -- Getting structure's data --------------------------------------
            if ii == 0:
                nvert = lnvert[0]
                totnvert = 0
            else:
                totnvert = lnvert[ii-1]
                nvert = lnvert[ii] - totnvert
            ind_struct = lsz_lim[ii]
            # -- Working on the structure limited ------------------------------
            for jj in range(lstruct_nlim[ii]):
                lim_min = langles[(ind_struct+jj)*2]
                lim_max = langles[(ind_struct+jj)*2 + 1]
                lim_is_none = lis_limited[ind_struct+jj] == 1
                # We test if it is really necessary to compute the inter
                # ie. we check if the ray intersects the bounding box

                inter_bbox = inter_ray_aabb_box(sign_ray, invr_ray,
                                                &lbounds[(ind_struct
                                                          + jj)*6],
                                                loc_org, True)
                if not inter_bbox:
                    continue
                # We check that the bounding box is not "behind"
                # the last POut encountered
                inter_bbox = inter_ray_aabb_box(sign_ray, invr_ray,
                                                &lbounds[(ind_struct
                                                          + jj)*6],
                                                last_pout, False)
                if inter_bbox:
                    continue
                 # Else, we compute the new values
                found_new_kout \
                    = comp_inter_los_vpoly(loc_org,
                                           loc_dir,
                                           &lstruct_polyx[totnvert],
                                           &lstruct_polyy[totnvert],
                                           &lstruct_normx[totnvert-ii],
                                           &lstruct_normy[totnvert-ii],
                                           nvert-1,
                                           lim_is_none,
                                           lim_min, lim_max,
                                           forbidbis,
                                           upscaDp, upar2,
                                           dpar2, invuz,
                                           s1x, s1y,
                                           s2x, s2y,
                                           crit2, eps_uz,
                                           eps_vz, eps_a,
                                           eps_b, eps_plane,
                                           False,
                                           kpin_loc,
                                           kpout_loc,
                                           ind_loc,
                                           loc_vp)
                if found_new_kout:
                    coeff_inter_out[ind_los] = kpin_loc[0]
                    vperp_out[0+3*ind_los] = loc_vp[0]
                    vperp_out[1+3*ind_los] = loc_vp[1]
                    vperp_out[2+3*ind_los] = loc_vp[2]
                    ind_inter_out[2+3*ind_los] = ind_loc[0]
                    ind_inter_out[0+3*ind_los] = 1+ii
                    ind_inter_out[1+3*ind_los] = jj
                    last_pout[0] = (coeff_inter_out[ind_los] *
                                    loc_dir[0]) + loc_org[0]
                    last_pout[1] = (coeff_inter_out[ind_los] *
                                    loc_dir[1]) + loc_org[1]
                    last_pout[2] = (coeff_inter_out[ind_los] *
                                    loc_dir[2]) + loc_org[2]
                    if lim_is_none:
                        break
    else:
        # == Case "IN" structure ===============================================
        # Nothing to do but compute intersection between vessel and LOS
        found_new_kout = comp_inter_los_vpoly(loc_org, loc_dir,
                                              lstruct_polyx,
                                              lstruct_polyy,
                                              lstruct_normx,
                                              lstruct_normy,
                                              nstruct_lim,
                                              lis_limited[0],
                                              langles[0], langles[1],
                                              forbidbis,
                                              upscaDp, upar2,
                                              dpar2, invuz,
                                              s1x, s1y, s2x, s2y,
                                              crit2, eps_uz, eps_vz,
                                              eps_a,eps_b, eps_plane,
                                              True,
                                              kpin_loc, kpout_loc,
                                              ind_loc, loc_vp)
        if found_new_kout:
            coeff_inter_in[ind_los]  = kpin_loc[0]
            coeff_inter_out[ind_los] = kpout_loc[0]
            ind_inter_out[2+3*ind_los] = ind_loc[0]
            ind_inter_out[0+3*ind_los] = 0
            ind_inter_out[1+3*ind_los] = 0
            vperp_out[0+3*ind_los] = loc_vp[0]
            vperp_out[1+3*ind_los] = loc_vp[1]
            vperp_out[2+3*ind_los] = loc_vp[2]

        else:
            coeff_inter_in[ind_los]  = C_NAN
            coeff_inter_out[ind_los] = C_NAN
            ind_inter_out[2+3*ind_los] = 0
            ind_inter_out[0+3*ind_los] = 0
            ind_inter_out[1+3*ind_los] = 0
            vperp_out[0+3*ind_los] = 0.
            vperp_out[1+3*ind_los] = 0.
            vperp_out[2+3*ind_los] = 0.
    # end case IN/OUT

    return


# ------------------------------------------------------------------
cdef inline bint comp_inter_los_vpoly(const double[3] ray_orig,
                                      const double[3] ray_vdir,
                                      const double* lpolyx,
                                      const double* lpolyy,
                                      const double* normx,
                                      const double* normy,
                                      const int nvert,
                                      const bint lim_is_none,
                                      const double lim_min,
                                      const double lim_max,
                                      const bint forbidbis,
                                      const double upscaDp, const double upar2,
                                      const double dpar2, const double invuz,
                                      const double s1x,   const double s1y,
                                      const double s2x, const double s2y,
                                      const double crit2, const double eps_uz,
                                      const double eps_vz, const double eps_a,
                                      const double eps_b, const double eps_pln,
                                      const bint is_in_struct,
                                      double[1] kpin_loc, double[1] kpout_loc,
                                      int[1] ind_loc, double[3] vperpin) nogil:
    """
    Computes the entry and exit point of ONE provided LOS/rays for a single
    structure that can be of type "OUT" (is_out_struct=True) or "IN"
    (is_out_struct=False). An "OUT" structure cannot be penetrated whereas an
    "IN" structure can. The latter is typically a vessel and are toroidally
    continous. If a structure is limited we can determine the number of limits
    and the limits itself. For optimization reasons we will also pass the
    bounding box limits. And the information of the last intersected point, if
    any.

    Params
    ======
    ray_vdir : (3) double array
       LOS normalized direction vector
    ray_orig : (3) double array
       LOS origin points coordinates
    lpolyx : (ntotnvert)
       List of "x" coordinates of the polygon's vertices of the structures on
       the poloidal plane
    lpolyy : (ntotnvert)
       List of "y" coordinates of the polygon's vertices of the structures on
       the poloidal plane
    normx : (2, num_vertex-1) double array
       List of "x" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by lpolyx/y
    normy : (2, num_vertex-1) double array
       List of "y" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by lpolyxy
    nvert : int
       Number of vertices on the polygon
    lim_is_none : bint
       Bool to know if the structures (or the vessel) is limited or not.
    lim_min : double
       Minimum angle where the structure lives. If the structure
       is toroidallykpin_loc continous then lim_min = 0
    lim_max : double
       Maximum angle where the structure lives. If the structure
       is toroidally continous then lim_min = 0
    forbidbis: bint
       Should we forbid values behind visible radius for each LOS ?
    upscaDp: double
       Scalar product between LOS' origin and direction
    upar2 : double
       Norm direction LOS
    dpar2 : double
       Norm origin LOS
    invuz : double
       Inverse of 3rd component of direction. ie. if direction is (ux, uy, uz)
       then invuz = 1/uz
    s1x, s1y, s2x, s2y : double
       Compute coordinates of the 2 points where the tangents touch the inner
       circle of the Tore, only needed if forbidbis = 0
    crit2 : double
       Critical value to evaluate for each LOS if horizontal or not
    eps<val> : double
       Small value, acceptance of error
    is_in_struct : bint
       Bool to determine if the structure is "OUT" or "IN". An "OUT" structure
       cannot be penetrated whereas an "IN" structure can. The latter is
       typically a vessel and are toroidally continous.
    kpout_loc : double array <INOUT>
       Coefficient of exit (kout) of the last point of intersection for the LOS
       with the structure or vessel
    kpin_loc : double array <INOUT>
       Coefficient of exit (kin) of the last point of intersection for the LOS
       with the structure or vessel
    vperpin : (3) double array <INOUT>
       Coordinates of the normal vector of impact of the LOS (0 if none)
    Return
    ======
    bool : If true, there was in impact
           If false, no intersection between LOS and structure
    """
    cdef int jj
    cdef int done=0
    cdef int indin=0
    cdef int indout=0
    cdef bint inter_bbox
    cdef double kout, kin
    cdef double res_kin = kpin_loc[0]
    cdef double res_kout = kpout_loc[0]
    cdef double sca=0., sca0=0., sca1=0., sca2=0.
    cdef double q, coeff, delta, sqd, k, sol0, sol1, phi=0.
    cdef double v0, v1, val_a, val_b, ephi_in0, ephi_in1
    cdef double sout1, sout0
    cdef double sin1, sin0
    cdef double invupar2
    cdef double cosl0, cosl1, sinl0, sinl1

    # -- Computing some seful values -------------------------------------------
    cosl0 = c_cos(lim_min)
    cosl1 = c_cos(lim_max)
    sinl0 = c_sin(lim_min)
    sinl1 = c_sin(lim_max)
    invupar2 = 1./upar2
    # == Compute all solutions =================================================
    # Set tolerance value for ray_vdir[2,ii]
    # eps_uz is the tolerated DZ across 20m (max Tokamak size)
    kout, kin, done = 1.e12, 1.e12, 0
    if ray_vdir[2] * ray_vdir[2] < crit2:
        # -- Case with horizontal semi-line ------------------------------------
        for jj in range(nvert):
            # Solutions exist only in the case with non-horizontal
            # segment (i.e.: cone, not plane)
            if (lpolyy[jj+1] - lpolyy[jj])**2 > eps_vz * eps_vz:
                q = (ray_orig[2] - lpolyy[jj]) / (lpolyy[jj+1] - lpolyy[jj])
                # The intersection must stand on the segment
                if q>=0 and q<1:
                    coeff = q * q * (lpolyx[jj+1]-lpolyx[jj])**2 + \
                        2. * q * lpolyx[jj] * (lpolyx[jj+1] - lpolyx[jj]) + \
                        lpolyx[jj] * lpolyx[jj]
                    delta = upscaDp * upscaDp - upar2 * (dpar2 - coeff)
                    if delta>0.:
                        sqd = c_sqrt(delta)
                        # The intersection must be on the semi-line (i.e.: k>=0)
                        # First solution
                        if -upscaDp - sqd >= 0:
                            k = (-upscaDp - sqd) * invupar2
                            sol0 = ray_orig[0] + k * ray_vdir[0]
                            sol1 = ray_orig[1] + k * ray_vdir[1]
                            if forbidbis:
                                sca0 = (sol0-s1x)*ray_orig[0] + \
                                       (sol1-s1y)*ray_orig[1]
                                sca1 = (sol0-s1x)*s1x + (sol1-s1y)*s1y
                                sca2 = (sol0-s2x)*s2x + (sol1-s2y)*s2y
                            if not forbidbis or (forbidbis and not
                                                 (sca0<0 and sca1<0 and
                                                  sca2<0)):
                                # Get the normalized perpendicular vector
                                # at intersection
                                phi = c_atan2(sol1, sol0)
                                # Check sol inside the Lim
                                if lim_is_none or (not lim_is_none and
                                                   ((lim_min<lim_max and
                                                     lim_min<=phi and
                                                     phi<=lim_max)
                                                    or (lim_min>lim_max and
                                                        (phi>=lim_min or
                                                         phi<=lim_max)))):
                                    # Get the scalar product to determine
                                    # entry or exit point
                                    sca = c_cos(phi)*normx[jj]*ray_vdir[0] + \
                                          c_sin(phi)*normx[jj]*ray_vdir[1] + \
                                          normy[jj]*ray_vdir[2]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        done = 1
                                        indout = jj
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj

                        # Second solution
                        if -upscaDp + sqd >=0:
                            k = (-upscaDp + sqd)*invupar2
                            sol0 = ray_orig[0] + k * ray_vdir[0]
                            sol1 = ray_orig[1] + k * ray_vdir[1]
                            if forbidbis:
                                sca0 = (sol0-s1x) * ray_orig[0] + \
                                       (sol1-s1y) * ray_orig[1]
                                sca1 = (sol0-s1x) * s1x + (sol1-s1y) * s1y
                                sca2 = (sol0-s2x) * s2x + (sol1-s2y) * s2y
                            if not forbidbis or (forbidbis and not
                                                 (sca0<0 and sca1<0 and
                                                  sca2<0)):
                                # Get the normalized perpendicular vector
                                # at intersection
                                phi = c_atan2(sol1,sol0)
                                if lim_is_none or (not lim_is_none and
                                                   ((lim_min<lim_max and
                                                     lim_min<=phi and
                                                     phi<=lim_max) or
                                                    (lim_min>lim_max and
                                                     (phi>=lim_min or
                                                      phi<=lim_max))
                                                   )):
                                    # Get the scalar product to determine
                                    # entry or exit point
                                    sca = c_cos(phi)*normx[jj]*ray_vdir[0] + \
                                          c_sin(phi)*normx[jj]*ray_vdir[1] + \
                                          normy[jj]*ray_vdir[2]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        done = 1
                                        indout = jj
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
    else:
        # == More general non-horizontal semi-line case ========================
        for jj in range(nvert):
            v0 = lpolyx[jj+1]-lpolyx[jj]
            v1 = lpolyy[jj+1]-lpolyy[jj]
            val_a = v0 * v0 - upar2 * v1 * invuz * v1 * invuz
            val_b = lpolyx[jj] * v0 + v1 * (ray_orig[2] - lpolyy[jj]) * upar2 *\
                    invuz * invuz - upscaDp * v1 * invuz
            coeff = - upar2 * (ray_orig[2] - lpolyy[jj])**2 * invuz * invuz +\
                    2. * upscaDp * (ray_orig[2]-lpolyy[jj]) * invuz -\
                    dpar2 + lpolyx[jj] * lpolyx[jj]
            if ((val_a * val_a < eps_a * eps_a) and
                (val_b * val_b > eps_b * eps_b)):
                q = -coeff / (2. * val_b)
                if q >= 0. and q < 1.:
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k >= 0:
                        sol0 = ray_orig[0] + k * ray_vdir[0]
                        sol1 = ray_orig[1] + k * ray_vdir[1]
                        if forbidbis:
                            sca0 = (sol0-s1x)*ray_orig[0] + \
                                   (sol1-s1y)*ray_orig[1]
                            sca1 = (sol0-s1x)*s1x + (sol1-s1y)*s1y
                            sca2 = (sol0-s2x)*s2x + (sol1-s2y)*s2y
                            if sca0<0 and sca1<0 and sca2<0:
                                continue
                        # Get the normalized perpendicular vect at intersection
                        phi = c_atan2(sol1,sol0)
                        if lim_is_none or (not lim_is_none and
                                           ((lim_min < lim_max and
                                             lim_min <= phi and
                                             phi <= lim_max) or
                                            (lim_min > lim_max and
                                             (phi >= lim_min or
                                              phi <= lim_max)))):
                            # Get the scal prod to determine entry or exit point
                            sca = c_cos(phi) * normx[jj] * ray_vdir[0] + \
                                  c_sin(phi) * normx[jj] * ray_vdir[1] + \
                                  normy[jj] * ray_vdir[2]
                            if sca<=0 and k<kout:
                                kout = k
                                done = 1
                                indout = jj
                            elif sca>=0 and k<min(kin,kout):
                                kin = k
                                indin = jj
            elif ((val_a * val_a >= eps_a * eps_a) and
                  (val_b * val_b > val_a * coeff)):
                sqd = c_sqrt(val_b * val_b - val_a * coeff)
                # First solution
                q = (-val_b + sqd) / val_a
                if q >= 0. and q < 1.:
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k >= 0.:
                        sol0 = ray_orig[0] + k * ray_vdir[0]
                        sol1 = ray_orig[1] + k * ray_vdir[1]
                        if forbidbis:
                            sca0 = (sol0-s1x) * ray_orig[0] + \
                                   (sol1-s1y) * ray_orig[1]
                            sca1 = (sol0-s1x) * s1x + (sol1-s1y) * s1y
                            sca2 = (sol0-s2x) * s2x + (sol1-s2y) * s2y
                        if not forbidbis or (forbidbis and
                                             not (sca0<0 and sca1<0 and
                                                  sca2<0)):
                            # Get the normalized perpendicular vector at inter
                            phi = c_atan2(sol1, sol0)
                            if lim_is_none or (not lim_is_none and
                                               ((lim_min < lim_max and
                                                 lim_min <= phi and
                                                 phi <= lim_max) or
                                                (lim_min > lim_max and
                                                 (phi >= lim_min or
                                                  phi <= lim_max)))):
                                # Get the scal prod to determine in or out point
                                sca = c_cos(phi) * normx[jj] * ray_vdir[0] + \
                                      c_sin(phi) * normx[jj] * ray_vdir[1] + \
                                      normy[jj] * ray_vdir[2]
                                if sca<=0 and k<kout:
                                    kout = k
                                    done = 1
                                    indout = jj
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj

                # == Second solution ===========================================
                q = (-val_b - sqd) / val_a
                if q >= 0. and q < 1.:
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k>=0.:
                        sol0 = ray_orig[0] + k * ray_vdir[0]
                        sol1 = ray_orig[1] + k * ray_vdir[1]
                        if forbidbis:
                            sca0 = (sol0-s1x) * ray_orig[0] + \
                                   (sol1-s1y) * ray_orig[1]
                            sca1 = (sol0-s1x) * s1x + (sol1-s1y) * s1y
                            sca2 = (sol0-s2x) * s2x + (sol1-s2y) * s2y
                        if not forbidbis or (forbidbis and
                                             not (sca0<0 and sca1<0 and
                                                  sca2<0)):
                            # Get the normalized perpendicular vector at inter
                            phi = c_atan2(sol1,sol0)
                            if lim_is_none or (not lim_is_none and
                                               ((lim_min < lim_max and
                                                 lim_min <= phi and
                                                 phi <= lim_max) or
                                                (lim_min>lim_max and
                                                 (phi>=lim_min or
                                                  phi<=lim_max)))):
                                # Get the scal prod to determine if in or out
                                sca = c_cos(phi) * normx[jj] * ray_vdir[0] + \
                                      c_sin(phi) * normx[jj] * ray_vdir[1] + \
                                      normy[jj] * ray_vdir[2]
                                if sca<=0 and k<kout:
                                    kout = k
                                    done = 1
                                    indout = jj
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj

    if not lim_is_none:
        ephi_in0 = -sinl0
        ephi_in1 =  cosl0
        if c_abs(ray_vdir[0] * ephi_in0 + ray_vdir[1] * ephi_in1) > eps_pln:
            k = -(ray_orig[0] * ephi_in0 + ray_orig[1] * ephi_in1) \
                /(ray_vdir[0] * ephi_in0 + ray_vdir[1] * ephi_in1)
            if k >= 0:
                # Check if in ves_poly
                sol0 = (ray_orig[0] + k * ray_vdir[0]) * cosl0 + \
                       (ray_orig[1] + k * ray_vdir[1]) * sinl0
                sol1 =  ray_orig[2] + k * ray_vdir[2]
                inter_bbox = is_point_in_path(nvert, lpolyx, lpolyy, sol0, sol1)
                if inter_bbox:
                    # Check PIn (POut not possible for limited torus)
                    sca = ray_vdir[0] * ephi_in0 + ray_vdir[1] * ephi_in1
                    if sca<=0 and k<kout:
                        kout = k
                        done = 1
                        indout = -1
                    elif sca>=0 and k<min(kin,kout):
                        kin = k
                        indin = -1

        ephi_in0 =  sinl1
        ephi_in1 = -cosl1
        if c_abs(ray_vdir[0] * ephi_in0 + ray_vdir[1] * ephi_in1) > eps_pln:
            k = -(ray_orig[0] * ephi_in0 + ray_orig[1] * ephi_in1)\
                /(ray_vdir[0] * ephi_in0 + ray_vdir[1] * ephi_in1)
            if k >= 0:
                sol0 = (ray_orig[0] + k * ray_vdir[0]) * cosl1 +\
                       (ray_orig[1] + k * ray_vdir[1]) * sinl1
                sol1 =  ray_orig[2] + k * ray_vdir[2]
                # Check if in ves_poly
                inter_bbox = is_point_in_path(nvert, lpolyx, lpolyy, sol0, sol1)
                if inter_bbox:
                    # Check PIn (POut not possible for limited torus)
                    sca = ray_vdir[0]*ephi_in0 + ray_vdir[1]*ephi_in1
                    if sca<=0 and k<kout:
                        kout = k
                        done = 1
                        indout = -2
                    elif sca>=0 and k<min(kin,kout):
                        kin = k
                        indin = -2
    # == Analyzing if there was impact =========================================
    if done==1:
        if is_in_struct :
            kpout_loc[0] = kout
            if indout==-1:
                vperpin[0] = -sinl0
                vperpin[1] = cosl0
                vperpin[2] = 0.
            elif indout==-2:
                vperpin[0] = sinl1
                vperpin[1] = -cosl1
                vperpin[2] = 0.
            else:
                sout0 = ray_orig[0] + kout * ray_vdir[0]
                sout1 = ray_orig[1] + kout * ray_vdir[1]
                phi = c_atan2(sout1, sout0)
                vperpin[0] = c_cos(phi) * normx[indout]
                vperpin[1] = c_sin(phi) * normx[indout]
                vperpin[2] = normy[indout]
            ind_loc[0] = indout
            if kin<kout:
                kpin_loc[0] = kin
        elif kin < kout and kin < res_kin:
            kpin_loc[0] = kin
            if indin==-1:
                vperpin[0] = sinl0
                vperpin[1] = -cosl0
                vperpin[2] = 0.
            elif indin==-2:
                vperpin[0] = -sinl1
                vperpin[1] = cosl1
                vperpin[2] = 0.
            else:
                sin0 = ray_orig[0] + kin * ray_vdir[0]
                sin1 = ray_orig[1] + kin * ray_vdir[1]
                phi = c_atan2(sin1,sin0)
                vperpin[0] = -c_cos(phi) * normx[indin]
                vperpin[1] = -c_sin(phi) * normx[indin]
                vperpin[2] = -normy[indin]
            ind_loc[0] = indin
    return (res_kin != kpin_loc[0]) or (res_kout != kpout_loc[0]
                                        and is_in_struct)

# ==============================================================================
# =  Raytracing on a Cylinder (Linear case)
# ==============================================================================
cdef inline void raytracing_inout_struct_lin(const int Nl,
                                             const double[:,::1] Ds,
                                             const double [:,::1] us,
                                             const int Ns,
                                             const double* polyx_tab,
                                             const double* polyy_tab,
                                             const double* normx_tab,
                                             const double* normy_tab,
                                             const double L0,
                                             const double L1,
                                             double[::1] kin_tab,
                                             double[::1] kout_tab,
                                             double[::1] vperpout_tab,
                                             int[::1] indout_tab,
                                             const double EpsPlane,
                                             const int ind_struct,
                                             const int ind_lim_struct) nogil:

    cdef bint is_in_path
    cdef int ii=0, jj=0
    cdef double kin, kout, scauVin, q, X, sca, k
    cdef int indin=0, indout=0, Done=0

    if ind_struct == 0 and ind_lim_struct == 0 :
        # If it is the first struct,
        # we have to initialize values even if no impact
        kin_tab[ii]  = C_NAN
        kout_tab[ii] = C_NAN

    for ii in range(0,Nl):
        kout, kin, Done = 1.e12, 1e12, 0
        # For cylinder
        for jj in range(0,Ns):
            scauVin = us[1,ii] * normx_tab[jj] + us[2,ii] * normy_tab[jj]
            # Only if plane not parallel to line
            if c_abs(scauVin)>EpsPlane:
                k = -( (Ds[1,ii] - polyx_tab[jj]) * normx_tab[jj] +
                       (Ds[2,ii] - polyy_tab[jj]) * normy_tab[jj]) \
                       / scauVin
                # Only if on good side of semi-line
                if k>=0.:
                    V1 = polyx_tab[jj+1]-polyx_tab[jj]
                    V2 = polyy_tab[jj+1]-polyy_tab[jj]
                    if (V1**2 + V2**2 > _VSMALL):
                        q = (  (Ds[1,ii] + k * us[1,ii] - polyx_tab[jj]) * V1
                             + (Ds[2,ii] + k * us[2,ii] - polyy_tab[jj]) * V2) \
                             / (V1*V1 + V2*V2)
                        # Only of on the fraction of plane
                        if q>=0. and q<1.:
                            X = Ds[0,ii] + k*us[0,ii]
                            # Only if within limits
                            if X>=L0 and X<=L1:
                                sca = us[1,ii] * normx_tab[jj] \
                                      + us[2,ii] * normy_tab[jj]
                                # Only if new
                                if sca<=0 and k<kout:
                                    kout = k
                                    indout = jj
                                    Done = 1
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj
                    else:
                        with gil:
                            from warnings import warn
                            warn("The polygon has double identical points",
                                 Warning)
        # For two faces
        # Only if plane not parallel to line
        if c_abs(us[0,ii])>EpsPlane:
            # First face
            k = -(Ds[0,ii]-L0)/us[0,ii]
            # Only if on good side of semi-line
            if k>=0.:
                # Only if inside VPoly
                is_in_path = is_point_in_path(Ns, polyx_tab, polyy_tab,
                                              Ds[1,ii]+k*us[1,ii],
                                              Ds[2,ii]+k*us[2,ii])
                if is_in_path:
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
                is_in_path = is_point_in_path(Ns, polyx_tab, polyy_tab,
                                              Ds[1,ii]+k*us[1,ii],
                                              Ds[2,ii]+k*us[2,ii])
                if is_in_path:
                    if us[0,ii]>=0 and k<kout:
                        kout = k
                        indout = -2
                        Done = 1
                    elif us[0,ii]<=0 and k<min(kin,kout):
                        kin = k
                        indin = -2
        # == Analyzing if there was impact ====================================
        if Done==1:
            if (ind_struct == 0 and ind_lim_struct == 0):
                kout_tab[ii] = kout
                if kin < kout:
                    kin_tab[ii] = kin
                # To be finished
                if indout==-1:
                    vperpout_tab[0 + 3 * ii] = 1.
                    vperpout_tab[1 + 3 * ii] = 0.
                    vperpout_tab[2 + 3 * ii] = 0.
                elif indout==-2:
                    vperpout_tab[0 + 3 * ii] = -1.
                    vperpout_tab[1 + 3 * ii] = 0.
                    vperpout_tab[2 + 3 * ii] = 0.
                else:
                    vperpout_tab[0 + 3 * ii] = 0.
                    vperpout_tab[1 + 3 * ii] = normx_tab[indout]
                    vperpout_tab[2 + 3 * ii] = normy_tab[indout]
                indout_tab[0 + 3 * ii] = 0
                indout_tab[1 + 3 * ii] = 0
                indout_tab[2 + 3 * ii] = indout
            elif kin<kout_tab[ii] and kin < kout:
                kout_tab[ii] = kin
                indout_tab[0 + 3 * ii] = ind_struct
                indout_tab[1 + 3 * ii] = ind_lim_struct
                indout_tab[2 + 3 * ii] = indin
                if indout==-1:
                    vperpout_tab[0 + 3 * ii] = 1.
                    vperpout_tab[1 + 3 * ii] = 0.
                    vperpout_tab[2 + 3 * ii] = 0.
                elif indout==-2:
                    vperpout_tab[0 + 3 * ii] = -1.
                    vperpout_tab[1 + 3 * ii] = 0.
                    vperpout_tab[2 + 3 * ii] = 0.
                else:
                    vperpout_tab[0 + 3 * ii] = 0.
                    vperpout_tab[1 + 3 * ii] = normx_tab[indout]
                    vperpout_tab[2 + 3 * ii] = normy_tab[indout]
    return


cdef inline void compute_inout_tot(const int num_los,
                                   const int npts_poly,
                                   const double[:, ::1] ray_orig,
                                   const double[:, ::1] ray_vdir,
                                   const double[:, ::1] ves_poly,
                                   const double[:, ::1] ves_norm,
                                   const long[::1] lstruct_nlim_org,
                                   const double[::1] ves_lims,
                                   const double[::1] lstruct_polyx,
                                   const double[::1] lstruct_polyy,
                                   const double[::1] lstruct_lims,
                                   const double[::1] lstruct_normx,
                                   const double[::1] lstruct_normy,
                                   const long[::1] lnvert,
                                   const int nstruct_tot,
                                   const int nstruct_lim,
                                   const int sz_ves_lims,
                                   const double min_poly_r,
                                   const double rmin_org,
                                   const double eps_uz, const double eps_a,
                                   const double eps_vz, const double eps_b,
                                   const double eps_plane, bint is_tor,
                                   const bint forbid, const int num_threads,
                                   double[::1] coeff_inter_out,
                                   double[::1] coeff_inter_in,
                                   double[::1] vperp_out,
                                   int[::1] ind_inter_out) nogil:
    cdef int ii, jj
    cdef int ind_struct = 0
    cdef int len_lim
    cdef int ind_min
    cdef int nvert
    cdef int lsl_ind = 0
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef double lim_min = 0.
    cdef double lim_max = 0.
    cdef double rmin = rmin_org
    cdef double rmin2 = 0.
    cdef bint forbidbis, forbid0
    cdef double *lbounds = <double *>malloc(nstruct_tot * 6 * sizeof(double))
    cdef double *langles = <double *>malloc(nstruct_tot * 2 * sizeof(double))
    cdef int *llimits = NULL
    cdef long *lsz_lim = NULL
    cdef long* lstruct_nlim = NULL
    cdef int[1] llim_ves
    cdef double[2] lbounds_ves
    cdef double[2] lim_ves
    # ==========================================================================
    if is_tor:
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None or sz_ves_lims == 0:
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
            llim_ves[0] = 1
        else:
            lbounds_ves[0] = c_atan2(c_sin(ves_lims[0]), c_cos(ves_lims[0]))
            lbounds_ves[1] = c_atan2(c_sin(ves_lims[1]), c_cos(ves_lims[1]))
            llim_ves[0] = 0
        # -- Toroidal case -----------------------------------------------------
        # rmin is necessary to avoid looking on the other side of the tokamak
        if rmin < 0.:
            rmin = 0.95*min(min_poly_r,
                            _bgt.comp_min_hypot(ray_orig[0, ...],
                                                ray_orig[1, ...],
                                                num_los))
        rmin2 = rmin*rmin
        # Variable to avoid looking "behind" blind spot of tore
        if forbid:
            forbid0, forbidbis = 1, 1
        else:
            forbid0, forbidbis = 0, 0

        # -- Computing intersection between LOS and Vessel ---------------------
        raytracing_inout_struct_tor(num_los, ray_vdir, ray_orig,
                                    coeff_inter_out, coeff_inter_in,
                                    vperp_out, NULL, ind_inter_out,
                                    forbid0, forbidbis,
                                    rmin, rmin2, crit2_base,
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
            lstruct_nlim = <long *>malloc(nstruct_lim * sizeof(long))
            for ii in range(nstruct_lim):
                # We get the number of vertices and limits of the struct's poly
                if ii == 0:
                    lsz_lim[0] = 0
                    nvert = lnvert[0]
                    ind_min = 0
                else:
                    nvert = lnvert[ii] - lnvert[ii - 1]
                    lsz_lim[ii] = lstruct_nlim[ii-1] + lsz_lim[ii-1]
                    ind_min = lnvert[ii-1]
                # For fast accessing
                len_lim = lstruct_nlim_org[ii]

                # We get the limits if any
                lstruct_nlim[ii] = lstruct_nlim_org[ii]
                if len_lim == 0:
                    lstruct_nlim[ii] = lstruct_nlim[ii] + 1
                    # computing structure bounding box (no limits)
                    llimits[ind_struct] = 1 # True : is continous
                    comp_bbox_poly_tor(nvert,
                                       &lstruct_polyx[ind_min],
                                       &lstruct_polyy[ind_min],
                                       &lbounds[ind_struct*6])
                    langles[ind_struct*2] = 0
                    langles[ind_struct*2 + 1] = 0
                    ind_struct = 1 + ind_struct
                    lsl_ind += 1
                    lim_min = 0.
                    lim_max = 0.
                elif len_lim == 1:
                    # computing structure bounding box (just one lim)
                    lim_ves[0] = lstruct_lims[lsl_ind]
                    lim_ves[1] = lstruct_lims[lsl_ind + 1]
                    lsl_ind += 2
                    llimits[ind_struct] = 0 # False : struct is limited
                    lim_min = c_atan2(c_sin(lim_ves[0]), c_cos(lim_ves[0]))
                    lim_max = c_atan2(c_sin(lim_ves[1]), c_cos(lim_ves[1]))
                    comp_bbox_poly_tor_lim(nvert,
                                           &lstruct_polyx[ind_min],
                                           &lstruct_polyy[ind_min],
                                           &lbounds[ind_struct*6],
                                           lim_min, lim_max)
                    langles[ind_struct*2] = lim_min
                    langles[ind_struct*2 + 1] = lim_max
                    ind_struct = 1 + ind_struct
                else:
                    # loop over the limits (one continous structure)
                    for jj in range(len_lim):
                        # We compute the structure's bounding box:
                        lim_ves[0] = lstruct_lims[lsl_ind]
                        lim_ves[1] = lstruct_lims[lsl_ind + 1]
                        lsl_ind += 2
                        llimits[ind_struct] = 0 # False : struct is limited
                        lim_min = c_atan2(c_sin(lim_ves[0]), c_cos(lim_ves[0]))
                        lim_max = c_atan2(c_sin(lim_ves[1]), c_cos(lim_ves[1]))
                        comp_bbox_poly_tor_lim(nvert,
                                               &lstruct_polyx[ind_min],
                                               &lstruct_polyy[ind_min],
                                               &lbounds[ind_struct*6],
                                               lim_min, lim_max)
                        langles[ind_struct*2] = lim_min
                        langles[ind_struct*2 + 1] = lim_max
                        ind_struct = 1 + ind_struct
            # end loops over structures
            # -- Computing intersection between structures and LOS -------------
            raytracing_inout_struct_tor(num_los, ray_vdir, ray_orig,
                                        coeff_inter_out, coeff_inter_in,
                                        vperp_out, &lstruct_nlim[0],
                                        ind_inter_out,
                                        forbid0, forbidbis,
                                        rmin, rmin2, crit2_base,
                                        nstruct_lim,
                                        lbounds, langles, llimits,
                                        &lnvert[0], lsz_lim,
                                        &lstruct_polyx[0],
                                        &lstruct_polyy[0],
                                        &lstruct_normx[0],
                                        &lstruct_normy[0],
                                        eps_uz, eps_vz, eps_a,
                                        eps_b, eps_plane,
                                        num_threads,
                                        True) # the structure is "OUT"

            free(lsz_lim)
            free(llimits)
    else:
        # -- Cylindrical case --------------------------------------------------
        # .. if there are, we get the limits for the vessel ....................
        if ves_lims is None  or sz_ves_lims == 0:
            lbounds_ves[0] = 0
            lbounds_ves[1] = 0
        else:
            lbounds_ves[0] = ves_lims[0]
            lbounds_ves[1] = ves_lims[1]

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
                if ii == 0:
                    nvert = lnvert[0]
                    ind_min = 0
                else:
                    nvert = lnvert[ii] - lnvert[ii - 1]
                    ind_min = lnvert[ii-1]
                # getting number limits
                len_lim = lstruct_nlim_org[ii]
                # We get the limits if any
                if len_lim == 0:
                    raytracing_inout_struct_lin(num_los, ray_orig, ray_vdir,
                                                nvert-1,
                                                &lstruct_polyx[ind_min],
                                                &lstruct_polyy[ind_min],
                                                &lstruct_normx[ind_min-ii],
                                                &lstruct_normy[ind_min-ii],
                                                lbounds_ves[0],
                                                lbounds_ves[1],
                                                coeff_inter_in,
                                                coeff_inter_out,
                                                vperp_out, ind_inter_out,
                                                eps_plane, ii+1, 0)
                    lsl_ind += 1
                elif len_lim == 1:
                    lbounds_ves[0] = lstruct_lims[lsl_ind]
                    lbounds_ves[1] = lstruct_lims[lsl_ind + 1]
                    lsl_ind += 2
                    raytracing_inout_struct_lin(num_los, ray_orig, ray_vdir,
                                                nvert-1,
                                                &lstruct_polyx[ind_min],
                                                &lstruct_polyy[ind_min],
                                                &lstruct_normx[ind_min-ii],
                                                &lstruct_normy[ind_min-ii],
                                                lbounds_ves[0],
                                                lbounds_ves[1],
                                                coeff_inter_in,
                                                coeff_inter_out,
                                                vperp_out, ind_inter_out,
                                                eps_plane, ii+1, 0)

                else:
                    # and loop over the limits (one continous structure)
                    for jj in range(len_lim):
                        lbounds_ves[0] = lstruct_lims[lsl_ind]
                        lbounds_ves[1] = lstruct_lims[lsl_ind + 1]
                        lsl_ind += 2
                        raytracing_inout_struct_lin(num_los, ray_orig, ray_vdir,
                                                    nvert-1,
                                                    &lstruct_polyx[ind_min],
                                                    &lstruct_polyy[ind_min],
                                                    &lstruct_normx[ind_min-ii],
                                                    &lstruct_normy[ind_min-ii],
                                                    lbounds_ves[0],
                                                    lbounds_ves[1],
                                                    coeff_inter_in,
                                                    coeff_inter_out,
                                                    vperp_out, ind_inter_out,
                                                    eps_plane, ii+1, jj)
    free(lbounds)
    free(langles)
    return


# ==============================================================================
# =  Raytracing on a Torus only KMin and KMax
# ==============================================================================
cdef inline void raytracing_minmax_struct_tor(const int num_los,
                                              const double[:,::1] ray_vdir,
                                              const double[:,::1] ray_orig,
                                              double* coeff_inter_out,
                                              double* coeff_inter_in,
                                              const bint forbid0,
                                              const bint forbidbis_org,
                                              const double rmin, double rmin2,
                                              const double crit2_base,
                                              const int npts_poly,
                                              const double* langles,
                                              const bint is_limited,
                                              const double* surf_polyx,
                                              const double* surf_polyy,
                                              const double* surf_normx,
                                              const double* surf_normy,
                                              const double eps_uz,
                                              const double eps_vz,
                                              const double eps_a,
                                              const double eps_b,
                                              const double eps_plane,
                                              const int num_threads) nogil:
    """
    Computes the entry and exit point of all provided LOS/rays for a set of
    "IN" structures in a TORE. A "in" structure is typically a vessel, or
    flux surface and are (noramally) toroidally continous but you can specify
    if it is otherwise with lis_limited and langles.
    This functions is parallelized.

    Params
    ======
    num_los : int
       Total number of lines of sight (LOS) (aka. rays)
    ray_vdir : (3, num_los) double array
       LOS normalized direction vector
    ray_orig : (3, num_los) double array
       LOS origin points coordinates
    coeff_inter_out : (num_los*num_surf) double array <INOUT>
       Coefficient of exit (kout) of the last point of intersection for each LOS
       with the global geometry (with ALL structures)
    coeff_inter_in : (num_los*num_surf) double array <INOUT>
       Coefficient of entry (kin) of the last point of intersection for each LOS
       with the global geometry (with ALL structures). If intersection at origin
       k = 0, if no intersection k = NAN
    forbid0 : bool
       Should we forbid values behind visible radius ? (see Rmin). If false,
       will test "hidden" part always, else, it will depend on the LOS and
       on forbidbis.
    forbidbis: bint
       Should we forbid values behind visible radius for each LOS ?
    rmin : double
       Minimal radius of vessel to take into consideration
    rmin2 : double
       Squared valued of the minimal radius
    crit2_base : double
       Critical value to evaluate for each LOS if horizontal or not
    npts_poly : int
       Number of OUT structures (not counting the limited versions).
       If not is_out_struct then length of vpoly.
    langles : (2 * nstruct) double array
       Minimum and maximum angles where the structure lives. If the structure
       number 'i' is toroidally continous then langles[i:i+2] = [0, 0].
    is_limited : bint
       bool to know if the flux surface is limited or not
    surf_polyx : (ntotnvert)
       List of "x" coordinates of the polygon's vertices on
       the poloidal plane
    surf_polyy : (ntotnvert)
       List of "y" coordinates of the polygon's vertices on
       the poloidal plane
    surf_normx : (2, num_vertex-1) double array
       List of "x" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by surf_poly
    surf_normy : (2, num_vertex-1) double array
       List of "y" coordinates of the normal vectors going "inwards" of the
        edges of the Polygon defined by surf_poly
    eps<val> : double
       Small value, acceptance of error
    num_threads : int
       The num_threads argument indicates how many threads the team should
       consist of. If not given, OpenMP will decide how many threads to use.
       Typically this is the number of cores available on the machine.
    """
    cdef double upscaDp=0., upar2=0., dpar2=0., crit2=0., idpar2=0.
    cdef double dist = 0., s1x = 0., s1y = 0., s2x = 0., s2y = 0.
    cdef double invuz=0.
    cdef int ind_struct
    cdef int ind_los
    cdef bint found_new_kout
    cdef bint forbidbis = forbidbis_org
    cdef double[3] dummy
    cdef int[1] silly
    cdef double* kpout_loc = NULL
    cdef double* kpin_loc = NULL
    cdef double* loc_org = NULL
    cdef double* loc_dir = NULL

    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
        # We use local arrays for each thread so
        loc_org   = <double *> malloc(sizeof(double) * 3)
        loc_dir   = <double *> malloc(sizeof(double) * 3)
        kpin_loc  = <double *> malloc(sizeof(double) * 1)
        kpout_loc = <double *> malloc(sizeof(double) * 1)
        # == The parallelization over the LOS ==================================
        for ind_los in prange(num_los, schedule='dynamic'):
            ind_struct = 0
            loc_org[0] = ray_orig[0, ind_los]
            loc_org[1] = ray_orig[1, ind_los]
            loc_org[2] = ray_orig[2, ind_los]
            loc_dir[0] = ray_vdir[0, ind_los]
            loc_dir[1] = ray_vdir[1, ind_los]
            loc_dir[2] = ray_vdir[2, ind_los]
            kpout_loc[0] = 0
            kpin_loc[0] = 0
            # -- Computing values that depend on the LOS/ray -------------------
            upscaDp = loc_dir[0]*loc_org[0] + loc_dir[1]*loc_org[1]
            upar2   = loc_dir[0]*loc_dir[0] + loc_dir[1]*loc_dir[1]
            dpar2   = loc_org[0]*loc_org[0] + loc_org[1]*loc_org[1]
            idpar2 = 1./dpar2
            invuz = 1./loc_dir[2]
            crit2 = upar2*crit2_base

            # -- Prepare in case forbid is True --------------------------------
            if forbid0 and not dpar2>0:
                forbidbis = 0
            if forbidbis:
                # Compute coordinates of the 2 points where the tangents touch
                # the inner circle
                dist = c_sqrt(dpar2-rmin2)
                s1x = (rmin2 * loc_org[0] + rmin * loc_org[1] * dist) * idpar2
                s1y = (rmin2 * loc_org[1] - rmin * loc_org[0] * dist) * idpar2
                s2x = (rmin2 * loc_org[0] - rmin * loc_org[1] * dist) * idpar2
                s2y = (rmin2 * loc_org[1] + rmin * loc_org[0] * dist) * idpar2

            # == Case "IN" structure =======================================
            # Nothing to do but compute intersection between vessel and LOS
            found_new_kout = comp_inter_los_vpoly(loc_org, loc_dir,
                                                  surf_polyx,
                                                  surf_polyy,
                                                  surf_normx,
                                                  surf_normy,
                                                  npts_poly-1,
                                                  not is_limited,
                                                  langles[0], langles[1],
                                                  forbidbis,
                                                  upscaDp, upar2,
                                                  dpar2, invuz,
                                                  s1x, s1y, s2x, s2y,
                                                  crit2, eps_uz, eps_vz,
                                                  eps_a,eps_b, eps_plane,
                                                  True,
                                                  kpin_loc, kpout_loc,
                                                  silly, dummy)
            if found_new_kout:
                coeff_inter_in[ind_los]  = kpin_loc[0]
                coeff_inter_out[ind_los] = kpout_loc[0]
            else:
                coeff_inter_in[ind_los]  = C_NAN
                coeff_inter_out[ind_los] = C_NAN
        free(loc_org)
        free(loc_dir)
        free(kpin_loc)
        free(kpout_loc)
    return


# ==============================================================================
# =  Raytracing on a Cylinder only KMin and KMax
# ==============================================================================
cdef inline void raytracing_minmax_struct_lin(const int Nl,
                                              const double[:,::1] Ds,
                                              const double [:,::1] us,
                                              const int Ns,
                                              const double* polyx_tab,
                                              const double* polyy_tab,
                                              const double* normx_tab,
                                              const double* normy_tab,
                                              const double L0,
                                              const double L1,
                                              double* kin_tab,
                                              double* kout_tab,
                                              const double EpsPlane) nogil:
    cdef bint is_in_path
    cdef int ii=0, jj=0
    cdef double kin, kout, scauVin, q, X, sca, k
    cdef int indin=0, indout=0, done=0

    kin_tab[ii]  = C_NAN
    kout_tab[ii] = C_NAN

    for ii in range(0,Nl):
        kout = 1.e12
        kin  = 1.e12
        done = 0
        # For cylinder
        for jj in range(0,Ns):
            scauVin = us[1,ii] * normx_tab[jj] + us[2,ii] * normy_tab[jj]
            # Only if plane not parallel to line
            if c_abs(scauVin)>EpsPlane:
                k = -( (Ds[1,ii] - polyx_tab[jj]) * normx_tab[jj] +
                       (Ds[2,ii] - polyy_tab[jj]) * normy_tab[jj]) \
                       / scauVin
                # Only if on good side of semi-line
                if k>=0.:
                    V1 = polyx_tab[jj+1]-polyx_tab[jj]
                    V2 = polyy_tab[jj+1]-polyy_tab[jj]
                    q = (  (Ds[1,ii] + k * us[1,ii] - polyx_tab[jj]) * V1
                         + (Ds[2,ii] + k * us[2,ii] - polyy_tab[jj]) * V2) \
                         / (V1*V1 + V2*V2)
                    # Only of on the fraction of plane
                    if q>=0. and q<1.:
                        X = Ds[0,ii] + k*us[0,ii]

                        # Only if within limits
                        if X>=L0 and X<=L1:
                            sca = us[1,ii] * normx_tab[jj] \
                                  + us[2,ii] * normy_tab[jj]
                            # Only if new
                            if sca<=0 and k<kout:
                                kout = k
                                indout = jj
                                done = 1
                            elif sca>=0 and k<min(kin,kout):
                                kin = k
                                indin = jj

        # For two faces
        # Only if plane not parallel to line
        if c_abs(us[0,ii])>EpsPlane:
            # First face
            k = -(Ds[0,ii]-L0)/us[0,ii]
            # Only if on good side of semi-line
            if k>=0.:
                # Only if inside VPoly
                is_in_path = is_point_in_path(Ns, polyx_tab, polyy_tab,
                                              Ds[1,ii]+k*us[1,ii],
                                              Ds[2,ii]+k*us[2,ii])
                if is_in_path:
                    if us[0,ii]<=0 and k<kout:
                        kout = k
                        indout = -1
                        done = 1
                    elif us[0,ii]>=0 and k<min(kin,kout):
                        kin = k
                        indin = -1
            # Second face
            k = -(Ds[0,ii]-L1)/us[0,ii]
            # Only if on good side of semi-line
            if k>=0.:
                # Only if inside VPoly
                is_in_path = is_point_in_path(Ns, polyx_tab, polyy_tab,
                                              Ds[1,ii]+k*us[1,ii],
                                              Ds[2,ii]+k*us[2,ii])
                if is_in_path:
                    if us[0,ii]>=0 and k<kout:
                        kout = k
                        indout = -2
                        done = 1
                    elif us[0,ii]<=0 and k<min(kin,kout):
                        kin = k
                        indin = -2
        # == Analyzing if there was impact ====================================
        if done==1:
            kout_tab[ii] = kout
            if kin<kin_tab[ii]:
                kin_tab[ii] = kin
    return


# ==============================================================================
# = Checking if points are visible
# ==============================================================================
cdef inline void is_visible_pt_vec(double pt0, double pt1, double pt2,
                                   double[:, ::1] pts, int npts,
                                   double[:, ::1] ves_poly,
                                   double[:, ::1] ves_norm,
                                   long* is_vis,
                                   double[::1] dist,
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
                                   double eps_plane, bint is_tor,
                                   bint forbid,
                                   int num_threads):
    cdef array vperp_out = clone(array('d'), npts * 3, True)
    cdef array coeff_inter_in  = clone(array('d'), npts, True)
    cdef array coeff_inter_out = clone(array('d'), npts, True)
    cdef array ind_inter_out = clone(array('i'), npts * 3, True)
    cdef int sz_ves_lims = np.size(ves_lims)
    cdef int npts_poly = ves_norm.shape[1]
    cdef double[:, ::1] ray_orig = view.array(shape=(3,npts),
                                              itemsize=sizeof(double),
                                              format="d")
    cdef double[:, ::1] ray_vdir = view.array(shape=(3,npts),
                                              itemsize=sizeof(double),
                                              format="d")
    # --------------------------------------------------------------------------
    if dist is None:
        is_visible_pt_vec_core_nd(pt0, pt1, pt2,
                                  pts, npts,
                                  ves_poly, ves_norm,
                                  is_vis, ves_lims,
                                  lstruct_nlim,
                                  lstruct_polyx, lstruct_polyy,
                                  lstruct_lims,
                                  lstruct_normx, lstruct_normy,
                                  lnvert, vperp_out,
                                  coeff_inter_in, coeff_inter_out,
                                  ind_inter_out, sz_ves_lims,
                                  ray_orig, ray_vdir, npts_poly,
                                  nstruct_tot, nstruct_lim,
                                  rmin, eps_uz, eps_a, eps_vz, eps_b,
                                  eps_plane, is_tor,
                                  forbid, num_threads)
    else:
        is_visible_pt_vec_core(pt0, pt1, pt2,
                               pts, npts,
                               ves_poly, ves_norm,
                               is_vis, &dist[0], ves_lims,
                               lstruct_nlim,
                               lstruct_polyx, lstruct_polyy,
                               lstruct_lims,
                               lstruct_normx, lstruct_normy,
                               lnvert, vperp_out,
                               coeff_inter_in, coeff_inter_out,
                               ind_inter_out, sz_ves_lims,
                               ray_orig, ray_vdir, npts_poly,
                               nstruct_tot, nstruct_lim,
                               rmin, eps_uz, eps_a, eps_vz, eps_b,
                               eps_plane, is_tor,
                               forbid, num_threads)

    return


cdef inline void is_visible_pt_vec_core(double pt0, double pt1, double pt2,
                                        double[:, ::1] pts, int npts,
                                        double[:, ::1] ves_poly,
                                        double[:, ::1] ves_norm,
                                        long* is_vis,
                                        double* dist,
                                        double[::1] ves_lims,
                                        long[::1] lstruct_nlim,
                                        double[::1] lstruct_polyx,
                                        double[::1] lstruct_polyy,
                                        double[::1] lstruct_lims,
                                        double[::1] lstruct_normx,
                                        double[::1] lstruct_normy,
                                        long[::1] lnvert,
                                        # results:
                                        double[::1] vperp_out,
                                        double[::1] coeff_inter_in,
                                        double[::1] coeff_inter_out,
                                        int[::1] ind_inter_out,
                                        int sz_ves_lims,
                                        double[:, ::1] ray_orig,
                                        double[:, ::1] ray_vdir,
                                        int npts_poly,
                                        # ...
                                        int nstruct_tot,
                                        int nstruct_lim,
                                        double rmin,
                                        double eps_uz, double eps_a,
                                        double eps_vz, double eps_b,
                                        double eps_plane, bint is_tor,
                                        bint forbid,
                                        int num_threads) nogil:
    cdef double min_poly_r
    # --------------------------------------------------------------------------
    # Initialization : creation of the rays between points pts and P
    _bgt.tile_3_to_2d(pt0, pt1, pt2, npts, ray_orig)
    _bgt.compute_diff_div(pts, ray_orig, &dist[0], npts, ray_vdir)
    # --------------------------------------------------------------------------
    min_poly_r = _bgt.comp_min(ves_poly[0, ...], npts_poly-1)
    compute_inout_tot(npts, npts_poly,
                      ray_orig, ray_vdir,
                      ves_poly, ves_norm,
                      lstruct_nlim, ves_lims,
                      lstruct_polyx, lstruct_polyy,
                      lstruct_lims, lstruct_normx,
                      lstruct_normy, lnvert,
                      nstruct_tot, nstruct_lim,
                      sz_ves_lims, min_poly_r, rmin,
                      eps_uz, eps_a, eps_vz, eps_b,
                      eps_plane, is_tor,
                      forbid, num_threads,
                      coeff_inter_out, coeff_inter_in, vperp_out,
                      ind_inter_out)
    # --------------------------------------------------------------------------
    # Get ind
    is_vis_mask(is_vis, dist, coeff_inter_out, npts, num_threads)

    return


cdef inline void is_visible_pt_vec_core_nd(double pt0, double pt1, double pt2,
                                           double[:, ::1] pts, int npts,
                                           double[:, ::1] ves_poly,
                                           double[:, ::1] ves_norm,
                                           long* is_vis,
                                           double[::1] ves_lims,
                                           long[::1] lstruct_nlim,
                                           double[::1] lstruct_polyx,
                                           double[::1] lstruct_polyy,
                                           double[::1] lstruct_lims,
                                           double[::1] lstruct_normx,
                                           double[::1] lstruct_normy,
                                           long[::1] lnvert,
                                           # results:
                                           double[::1] vperp_out,
                                           double[::1] coeff_inter_in,
                                           double[::1] coeff_inter_out,
                                           int[::1] ind_inter_out,
                                           int sz_ves_lims,
                                           double[:, ::1] ray_orig,
                                           double[:, ::1] ray_vdir,
                                           int npts_poly,
                                           # ...
                                           int nstruct_tot,
                                           int nstruct_lim,
                                           double rmin,
                                           double eps_uz, double eps_a,
                                           double eps_vz, double eps_b,
                                           double eps_plane, bint is_tor,
                                           bint forbid,
                                           int num_threads) nogil:
    """
    Same as `is_visible_pt_vec_core` but when distance is not given.
    """
    cdef double* dist_arr = NULL
    cdef double min_poly_r
    # --------------------------------------------------------------------------
    # Initialization : creation of the rays between points pts and P
    _bgt.tile_3_to_2d(pt0, pt1, pt2, npts, ray_orig)
    dist_arr = <double*> malloc(npts*sizeof(double))
    _bgt.compute_dist_pt_vec(pt0, pt1, pt2, npts, pts, &dist_arr[0])
    _bgt.compute_diff_div(pts, ray_orig, dist_arr, npts, ray_vdir)
    # --------------------------------------------------------------------------
    min_poly_r = _bgt.comp_min(ves_poly[0, ...], npts_poly-1)
    compute_inout_tot(npts, npts_poly,
                      ray_orig, ray_vdir,
                      ves_poly, ves_norm,
                      lstruct_nlim, ves_lims,
                      lstruct_polyx, lstruct_polyy,
                      lstruct_lims, lstruct_normx,
                      lstruct_normy, lnvert,
                      nstruct_tot, nstruct_lim,
                      sz_ves_lims, min_poly_r, rmin,
                      eps_uz, eps_a, eps_vz, eps_b,
                      eps_plane, is_tor,
                      forbid, num_threads,
                      coeff_inter_out, coeff_inter_in, vperp_out,
                      ind_inter_out)
    # --------------------------------------------------------------------------
    # Get ind
    is_vis_mask(is_vis, dist_arr, coeff_inter_out, npts,
                num_threads)
    return


cdef inline void is_vis_mask(long* is_vis, double* dist,
                             double[::1] coeff_inter_out,
                             int npts,
                             int num_threads) nogil:
    cdef int ii
    for ii in range(npts):
        is_vis[ii] = 1
        if dist[ii] > coeff_inter_out[ii]:
            is_vis[ii] = 0
    return


cdef inline void are_visible_vec_vec(double[:, ::1] pts1, int npts1,
                                     double[:, ::1] pts2, int npts2,
                                     double[:, ::1] ves_poly,
                                     double[:, ::1] ves_norm,
                                     long[:, ::1] is_vis,
                                     double[:, ::1] dist,
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
                                     double eps_plane, bint is_tor,
                                     bint forbid,
                                     int num_threads):
    cdef int ii
    cdef array vperp_out = clone(array('d'), npts2 * 3, True)
    cdef array coeff_inter_in  = clone(array('d'), npts2, True)
    cdef array coeff_inter_out = clone(array('d'), npts2, True)
    cdef array ind_inter_out = clone(array('i'), npts2 * 3, True)
    cdef int sz_ves_lims = np.size(ves_lims)
    cdef int npts_poly = ves_norm.shape[1]
    cdef double[:, ::1] ray_orig = view.array(shape=(3,npts2),
                                              itemsize=sizeof(double),
                                              format="d")
    cdef double[:, ::1] ray_vdir = view.array(shape=(3,npts2),
                                              itemsize=sizeof(double),
                                              format="d")
    # We compute for each point in the polygon
    if dist is not None:
        for ii in range(npts1):
            is_visible_pt_vec_core(pts1[0,ii], pts1[1,ii], pts1[2,ii],
                                   pts2, npts2,
                                   ves_poly, ves_norm,
                                   &is_vis[ii, 0], &dist[ii, 0], ves_lims,
                                   lstruct_nlim,
                                   lstruct_polyx, lstruct_polyy,
                                   lstruct_lims,
                                   lstruct_normx, lstruct_normy,
                                   lnvert, vperp_out,
                                   coeff_inter_in, coeff_inter_out,
                                   ind_inter_out, sz_ves_lims,
                                   ray_orig, ray_vdir, npts_poly,
                                   nstruct_tot, nstruct_lim,
                                   rmin, eps_uz, eps_a, eps_vz, eps_b,
                                   eps_plane, is_tor,
                                   forbid, num_threads)
    else:
        for ii in range(npts1):
            is_visible_pt_vec_core_nd(pts1[0,ii], pts1[1,ii], pts1[2,ii],
                                      pts2, npts2,
                                      ves_poly, ves_norm,
                                      &is_vis[ii, 0], ves_lims,
                                      lstruct_nlim,
                                      lstruct_polyx, lstruct_polyy,
                                      lstruct_lims,
                                      lstruct_normx, lstruct_normy,
                                      lnvert, vperp_out,
                                      coeff_inter_in, coeff_inter_out,
                                      ind_inter_out, sz_ves_lims,
                                      ray_orig, ray_vdir, npts_poly,
                                      nstruct_tot, nstruct_lim,
                                      rmin, eps_uz, eps_a, eps_vz, eps_b,
                                      eps_plane, is_tor,
                                      forbid, num_threads)
    return
