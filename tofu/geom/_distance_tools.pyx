# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions related to the distance between a LOS and a VPoly:
#    - Compute distance between LOS and a circle (unit and vectorial)
#    - Test if a LOS and a circle are close (unit and vectorial) to a epsilon
#    - Compute distance between LOS and VPoly (unit and vectorial)
#    - Test if a LOS and a VPoly are close (unit and vectorial) to a epsilon
#    - Compute which LOS is closer to a VPoly
#    - Compute which VPoly is closer to a LOS
################################################################################
cimport cython
from cython.parallel import prange
from cython.parallel cimport parallel
from libc.math cimport fabs as c_abs
from libc.math cimport sqrt as c_sqrt
from libc.math cimport NAN as C_NAN
from libc.stdlib cimport malloc, free
# from libc.stdio cimport printf    # for debug
from ._basic_geom_tools cimport _VSMALL
from . cimport _basic_geom_tools as _bgt
from . cimport _sampling_tools as _st

# ==============================================================================
# == DISTANCE CIRCLE - LOS
# ==============================================================================
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
    cdef double lambd, m2b2, b1sqr, b1, twoThirds, sHat, gHat, cutoff, s
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
        norm_dir = c_sqrt(_bgt.compute_dot_prod(direct, direct))
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
    _bgt.compute_cross_prod(direction, circle_normal, MxN)
    _bgt.compute_cross_prod(D, circle_normal, DxN)
    m0sqr = _bgt.compute_dot_prod(MxN, MxN)

    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        numRoots = 0

        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*direction =
        # (0,b1',b2').
        m0 = c_sqrt(m0sqr)
        rm0 = radius * m0
        lambd = -_bgt.compute_dot_prod(MxN, DxN) / m0sqr
        for i in range(3):
            D[i] += lambd * direction[i]
            DxN[i] += lambd * MxN[i]
        m2b2 = _bgt.compute_dot_prod(direction, D)
        b1sqr = _bgt.compute_dot_prod(DxN, DxN)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = c_sqrt(b1sqr)
            rm0sqr = radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = c_sqrt(c_abs((rm0sqr * b1sqr)**twoThirds - b1sqr)) / m0
                gHat = rm0sqr * sHat / c_sqrt(c_abs(m0sqr * sHat * sHat + b1sqr))
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2)
                        roots[numRoots] = s
                        numRoots += 1
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
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
        _bgt.compute_cross_prod(circle_normal, line_closest, NxDelta)
        if not (c_abs(NxDelta[0]) <= _VSMALL
                and c_abs(NxDelta[1]) <= _VSMALL
                and c_abs(NxDelta[2]) <= _VSMALL):
            norm_ppar = c_sqrt(line_closest[0]*line_closest[0]
                              + line_closest[1]*line_closest[1])
            circle_closest[0] = radius * line_closest[0] / norm_ppar
            circle_closest[1] = radius * line_closest[1] / norm_ppar
            circle_closest[2] = circle_center[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
        else:
            diff[0] = line_closest[0] - radius
            diff[1] = line_closest[1]
            diff[2] = line_closest[2] - circle_center[2]
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
    else:
        # The line direction and the plane normal are parallel.
        # There is only one solution the intersection between line and plane
        if not (c_abs(DxN[0]) <= _VSMALL
                and c_abs(DxN[1]) <= _VSMALL
                and c_abs(DxN[2]) <= _VSMALL):
            # The line is A+t*N but with A != C.
            t = -_bgt.compute_dot_prod(direction, D)
            # We compute line closest
            line_closest[0] = origin[0] + t * direction[0]
            line_closest[1] = origin[1] + t * direction[1]
            line_closest[2] = origin[2] + t * direction[2]
            # We compute cirlce closest
            for i in range(3):
                diff[i] = line_closest[i] - circle_center[i]
            distance = radius / c_sqrt(_bgt.compute_dot_prod(diff, diff))
            circle_closest[0] = line_closest[0] * distance
            circle_closest[1] = line_closest[1] * distance
            circle_closest[2] = circ_z + (line_closest[2] - circ_z) * distance
            if t < 0:
                # fi t is negative, we take origin as closest point
                t = 0.
                line_closest[0] = origin[0]
                line_closest[1] = origin[1]
                line_closest[2] = origin[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
            result[0] = t
            result[1] = distance
        else:
            # The line direction and the normal vector are on the same line
            # so C is the closest point for the circle and the distance is
            # the radius unless the ray's origin is after the circle center
            if (origin[2] * direction[2] <= circle_center[2] * direction[2]) :
                t = c_abs(circle_center[2] - origin[2])
                result[0] = t
                result[1] = radius
            else:
                t = c_abs(circle_center[2] - origin[2])
                result[0] = 0
                result[1] = c_sqrt(radius*radius + t*t)
    result[0] = result[0] * inv_norm_dir
    return

cdef inline void comp_dist_los_circle_vec_core(int num_los, int num_cir,
                                               double* los_directions,
                                               double* los_origins,
                                               double* circle_radius,
                                               double* circle_z,
                                               double* norm_dir_tab,
                                               double[::1] res_k,
                                               double[::1] res_dist,
                                               int num_threads) nogil:
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
    # Python, if you need it from cython, use `dist_los_circle_core`
    """
    cdef int i, ind_los, ind_cir
    cdef double* loc_res
    cdef double* dirv
    cdef double* orig
    cdef double radius, circ_z, norm_dir
    with nogil, parallel(num_threads=num_threads):
        dirv = <double*>malloc(3*sizeof(double))
        orig = <double*>malloc(3*sizeof(double))
        loc_res = <double*>malloc(2*sizeof(double))
        for ind_los in prange(num_los):
            for i in range(3):
                dirv[i] = los_directions[ind_los * 3 + i]
                orig[i] = los_origins[ind_los * 3 + i]
            norm_dir = norm_dir_tab[ind_los]
            if norm_dir < 0.:
                norm_dir = c_sqrt(_bgt.compute_dot_prod(dirv, dirv))
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
# == TEST CLOSENESS CIRCLE - LOS
# ==============================================================================
cdef inline bint is_close_los_circle_core(const double[3] direct,
                                          const double[3] origin,
                                          double radius, double circ_z,
                                          double norm_dir, double eps) nogil:
    # Source: https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
    # The line is P(t) = B+t*M.  The circle is |X-C| = r with Dot(N,X-C)=0.
    cdef int numRoots, i
    cdef double zero = 0., m0sqr, m0, rm0
    cdef double lambd, m2b2, b1sqr, b1, twoThirds, sHat, gHat, cutoff, s
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
        norm_dir = c_sqrt(_bgt.compute_dot_prod(direct, direct))
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
    _bgt.compute_cross_prod(direction, circle_normal, MxN)
    _bgt.compute_cross_prod(D, circle_normal, DxN)
    m0sqr = _bgt.compute_dot_prod(MxN, MxN)

    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        numRoots = 0
        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*direction =
        # (0,b1',b2').
        m0 = c_sqrt(m0sqr)
        rm0 = radius * m0
        lambd = -_bgt.compute_dot_prod(MxN, DxN) / m0sqr
        for i in range(3):
            D[i] += lambd * direction[i]
            DxN[i] += lambd * MxN[i]
        m2b2 = _bgt.compute_dot_prod(direction, D)
        b1sqr = _bgt.compute_dot_prod(DxN, DxN)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = c_sqrt(b1sqr)
            rm0sqr = radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = c_sqrt(c_abs((rm0sqr * b1sqr)**twoThirds - b1sqr)) / m0
                gHat = rm0sqr * sHat / c_sqrt(c_abs(m0sqr * sHat * sHat + b1sqr))
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2)
                        roots[numRoots] = s
                        numRoots += 1
                        s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = _bgt.compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
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
        _bgt.compute_cross_prod(circle_normal, line_closest, NxDelta)
        if not (c_abs(NxDelta[0]) <= _VSMALL
                and c_abs(NxDelta[1]) <= _VSMALL
                and c_abs(NxDelta[2]) <= _VSMALL):
            norm_ppar = c_sqrt(line_closest[0]*line_closest[0]
                              + line_closest[1]*line_closest[1])
            circle_closest[0] = radius * line_closest[0] / norm_ppar
            circle_closest[1] = radius * line_closest[1] / norm_ppar
            circle_closest[2] = circle_center[2]
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
            are_close = distance < eps
            return are_close
        else:
            diff[0] = line_closest[0] + radius
            diff[1] = line_closest[1]
            diff[2] = line_closest[2] - circle_center[2]
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
            are_close = distance < eps
            return are_close
    else:
        # The line direction and the plane normal are parallel.
        # There is only one solution the intersection between line and plane
        if not (c_abs(DxN[0]) <= _VSMALL
                and c_abs(DxN[1]) <= _VSMALL
                and c_abs(DxN[2]) <= _VSMALL):
            # The line is A+t*N but with A != C.
            t = -_bgt.compute_dot_prod(direction, D)
            # We compute line closest
            line_closest[0] = origin[0] + t * direction[0]
            line_closest[1] = origin[1] + t * direction[1]
            line_closest[2] = origin[2] + t * direction[2]
            # We compute cirlce closest
            for i in range(3):
                diff[i] = line_closest[i] - circle_center[i]
            distance = radius / c_sqrt(_bgt.compute_dot_prod(diff, diff))
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
            distance = c_sqrt(_bgt.compute_dot_prod(diff, diff))
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
                t = c_abs(circle_center[2] - origin[2])
                are_close = c_sqrt(radius*radius + t*t) < eps
                return are_close

cdef inline void is_close_los_circle_vec_core(int num_los, int num_cir,
                                              double eps,
                                              double* los_directions,
                                              double* los_origins,
                                              double* circle_radius,
                                              double* circle_z,
                                              double* norm_dir_tab,
                                              int[::1] res,
                                              int num_threads) nogil:
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
    Python, if you need it from cython, use `dist_los_circle_core`
    """
    cdef int i, ind_los, ind_cir
    cdef double* dirv
    cdef double* orig
    cdef double radius, circ_z, norm_dir
    with nogil, parallel(num_threads=num_threads):
        dirv = <double*>malloc(3*sizeof(double))
        orig = <double*>malloc(3*sizeof(double))
        for ind_los in prange(num_los):
            for i in range(3):
                dirv[i] = los_directions[ind_los * 3 + i]
                orig[i] = los_origins[ind_los * 3 + i]
            norm_dir = norm_dir_tab[ind_los]
            if norm_dir < 0.:
                norm_dir = c_sqrt(_bgt.compute_dot_prod(dirv, dirv))
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
# == DISTANCE BETWEEN LOS AND EXT-POLY
# ==============================================================================
cdef inline void comp_dist_los_vpoly_vec_core(int num_poly, int nlos,
                                              double* ray_orig,
                                              double* ray_vdir,
                                              double[:,:,::1] ves_poly,
                                              double eps_uz,
                                              double eps_a,
                                              double eps_vz,
                                              double eps_b,
                                              double eps_plane,
                                              int ves_type,
                                              int algo_type,
                                              double* res_k,
                                              double* res_dist,
                                              double disc_step,
                                              int num_threads) nogil:
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
        algo_type : int
           If algo_type = 0, then simple algo will be used
        ves_type : int
           If ves_type = 1, then geo is TOROIDAL
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
    cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, ind_pol2
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.
    cdef double** list_vpoly_x = NULL
    cdef double** list_vpoly_y = NULL
    cdef int* list_npts = NULL
    # == Tests and warnings ====================================================
    with gil:
        if not algo_type == 0 or not ves_type == 1:
            assert False
        from warnings import warn
        warn("This algo supposes that the polys are nested from inner to outer",
             Warning)
    # == Discretizing vpolys ===================================================
    list_vpoly_x = <double**>malloc(sizeof(double*)*num_poly)
    list_vpoly_y = <double**>malloc(sizeof(double*)*num_poly)
    list_npts = <int*>malloc(sizeof(int)*num_poly)
    for ind_pol in range(num_poly):
        list_vpoly_x[ind_pol] = NULL
        list_vpoly_y[ind_pol] = NULL
        _st.simple_discretize_vpoly_core(ves_poly[ind_pol],
                                        ves_poly[ind_pol].shape[1],
                                        disc_step, # discretization step
                                        &list_vpoly_x[ind_pol],
                                        &list_vpoly_y[ind_pol],
                                        &list_npts[ind_pol],
                                        0, # mode = absolute
                                        _VSMALL)
    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
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
                simple_dist_los_vpoly_core(loc_org, loc_dir,
                                           list_vpoly_x[ind_pol],
                                           list_vpoly_y[ind_pol],
                                           list_npts[ind_pol],
                                           upscaDp,
                                           upar2, dpar2,
                                           invuz, crit2,
                                           eps_uz, eps_vz,
                                           eps_a, eps_b,
                                           loc_res)
                if not res_k == NULL:
                    res_k[ind_los * num_poly + ind_pol] = loc_res[0]
                res_dist[ind_los * num_poly + ind_pol] = loc_res[1]
                if not loc_res[1] == loc_res[1] : #is nan
                    for ind_pol2 in range(ind_pol, num_poly):
                        res_k[ind_los * num_poly + ind_pol2] = C_NAN
                        res_dist[ind_los * num_poly + ind_pol2] = C_NAN
                    continue
        free(loc_dir)
        free(loc_org)
        free(loc_res)
    free(list_vpoly_x)
    free(list_vpoly_y)
    free(list_npts)
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
           inverse of uz (vdir[2])
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
             if the i-th LOS intersects the poly, then distance[i] = C_NAN
    ---
    This is the cython version, only accessible from cython. If you need
    to use it from Python please use: comp_dist_los_vpoly
    """
    cdef int jj
    cdef double norm_dir2, norm_dir2_ori
    cdef double radius_z
    cdef double q = C_NAN, coeff = C_NAN, sqd = C_NAN, k = C_NAN
    cdef double v0, v1, val_a, val_b
    cdef double[2] res_a
    cdef double[2] res_b
    cdef double[3] circle_tangent
    cdef double[3] ray_vdir_norm
    cdef double rdotvec
    res_final[0] = 1000000000
    res_final[1] = 1000000000

    # == Compute all solutions =================================================
    # Set tolerance value for ray_vdir[2,ii]
    # eps_uz is the tolerated DZ across 20m (max Tokamak size)
    norm_dir2 = c_sqrt(_bgt.compute_dot_prod(ray_vdir, ray_vdir))

    norm_dir2_ori = norm_dir2
    for jj in range(3):
        ray_vdir_norm[jj] = ray_vdir[jj] / norm_dir2
    norm_dir2 = 1.

    if ray_vdir_norm[2] * ray_vdir_norm[2] < crit2:
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
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                        lpolyx[jj], lpolyy[jj],
                                        norm_dir2, res_a)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_a)
                else:
                    # The we need to compute the radius (the height is Z_D)
                    # of the circle in the same plane as the LOS and compute the
                    # distance between the LOS and circle.
                    radius_z = q * (lpolyx[jj+1] - lpolyx[jj]) + lpolyx[jj]
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         radius_z, ray_orig[2],
                                         norm_dir2, res_a)
                    if res_a[1] < _VSMALL:
                        # The line is either tangent or intersects the frustum
                        # we need to make the difference
                        k = res_a[0]
                        # we compute the ray from circle center to P
                        circle_tangent[0] = -ray_orig[0] - k * ray_vdir_norm[0]
                        circle_tangent[1] = -ray_orig[1] - k * ray_vdir_norm[1]
                        circle_tangent[2] = 0. # the line is horizontal
                        rdotvec = _bgt.compute_dot_prod(circle_tangent,
                                                        ray_vdir_norm)
                        if c_abs(rdotvec) > _VSMALL:
                            # There is an intersection, distance = C_NAN
                            res_final[1] = C_NAN # distance
                            res_final[0] = C_NAN # k
                            # no need to continue
                            return
                if (res_final[1] - res_a[1] > _VSMALL
                    or (res_final[1] == res_a[1]
                        and res_final[0] - res_a[0] > _VSMALL)):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance
            else:
                # -- case with horizontal cone (aka cone is a plane annulus) ---
                if (c_abs(ray_orig[2] - lpolyy[jj]) > _VSMALL
                    and c_sqrt(dpar2) >= min(lpolyx[jj], lpolyx[jj+1])
                    and c_sqrt(dpar2) <= max(lpolyx[jj], lpolyx[jj+1])):
                    # if ray and annulus are NOT on the same plane:
                    # AND the origin is somewhere in the annulus (on the
                    # X,Y plane), the origin is the closest and distance
                    # is the difference of height
                    if c_abs(ray_orig[2] - lpolyy[jj]) <= res_final[1]:
                        res_final[0] = 0 # k
                        res_final[1] = c_abs(ray_orig[2] - lpolyy[jj])
                else:
                    # Then the shortest distance is the distance to the
                    # outline circles
                    # computing dist to cricle C_A of radius R_A and height Z_A
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj], lpolyy[jj],
                                         norm_dir2, res_a)
                    if res_a[1] < _VSMALL:
                        # The line is either tangent or intersects the frustum
                        # we need to make the difference
                        k = res_a[0]
                        # we compute the ray from circle center to P
                        circle_tangent[0] = -ray_orig[0] - k * ray_vdir_norm[0]
                        circle_tangent[1] = -ray_orig[1] - k * ray_vdir_norm[1]
                        circle_tangent[2] = 0. # the ray is horizontal
                        rdotvec = _bgt.compute_dot_prod(circle_tangent,
                                                        ray_vdir_norm)
                        if c_abs(rdotvec) > _VSMALL:
                            # There is an intersection, distance = C_NAN
                            res_final[1] = C_NAN # distance
                            res_final[0] = C_NAN # k
                            # no need to continue
                            return
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_b)
                    if res_b[1] < _VSMALL:
                        # The line is either tangent or intersects the frustum
                        # we need to make the difference
                        k = res_b[0]
                        # we compute the ray from circle center to P
                        circle_tangent[0] = -ray_orig[0] - k * ray_vdir_norm[0]
                        circle_tangent[1] = -ray_orig[1] - k * ray_vdir_norm[1]
                        circle_tangent[2] = 0. # the ray is horizontal
                        rdotvec = _bgt.compute_dot_prod(circle_tangent,
                                                        ray_vdir_norm)
                        if c_abs(rdotvec) > _VSMALL:
                            # There is an intersection, distance = C_NAN
                            res_final[1] = C_NAN # distance
                            res_final[0] = C_NAN # k
                            # no need to continue
                            return
                    # The result is the one associated to the shortest distance
                    if (res_final[1] - res_a[1] > _VSMALL
                        or (res_final[1] == res_a[1]
                         and res_final[0] - res_a[0] > _VSMALL)):
                        res_final[0] = res_a[0] # k
                        res_final[1] = res_a[1] # distance
                    if (res_final[1] - res_b[1] > _VSMALL
                        or (res_final[1] == res_b[1]
                         and res_final[0] - res_b[0] > _VSMALL)):
                        res_final[0] = res_b[0] # k
                        res_final[1] = res_b[1] # distance

    else:
        # == More general non-horizontal semi-line case ========================
        for jj in range(nvert-1):
            v0 = lpolyx[jj+1]-lpolyx[jj]
            v1 = lpolyy[jj+1]-lpolyy[jj]
            val_a = v0 * v0 - upar2 * v1 * invuz * v1 * invuz
            val_b = lpolyx[jj] * v0 + v1 * invuz * (
                (ray_orig[2] - lpolyy[jj]) * upar2 * invuz - upscaDp)
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
                    elif v0 * v0 < eps_a and upar2 * upar2 < eps_a:
                        # cylinder and vertical line
                        if (ray_orig[2] <= max(lpolyy[jj], lpolyy[jj+1])
                            and ray_orig[2] >= min(lpolyy[jj], lpolyy[jj+1])):
                            # origin of line in the length of cylinder:
                            res_a[0] = 0
                            res_a[1] = c_abs(upar2 - lpolyx[jj])
                        elif (lpolyy[jj] >= ray_orig[2]
                              and ray_orig[2] <= lpolyy[jj+1]):
                            # ray origin below cylinder
                            if ray_vdir_norm[2] < 0:
                                # if the ray is going down origin is the closest
                                res_a[0] = 0
                                res_a[1] = c_sqrt(c_abs(
                                    (lpolyx[jj] - upar2)**2
                                    + (min(lpolyy[jj], lpolyy[jj+1]) - ray_orig[2])**2
                                ))
                            else:
                                res_a[0] = (min(lpolyy[jj], lpolyy[jj+1])
                                            - ray_orig[2]) * invuz
                                res_a[1] = c_abs(lpolyx[jj] - upar2)
                        else:
                            # ray origin above cylinder
                            if ray_vdir_norm[2] > 0:
                                # if the ray is going up origin is the closest
                                res_a[0] = 0
                                res_a[1] = c_sqrt(c_abs(
                                    (lpolyx[jj] - upar2)**2
                                    + (max(lpolyy[jj], lpolyy[jj+1]) - ray_orig[2])**2
                                ))
                            else:
                                res_a[0] = (max(lpolyy[jj], lpolyy[jj+1])
                                            - ray_orig[2]) * invuz
                                res_a[1] = c_abs(lpolyx[jj] - upar2)
                else: # (val_b * val_b > eps_b * eps_b):
                    q = -coeff / (2. * val_b)
                    if q < 0. :
                        # Then we only need to compute distance to circle C_A
                        dist_los_circle_core(ray_vdir_norm, ray_orig,
                                            lpolyx[jj], lpolyy[jj],
                                             norm_dir2, res_a)
                    elif q > 1:
                        # Then we only need to compute distance to circle C_B
                        dist_los_circle_core(ray_vdir_norm, ray_orig,
                                            lpolyx[jj+1], lpolyy[jj+1],
                                             norm_dir2, res_a)
                    else :
                        k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                        if k >= 0.:
                            # Then there is an intersection
                            res_final[0] = C_NAN
                            res_final[1] = C_NAN
                            return # no need to move forward
                        else:
                            # The closest point on the line is the LOS origin
                            res_a[0] = 0
                            res_a[1] = -k * c_sqrt(norm_dir2)

                if (res_final[1] - res_a[1] > _VSMALL
                    or (res_final[1] == res_a[1]
                        and res_final[0] - res_a[0] > _VSMALL)):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance

            elif (val_b * val_b >= val_a * coeff):

                sqd = c_sqrt(c_abs(val_b * val_b - val_a * coeff))
                # First solution
                q = (-val_b + sqd) / val_a
                if q < 0:
                    # Then we only need to compute distance to circle C_A
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj], lpolyy[jj],
                                         norm_dir2, res_a)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_a)
                else :
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k >= 0.:
                        # There is an intersection
                        res_final[0] = C_NAN
                        res_final[1] = C_NAN
                        return # no need to continue
                    else:
                        # The closest point on the LOS is its origin
                        res_a[0] = 0
                        res_a[1] = -k * c_sqrt(norm_dir2)

                if (res_final[1] - res_a[1] > _VSMALL
                    or (res_final[1] == res_a[1]
                        and res_final[0] - res_a[0] > _VSMALL)):
                    res_final[0] = res_a[0] # k
                    res_final[1] = res_a[1] # distance

                # Second solution
                q = (-val_b - sqd) / val_a
                if q < 0:
                    # Then we only need to compute distance to circle C_A
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj], lpolyy[jj],
                                         norm_dir2, res_b)
                elif q > 1:
                    # Then we only need to compute distance to circle C_B
                    dist_los_circle_core(ray_vdir_norm, ray_orig,
                                         lpolyx[jj+1], lpolyy[jj+1],
                                         norm_dir2, res_b)
                else:
                    k = (q * v1 - (ray_orig[2] - lpolyy[jj])) * invuz
                    if k>=0.:
                        # there is an intersection
                        res_final[0] = C_NAN
                        res_final[1] = C_NAN
                        return # no need to continue
                    else:
                        # The closest point on the LOS is its origin
                        res_b[0] = 0
                        res_b[1] = -k * c_sqrt(norm_dir2)

                if (res_final[1] - res_b[1] > _VSMALL
                    or (res_final[1] == res_b[1]
                        and res_final[0] - res_b[0] > _VSMALL)):
                    res_final[0] = res_b[0]
                    res_final[1] = res_b[1]

    res_final[0] = res_final[0] / norm_dir2_ori
    return

# =============================================================================
# == ARE LOS AND EXT-POLY CLOSE
# =============================================================================

cdef inline void is_close_los_vpoly_vec_core(int num_poly, int nlos,
                                             double* ray_orig,
                                             double* ray_vdir,
                                             double[:,:,::1] ves_poly,
                                             double eps_uz,
                                             double eps_a,
                                             double eps_vz,
                                             double eps_b,
                                             double eps_plane,
                                             double epsilon,
                                             int[::1] are_close,
                                             int num_threads) nogil:
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
    cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.

    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
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
# == WHICH LOS/VPOLY IS CLOSER
# ==============================================================================
cdef inline void which_los_closer_vpoly_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 int[::1] ind_close_tab,
                                                 int num_threads) nogil:
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
    cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int ind_los, ind_pol, indloc
    cdef double loc_dist
    cdef double* dist = <double*>malloc(sizeof(double)*num_poly*nlos)

    for indloc in range(num_poly):
        ind_close_tab[indloc] = -1
    comp_dist_los_vpoly_vec_core(num_poly, nlos,
                                 ray_orig,
                                 ray_vdir,
                                 ves_poly,
                                 eps_uz, eps_a,
                                 eps_vz, eps_b,
                                 eps_plane,
                                 True, #ves_type is Tor
                                 0, #algo_type is Simple
                                 NULL, dist,
                                 0.01,
                                 num_threads)

    # We use local arrays for each thread so...
    for ind_pol in range(num_poly):
        loc_dist = 100000000.
        for ind_los in range(nlos):
            if (dist[ind_los*num_poly + ind_pol] < loc_dist):
                ind_close_tab[ind_pol] = ind_los
                loc_dist = dist[ind_los*num_poly + ind_pol]
    free(dist)
    return

cdef inline void which_vpoly_closer_los_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 int[::1] ind_close_tab,
                                                 int num_threads) nogil:
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
    cython, if you need it from Python, use `comp_dist_los_vpoly_vec`
    """
    cdef int i, ind_los, ind_pol, indloc
    cdef int npts_poly
    cdef double* loc_res
    cdef double* loc_dir
    cdef double* loc_org
    cdef double crit2, invuz,  dpar2, upar2, upscaDp
    cdef double crit2_base = eps_uz * eps_uz /400.

    # initialization ...............................................
    for indloc in range(nlos):
        ind_close_tab[indloc] = num_poly-1

    # == Defining parallel part ================================================
    with nogil, parallel(num_threads=num_threads):
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