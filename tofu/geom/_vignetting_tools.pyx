# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for the earclipping techinque for triangulation of a
# polygon. This is really useful when vignetting, as the computation to know
# if a ray intersected a polygon, is easier if we check if the polygon is
# discretized in triangles and then we check if the ray intersected each
# triangle.
################################################################################
cimport cython
from cython.parallel import prange
from cython.parallel cimport parallel
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as c_sqrt
from . cimport _raytracing_tools as _rt
from . cimport _basic_geom_tools as _bgt
from . cimport _chained_list as _cl
from . cimport _sorted_set as _ss


# ===============================================================
#                   Basic utilities:
#                   vector diff
# ===============================================================


cdef inline void compute_diff3d(
    double* orig,
    int nvert,
    double* diff,
) nogil:

    cdef int ivert

    for ivert in range(nvert-1):
        diff[ivert*3 + 0] = orig[0*nvert+(ivert+1)] - orig[0*nvert+ivert]
        diff[ivert*3 + 1] = orig[1*nvert+(ivert+1)] - orig[1*nvert+ivert]
        diff[ivert*3 + 2] = orig[2*nvert+(ivert+1)] - orig[2*nvert+ivert]

    # doing the last point:
    diff[3*(nvert-1) + 0] = orig[0*nvert] - orig[0*nvert+(nvert-1)]
    diff[3*(nvert-1) + 1] = orig[1*nvert] - orig[1*nvert+(nvert-1)]
    diff[3*(nvert-1) + 2] = orig[2*nvert] - orig[2*nvert+(nvert-1)]
    return


cdef inline void compute_diff2d(
    double* orig,
    int nvert,
    double* diff,
) nogil:

    cdef int ivert

    for ivert in range(nvert-1):
        diff[0*nvert + ivert] = orig[0*nvert + (ivert+1)] - orig[0*nvert + ivert]
        diff[1*nvert + ivert] = orig[1*nvert + (ivert+1)] - orig[1*nvert + ivert]

    # doing the last point:
    diff[0*nvert + (nvert-1)] = orig[0*nvert] - orig[0*nvert + (nvert-1)]
    diff[1*nvert + (nvert-1)] = orig[1*nvert] - orig[1*nvert + (nvert-1)]
    return


# ===============================================================
#                   Basic utilities:
#           reflex angle, point in triangle,...
# ===============================================================


cdef inline bint is_reflex_2d(
    const double[1] u0,
    const double[1] u1,
    const double[1] v0,
    const double[1] v1,
) nogil:
    """ test if a point is reflex

    - reflex = angle > pi between vectors

    Assumes:
        - polygon is 3d
        - vect_cc is a normal vector to the polygon quasi-plane P
        - in P, the polygon is counter-clockwise

    Depends on the orientation of vect_cc

    """
    return (u0[0]*v1[0] - u1[0]*v0[0]) < 0.


cdef inline bint is_reflex_3d(
    const double[3] u,
    const double[3] v,
    double[3] vect_cc
) nogil:
    """ test if a point is reflex

    - reflex = angle > pi between vectors

    Assumes:
        - polygon is 3d
        - vect_cc is a normal vector to the polygon quasi-plane P
        - in P, the polygon is counter-clockwise

    Depends on the orientation of vect_cc

    """
    return (
        (u[1]*v[2] - u[2]*v[1]) * vect_cc[0]
        + (u[2]*v[0] - u[0]*v[2]) * vect_cc[1]
        + (u[0]*v[1] - u[1]*v[0]) * vect_cc[2]
    ) < 0.


cdef inline void are_points_reflex_2d(
    int nvert,
    double* diff,
    bint* are_reflex,
) nogil:
    """
    Determines if the interior angles of a polygons are reflex
    (angle > pi) or not.
    """
    cdef int ivert

    # .. Computing if reflex or not ..........................
    for ivert in range(1, nvert):
        are_reflex[ivert] = is_reflex_2d(
            &diff[0*nvert + ivert - 1],         # u0
            &diff[1*nvert + ivert - 1],         # u1
            &diff[0*nvert + ivert],     # v0
            &diff[1*nvert + ivert],     # v1
        )

    # do first point:
    are_reflex[0] = is_reflex_2d(
        &diff[0*nvert + 9],         # u0
        &diff[1*nvert + 9],         # u1
        &diff[0*nvert + 0],     # v0
        &diff[1*nvert + 0],     # v1
    )
    return


cdef inline void are_points_reflex_3d(
    int nvert,
    double* diff,
    bint* are_reflex,
    double[3] vect_cc,
) nogil:
    """
    Determines if the interior angles of a polygons are reflex
    (angle > pi) or not.
    """
    cdef int ivert

    # .. Computing if reflex or not ..........................
    for ivert in range(1, nvert):
        are_reflex[ivert] = is_reflex_3d(
            &diff[ivert*2],
            &diff[(ivert-1)*2],
            vect_cc,
        )

    # do first point:
    are_reflex[0] = is_reflex_3d(
        &diff[0],
        &diff[(nvert-1)*2],
        vect_cc,
    )
    return


# ===============================================================
#                   Basic utilities:
#                   point in triangle,...
# ===============================================================


cdef inline bint is_pt_in_tri_2d(
    double Ax,
    double Ay,
    double Bx,
    double By,
    double Cx,
    double Cy,
    double Px,
    double Py,
) nogil:
    """
    Tests if point P is on the triangle A, B, C such that
        v0 = C - A
        v1 = -B + A
    and A = (Ax, Ay, Az) and P = (px, py, pz)
    """
    cdef bint bA, bB, bC

    # if inside => sign of vector products are all the same
    bA = (Px - Ax)*(By - Ay) - (Py - Ay)*(Bx - Ax) > 0.
    bB = (Px - Bx)*(Cy - By) - (Py - By)*(Cx - Bx) > 0.
    bC = (Px - Cx)*(Ay - Cy) - (Py - Cy)*(Ax - Cx) > 0.

    return (bA == bB and bA == bC)


# Wrong ?
cdef inline bint is_pt_in_tri_3d(double[3] v0, double[3] v1,
                              double Ax, double Ay, double Az,
                              double px, double py, double pz) nogil:
    """
    Tests if point P is on the triangle A, B, C such that
        v0 = C - A
        v1 = -B + A
    and A = (Ax, Ay, Az) and P = (px, py, pz)
    """
    cdef double[3] v2
    cdef double dot00, dot01, dot02
    cdef double dot11, dot12
    cdef double invDenom
    cdef double u, v
    cdef double denom
    # computing vector between A and P
    v2[0] = px - Ax
    v2[1] = py - Ay
    v2[2] = pz - Az
    # compute dot products
    dot00 = _bgt.compute_dot_prod(v0, v0)
    dot01 = -_bgt.compute_dot_prod(v0, v1)
    dot02 = _bgt.compute_dot_prod(v0, v2)
    dot11 = _bgt.compute_dot_prod(v1, v1)
    dot12 = -_bgt.compute_dot_prod(v1, v2)
    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    invDenom = 1. / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)


# ===============================================================
#                   Earclipping method:
#       getting one ear of a poly, triangulate poly, ...
# ===============================================================


cdef inline int get_one_ear(
    double* polygon,
    bint* lref,
    _cl.ChainedList* working_index,
    int nv,
    int nvert,
) nogil:
    """
    A polygon's "ear" is defined as a triangle of vert_i-1, vert_i, vert_i+1,
    points on the polygon, where the point vert_i has the following properties:
        - Its interior angle is not reflex (angle < pi)
        - None of the other vertices of the polygon are in the triangle formed
          by its two annexing points
          Note: only reflex edges can be on the triangle
    polygon : (2*nv) [x0, x1, x2..., x_nv, y0, y1, ... y_nv]
    lref : (nv) [is_reflex(P0), is_reflex(P1), ...]
    working_index : to avoid memory allocation and deallocation, we work with
        with only ONE vector, that allow us to know which of the orignal
        nvert vertices is still being used.
        At the beginning working_index = range(nvert)
        if we took out ONE ear (vertex), for example vertex ii, then:
             nv = nvert - 1
             working_index = [0, ..., ii-1, ii+1, ii+2, ..., nvert]
             and the other tabs are also updated:
                lref = [ .. is_reflex(Pii-1), X, is_reflex(Pii+1),..]
                where X represents values that will never be used !
    """

    # declarations
    cdef int i, j
    cdef int wi, wj
    cdef int wip1, wim1
    cdef bint a_pt_in_tri

    # loop on points to identify an ear
    for i in range(1, nv - 1):
        wi = <int>_cl.get_at_pos(working_index, i)
        if not lref[wi]:
            # angle is not reflex
            a_pt_in_tri = False
            # we get some useful values
            wip1 = <int>_cl.get_at_pos(working_index, i+1)
            wim1 = <int>_cl.get_at_pos(working_index, i-1)
            # We can test if there is another vertex in the 'ear'
            for j in range(nv):
                wj = <int>_cl.get_at_pos(working_index, j)
                # We only test reflex angles, and points that are not
                # edges of the triangle
                if (lref[wj] and wj != wim1 and wj != wip1 and wj != wi):
                    a_pt_in_tri = is_pt_in_tri_2d(
                        # triangle
                        polygon[0*nvert + wim1],
                        polygon[1*nvert + wim1],
                        polygon[0*nvert + wi],
                        polygon[1*nvert + wi],
                        polygon[0*nvert + wip1],
                        polygon[1*nvert + wip1],
                        # point tested
                        polygon[0*nvert + wj],
                        polygon[1*nvert + wj],
                    )
                    if a_pt_in_tri:
                        # point in triangle => not ear, stop test
                        break

            # Let's check if there was a point in the triangle....
            if not a_pt_in_tri:
                return i # if not, we found an ear

    # if we havent returned, either, there was an error somerwhere
    with gil:
        msg = (
            "Got here but shouldn't have\n"
            f"\t- polygon x: {[polygon[0*nvert + ii] for ii in range(nv)]}\n"
            f"\t- polygon y: {[polygon[1*nvert + ii] for ii in range(nv)]}\n"
            f"\t- i: {i} / {nv-1}\n"
            f"\t- j: {j} / {nv}\n"
        )
        raise Exception(msg)

    return -1


cdef inline void earclipping_poly_2d(
    double* vignett,        # 2d polygon coordinates
    long* ltri,             # pre-allocated array of bool (bint)
    double* diff,           # 2d vectors of poolygon edges
    bint* lref,             # array of bool indicating which points are reflex
    int nvert,             # nb of vertices
) nogil:
    """
    Triangulates a polygon by earclipping an edge at a time.
        vignett : (2*nvert) coordinates of 2d poly
        nvert : number of vertices
    Result
        ltri : (3*(nvert-2)) int array, indices of triangles
    """

    # init...
    cdef int nv = nvert
    cdef int itri = 0
    cdef int wi, wim1, wip1, itemp
    cdef int iear
    cdef _cl.ChainedList* working_index

    # trivial case: triangle
    if nvert == 3:
        ltri[0] = 0
        ltri[1] = 1
        ltri[2] = 2
        return

    # .. First computing the edges coodinates .................................
    # .. and checking if the angles defined by the edges are reflex or not.....
    # initialization of working index tab:
    working_index = _cl.create_ordered(nvert)
    # .. Loop ..................................................................
    for itri in range(nvert-3):

        # get first ear
        iear =  get_one_ear(
            vignett,
            &lref[0],
            working_index,
            nv,
            nvert,
        )

        wim1 = <int>_cl.get_at_pos(working_index, iear-1)
        wi   = <int>_cl.get_at_pos(working_index, iear)
        wip1 = <int>_cl.get_at_pos(working_index, iear+1)

        ltri[itri*3]   = wim1
        ltri[itri*3 + 1] = wi
        ltri[itri*3 + 2] = wip1

        # updates on the "information" arrays:
        diff[0*nvert + wim1] = vignett[0*nvert + wip1] - vignett[0*nvert + wim1]
        diff[1*nvert + wim1] = vignett[1*nvert + wip1] - vignett[1*nvert + wim1]

        #... theoritically we should get rid of off diff[wip1] as well but
        # we'll just not use it, however we have to update lref
        # if an angle is not reflex, then it will stay so, only chage if reflex
        if lref[wim1]:
            if iear == 1:
                itemp = <int>_cl.get_at_pos(working_index, nv - 1)
            else:
                itemp = <int>_cl.get_at_pos(working_index, iear - 2)
            lref[wim1] = is_reflex_2d(
                &diff[0*nvert + itemp],
                &diff[1*nvert + itemp],
                &diff[0*nvert + wim1],
                &diff[1*nvert + wim1],
            )

        if lref[wip1]:
            lref[wip1] = is_reflex_2d(
                &diff[0*nvert + wim1],
                &diff[1*nvert + wim1],
                &diff[0*nvert + wip1],
                &diff[1*nvert + wip1],
            )

        # last but not least update on number of vertices and working indices
        nv -= 1
        _cl.pop_at_pos(&working_index, iear)

    # we only have three points left, so that is the last triangle:
    ltri[(itri+1)*3]   = <int>_cl.get_at_pos(working_index, 0)
    ltri[(itri+1)*3+1] = <int>_cl.get_at_pos(working_index, 1)
    ltri[(itri+1)*3+2] = <int>_cl.get_at_pos(working_index, 2)
    _cl.free_cl(&working_index)
    return


# ===============================================================
#
#       Polygons triangulation and Intersection Ray-Poly
#
# ===============================================================


cdef inline void triangulate_poly(
    double* vignett_poly,
    long nvert,
    long** ltri,
) nogil:
    """
    Triangulates a single 3d polygon using the earclipping technique
    https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    Returns
        ltri: 3*(nvert-2) :
            {tri_0_0, tri_0_1, ... tri_0_nvert0}
            where tri_i_j are the 3 indices of the vertex forming a sub-triangle
            on each vertex (-2) and for each vignett
    """

    cdef double* diff = NULL
    cdef bint* lref = NULL

    # ...
    # -- Defining parallel part ------------------------------------------------

    diff = <double*>malloc(3*nvert*sizeof(double))
    lref = <bint*>malloc(nvert*sizeof(bint))
    ltri[0] = <long*>malloc((nvert-2)*3*sizeof(long))

    # diff + reflex
    compute_diff3d(vignett_poly, nvert, &diff[0])
    are_points_reflex_2d(nvert, diff, &lref[0])

    # earclipping
    earclipping_poly_2d(
        vignett_poly,
        &ltri[0][0],
        &diff[0],
        &lref[0],
        nvert,
    )
    return


cdef inline int triangulate_polys(
    double** vignett_poly,
    long* lnvert,
    int nvign,
    long** ltri,
    int num_threads,
) nogil except -1:
    """
    Triangulates a list 3d polygon using the earclipping techinque
    https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    Returns
        ltri: 3*(nvert-2)*nvign :
            = [{tri_0_0, tri_0_1, ... tri_0_nvert0}, ..., {tri_nvign_0, ...}]
            where tri_i_j are the 3 indices of the vertex forming a sub-triangle
            on each vertex (-2) and for each vignett
    """
    cdef int ivign
    cdef int nvert
    cdef double* diff = NULL
    cdef bint* lref = NULL
    # ...
    # -- Defining parallel part ------------------------------------------------
    with nogil, parallel(num_threads=num_threads):
        for ivign in prange(nvign):
            nvert = lnvert[ivign]
            diff = <double*>malloc(3*nvert*sizeof(double))
            lref = <bint*>malloc(nvert*sizeof(bint))
            ltri[ivign] = <long*>malloc((nvert-2)*3*sizeof(long))
            if not diff or not lref or not ltri[ivign]:
                with gil:
                    raise MemoryError()
            try:
                compute_diff2d(vignett_poly[ivign], nvert, &diff[0])
                are_points_reflex_2d(nvert, diff, &lref[0])
                earclipping_poly_2d(vignett_poly[ivign], &ltri[ivign][0],
                                 &diff[0], &lref[0], nvert)
            finally:
                free(diff)
                free(lref)

    return 0


# ===============================================================
#               Vignetting
# ===============================================================


cdef inline bint inter_ray_poly(const double[3] ray_orig,
                                const double[3] ray_vdir,
                                double* vignett,
                                int nvert,
                                long* ltri) nogil:
    cdef int ii, jj
    cdef double[3] pt1
    cdef double[3] pt2
    cdef double[3] pt3
    #...
    for ii in range(nvert-2):
        for jj in range(3):
            pt1[jj] = vignett[ltri[3*ii+0] + jj * nvert]
            pt2[jj] = vignett[ltri[3*ii+1] + jj * nvert]
            pt3[jj] = vignett[ltri[3*ii+2] + jj * nvert]
        if _rt.inter_ray_triangle(ray_orig, ray_vdir, pt1, pt2, pt3):
            return True
    return False


cdef inline void vignetting_core(double[:, ::1] ray_orig,
                                 double[:, ::1] ray_vdir,
                                 double** vignett,
                                 long* lnvert,
                                 double* lbounds,
                                 long** ltri,
                                 int nvign,
                                 int nlos,
                                 bint* goes_through,
                                 int num_threads) nogil:
    cdef int ilos, ivign
    cdef int jj
    cdef int nvert
    cdef bint inter_bbox
    cdef double* loc_org = NULL
    cdef double* loc_dir = NULL
    cdef double* invr_ray = NULL
    cdef int* sign_ray = NULL

    # == Defining parallel part ================================

    with nogil, parallel(num_threads=num_threads):
        # We use local arrays for each thread so
        loc_org   = <double*>malloc(sizeof(double) * 3)
        loc_dir   = <double*>malloc(sizeof(double) * 3)
        invr_ray  = <double*>malloc(sizeof(double) * 3)
        sign_ray  = <int *> malloc(sizeof(int) * 3)
        for ilos in prange(nlos):
            loc_org[0] = ray_orig[0, ilos]
            loc_org[1] = ray_orig[1, ilos]
            loc_org[2] = ray_orig[2, ilos]
            loc_dir[0] = ray_vdir[0, ilos]
            loc_dir[1] = ray_vdir[1, ilos]
            loc_dir[2] = ray_vdir[2, ilos]
            _bgt.compute_inv_and_sign(loc_dir, sign_ray, invr_ray)
            jj = ilos*nvign
            for ivign in range(nvign):
                nvert = lnvert[ivign]
                # -- We check if intersection with  bounding box ---------------
                inter_bbox = _rt.inter_ray_aabb_box(sign_ray, invr_ray,
                                                    &lbounds[6*ivign],
                                                    &loc_org[0],
                                                    countin=True)
                if not inter_bbox:
                    goes_through[ivign + jj] = False
                    continue
                # -- if none, we continue --------------------------------------
                goes_through[ivign + jj] = inter_ray_poly(&loc_org[0],
                                                          &loc_dir[0],
                                                          vignett[ivign],
                                                          nvert,
                                                          ltri[ivign])

        free(loc_org)
        free(loc_dir)
        free(invr_ray)
        free(sign_ray)
    return


# ===============================================================
#               Vignetting Vmesh with VPoly
# ===============================================================


cdef inline int vignetting_vmesh_vpoly(int npts, int sz_r,
                                       bint is_cart,
                                       double[:, ::1] vpoly,
                                       double[:, ::1] pts,
                                       double[::1] vol_resol,
                                       double[::1] r_on_phi,
                                       double* disc_r,
                                       long[::1] lind,
                                       double** res_x,
                                       double** res_y,
                                       double** res_z,
                                       double** res_vres,
                                       double** res_rphi,
                                       long** res_lind,
                                       long* sz_rphi,
                                       int num_threads) nogil:
    # we keep only the points in vpoly
    cdef int ii, jj
    cdef int npts_vpoly
    cdef int nb_in_poly = 0
    cdef double loc_hypot
    cdef _ss.SortedSet* set_r = NULL
    cdef _cl.ChainedList* vec_x = NULL
    cdef _cl.ChainedList* vec_y = NULL
    cdef _cl.ChainedList* vec_z = NULL
    cdef _cl.ChainedList* vec_rphi = NULL
    cdef _cl.ChainedList* vec_vres = NULL
    cdef _cl.ChainedList* vec_lind = NULL
    # -- initialization --------------------------------------------------------
    npts_vpoly = vpoly.shape[1] - 1
    # -- Main loops by case ----------------------------------------------------
    if is_cart:
        for ii in range(npts):
            loc_hypot = c_sqrt(pts[0,ii]*pts[0,ii] + pts[1,ii]*pts[1,ii])
            if _bgt.is_point_in_path(npts_vpoly, &vpoly[0][0], &vpoly[1][0],
                                loc_hypot, pts[2,ii]):
                nb_in_poly += 1
                _cl.push_back(&vec_x, pts[0,ii])
                _cl.push_back(&vec_y, pts[1,ii])
                _cl.push_back(&vec_z, pts[2,ii])
                _cl.push_back(&vec_vres, vol_resol[ii])
                _cl.push_back(&vec_lind, lind[ii])
                # we create a set for the new radius in vpoly:
                _ss.insert(&set_r, loc_hypot)
        # We initialize the arrays:
        res_x[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_y[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_z[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_vres[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_lind[0] = <long*> malloc(nb_in_poly * sizeof(long))
        with nogil, parallel(num_threads=num_threads):
            for ii in prange(nb_in_poly):
                res_x[0][ii] = _cl.get_at_pos(vec_x, ii)
                res_y[0][ii] = _cl.get_at_pos(vec_y, ii)
                res_z[0][ii] = _cl.get_at_pos(vec_z, ii)
                res_vres[0][ii] = _cl.get_at_pos(vec_vres, ii)
                res_lind[0][ii] = <int> _cl.get_at_pos(vec_lind, ii)
        # we have to keep only the rphi in vpoly
        for ii in range(sz_r):
            if _ss.count(set_r, disc_r[ii]) > 0:
                _cl.push_back(&vec_rphi, r_on_phi[ii])
        # we transform the set of rphi to an array
        sz_rphi[0] = vec_rphi.size
        res_rphi[0] = <double*> malloc(vec_rphi.size * sizeof(double))
        with nogil, parallel(num_threads=num_threads):
            for ii in prange(sz_rphi[0]):
                res_rphi[0][ii] = _cl.get_at_pos(vec_rphi, ii)
    else:
        for ii in range(npts):
            if _bgt.is_point_in_path(npts_vpoly, &vpoly[0][0], &vpoly[1][0],
                                pts[0,ii], pts[1,ii]):
                nb_in_poly += 1
                _cl.push_back(&vec_x, pts[0,ii])
                _cl.push_back(&vec_y, pts[1,ii])
                _cl.push_back(&vec_z, pts[2,ii])
                _cl.push_back(&vec_vres, vol_resol[ii])
                _cl.push_back(&vec_lind, lind[ii])
                # we create a set for the new radius in vpoly:
                _ss.insert(&set_r, pts[0,ii])
        # We initialize the arrays:
        res_x[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_y[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_z[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_vres[0] = <double*> malloc(nb_in_poly * sizeof(double))
        res_lind[0] = <long*> malloc(nb_in_poly * sizeof(long))
        with nogil, parallel(num_threads=num_threads):
            for ii in prange(nb_in_poly):
                res_x[0][ii] = _cl.get_at_pos(vec_x, ii)
                res_y[0][ii] = _cl.get_at_pos(vec_y, ii)
                res_z[0][ii] = _cl.get_at_pos(vec_z, ii)
                res_vres[0][ii] = _cl.get_at_pos(vec_vres, ii)
                res_lind[0][ii] = <int>_cl.get_at_pos(vec_lind, ii)
        for ii in range(sz_r):
            if _ss.count(set_r, disc_r[ii]) > 0:
                _cl.push_back(&vec_rphi, r_on_phi[ii])
        # we transform the set of rphi to an array
        sz_rphi[0] = vec_rphi.size
        res_rphi[0] = <double*> malloc(vec_rphi.size * sizeof(double))
        for ii in prange(sz_rphi[0], num_threads=num_threads):
            res_rphi[0][ii] = _cl.get_at_pos(vec_rphi, ii)

    _ss.free_ss(&set_r)
    _cl.free_cl(&vec_x)
    _cl.free_cl(&vec_y)
    _cl.free_cl(&vec_z)
    _cl.free_cl(&vec_rphi)
    _cl.free_cl(&vec_vres)
    _cl.free_cl(&vec_lind)

    return nb_in_poly


# ===============================================================
#           Vignetting Vmesh with VPoly
# ===============================================================


cdef inline int are_in_vignette(int sz_r, int sz_z,
                                double[:, ::1] vpoly,
                                int npts_vpoly,
                                double* disc_r,
                                double* disc_z,
                                long[:, ::1] is_in_vignette) nogil:
    # we keep only the points in vpoly
    cdef int ii, jj
    cdef int nb_in_poly = 0

    for ii in range(sz_r):
        for jj in range(sz_z):
            if _bgt.is_point_in_path(npts_vpoly,
                                     &vpoly[0][0], &vpoly[1][0],
                                     disc_r[ii], disc_z[jj]):
                nb_in_poly += 1
                is_in_vignette[ii, jj] = 1
            else:
                is_in_vignette[ii, jj] = 0
    return nb_in_poly
