# distutils: language=c++
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
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport _raytracing_tools as _rt
cimport _basic_geom_tools as _bgt


# ==============================================================================
# =  Basic utilities: is angle reflex, vector np.diff, is point in triangle,...
# ==============================================================================
cdef inline bint is_reflex(const double[3] u,
                           const double[3] v) nogil:
    """
    Determines if the angle between U and -V is reflex (angle > pi) or not.
    Warning: note the MINUS in front of V, this was done as this is the only
             form how we will need this function. but it is NOT general.
    """
    cdef int ii
    cdef double sumc
    cdef double[3] ucrossv
    # ...
    _bgt.compute_cross_prod(u, v, ucrossv)
    sumc = 0.0
    for ii in range(3):
        # normally it should be a sum, but it is a minus cause is we have (U,-V)
        sumc -= ucrossv[ii]
    return sumc >= 0.

cdef inline void compute_diff3d(double[:, ::1] orig,
                                int nvert,
                                double* diff) nogil:
    cdef int ivert
    for ivert in range(nvert-1):
        diff[ivert*3 + 0] = orig[0,ivert+1] - orig[0,ivert]
        diff[ivert*3 + 1] = orig[1,ivert+1] - orig[1,ivert]
        diff[ivert*3 + 2] = orig[2,ivert+1] - orig[2,ivert]
    # doing the last point:
    diff[3*(nvert-1) + 0] = orig[0,0] - orig[0,nvert-1]
    diff[3*(nvert-1) + 1] = orig[1,0] - orig[1,nvert-1]
    diff[3*(nvert-1) + 2] = orig[2,0] - orig[2,nvert-1]
    return

cdef inline void are_points_reflex(double[:,::1] vignett,
                                   int nvert,
                                   double* diff,
                                   bint* are_reflex) nogil:
    """
    Determines if the interior angles of a polygons are reflex
    (angle > pi) or not.
    """
    cdef int ivert
    cdef int icoord
    cdef double[3] u1, v1, un, vn
    # .. Computing if reflex or not ...........................................
    for ivert in range(1,nvert):
        are_reflex[ivert] = is_reflex(&diff[ivert*3], &diff[(ivert-1)*3])
    # doing first point:
    are_reflex[0] = is_reflex(&diff[0], &diff[(nvert-1)*3])
    return

cdef inline bint is_pt_in_tri(double[3] v0, double[3] v1,
                              double Ax, double Ay, double Az,
                              double px, double py, double pz) nogil:
    """
    Tests if point P is on the triangle A, B, C such that
        v0 = C - A
        v1 = B - A
    and A = (Ax, Ay, Az) and P = (px, py, pz)
    """
    cdef double[3] v2
    cdef double dot00, dot01, dot02
    cdef double dot11, dot12
    cdef double invDenom
    cdef double u, v
    # computing vector between A and P
    v2[0] = px - Ax
    v2[1] = py - Ay
    v2[2] = pz - Az
    # compute dot products
    dot00 = _bgt.compute_dot_prod(v0, v0)
    dot01 = _bgt.compute_dot_prod(v0, v1)
    dot02 = _bgt.compute_dot_prod(v0, v2)
    dot11 = _bgt.compute_dot_prod(v1, v1)
    dot12 = _bgt.compute_dot_prod(v1, v2)
    # Compute barycentric coordinates
    invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)


# ==============================================================================
# =  Earclipping method: getting one ear of a poly, triangulate poly, ...
# ==============================================================================
cdef inline int get_one_ear(double[:,::1] polygon,
                            double* diff,
                            bint* lref,
                            vector[int] working_index,
                            int nvert) nogil:
    """
    A polygon's "ear" is defined as a triangle of vert_i-1, vert_i, vert_i+1,
    points on the polygon, where the point vert_i has the following properties:
        - Its interior angle is not reflex (angle < pi)
        - None of the other vertices of the polygon are in the triangle formed
          by its two annexing points
          Note: only reflex edges can be on the triangle
    """
    cdef int iloc
    cdef int i, j
    cdef int wi, wj
    cdef int wip1, wim1
    cdef bint a_pt_in_tri
    for i in range(1, nvert-1):
        wi = working_index[i]
        if wi != -1 and not lref[wi]:
            # angle is not reflex
            a_pt_in_tri = False
            # we get some useful values
            wip1 = working_index[i+1]
            wim1 = working_index[i-1]
            # We can test if there is another vertex in the 'ear'
            for j in range(nvert):
                wj = working_index[j]
                # We only test reflex angles:
                if (wj != -1 and lref[wj] # and wj is not a vertex of triangle
                    and wj != wim1 and wj != wip1 and wj != wi):
                    if is_pt_in_tri(&diff[wi*3], &diff[(wim1)*3],
                                    polygon[0,wi], polygon[1,wi],
                                    polygon[2,wi],
                                    polygon[0,wj], polygon[1,wj],
                                    polygon[2,wj]):
                        # We found a point in the triangle, thus is not ear
                        # no need to keep testing....
                        a_pt_in_tri = True
                        break
            # Let's check if there was a point in the triangle....
            if not a_pt_in_tri:
                return i# , [lpts[i-1], lpts[i], lpts[ip1]]
    # if we havent returned, either, there was an error somerwhere
    with gil:
        assert(False)
    return -1

cdef inline void earclipping_poly(double[:,::1] vignett,
                                  long* ltri,
                                  int nvert) nogil:
    """
    Triangulates a polygon by earclipping an edge at a time.
        vignett : (3,nvert) coordinates of poly
        nvert : number of vertices
    Result
        ltri : (3*(nvert-2)) int array, indices of triangles
    """
    # init...
    cdef double* diff = NULL
    cdef bint* lref = NULL
    cdef int loc_nv = nvert
    cdef int itri = 0
    cdef int ii
    cdef int wi, wim1, wip1
    cdef int iear
    cdef vector[int] working_index
    # .. First computing the edges coodinates .................................
    # .. and checking if the angles defined by the edges are reflex or not.....
    diff = <double*>malloc(3*nvert*sizeof(double))
    lref = <bint*>malloc(nvert*sizeof(bint))
    compute_diff3d(vignett, nvert, diff)
    are_points_reflex(vignett, nvert, diff, lref)
    # initialization of working index tab:
    for ii in range(nvert):
        working_index.push_back(ii)
    # .. Loop ..................................................................
    while loc_nv > 3:
        iear = get_one_ear(vignett, diff, lref, working_index, loc_nv)
        wim1 = working_index[iear-1]
        wi   = working_index[iear]
        wip1 = working_index[iear+1]
        ltri[itri*3]   = wim1
        ltri[itri*3+1] = wi
        ltri[itri*3+2] = wip1
        # updates on the "information" arrays:
        diff[wim1*3]   = vignett[0,wip1] - vignett[0,wim1]
        diff[wim1*3+1] = vignett[1,wip1] - vignett[1,wim1]
        diff[wim1*3+2] = vignett[2,wip1] - vignett[2,wim1]
        #... theoritically we should get rid of off diff[wip1] as well but
        # we'll just not use it, however we have to update lref
        # if an angle is not reflex, then it will stay so, only chage if reflex
        if lref[wim1]:
            if iear >= 2:
                lref[wim1] = is_reflex(&diff[3*wim1],
                                       &diff[3*working_index[iear-2]])
            else:
                lref[wim1] = is_reflex(&diff[3*wim1],
                                       &diff[3*working_index[loc_nv-1]])
        if lref[wip1]:
            lref[wip1] = is_reflex(&diff[wip1*3],
                                   &diff[wim1*3])
        # last but not least update on number of vertices and working indices
        itri = itri + 1
        loc_nv = loc_nv - 1
        working_index.erase(working_index.begin()+iear)
    # .. Cleaning up ...........................................................
    free(diff)
    free(lref)
    return

# ==============================================================================
# =  Polygons triangulation and Intersection Ray-Poly
# ==============================================================================
cdef inline void triangulate_polys(double[:, :, ::1] vignett_poly,
                                   long* lnvert,
                                   int nvign,
                                   long** ltri,
                                   int num_threads=16) nogil:
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
    # ...
    # -- Defining parallel part ------------------------------------------------
    with nogil, parallel(num_threads=num_threads):
        for ivign in prange(nvign):
            nvert = lnvert[ivign]
            ltri[ivign] = <long*>malloc((nvert-2)*3*sizeof(int))
            earclipping_poly(vignett_poly[ivign], ltri[ivign], nvert)
    return

cdef inline bint inter_ray_poly(const double[3] ray_orig,
                                const double[3] ray_vdir,
                                double[:, ::1] vignett,
                                int nvert,
                                long* ltri) nogil:
    cdef int ii
    for ii in range(nvert-2):
        if _rt.inter_ray_triangle(ray_orig, ray_vdir,
                              vignett[:,ltri[3*ii]],
                              vignett[:,ltri[3*ii+1]],
                              vignett[:,ltri[3*ii+2]]):
            return True
    return False

# ==============================================================================
# =  Vignetting
# ==============================================================================
cdef inline void vignetting_core(double[:, ::1] ray_orig,
                                 double[:, ::1] ray_vdir,
                                 double[:, :, ::1] vignett,
                                 long* lnvert,
                                 double* lbounds,
                                 long** ltri,
                                 int nvign,
                                 int nlos,
                                 bint* goes_through,
                                 int num_threads=16) nogil:
    cdef int ilos, ivign
    cdef int nvert
    cdef bint inter_bbox
    # == Defining parallel part ================================================
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
            for ivign in range(nvign):
                nvert = lnvert[ivign]
                # -- We check if intersection with  bounding box ---------------
                inter_bbox = _rt.inter_ray_aabb_box(sign_ray, invr_ray,
                                                    &lbounds[6*ivign],
                                                    loc_org,
                                                    countin=True)
                if not inter_bbox:
                    goes_through[ivign*nlos + ilos] = 0 # False
                    continue
                # -- if none, we continue --------------------------------------
                goes_through[ivign*nlos + ilos] = inter_ray_poly(loc_org,
                                                                 loc_dir,
                                                                 vignett[ivign],
                                                                 nvert,
                                                                 ltri[ivign])
    return
