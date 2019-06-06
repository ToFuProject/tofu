# distutils: language=c++
from libcpp.vector cimport vector

cdef inline bint is_reflex(const double[3] u,
                           const double[3] v,
                           const bint not_normed=True) nogil:
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
                                double* diff):
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

cdef inline void is_pt_in_tri(double[3] v0, double[3] v1, double[3] A,
                              double[3] point) nogil:
    """
    Tests if point P is on the triangle A, B, C such that
        v0 = C - A
        v1 = B - A
    """
    cdef double[3] v2
    cdef double dot00, dot01, dot02
    cdef double dot11, dot12
    cdef double invDenom
    cdef double u, v
    # computing vector between A and P
    v2[0] = point[0] - A[0]
    v2[1] = point[1] - A[1]
    v2[2] = point[2] - A[2]
    # compute dot products
    dot00 = compute_dot_prod(v0, v0)
    dot01 = compute_dot_prod(v0, v1)
    dot02 = compute_dot_prod(v0, v2)
    dot11 = compute_dot_prod(v1, v1)
    dot12 = compute_dot_prod(v1, v2)
    # Compute barycentric coordinates
    invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)


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
                    if is_pt_in_tri(diff[wi*3], diff[(wim1)*3])
                                    polygon[:,wi], polygon[:,wj]):
                        # We found a point in the triangle, thus is not ear
                        # no need to keep testing....
                        a_pt_in_tri = True
                        break
            # Let's check if there was a point in the triangle....
            if not a_pt_in_tri:
                return i# , [lpts[i-1], lpts[i], lpts[ip1]]
    # if we havent returned, either, there was an error somerwhere
    assert(False)
    return -1

cdef inline void earclipping_poly(double[:,::1] vignett,
                                  int* ltri,
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
                lref[wim1] = is_reflex(&diff[3*wim1]
                                       &diff[3*working_index[iear-2]])
            else:
                lref[wim1] = is_reflex(&diff[3*wim1]
                                       &diff[3*working_index[loc_nv-1]])
        if lref[wip1]:
            lref[wip1] = is_reflex(&diff[wip1*3],
                                   &diff[wim1*3])
        # last but not least update on number of vertices and working indices
        itri = itri + 1
        loc_nv = loc_nv - 1
        working_index.erase(iear)
    # .. Cleaning up ...........................................................
    free(diff)
    free(lref)
    return
