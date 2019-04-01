#nouveau
cdef inline double* dist_los_circle_core(const double[3] direction,
                                         const double[3] origin,
                                         double radius, double circ_z,
                                         double norm_dir,
                                         double[2] result) nogil:
    # This function computes the intersection of a Ray (or Line Of Sight)
    # and a circle in 3D. It returns `kmin`, the coefficient such that the
    # ray of origin O = [ori1, ori2, ori3] and of directional vector
    # D = [dir1, dir2, dir3] is closest to the circle of radius `radius`
    # and centered `(0, 0, circ_z)` at the point P = O + kmin * D.
    # And `distance` the distance between the two closest points
    # The variable `norm_dir` is the squared norm of the direction of the ray.
    # ---
    # Source: https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
    # The line is P(t) = B+t*M.  The circle is |X-C| = r with Dot(N,X-C)=0.
    cdef int numRoots, i
    cdef double zero = 0., m0sqr, m0, rm0
    cdef double lambd, m2b2, b1sqr, b1, r0sqr, twoThirds, sHat, gHat, cutoff, s
    cdef double[3] vzero
    cdef double[3] D, oldD
    cdef double[3] MxN
    cdef double[3] DxN
    cdef double[3] NxDelta
    cdef double[3] circle_normal
    cdef double[3] roots
    cdef double[3] diff
    cdef double[3] circle_center
    cdef double[3] circle_closest
    cdef double[3] line_closest
    cdef double tmin
    cdef double distance

    # .. initialization .....
    for i in range(3):
        circle_center[i] = 0.
        circle_normal[i] = 0.
        vzero[i] = 0.
        roots[i] = 0.
    circle_normal[2] = 1
    circle_center[2] = circ_z

    D[0] = origin[0]
    D[1] = origin[1]
    D[2] = origin[2] - circ_z
    compute_cross_prod(direction, circle_normal, MxN)
    compute_cross_prod(D, circle_normal, DxN)
    m0sqr = compute_dot_prod(MxN, MxN)

    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        numRoots = 0

        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*direction =
        # (0,b1',b2').
        m0 = Csqrt(m0sqr)
        rm0 = radius * m0
        lambd = -compute_dot_prod(MxN, DxN) / m0sqr
        for i in range(3):
            D[i] += lambd * direction[i]
            DxN[i] += lambd * MxN[i]
        m2b2 = compute_dot_prod(direction, D)
        b1sqr = compute_dot_prod(DxN, DxN)
        with gil:
            print
            print("DxN = ", DxN[0], DxN[1], DxN[2])
            print(lambd)
            print(b1sqr)
            print(radius*m0sqr)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = Csqrt(b1sqr)
            rm0sqr = radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = Csqrt((rm0sqr * b1sqr)**twoThirds - b1sqr) / m0
                gHat = rm0sqr * sHat / Csqrt(m0sqr * sHat * sHat + b1sqr)
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2)
                        roots[numRoots] = s
                        numRoots += 1
                        s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = compute_bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
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
        tmin = roots[0]# + lambd
        for i in range(1,numRoots):
            t = roots[i] + lambd
            if (t>0 and t<tmin):
                tmin = t
        if tmin < 0:
            tmin = 0.
        # Now that we know the closest point on the line we can compute the
        # closest point on the circle and compute the distance
        tmin = tmin / norm_dir
        line_closest[0] = origin[0] + tmin * direction[0]
        line_closest[1] = origin[1] + tmin * direction[1]
        line_closest[2] = origin[2] + tmin * direction[2]
        compute_cross_prod(circle_normal, line_closest, NxDelta)
        if not (NxDelta[0] == 0. or NxDelta[1] == 0. or NxDelta[2] == 0.):
            NdotDelta = compute_dot_prod(circle_normal, line_closest)
            for i in range(3):
                line_closest[i] = line_closest[i] - NdotDelta * circle_normal[i]
            norm_delta = Csqrt(compute_dot_prod(line_closest, line_closest))
            for i in range(3):
                line_closest[i] = line_closest[i] / norm_delta
                circle_closest[i] = circle_center[i] + radius * line_closest[i]
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
            return result
        else:
            diff[0] = line_closest[0] - circle_center[0] - radius
            diff[1] = line_closest[1] - circle_center[1]
            diff[2] = line_closest[2] - circle_center[2]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = tmin
            result[1] = distance
            return result
    else:
        # The line direction and the plane normal are parallel.
        # There is only one solution the intersection between line and plane
        if (DxN != vzero) :
            # The line is A+t*N but with A != C.
            t = -compute_dot_prod(direction, D)
            if t > 0:
                t = t / norm_dir
            else:
                t = 0.
            # We compute line closest
            line_closest[0] = origin[0] + t * direction[0]
            line_closest[1] = origin[1] + t * direction[1]
            line_closest[2] = origin[2] + t * direction[2]
            # We compute cirlce closest
            for i in range(3):
                diff[i] = line_closest[i] - circle_center[i]
            distance = radius / Csqrt(compute_dot_prod(diff, diff))
            circle_closest[0] = line_closest[0] * distance
            circle_closest[1] = line_closest[1] * distance
            circle_closest[2] = circ_z + (line_closest[2] - circ_z) * distance
            for i in range(3):
                diff[i] = line_closest[i] - circle_closest[i]
            distance = Csqrt(compute_dot_prod(diff, diff))
            result[0] = t
            result[1] = distance
            return result
        else:
            # The line is C+t*N, so C is the closest point for the line and
            # all circle points are equidistant from it.
            t = 0.
            result[0] = t
            result[1] = radius
            return result
