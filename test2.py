import numpy as np


class Vector3:
    x = 0
    y = 0
    z = 0
    def __init__(self, x=0, y=0, z=0, copy=None):
        if copy is None:
            self.x = x
            self.y = y
            self.z = z
        else:
            self.x = copy.x
            self.y = copy.y
            self.z = copy.z
    def __mul__(self, gamma):
        return Vector3(gamma*self.x, gamma*self.y, gamma*self.z)
    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)
    def __truediv__(self, other):
        res = Vector3(self.x/other, self.y/other, self.z/other)
        return res
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
    def __sub__(self, other):
        res = Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        return res
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__
    def __add__(self, other):
        """ Returns the vector addition of self and other """
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

class Line3:
    def __init__(self, origin=Vector3(), direction=Vector3()):
        self.origin = origin
        self.direction = direction

class Circle3:
    def __init__(self, center=Vector3(), normal=Vector3(0, 0, 1), radius=1.0):
        self.center = center
        self.normal = normal
        self.radius = radius

class ClosestInfo:
    sqrDistance = 0
    lineClosest = Vector3()
    circleClosest = Vector3()
    equidistant = False
    def __init__(self):
        return
    def __lt__(self, other):
        return self.sqrDistance < other.sqrDistance


class Result:
    distance = 0.0
    sqrDistance = 0.0
    numClosestPairs = 0
    lineClosest = [Vector3(), Vector3()]
    circleClosest = [Vector3(), Vector3()]
    equidistant = False
    def __init__(self):
        return
    def __str__(self):
        string = "Distance = " + str(self.distance) + "\n"
        string += "Square Distance = " + str(self.sqrDistance) + "\n"
        string += "numClosestPairs = " + str(self.numClosestPairs) + "\n"
        string += "Line Closest[0] = " + str(self.lineClosest[0]) + "\n"
        string += "Line Closest[1] = " + str(self.lineClosest[1]) + "\n"
        string += "CircleClosest[0]= " + str(self.circleClosest[0]) + "\n"
        string += "CircleClosest[1]= " + str(self.circleClosest[1]) + "\n"
        string += "equidistant = " + str(self.equidistant) + "\n"
        return string

def Normalize(v):
    length = np.sqrt(Dot(v, v))
    if (length > 0.0):
        v /= length
    else:
        for i in range(3):
            v[i] = 0.
    return v

def GetPair(line, circle, D, t):
    delta = D + t * line.direction / Dot(line.direction, line.direction)
    lineClosest = circle.center + delta
    delta -= Dot(circle.normal, delta) * circle.normal
    delta = Normalize(delta)
    circleClosest = circle.center + circle.radius * delta
    return lineClosest, circleClosest

def Cross(v1, v2):
    x0 = v1.x
    y0 = v2.x
    x1 = v1.y
    y1 = v2.y
    x2 = v1.z
    y2 = v2.z
    return Vector3(x1*y2-x2*y1, x2*y0-x0*y2, x0*y1-x1*y0)

def Dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def Bisect(m2b2, rm0sqr, m0sqr, b1sqr, smin, smax):
    G = lambda s: s + m2b2 - rm0sqr*s / np.sqrt(m0sqr*s*s + b1sqr)
    # The function is known to be increasing, so we can specify -1 and +1
    # as the function values at the bounding interval endpoints.  The use
    # of 'double' is intentional in case Real is a BSNumber or BSRational
    # type.  We want the bisections to terminate in a reasonable amount of
    # time.
    maxIterations = 10000
    root = 0
    gbg, root = Find(G, smin, smax, -1.0, 1.0, maxIterations, root)
    gmin = G(root)
    return root


def Find(F, t0, t1, f0, f1, maxIterations, root):
    if (t0 < t1):
        # Test the endpoints to see whether F(t) is zero.
        if f0 == 0.:
            root = t0
            return 1, root
        if f1 == 0.:
            root = t1
            return 1, root
        if f0*f1 > 0.:
            # It is not known whether the interval bounds a root.
            return 0, root
        for i in range(2, maxIterations+1):
            root = (0.5) * (t0 + t1)
            if (root == t0 or root == t1):
                # The numbers t0 and t1 are consecutive floating-point
                # numbers.
                break
            fm = F(root)
            product = fm * f0;
            if (product < 0.):
                t1 = root
                f1 = fm
            elif (product > 0.):
                t0 = root
                f0 = fm
            else:
                break
        return i, root
    else:
        return 0, root


def GetOrthogonal(v, unitLength):
    cmax = abs(v[0])
    imax = 0
    for i in range(1, N):
        c = abs(v[i])
        if (c > cmax):
            cmax = c
            imax = i

    result = np.zeros(N)
    inext = imax + 1
    if (inext == N):
        inext = 0
    result[imax] = v[inext];
    result[inext] = -v[imax];
    if (unitLength):
        sqrDistance = result[imax] * result[imax] + result[inext] * result[inext]
        invLength = 1.0 / np.sqrt(sqrDistance)
        result[imax] *= invLength
        result[inext] *= invLength
    return result

def comp_dist_los_circle(line, circle):
    # The line is P(t) = B+t*M.  The circle is |X-C| = r with Dot(N,X-C)=0.
    vzero = Vector3()
    zero = 0.
    result = Result()
    D = line.origin - circle.center
    MxN = Cross(line.direction, circle.normal)
    DxN = Cross(D, circle.normal)
    m0sqr = Dot(MxN, MxN)
    if (m0sqr > zero):
        # Compute the critical points s for F'(s) = 0.
        # Real s, t;
        numRoots = 0
        roots = np.zeros(3)

        # The line direction M and the plane normal N are not parallel.  Move
        # the line origin B = (b0,b1,b2) to B' = B + lambd*line.direction =
        # (0,b1',b2').
        m0 = np.sqrt(m0sqr)
        rm0 = circle.radius * m0
        lambd = -Dot(MxN, DxN) / m0sqr
        oldD = Vector3(copy=D)
        D += lambd * line.direction
        DxN += lambd * MxN
        m2b2 = Dot(line.direction, D)
        b1sqr = Dot(DxN, DxN)
        if (b1sqr > zero) :
            # B' = (0,b1',b2') where b1' != 0.  See Sections 1.1.2 and 1.2.2
            # of the PDF documentation.
            b1 = np.sqrt(b1sqr)
            rm0sqr = circle.radius * m0sqr
            if (rm0sqr > b1):
                twoThirds = 2.0 / 3.0
                sHat = np.sqrt(np.power(rm0sqr * b1sqr, twoThirds) - b1sqr) / m0
                gHat = rm0sqr * sHat / np.sqrt(m0sqr * sHat * sHat + b1sqr)
                cutoff = gHat - sHat
                if (m2b2 <= -cutoff):
                    s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2, -m2b2 + rm0)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == -cutoff):
                        roots[numRoots] = -sHat
                        numRoots += 1
                elif (m2b2 >= cutoff):
                    s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                        -m2b2)
                    roots[numRoots] = s
                    numRoots += 1
                    if (m2b2 == cutoff):
                        roots[numRoots] = sHat;
                        numRoots += 1
                else:
                    if (m2b2 <= zero):
                        s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                            -m2b2 + rm0)
                        roots[numRoots] = s
                        numRoots += 1
                        s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -sHat)
                        roots[numRoots] = s
                        numRoots += 1
                    else:
                        s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
                            -m2b2);
                        roots[numRoots] = s
                        numRoots += 1
                        s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, sHat,
                            -m2b2 + rm0);
                        roots[numRoots] = s
                        numRoots += 1
            else:
                if (m2b2 < zero):
                    s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2,
                        -m2b2 + rm0)
                elif (m2b2 > zero):
                    s = Bisect(m2b2, rm0sqr, m0sqr, b1sqr, -m2b2 - rm0,
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
                s = -m2b2 + rm0;
                roots[numRoots] = s
                numRoots += 1
                s = -m2b2 - rm0;
                roots[numRoots] = s
                numRoots += 1

        #std::array<ClosestInfo, 4> candidates;
        candidates = [ClosestInfo(), ClosestInfo(), ClosestInfo(), ClosestInfo()]
        for i in range(numRoots):
            t = roots[i] + lambd
            info = ClosestInfo()
            NxDelta = Cross(circle.normal, oldD + t * line.direction)
            if (NxDelta != vzero):
                info.lineClosest, \
                  info.circleClosest = GetPair(line, circle, oldD, t)
                info.equidistant = False
            else:
                U = GetOrthogonal(circle.normal, True)
                info.lineClosest = circle.center
                info.circleClosest = circle.center + circle.radius * U
                info.equidistant = True
            diff = info.lineClosest - info.circleClosest
            info.sqrDistance = Dot(diff, diff)
            candidates[i] = info

        np.sort(candidates)

        result.numClosestPairs = 1
        result.lineClosest[0] = candidates[0].lineClosest
        result.circleClosest[0] = candidates[0].circleClosest
        if (numRoots > 1 and
            candidates[1].sqrDistance == candidates[0].sqrDistance):
            result.numClosestPairs = 2
            result.lineClosest[1] = candidates[1].lineClosest
            result.circleClosest[1] = candidates[1].circleClosest
        result.equidistant = False
    else:
        # The line direction and the plane normal are parallel.
        if (DxN != vzero) :
            # The line is A+t*N but with A != C.
            result.numClosestPairs = 1
            result.lineClosest[0], \
              result.circleClosest[0] = GetPair(line, circle, D,
                                                -Dot(line.direction, D))
            result.equidistant = False
        else:
            # The line is C+t*N, so C is the closest point for the line and
            # all circle points are equidistant from it.
            U = GetOrthogonal(circle.normal, True)
            result.numClosestPairs = 1
            result.lineClosest[0] = circle.center
            result.circleClosest[0] = circle.center + circle.radius * U
            result.equidistant = True
    diff = result.lineClosest[0] - result.circleClosest[0]
    result.sqrDistance = Dot(diff, diff)
    result.distance = np.sqrt(result.sqrDistance)
    return result


if __name__ == '__main__':
    origin = Vector3(0, 1, 2)
    direction = Vector3(4./5., 0, 3./5.)
    line = Line3(origin=origin, direction=direction)
    circle = Circle3()
    res = comp_dist_los_circle(line, circle)
    print(res)

    print('')
    print('--------------------')
    origin = Vector3(2,0,0)
    direction = Vector3(0,0,1)
    line = Line3(origin=origin, direction=direction)
    circle = Circle3()
    res = comp_dist_los_circle(line, circle)
    print(res)
    exact = Vector3(2, 0, 0)
    print(" ================> Error : ", res.circleClosest[0] - exact,
          res.lineClosest[0] - exact)

    print('')
    print('--------------------')
    origin = Vector3(0,0,0)
    direction = Vector3(1,1,0)
    line = Line3(origin=origin, direction=direction)
    center = Vector3(0,0,0)
    circle = Circle3(center=center, radius=4.0)
    res = comp_dist_los_circle(line, circle)
    exact = Vector3(np.cos(np.pi/4)*4, np.cos(np.pi/4)*4, 0)
    print(res)
    print(" ================> Error 1 : ", res.circleClosest[0] - exact,
          res.lineClosest[0] - exact)
    exact = Vector3(-np.cos(np.pi/4)*4, -np.cos(np.pi/4)*4, 0)
    print(" ================> Error 2 : ", res.circleClosest[1] - exact,
          res.lineClosest[1] - exact)

    print('')
    print('--------------------')
    origin = Vector3(-1.,-1.,-1.)
    direction = Vector3(3.,1.,1.)/np.sqrt(11)
    line = Line3(origin=origin, direction=direction)
    print("  *** Line origin = ", line.origin, " direction = ", line.direction)
    center = Vector3(0,0,0)
    circle = Circle3(center=center, radius=2.0)
    print("  *** Circle center = ", circle.center, " normal = ", circle.normal,
          " radius = ", circle.radius)
    res = comp_dist_los_circle(line, circle)
    print(res)
    exact = Vector3(2,0,0)
    print(" ================> Error : ", res.circleClosest[0] - exact,
          res.lineClosest[0] - exact)
