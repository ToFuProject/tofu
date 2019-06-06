from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import sqrt, cos, sin, pi, arctan2 as atan2
import matplotlib.pyplot as plt
from matplotlib import style
import time


def norm(u):
    return sqrt(u[0] * u[0] + u[1]*u[1] + u[2]*u[2])

def dot(u,v):
    return u[0] * v[0] + u[1]*v[1] + u[2]*v[2]

def cross(u,v):
    res = np.zeros(3)
    res[0] = u[1]*v[2]-u[2]*v[1]
    res[1] = u[2]*v[0]-u[0]*v[2]
    res[2] = u[0]*v[1]-u[1]*v[0]
    return res

def angle(u,v):
    return dot(u,v)/norm(u)/norm(v)

#@profile
def is_pt_in_tri(A, B, C, P):
    #Compute vectors
    v0 = C - A
    v1 = B - A
    v2 = P - A
    # Compute dot products
    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)
    # Compute barycentric coordinates
    invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)

def compute_det(u,v):
    nor = cross(u,v)
    nor = nor / norm(nor)
    return dot(nor, cross(u,v)), np.sum(nor) > 0.

def compute_one_angle(u,v):
    normu = u/norm(u)
    normv = v/norm(v)
    udotv = dot(normu, normv)
    det, signcross = compute_det(normu, normv)
    angle = atan2(det, udotv)
    if signcross:
        angle = 2.*pi - angle
    return angle

def compute_angles(lpts):
    angle0 = compute_one_angle(lpts[-2]-lpts[0], lpts[1]-lpts[0])
    ang = np.r_[angle0, [compute_one_angle(pts0-pts1,
                                          pts2-pts1) for (pts0, pts1, pts2)
                in zip(lpts[:-2], lpts[1:-1], lpts[2:])]]
    return ang

#@profile
def is_reflex(u,v):
    nor = cross(u,v)
    return np.sum(nor) >= 0.

#@profile
def is_reflex_norm(u,v):
    nor = cross(u,v)
    return np.sum(nor) >= 0.


#@profile
def are_reflex(lpts):
    ndiff = np.diff(lpts, axis=0)
    # ndiff = [di/norm(di) for di in diff]
    isr0 = is_reflex(lpts[-1]-lpts[0], ndiff[0])
    isr1 = is_reflex(-ndiff[-1], lpts[0]-lpts[-1])
    res = [isr0]
    for i in range(np.shape(ndiff)[0]-1):
        res.append(is_reflex(ndiff[i], -ndiff[i+1]))
    res.append(isr1)
    return np.array(res)

#@profile
def get_one_ears(lpts, lref):
    # lpts = [ptsn, pts0, pts1, ..., ptsn-1, ptsn, pts0]
    npts = lpts.shape[0]
    ltris = []
    for i in range(1,npts-1):
        if not lref[i]:
            a_pt_in_tri = False
            for j in range(npts):
                if lref[j] and (j<i-1 or j > i+1):
                    if is_pt_in_tri(lpts[i-1], lpts[i], lpts[i+1], lpts[j]):
                        a_pt_in_tri = True
                        break
            if not a_pt_in_tri:
                return i, [lpts[i-1], lpts[i], lpts[i+1]]
    assert(False)
    return

#@profile
def get_all_ears(lpts):
    # https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    lref = are_reflex(lpts)
    npts = lpts.shape[0]
    ltri = []
    while npts > 3:
        iear, tri = get_one_ears(lpts, lref)
        ltri.append(tri)
        lpts = np.delete(lpts, iear, axis=0)
        npts = npts-1 # TODOOOOOOOO
        if iear == npts:
            is_ref_im1 = lref[iear-1]
            is_ref_ip1 = lref[0]
            lref = np.delete(lref, iear, axis=0)
            if is_ref_im1:
                lref[iear-1] = is_reflex(lpts[iear-2] - lpts[iear-1],
                                         lpts[0] - lpts[iear-1])
            if is_ref_ip1:
                lref[0] = is_reflex(lpts[iear-1] - lpts[0],
                                    lpts[1] - lpts[0])
        else:
            ip1 = iear + 1
            is_ref_im1 = lref[iear-1]
            is_ref_ip1 = lref[ip1]
            lref = np.delete(lref, iear, axis=0)
            if is_ref_im1:
                lref[iear-1] = is_reflex(lpts[iear-2] - lpts[iear-1],
                                        lpts[iear] - lpts[iear-1])
            if is_ref_ip1:
                lref[iear] = is_reflex(lpts[iear-1] - lpts[iear],
                                    lpts[ip1] - lpts[iear])
    ltri.append(lpts)
    return(ltri)

#@profile
def intersect_triangle(orig, dire,
                       vert0, vert1, vert2):
    # https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    small = 10**-6
    edge1 = vert1 - vert0
    edge2 = vert2 - vert0
    # begin calculating determinant  also used to calculate U parameter
    pvec = cross(dire, edge2)
    # if determinant is near zero ray lies in plane of triangle
    det = dot(edge1, pvec)
    if abs(det) < small:
        return False
    invdet = 1./det
    # calculate distance from vert to ray origin
    tvec = orig - vert0
    # calculate U parameter and test bounds
    u = dot(tvec, pvec) * invdet
    if u < 0. or u > 1.:
        return False
    # prepare to test V parameter
    qvec = cross(tvec, edge1)
    # calculate V parameter and test bounds
    v = dot(dire, qvec) * invdet
    if v < 0. or u + v > 1.:
        return False
    return True

#@profile
def intersect_poly(orig, dire, list_pts):
    list_tri = get_all_ears(list_pts)
    for tri in list_tri:
        if intersect_triangle(orig, dire, tri[0], tri[1], tri[2]):
            return True
    return False

def get_easy_triangles(lpts):
    npts = lpts.shape[0]
    mx = np.mean(lpts[:,0])
    my = np.mean(lpts[:,1])
    mz = np.mean(lpts[:,2])
    ltri = [[np.r_[p0[0],p0[1],p0[2]],
             np.r_[mx,my,mz],
             np.r_[p1[0],p1[1],p1[2]]]
            for (p0, p1) in zip(lpts[:-1],lpts[1:])]
    return ltri

def intersect_poly_easy(orig, dire, list_pts):
    list_tri = get_easy_triangles(list_pts)
    num_inter = 0
    for tri in list_tri:
        if intersect_triangle(orig, dire, tri[0], tri[1], tri[2]):
            num_inter += 1
    return num_inter > 0 and num_inter % 2 == 1


#...........;
def main_test(x1, x2, list_pts1, list_pts2, plot=False):
    if np.array_equal(list_pts1[-1], list_pts1[0]):
        list_pts1 = list_pts1[:-1]
    if np.array_equal(list_pts2[-1], list_pts2[0]):
        list_pts2 = list_pts2[:-1]
    # ...
    num_pts1 = np.size(x1)
    list_tri1 = get_all_ears(list_pts1)
    num_pts2 = np.size(x2)
    list_tri2 = get_all_ears(list_pts2)
    #....
    orig = np.r_[3.75, 2.5, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    orig = np.r_[5, 3.1, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    orig = np.r_[0, 0, 5]
    dire = np.r_[4, 1,  -5]/2.
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    print()
    print("... 3d tests ...")
    orig = np.r_[0, 2.5, 1]
    fina = np.r_[6.1, 2., 0]
    dire = fina - orig
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)

    orig2 = np.r_[0, 2.5, 1]
    fina2 = np.r_[6., 6., 0]
    dire2 = fina2 - orig2
    inter_p1 = intersect_poly(orig2, dire2, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly(orig2, dire2, list_pts2)
    print(" Ray intersects p2 =", inter_p2)

    if plot:
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.plot(x1,y1,z1,color='blue')
        ax1.set_top_view()
        ax1.view_init(azim=-90, elev=90)
        plt.pause(1)
        for tri in list_tri1:
            xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
            ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
            zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
            ax1.plot(xs,ys,zs)
        plt.pause(1)
        ax1.plot([orig[0], fina[0]],
                 [orig[1], fina[1]],
                 [orig[2], fina[2]], color='red')
        ax1.plot([orig2[0], fina2[0]],
                 [orig2[1], fina2[1]],
                 [orig2[2], fina2[2]], color='green')
        plt.show(block=True)
        #..
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.plot(x2,y2,z2,color='blue')
        ax2.set_top_view()
        ax2.view_init(azim=-90, elev=90)
        plt.pause(1)
        for tri in list_tri2:
            xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
            ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
            zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
            ax2.plot(xs,ys,zs)
        plt.pause(1)
        ax2.plot([orig[0], fina[0]],
                 [orig[1], fina[1]],
                 [orig[2], fina[2]], color='red')
        ax2.plot([orig2[0], fina2[0]],
                 [orig2[1], fina2[1]],
                 [orig2[2], fina2[2]], color='green')
        plt.show(block=True)


def main_test_easy(plot=True):
    style.use('ggplot')
    # Conf 1
    list_tri1 = get_easy_triangles(list_pts1)
    # Conf 2
    list_tri2 = get_easy_triangles(list_pts2)
    # Tests ....
    orig = np.r_[3.75, 2.5, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    orig = np.r_[5, 3.1, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    orig = np.r_[0, 0, 5]
    dire = np.r_[4, 1,  -5]/2.
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    print(" Ray intersects p1 =", inter_p1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    print(" Ray intersects p2 =", inter_p2)
    print()
    print("... 3d tests ...")
    orig = np.r_[0, 2.5, 1]
    fina = np.r_[6.1, 2., 0]
    dire = fina - orig
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    print(" Ray 1 intersects p1 =", inter_p1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    print(" Ray 1 intersects p2 =", inter_p2)
    orig2 = np.r_[0, 2.5, 1]
    fina2 = np.r_[6., 6., 0]
    dire2 = fina2 - orig2
    inter_p1 = intersect_poly_easy(orig2, dire2, list_pts1)
    print(" Ray 2 intersects p1 =", inter_p1)
    inter_p2 = intersect_poly_easy(orig2, dire2, list_pts2)
    print(" Ray 2 intersects p2 =", inter_p2)

    if plot:
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.plot(x1,y1,z1,color='blue')
        ax1.set_top_view()
        ax1.view_init(azim=-90, elev=90)
        plt.pause(1)
        for tri in list_tri1:
            xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
            ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
            zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
            ax1.plot(xs,ys,zs)
        plt.pause(1)
        ax1.plot([orig[0], fina[0]],
                 [orig[1], fina[1]],
                 [orig[2], fina[2]], color='red')
        ax1.plot([orig2[0], fina2[0]],
                 [orig2[1], fina2[1]],
                 [orig2[2], fina2[2]], color='green')
        plt.show(block=True)
        #..
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.plot(x2,y2,z2,color='blue')
        ax2.set_top_view()
        ax2.view_init(azim=-90, elev=90)
        plt.pause(1)
        for tri in list_tri2:
            xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
            ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
            zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
            ax2.plot(xs,ys,zs)
        plt.pause(1)
        ax2.plot([orig[0], fina[0]],
                 [orig[1], fina[1]],
                 [orig[2], fina[2]], color='red')
        ax2.plot([orig2[0], fina2[0]],
                 [orig2[1], fina2[1]],
                 [orig2[2], fina2[2]], color='green')
        plt.show(block=True)


def time_test_earclipping():
    # -- First ray
    orig = np.r_[3.75, 2.5, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    # -- Second ray
    orig = np.r_[5, 3.1, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    # -- Third ray
    orig = np.r_[0, 0, 5]
    dire = np.r_[4, 1,  -5]/2.
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    # ==== 3D TESTS ====
    orig = np.r_[0, 2.5, 1]
    fina = np.r_[6.1, 2., 0]
    dire = fina - orig
    inter_p1 = intersect_poly(orig, dire, list_pts1)
    inter_p2 = intersect_poly(orig, dire, list_pts2)
    # Another ray
    orig2 = np.r_[0, 2.5, 1]
    fina2 = np.r_[6., 6., 0]
    dire2 = fina2 - orig2
    inter_p1 = intersect_poly(orig2, dire2, list_pts1)
    inter_p2 = intersect_poly(orig2, dire2, list_pts2)

def time_test_easy():
    # -- First ray
    orig = np.r_[3.75, 2.5, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    # -- Second ray
    orig = np.r_[5, 3.1, -2]
    dire = np.r_[0, 0,  1]
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    # -- Third ray
    orig = np.r_[0, 0, 5]
    dire = np.r_[4, 1,  -5]/2.
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    # ==== 3D TESTS ====
    orig = np.r_[0, 2.5, 1]
    fina = np.r_[6.1, 2., 0]
    dire = fina - orig
    inter_p1 = intersect_poly_easy(orig, dire, list_pts1)
    inter_p2 = intersect_poly_easy(orig, dire, list_pts2)
    # Another ray
    orig2 = np.r_[0, 2.5, 1]
    fina2 = np.r_[6., 6., 0]
    dire2 = fina2 - orig2
    inter_p1 = intersect_poly_easy(orig2, dire2, list_pts1)
    inter_p2 = intersect_poly_easy(orig2, dire2, list_pts2)

def test_star_earclipping(lpts):
    # -- Hit
    ori = np.r_[-1.75, 1.2, -2]
    fin = np.r_[0., 0., 0]
    div = fin - ori
    inter = intersect_poly(ori, div, lpts)
    # -- Miss
    fin = np.r_[15., 15., 0.]
    div = fin - ori
    inter = intersect_poly(ori, div, lpts)

def test_star_easy(lpts):
    # -- Hit
    ori = np.r_[-1.75, 1.2, -2]
    fin = np.r_[0., 0., 0]
    div = fin - ori
    inter = intersect_poly_easy(ori, div, lpts)
    # -- Miss
    fin = np.r_[15., 15., 0.]
    div = fin - ori
    inter = intersect_poly_easy(ori, div, lpts)


if __name__ == '__main__':
    style.use('ggplot')
    # ... First test case ......................................................
    x1 = np.r_[2,3,4,5,6,6,  6,5,   4,3.5,3,3.5,4,3.5,3,3.5,2,2,2]
    y1 = np.r_[2,1,0,1,2,3.5,5,3.5, 2,2.0,2,2.5,3,3.5,4,3.5,3,2.5,2]
    x1 = np.r_[2,4,6,6,4,3,4,3,2,2]
    y1 = np.r_[2,0,2,5,2,2,3,4,3,2]
    z1 = np.zeros_like(x1)
    print(" NPTS for first coordinates = ", x1.shape)
    list_pts1 = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x1,y1,z1)])
    # ... Second test case .....................................................
    x2 = np.r_[0,3.5,5.5,7,8,7, 6,5,3,4, 0]
    y2 = np.r_[2.5,0,1.5,1,5,4.5, 6,3,4,8, 2.5]
    z2 = np.array([0 if xi < 5. else 1. for xi in x2])
    list_pts2 = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x2,y2,z2)])
    print(" NPTS for first coordinates = ", x2.shape)
    # ..
    import timeit
    print("easy test case :")
    print(timeit.timeit("time_test_easy()",
                        setup="from __main__ import time_test_easy, x1, x2, list_pts1, list_pts2",
                        number=1000))
    list_pts1 = list_pts1[:-1]
    list_pts2 = list_pts2[:-1]
    print("earclipping test case :")
    print(timeit.timeit("time_test_earclipping()",
                        setup="from __main__ import time_test_earclipping, x1, x2, list_pts1, list_pts2",
                        number=1000))
    #....
    main_test(x1, x2, list_pts1, list_pts2, plot=False)
    print()
    print("easy")
    print()
    main_test_easy(plot=False)

    # from planar import Polygon
    # for i in range(3,12):
    #     # star = np.array(Polygon.star(i, 1, 4))
    #     # npts = i*2
    #     star = np.array(Polygon.regular(i, radius=1))
    #     npts = i
    #     x = np.zeros(npts+1)
    #     y = np.zeros(npts+1)
    #     z = np.zeros(npts+1)
    #     x[:-1] = star[:,0]
    #     y[:-1] = star[:,1]
    #     x[-1] = x[0]
    #     y[-1] = y[0]
    #     lpts = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x,y,z)])
    #     # ...
    #     start1 = time.time()
    #     test_star_easy(lpts)
    #     end1 = time.time()
    #     print("For i =", i, " easy method =", end1 - start1)
    #     # ...
    #     lpts = lpts[:-1]
    #     start2 = time.time()
    #     test_star_earclipping(lpts)
    #     end2 = time.time()
    #     print("For i =", i, " earclipping method =", end2 - start2)
    #     print(" * Ear clip won :", end2-start2 <= end1-start1)
    #     print()
