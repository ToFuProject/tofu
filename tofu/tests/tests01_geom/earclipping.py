from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import sqrt, cos, sin, pi, arctan2 as atan2
import matplotlib.pyplot as plt
from matplotlib import style


def dot(u,v):
    return u[0] * v[0] + u[1]*v[1] + u[2]*v[2]

def cross(u,v):
    return np.r_[u[1]*v[2]-u[2]*v[1],
                 u[2]*v[0]-u[0]*v[2],
                 u[0]*v[1]-u[1]*v[0]]

def norm(u):
    return sqrt(dot(u,u))

def angle(u,v):
    return dot(u,v)/norm(u)/norm(v)

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

def is_reflex(u,v):
    normu = u/norm(u)
    normv = v/norm(v)
    nor = cross(normu,normv)
    return np.sum(nor) > 0.

def are_reflex(lpts):
    isr0 = is_reflex(lpts[-1]-lpts[0], lpts[1]-lpts[0])
    isr1 = is_reflex(lpts[-2]-lpts[-1], lpts[0]-lpts[-1])
    reflex = np.r_[isr0,
                   [is_reflex(pts0-pts1,
                              pts2-pts1) for (pts0, pts1, pts2)
                        in zip(lpts[:-2], lpts[1:-1], lpts[2:])],
                   isr1]
    return reflex


def get_one_ears(lpts, lref):
    # lpts = [ptsn, pts0, pts1, ..., ptsn-1, ptsn, pts0]
    npts = lpts.shape[0]
    ltris = []
    for i in range(npts):
        if not lref[i]:
            a_pt_in_tri = False
            for j in np.r_[np.r_[0:i-2], np.r_[i+2:npts]]:
                if lref[j]:
                    if is_pt_in_tri(lpts[i-1], lpts[i], lpts[i+1], lpts[j]):
                        a_pt_in_tri = True
                        break
            if not a_pt_in_tri:
                return i, [lpts[i-1], lpts[i], lpts[i+1]]
    assert(False)
    return

def get_all_ears(llpts):
    # https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    lref = are_reflex(llpts)
    lpts = np.copy(llpts)
    npts = lpts.shape[0]
    ltri = []
    while npts > 3:
        iear, tri = get_one_ears(lpts, lref)
        ltri.append(tri)
        lpts = np.r_[lpts[0:iear], lpts[iear+1:]]
        is_ref_im1 = lref[iear-1]
        is_ref_ip1 = lref[iear+1]
        lref = np.r_[lref[0:iear], lref[iear+1:]]
        if is_ref_im1:
            lref[iear-1] = is_reflex(lpts[iear-2] - lpts[iear-1],
                                     lpts[iear] - lpts[iear-1])
        if is_ref_ip1:
            lref[iear] = is_reflex(lpts[iear-1] - lpts[iear],
                                   lpts[iear+1] - lpts[iear])
        npts = npts-1
    ltri.append(lpts)
    return(ltri)

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
        return 0
    invdet = 1./det
    # calculate distance from vert to ray origin
    tvec = orig - vert0
    # calculate U parameter and test bounds
    u = dot(tvec, pvec) * invdet
    if u < 0. or u > 1.:
        return 0
    # prepare to test V parameter
    qvec = cross(tvec, edge1)
    # calculate V parameter and test bounds
    v = dot(dire, qvec) * invdet
    if v < 0. or u + v > 1.:
        return 0
    return 1

def intersect_poly(orig, dire, list_pts):
    list_tri = get_all_ears(list_pts)
    for tri in list_tri:
        if intersect_triangle(orig, dire, tri[0], tri[1], tri[2]):
            return True
    return False

#...........;

x = np.r_[2,4,6,6,4,3,4,3,2,2]
y = np.r_[2,0,2,5,2,2,3,4,3,2]
z = np.zeros_like(x)
num_pts = np.size(x)

mx = np.mean(x)
my = np.mean(y)
mz = np.mean(z)

#list_pts = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x,y,z)])
# print("one obvious no :", is_pt_in_tri(list_pts[0], list_pts[1], list_pts[2],
#                                        list_pts[3]))
# print("one YES :", is_pt_in_tri(list_pts[1], list_pts[3], list_pts[7],
#                                 list_pts[4]))

# x = np.r_[0,0,1,1,0]
# y = np.r_[1,0,0,1,1]
# z = np.zeros_like(x)
# npts = np.size(x)

list_pts = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x,y,z)])
list_pts = list_pts[:-1]
list_tri = get_all_ears(list_pts)


style.use('ggplot')
fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot(x,y,z,color='blue')
ax1.scatter(mx,my,mz,color='red',s=6)
for i in range(num_pts-1):
    ax1.text(x[i], y[i], z[i], str(i))
ax1.set_top_view()
ax1.view_init(azim=-90, elev=90)

plt.pause(1)

for tri in list_tri:
    xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
    ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
    zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
    ax1.plot(xs,ys,zs)

plt.pause(1)


print()
print("-------------------------------------------")
print("              example 2")

x = np.r_[0,3.5,5.5,7,8,7, 6,5,3,4, 0]
y = np.r_[2.5,0,1.5,1,5,4.5, 6,3,4,8, 2.5]
z = np.array([0 if xi < 5. else 1. for xi in x])
num_pts = np.size(x)
list_pts2 = np.array([np.r_[xi,yi,zi] for (xi,yi,zi) in zip(x,y,z)])
list_pts2 = list_pts2[:-1]
list_tri = get_all_ears(list_pts2)

fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot(x,y,z,color='blue')
for i in range(num_pts-1):
    ax1.text(x[i], y[i], z[i], str(i))
ax1.set_top_view()
ax1.view_init(azim=-90, elev=90)

plt.pause(1)
for tri in list_tri:
    xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
    ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
    zs = [tri[0][2], tri[1][2], tri[2][2], tri[0][2]]
    ax1.plot(xs,ys,zs)

plt.pause(1)


orig = np.r_[3.75, 2.5, -2]
dire = np.r_[0, 0,  1]
inter_p1 = intersect_poly(orig, dire, list_pts)
print(" Ray intersects p1 =", inter_p1)
inter_p2 = intersect_poly(orig, dire, list_pts2)
print(" Ray intersects p2 =", inter_p2)

orig = np.r_[5, 3.1, -2]
dire = np.r_[0, 0,  1]
inter_p1 = intersect_poly(orig, dire, list_pts)
print(" Ray intersects p1 =", inter_p1)
inter_p2 = intersect_poly(orig, dire, list_pts2)
print(" Ray intersects p2 =", inter_p2)

orig = np.r_[0, 0, 5]
dire = np.r_[4, 1,  -5]/2.
inter_p1 = intersect_poly(orig, dire, list_pts)
print(" Ray intersects p1 =", inter_p1)
inter_p2 = intersect_poly(orig, dire, list_pts2)
print(" Ray intersects p2 =", inter_p2)

print()
print("... 3d tests ...")
orig = np.r_[0, 2.5, 1]
fina = np.r_[6.1, 2., 0]
dire = fina - orig
inter_p1 = intersect_poly(orig, dire, list_pts)
print(" Ray intersects p1 =", inter_p1)
inter_p2 = intersect_poly(orig, dire, list_pts2)
print(" Ray intersects p2 =", inter_p2)

orig2 = np.r_[0, 2.5, 1]
fina2 = np.r_[6., 6., 0]
dire2 = fina2 - orig2
inter_p1 = intersect_poly(orig2, dire2, list_pts)
print(" Ray intersects p1 =", inter_p1)
inter_p2 = intersect_poly(orig2, dire2, list_pts2)
print(" Ray intersects p2 =", inter_p2)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot(x,y,z,color='blue')
ax2.plot([orig[0], fina[0]],
         [orig[1], fina[1]],
         [orig[2], fina[2]], color='red')
ax2.plot([orig2[0], fina2[0]],
         [orig2[1], fina2[1]],
         [orig2[2], fina2[2]], color='green')
ax2.set_top_view()
ax2.view_init(azim=-90, elev=90)

plt.show(block=True)


x = np.r_[2,4,6,6,4,3,4,3,2,2]
y = np.r_[2,0,2,5,2,2,3,4,3,2]
z = np.zeros_like(x)
num_pts = np.size(x)
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot(x,y,z,color='blue')
ax2.plot([orig[0], fina[0]],
         [orig[1], fina[1]],
         [orig[2], fina[2]], color='red')
ax2.plot([orig2[0], fina2[0]],
         [orig2[1], fina2[1]],
         [orig2[2], fina2[2]], color='green')
ax2.set_top_view()
ax2.view_init(azim=-90, elev=90)

plt.show(block=True)
