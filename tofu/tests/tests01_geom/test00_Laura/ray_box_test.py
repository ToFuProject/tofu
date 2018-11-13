from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

class Vector3(object):
    x = 0.
    y = 0.
    z = 0.
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Ray(object):
    def __init__(self, o, d, notVec = False):
        if notVec :
            self.origin = Vector3(o[0], o[1], o[2])
            self.direction = Vector3(d[0], d[1], d[2])
        else:
            self.origin = o
            self.direction = d
        try:
            dx = 1./self.direction.x
        except ZeroDivisionError:
            dx = float('Inf')
        try:
            dy = 1./self.direction.y
        except ZeroDivisionError:
            dy = float('Inf')
        try:
            dz = 1./self.direction.z
        except ZeroDivisionError:
            dz = float('Inf')
        self.inv_direction = Vector3(dx, dy, dz)
        self.sign = [0, 0, 0]
        self.sign[0] = int(self.inv_direction.x < 0)
        self.sign[1] = int(self.inv_direction.y < 0)
        self.sign[2] = int(self.inv_direction.z < 0)
    def plot(self, ax=None,block=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect("equal")
        soa = np.array([[self.origin.x,
                  self.origin.y,
                  self.origin.z,
                  self.direction.x,
                  self.direction.y,
                  self.direction.z]])

        X, Y, Z, U, V, W = zip(*soa)
        ax.quiver(X, Y, Z, U, V, W,
                  color="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.quiver(self.origin.x,
        #           self.origin.y,
        #           self.origin.z,
        #           self.direction.x,
        #           self.direction.y,
        #           self.direction.z,
        #           color="r")
        plt.title("Showing the Rays")
        plt.show(block=block)

class Box(object):
    def __init__(self, vmin, vmax):
        self.bounds = []
        self.bounds.append(vmin)
        self.bounds.append(vmax)
    def __init__(self, bounds):
        bmin = Vector3(bounds[0], bounds[1], bounds[2])
        bmax = Vector3(bounds[3], bounds[4], bounds[5])
        self.bounds = []
        self.bounds.append(bmin)
        self.bounds.append(bmax)


    def intersect(self, r, t0, t1) :
        bounds = self.bounds
        tmin = (bounds[r.sign[0]].x - r.origin.x) * r.inv_direction.x;
        tmax = (bounds[1-r.sign[0]].x - r.origin.x) * r.inv_direction.x;
        tymin = (bounds[r.sign[1]].y - r.origin.y) * r.inv_direction.y;
        tymax = (bounds[1-r.sign[1]].y - r.origin.y) * r.inv_direction.y;
        if ( (tmin > tymax) or (tymin > tmax) ):
            return False
        if (tymin > tmin):
            tmin = tymin
        if (tymax < tmax):
            tmax = tymax
        tzmin = (bounds[r.sign[2]].z - r.origin.z) * r.inv_direction.z
        tzmax = (bounds[1-r.sign[2]].z - r.origin.z) * r.inv_direction.z
        if ( (tmin > tzmax) or (tzmin > tmax) ):
            return False
        if (tzmin > tmin):
            tmin = tzmin
        if (tzmax < tmax):
            tmax = tzmax
        return ( (tmin < t1) and (tmax > t0) )
    
    def plot(self, ax=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        rx = [self.bounds[0].x, self.bounds[1].x]
        ry = [self.bounds[0].y, self.bounds[1].y]
        rz = [self.bounds[0].z, self.bounds[1].z]
        for s, e in combinations(np.array(list(product(rx, ry, rz))), 2):
            #if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")
        return ax

def check_inter_bbox_ray(bounds, us, ds):
    # print("  Ray u =", us.shape, us[0], us[1], us[2],
    #       "  Ds = ", ds.shape, ds[0], ds[1], ds[2])
    bbox = Box(bounds)
    ray = Ray(us, ds, notVec = True)
    return bbox.intersect(ray, -100000,100000)

if __name__ == "__main__":
    b_min = Vector3(0., 0., 0.)
    b_max = Vector3(1., 1., 1.)
    box = Box(b_min, b_max)

    origin = Vector3(1., 0., 0.)
    direction = Vector3(1., 0.5, 0.5)
    ray = Ray(origin, direction)

    print(box.intersect(ray, -10000, 10000))
