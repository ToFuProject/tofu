class Vector3(object):
    x = 0.
    y = 0.
    z = 0.
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Ray(object):
    def __init__(self, o, d):
        self.origin = o
        self.direction = d
        try:
            dx = 1./d.x
        except ZeroDivisionError:
            dx = float('Inf')
        try:
            dy = 1./d.y
        except ZeroDivisionError:
            dy = float('Inf')
        try:
            dz = 1./d.z
        except ZeroDivisionError:
            dz = float('Inf')
        self.inv_direction = Vector3(dx, dy, dz)
        self.sign = [0, 0, 0]
        self.sign[0] = int(self.inv_direction.x < 0)
        self.sign[1] = int(self.inv_direction.y < 0)
        self.sign[2] = int(self.inv_direction.z < 0)


class Box(object):
    def __init__(self,vmin, vmax):
        self.bounds = []
        self.bounds.append(vmin)
        self.bounds.append(vmax)

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


if __name__ == "__main__":
    b_min = Vector3(0., 0., 0.)
    b_max = Vector3(1., 1., 1.)
    box = Box(b_min, b_max)

    origin = Vector3(1., 0., 0.)
    direction = Vector3(1., 0.5, 0.5)
    ray = Ray(origin, direction)

    print(box.intersect(ray, -10000, 10000))
