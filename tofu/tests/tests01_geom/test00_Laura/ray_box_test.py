def compute_inv(direction):
    try:
        dx = 1./direction[0]
    except ZeroDivisionError:
        dx = float('Inf')
    try:
        dy = 1./direction[1]
    except ZeroDivisionError:
        dy = float('Inf')
    try:
        dz = 1./direction[2]
    except ZeroDivisionError:
        dz = float('Inf')
    inv_direction = [dx, dy, dz]
    return inv_direction

def compute_sign(direction):
    inv_direction = compute_inv(direction)
    # sign = [0, 0, 0]
    # sign[0] = int(inv_direction[0] < 0)
    # sign[1] = int(inv_direction[1] < 0)
    # sign[2] = int(inv_direction[2] < 0)
    sign = [1 if inv < 0 else 0 for inv in inv_direction]
    return inv_direction, sign

def intersect(bounds, inv_direction, sign, origin, t0, t1) :
    tmin = (bounds[sign[0]][0] - origin[0]) * inv_direction[0];
    tmax = (bounds[1-sign[0]][0] - origin[0]) * inv_direction[0];
    tymin = (bounds[sign[1]][1] - origin[1]) * inv_direction[1];
    tymax = (bounds[1-sign[1]][1] - origin[1]) * inv_direction[1];
    if ( (tmin > tymax) or (tymin > tmax) ):
        return False
    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax
    tzmin = (bounds[sign[2]][2] - origin[2]) * inv_direction[2]
    tzmax = (bounds[1-sign[2]][2] - origin[2]) * inv_direction[2]
    if ( (tmin > tzmax) or (tzmin > tmax) ):
        return False
    if (tzmin > tmin):
        tmin = tzmin
    if (tzmax < tmax):
        tmax = tzmax
    return ( (tmin < t1) and (tmax > t0) )

def check_inter_bbox_ray(bb_xmin, bb_ymin, bb_zmin, bb_xmax, bb_ymax, bb_zmax, us, ds):
    # print("  Ray u =", us.shape, us[0], us[1], us[2],
    #       "  Ds = ", ds.shape, ds[0], ds[1], ds[2])
    bounds = [[bb_xmin, bb_ymin, bb_zmin], [bb_xmax, bb_ymax, bb_zmax]]
    inv, sign = compute_sign(ds)
    return intersect(bounds, inv, sign, us, -100000,100000)
