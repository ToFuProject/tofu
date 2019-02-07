# coding: utf-8
import numpy as np
import tofu.geom._GG_LM as _GG
import time
import matplotlib.pyplot as plt
from test2 import Vector3, Line3, Circle3, comp_dist_los_circle

def test_Dist_Cricle_LOS_DV(los_dir, los_org, cir_rad, cir_z, klim = np.inf):
    [D0, D1, D2] = los_dir
    [u0, u1, u2] = los_org
    RZ0 = cir_rad
    RZ1 = cir_z
    uN = np.sqrt(u0**2+u1**2+u2**2)
    uParN = np.sqrt(u0**2+u1**2)
    DParN = np.sqrt(D0**2+D1**2)
    Sca = u0*D0+u1*D1+u2*D2
    ScaP = u0*D0+u1*D1
    if uParN == 0.:
        kPMin = (RZ1-D2)/u2
    else:
        kPMin = _GG.LOS_sino_findRootkPMin_Tor(uParN, uN, Sca, RZ0, RZ1, ScaP,
                                               DParN, klim, D0, D1, D2, u0, u1,
                                               u2, Mode="LOS")
    return kPMin

def test_Dist_Cricle_LOS_LM(los_dir, los_org, cir_rad, cir_z, klim = np.inf):
    origin = Vector3(los_org[0], los_org[1], los_org[2])
    direction = Vector3(los_dir[0], los_dir[1], los_dir[2])
    line = Line3(origin=origin, direction=direction)
    center = Vector3(0, 0, cir_z)
    circle = Circle3(center=center, radius=cir_rad)
    res = comp_dist_los_circle(line, circle)
    last_pout = res.lineClosest[0]
    [kpmin_x, kpmin_y, kpmin_z] = [0., 0., 0.]
    if not direction.x == 0.:
        kpmin = (last_pout.x - origin.x) / direction.x
    elif not direction.y == 0:
        kpmin = (last_pout.y - origin.y) / direction.y
    else :
        print(last_pout)
        print(origin)
        print(direction)
        kpmin = (last_pout.z - origin.z) / direction.z
    return kpmin

if __name__ == "__main__":
    origin = [2, 0, 0]
    direction = [0, 0, 1]
    radius = 1.0
    circ_z = 0.
    print("res for DV algo =", test_Dist_Cricle_LOS_DV(direction, origin,
                                                       radius, circ_z))
    print("res for LM algo =", test_Dist_Cricle_LOS_LM(direction, origin,
                                                       radius, circ_z))
