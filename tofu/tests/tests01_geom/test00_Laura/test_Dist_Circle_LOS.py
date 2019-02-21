# coding: utf-8
import numpy as np
import tofu.geom._GG as _GG

import timeit
import matplotlib.pyplot as plt
from test2 import Vector3, Line3, Circle3, comp_dist_los_circle

def test_Dist_Cricle_LOS_DV(los_dir, los_ori, cir_rad, cir_z, klim = np.inf,
                            exact=[0,0,0]):
    [D0, D1, D2] = los_ori
    [u0, u1, u2] = los_dir
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

    for i in range(3):
        if not los_dir[i] == 0:
            kmin_ex = (exact[i] - los_ori[i]) / los_dir[i]
            err = abs(kPMin - kmin_ex)
            return kPMin, err

    print("ERRORRRRRRRRRRRRRRRR", los_dir)
    return

def test_Dist_Cricle_LOS_LM2(los_dir, los_ori, cir_rad, cir_z,
                            exact=[0,0,0]):
    [D0, D1, D2] = los_ori
    [u0, u1, u2] = los_dir
    uParN = np.sqrt(u0**2+u1**2)
    if uParN == 0.:
        kPMin = (cir_z-D2)/u2
    else:
        kPMin = _GG.comp_dist_los_circle2(u0, u1, u2,
                                          D0, D1, D2,
                                          cir_rad, cir_z)
    for i in range(3):
        if not los_dir[i] == 0:
            kmin_ex = (exact[i] - los_ori[i]) / los_dir[i]
            err = abs(kPMin - kmin_ex)
            return kPMin, err

    print("ERRORRRRRRRRRRRRRRRR", los_dir)
    return


def test_Dist_Cricle_LOS_LM(los_dir, los_org, cir_rad, cir_z, klim = np.inf,
                            exact=[0,0,0]):
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
        kpmin = (last_pout.z - origin.z) / direction.z

    for i in range(3):
        if not los_dir[i] == 0:
            kmin_ex = (exact[i] - los_org[i]) / los_dir[i]
            err = abs(kpmin - kmin_ex)
            return kpmin, err

    print("ERRORRRRRRRRRRRRRRRR")
    return


def LOS_DV(los_dir, los_ori, cir_rad, cir_z,
                            exact=[0,0,0], klim = np.inf):
    [D0, D1, D2] = los_ori
    [u0, u1, u2] = los_dir
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
    return

def LOS_LM(los_dir, los_org, cir_rad, cir_z, klim = np.inf,
                            exact=[0,0,0]):
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
        kpmin = (last_pout.z - origin.z) / direction.z

    return

def LOS_LM2(los_dir, los_ori, cir_rad, cir_z,
                            exact=[0,0,0]):
    [D0, D1, D2] = los_ori
    [u0, u1, u2] = los_dir
    uParN = np.sqrt(u0**2+u1**2)
    if uParN == 0.:
        kPMin = (cir_z-D2)/u2
    else:
        kPMin = _GG.comp_dist_los_circle2(u0, u1, u2,
                                          D0, D1, D2,
                                          cir_rad, cir_z)
    return

if __name__ == "__main__":
    # origin = [2, 0, 0]
    # direction = [0, 0, 1]
    # radius = 1.0
    # circ_z = 0.
    # exact = [2, 0, 0]
    # print("res for DV algo =", test_Dist_Cricle_LOS_DV(direction, origin,
    #                                                    radius, circ_z, exact))
    # print("res for LM algo =", test_Dist_Cricle_LOS_LM(direction, origin,
    #                                                    radius, circ_z, exact))
    # print("res for LM2 algo =", test_Dist_Cricle_LOS_LM2(direction, origin,
    #                                                    radius, circ_z, exact))

    # print("DV: ", timeit.timeit('LOS_DV(direction, origin, radius, circ_z, exact=exact)',
    #                     globals=globals()))
    # print("LM: ", timeit.timeit('LOS_LM(direction, origin, radius, circ_z, exact=exact)',
    #                     globals=globals()))
    # print("LM2: ", timeit.timeit('LOS_LM2(direction, origin, radius, circ_z, exact=exact)',
    #                     globals=globals()))

    # origin = [0, 0, 0]
    # direction = [1, 1, 0]
    # radius = 4.0
    # circ_z = 0.
    # exact = [np.cos(np.pi/4)*4, np.cos(np.pi/4)*4, 0]
    # print("res for DV algo =", test_Dist_Cricle_LOS_DV(direction, origin,
    #                                                    radius, circ_z, exact=exact))
    # print("res for LM algo =", test_Dist_Cricle_LOS_LM(direction, origin,
    #                                                    radius, circ_z, exact=exact))
    # print("res for LM2 algo =", test_Dist_Cricle_LOS_LM2(direction, origin,
    #                                                    radius, circ_z, exact=exact))
    # print("DV: ", timeit.timeit('LOS_DV(direction, origin, radius, circ_z, exact=exact)',
    #                     globals=globals()))
    # print("LM2: ", timeit.timeit('LOS_LM2(direction, origin, radius, circ_z, exact=exact)',
    #                     globals=globals()))

    origin = [-1, -1, -1]
    direction = [3./np.sqrt(11), 1./np.sqrt(11), 1./np.sqrt(11)]
    radius = 2.0
    circ_z = 0.
    exact = [2, 0, 0]
    print("res for DV algo =", test_Dist_Cricle_LOS_DV(direction, origin,
                                                       radius, circ_z, exact=exact))
    print("res for LM algo =", test_Dist_Cricle_LOS_LM(direction, origin,
                                                       radius, circ_z, exact=exact))
    print("res for LM2 algo =", test_Dist_Cricle_LOS_LM2(direction, origin,
                                                       radius, circ_z, exact=exact))
    print("DV: ", timeit.timeit('LOS_DV(direction, origin, radius, circ_z, exact=exact)',
                        globals=globals()))
    print("LM2: ", timeit.timeit('LOS_LM2(direction, origin, radius, circ_z, exact=exact)',
                        globals=globals()))
