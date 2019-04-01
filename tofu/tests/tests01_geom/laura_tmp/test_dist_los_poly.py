# coding: utf-8
from tofu_LauraBenchmarck_load_config import *
import tofu.geom._GG as _GG
import time
import matplotlib.pyplot as plt
# #import line_profiler
# import pstats, cProfile
# from pathlib import Path
# from resource import getpagesize
# import os
# import psutil
# #from memory_profiler import profile

Cams = ["V1", "V10", "V100", "V1000", "V10000",
        "V100000", "V1000000"]
CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000",
         "VA100000"]
CamsA3 = ["V31", "V310", "V3100", "V31000", "V310000",
         "V3100000"]

def prepare_inputs(vcam, config, method='ref'):

    D, u, loscam = get_Du(vcam)
    u = u/np.sqrt(np.sum(u**2,axis=0))[np.newaxis,:]
    D = np.ascontiguousarray(D)
    u = np.ascontiguousarray(u)

    # Get reference
    lS = config.lStruct
    lSIn = [ss for ss in lS if ss._InOut=='in']
    if len(lSIn)==0:
        msg = "self.config must have at least a StructIn subclass !"
        assert len(lSIn)>0, msg
    elif len(lSIn)>1:
        S = lSIn[np.argmin([ss.dgeom['Surf'] for ss in lSIn])]
    else:
        S = lSIn[0]
    VPoly = S.Poly_closed
    VType = config.Id.Type

    largs = [D, u, VPoly]
    dkwd = dict(ves_type=VType,
                eps_uz=1.e-6, eps_vz=1.e-9,
                eps_a=1.e-9, eps_b=1.e-9, eps_plane=1.e-9)

    return largs, dkwd


num_rays = 11
def artificial_case():
    ves_poly = np.zeros((2, 9))
    ves_poly0 = [2, 3, 4, 5, 5, 4, 3, 2, 2]
    ves_poly1 = [2, 1, 1, 2, 3, 4, 4, 3, 2]
    ves_poly[0] = np.asarray(ves_poly0)
    ves_poly[1] = np.asarray(ves_poly1)
    # rays :
    ray_orig = np.zeros((3,num_rays))
    ray_vdir = np.zeros((3,num_rays))
    # ray 0 :
    ray_orig[0][0] = 0
    ray_orig[2][0] = 5
    ray_vdir[0][0] = 1
    # ray 1 :
    ray_orig[0][1] = 3.5
    ray_orig[2][1] = 5
    ray_vdir[0][1] = 1
    # ray 2 :
    ray_orig[0][2] = 3.5
    ray_orig[2][2] = 5
    ray_orig[1][2] = -1
    ray_vdir[0][2] = -1
    # ray 3:
    ray_orig[0][3] = 4
    ray_orig[2][3] = -1
    ray_vdir[0][3] = 1
    ray_vdir[2][3] = 1
    # ray 4:
    ray_orig[0][4] = 7
    ray_orig[2][4] = 3
    ray_vdir[0][4] = 1
    ray_vdir[2][4] = 1
    # ray 5:
    ray_orig[0][5] = 6
    ray_orig[2][5] = 2.4
    ray_orig[1][5] = -1.3
    ray_vdir[1][5] = 1
    ray_vdir[2][5] = 0.01
    # ray 6:
    ray_orig[0][6] = 0.
    ray_orig[1][6] = 0.
    ray_orig[2][6] = -1.
    ray_vdir[2][6] = 0.5
    # ray 7:
    ray_orig[0][7] = 0.
    ray_orig[1][7] = 0.
    ray_orig[2][7] = 4.
    ray_vdir[2][7] = -1.
    # ray 8:
    ray_orig[0][8] = 1.
    ray_orig[1][8] = 0.
    ray_orig[2][8] = 2.
    ray_vdir[2][8] = -1.
    # ray 9:
    ray_orig[0][9] = 3.5
    ray_orig[1][9] = 0.
    ray_orig[2][9] = 0.5
    ray_vdir[2][9] = -1.
    # ray 10:
    ray_orig[0][10] = 5.5
    ray_orig[1][10] = 0.
    ray_orig[2][10] = 2.5
    ray_vdir[0][10] = 1.

    ray_min = 0#num_rays-1
    ray_max = num_rays-1
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    ax1 = plt.subplot(121)
    ax1.plot(ves_poly0, ves_poly1)
    ax1.scatter(0, 0, 10)
    for i in range(ray_min,ray_max+1):
        ax1.plot([ray_orig[0][i],
                  ray_orig[0][i] + ray_vdir[0][i]],
                 [ray_orig[2][i],
                  ray_orig[2][i] + ray_vdir[2][i]],
                 linewidth=2.0, label="ray"+str(i))
        ax1.scatter(ray_orig[0][i], ray_orig[2][i], 10)
    ax1.legend()
    # plt.show(block=True)
    # plt.clf()
    theta = np.linspace(0, 2.*np.pi, 100)
    rmin = ves_poly[0][5]#min(ves_poly[0])
    rmax = ves_poly[0][6]#max(ves_poly[0])
    print("rmin, rmax =", rmin, rmax)
    #plt.show()
    ax2 = plt.subplot(122)
    ax2.plot(rmin * np.cos(theta), rmin * np.sin(theta))
    ax2.plot(rmax * np.cos(theta), rmax * np.sin(theta))
    ax2.plot(5 * np.cos(theta), 5 * np.sin(theta))
    ax2.plot(2 * np.cos(theta), 2 * np.sin(theta))
    for i in range(ray_min,ray_max+1):
        ax2.plot([ray_orig[0][i],
                  ray_orig[0][i] + ray_vdir[0][i]],
                 [ray_orig[1][i],
                  ray_orig[1][i]  + ray_vdir[1][i]],
                 linewidth=2.0, label="ray"+str(i))
        ax2.scatter(ray_orig[0][i], ray_orig[1][i], 10)
    # foundk = 1.414213562373094
    # ax1.scatter(ray_orig[0][ray_min] + ray_vdir[0][ray_min] * foundk,
    #             ray_orig[2][ray_min] + ray_vdir[2][ray_min] * foundk, 15)
    # ax2.scatter(ray_orig[0][ray_min] + ray_vdir[0][ray_min] * foundk,
    #             ray_orig[1][ray_min] + ray_vdir[1][ray_min] * foundk, 15)
    ax2.legend()
    plt.show()
    # out :
    print("************************************************")
    print(" Oris => \n", ray_orig[:,ray_min:ray_max+1])
    print(" Dirs => \n", ray_vdir[:,ray_min:ray_max+1])
    print("************************************************")
    out = _GG.comp_dist_los_vpoly(np.ascontiguousarray(ray_orig[:,ray_min:ray_max+1], dtype=np.float64),
                                  np.ascontiguousarray(ray_vdir[:,ray_min:ray_max+1], dtype=np.float64),
                                  ves_poly, num_threads=1)

    return out

def test_LOS_west_configs(config="A1", cams=["V10"]):
    dconf = load_config(config)
    out_tot = []
    for vcam in cams:
        largs, dkwd = prepare_inputs(vcam, dconf)
        start = time.time()
        out = _GG.comp_dist_los_vpoly(*largs, **dkwd)
        elapsed = time.time() - start
        out_tot.append(out)
    return out_tot

if __name__ == "__main__":
    # out = test_LOS_west_configs()
    # print(out[0][0])
    # print(out[0][1])

    out = artificial_case()
    np.set_printoptions(precision=3)
    for i in range(num_rays):
        print("For Ray =", i, " k =", out[0][i], ", distance =", out[1][i])
    plt.show(block=True)
