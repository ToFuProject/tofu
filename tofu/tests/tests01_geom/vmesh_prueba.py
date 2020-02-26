# External modules
import os
import timeit
import numpy as np
import matplotlib
import tofu.geom as tfg
import tofu.geom._GG as GG
import time

matplotlib.use("agg")
# Nose-specific
_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = "tofu.geom.tests03_core"
keyVers = "Vers"
_Exp = "WEST"


def bigger_test():
    """ exactly like test13_get_sampleV(self) """

    path = os.path.join(_here, "tests03_core_data")
    lf = os.listdir(path)
    lf = [f for f in lf if all([s in f for s in [_Exp, ".txt"]])]
    lCls = sorted(set([f.split("_")[1] for f in lf]))
    dobj = {"Tor": {}}  # , "Lin": {}}
    for tt in dobj.keys():
        for cc in lCls:
            lfc = [f for f in lf if f.split("_")[1] == cc and "V0" in f]
            ln = []
            for f in lfc:
                if "CoilCS" in f:
                    ln.append(f.split("_")[2].split(".")[0])
                else:
                    ln.append(f.split("_")[2].split(".")[0])
            lnu = sorted(set(ln))
            if not len(lnu) == len(ln):
                msg = "Non-unique name list for {0}:".format(cc)
                msg += "\n    ln = [{0}]".format(", ".join(ln))
                msg += "\n    lnu = [{0}]".format(", ".join(lnu))
                raise Exception(msg)
            dobj[tt][cc] = {}
            for ii in range(0, len(ln)):
                if "BumperOuter" in ln[ii]:
                    Lim = np.r_[10.0, 20.0] * np.pi / 180.0
                elif "BumperInner" in ln[ii]:
                    t0 = np.arange(0, 360, 60) * np.pi / 180.0
                    Dt = 5.0 * np.pi / 180.0
                    Lim = (
                        t0[np.newaxis, :] + Dt * np.r_[-1.0, 1.0][:, np.newaxis]
                    )  # noqa
                elif "Ripple" in ln[ii]:
                    t0 = np.arange(0, 360, 30) * np.pi / 180.0
                    Dt = 2.5 * np.pi / 180.0
                    Lim = (
                        t0[np.newaxis, :] + Dt * np.r_[-1.0, 1.0][:, np.newaxis]
                    )  # noqa
                elif "IC" in ln[ii]:
                    t0 = np.arange(0, 360, 120) * np.pi / 180.0
                    Dt = 10.0 * np.pi / 180.0
                    Lim = (
                        t0[np.newaxis, :] + Dt * np.r_[-1.0, 1.0][:, np.newaxis]
                    )  # noqa
                elif "LH" in ln[ii]:
                    t0 = np.arange(-180, 180, 120) * np.pi / 180.0
                    Dt = 10.0 * np.pi / 180.0
                    Lim = (
                        t0[np.newaxis, :] + Dt * np.r_[-1.0, 1.0][:, np.newaxis]
                    )  # noqa
                elif tt == "Lin":
                    Lim = np.r_[0.0, 10.0]
                else:
                    Lim = None
                Poly = np.loadtxt(os.path.join(path, lfc[ii]))
                assert Poly.ndim == 2
                assert Poly.size >= 2 * 3
                kwd = dict(  # noqa
                    Name=ln[ii] + tt,
                    Exp=_Exp,
                    SavePath=_here,
                    Poly=Poly,
                    Lim=Lim,
                    Type=tt,
                )  # noqa
                dobj[tt][cc][ln[ii]] = eval("tfg.%s(**kwd)" % cc)
    for typ in dobj.keys():
        # Todo : introduce possibility of choosing In coordinates !
        for c in dobj[typ].keys():
            if issubclass(eval('tfg.%s' % c), tfg._core.StructOut):
                continue
            for n in dobj[typ][c].keys():
                print("\n For type = " + str(typ) + " c = " + str(c)
                      + " n = ", n)
                obj = dobj[typ][c][n]
                print("obj = ", obj)
                box = None  # [[2.,3.], [0.,5.], [0.,np.pi/2.]]
                try:
                    ii = 0
                    reso = 0.02
                    start = time.perf_counter()
                    out = obj.get_sampleV(reso, resMode='abs', DV=box,
                                          Out='(X,Y,Z)')
                    print("NEW sample V total time = ", time.perf_counter() - start)
                    pts0, ind = out[0], out[2]
                    # start = time.perf_counter()
                    # out = obj.get_sampleV(reso, resMode='abs', DV=box,
                    #                       Out='(X,Y,Z)', algo="old")
                    # print("OLD sample V total time = ", time.perf_counter() - start)
                    # pts1, ind1 = out[0], out[2]
                    # assert np.allclose(ind1, ind)
                    ii = 1
                    start = time.perf_counter()
                    out = obj.get_sampleV(reso, resMode='abs', ind=ind,
                                          Out='(X,Y,Z)', num_threads=48)
                    print("NEW sample V total time = ", time.perf_counter() - start)
                    pts4 = out[0]
                    start = time.perf_counter()
                    out = obj.get_sampleV(reso, resMode='abs', ind=ind,
                                          Out='(X,Y,Z)', algo="old")
                    print("OLD sample V total time = ", time.perf_counter() - start)
                    pts3 = out[0]
                except Exception as err:
                    msg = str(err)
                    msg += "\nFailed for {0}_{1}_{2}".format(typ, c, n)
                    msg += "\n    ii={0}".format(ii)
                    msg += "\n    Lim={0}".format(str(obj.Lim))
                    msg += "\n    DS={0}".format(str(box))
                    raise Exception(msg)

                if type(pts0) is list:
                    # assert all([np.allclose(pts0[ii], pts1[ii])
                    #             for ii in range(0, len(pts0))])
                    assert all([np.allclose(pts3[ii], pts4[ii])
                                for ii in range(0, len(pts3))])
                    assert all([np.allclose(pts0[ii], pts4[ii])
                                for ii in range(0, len(pts0))])
                else:
                    # assert np.allclose(pts0, pts1)
                    assert np.allclose(pts3, pts4)
                    assert np.allclose(pts0, pts4)


def small_test():
    """Test vmesh"""

    # VPoly
    thet = np.linspace(0.0, 2.0 * np.pi, 100)
    VPoly = np.array([2.0 + 1.0 * np.cos(thet), 0.0 + 1.0 * np.sin(thet)])
    RMinMax = np.array([np.min(VPoly[0, :]), np.max(VPoly[0, :])])
    ZMinMax = np.array([np.min(VPoly[1, :]), np.max(VPoly[1, :])])
    dR, dZ, dRPhi = 0.025, 0.025, 0.025
    LDPhi = [
        None,  # noqa
        [3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0],  # noqa
        [-np.pi / 4.0, np.pi / 4.0],
    ]  # noqa
    for ii in range(0, len(LDPhi)):
        Pts, dV, ind, dRr, dZr, dRPhir = GG._Ves_Vmesh_Tor_SubFromD_cython(
            dR,
            dZ,
            dRPhi,
            RMinMax,
            ZMinMax,
            DR=np.array([0.5, 2.0]),
            DZ=np.array([0.0, 1.2]),
            DPhi=LDPhi[ii],
            VPoly=VPoly,
            Out="(R,Z,Phi)",
            margin=1.0e-9,
        )
        assert Pts.ndim == 2 and Pts.shape[0] == 3
        assert np.all(Pts[0, :] >= 1.0)
        assert np.all(Pts[0, :] <= 2.0)
        assert np.all(Pts[1, :] >= 0.0)
        assert np.all(Pts[1, :] <= 1.0)
        marg = np.abs(np.arctan(np.mean(dRPhir) / np.min(VPoly[1, :])))
        if not LDPhi[ii] is None:
            LDPhi[ii][0] = np.arctan2(
                np.sin(LDPhi[ii][0]), np.cos(LDPhi[ii][0])
            )  # noqa
            LDPhi[ii][1] = np.arctan2(
                np.sin(LDPhi[ii][1]), np.cos(LDPhi[ii][1])
            )  # noqa
            if LDPhi[ii][0] <= LDPhi[ii][1]:
                assert np.all(Pts[2, :] >= LDPhi[ii][0] - marg)
                assert np.all(Pts[2, :] <= LDPhi[ii][1] + marg)
            else:
                assert np.all(
                    (Pts[2, :] >= LDPhi[ii][0] - marg)  # noqa
                    | (Pts[2, :] <= LDPhi[ii][1] + marg)  # noqa
                )
        assert dV.shape == (Pts.shape[1],)
        assert ind.shape == (Pts.shape[1],)
        assert ind.dtype == int
        assert np.unique(ind).size == ind.size
        assert np.all(ind == np.unique(ind))
        assert np.all(ind >= 0)
        assert all(
            [
                ind.shape == (Pts.shape[1],),
                ind.dtype == int,
                np.unique(ind).size == ind.size,
                np.all(ind == np.unique(ind)),
                np.all(ind >= 0),
            ]
        )
        assert dRPhir.ndim == 1
        Ptsi, dVi, dRri, dZri, dRPhiri = GG._Ves_Vmesh_Tor_SubFromInd_cython(
            dR, dZ, dRPhi, RMinMax, ZMinMax, ind, Out="(R,Z,Phi)", margin=1.0e-9
        )
        assert np.allclose(Pts, Ptsi)
        assert np.allclose(dV, dVi)
        assert dRr == dRri and dZr == dZri
        assert np.allclose(dRPhir, dRPhiri)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing vmesh algo")
    parser.add_argument(
        "-m",
        "--mode",
        help="small, big or timeit",
        required=False,
        choices=["big", "small"],
        default="small",
    )
    parser.add_argument('--timeit', dest='timeit', action='store_true')
    parser.add_argument('--no-timeit', dest='timeit', action='store_false')
    parser.set_defaults(timeit=False)
    args = parser.parse_args()
    print(".-.-.-.-.-.-.-. ", args.mode, " .-.-.-.-.-.-.-.-")
    if args.mode.lower() == "small":
        if args.timeit:
            print(
                timeit.timeit(
                    "small_test()",
                    setup="from __main__ import small_test",
                    number=500,  # fmt: off
                )
            )
        else:
            small_test()
    elif args.mode.lower() == "big":
        if args.timeit:
            print(
                timeit.timeit(
                    "bigger_test()",
                    setup="from __main__ import bigger_test",
                    number=50,  # fmt: off
                )
            )
        else:
            print(".................... ONE CALL ...................")
            bigger_test()
