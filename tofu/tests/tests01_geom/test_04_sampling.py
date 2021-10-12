"""Tests of the functions in `sampling_tools.pxd` or their wrappers found in
`tofu.geom`.
"""
import itertools
import numpy as np
import tofu as tf
import tofu.geom._GG as GG
from matplotlib.path import Path
from .testing_tools import compute_ves_norm
from .testing_tools import compute_min_max_r_and_z

# Global variables for testing
theta = np.linspace(0., 2. * np.pi, 100)
VPoly = np.array([2. + 1. * np.cos(theta), 0. + 1. * np.sin(theta)])
VIn = compute_ves_norm(VPoly)


# =============================================================================
# Vessel  - Line and VPoly sampling
# =============================================================================
def test01_ves_mesh_dlfroml():
    LMinMax = np.array([0., 10.])
    L, dLr, indL, N = GG.discretize_line1d(LMinMax, 20., DL=None,
                                           Lim=True, margin=1.e-9)

    assert np.allclose(L, [5.]) and dLr == 10. and np.allclose(indL, [0])
    assert N == 1
    L, dLr, indL, N = GG.discretize_line1d(LMinMax, 1., DL=None,
                                           Lim=True, margin=1.e-9)
    assert np.allclose(L, 0.5 + np.arange(0, 10)) and dLr == 1. and \
           np.allclose(indL, range(0, 10)) and N == 10
    DL = [2., 8.]
    L, dLr, indL, N = GG.discretize_line1d(LMinMax, 1., DL=DL,
                                           Lim=True, margin=1.e-9)
    assert np.allclose(L, 0.5 + np.arange(2, 8)) and dLr == 1. and \
           np.allclose(indL, range(2, 8)) and N == 10
    DL = [2., 12.]
    L, dLr, indL, N = GG.discretize_line1d(LMinMax, 1., DL=DL,
                                           Lim=True, margin=1.e-9)
    assert np.allclose(L, 0.5 + np.arange(2, 10)) and dLr == 1. and \
           np.allclose(indL, range(2, 10)) and N == 10
    DL = [2., 12.]
    L, dLr, indL, N = GG.discretize_line1d(LMinMax, 1., DL=DL,
                                           Lim=False, margin=1.e-9)
    assert np.allclose(L, 0.5 + np.arange(2, 12)) and dLr == 1. and \
           np.allclose(indL, range(2, 12)) and N == 10


def test02_discretize_vpoly(VPoly=VPoly):
    VIn = compute_ves_norm(VPoly)
    dL = 0.01

    PtsCross, dLr, ind, N, Rref, VPbis = GG.discretize_vpoly(VPoly, dL,
                                                             D1=None,
                                                             D2=None,
                                                             margin=1.e-9,
                                                             DIn=0., VIn=VIn)
    assert PtsCross.ndim == 2 and PtsCross.shape[1] >= VPoly.shape[1] - 1 and \
           not np.allclose(PtsCross[:, 0], PtsCross[:, -1])
    assert dLr.shape == (PtsCross.shape[1],) and np.all(dLr <= dL)
    assert ind.shape == (PtsCross.shape[1],)
    assert np.all(np.unique(ind) == ind)
    assert np.all(~np.isnan(ind))
    assert np.max(ind) < PtsCross.shape[1]
    assert N.shape == (VPoly.shape[1] - 1,) and np.all(N >= 1)
    assert Rref.shape == (PtsCross.shape[1],)
    assert np.all(Rref == PtsCross[0, :])
    assert VPbis.ndim == 2 and VPbis.shape[1] >= VPoly.shape[1]

    PtsCross, dLr, ind, N, Rref, VPbis = GG.discretize_vpoly(VPoly, dL,
                                                             D1=[0., 2.],
                                                             D2=[-2., 0.],
                                                             margin=1.e-9,
                                                             DIn=0.05, VIn=VIn)
    assert np.all(PtsCross[0, :] >= 0.) and np.all(PtsCross[0, :] <= 2.) and \
           np.all(PtsCross[1, :] >= -2.) and np.all(PtsCross[1, :] <= 0.)
    assert np.all(Path(VPoly.T).contains_points(PtsCross.T))
    assert dLr.shape == (PtsCross.shape[1],) and np.all(dLr <= dL)
    assert ind.shape == (PtsCross.shape[1],) and np.all(np.unique(ind) == ind)
    assert np.all(~np.isnan(ind))
    assert N.shape == (VPoly.shape[1] - 1,) and np.all(N >= 1)
    assert Rref.size > 3 * PtsCross.shape[1]
    assert VPbis.ndim == 2 and VPbis.shape[1] >= VPoly.shape[1]

    PtsCross, dLr, ind, N, Rref, VPbis = GG.discretize_vpoly(VPoly, dL,
                                                             D1=[0., 2.],
                                                             D2=[-2., 0.],
                                                             margin=1.e-9,
                                                             DIn=-0.05,
                                                             VIn=VIn)
    assert np.all(PtsCross[0, :] >= 0. - 0.05)
    assert np.all(PtsCross[0, :] <= 2.)
    assert np.all(PtsCross[1, :] >= -2. - 0.05)
    assert np.all(PtsCross[1, :] <= 0.)
    assert np.all(~Path(VPoly.T).contains_points(PtsCross.T))


# =============================================================================
# Ves  - Vmesh
# =============================================================================
def test03_Ves_Vmesh_Tor():
    r_min_max, z_min_max = compute_min_max_r_and_z(VPoly)
    dR, dZ, dRPhi = 0.05, 0.05, 0.05
    LDPhi = [None,
             [3. * np.pi / 4., 5. * np.pi / 4.],
             [-np.pi / 4., np.pi / 4.]]

    for ii in range(len(LDPhi)):
        out = GG._Ves_Vmesh_Tor_SubFromD_cython(dR, dZ, dRPhi,
                                                r_min_max,
                                                z_min_max,
                                                DR=[0.5, 2.],
                                                DZ=[0., 1.2],
                                                DPhi=LDPhi[ii],
                                                limit_vpoly=VPoly,
                                                out_format='(R,Z,Phi)',
                                                margin=1.e-9)
        pts, vol_res, ind, r_res, z_res, vec_phi_res, sz_r, sz_z = out
        assert pts.ndim == 2 and pts.shape[0] == 3
        assert np.all(pts[0, :] >= 1.) and np.all(pts[0, :] <= 2.), \
            " X coordinates not in right bounds"
        assert np.all(pts[1, :] >= 0.) and np.all(pts[1, :] <= 1.), \
            " Y coordinates not in right bounds"
        marg = np.abs(np.arctan(np.mean(vec_phi_res) / np.min(VPoly[1, :])))
        if LDPhi[ii] is not None:
            LDPhi[ii][0] = np.arctan2(np.sin(LDPhi[ii][0]),
                                      np.cos(LDPhi[ii][0]))
            LDPhi[ii][1] = np.arctan2(np.sin(LDPhi[ii][1]),
                                      np.cos(LDPhi[ii][1]))
            if LDPhi[ii][0] <= LDPhi[ii][1]:
                assert np.all((pts[2, :] >= LDPhi[ii][0] - marg) &
                              (pts[2, :] <= LDPhi[ii][1] + marg))
            else:
                assert np.all((pts[2, :] >= LDPhi[ii][0] - marg) |
                              (pts[2, :] <= LDPhi[ii][1] + marg))
        assert vol_res.shape == (pts.shape[1],)
        assert all([ind.shape == (pts.shape[1],),
                    ind.dtype == int,
                    np.unique(ind).size == ind.size,
                    np.all(ind == np.unique(ind)),
                    np.all(ind >= 0)])
        assert vec_phi_res.ndim == 1

        out = GG._Ves_Vmesh_Tor_SubFromInd_cython(dR, dZ, dRPhi,
                                                  r_min_max, z_min_max, ind,
                                                  Out='(R,Z,Phi)',
                                                  margin=1.e-9)
        Ptsi, dVi, dRri, dZri, dRPhiri = out
        assert np.allclose(pts, Ptsi)
        assert np.allclose(vol_res, dVi)
        assert r_res == dRri and z_res == dZri
        assert np.allclose(vec_phi_res, dRPhiri)


def test04_ves_vmesh_lin():
    XMinMax = np.array([0., 10.])
    YMinMax = np.array([np.min(VPoly[0, :]), np.max(VPoly[0, :])])
    ZMinMax = np.array([np.min(VPoly[1, :]), np.max(VPoly[1, :])])
    dX, dY, dZ = 0.05, 0.05, 0.05

    out = GG._Ves_Vmesh_Lin_SubFromD_cython(dX, dY, dZ, XMinMax,
                                            YMinMax, ZMinMax,
                                            DX=[8., 15.],
                                            DY=[0.5, 2.],
                                            DZ=[0., 1.2],
                                            limit_vpoly=VPoly,
                                            margin=1.e-9)
    Pts, dV, ind, dXr, dYr, dZr = out
    assert Pts.ndim == 2 and Pts.shape[0] == 3
    assert np.all(Pts[0, :] >= 8.) and np.all(Pts[0, :] <= 10.) and \
           np.all(Pts[1, :] >= 1.) and np.all(Pts[1, :] <= 2.) and \
           np.all(Pts[2, :] >= 0.) and np.all(Pts[2, :] <= 1.)
    assert all([ind.shape == (Pts.shape[1],), ind.dtype == int,
                np.unique(ind).size == ind.size,
                np.all(ind == np.unique(ind)), np.all(ind >= 0)])

    out = GG._Ves_Vmesh_Lin_SubFromInd_cython(dX, dY, dZ,
                                              XMinMax, YMinMax, ZMinMax,
                                              ind, margin=1.e-9)
    Ptsi, dVi, dXri, dYri, dZri = out
    assert np.allclose(Pts, Ptsi)
    assert np.allclose(dV, dVi)
    assert dXr == dXri and dYr == dYri and dZr == dZri


# =============================================================================
# Ves  - Solid angles
# =============================================================================
def test05_sa_integ_map(ves_poly=VPoly, debug=3):
    import matplotlib.pyplot as plt
    print()

    lblock = [False, True]
    lstep_rz = [0.015, 0.02]

    for (block, step_rz) in zip(lblock, lstep_rz):

        config = tf.load_config("A1")
        kwdargs = config.get_kwdargs_LOS_isVis()
        ves_poly = kwdargs["ves_poly"]

        # coordonnées en x,y,z:
        part = np.array([[3, 0.1, 0.1],
                         [3, 0.1, 0.]], order='F').T
        part_rad = np.r_[0.001,
                         0.0012]
        rstep = zstep = step_rz
        phistep = 0.08
        limits_r, limits_z = compute_min_max_r_and_z(ves_poly)
        DR = None
        DZ = None
        DPhi = [-0.1, 0.1]

        # -- Getting cython APPROX computation --------------------------------
        res = GG.compute_solid_angle_map(part, part_rad,
                                         rstep, zstep, phistep,
                                         limits_r, limits_z,
                                         DR=DR, DZ=DZ,
                                         DPhi=DPhi,
                                         block=block,
                                         limit_vpoly=ves_poly,
                                         **kwdargs,
                                         )
        pts, sa_map_cy, ind, reso_r_z = res

        # check sizes
        npts_ind = np.size(ind)
        dim, npts = np.shape(pts)
        npts_sa, sz_p = np.shape(sa_map_cy)

        if debug > 2:
            print(f"sa_map_cy is of size {npts_sa},{sz_p}")

        # Checking shapes, sizes, types
        assert dim == 2
        assert sz_p == np.shape(part)[1]
        assert npts_ind == npts
        assert npts == npts_sa
        assert isinstance(reso_r_z, float)

        # -- Getting cython EXACT computation ---------------------------------
        kwdargs = config.get_kwdargs_LOS_isVis()
        res = GG.compute_solid_angle_map(part, part_rad,
                                         rstep, zstep, phistep,
                                         limits_r, limits_z,
                                         approx=False,
                                         DR=DR, DZ=DZ,
                                         DPhi=DPhi,
                                         block=block,
                                         **kwdargs,
                                         limit_vpoly=ves_poly
                                         )
        pts_ex, sa_map_cy_ex, ind_ex, reso_r_z_ex = res

        if debug > 2:
            print(f"sa_map_cy_ex is of size {npts_sa},{sz_p}")

        # Checking if pts, indices and resolution same as with approx
        assert np.allclose(pts_ex, pts)
        assert np.allclose(ind_ex, ind)
        assert reso_r_z_ex == reso_r_z

        # ... Testing with python function ....................................
        res = GG._Ves_Vmesh_Tor_SubFromD_cython(rstep, zstep, phistep,
                                                limits_r, limits_z,
                                                limit_vpoly=ves_poly,
                                                DR=DR, DZ=DZ,
                                                DPhi=DPhi,
                                                out_format='(X,Y,Z)',
                                                margin=1.e-9)

        pts_xyz, dvol, ind_disc, reso_r, reso_z, reso_phi, sz_r, sz_z = res

        npts_disc = np.shape(pts_xyz)[1]

        sang = config.calc_solidangle_particle(pts_xyz,
                                               part,
                                               part_rad,
                                               block=block,
                                               approx=True)

        sang_ex = config.calc_solidangle_particle(pts_xyz,
                                                  part,
                                                  part_rad,
                                                  block=block,
                                                  approx=False)

        pts_disc = GG.coord_shift(pts_xyz, in_format='(X,Y,Z)',
                                  out_format='(R,Z)')

        assert (npts_disc, sz_p) == np.shape(sang)
        assert npts_disc >= npts
        assert reso_r * reso_z == reso_r_z
        if ves_poly is None:
            assert npts_sa == sz_z * sz_r, f"sizes r and z = {sz_r}{sz_z}"

        sa_map_py = np.zeros((npts_sa, sz_p))
        sa_map_py_ex = np.zeros((npts_sa, sz_p))

        r0 = limits_r[0]
        z0 = limits_z[0]

        lvisited = []
        for ii in range(npts_disc):
            i_r = int(np.abs(r0 - pts_disc[0, ii]) // reso_r)
            i_z = int(np.abs(z0 - pts_disc[1, ii]) // reso_z)
            ind_pol = int(i_r * sz_z + i_z)
            if ves_poly is not None:
                wh_eq = np.where(ind == ind_pol)
                ind_pol_u = wh_eq[0][0]
            else:
                ind_pol_u = ind_pol
            if ind_pol_u not in lvisited:
                lvisited.append(ind_pol_u)
            for pp in range(sz_p):
                sa_map_py[ind_pol_u, pp] += sang[ii, pp] * reso_phi[i_r]
                sa_map_py_ex[ind_pol_u, pp] += sang_ex[ii, pp] * reso_phi[i_r]

        if debug > 0:
            for pp in range(sz_p):
                fig = plt.figure(figsize=(14, 8))
                ax = plt.subplot(121)
                # import pdb; pdb.set_trace()
                ax.scatter(pts[0, :], pts[1, :],
                           marker="s", edgecolors="None",
                           s=10, c=sa_map_cy[:, pp].flatten(),
                           vmin=0, vmax=max(sa_map_cy[:, pp].max(),
                                            sa_map_py[:, pp].max()))
                ax.set_title("cython function")
                ax = plt.subplot(122)
                im = ax.scatter(pts[0, :], pts[1, :],
                                marker="s", edgecolors="None",
                                s=10, c=sa_map_py[:, pp].flatten(),
                                vmin=0, vmax=max(sa_map_cy[:, pp].max(),
                                                 sa_map_py[:, pp].max()))
                fig.colorbar(im, ax=ax)
                ax.set_title("python reconstruction")
                plt.savefig("comparaison_part"+str(pp))

        max_py = np.max(sa_map_py)
        max_cy = np.max(sa_map_cy)
        max_py_ex = np.max(sa_map_py_ex)
        max_cy_ex = np.max(sa_map_cy_ex)

        err = np.abs(sa_map_py - sa_map_cy) / max(max_py, max_cy)
        print("max error approx py vs cy =", np.max(err))

        err = np.abs(sa_map_py_ex - sa_map_cy_ex) / max(max_py_ex, max_cy_ex)
        print("max error exacts py vs cy =", np.max(err))

        err = np.abs(sa_map_py_ex - sa_map_py) / max(max_py, max_py_ex)
        print("max error python approx vs exact =", np.max(err))

        err = np.abs(sa_map_cy_ex - sa_map_cy) / max(max_cy, max_cy_ex)
        print("max error cython approx vs exact =", np.max(err))

        assert np.allclose(sa_map_cy, sa_map_py, atol=1e-16, rtol=1e-16,
                           equal_nan=True)
        assert np.allclose(sa_map_cy_ex, sa_map_py_ex, atol=1e-16, rtol=1e-16,
                           equal_nan=True)
        assert np.allclose(sa_map_cy, sa_map_cy_ex, atol=1e-14, rtol=1e-16,
                           equal_nan=True)
        assert np.allclose(sa_map_py, sa_map_py_ex, atol=1e-14, rtol=1e-16,
                           equal_nan=True)
    # ...
    return


def test06_sa_integ_poly_map(ves_poly=VPoly, debug=3):
    import matplotlib.pyplot as plt
    print()

    config = tf.load_config("A1")
    for name_pfc in ['BaffleV0', "DivUpV1", "DivLowITERV1"]:
        try:
            config.remove_Struct("PFC", name_pfc)
        except:
            pass

    kwdargs = config.get_kwdargs_LOS_isVis()
    ves_poly = kwdargs["ves_poly"]

    if debug > 2:
        config.plot()
        fig = plt.gcf()
        fig.savefig("configuration")

    # coordonnées en x,y,z:
    poly_coords = [
        np.array([
            [2.7, 0, -0.4],
            [2.75, 0, -0.4],
            [2.75, 0, -0.1],
            [2.70, 0, -0.1],
            [2.60, 0, -0.1],
            [2.50, 0, -0.1],
            [2.50, 0, -0.4],
        ]).T,  # 1st polygon
        np.array([
            [2.50, 0, -0.1],
            [2.60, 0, -0.1],
            [2.70, 0, -0.1],
            [2.75, 0, -0.1],
            [2.75, 0, -0.4],
            [2.7, 0, -0.4],
            [2.50, 0, -0.4],
        ]).T,  # 2nd polygon
        np.array([
            [2.60, 0, -0.1],
            [2.70, 0, -0.1],
            [2.75, 0, -0.1],
            [2.75, 0, -0.4],
            [2.7, 0, -0.4],
            [2.50, 0, -0.4],
            [2.50, 0, -0.1],
        ]).T,  # 3rd polygon
        np.array([
            [2.70, 0, -0.1],
            [2.75, 0, -0.1],
            [2.75, 0, -0.4],
            [2.7, 0, -0.4],
            [2.50, 0, -0.4],
            [2.50, 0, -0.1],
            [2.60, 0, -0.1],
        ]).T,  # 4th polygon
        np.array([
            [2.75, 0, -0.1],
            [2.75, 0, -0.4],
            [2.7, 0, -0.4],
            [2.50, 0, -0.4],
            [2.50, 0, -0.1],
            [2.60, 0, -0.1],
            [2.70, 0, -0.1],
        ]).T,  # 5th polygon
        np.array([
            [2.2, 0., 0.25],
            [2.2, 0., 0.50],
            [2.5, 0., 0.50],
            [2.7, 0., 0.50],
            [3.0, 0., 0.50],
            [3.0, 0., 0.25],
            [2.6, 0., 0.25],
        ]).T, # 6th polygon
    ]
    poly_coords = [np.ascontiguousarray(poly) for poly in poly_coords]
    poly_lnorms = np.array([
        [0, 1., 0],
        [0, 1., 0],
        [0, 1., 0],
        [0, 1., 0],
        [0, 1., 0],
        [0, 1., 0],
    ])
    poly_lnvert = np.array([poly.shape[1] for poly in poly_coords])
    limits_r, limits_z = compute_min_max_r_and_z(ves_poly)

    lblock = [False]  #, True]
    lstep_rz = [
        #  0.01,
        0.01,
    ]

    test_cases = list(itertools.product(lblock, lstep_rz))

    for (block, step_rz) in test_cases:

        rstep = zstep = step_rz
        zstep = step_rz
        phistep = 0.01
        DR = None  # [2.45, 2.8]
        DZ = None  # [-0.5, 0.]
        DPhi = None  # [-0.1, 0.1]

        # -- Getting cython APPROX computation --------------------------------
        res = GG.compute_solid_angle_poly_map(
            poly_coords,
            poly_lnorms,
            poly_lnvert,
            rstep, zstep, phistep,
            limits_r, limits_z,
            DR=DR, DZ=DZ,
            DPhi=DPhi,
            block=block,
            limit_vpoly=ves_poly,
            **kwdargs,
        )
        pts, sa_map_cy, ind, reso_r_z = res

        # check sizes
        npts_ind = np.size(ind)
        dim, npts = np.shape(pts)
        npts_sa, npoly = np.shape(sa_map_cy)

        if debug > 2:
            print(f"sa_map_cy is of size {npts_sa},{npoly}")

        # Checking shapes, sizes, types
        assert dim == 2
        assert npoly == len(poly_coords), str(len(poly_coords))
        assert npts_ind == npts
        assert npts == npts_sa
        assert isinstance(reso_r_z, float)

        if debug > 0:
            for pp in range(npoly):
                plt.clf()
                fig = plt.figure()
                ax = plt.subplot(111)
                # import pdb; pdb.set_trace()
                im = ax.scatter(pts[0, :], pts[1, :],
                                marker="s", edgecolors="None",
                                s=40, c=sa_map_cy[:, pp].flatten(),
                                vmin=sa_map_cy[:, pp].min(),
                                vmax=sa_map_cy[:, pp].max())
                poly = poly_coords[pp]
                xpoly = np.sqrt(poly[0]**2 + poly[1]**2)
                zpoly = poly[2]
                ax.plot(
                    xpoly[[iii for iii in range(np.size(xpoly))]+[0]],
                    zpoly[[iii for iii in range(np.size(xpoly))]+[0]],
                    "r-", marker='o',
                    linewidth=2,
                )
                # for iii in range(np.size(xpoly)):
                #     ax.annotate(iii,
                #                 (xpoly[iii],
                #                  zpoly[iii]))
                ax.plot()
                ax.set_title("cython function")
                fig.colorbar(im, ax=ax)
                plt.savefig("sa_map_poly" + str(pp)
                            + "_block" + str(block)
                            + "_steprz" + str(step_rz).replace(".", "_"))

        # max_py = np.max(sa_map_py)
        # max_cy = np.max(sa_map_cy)
        # max_py_ex = np.max(sa_map_py_ex)
        # max_cy_ex = np.max(sa_map_cy_ex)

        # err = np.abs(sa_map_py - sa_map_cy) / max(max_py, max_cy)
        # print("max error approx py vs cy =", np.max(err))

        # err = np.abs(sa_map_py_ex - sa_map_cy_ex) / max(max_py_ex, max_cy_ex)
        # print("max error exacts py vs cy =", np.max(err))

        # err = np.abs(sa_map_py_ex - sa_map_py) / max(max_py, max_py_ex)
        # print("max error python approx vs exact =", np.max(err))

        # err = np.abs(sa_map_cy_ex - sa_map_cy) / max(max_cy, max_cy_ex)
        # print("max error cython approx vs exact =", np.max(err))

        # assert np.allclose(sa_map_cy, sa_map_py, atol=1e-16, rtol=1e-16,
        #                    equal_nan=True)
        # assert np.allclose(sa_map_cy_ex, sa_map_py_ex, atol=1e-16, rtol=1e-16,
        #                    equal_nan=True)
        # assert np.allclose(sa_map_cy, sa_map_cy_ex, atol=1e-14, rtol=1e-16,
        #                    equal_nan=True)
        # assert np.allclose(sa_map_py, sa_map_py_ex, atol=1e-14, rtol=1e-16,
        #                    equal_nan=True)
    # ...
    return
