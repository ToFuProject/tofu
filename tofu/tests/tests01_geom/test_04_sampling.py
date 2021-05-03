"""Tests of the functions in `sampling_tools.pxd` or their wrappers found in
`tofu.geom`.
"""
import numpy as np
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
    assert np.all(PtsCross[0, :] >= 0. - 0.05) and np.all(PtsCross[0, :] <= 2.)
    assert np.all(PtsCross[1, :] >= -2. - 0.05) and np.all(PtsCross[1, :] <= 0.)
    assert np.all(~Path(VPoly.T).contains_points(PtsCross.T))


# =============================================================================
# Ves  - Vmesh
# =============================================================================
def test03_Ves_Vmesh_Tor():
    r_min_max, z_min_max = compute_min_max_r_and_z(VPoly)
    dR, dZ, dRPhi = 0.05, 0.05, 0.05
    LDPhi = [None, [3. * np.pi / 4., 5. * np.pi / 4.], [-np.pi / 4., np.pi / 4.]]

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

    Pts, dV, ind, \
    dXr, dYr, dZr = GG._Ves_Vmesh_Lin_SubFromD_cython(dX, dY, dZ, XMinMax,
                                                      YMinMax, ZMinMax,
                                                      DX=[8., 15.],
                                                      DY=[0.5, 2.],
                                                      DZ=[0., 1.2],
                                                      limit_vpoly=VPoly,
                                                      margin=1.e-9)
    assert Pts.ndim == 2 and Pts.shape[0] == 3
    assert np.all(Pts[0, :] >= 8.) and np.all(Pts[0, :] <= 10.) and \
           np.all(Pts[1, :] >= 1.) and np.all(Pts[1, :] <= 2.) and \
           np.all(Pts[2, :] >= 0.) and np.all(Pts[2, :] <= 1.)
    assert all([ind.shape == (Pts.shape[1],), ind.dtype == int,
                np.unique(ind).size == ind.size,
                np.all(ind == np.unique(ind)), np.all(ind >= 0)])

    Ptsi, dVi, \
    dXri, dYri, \
    dZri = GG._Ves_Vmesh_Lin_SubFromInd_cython(dX, dY, dZ,
                                               XMinMax, YMinMax, ZMinMax,
                                               ind, margin=1.e-9)
    assert np.allclose(Pts, Ptsi)
    assert np.allclose(dV, dVi)
    assert dXr == dXri and dYr == dYri and dZr == dZri


# =============================================================================
# Ves  - Solid angles
# =============================================================================
def test25_sa_integ_map(ves_poly=VPoly, debug=1):
    import tofu.geom as tfg
    import matplotlib.pyplot as plt

    block = False

    ves = tfg.Ves(
        Name="DebugVessel",
        Poly=ves_poly,
        Type="Tor",
        Exp="Misc",
        shot=0
    )

    config = tfg.Config(Name="test",
                        Exp="Misc",
                        lStruct=[ves])

    part = np.array([[10., -0.0, 0]], order='F').T
    part = np.array([[10., 0.0, 0]], order='F').T
    part_rad = np.r_[0.01]
    rstep = zstep = 0.1
    phistep = 0.1
    limits_r, limits_z = compute_min_max_r_and_z(ves_poly)
    DR = None # [1.75, 2.]
    DZ = None # [-.25, 0.25]
    DPhi = None # [-0.01, 0.01]

    kwdargs = config.get_kwdargs_LOS_isVis()
    vpoly = None#kwdargs["ves_poly"]
    res = GG.compute_solid_angle_map(part, part_rad,
                                     rstep, zstep, phistep,
                                     limits_r, limits_z,
                                     DR=DR, DZ=DZ,
                                     DPhi=DPhi,
                                     block=block,
                                     **kwdargs,
                                     limit_vpoly=vpoly
                                     )
    pts, sa_map, ind, reso_r_z  = res

    # check sizes
    npts_ind = np.size(ind)
    dim, npts = np.shape(pts)
    npts_sa, sz_p = np.shape(sa_map)

    # if debug > 0:
    #     print(f"sa_map is of size {npts_sa},{sz_p}")

    # Checking shapes, sizes, types
    assert dim == 2
    assert sz_p == np.shape(part)[1]
    assert npts == npts_sa
    assert isinstance(reso_r_z, float)

    # ... Testing with exact function .........................................
    res = GG._Ves_Vmesh_Tor_SubFromD_cython(rstep, zstep, phistep,
                                            limits_r, limits_z,
                                            limit_vpoly=vpoly,
                                            DR=DR, DZ=DZ,
                                            DPhi=DPhi,
                                            out_format='(R,Z,Phi)',
                                            margin=1.e-9)

    pts_disc, dvol, ind, reso_r, reso_z, reso_phi, sz_r, sz_z = res

    npts_disc = np.shape(pts_disc)[1]

    sang = config.calc_solidangle_particle(pts_disc,
                                           part,
                                           part_rad,
                                           block=block,
                                           approx=True)

    sang_ex = config.calc_solidangle_particle(pts_disc,
                                              part,
                                              part_rad,
                                              block=block,
                                              approx=False)

    if debug > 0:
        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(121)
        ax.plot(pts_disc[0, :], pts_disc[1, :], '.b')
        ax.plot(part[0, :], part[1, :], '*r')
        ax2 = plt.subplot(122)
        ax2.plot(pts[0, :], pts[1, :], '.b')
        ax2.plot(part[0, :], part[1, :], '*r')
        if vpoly is not None:
            ax2.plot(vpoly[0, :], vpoly[1, :])
        fig.suptitle("Discretization points and particle traj")
        plt.savefig("discretization_and_traj")

    assert (npts_disc, sz_p) == np.shape(sang)
    assert npts_disc >= npts
    assert reso_r * reso_z == reso_r_z
    if vpoly is None:
        assert npts_sa == sz_z * sz_r, f"sizes r and z = {sz_r}{sz_z}"

    sa_map_py = np.zeros((npts_sa, sz_p))
    sa_map_py_ex = np.zeros((npts_sa, sz_p))

    r0 = pts_disc[0, 0]
    z0 = pts_disc[1, 0]

    for ii in range(npts_disc):
        i_r = int(np.round(np.abs(r0 - pts_disc[0, ii]) / reso_r))
        i_z = int(np.round(np.abs(z0 - pts_disc[1, ii]) / reso_z))
        ind_pol = int(i_r * sz_z + i_z)
        for pp in range(sz_p):
            sa_map_py[ind_pol, pp] += sang[ii, pp] * reso_phi[i_r]
            sa_map_py_ex[ind_pol, pp] += sang_ex[ii, pp] * reso_phi[i_r]

    if debug > 0:
        print("")
        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(121)
        # import pdb; pdb.set_trace()
        ax.scatter(pts[0, :], pts[1, :],
                   marker="s", edgecolors="None",
                   s=10, c=sa_map.flatten(),
                   vmin=0, vmax=max(sa_map.max(), sa_map_py.max()))
        ax.set_title("cython function")
        ax = plt.subplot(122)
        ax.scatter(pts[0, :], pts[1, :],
                   marker="s", edgecolors="None",
                   s=10, c=sa_map_py.flatten(),
                   vmin=0, vmax=max(sa_map.max(), sa_map_py.max()))
        ax.set_title("python reconstruction")
        plt.savefig("comparaison")

    print("max error approx =", np.max(np.abs(sa_map_py - sa_map)/sa_map_py))
    print("max error exacts =", np.max(np.abs(sa_map_py_ex - sa_map)/sa_map_py_ex))
    print("max error python =", np.max(np.abs(sa_map_py - sa_map_py_ex)/sa_map_py_ex))

    assert np.allclose(sa_map, sa_map_py, rtol=1)
    assert np.allclose(sa_map, sa_map_py_ex, rtol=1)
    assert np.allclose(sa_map_py, sa_map_py_ex, rtol=1)

    # ...
    return
