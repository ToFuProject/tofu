"""
This module is the computational part of the geometrical module of ToFu
"""

# Built-in
import sys
import warnings

# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpintg
from inspect import signature as insp

# ToFu-specific
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from . import _def as _def
    from . import _GG as _GG


###############################################################################
#                            Ves functions
###############################################################################


# ==============================================================================
# = Ves sub-functions
# ==============================================================================
def _Struct_set_Poly(
    Poly, pos=None, extent=None, arrayorder="C", Type="Tor", Clock=False
):
    """ Compute geometrical attributes of a Struct object """

    # Make Poly closed, counter-clockwise, with '(cc,N)' layout and arrayorder
    Poly = _GG.Poly_Order(
        Poly, order="C", Clock=False, close=True, layout="(cc,N)", Test=True
    )
    assert Poly.shape[0] == 2, "Arg Poly must be a 2D polygon !"
    fPfmt = np.ascontiguousarray if arrayorder == "C" else np.asfortranarray

    # Get all remarkable points and moments
    NP = Poly.shape[1] - 1
    P1Max = Poly[:, np.argmax(Poly[0, :])]
    P1Min = Poly[:, np.argmin(Poly[0, :])]
    P2Max = Poly[:, np.argmax(Poly[1, :])]
    P2Min = Poly[:, np.argmin(Poly[1, :])]
    BaryP = np.sum(Poly[:, :-1], axis=1, keepdims=False) / (Poly.shape[1] - 1)
    BaryL = np.array(
        [(P1Max[0] + P1Min[0]) / 2.0, (P2Max[1] + P2Min[1]) / 2.0]
    )
    BaryS, Surf = _GG.poly_area_and_barycenter(Poly, NP)

    # Get lim-related indicators
    noccur = int(pos.size)
    Multi = noccur > 1

    # Get Tor-related quantities
    if Type.lower() == "lin":
        Vol, BaryV = None, None
    else:
        Vol, BaryV = _GG.Poly_VolAngTor(Poly)
        msg = "Pb. with volume computation for Ves object of type 'Tor' !"
        assert Vol > 0.0, msg

    # Compute the non-normalized vector of each side of the Poly
    Vect = np.diff(Poly, n=1, axis=1)
    Vect = fPfmt(Vect)

    # Compute the normalised vectors directed inwards
    Vin = np.array([Vect[1, :], -Vect[0, :]])
    Vin = -Vin  # Poly is Counter Clock-wise as defined above
    Vin = Vin / np.hypot(Vin[0, :], Vin[1, :])[np.newaxis, :]
    Vin = fPfmt(Vin)

    poly = _GG.Poly_Order(
        Poly,
        order=arrayorder,
        Clock=Clock,
        close=False,
        layout="(cc,N)",
        Test=True,
    )

    # Get bounding circle
    circC = BaryS
    r = np.sqrt(np.sum((poly - circC[:, np.newaxis]) ** 2, axis=0))
    circr = np.max(r)

    dout = {
        "Poly": poly,
        "pos": pos,
        "extent": extent,
        "noccur": noccur,
        "Multi": Multi,
        "nP": NP,
        "P1Max": P1Max,
        "P1Min": P1Min,
        "P2Max": P2Max,
        "P2Min": P2Min,
        "BaryP": BaryP,
        "BaryL": BaryL,
        "BaryS": BaryS,
        "BaryV": BaryV,
        "Surf": Surf,
        "VolAng": Vol,
        "Vect": Vect,
        "VIn": Vin,
        "circ-C": circC,
        "circ-r": circr,
        "Clock": Clock,
    }
    return dout


def _Ves_get_InsideConvexPoly(
    Poly,
    P2Min,
    P2Max,
    BaryS,
    RelOff=_def.TorRelOff,
    ZLim="Def",
    Spline=True,
    Splprms=_def.TorSplprms,
    NP=_def.TorInsideNP,
    Plot=False,
    Test=True,
):
    if Test:
        assert type(RelOff) is float, "Arg RelOff must be a float"
        assert (
            ZLim is None or ZLim == "Def" or type(ZLim) in [tuple, list]
        ), "Arg ZLim must be a tuple (ZlimMin, ZLimMax)"
        assert type(Spline) is bool, "Arg Spline must be a bool !"
    if ZLim is not None:
        if ZLim == "Def":
            ZLim = (
                P2Min[1] + 0.1 * (P2Max[1] - P2Min[1]),
                P2Max[1] - 0.05 * (P2Max[1] - P2Min[1]),
            )
        indZLim = (Poly[1, :] < ZLim[0]) | (Poly[1, :] > ZLim[1])
        if Poly.shape[1] - indZLim.sum() < 10:
            msg = "Poly seems to be Convex and simple enough !"
            msg += "\n  Poly.shape[1] - indZLim.sum() < 10"
            warnings.warn(msg)
            return Poly
        Poly = np.delete(Poly, indZLim.nonzero()[0], axis=1)
    if np.all(Poly[:, 0] == Poly[:, -1]):
        Poly = Poly[:, :-1]
    Np = Poly.shape[1]
    if Spline:
        BarySbis = np.tile(BaryS, (Np, 1)).T
        Ptemp = (1.0 - RelOff) * (Poly - BarySbis)
        # Poly = BarySbis + Ptemp
        Ang = np.arctan2(Ptemp[1, :], Ptemp[0, :])
        Ang, ind = np.unique(Ang, return_index=True)
        Ptemp = Ptemp[:, ind]
        # spline parameters
        ww = Splprms[0] * np.ones((Np + 1,))
        ss = Splprms[1] * (Np + 1)  # smoothness parameter
        kk = Splprms[2]  # spline order
        nest = int(
            (Np + 1) / 2.0
        )  # estimate of number of knots needed (-1 = maximal)
        # Find the knot points

        # TODO @DV : we can probably get rid of this
        # tckp,uu = scpinterp.splprep([np.append(Ptemp[0,:],Ptemp[0,0]),
        # np.append(Ptemp[1,:],Ptemp[1,0]),np.append(Ang,Ang[0]+2.*np.pi)],
        # w=ww, s=ss, k=kk, nest=nest)
        tckp, uu = scpinterp.splprep(
            [
                np.append(Ptemp[0, :], Ptemp[0, 0]),
                np.append(Ptemp[1, :], Ptemp[1, 0]),
            ],
            u=np.append(Ang, Ang[0] + 2.0 * np.pi),
            w=ww,
            s=ss,
            k=kk,
            nest=nest,
            full_output=0,
        )
        xnew, ynew = scpinterp.splev(np.linspace(-np.pi, np.pi, NP), tckp)
        Poly = np.array([xnew + BaryS[0], ynew + BaryS[1]])
        Poly = np.concatenate((Poly, Poly[:, 0:1]), axis=1)
    if Plot:
        import matplotlib.pyplot as plt

        f = plt.figure(facecolor="w", figsize=(8, 10))
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(Poly[0, :], Poly[1, :], "-k", Poly[0, :], Poly[1, :], "-r")
        ax.set_aspect(aspect="equal", adjustable="datalim"), ax.set_xlabel(
            r"R (m)"
        ), ax.set_ylabel(r"Z (m)")
        f.canvas.draw()
    return Poly


def _Ves_get_sampleEdge(
    VPoly, dL, DS=None, dLMode="abs", DIn=0.0, VIn=None, margin=1.0e-9
):
    types = [int, float, np.int32, np.int64, np.float32, np.float64]
    assert type(dL) in types and type(DIn) in types
    assert DS is None or (hasattr(DS, "__iter__") and len(DS) == 2)
    if DS is None:
        DS = [None, None]
    else:
        assert all(
            [
                ds is None
                or (
                    hasattr(ds, "__iter__")
                    and len(ds) == 2
                    and all([ss is None or type(ss) in types for ss in ds])
                )
                for ds in DS
            ]
        )
    assert type(dLMode) is str and dLMode.lower() in [
        "abs",
        "rel",
    ], "Arg dLMode must be in ['abs','rel'] !"
    # assert ind is None or (type(ind) is np.ndarray and ind.ndim==1
    # and ind.dtype in ['int32','int64'] and np.all(ind>=0)),
    #    "Arg ind must be None or 1D np.ndarray of positive int !"
    Pts, dLr, ind, N, Rref, VPolybis = _GG.discretize_vpoly(
        VPoly,
        float(dL),
        mode=dLMode.lower(),
        D1=DS[0],
        D2=DS[1],
        margin=margin,
        DIn=float(DIn),
        VIn=VIn,
    )
    return Pts, dLr, ind


def _Ves_get_sampleCross(
    VPoly,
    Min1,
    Max1,
    Min2,
    Max2,
    dS,
    DS=None,
    dSMode="abs",
    ind=None,
    margin=1.0e-9,
    mode="flat",
):
    assert mode in ["flat", "imshow"]
    types = [int, float, np.int32, np.int64, np.float32, np.float64]
    c0 = (
        hasattr(dS, "__iter__")
        and len(dS) == 2
        and all([type(ds) in types for ds in dS])
    )
    assert (
        c0 or type(dS) in types
    ), "Arg dS must be a float or a list 2 floats!"
    dS = (
        [float(dS), float(dS)]
        if type(dS) in types
        else [float(dS[0]), float(dS[1])]
    )
    assert DS is None or (hasattr(DS, "__iter__") and len(DS) == 2)
    if DS is None:
        DS = [None, None]
    else:
        assert all(
            [
                ds is None
                or (
                    hasattr(ds, "__iter__")
                    and len(ds) == 2
                    and all([ss is None or type(ss) in types for ss in ds])
                )
                for ds in DS
            ]
        )
    assert type(dSMode) is str and dSMode.lower() in [
        "abs",
        "rel",
    ], "Arg dSMode must be in ['abs','rel'] !"
    assert ind is None or (
        type(ind) is np.ndarray
        and ind.ndim == 1
        and ind.dtype in ["int32", "int64"]
        and np.all(ind >= 0)
    ), "Arg ind must be None or 1D np.ndarray of positive int !"

    MinMax1 = np.array([Min1, Max1])
    MinMax2 = np.array([Min2, Max2])
    if ind is None:
        if mode == "flat":
            Pts, dS, ind, d1r, d2r = _GG.discretize_segment2d(
                MinMax1,
                MinMax2,
                dS[0],
                dS[1],
                D1=DS[0],
                D2=DS[1],
                mode=dSMode,
                VPoly=VPoly,
                margin=margin,
            )
            out = (Pts, dS, ind, (d1r, d2r))
        else:
            x1, d1r, ind1, N1 = _GG._Ves_mesh_dlfromL_cython(
                MinMax1, dS[0], DS[0], Lim=True, dLMode=dSMode, margin=margin
            )
            x2, d2r, ind2, N2 = _GG._Ves_mesh_dlfromL_cython(
                MinMax2, dS[1], DS[1], Lim=True, dLMode=dSMode, margin=margin
            )
            xx1, xx2 = np.meshgrid(x1, x2)
            pts = np.squeeze([xx1, xx2])
            extent = (
                x1[0] - d1r / 2.0,
                x1[-1] + d1r / 2.0,
                x2[0] - d2r / 2.0,
                x2[-1] + d2r / 2.0,
            )
            out = (pts, x1, x2, extent)
    else:
        assert mode == "flat"
        c0 = type(ind) is np.ndarray and ind.ndim == 1
        c0 = c0 and ind.dtype in ["int32", "int64"] and np.all(ind >= 0)
        assert c0, "Arg ind must be a np.ndarray of int !"
        Pts, dS, d1r, d2r = _GG._Ves_meshCross_FromInd(
            MinMax1, MinMax2, dS[0], dS[1], ind, dSMode=dSMode, margin=margin
        )
        out = (Pts, dS, ind, (d1r, d2r))
    return out


def _Ves_get_sampleV(
    VPoly,
    Min1,
    Max1,
    Min2,
    Max2,
    dV,
    DV=None,
    dVMode="abs",
    ind=None,
    VType="Tor",
    VLim=None,
    Out="(X,Y,Z)",
    margin=1.0e-9,
):
    types = [int, float, np.int32, np.int64, np.float32, np.float64]
    assert type(dV) in types or (
        hasattr(dV, "__iter__")
        and len(dV) == 3
        and all([type(ds) in types for ds in dV])
    ), "Arg dV must be a float or a list 3 floats !"
    dV = (
        [float(dV), float(dV), float(dV)]
        if type(dV) in types
        else [float(dV[0]), float(dV[1]), float(dV[2])]
    )
    assert DV is None or (hasattr(DV, "__iter__") and len(DV) == 3)
    if DV is None:
        DV = [None, None, None]
    else:
        assert all(
            [
                ds is None
                or (
                    hasattr(ds, "__iter__")
                    and len(ds) == 2
                    and all([ss is None or type(ss) in types for ss in ds])
                )
                for ds in DV
            ]
        ), "Arg DV must be a list of 3 lists of 2 floats !"
    assert type(dVMode) is str and dVMode.lower() in [
        "abs",
        "rel",
    ], "Arg dVMode must be in ['abs','rel'] !"
    assert ind is None or (
        type(ind) is np.ndarray
        and ind.ndim == 1
        and ind.dtype in ["int32", "int64"]
        and np.all(ind >= 0)
    ), "Arg ind must be None or 1D np.ndarray of positive int !"

    MinMax1 = np.array([Min1, Max1])
    MinMax2 = np.array([Min2, Max2])
    VLim = None if VType.lower() == "tor" else np.array(VLim).ravel()
    dVr = [None, None, None]
    if ind is None:
        if VType.lower() == "tor":
            Pts, dV, ind, dVr[0], dVr[1], dVr[
                2
            ] = _GG._Ves_Vmesh_Tor_SubFromD_cython(
                dV[0],
                dV[1],
                dV[2],
                MinMax1,
                MinMax2,
                DR=DV[0],
                DZ=DV[1],
                DPhi=DV[2],
                VPoly=VPoly,
                Out=Out,
                margin=margin,
            )
        else:
            Pts, dV, ind, dVr[0], dVr[1], dVr[
                2
            ] = _GG._Ves_Vmesh_Lin_SubFromD_cython(
                dV[0],
                dV[1],
                dV[2],
                VLim,
                MinMax1,
                MinMax2,
                DX=DV[0],
                DY=DV[1],
                DZ=DV[2],
                VPoly=VPoly,
                margin=margin,
            )
    else:
        if VType.lower() == "tor":
            Pts, dV, dVr[0], dVr[1], dVr[
                2
            ] = _GG._Ves_Vmesh_Tor_SubFromInd_cython(
                dV[0],
                dV[1],
                dV[2],
                MinMax1,
                MinMax2,
                ind,
                Out=Out,
                margin=margin,
            )
        else:
            Pts, dV, dVr[0], dVr[1], dVr[
                2
            ] = _GG._Ves_Vmesh_Lin_SubFromInd_cython(
                dV[0], dV[1], dV[2], VLim, MinMax1, MinMax2, ind, margin=margin
            )
    return Pts, dV, ind, dVr


def _Ves_get_sampleS(
    VPoly,
    dS,
    DS=None,
    dSMode="abs",
    ind=None,
    DIn=0.0,
    VIn=None,
    VType="Tor",
    VLim=None,
    nVLim=None,
    Out="(X,Y,Z)",
    margin=1.0e-9,
    Multi=False,
    Ind=None,
):
    types = [int, float, np.int32, np.int64, np.float32, np.float64]
    assert type(dS) in types or (
        hasattr(dS, "__iter__")
        and len(dS) == 2
        and all([type(ds) in types for ds in dS])
    ), "Arg dS must be a float or a list of 2 floats !"
    dS = (
        [float(dS), float(dS), float(dS)]
        if type(dS) in types
        else [float(dS[0]), float(dS[1]), float(dS[2])]
    )
    assert DS is None or (hasattr(DS, "__iter__") and len(DS) == 3)
    msg = "type(nVLim)={0} and nVLim={1}".format(str(type(nVLim)), nVLim)
    assert type(nVLim) is int and nVLim >= 0, msg
    if DS is None:
        DS = [None, None, None]
    else:
        assert all(
            [
                ds is None
                or (
                    hasattr(ds, "__iter__")
                    and len(ds) == 2
                    and all([ss is None or type(ss) in types for ss in ds])
                )
                for ds in DS
            ]
        ), "Arg DS must be a list of 3 lists of 2 floats !"
    assert type(dSMode) is str and dSMode.lower() in [
        "abs",
        "rel",
    ], "Arg dSMode must be in ['abs','rel'] !"
    assert type(Multi) is bool, "Arg Multi must be a bool !"

    VLim = None if (VLim is None or nVLim == 0) else np.array(VLim)

    # Check if Multi
    if nVLim > 1:
        assert VLim is not None, "For multiple Struct, Lim cannot be None !"
        assert all([hasattr(ll, "__iter__") and len(ll) == 2 for ll in VLim])
        if Ind is None:
            Ind = np.arange(0, nVLim)
        else:
            Ind = [Ind] if not hasattr(Ind, "__iter__") else Ind
            Ind = np.asarray(Ind).astype(int)
        if ind is not None:
            assert hasattr(ind, "__iter__") and len(ind) == len(
                Ind
            ), "For multiple Struct, ind must be a list of len() = len(Ind) !"
            assert all(
                [
                    type(ind[ii]) is np.ndarray
                    and ind[ii].ndim == 1
                    and ind[ii].dtype in ["int32", "int64"]
                    and np.all(ind[ii] >= 0)
                    for ii in range(0, len(ind))
                ]
            ), "For multiple Struct, ind must be a list of index arrays !"

    else:
        VLim = [None] if VLim is None else [VLim.ravel()]
        assert ind is None or (
            type(ind) is np.ndarray
            and ind.ndim == 1
            and ind.dtype in ["int32", "int64"]
            and np.all(ind >= 0)
        ), "Arg ind must be None or 1D np.ndarray of positive int !"
        Ind = [0]

    if ind is None:
        Pts, dS, ind, dSr = (
            [0 for ii in Ind],
            [dS for ii in Ind],
            [0 for ii in Ind],
            [[0, 0] for ii in Ind],
        )
        if VType.lower() == "tor":
            for ii in range(0, len(Ind)):
                if VLim[Ind[ii]] is None:
                    Pts[ii], dS[ii], ind[ii], NL, dSr[ii][0], Rref, dSr[ii][
                        1
                    ], nRPhi0, VPbis = _GG._Ves_Smesh_Tor_SubFromD_cython(
                        dS[ii][0],
                        dS[ii][1],
                        VPoly,
                        DR=DS[0],
                        DZ=DS[1],
                        DPhi=DS[2],
                        DIn=DIn,
                        VIn=VIn,
                        PhiMinMax=None,
                        Out=Out,
                        margin=margin,
                    )
                else:
                    Pts[ii], dS[ii], ind[ii], NL, dSr[ii][
                        0
                    ], Rref, dR0r, dZ0r, dSr[ii][
                        1
                    ], VPbis = _GG._Ves_Smesh_TorStruct_SubFromD_cython(
                        VLim[Ind[ii]],
                        dS[ii][0],
                        dS[ii][1],
                        VPoly,
                        DR=DS[0],
                        DZ=DS[1],
                        DPhi=DS[2],
                        DIn=DIn,
                        VIn=VIn,
                        Out=Out,
                        margin=margin,
                    )
                    dSr[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0, len(Ind)):
                Pts[ii], dS[ii], ind[ii], NL, dSr[ii][0], Rref, dSr[ii][
                    1
                ], dY0r, dZ0r, VPbis = _GG._Ves_Smesh_Lin_SubFromD_cython(
                    VLim[Ind[ii]],
                    dS[ii][0],
                    dS[ii][1],
                    VPoly,
                    DX=DS[0],
                    DY=DS[1],
                    DZ=DS[2],
                    DIn=DIn,
                    VIn=VIn,
                    margin=margin,
                )
                dSr[ii] += [dY0r, dZ0r]
    else:
        ind = ind if Multi else [ind]
        Pts, dS, dSr = (
            [np.ones((3, 0)) for ii in Ind],
            [dS for ii in Ind],
            [[0, 0] for ii in Ind],
        )
        if VType.lower() == "tor":
            for ii in range(0, len(Ind)):
                if ind[Ind[ii]].size > 0:
                    if VLim[Ind[ii]] is None:
                        out_loc = _GG._Ves_Smesh_Tor_SubFromInd_cython(
                            dS[ii][0],
                            dS[ii][1],
                            VPoly,
                            ind[Ind[ii]],
                            DIn=DIn,
                            VIn=VIn,
                            PhiMinMax=None,
                            Out=Out,
                            margin=margin,
                        )
                        Pts[ii], dS[ii], NL, dSr[ii][0], Rref = out_loc[:5]
                        dSr[ii][1], nRPhi0, VPbis = out_loc[5:]
                    else:
                        out_loc = _GG._Ves_Smesh_TorStruct_SubFromInd_cython(
                            VLim[Ind[ii]],
                            dS[ii][0],
                            dS[ii][1],
                            VPoly,
                            ind[Ind[ii]],
                            DIn=DIn,
                            VIn=VIn,
                            Out=Out,
                            margin=margin,
                        )
                        Pts[ii], dS[ii], NL, dSr[ii][0], Rref = out_loc[:5]
                        dR0r, dZ0r, dSr[ii][1], VPbis = out_loc[5:]
                        dSr[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0, len(Ind)):
                if ind[Ind[ii]].size > 0:
                    out_loc = _GG._Ves_Smesh_Lin_SubFromInd_cython(
                        VLim[Ind[ii]],
                        dS[ii][0],
                        dS[ii][1],
                        VPoly,
                        ind[Ind[ii]],
                        DIn=DIn,
                        VIn=VIn,
                        margin=margin,
                    )
                    Pts[ii], dS[ii], NL, dSr[ii][0], Rref = out_loc[:5]
                    dSr[ii][1], dY0r, dZ0r, VPbis = out_loc[5:]
                    dSr[ii] += [dY0r, dZ0r]

    if len(VLim) == 1:
        Pts, dS, ind, dSr = Pts[0], dS[0], ind[0], dSr[0]
    return Pts, dS, ind, dSr


# ==============================================================================
# =  phi / theta projections for magfieldlines
# ==============================================================================
def _Struct_get_phithetaproj(ax=None, poly_closed=None, lim=None, noccur=0):

    # phi = toroidal angle
    if noccur == 0:
        Dphi = np.array([[-np.pi, np.pi]])
        nphi = np.r_[1]
    else:
        assert lim.ndim == 2, str(lim)
        nphi = np.ones((noccur,), dtype=int)
        ind = (lim[:, 0] > lim[:, 1]).nonzero()[0]
        Dphi = np.concatenate((lim, np.full((noccur, 2), np.nan)), axis=1)
        if ind.size > 0:
            for ii in ind:
                Dphi[ii, :] = [lim[ii, 0], np.pi, -np.pi, lim[ii, 1]]
                nphi[ii] = 2

    # theta = poloidal angle
    Dtheta = np.arctan2(poly_closed[1, :] - ax[1], poly_closed[0, :] - ax[0])
    Dtheta = np.r_[np.min(Dtheta), np.max(Dtheta)]
    if Dtheta[0] > Dtheta[1]:
        ntheta = 2
        Dtheta = [Dtheta[0], np.pi, -np.pi, Dtheta[1]]
    else:
        ntheta = 1

    return nphi, Dphi, ntheta, Dtheta


def _get_phithetaproj_dist(
    poly_closed,
    ax,
    Dtheta,
    nDtheta,
    Dphi,
    nDphi,
    theta,
    phi,
    ntheta,
    nphi,
    noccur,
):

    if nDtheta == 1:
        ind = (theta >= Dtheta[0]) & (theta <= Dtheta[1])
    else:
        ind = (theta >= Dtheta[0]) | (theta <= Dtheta[1])

    disttheta = np.full((theta.size,), np.nan)

    # phi within Dphi
    if noccur > 0:
        indphi = np.zeros((nphi,), dtype=bool)
        for ii in range(0, noccur):
            for jj in range(0, nDphi[ii]):
                indphi |= (phi >= Dphi[ii, jj]) & (phi <= Dphi[ii, jj + 1])
        if not np.any(indphi):
            return disttheta, indphi
    else:
        indphi = np.ones((nphi,), dtype=bool)

    # No theta within Dtheta
    if not np.any(ind):
        return disttheta, indphi

    # Check for non-parallel AB / u pairs
    u = np.array([np.cos(theta), np.sin(theta)])
    AB = np.diff(poly_closed, axis=1)
    detABu = AB[0, :, None] * u[1, None, :] - AB[1, :, None] * u[0, None, :]
    inddet = ind[None, :] & (np.abs(detABu) > 1.0e-9)
    if not np.any(inddet):
        return disttheta, indphi

    nseg = poly_closed.shape[1] - 1
    k = np.full((nseg, ntheta), np.nan)

    OA = poly_closed[:, :-1] - ax[:, None]
    detOAu = (OA[0, :, None] * u[1, None, :] - OA[1, :, None] * u[0, None, :])[
        inddet
    ]
    ss = -detOAu / detABu[inddet]
    inds = (ss >= 0.0) & (ss < 1.0)
    inddet[inddet] = inds

    if not np.any(inds):
        return disttheta, indphi

    scaOAu = (OA[0, :, None] * u[0, None, :] + OA[1, :, None] * u[1, None, :])[
        inddet
    ]
    scaABu = (AB[0, :, None] * u[0, None, :] + AB[1, :, None] * u[1, None, :])[
        inddet
    ]
    k[inddet] = scaOAu + ss[inds] * scaABu
    indk = k[inddet] > 0.0
    inddet[inddet] = indk

    if not np.any(indk):
        return disttheta, indphi

    k[~inddet] = np.nan
    indok = np.any(inddet, axis=0)
    disttheta[indok] = np.nanmin(k[:, indok], axis=0)

    return disttheta, indphi


# ==============================================================================
# =  LOS functions
# ==============================================================================
def LOS_PRMin(Ds, us, kOut=None, Eps=1.0e-12, squeeze=True, Test=True):
    """  Compute the point on the LOS where the major radius is minimum """
    if Test:
        assert Ds.ndim in [1, 2, 3] and 3 in Ds.shape and Ds.shape == us.shape
    if kOut is not None:
        kOut = np.atleast_1d(kOut)
        assert kOut.size == Ds.size / 3

    if Ds.ndim == 1:
        Ds, us = Ds[:, None, None], us[:, None, None]
    elif Ds.ndim == 2:
        Ds, us = Ds[:, :, None], us[:, :, None]
    if kOut is not None:
        if kOut.ndim == 1:
            kOut = kOut[:, None]
    _, nlos, nref = Ds.shape

    kRMin = np.full((nlos, nref), np.nan)
    uparN = np.sqrt(us[0, :, :] ** 2 + us[1, :, :] ** 2)

    # Case with u vertical
    ind = uparN > Eps
    kRMin[~ind] = 0.0

    # Else
    kRMin[ind] = (
        -(us[0, ind] * Ds[0, ind] + us[1, ind] * Ds[1, ind]) / uparN[ind] ** 2
    )

    # Check
    kRMin[kRMin <= 0.0] = 0.0
    if kOut is not None:
        kRMin[kRMin > kOut] = kOut[kRMin > kOut]

    # squeeze
    if squeeze:
        if nref == 1 and nlos == 11:
            kRMin = kRMin[0, 0]
        elif nref == 1:
            kRMin = kRMin[:, 0]
        elif nlos == 1:
            kRMin = kRMin[0, :]
    return kRMin


def LOS_CrossProj(
    VType,
    Ds,
    us,
    kOuts,
    proj="All",
    multi=False,
    num_threads=16,
    return_pts=False,
    Test=True,
):
    """ Compute the parameters to plot the poloidal projection of the LOS  """
    assert type(VType) is str and VType.lower() in ["tor", "lin"]
    dproj = {
        "cross": ("R", "Z"),
        "hor": ("x,y"),
        "all": ("R", "Z", "x", "y"),
        "3d": ("x", "y", "z"),
    }
    assert type(proj) in [str, tuple]
    if type(proj) is tuple:
        assert all([type(pp) is str for pp in proj])
        lcoords = proj
    else:
        proj = proj.lower()
        assert proj in dproj.keys()
        lcoords = dproj[proj]
    if return_pts:
        assert proj in ["cross", "hor", "3d"]

    lc = [Ds.ndim == 3, Ds.shape == us.shape]
    if not all(lc):
        msg = "Ds and us must have the same shape and dim in [2,3]:\n"
        msg += "    - provided Ds.shape: %s\n" % str(Ds.shape)
        msg += "    - provided us.shape: %s" % str(us.shape)
        raise Exception(msg)
    lc = [kOuts.size == Ds.size / 3, kOuts.shape == Ds.shape[1:]]
    if not all(lc):
        msg = "kOuts must have the same shape and ndim = Ds.ndim-1:\n"
        msg += "    - Ds.shape    : %s\n" % str(Ds.shape)
        msg += "    - kOutss.shape: %s" % str(kOuts.shape)
        raise Exception(msg)

    # Prepare inputs
    _, nlos, nseg = Ds.shape

    # Detailed sampling for 'tor' and ('cross' or 'all')
    R, Z = None, None
    if "R" in lcoords or "Z" in lcoords:
        angcross = np.arccos(
            np.sqrt(us[0, ...] ** 2 + us[1, ...] ** 2)
            / np.sqrt(np.sum(us ** 2, axis=0))
        )
        resnk = np.ceil(25.0 * (1 - (angcross / (np.pi / 4) - 1) ** 2) + 5)
        resnk = 1.0 / resnk.ravel()

        # Use optimized get sample
        DL = np.vstack((np.zeros((nlos * nseg,), dtype=float), kOuts.ravel()))

        k, reseff, lind = _GG.LOS_get_sample(
            nlos * nseg,
            resnk,
            DL,
            dmethod="rel",
            method="simps",
            num_threads=num_threads,
            Test=Test,
        )

        assert lind.size == nseg * nlos - 1
        ind = lind[nseg - 1 :: nseg]  # noqa
        nbrep = np.r_[lind[0], np.diff(lind), k.size - lind[-1]]
        pts = np.repeat(Ds.reshape((3, nlos * nseg)), nbrep, axis=1) + k[
            None, :
        ] * np.repeat(us.reshape((3, nlos * nseg)), nbrep, axis=1)

        if return_pts:
            pts = np.array([np.hypot(pts[0, :], pts[1, :]), pts[2, :]])
            if multi:
                pts = np.split(pts, ind, axis=1)
            else:
                pts = np.insert(pts, ind, np.nan, axis=1)
        else:
            if multi:
                if "R" in lcoords:
                    R = np.split(np.hypot(pts[0, :], pts[1, :]), ind)
                if "Z" in lcoords:
                    Z = np.split(pts[2, :], ind)
            else:
                if "R" in lcoords:
                    R = np.insert(np.hypot(pts[0, :], pts[1, :]), ind, np.nan)
                if "Z" in lcoords:
                    Z = np.insert(pts[2, :], ind, np.nan)

    # Normal sampling => pts
    # unnecessary only if 'tor' and 'cross'
    x, y, z = None, None, None
    if "x" in lcoords or "y" in lcoords or "z" in lcoords:
        pts = np.concatenate(
            (Ds, Ds[:, :, -1:] + kOuts[None, :, -1:] * us[:, :, -1:]), axis=-1
        )

        if multi:
            ind = np.arange(1, nlos) * (nseg + 1)
            pts = pts.reshape((3, nlos * (nseg + 1)))
        else:
            nancoords = np.full((3, nlos, 1), np.nan)
            pts = np.concatenate((pts, nancoords), axis=-1)
            pts = pts.reshape((3, nlos * (nseg + 2)))

        if return_pts:
            assert proj in ["hor", "3d"]
            if multi:
                if proj == "hor":
                    pts = np.split(pts[:2, :], ind, axis=1)
                else:
                    pts = np.split(pts, ind, axis=1)
            elif proj == "hor":
                pts = pts[:2, :]

        else:
            if multi:
                if "x" in lcoords:
                    x = np.split(pts[0, :], ind)
                if "y" in lcoords:
                    y = np.split(pts[1, :], ind)
                if "z" in lcoords:
                    z = np.split(pts[2, :], ind)
            else:
                if "x" in lcoords:
                    x = pts[0, :]
                if "y" in lcoords:
                    y = pts[1, :]
                if "z" in lcoords:
                    z = pts[2, :]

    if return_pts:
        return pts
    else:
        return R, Z, x, y, z


# ==============================================================================
# =  Meshing & signal
# ==============================================================================
def LOS_get_sample(D, u, dL, DL=None, dLMode="abs", method="sum", Test=True):
    """ Return the sampled line, with the specified method

    'linspace': return the N+1 edges, including the first and last point
    'sum' : return the N middle of the segments
    'simps': return the N+1 egdes, where N has to be even
             (scipy.simpson requires an even number of intervals)
    'romb' : return the N+1 edges, where N+1 = 2**k+1
             (fed to scipy.romb for integration)
    """
    if Test:
        assert all(
            [type(dd) is np.ndarray and dd.shape == (3,) for dd in [D, u]]
        )
        assert not hasattr(dL, "__iter__")
        assert DL is None or all(
            [
                hasattr(DL, "__iter__"),
                len(DL) == 2,
                all([not hasattr(dd, "__iter__") for dd in DL]),
            ]
        )
        assert dLMode in ["abs", "rel"]
        assert type(method) is str and method in [
            "linspace",
            "sum",
            "simps",
            "romb",
        ]
    # Compute the min number of intervals to satisfy the specified resolution
    N = (
        int(np.ceil((DL[1] - DL[0]) / dL))
        if dLMode == "abs"
        else int(np.ceil(1.0 / dL))
    )
    # Modify N according to the desired method
    if method == "simps":
        N = N if N % 2 == 0 else N + 1
    elif method == "romb":
        N = 2 ** int(np.ceil(np.log(N) / np.log(2.0)))

    # Derive k and dLr
    if method == "sum":
        dLr = (DL[1] - DL[0]) / N
        k = DL[0] + (0.5 + np.arange(0, N)) * dLr
    else:
        k, dLr = np.linspace(
            DL[0], DL[1], N + 1, endpoint=True, retstep=True, dtype=float
        )

    Pts = D[:, np.newaxis] + k[np.newaxis, :] * u[:, np.newaxis]
    return Pts, k, dLr


def LOS_calc_signal(
    ff, D, u, dL, DL=None, dLMode="abs", method="romb", Test=True
):
    assert hasattr(ff, "__call__"), (
        "Arg ff must be a callable (function) taking at least 1 positional ",
        "Pts (a (3,N) np.ndarray of cartesian (X,Y,Z) coordinates) !",
    )
    assert not method == "linspace"
    Pts, k, dLr = LOS_get_sample(
        D, u, dL, DL=DL, dLMode=dLMode, method=method, Test=Test
    )
    out = insp(ff)
    N = np.sum(
        [
            (
                pp.kind == pp.POSITIONAL_OR_KEYWORD
                and pp.default is pp.empty
            )
            for pp in out.parameters.values()
        ]
    )

    if N == 1:
        Vals = ff(Pts)
    elif N == 2:
        Vals = ff(Pts, np.tile(-u, (Pts.shape[1], 1)).T)
    else:
        raise ValueError(
            "The function (ff) assessing the emissivity locally "
            + "must take a single positional argument: Pts a (3,N)"
            + " np.ndarray of (X,Y,Z) cartesian coordinates !"
        )

    Vals[np.isnan(Vals)] = 0.0
    if method == "sum":
        Int = np.sum(Vals) * dLr
    elif method == "simps":
        Int = scpintg.simps(Vals, x=None, dx=dLr)
    elif method == "romb":
        Int = scpintg.romb(Vals, dx=dLr, show=False)
    return Int


# ==============================================================================
# =  Solid Angle particle
# ==============================================================================


def calc_solidangle_particle(
    traj, pts, r=1.0, config=None, approx=True, aniso=False, block=True
):
    """ Compute the solid angle subtended by a particle along a trajectory

    The particle has radius r, and trajectory (array of points) traj
    It is observed from pts (array of points)

    traj and pts are (3,N) and (3,M) arrays of cartesian coordinates

    approx = True => use approximation
    aniso = True => return also unit vector of emission
    block = True consider LOS collisions (with Ves, Struct...)

    if block:
        config = config used for LOS collisions

    Return:
    -------
    sang: np.ndarray
        (N,M) Array of floats, solid angles

    """
    ################
    # Prepare inputs
    traj = np.ascontiguousarray(traj, dtype=float)
    pts = np.ascontiguousarray(pts, dtype=float)
    r = np.r_[r].astype(float).ravel()

    # Check booleans
    assert type(approx) is bool
    assert type(aniso) is bool
    assert type(block) is bool

    # Check config
    assert config is None or config.__class__.__name__ == "Config"
    assert block == (config is not None)

    # Check pts, traj and r are array of good shape
    assert traj.ndim in [1, 2]
    assert pts.ndim in [1, 2]
    assert 3 in traj.shape and 3 in pts.shape
    if traj.ndim == 1:
        traj = traj.reshape((3, 1))
    if traj.shape[0] != 3:
        traj = traj.T
    if pts.ndim == 1:
        pts = pts.reshape((3, 1))
    if pts.shape[0] != 3:
        pts = pts.T

    # get npart
    ntraj = traj.shape[1]
    nr = r.size

    npart = max(nr, ntraj)
    assert nr in [1, npart]
    assert ntraj in [1, npart]
    if nr < npart:
        r = np.full((npart,), r[0])
    if ntraj < npart:
        traj = np.repeat(traj, npart, axis=1)

    ################
    # Main computation

    # traj2pts vector, with length (3d array (3,N,M))
    vect = pts[:, None, :] - traj[:, :, None]
    len_v = np.sqrt(np.sum(vect ** 2, axis=0))

    # If aniso or block, normalize
    if aniso or block:
        vect = vect / len_v[None, :, :]

    # Solid angle
    if approx:
        sang = np.pi * r[None, :] ** 2 / len_v ** 2
    else:
        sang = 2.0 * np.pi * (1 - np.sqrt(1.0 - r ** 2[None, :] / len_v ** 2))

    # block
    if block:
        kwdargs = config._get_kwdargs_LOS_isVis()
        # TODO : modify this function along issue #102
        indnan = _GG.LOS_areVis_PtsFromPts_VesStruct(
            traj, pts, k=len_v, vis=False, **kwdargs
        )
        sang[indnan] = 0.0
        vect[indnan, :] = np.nan

    ################
    # Return

    if aniso:
        return sang, vect
    else:
        return sang


def calc_solidangle_particle_integ(
    traj, r=1.0, config=None, approx=True, block=True, res=0.01
):

    # step0: if block : generate kwdargs from config

    # step 1: sample cross-section

    # step 2: loop on R of  pts of cross-section (parallelize ?)
    # => fix nb. of phi for the rest of the loop

    # loop of Z

    # step 3: loop phi
    # Check visibility (if block = True) for each phi (LOS collision)
    # If visible => compute solid angle
    # integrate (sum * res) on each phi the solid angle

    # Return sang as (N,nR,nZ) array
    return
