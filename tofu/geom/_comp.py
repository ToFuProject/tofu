"""
This module is the computational part of the geometrical module of ToFu
"""

# Built-in
import os
import warnings
from xml.dom import minidom

# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpintg
import matplotlib.colors as mplcol
from inspect import signature as insp


# ToFu-specific
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from . import _def as _def
    from . import _GG as _GG


_LTYPES = [int, float, np.int_, np.float64]
_RES = 0.1


###############################################################################
#                            Default parameters
###############################################################################


_SAMPLE_RES = {
    'edge': 0.02,
    'cross': 0.1,
    'surface': 0.1,
    'volume': 0.1,
}
_SAMPLE_RESMODE = {
    'edge': 'abs',
    'cross': 'abs',
    'surface': 'abs',
    'volume': 'abs',
}


def _check_float(var=None, varname=None, vardef=None):
    if var is None:
        var = vardef
    if not type(var) in _LTYPES:
        msg = (
            "Arg {} must be a float!\n".format(varname)
            + "Provided: {}".format(type(var))
        )
        raise Exception(msg)
    return var

###############################################################################
#                            Ves functions
###############################################################################


# ==============================================================================
# Interfacing functions
# ==============================================================================


def _get_pts_from_path_svg(
    path_str=None,
    res=None,
    k0=None,
):

    # Check inputs
    res = _check_float(var=res, varname='res', vardef=_RES)

    # try loading
    try:
        from svg.path import parse_path
    except Exception as err:
        msg = (
            str(err)
            + "\n\nYou do not seem to have svg.path installed\n"
            + "It is an optional dependency only used for this method\n"
            + "To use from_svg(), please install svg.path using:\n"
            + "\tpip install svg.path"
        )
        raise Exception(msg)

    lpath = parse_path(path_str)

    lpath._calc_lengths()
    fract = lpath._fractions

    pos = []
    for ii, pat in enumerate(lpath):
        if pat.__class__.__name__ == 'Line':
            pos.append(np.r_[fract[ii]])
        elif pat.__class__.__name__ == 'Move':
            pos.append(np.r_[fract[ii]])
        elif pat.__class__.__name__ == 'Close':
            pos.append(np.r_[fract[ii]])
        else:
            npts = int(np.ceil(pat.length() / res))
            pos.append(
                np.linspace(fract[ii], fract[ii+1], npts, endpoint=False)
            )

    pos = np.unique(np.concatenate(pos))
    ind1 = np.abs(pos-1.) < 1e-14
    if np.sum(ind1) == 1:
        pos[ind1] = 1.
    elif np.sum(ind1) > 1:
        msg = "Several 1!"
        raise Exception(msg)
    pts = np.array([lpath.point(po) for po in pos])
    pts = np.array([pts.real, pts.imag])

    # Check for reference line
    isref = False
    if 'z' not in path_str.lower():
        if pts.shape[1] == 2:
            isref = True
        elif np.allclose(pts[:, 0], pts[:, -1]):
            pass
        else:
            pts = np.concatenate((pts, pts[:, 0:1]), axis=1)
            msg = (
                f"Non-conform path '{k0}' identified!\n"
                "All path must be either:\n"
                "\t- closed\n"
                "\t- or a unique straight line with 2 points\n"
                "Provided:\n"
                f"path_str: {path_str}\n"
                f"  => closed automatically"
            )
            warnings.warn(msg)

    return pts, isref


def get_paths_from_svg(
    pfe=None,
    res=None,
    r0=None,
    z0=None,
    point_ref1=None,
    point_ref2=None,
    length_ref=None,
    scale=None,
    verb=None,
):

    # check input
    c0 = isinstance(pfe, str) and os.path.isfile(pfe) and pfe.endswith('.svg')
    if not c0:
        msg = (
            "Arg pfe should be a path to a valid .svg file!\n"
            + "Provided:\n\t{}".format(pfe)
        )
        raise Exception(msg)
    pfe = os.path.abspath(pfe)

    # r0, z0, scale
    z0 = _check_float(var=z0, varname='z0', vardef=0.)
    r0 = _check_float(var=r0, varname='r0', vardef=0.)
    scale = _check_float(var=scale, varname='scale', vardef=1.)

    # verb
    if verb is None:
        verb = True
    if not isinstance(verb, bool):
        msg = (
            "Arg verb must be a bool!\n"
            + "Provided:\n\t{}".format(verb)
        )
        raise Exception(msg)

    # Predefine useful var
    doc = minidom.parse(pfe)

    # Try extract raw data
    try:
        dpath = {
            path.getAttribute('id').replace('\n', '').replace('""', ''): {
                'poly': path.getAttribute('d'),
                'color': path.getAttribute('style')
            }
            for path in doc.getElementsByTagName('path')
        }
    except Exception as err:
        msg = (
            "Could not extract path coordinates from {}".format(pfe)
        )
        raise Exception(msg)

    # Derive usable data
    kstr = 'fill:'
    dpath = {k0: v0 for k0, v0 in dpath.items() if kstr in v0['color']}
    lk = list(dpath.keys())
    ref = None
    for ii, k0 in enumerate(lk):

        v0 = dpath[k0]
        poly, isref = _get_pts_from_path_svg(v0['poly'], res=res, k0=k0)
        if isref is True:
            ref = poly
            del dpath[k0]
            continue
        dpath[k0]['poly'] = poly

        # class and color
        color = v0['color'][v0['color'].index(kstr) + len(kstr):].split(';')[0]
        if color == 'none':
            dpath[k0]['cls'] = 'Ves'
            color = None
        else:
            if mplcol.to_rgb(color) == (1., 0., 0.):
                dpath[k0]['cls'] = 'CoilPF'
            else:
                dpath[k0]['cls'] = 'PFC'
        dpath[k0]['color'] = color

    # Check for negative r
    lkneg = [k0 for k0, v0 in dpath.items() if np.any(v0['poly'][0, :] <= 0.)]
    if len(lkneg) > 0.:
        lstr = ['\t- {}'.format(k0) for k0 in lkneg]
        msg = (
            "With the chosen r0 ({}) some structure have negative r values\n"
            + "This is impossible in a toroidal coordinate system\n"
            + "  => the following structures are removed:\n"
            + "\n".join(lstr)
        )
        if len(lkneg) == len(dpath):
            raise Exception(msg)
        else:
            warnings.warn(msg)
        dpath = {k0: dpath[k0] for k0 in dpath.keys() if k0 not in lkneg}

    # Set origin and rescale
    if ref is not None:
        lc = [
            point_ref1 is not None and point_ref2 is not None,
            point_ref1 is not None and length_ref is not None,
        ]
        if not any(lc):
            msg = (
                "Arg reference line for scaling has been detected!\n"
                + "But it cannot be used without providing:\n"
                + "\t- point_ref1 + point_ref2: iterables of len() = 2\n"
                + "\t- point_Ref1 + length_ref: iterable len() = 2 + scalar\n"
            )
            warnings.warn(msg)
        else:
            unit = np.diff(ref, axis=1)
            unit = unit / np.linalg.norm(unit)
            unit = np.array([[unit[0, 0]], [-unit[1, 0]]])
            if not lc[0]:
                point_ref2 = np.array(point_ref1)[:, None] + length_ref*unit

            # if horizontal (resp. vertical line) => coef = inf
            # => assume equal scale for r and z instead to avoid inf
            eps = 1.e-8
            if np.abs(unit[0, 0]) > eps:
                r_coef = (
                    (point_ref2[0]-point_ref1[0]) / (ref[0, 1] - ref[0, 0])
                )
            if np.abs(unit[1, 0]) > eps:
                z_coef = (
                    (point_ref2[1]-point_ref1[1]) / (ref[1, 1] - ref[1, 0])
                )
            if np.abs(unit[0, 0]) < eps:
                # vertical line => assume rscale  = zscale
                r_coef = -z_coef
            if np.abs(unit[1, 0]) < eps:
                # horizontal line => assume rscale  = zscale
                z_coef = -r_coef
            r_offset = point_ref1[0] - r_coef*ref[0, 0]
            z_offset = point_ref1[1] - z_coef*ref[1, 0]

            for k0 in dpath.keys():
                dpath[k0]['poly'] = np.array([
                    r_coef*dpath[k0]['poly'][0, :] + r_offset,
                    z_coef*dpath[k0]['poly'][1, :] + z_offset,
                ])
    else:
        for k0 in dpath.keys():
            dpath[k0]['poly'] = np.array([
                scale*(dpath[k0]['poly'][0, :] - r0),
                scale*(-dpath[k0]['poly'][1, :] - z0),
            ])

    # verb
    if verb is True:
        lVes = sorted([k0 for k0, v0 in dpath.items() if v0['cls'] == 'Ves'])
        lPFC = sorted([k0 for k0, v0 in dpath.items() if v0['cls'] == 'PFC'])
        lCoilPF = sorted([
            k0 for k0, v0 in dpath.items() if v0['cls'] == 'CoilPF'
        ])
        lobj = [
            '\t- {}: {} ({} pts, {})'.format(
                dpath[k0]['cls'], k0,
                dpath[k0]['poly'].shape[1], dpath[k0]['color'],
            )
            for k0 in lVes + lPFC + lCoilPF
        ]
        msg = (
            "The following structures were loaded:\n".format(pfe)
            + "\n".join(lobj)
            + "\nfrom {}".format(pfe)
        )
        print(msg)

    return dpath


# ==============================================================================
# = Ves sub-functions
# ==============================================================================


def _Struct_set_Poly(
    Poly, pos=None, extent=None, arrayorder="C", Type="Tor", Clock=False
):
    """ Compute geometrical attributes of a Struct object """

    # Make Poly closed, counter-clockwise, with '(cc,N)' layout and arrayorder
    try:
        Poly = _GG.format_poly(Poly, order="C", Clock=False, close=True,
                               Test=True)
    except Exception as excp:
        print(excp)
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
        if Vol <= 0.0:
            msg = ("Pb. with volume computation for Struct of type 'Tor' !\n"
                   + "\t- Vol = {}\n".format(Vol)
                   + "\t- Poly = {}\n\n".format(str(Poly))
                   + "  => Probably corrupted polygon\n"
                   + "  => Please check polygon is not self-intersecting")
            raise Exception(msg)

    # Compute the non-normalized vector of each side of the Poly
    Vect = np.diff(Poly, n=1, axis=1)
    Vect = fPfmt(Vect)

    # Compute the normalised vectors directed inwards
    Vin = np.array([Vect[1, :], -Vect[0, :]])
    Vin = -Vin  # Poly is Counter Clock-wise as defined above
    Vin = Vin / np.hypot(Vin[0, :], Vin[1, :])[np.newaxis, :]
    Vin = fPfmt(Vin)

    poly = _GG.format_poly(
        Poly,
        order=arrayorder,
        Clock=Clock,
        close=False,
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


# ==============================================================================
# = Ves sampling functions
# ==============================================================================


def _Ves_get_sample_checkinputs(
    res=None,
    domain=None,
    resMode=None,
    ind=None,
    which='volume'
):
    """ Check inputs for all sampling routines """

    # res
    dres = {'edge': 1, 'cross': 2, 'surface': 2, 'volume': 3}
    if res is None:
        res = _SAMPLE_RES[which]
    ltypes = [int, float, np.int_, np.float64]
    c0 = (type(res) in ltypes
          or (hasattr(res, "__iter__")
              and len(res) == dres[which]
              and all([type(ds) in ltypes for ds in res])))
    if not c0:
        msg = ("Arg res must be either:\n"
               + "\t- float: unique resolution for all directions\n"
               + "\t- iterable of {} floats\n".format(dres[which])
               + "  You provided:\n{}".format(res))
        raise Exception(msg)
    if type(res) in ltypes:
        if which != 'edge':
            res = [float(res) for ii in range(dres[which])]
    else:
        if which == 'edge':
            msg = ("For edge, res cannot be an iterable!\n"
                   + "\t- res: {}".format(res))
            raise Exception(msg)
        res = [float(res[ii]) for ii in range(dres[which])]

    # domain (i.e.: sub-domain to be sampled, defined by its limits)
    ddomain = {'edge': 2, 'cross': 2, 'surface': 3, 'volume': 3}
    if domain is None:
        domain = [None for ii in range(ddomain[which])]
    c0 = (hasattr(domain, "__iter__")
          and len(domain) == ddomain[which]
          and all([dd is None
                   or (hasattr(dd, "__iter__")
                       and len(dd) == 2
                       and all([ss is None
                                or type(ss) in ltypes
                                for ss in dd]))
                   for dd in domain]))
    if not c0:
        msg = ("Arg domain must be a len()={} iterable".format(ddomain[which])
               + " where each element can be:\n"
               + "\t- an iterable 2 floats: [lower, upper] bounds\n"
               + "\t- None: no bounds\n"
               + "  You provided:\n{}".format(domain))
        raise Exception(msg)
    for ii in range(len(domain)):
        if domain[ii] is not None:
            domain[ii] = [float(domain[ii][0])
                          if domain[ii][0] is not None else None,
                          float(domain[ii][1])
                          if domain[ii][1] is not None else None]

    # resMode
    if resMode is None:
        resMode = _SAMPLE_RESMODE[which]
    c0 = isinstance(resMode, str) and resMode.lower() in ["abs", "rel"]
    if not c0:
        msg = ("Arg resMode must be in ['abs','rel']!\n"
               + "  You provided:\n{}".format(resMode))
        raise Exception(msg)
    resMode = resMode.lower()

    # ind (indices of points to be recovered)
    c0 = (ind is None
          or (isinstance(ind, np.ndarray)
              and ind.ndim == 1
              and 'int' in ind.dtype.name
              and np.all(ind >= 0))
          or (which == 'surface'
              and isinstance(ind, list)
              and all([isinstance(indi, np.ndarray)
                       and indi.ndim == 1
                       and 'int' in indi.dtype.name
                       and np.all(indi >= 0) for indi in ind])))
    if not c0:
        msg = ("Arg ind must be either:\n"
               + "\t- None: domain is used instead\n"
               + "\t- 1d np.ndarray of positive int: indices\n")
        if which == 'surface':
            msg += "\t- list of 1d np.ndarray of positive indices\n"
        msg += "  You provided:\n{}".format(ind)
        raise Exception(msg)

    if isinstance(ind, np.ndarray):
        ind = ind.astype(np.int64)
    elif isinstance(ind, list):
        ind = [ii.astype(np.int64) for ii in ind]

    return res, domain, resMode, ind


def _Ves_get_sampleEdge(
    VPoly,
    res=None,
    domain=None,
    resMode=None,
    offsetIn=0.0,
    VIn=None,
    margin=1.0e-9
):

    # -------------
    #  Check inputs

    # standard
    res, domain, resMode, ind = _Ves_get_sample_checkinputs(
        res=res,
        domain=domain,
        resMode=resMode,
        ind=None,
        which='edge',
    )

    # specific
    ltypes = [int, float, np.int_, np.float64]
    assert type(offsetIn) in ltypes

    # -------------
    #  Compute

    Pts, reseff, ind, N, Rref, VPolybis = _GG.discretize_vpoly(
        VPoly,
        float(res),
        mode=resMode,
        D1=domain[0],
        D2=domain[1],
        margin=margin,
        DIn=float(offsetIn),
        VIn=VIn,
    )
    return Pts, reseff, ind


def _Ves_get_sampleCross(
    VPoly,
    Min1,
    Max1,
    Min2,
    Max2,
    res=None,
    domain=None,
    resMode=None,
    ind=None,
    margin=1.0e-9,
    mode="flat",
):

    # -------------
    #  Check inputs

    # standard
    res, domain, resMode, ind = _Ves_get_sample_checkinputs(
        res=res,
        domain=domain,
        resMode=resMode,
        ind=ind,
        which='cross',
    )

    # specific
    assert mode in ["flat", "imshow"]

    # -------------
    #  Compute

    MinMax1 = np.array([Min1, Max1])
    MinMax2 = np.array([Min2, Max2])
    if ind is None:
        if mode == "flat":
            Pts, dS, ind, d1r, d2r = _GG.discretize_segment2d(
                MinMax1,
                MinMax2,
                res[0],
                res[1],
                D1=domain[0],
                D2=domain[1],
                mode=resMode,
                VPoly=VPoly,
                margin=margin,
            )
            out = (Pts, dS, ind, (d1r, d2r))
        else:
            x1, d1r, ind1, N1 = _GG._Ves_mesh_dlfromL_cython(
                MinMax1, res[0], domain[0], Lim=True,
                dLMode=resMode, margin=margin
            )
            x2, d2r, ind2, N2 = _GG._Ves_mesh_dlfromL_cython(
                MinMax2, res[1], domain[1], Lim=True,
                dLMode=resMode, margin=margin
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
            MinMax1,
            MinMax2,
            res[0],
            res[1],
            ind,
            dSMode=resMode,
            margin=margin,
        )
        out = (Pts, dS, ind, (d1r, d2r))
    return out


def _Ves_get_sampleS(
    VPoly,
    res=None,
    domain=None,
    resMode="abs",
    ind=None,
    offsetIn=0.0,
    VIn=None,
    VType="Tor",
    VLim=None,
    nVLim=None,
    returnas="(X,Y,Z)",
    margin=1.0e-9,
    Multi=False,
    Ind=None,
):
    """ Sample the surface """

    # -------------
    #  Check inputs

    # standard
    res, domain, resMode, ind = _Ves_get_sample_checkinputs(
        res=res,
        domain=domain,
        resMode=resMode,
        ind=ind,
        which='surface',
    )

    # nVLim and VLim
    if not (type(nVLim) in [int, np.int_] and nVLim >= 0):
        msg = ("Arg nVLim must be a positive int\\n"
               + "  You provided:\n{} ({})".format(nVLim, type(nVLim)))
        raise Exception(msg)
    VLim = None if (VLim is None or nVLim == 0) else np.array(VLim)

    if not isinstance(Multi, bool):
        msg = ("Arg Multi must be a bool!\n"
               + "  You provided:\n{}".format(Multi))
        raise Exception(msg)

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
            if isinstance(ind, np.ndarray):
                ind = [ind for ii in range(len(Ind))]
            elif not (isinstance(ind, list) and len(ind) == len(Ind)):
                msg = ("Arg ind must be a list of same len() as Ind!\n"
                       + "\t- provided: {}".format(ind))
                raise Exception(msg)

    else:
        VLim = [None] if VLim is None else [VLim.ravel()]
        if ind is not None:
            if not isinstance(ind, np.ndarray):
                msg = ("ind must be a np.ndarray if nVLim == 1\n"
                       + "\t- provided: {}".format(ind))
                raise Exception(msg)
        Ind = [0]

    if ind is None:
        pts, dS, ind, reseff = (
            [0 for ii in Ind],
            [0 for ii in Ind],
            [0 for ii in Ind],
            [[0, 0] for ii in Ind],
        )
        if VType.lower() == "tor":
            for ii in range(0, len(Ind)):
                if VLim[Ind[ii]] is None:
                    (pts[ii],
                     dS[ii],
                     ind[ii],
                     NL,
                     reseff[ii][0],
                     Rref,
                     reseff[ii][1],
                     nRPhi0,
                     VPbis) = _GG._Ves_Smesh_Tor_SubFromD_cython(
                         res[0],
                         res[1],
                         VPoly,
                         DR=domain[0],
                         DZ=domain[1],
                         DPhi=domain[2],
                         DIn=offsetIn,
                         VIn=VIn,
                         PhiMinMax=None,
                         Out=returnas,
                         margin=margin,
                     )
                else:
                    (pts[ii],
                     dS[ii],
                     ind[ii],
                     NL,
                     reseff[ii][0],
                     Rref,
                     dR0r,
                     dZ0r,
                     reseff[ii][1],
                     VPbis) = _GG._Ves_Smesh_TorStruct_SubFromD_cython(
                         VLim[Ind[ii]],
                         res[0],
                         res[1],
                         VPoly,
                         DR=domain[0],
                         DZ=domain[1],
                         DPhi=domain[2],
                         DIn=offsetIn,
                         VIn=VIn,
                         Out=returnas,
                         margin=margin,
                     )
                    reseff[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0, len(Ind)):
                (pts[ii],
                 dS[ii],
                 ind[ii],
                 NL,
                 reseff[ii][0],
                 Rref,
                 reseff[ii][1],
                 dY0r,
                 dZ0r,
                 VPbis) = _GG._Ves_Smesh_Lin_SubFromD_cython(
                     VLim[Ind[ii]],
                     res[0],
                     res[1],
                     VPoly,
                     DX=domain[0],
                     DY=domain[1],
                     DZ=domain[2],
                     DIn=offsetIn,
                     VIn=VIn,
                     margin=margin,
                 )
                reseff[ii] += [dY0r, dZ0r]
    else:
        ind = ind if Multi else [ind]
        pts, dS, reseff = (
            [np.ones((3, 0)) for ii in Ind],
            [0 for ii in Ind],
            [[0, 0] for ii in Ind],
        )
        if VType.lower() == "tor":
            for ii in range(0, len(Ind)):
                if ind[Ind[ii]].size > 0:
                    if VLim[Ind[ii]] is None:
                        out_loc = _GG._Ves_Smesh_Tor_SubFromInd_cython(
                            res[0],
                            res[1],
                            VPoly,
                            ind[Ind[ii]],
                            DIn=offsetIn,
                            VIn=VIn,
                            PhiMinMax=None,
                            Out=returnas,
                            margin=margin,
                        )
                        pts[ii], dS[ii], NL, reseff[ii][0], Rref = out_loc[:5]
                        reseff[ii][1], nRPhi0, VPbis = out_loc[5:]
                    else:
                        out_loc = _GG._Ves_Smesh_TorStruct_SubFromInd_cython(
                            VLim[Ind[ii]],
                            res[0],
                            res[1],
                            VPoly,
                            ind[Ind[ii]],
                            DIn=offsetIn,
                            VIn=VIn,
                            Out=returnas,
                            margin=margin,
                        )
                        pts[ii], dS[ii], NL, reseff[ii][0], Rref = out_loc[:5]
                        dR0r, dZ0r, reseff[ii][1], VPbis = out_loc[5:]
                        reseff[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0, len(Ind)):
                if ind[Ind[ii]].size > 0:
                    out_loc = _GG._Ves_Smesh_Lin_SubFromInd_cython(
                        VLim[Ind[ii]],
                        res[0],
                        res[1],
                        VPoly,
                        ind[Ind[ii]],
                        DIn=offsetIn,
                        VIn=VIn,
                        margin=margin,
                    )
                    pts[ii], dS[ii], NL, reseff[ii][0], Rref = out_loc[:5]
                    reseff[ii][1], dY0r, dZ0r, VPbis = out_loc[5:]
                    reseff[ii] += [dY0r, dZ0r]

    if len(VLim) == 1:
        pts, dS, ind, reseff = pts[0], dS[0], ind[0], reseff[0]
    return pts, dS, ind, reseff


def _Ves_get_sampleV(
    VPoly,
    Min1,
    Max1,
    Min2,
    Max2,
    res=None,
    domain=None,
    resMode=None,
    ind=None,
    VType="Tor",
    VLim=None,
    returnas="(X,Y,Z)",
    margin=1.0e-9,
    algo="new",
    num_threads=48,
):
    """ Sample the volume """

    # -------------
    #  Check inputs
    res, domain, resMode, ind = _Ves_get_sample_checkinputs(
        res=res,
        domain=domain,
        resMode=resMode,
        ind=ind,
        which='volume',
    )

    # ------------
    # Computation

    MinMax1 = np.array([Min1, Max1])
    MinMax2 = np.array([Min2, Max2])
    VLim = None if VType.lower() == "tor" else np.array(VLim).ravel()
    reseff = [None, None, None]
    if ind is None:
        if VType.lower() == "tor":
            if algo.lower() == "new":
                (pts, dV, ind,
                 reseff[0],
                 reseff[1],
                 reseff[2],
                 sz_r, sz_z,
                 ) = _GG._Ves_Vmesh_Tor_SubFromD_cython(
                    res[0],
                    res[1],
                    res[2],
                    MinMax1,
                    MinMax2,
                    DR=domain[0],
                    DZ=domain[1],
                    DPhi=domain[2],
                    limit_vpoly=VPoly,
                    out_format=returnas,
                    margin=margin,
                    num_threads=num_threads,
                )
            else:
                (pts, dV, ind,
                 reseff[0],
                 reseff[1],
                 reseff[2]) = _GG._Ves_Vmesh_Tor_SubFromD_cython_old(
                    res[0],
                    res[1],
                    res[2],
                    MinMax1,
                    MinMax2,
                    DR=domain[0],
                    DZ=domain[1],
                    DPhi=domain[2],
                    VPoly=VPoly,
                    Out=returnas,
                    margin=margin,
                )

        else:
            (pts, dV, ind,
             reseff[0],
             reseff[1],
             reseff[2]) = _GG._Ves_Vmesh_Lin_SubFromD_cython(
                res[0],
                res[1],
                res[2],
                VLim,
                MinMax1,
                MinMax2,
                DX=domain[0],
                DY=domain[1],
                DZ=domain[2],
                limit_vpoly=VPoly,
                margin=margin,
            )
    else:
        if VType.lower() == "tor":
            if algo.lower() == "new":
                (pts, dV,
                 reseff[0],
                 reseff[1],
                 reseff[2]) = _GG._Ves_Vmesh_Tor_SubFromInd_cython(
                    res[0],
                    res[1],
                    res[2],
                    MinMax1,
                    MinMax2,
                    ind,
                    Out=returnas,
                    margin=margin,
                    num_threads=num_threads,
                )
            else:
                (pts, dV,
                 reseff[0],
                 reseff[1],
                 reseff[2]) = _GG._Ves_Vmesh_Tor_SubFromInd_cython_old(
                    res[0],
                    res[1],
                    res[2],
                    MinMax1,
                    MinMax2,
                    ind,
                    Out=returnas,
                    margin=margin,
                )
        else:
            (pts, dV,
             reseff[0],
             reseff[1],
             reseff[2]) = _GG._Ves_Vmesh_Lin_SubFromInd_cython(
                 res[0],
                 res[1],
                 res[2],
                 VLim,
                 MinMax1,
                 MinMax2,
                 ind,
                 margin=margin
             )
    return pts, dV, ind, reseff


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
        nphi = np.ones((noccur,), dtype=np.int64)
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
        Int = scpintg.simpson(Vals, x=None, dx=dLr)
    elif method == "romb":
        Int = scpintg.romb(Vals, dx=dLr, show=False)
    return Int