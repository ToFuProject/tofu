

import copy
import warnings


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import Polygon as plg
import datastock as ds


from . import _comp_solidangles


__all__ = ['compute_etendue']


# #############################################################################
# #############################################################################
#                    Main routines
# #############################################################################


def compute_etendue(
    det=None,
    aperture=None,
    analytical=None,
    numerical=None,
    check=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    verb=None,
    plot=None,
):
    """ Only works for a set of detectors associated to a single aperture

    Typical use:
        - pinhole cameras

    """

    # -------------
    # check inputs

    (
        det, ldeti, aperture,
        analytical, numerical,
        res, margin_par, margin_perp,
        check, verb, plot,
    ) = _compute_etendue_check(
        det=det,
        aperture=aperture,
        analytical=analytical,
        numerical=numerical,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        check=check,
        verb=verb,
        plot=plot,
    )

    # ----------
    # prepare

    (
        det_area, ap_area, distances,
        los_x, los_y, los_z,
        cos_los_det, cos_los_ap,
        solid_angles, res, pix_ap,
    ) = _compute_etendue_prepare(
        ldeti=ldeti,
        aperture=aperture,
        res=res,
    )

    shape = distances.shape

    # --------------------
    # compute analytically

    if analytical is True:
        etend0 = np.full(tuple(np.r_[3, shape]), np.nan)

        # 0th order
        etend0[0, :] = ap_area * det_area / distances**2

        # 1st order
        etend0[1, :] = (
            cos_los_ap * ap_area
            * cos_los_det * det_area / distances**2
        )

        # 2nd order
        etend0[2, :] = cos_los_ap * ap_area * solid_angles

    else:
        etend0 = None

    # --------------------
    # compute numerically

    if numerical is True:
        etend1 = _compute_etendue_numerical(
            ldeti=ldeti,
            aperture=aperture,
            pix_ap=pix_ap,
            res=res,
            los_x=los_x,
            los_y=los_y,
            los_z=los_z,
            margin_par=margin_par,
            margin_perp=margin_perp,
            check=check,
            verb=verb,
        )

    else:
        etend1 = None

    # --------------------
    # optional plotting

    if plot is True:
        dax = _plot_etendues(
            etend0=etend0,
            etend1=etend1,
            res=res,
        )

    # --------
    # reshape

    sh0 = det['cents_x'].shape
    if sh0 == ():
        sh0 = (1,)

    # etend0
    if etend0 is not None and etend0.shape[1:] != sh0:
        etend0 = etend0.reshape(tuple(np.r_[etend0.shape[0], sh0]))

    # etend1
    if etend1 is not None and etend1.shape[1:] != sh0:
        etend1 = etend1.reshape(tuple(np.r_[res.size, sh0]))

    # los
    if los_x.shape != sh0:
        los_x = los_x.reshape(sh0)
        los_y = los_y.reshape(sh0)
        los_z = los_z.reshape(sh0)

    # --------------------
    # return

    dout = {
        'analytical': etend0,
        'numerical': etend1,
        'res': res,
        'los_x': los_x,
        'los_y': los_y,
        'los_z': los_z,
    }

    return dout


# #############################################################################
# #############################################################################
#                   input checking routine
# #############################################################################


def _compute_etendue_check(
    det=None,
    aperture=None,
    analytical=None,
    numerical=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    check=None,
    verb=None,
    plot=None,
):
    """ Check conformity of inputs

    """

    # -----------
    # det

    # check keys
    lk = [
        'cents_x', 'cents_y', 'cents_z',
        'nin_x', 'nin_y', 'nin_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
        'outline_x0', 'outline_x1',
    ]

    c0 = (
        isinstance(det, dict)
        and all([kk in det.keys() for kk in lk])
    )
    if not c0:
        lstr = [f"\t- {k0}" for k0 in lk]
        msg = (
            "Arg det must be a dict with the following keys:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # check values
    for k0 in lk:
        if isinstance(det[k0], (list, tuple)):
            det[k0] = np.atleast_1d(det[k0]).ravel()

        c0 = isinstance(det[k0], np.ndarray) or np.isscalar(det[k0])
        if not c0:
            msg = f"Arg det['{k0}'] must be a np.ndarray or scalar"
            raise Exception(msg)

        if k0 in ['outline_x0', 'outline_x1'] and det[k0].ndim > 1:
            msg = "Arg det['outline_x0'] and det['outline_x1'] must be 1d"
            raise Exception(msg)

    # check shapes
    dshape = {
        0: ['outline_x0', 'outline_x1'],
        1: [
            'nin_x', 'nin_y', 'nin_z',
            'e0_x', 'e0_y', 'e0_z',
            'e1_x', 'e1_y', 'e1_z',
        ],
        2: ['cents_x', 'cents_y', 'cents_z'],
    }
    for k0, v0 in dshape.items():
        if len(set([det[v1].shape for v1 in v0])) > 1:
            lstr = [f"\t- {v1}" for v1 in v0]
            msg = (
                "The following args must share the same shape:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    shaped = det['cents_x'].shape
    if det['cents_x'].shape != det['nin_x'].shape:
        if np.isscalar(det['nin_x']):
            assert det.get('parallel', True) is True
        else:
            msg = (
                "Arg det['nin_x'], det['nin_y'], det['nin_z'] must have "
                "the same shape as det['cents_z']"
            )
            raise Exception(msg)

    # check outline not closed
    if (
        det['outline_x0'][0] == det['outline_x0'][-1]
        and det['outline_x1'][0] == det['outline_x1'][-1]
    ):
        det['outline_x0'] = det['outline_x0'][:-1]
        det['outline_x1'] = det['outline_x1'][:-1]

    # normalization
    norms = np.sqrt(det['nin_x']**2 + det['nin_y']**2 + det['nin_z']**2)
    det['nin_x'] = det['nin_x'] / norms
    det['nin_y'] = det['nin_y'] / norms
    det['nin_z'] = det['nin_z'] / norms

    # -----------
    # ldeti

    lk = [
        k0 for k0 in det.keys()
        if any([k0.endswith(ss) for ss in ['_x', '_y', '_z']])
        and not np.isscalar(det[k0])
    ]
    deti = {k0: det[k0].ravel() for k0 in lk}

    ldeti = [
        {
            k0: deti[k0][ii]
            if k0 in lk else v0
            for k0, v0 in det.items()
        }
        for ii in range(det['cents_x'].size)
    ]

    # -----------
    # aperture

    lk = [
        'poly_x', 'poly_y', 'poly_z',
        'nin', 'e0', 'e1',
    ]

    c0 = (
        isinstance(aperture, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            and isinstance(v0.get('reflector', False), bool)
            and all([kk in v0.keys() for kk in lk])
            for k0, v0 in aperture.items()
        ])
    )
    if not c0:
        lstr = [f"\t- {k0}" for k0 in lk]
        msg = (
            "Arg aperture must be a dict of sub-dict with keys:\n"
            + "\n".join(lstr)
            + f"\nProvided:\n{aperture}"
        )
        raise Exception(msg)

    # check each case 
    for k0, v0 in aperture.items():

        # check values
        for k1 in lk:
            if isinstance(v0[k1], (list, tuple)):
                aperture[k0][k1] = np.atleast_1d(v0[k1]).ravel().astype(float)

            if not isinstance(aperture[k0][k1], np.ndarray):
                msg = f"Arg aperture['{k0}']['{k1}'] must ba a np.ndarray"
                raise Exception(msg)

            if aperture[k0][k1].ndim > 1:
                msg = f"Arg aperture['{k0}']['{k1}'] must be 1d"
                raise Exception(msg)

            if 'poly_' not in k1 and aperture[k0][k1].shape != (3,):
                msg = f"Arg aperture['{k0}']['{k1}'] must have shape (3,)"
                raise Exception(msg)

        v0 = aperture[k0]

        # check shapes
        dshape = {
            0: ['poly_x', 'poly_y', 'poly_z'],
            1: ['nin', 'e0', 'e1'],
        }
        for k1, v1 in dshape.items():
            if len(set([v0[v2].shape for v2 in v1])) > 1:
                lstr = [f"\t- {v2}" for v2 in v1]
                msg = (
                    "The following args must share the same shape:\n"
                    + "\n".join(lstr)
                )
                raise Exception(msg)

        # check not closed poly
        if (
            v0['poly_x'][0] == v0['poly_x'][-1]
            and v0['poly_y'][0] == v0['poly_y'][-1]
            and v0['poly_z'][0] == v0['poly_z'][-1]
        ):
            v0['poly_x'] = v0['poly_x'][:-1]
            v0['poly_y'] = v0['poly_y'][:-1]
            v0['poly_z'] = v0['poly_z'][:-1]

        # normalization
        norm = np.linalg.norm(v0['nin'])
        v0['nin'] = v0['nin'] / norm

        # derive cents
        if 'cent' not in v0.keys():
            v0['cent'] = np.r_[
                np.mean(v0['poly_x']),
                np.mean(v0['poly_y']),
                np.mean(v0['poly_z']),
            ]

    # -----------
    # analytical

    analytical = ds._generic_check._check_var(
        analytical, 'analytical',
        types=bool,
        default=True,
    )

    # -----------
    # numerical

    numerical = ds._generic_check._check_var(
        numerical, 'numerical',
        types=bool,
        default=True,
    )

    # -----------
    # res

    if res is not None:
        res = np.atleast_1d(res).ravel()

    # -----------
    # margin_par

    margin_par = ds._generic_check._check_var(
        margin_par, 'margin_par',
        types=float,
        default=0.05,
    )

    # -----------
    # margin_perp

    margin_perp = ds._generic_check._check_var(
        margin_perp, 'margin_perp',
        types=float,
        default=0.05,
    )

    # -----------
    # check

    check = ds._generic_check._check_var(
        check, 'check',
        types=bool,
        default=True,
    )

    # -----------
    # verb

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # -----------
    # plot

    if plot is None:
        plot = True
    if not isinstance(plot, bool):
        msg = "Arg plot must be a bool"
        raise Exception(msg)

    return (
        det, ldeti, aperture, analytical, numerical,
        res, margin_par, margin_perp, check, verb, plot,
    )


# #############################################################################
# #############################################################################
#                   preparation routine
# #############################################################################


def _project_poly_on_plane_from_pt(
    pt=None,
    lpoly_x=None,
    lpoly_y=None,
    lpoly_z=None,
    plane_pt=None,
    plane_nin=None,
    plane_e0=None,
    plane_e1=None,
):

    isok = True
    for ii in range(len(lpoly_x)):

        sca0 = (
            (plane_pt[0] - pt[0])*plane_nin[0]
            + (plane_pt[1] - pt[1])*plane_nin[1]
            + (plane_pt[2] - pt[2])*plane_nin[2]
        )

        vx = lpoly_x[ii] - pt[0]
        vy = lpoly_y[ii] - pt[1]
        vz = lpoly_z[ii] - pt[2]

        sca1 = vx*plane_nin[0] + vy*plane_nin[1] + vz*plane_nin[2]

        k = sca0 / sca1

        px = pt[0] + k * vx
        py = pt[1] + k * vy
        pz = pt[2] + k * vz

        p0 = (
            (px - plane_pt[0])*plane_e0[0]
            + (py - plane_pt[1])*plane_e0[1]
            + (pz - plane_pt[2])*plane_e0[2]
        )
        p1 = (
            (px - plane_pt[0])*plane_e1[0]
            + (py - plane_pt[1])*plane_e1[1]
            + (pz - plane_pt[2])*plane_e1[2]
        )

        if ii == 0:
            p_a = plg.Polygon(np.array([p0, p1]).T)
        else:
            p_a = p_a & plg.Polygon(np.array([p0, p1]).T)

        if p_a.nPoints() < 3:
            isok = False
            break

    if isok is False:
        return None, None, None, None, None, None

    p0, p1 = np.array(p_a.contour(0)).T
    px = plane_pt[0] + p0*plane_e0[0] + p1*plane_e1[0]
    py = plane_pt[1] + p0*plane_e0[1] + p1*plane_e1[1]
    pz = plane_pt[2] + p0*plane_e0[2] + p1*plane_e1[2]

    return p_a, p0, p1, px, py, pz


def _compute_etendue_prepare(
    ldeti=None,
    aperture=None,
    res=None,
):

    # -------------------------
    # intersection of apertures

    kap_ref = list(aperture.keys())[0]
    lpoly_x = [v0['poly_x'] for v0 in aperture.values()]
    lpoly_y = [v0['poly_y'] for v0 in aperture.values()]
    lpoly_z = [v0['poly_z'] for v0 in aperture.values()]

    # prepare data
    nd = len(ldeti)
    ap_area = np.zeros((nd,), dtype=float)
    los_x = np.full((nd,), np.nan)
    los_y = np.full((nd,), np.nan)
    los_z = np.full((nd,), np.nan)
    solid_angles = np.zeros((nd,), dtype=float)
    mindiff = np.full((nd,), np.nan)

    # store projected intersection of apertures (3d), per pix
    # useful later for estimating the plane to be sample (numerical)
    pix_ap = []

    for ii in range(nd):

        # ap
        p_a, p0, p1, px, py, pz = _project_poly_on_plane_from_pt(
            pt=np.r_[
                ldeti[ii]['cents_x'],
                ldeti[ii]['cents_y'],
                ldeti[ii]['cents_z'],
            ],
            lpoly_x=lpoly_x,
            lpoly_y=lpoly_y,
            lpoly_z=lpoly_z,
            plane_pt=aperture[kap_ref]['cent'],
            plane_nin=aperture[kap_ref]['nin'],
            plane_e0=aperture[kap_ref]['e0'],
            plane_e1=aperture[kap_ref]['e1'],
        )

        if p_a is None:
            pix_ap.append(None)
            continue

        else:
            ap_area[ii] = p_a.area()
            ap_cent = (
                aperture[kap_ref]['cent']
                + p_a.center()[0] * aperture[kap_ref]['e0']
                + p_a.center()[1] * aperture[kap_ref]['e1']
            )
            mindiff[ii] = np.sqrt(np.min(np.diff(p0)**2 + np.diff(p1)**2))

            # ----------------------------------
            # los, distances, cosines

            los_x[ii] = ap_cent[0] - ldeti[ii]['cents_x']
            los_y[ii] = ap_cent[1] - ldeti[ii]['cents_y']
            los_z[ii] = ap_cent[2] - ldeti[ii]['cents_z']

            # ------------
            # solid angles

            solid_angles[ii] = _comp_solidangles.calc_solidangle_apertures(
                # observation points
                pts_x=ap_cent[0],
                pts_y=ap_cent[1],
                pts_z=ap_cent[2],
                # polygons
                apertures=None,
                detectors=ldeti[ii],
                # possible obstacles
                config=None,
                # parameters
                visibility=False,
                return_vector=False,
            ).ravel()[0]

            pix_ap.append((px, py, pz))

    # -------------
    # normalize los

    distances = np.sqrt(los_x**2 + los_y**2 + los_z**2)

    los_x = los_x / distances
    los_y = los_y / distances
    los_z = los_z / distances

    # ------
    # angles

    cos_los_det = (
        los_x * ldeti[ii]['nin_x']
        + los_y * ldeti[ii]['nin_y']
        + los_z * ldeti[ii]['nin_z']
    )

    cos_los_ap = (
        los_x * aperture[kap_ref]['nin'][0]
        + los_y * aperture[kap_ref]['nin'][1]
        + los_z * aperture[kap_ref]['nin'][2]
    )

    # -----------
    # surfaces

    # det
    if ldeti[ii].get('pix area') is None:
        det_area = plg.Polygon(np.array([
            ldeti[ii]['outline_x0'],
            ldeti[ii]['outline_x1'],
        ]).T).area()
    else:
        det_area = ldeti[ii]['pix area']

    # -------------------------------------
    # det outline discretization resolution

    if res is None:

        res = min(
            np.sqrt(det_area),
            np.sqrt(np.min(ap_area[ap_area > 0.])),
            np.nanmin(mindiff),
        ) * np.r_[1., 0.5, 0.1]

    iok = np.isfinite(res)
    iok[iok] = res[iok] > 0
    if not np.any(iok):
        res = np.r_[0.001]
    else:
        res = res[iok]

    return (
        det_area, ap_area, distances,
        los_x, los_y, los_z,
        cos_los_det, cos_los_ap, solid_angles, res, pix_ap,
    )


# #############################################################################
# #############################################################################
#           Numerical etendue estimation routine
# #############################################################################


def _compute_etendue_numerical(
    ldeti=None,
    aperture=None,
    pix_ap=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    los_x=None,
    los_y=None,
    los_z=None,
    check=None,
    verb=None,
):

    # shape0 = det['cents_x'].shape
    nd = len(ldeti)

    ap_ind = np.cumsum([v0['poly_x'].size for v0 in aperture.values()][:-1])

    ap_tot_px = np.concatenate(tuple(
        [v0['poly_x'] for v0 in aperture.values()]
    ))
    ap_tot_py = np.concatenate(tuple(
        [v0['poly_y'] for v0 in aperture.values()]
    ))
    ap_tot_pz = np.concatenate(tuple(
        [v0['poly_z'] for v0 in aperture.values()]
    ))

    # ------------------------------
    # Get plane perpendicular to los

    etendue = np.full((res.size, nd), np.nan)
    for ii in range(nd):

        if verb is True:
            msg = f"Numerical etendue for det {ii+1} / {nd}"
            print(msg)

        if np.isnan(los_x[ii]):
            continue

        # get det corners to aperture corners vectors
        out_c_x0 = np.r_[0, ldeti[ii]['outline_x0']]
        out_c_x1 = np.r_[0, ldeti[ii]['outline_x1']]

        # det poly 3d
        det_Px = (
            ldeti[ii]['cents_x']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_x']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_x']
        )
        det_Py = (
            ldeti[ii]['cents_y']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_y']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_y']
        )
        det_Pz = (
            ldeti[ii]['cents_z']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_z']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_z']
        )

        # det to ap vectors
        PA_x = ap_tot_px[:, None] - det_Px[None, :]
        PA_y = ap_tot_py[:, None] - det_Py[None, :]
        PA_z = ap_tot_pz[:, None] - det_Pz[None, :]

        sca1 = PA_x * los_x[ii] + PA_y * los_y[ii] + PA_z * los_z[ii]
        # get length along los
        k_los = (1. + margin_par) * np.max(sca1)

        # get center of plane perpendicular to los
        c_los_x = ldeti[ii]['cents_x'] + k_los * los_x[ii]
        c_los_y = ldeti[ii]['cents_y'] + k_los * los_y[ii]
        c_los_z = ldeti[ii]['cents_z'] + k_los * los_z[ii]

        # get projections of corners on plane perp. to los
        sca0 = (
            (c_los_x - det_Px[None, :]) * los_x[ii]
            + (c_los_y - det_Py[None, :]) * los_y[ii]
            + (c_los_z - det_Pz[None, :]) * los_z[ii]
        )
        k_plane = sca0 / sca1

        # get LOS-specific unit vectors

        e0_xi = (
            los_y[ii] * ldeti[ii]['e1_z'] - los_z[ii] * ldeti[ii]['e1_y']
        )
        e0_yi = (
            los_z[ii] * ldeti[ii]['e1_x'] - los_x[ii] * ldeti[ii]['e1_z']
        )
        e0_zi = (
            los_x[ii] * ldeti[ii]['e1_y'] - los_y[ii] * ldeti[ii]['e1_x']
        )

        e0_normi = np.sqrt(e0_xi**2 + e0_yi**2 + e0_zi**2)
        e0_xi = e0_xi / e0_normi
        e0_yi = e0_yi / e0_normi
        e0_zi = e0_zi / e0_normi

        e1_xi = los_y[ii] * e0_zi - los_z[ii] * e0_yi
        e1_yi = los_z[ii] * e0_xi - los_x[ii] * e0_zi
        e1_zi = los_x[ii] * e0_yi - los_y[ii] * e0_xi

        # get projections on det_e0 and det_e1 in plane

        x0 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e0_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e0_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e0_zi,
            ap_ind,
            axis=0,
        )
        x1 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e1_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e1_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e1_zi,
            ap_ind,
            axis=0,
        )

        x0_min = np.max([np.min(x0s) for x0s in x0])
        x0_max = np.min([np.max(x0s) for x0s in x0])
        x1_min = np.max([np.min(x1s) for x1s in x1])
        x1_max = np.min([np.max(x1s) for x1s in x1])

        w0 = x0_max - x0_min
        w1 = x1_max - x1_min

        min_res = min(2*margin_perp*w0, 2*margin_perp*w1)
        too_large = res >= min_res
        if np.any(too_large):
            msg = (
                f"Minimum etendue resolution for det {ii} / {nd}: {min_res}\n"
                "The following res values may lead to errors:\n"
                f"\t- res values = {res}\n"
                f"\t- too large  = {too_large}\n"
            )
            warnings.warn(msg)

        # -------------------
        # Discretize aperture

        for jj in range(res.size):

            coef = 1. + 2.*margin_perp
            n0 = int(np.ceil(coef*w0 / res[jj]))
            n1 = int(np.ceil(coef*w1 / res[jj]))

            d0 = coef*w0 / n0
            d1 = coef*w1 / n1

            ds = d0 * d1

            pts_0 = np.linspace(
                x0_min - margin_perp*w0,
                x0_max + margin_perp*w0,
                n0 + 1,
            )
            pts_1 = np.linspace(
                x1_min - margin_perp*w1,
                x1_max + margin_perp*w1,
                n1 + 1,
            )
            pts_0 = 0.5 * (pts_0[1:] + pts_0[:-1])
            pts_1 = 0.5 * (pts_1[1:] + pts_1[:-1])

            # debug
            # n0, n1 = 2, 2
            # pts_0 = np.r_[pts_0[0], pts_0[0]]
            # pts_1 = np.r_[0, 0]

            pts_x = (
                c_los_x + pts_0[:, None] * e0_xi + pts_1[None, :] * e1_xi
            ).ravel()
            pts_y = (
                c_los_y + pts_0[:, None] * e0_yi + pts_1[None, :] * e1_yi
            ).ravel()
            pts_z = (
                c_los_z + pts_0[:, None] * e0_zi + pts_1[None, :] * e1_zi
            ).ravel()

            if verb is True:
                msg = (
                    f"\tres = {res[jj]} ({jj+1} / {res.size})"
                    f"    nb. of points = {pts_x.size}"
                )
                print(msg)

            # ----------------------------------
            # compute solid angle for each pixel

            if check is True:
                solid_angle = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                )

                sar = solid_angle.reshape((n0, n1))
                c0 = (
                    ((pts_0[0] < x0_min) == np.all(sar[0, :] == 0))
                    and ((pts_0[-1] > x0_max) == np.all(sar[-1, :] == 0))
                    and ((pts_1[0] < x1_min) == np.all(sar[:, 0] == 0))
                    and ((pts_1[-1] > x1_max) == np.all(sar[:, -1] == 0))
                )
                if not c0 and not too_large[jj]:
                    # debug
                    plt.figure()
                    plt.imshow(
                        sar.T,
                        extent=(
                            x0_min - margin_perp*w0, x0_max + margin_perp*w0,
                            x1_min - margin_perp*w1, x1_max + margin_perp*w1,
                        ),
                        interpolation='nearest',
                        origin='lower',
                        aspect='equal',
                    )

                    lc = ['r', 'm', 'c', 'y']
                    for ss in range(len(x0)):
                        iss = np.r_[np.arange(0, x0[ss].shape[0]), 0]
                        plt.plot(
                            x0[ss][iss, :],
                            x1[ss][iss, :],
                            c=lc[ss%len(lc)],
                            marker='o',
                            ls='-',
                        )

                    plt.plot(
                        pts_0, np.mean(pts_1)*np.ones((n0,)),
                        c='k', marker='.', ls='None',
                    )
                    plt.plot(
                        np.mean(pts_0)*np.ones((n1,)), pts_1,
                        c='k', marker='.', ls='None',
                    )
                    plt.gca().set_xlabel('x0')
                    plt.gca().set_xlabel('x1')
                    # import pdb; pdb.set_trace()
                    msg = "Something is wrong with solid_angle or sampling"
                    raise Exception(msg)
                else:
                    etendue[jj, ii] = np.sum(solid_angle) * ds

            else:
                etendue[jj, ii] = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    summed=True,
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                ) * ds

    return etendue


# #############################################################################
# #############################################################################
#                   Plotting routine
# #############################################################################


def _plot_etendues(
    etend0=None,
    etend1=None,
    res=None,
):

    # -------------
    # prepare data

    nmax = 0
    if etend0 is not None:
        if etend0.ndim > 2:
            etend0 = etend0.reshape((etend0.shape[0], -1))
        nmax = max(nmax, etend0.shape[0])
    if etend1 is not None:
        if etend1.ndim > 2:
            etend1 = etend1.reshape((etend1.shape[0], -1))
        nmax = max(nmax, etend1.shape[0])

    x0 = None
    if etend0 is not None:
        x0 = [
            f'order {ii}' if ii < 3 else '' for ii in range(nmax)
        ]
    if etend1 is not None:
        x1 = [f'{res[ii]}' if ii < res.size-1 else '' for ii in range(nmax)]
        if x0 is None:
            x0 = x1
        else:
            x0 = [f'{x0[ii]}\n{x1[ii]}' for ii in range(nmax)]

    # -------------
    # prepare axes

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_ylabel('Etendue ' + r'($m^2.sr$)', size=12, fontweight='bold')
    ax.set_xlabel('order of approximation', size=12, fontweight='bold')

    ax.set_xticks(range(0, nmax))
    ax.set_xticklabels(x0)

    # -------------
    # plot

    if etend0 is not None:
        lines = ax.plot(
            etend0,
            ls='-',
            marker='o',
            ms=6,
        )
        lcol = [ll.get_color() for ll in lines]
    else:
        lcol = [None for ii in range(etend1.shape[1])]

    if etend1 is not None:
        for ii in range(etend1.shape[1]):
            ax.plot(
                etend1[:, ii],
                ls='--',
                marker='*',
                ms=6,
                color=lcol[ii],
            )

    # -------------
    # legend

    handles = [
        mlines.Line2D(
            [], [],
            c='k', marker='o', ls='-', ms=6,
            label='analytical',
        ),
        mlines.Line2D(
            [], [],
            c='k', marker='*', ls='--', ms=6,
            label='numerical',
        ),
    ]
    ax.legend(handles=handles)

    return ax
