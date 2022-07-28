

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


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
        det, aperture,
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
        det_surface, ap_surface, distances,
        los_x, los_y, los_z,
        cos_los_det, cos_los_ap,
        solid_angles, res,
    ) = _compute_etendue_prepare(
        det=det,
        aperture=aperture,
        res=res,
    )

    shape = distances.shape

    # --------------------
    # compute analytically

    if analytical is True:
        etend0 = np.full(tuple(np.r_[3, shape]), np.nan)

        # 0th order
        etend0[0, ...] = ap_surface * det_surface / distances**2

        # 1st order
        etend0[1, ...] = (
            cos_los_ap * ap_surface
            * cos_los_det * det_surface / distances**2
        )

        # 2nd order
        etend0[2, ...] = cos_los_ap * ap_surface * solid_angles

    else:
        etend0 = None

    # --------------------
    # compute numerically

    if numerical is True:
        etend1 = _compute_etendue_numerical(
            det=det,
            aperture=aperture,
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

    # --------------------
    # return

    dout = {
        'analytical': etend0,
        'numerical': etend1,
        'res': res,
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
        tok = (list, tuple, int, float, np.integer, np.float)
        if isinstance(det[k0], tok):
            det[k0] = np.atleast_1d(det[k0]).ravel()

        if not isinstance(det[k0], np.ndarray):
            msg = f"Arg det['{k0}'] must ba a np.ndarray"
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
        if det['nin_x'].shape == (1,):
            det['nin_x'] = np.full(shaped, det['nin_x'][0])
            det['nin_y'] = np.full(shaped, det['nin_y'][0])
            det['nin_z'] = np.full(shaped, det['nin_z'][0])
            det['e0_x'] = np.full(shaped, det['e0_x'][0])
            det['e0_y'] = np.full(shaped, det['e0_y'][0])
            det['e0_z'] = np.full(shaped, det['e0_z'][0])
            det['e1_x'] = np.full(shaped, det['e1_x'][0])
            det['e1_y'] = np.full(shaped, det['e1_y'][0])
            det['e1_z'] = np.full(shaped, det['e1_z'][0])
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
    # aperture 

    lk = [
        'poly_x', 'poly_y', 'poly_z',
        'nin_x', 'nin_y', 'nin_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
    ]

    c0 = (
        isinstance(aperture, dict)
        and all([kk in aperture.keys() for kk in lk])
    )
    if not c0:
        lstr = [f"\t- {k0}" for k0 in lk]
        msg = (
            "Arg aperture must be a dict with the following keys:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # check values
    for k0 in lk:
        tok = (list, tuple, int, float, np.integer, np.float)
        if isinstance(aperture[k0], tok):
            aperture[k0] = np.atleast_1d(aperture[k0]).ravel()

        if not isinstance(aperture[k0], np.ndarray):
            msg = f"Arg aperture['{k0}'] must ba a np.ndarray"
            raise Exception(msg)

        if aperture[k0].ndim > 1:
            msg = f"Arg aperture['{k0}'] must be 1d"
            raise Exception(msg)

        if 'poly_' not in k0 and aperture[k0].size > 1:
            msg = f"Arg aperture['{k0}'] must have size 1"
            raise Exception(msg)

    # check shapes
    dshape = {
        0: ['poly_x', 'poly_y', 'poly_z'],
        1: [
            'nin_x', 'nin_y', 'nin_z',
            'e0_x', 'e0_y', 'e0_z',
            'e1_x', 'e1_y', 'e1_z',
        ],
    }
    for k0, v0 in dshape.items():
        if len(set([aperture[v1].shape for v1 in v0])) > 1:
            lstr = [f"\t- {v1}" for v1 in v0]
            msg = (
                "The following args must share the same shape:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # check not closed poly
    if (
        aperture['poly_x'][0] == aperture['poly_x'][-1]
        and aperture['poly_y'][0] == aperture['poly_y'][-1]
        and aperture['poly_z'][0] == aperture['poly_z'][-1]
    ):
        aperture['poly_x'] = aperture['poly_x'][:-1]
        aperture['poly_y'] = aperture['poly_y'][:-1]
        aperture['poly_z'] = aperture['poly_z'][:-1]

    # normalization
    norm = np.sqrt(
        aperture['nin_x']**2
        + aperture['nin_y']**2
        + aperture['nin_z']**2
    )
    aperture['nin_x'] = aperture['nin_x'] / norm
    aperture['nin_y'] = aperture['nin_y'] / norm
    aperture['nin_z'] = aperture['nin_z'] / norm

    # derive cents
    aperture['cent_x'] = np.mean(aperture['poly_x'])
    aperture['cent_y'] = np.mean(aperture['poly_y'])
    aperture['cent_z'] = np.mean(aperture['poly_z'])

    # derive outline
    aperture['outline_x0'] = (
        (aperture['poly_x'] - aperture['cent_x'])*aperture['e0_x']
        + (aperture['poly_y'] - aperture['cent_y'])*aperture['e0_y']
        + (aperture['poly_z'] - aperture['cent_z'])*aperture['e0_z']
    )
    aperture['outline_x1'] = (
        (aperture['poly_x'] - aperture['cent_x'])*aperture['e1_x']
        + (aperture['poly_y'] - aperture['cent_y'])*aperture['e1_y']
        + (aperture['poly_z'] - aperture['cent_z'])*aperture['e1_z']
    )

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
        det, aperture, analytical, numerical,
        res, margin_par, margin_perp, check, verb, plot,
    )


# #############################################################################
# #############################################################################
#                   preparation routine
# #############################################################################


def _compute_etendue_prepare(
    det=None,
    aperture=None,
    res=None,
):

    # -----------------------
    # check outline is closed

    # det
    det_out_x0 = det['outline_x0']
    det_out_x1 = det['outline_x1']
    if (det_out_x0[0] != det_out_x0[-1]) or (det_out_x1[0] != det_out_x1[-1]):
        det_out_x0 = np.append(det_out_x0, det_out_x0[0])
        det_out_x1 = np.append(det_out_x1, det_out_x1[0])

    # ap
    ap_out_x0 = aperture['outline_x0']
    ap_out_x1 = aperture['outline_x1']
    if (ap_out_x0[0] != ap_out_x0[-1]) or (ap_out_x1[0] != ap_out_x1[-1]):
        ap_out_x0 = np.append(ap_out_x0, ap_out_x0[0])
        ap_out_x1 = np.append(ap_out_x1, ap_out_x1[0])


    # ----------------------------------
    # los, distances, cosines

    los_x = aperture['cent_x'] - det['cents_x']
    los_y = aperture['cent_y'] - det['cents_y']
    los_z = aperture['cent_z'] - det['cents_z']

    distances = np.sqrt(los_x**2 + los_y**2 + los_z**2)

    los_x = los_x / distances
    los_y = los_y / distances
    los_z = los_z / distances

    cos_los_det = (
        los_x * det['nin_x']
        + los_y * det['nin_y']
        + los_z * det['nin_z']
    )

    cos_los_ap = (
        los_x * aperture['nin_x']
        + los_y * aperture['nin_y']
        + los_z * aperture['nin_z']
    )

    # -----------
    # surfaces

    det_surface = 0.5*np.abs(
        np.sum(
            (det_out_x0[1:] + det_out_x0[:-1])
            * (det_out_x1[1:] - det_out_x1[:-1])
        )
        + (det_out_x0[0] + det_out_x0[-1])*(det_out_x1[0] - det_out_x1[-1])
    )

    ap_surface = 0.5*np.abs(
        np.sum(
            (ap_out_x0[1:] + ap_out_x0[:-1]) * (ap_out_x1[1:] - ap_out_x1[:-1])
        )
        + (ap_out_x0[0] + ap_out_x0[-1]) * (ap_out_x1[0] - ap_out_x1[-1])
    )

    # ------------
    # solid angles

    solid_angles = _comp_solidangles.calc_solidangle_apertures(
        # observation points
        pts_x=aperture['cent_x'],
        pts_y=aperture['cent_y'],
        pts_z=aperture['cent_z'],
        # polygons
        apertures=None,
        detectors=det,
        # possible obstacles
        config=None,
        # parameters
        visibility=False,
        return_vector=False,
    ).ravel()

    # -------------------------------------
    # det outline discretization resolution

    if res is None:
        res = min(
            np.sqrt(det_surface),
            np.sqrt(np.min(np.diff(ap_out_x0)**2 + np.diff(ap_out_x1)**2))
        ) * np.r_[2., 1., 0.5, 0.2]

    return (
        det_surface, ap_surface, distances,
        los_x, los_y, los_z,
        cos_los_det, cos_los_ap, solid_angles, res,
    )


# #############################################################################
# #############################################################################
#           Numerical etendue estimation routine
# #############################################################################


def _compute_etendue_numerical(
    det=None,
    aperture=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    los_x=None,
    los_y=None,
    los_z=None,
    check=None,
    verb=None,
):

    shape0 = det['cents_x'].shape
    cents_x = det['cents_x'].ravel()
    cents_y = det['cents_y'].ravel()
    cents_z = det['cents_z'].ravel()
    nin_x = det['nin_x'].ravel()
    nin_y = det['nin_y'].ravel()
    nin_z = det['nin_z'].ravel()
    e0_x = det['e0_x'].ravel()
    e0_y = det['e0_y'].ravel()
    e0_z = det['e0_z'].ravel()
    e1_x = det['e1_x'].ravel()
    e1_y = det['e1_y'].ravel()
    e1_z = det['e1_z'].ravel()
    nd = cents_x.size

    # ------------------------------
    # Get plane perpendicular to los

    etendue = np.full((res.size, cents_x.size), np.nan)
    for ii in range(nd):

        if verb is True:
            msg = f"Numerical etendue for det {ii+1} / {nd}"
            print(msg)

        # get individual det dict
        deti = dict(det)
        deti['cents_x'] = np.r_[cents_x[ii]]
        deti['cents_y'] = np.r_[cents_y[ii]]
        deti['cents_z'] = np.r_[cents_z[ii]]
        deti['nin_x'] = np.r_[nin_x[ii]]
        deti['nin_y'] = np.r_[nin_y[ii]]
        deti['nin_z'] = np.r_[nin_z[ii]]
        deti['e0_x'] = np.r_[e0_x[ii]]
        deti['e0_y'] = np.r_[e0_y[ii]]
        deti['e0_z'] = np.r_[e0_z[ii]]
        deti['e1_x'] = np.r_[e1_x[ii]]
        deti['e1_y'] = np.r_[e1_y[ii]]
        deti['e1_z'] = np.r_[e1_z[ii]]

        # get det corners to aperture corners vectors
        out_c_x0 = np.r_[0, deti['outline_x0']]
        out_c_x1 = np.r_[0, deti['outline_x1']]

        det_Px = cents_x[ii] + out_c_x0*e0_x[ii] + out_c_x1*e1_x[ii]
        det_Py = cents_y[ii] + out_c_x0*e0_y[ii] + out_c_x1*e1_y[ii]
        det_Pz = cents_z[ii] + out_c_x0*e0_z[ii] + out_c_x1*e1_z[ii]

        PA_x = aperture['poly_x'][:, None] - det_Px[None, :]
        PA_y = aperture['poly_y'][:, None] - det_Py[None, :]
        PA_z = aperture['poly_z'][:, None] - det_Pz[None, :]

        # get length along los
        k_los = (
            (1. + margin_par)
            * np.max(PA_x * los_x[ii] + PA_y * los_y[ii] + PA_z * los_z[ii])
        )

        # get center of plane perpendicular to los
        c_los_x = cents_x[ii] + k_los * los_x[ii]
        c_los_y = cents_y[ii] + k_los * los_y[ii]
        c_los_z = cents_z[ii] + k_los * los_z[ii]

        # get projections of corners on plane perp. to los
        sca0 = (
            (c_los_x - det_Px[None, :]) * los_x[ii]
            + (c_los_y - det_Py[None, :]) * los_y[ii]
            + (c_los_z - det_Pz[None, :]) * los_z[ii]
        )
        sca1 = PA_x * los_x[ii] + PA_y * los_y[ii] + PA_z * los_z[ii]

        k_plane = sca0 / sca1

        # get LOS-specific unit vectors

        e0_xi = los_y[ii] * e1_z[ii] - los_z[ii] * e1_y[ii]
        e0_yi = los_z[ii] * e1_x[ii] - los_x[ii] * e1_z[ii]
        e0_zi = los_x[ii] * e1_y[ii] - los_y[ii] * e1_x[ii]
        e0_normi = np.sqrt(e0_xi**2 + e0_yi**2 + e0_zi**2)
        e0_xi = e0_xi / e0_normi
        e0_yi = e0_yi / e0_normi
        e0_zi = e0_zi / e0_normi

        e1_xi = los_y[ii] * e0_zi - los_z[ii] * e0_yi
        e1_yi = los_z[ii] * e0_xi - los_x[ii] * e0_zi
        e1_zi = los_x[ii] * e0_yi - los_y[ii] * e0_xi

        # get projections on det_e0 and det_e1 in plane

        x0 = (
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e0_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e0_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e0_zi
        )
        x1 = (
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e1_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e1_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e1_zi
        )

        x0_min, x0_max = np.min(x0), np.max(x0)
        x1_min, x1_max = np.min(x1), np.max(x1)
        w0 = x0_max - x0_min
        w1 = x1_max - x1_min

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
                    detectors=deti,
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
                    np.allclose(sar[0, :], 0)
                    and np.allclose(sar[-1, :], 0)
                    and np.allclose(sar[:, 0], 0)
                    and np.allclose(sar[:, -1], 0)
                )
                if not c0:
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
                    plt.plot(
                        x0.ravel(), x1.ravel(), c='r', marker='o', ls='None',
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
                    import pdb; pdb.set_trace()
                    msg = "Something is wrong with solid_angle or sampling"
                    raise Exception(msg)
                else:
                    etendue[jj, ii] = np.sum(solid_angle) * ds

            else:
                etendue[jj, ii] = np.sum(
                    _comp_solidangles.calc_solidangle_apertures(
                        # observation points
                        pts_x=pts_x,
                        pts_y=pts_y,
                        pts_z=pts_z,
                        # polygons
                        apertures=aperture,
                        detectors=deti,
                        # possible obstacles
                        config=None,
                        # parameters
                        visibility=False,
                        return_vector=False,
                        return_flat_pts=True,
                        return_flat_det=True,
                    )
                ) * ds


    # --------------
    # reshape output

    if cents_x.shape != shape0:
        etendue = etendue.reshape(tuple(np.r_[res.size, shape0]))

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

    ax.set_ylabel('Etendue '+ r'($m^2.sr$)', size=12, fontweight='bold')
    ax.set_xlabel('order of approximation', size=12, fontweight='bold')

    ax.set_xticks(range(0, nmax))
    ax.set_xticklabels(x0)

    # -------------
    # plot

    lines = ax.plot(
        etend0,
        ls='-',
        marker='o',
        ms=6,
    )

    for ii in range(len(lines)):
        ax.plot(
            etend1[:, ii],
            ls='--',
            marker='*',
            ms=6,
            color=lines[ii].get_color(),
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
