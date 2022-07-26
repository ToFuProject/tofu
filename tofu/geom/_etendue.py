

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
    res=None,
    plot=None,
):
    """ Only works for a set of detectors associated to a single aperture

    Typical use:
        - pinhole cameras

    """

    # -------------
    # check inputs

    det, aperture, plot = _compute_etendue_check(
        det=det,
        aperture=aperture,
        analytical=analytical,
        numerical=numerical,
        res=res,
        plot=plot,
    )

    # ----------
    # prepare

    (
        det_surface, ap_surface,
        distances, cos_det, cos_ap,
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
        etend1 = np.full(tuple(np.r_[3, shape]), np.nan)

        # # poor resolution
        # etend1[0, ...] = _compute_etendue_numerical(
            # det=det,
            # aperture=aperture,
            # res=res[0],
        # )

        # # medium resolution
        # etend1[1, ...] = _compute_etendue_numerical(
            # det=det,
            # aperture=aperture,
            # res=res[1],
        # )

        # # good resolution
        # etend1[2, ...] = _compute_etendue_numerical(
            # det=det,
            # aperture=aperture,
            # res=res[2],
        # )

    else:
        etend1 = None

    # --------------------
    # optional plotting

    if plot is True:
        dax = _plot_etendues(
            etend0=etend0,
            etend1=etend1,
        )

    return etend0, etend1


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
        'analytical', analytical,
        types=bool,
        default=True,
    )

    # -----------
    # numerical

    numerical = ds._generic_check._check_var(
        'numerical', nmumerical,
        types=bool,
        default=True,
    )

    # -----------
    # res

    if res is not None:
        res = np.atleast_1d(res).ravel()

    # -----------
    # plot

    if plot is None:
        plot = True
    if not isinstance(plot, bool):
        msg = "Arg plot must be a bool"
        raise Exception(msg)

    return det, aperture, analytical, numerical, res, plot


# #############################################################################
# #############################################################################
#                   preparation routine
# #############################################################################


def _compute_etendue_prepare(
    det=None,
    aperture=None,
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
        los_x * ap['nin_x']
        + los_y * ap['nin_y']
        + los_z * ap['nin_z']
    )

    # -----------
    # surfaces

    det_surface = 0.5*np.sum(
        (det_out_x0[1:] + det_out_x0[:-1]) * (det_out_x1[1:] - det_out_x1[:-1])
    )
    ap_surface = 0.5*np.sum(
        (ap_out_x0[1:] + ap_out_x0[:-1]) * (ap_out_x1[1:] - ap_out_x1[:-1])
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
        detectors_normal=None,
        # possible obstacles
        config=None,
        # parameters
        visibility=False,
        return_vector=False,
    )

    import pdb; pdb.set_trace()     # DB

    # -------------------------------------
    # det outline discretization resolution

    if res is None:
        res = min(
            np.sqrt(det_surface),
            np.sqrt(np.min(np.diff(ap_out_x0)**2 + np.diff(ap_out_x1)**2))
        ) * np.r_[0.1, 0.05, 0.02, 0.01]

    return (
        det_surface, ap_surface, distances,
        cos_det, cos_ap, solid_angles, res,
    )


# #############################################################################
# #############################################################################
#           Numerical etendue estimation routine
# #############################################################################


def _compute_etendue_numerical(
    det=None,
    aperture=None,
    res=None,
):

    # -------------------
    # Discretize aperture

    cx, cy, cz = [aperture[k0] for k0 in ['cent_x', 'cent_y', 'cent_z']]
    out0, out1 = aperture['outline_x0'], aperture['outline_x1']

    min0, max0 = np.min(out0), np.max(out0)
    min1, max1 = np.min(out1), np.max(out1)

    n0 = int(np.ceil((max0 - min0) / res))
    n1 = int(np.ceil((max1 - min1) / res))

    d0 = (max0 - min0) / n0
    d1 = (max1 - min1) / n1

    ds = d0 * d1

    pts_0 = np.linspace(np.min(out0), np.max(out0), n0 + 1)
    pts_1 = np.linspace(np.min(out1), np.max(out1), n1 + 1)
    pts_0 = 0.5 * (pts_0[1:] + pts_0[:-1])
    pts_1 = 0.5 * (pts_1[1:] + pts_1[:-1])

    pts_x = (
        cx
        + pts_0[:, None] * aperture['ei_x']
        + pts_1[None, :] * aperture['ej_x']
    ).ravel()
    pts_y = (
        cy
        + pts_0[:, None] * aperture['ei_y']
        + pts_1[None, :] * aperture['ej_y']
    ).ravel()
    pts_z = (
        cz
        + pts_0[:, None] * aperture['ei_z']
        + pts_1[None, :] * aperture['ej_z']
    ).ravel()

    # ----------------------------------
    # compute solid angle for each pixel

    etendue = np.full(det['cents_x'].size, np.nan)
    for ii in range(det['cents_x'].size):
        solid_angle = np.nan
        etendue[ii] = np.sum(solid_angle) * ds

    # --------------
    # reshape output

    # if shape0 is not False:
        # etendue = etendue.reshape(shape0)

    return etendue


# #############################################################################
# #############################################################################
#                   Plotting routine
# #############################################################################


def _plot_etendues(
    etend0=None,
    etend1=None,
):

    # -------------
    # prepare data

    if etend0.ndim > 2:
        etend0 = etend0.reshape((etend0.shape[0], -1))
        etend1 = etend1.reshape((etend0.shape[0], -1))

    # -------------
    # prepare axes

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_ylabel('Etendue '+ r'($m^2.sr$)', size=12, fontweight='bold')
    ax.set_xlabel('order of approximation', size=12, fontweight='bold')

    ax.set_xticks(range(1, etend0.shape[0] + 1))
    ax.set_xticklabels(['0th', '1st', '2nd'])

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
