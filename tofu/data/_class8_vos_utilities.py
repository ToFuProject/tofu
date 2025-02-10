# -*- coding: utf-8 -*-


import warnings


import numpy as np
import bsplines2d as bs2
from contourpy import contour_generator
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import datastock as ds
import Polygon as plg


# ###############################################
# ###############################################
#           margin polygons
# ###############################################


def _get_poly_margin(
    # polygon
    p0=None,
    p1=None,
    # margin
    margin=None,
):

    # ----------
    # check

    margin = float(ds._generic_check._check_var(
        margin, 'margin',
        types=(float, int),
        default=0.3,
        sign='>0'
    ))

    # ---------------------------
    # add extra margin to pcross

    # get centroid
    cent = plg.Polygon(np.array([p0, p1]).T).center()

    # add margin
    return (
        cent[0] + (1. + margin) * (p0 - cent[0]),
        cent[1] + (1. + margin) * (p1 - cent[1]),
    )


# ###############################################
# ###############################################
#           overall polygons
# ###############################################


def _get_overall_polygons(
    coll=None,
    doptics=None,
    key_cam=None,
    poly=None,
    convexHull=None,
):

    # get temporary vos
    kp0, kp1 = doptics[key_cam]['dvos'][poly]
    shape = coll.ddata[kp0]['data'].shape
    p0 = coll.ddata[kp0]['data'].reshape((shape[0], -1))
    p1 = coll.ddata[kp1]['data'].reshape((shape[0], -1))

    # pix indices
    iok = np.isfinite(p0)

    # -----------------------
    # envelop pcross and phor

    if convexHull is True:
        # replace by convex hull
        pts = np.array([p0[iok], p1[iok]]).T
        return pts[ConvexHull(pts).vertices, :].T

    else:

        ipn = (np.all(iok, axis=0)).nonzero()[0]
        pp = plg.Polygon(np.array([p0[:, ipn[0]], p1[:, ipn[0]]]).T)
        for ii in ipn[1:]:
            pp |= plg.Polygon(np.array([p0[:, ii], p1[:, ii]]).T)

        if len(pp) > 1:

            # replace by convex hull
            pts = np.concatenate(
                tuple([np.array(pp.contour(ii)) for ii in range(len(pp))]),
                axis=0,
            )
            poly = pts[ConvexHull(pts).vertices, :].T

            # plot for debugging
            fig = plt.figure()
            fig.suptitle("_get_overall_polygons()", size=12)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_title(f"camera '{key_cam}', vos poly '{poly}'")
            for ii in range(len(pp)):
                ax.plot(
                    np.array(pp.contour(ii))[:, 0],
                    np.array(pp.contour(ii))[:, 1],
                    '.-',
                    poly[0, :],
                    poly[1, :],
                    '.-k',
                )
            msg = "multiple contours"
            warnings.warn(msg)
            return poly

        else:
            return np.array(pp.contour(0)).T


# ###############################################
# ###############################################
#               get polygons
# ###############################################


def _get_polygons(
    x0=None,
    x1=None,
    bool_cross=None,
    res=None,
):
    """ Get simplified contour polygon

    First computes contours
    Then simplifies it using a mix of convexHull and concave picking edges
    """

    # ------------
    # get contour

    contgen = contour_generator(
        x=x0,
        y=x1,
        z=bool_cross,
        name='serial',
        corner_mask=None,
        line_type='Separate',
        fill_type=None,
        chunk_size=None,
        chunk_count=None,
        total_chunk_count=None,
        quad_as_tri=True,       # for sub-mesh precision
        # z_interp=<ZInterp.Linear: 1>,
        thread_count=0,
    )

    no_cont, cj = bs2._class02_contours._get_contours_lvls(
        contgen=contgen,
        level=0.5,
        largest=True,
    )

    assert no_cont is False

    # -------------
    # simplify poly

    return _simplify_polygon(cj[:, 0], cj[:, 1], res=res)


def _simplify_polygon(c0, c1, res=None):

    # -----------
    # convex hull

    npts = c0.size

    # get hull
    convh = ConvexHull(np.array([c0, c1]).T)
    indh = convh.vertices
    ch0 = c0[indh]
    ch1 = c1[indh]
    nh = indh.size

    sign = np.median(np.diff(indh))

    # segments norms
    seg0 = np.r_[ch0[1:] - ch0[:-1], ch0[0] - ch0[-1]]
    seg1 = np.r_[ch1[1:] - ch1[:-1], ch1[0] - ch1[-1]]
    norms = np.sqrt(seg0**2 + seg1**2)

    # keep egdes that match res
    lind = []
    for ii, ih in enumerate(indh):

        # ind of points in between
        i1 = indh[(ii + 1) % nh]
        if sign > 0:
            if i1 > ih:
                ind = np.arange(ih, i1 + 1)
            else:
                ind = np.r_[np.arange(ih, npts), np.arange(0, i1 + 1)]
        else:
            if i1 < ih:
                ind = np.arange(ih, i1 - 1, -1)
            else:
                ind = np.r_[np.arange(ih, -1, -1), np.arange(npts - 1, i1 - 1, -1)]

        # trivial
        if ind.size == 2:
            lind.append((ih, i1))
            continue

        # get distances
        x0 = c0[ind]
        x1 = c1[ind]

        # segment unit vect
        vect0 = x0 - ch0[ii]
        vect1 = x1 - ch1[ii]

        # perpendicular distance
        cross = (vect0*seg1[ii] - vect1*seg0[ii]) / norms[ii]

        # criterion
        if np.all(np.abs(cross) <= 0.8*res):
            lind.append((ih, i1))
        else:
            lind += _simplify_concave(
                x0=x0,
                x1=x1,
                ind=ind,
                cross=cross,
                res=res,
            )

    # ------------------------------------
    # point by point on remaining segments

    iok = np.unique(np.concatenate(tuple(lind)))

    return c0[iok], c1[iok]


def _simplify_concave(
    x0=None,
    x1=None,
    ind=None,
    cross=None,
    res=None,
):

    # ------------
    # safety check

    imax = np.argmax(np.abs(cross))
    sign0 = np.sign(cross[imax])
    # sign0 = np.mean(sign)
    iok0 = cross * sign0 >= -1e-12
    if not np.all(iok0):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(
            x0, x1, '.-k',
        )
        msg = (
            "Non-conform cross * sign0 (>= -1e-12):\n"
            f"\t- x0 = {x0}\n"
            f"\t- x1 = {x1}\n"
            f"\t- ind = {ind}\n"
            f"\t- iok0 = {iok0}\n"
            # f"\t- sign = {sign}\n"
            f"\t- sign0 = {sign0}\n"
            f"\t- cross * sign0 = {cross * sign0}\n"
        )
        raise Exception(msg)

    # ------------
    # loop

    i0 = 0
    i1 = 1
    iok = 1
    lind_loc, lind = [], []
    while iok <= ind.size - 1:

        # reference normalized vector
        vref0, vref1 = x0[i1] - x0[i0], x1[i1] - x1[i0]
        normref = np.sqrt(vref0**2 + vref1**2)
        vref0, vref1 = vref0 / normref, vref1 / normref

        # intermediate vectors
        indi = np.arange(i0 + 1, i1)
        v0 = x0[indi] - x0[i0]
        v1 = x1[indi] - x1[i0]

        # sign and distance (from cross product)
        cross = v0 * vref1 - v1 * vref0
        dist = np.abs(cross)

        # conditions
        c0 = np.all(dist <= 0.8*res)
        c1 = np.all(cross * sign0 >= -1e-12)
        c2 = i1 == ind.size - 1

        append = False
        # cases
        if c0 and c1 and (not c2):
            iok = int(i1)
            i1 += 1
        elif c0 and c1 and c2:
            iok = int(i1)
            append = True
        elif c0 and (not c1) and (not c2):
            i1 += 1
        elif c0 and (not c1) and c2:
            append = True
        elif not c0:
            append = True

        # append
        if append is True:
            lind_loc.append((i0, iok))
            lind.append((ind[i0], ind[iok]))
            i0 = iok
            i1 = i0 + 1
            iok = int(i1)

        if i1 > ind.size - 1:
            break

    return lind


# ################################################################
# ################################################################
#               Get dphi from R and phor
# ################################################################


def _get_dphi_from_R_phor(
    R=None,
    phor0=None,
    phor1=None,
    phimin=None,
    phimax=None,
    res=None,
    out=None
):

    # ------------
    # check inputs

    # out
    out = ds._generic_check._check_var(
        out, 'out',
        types=bool,
        default=True,
    )
    sign = 1. if out is True else -1.

    # R
    R = np.unique(np.atleast_1d(R).ravel())

    # path
    path = Path(np.array([phor0, phor1]).T)

    # --------------
    # sample phi

    dphi = np.full((2, R.size), np.nan)
    for ir, rr in enumerate(R):

        nphi = np.ceil(rr*np.abs(phimax - phimin) / (0.05*res)).astype(int) + 1
        phi = np.linspace(phimin, phimax, nphi)

        ind = path.contains_points(
            np.array([rr*np.cos(phi), rr*np.sin(phi)]).T
        )

        if np.any(ind):
            dphi[0, ir] = np.min(phi[ind]) - sign*(phi[1] - phi[0])
            dphi[1, ir] = np.max(phi[ind]) + sign*(phi[1] - phi[0])

    return dphi