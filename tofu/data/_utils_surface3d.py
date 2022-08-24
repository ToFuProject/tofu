# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds


from ..geom._comp_solidangles import _check_polygon_2d, _check_polygon_3d


# #################################################################
# #################################################################
#               Surface 3d
# #################################################################


def _surface3d(
    key=None,
    # 2d outline
    outline_x0=None,
    outline_x1=None,
    cent=None,
    # 3d outline
    poly_x=None,
    poly_y=None,
    poly_z=None,
    # extenthalf
    extenthalf=None,
    # normal vector at cent
    nin=None,
    e0=None,
    e1=None,
    # curvature
    curve_r=None,
    curve_npts=None,
):

    # -----------
    # cent

    if cent is not None:
        cent = np.atleast_1d(cent).ravel().astype(float)
        assert cent.shape == (3,)

    # -----------
    # unit vectors

    # nin has to be provided
    nin = ds._generic_check._check_flat1darray(
        var=nin, varname='nin', dtype=float, size=3, norm=True,
    )

    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
    )

    # -----------------------------
    # outline vs poly vs extenthalf

    lc = [
        all([pp is not None for pp in [outline_x0, outline_x1]])
        and e0 is not None and cent is not None,
        extenthalf is not None
        and curve_r is not None
        and e0 is not None and cent is not None,
        all([pp is not None for pp in [poly_x, poly_y, poly_z]])
    ]
    if np.sum(lc) != 1:
        msg = (
            "Please provide either (xor):\n"
            "\t- planar: outline_x0, outline_x1 and cent, e0, e1\n"
            "xor\n"
            "\t- curved: extenthalf and cent, e0, e1\n"
            "xor\n"
            "\t- arbitrary 3d: poly_x, poly_y, poly_z"
        )
        raise Exception(msg)

    # --------------
    # outline

    if lc[0]:

        # check outline
        outline_x0, outline_x1, area = _check_polygon_2d(
            poly_x=outline_x0,
            poly_y=outline_x1,
            poly_name=f'{key}-outline',
            can_be_None=False,
            closed=False,
            counter_clockwise=True,
            return_area=True,
        )

        # derive poly 3d
        poly_x = cent[0] + outline_x0 * e0[0] + outline_x1 * e1[0]
        poly_y = cent[1] + outline_x0 * e0[1] + outline_x1 * e1[1]
        poly_z = cent[2] + outline_x0 * e0[2] + outline_x1 * e1[2]

        gtype = 'planar'

    # ----------
    # curvature

    if lc[1]:

        curve_r = ds._generic_check._check_flat1darray(
            curve_r, 'curve_r', size=[1, 2], dtype=float,
        )
        extenthalf = ds._generic_check._check_flat1darray(
            extenthalf, 'extenthalf', size=[1, 2], dtype=float,
        )
        if extenthalf.size == 1:
            extenthalf = np.r_[extenthalf, extenthalf]

        if np.any(np.isnan(curve_r)):
            msg = (
                f"Arg curve_r for 3d surface '{key}' must:\n"
                "\t- be array-like of floats\n"
                "\t- be of size <= 2\n"
                f"Provided: {curve_r}"
            )
            raise Exception(msg)

        if curve_r.size == 1:
            curve_r = np.r_[curve_r, curve_r]

        ninf = np.isinf(curve_r).sum()
        if ninf == 1:
            gtype = 'cylindrical'
        elif ninf == 2:
            gtype = 'planar'
        elif np.unique(curve_r).size == 1:
            gtype = 'spherical'
        else:
            gtype = 'toroidal'

        curve_r = tuple(curve_r)

        # area
        area = _get_curved_area(
            gtype=gtype,
            curve_r=np.abs(curve_r),
            extenthalf=extenthalf,
        )

        # outline if planar
        if gtype == 'planar':
            outline_x0 = extenthalf[0]*np.r_[-1, 1, 1, -1]
            outline_x1 = extenthalf[1]*np.r_[-1, -1, 1, 1]

        # poly
        poly_x, poly_y, poly_z = _get_curved_poly(
            gtype=gtype,
            curve_r=curve_r,
            curve_npts=curve_npts,
            extenthalf=extenthalf,
            cent=cent,
            nin=nin,
            e0=e0,
            e1=e1,
        )

    # --------------------
    # poly 3d sanity check

    poly_x, poly_y, poly_z = _check_polygon_3d(
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        poly_name=f'{key}-polygon',
        can_be_None=False,
        closed=False,
        counter_clockwise=True,
        normal=nin,
    )

    # --------------------------------------------
    # try to get 2d outline from 3d poly if planar

    if lc[2]:

        gtype, outline_x0, outline_x1, area = _get_outline_from_poly(
            key=key,
            cent=cent,
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            nin=nin,
            e0=e0,
            e1=e1,
        )

    # ----------
    # return

    return (
        cent,
        outline_x0, outline_x1,
        poly_x, poly_y, poly_z,
        nin, e0, e1,
        extenthalf, area, curve_r, gtype,
    )


# #################################################################
# #################################################################
#               curved surfaces
# #################################################################


def _get_curved_area(
    gtype=None,
    curve_r=None,
    extenthalf=None,
):

    # planar
    if gtype == 'planar':
        area = 4. * extenthalf[0] * extenthalf[1]

    # cylindrical
    if gtype == 'cylindrical':
        iplan = np.isinf(curve_r).nonzero()[0][0]
        icurv = 1 - iplan
        rc = curve_r[icurv]

        area = 2. * extenthalf[iplan] * rc * 2. * extenthalf[icurv]

    # spherical
    elif gtype == 'spherical':
        rc = curve_r[0]
        ind = np.argmax(extenthalf)
        dphi = extenthalf[ind]
        sindtheta = np.sin(extenthalf[ind-1])

        area = 4. * rc**2 * dphi * sindtheta

    # toroidal
    elif gtype == 'toroidal':
        imax = np.argmax(curve_r)
        imin = 1 - imax
        rmax = curve_r[imax]
        rmin = curve_r[imin]
        phi2 = extenthalf[imax]
        theta2 = extenthalf[imin]

        area = rmin * 2.*phi2 * (rmax*2*theta2 + 2*rmin*np.sin(theta2))

    return area


def _get_curved_poly(
    gtype=None,
    curve_r=None,
    curve_npts=None,
    extenthalf=None,
    cent=None,
    nin=None,
    e0=None,
    e1=None,
):

    # ------------
    # check inputs

    curve_npts = ds._generic_check._check_var(
        curve_npts, 'curve_npts',
        types=int,
        default=5,
    )
    assert curve_npts >= 0, curve_npts

    # ------------
    # compute

    add = np.ones((curve_npts,))
    ang_add = np.linspace(-1, 1, curve_npts + 2)[1:-1]

    # planar
    if gtype == 'planar':

        b0 = extenthalf[0] * np.r_[-1, 1, 1, -1]
        b1 = extenthalf[1] * np.r_[-1, -1, 1, 1]

        poly_x = cent[0] + b0 * e0[0] + b1 * e1[0]
        poly_y = cent[1] + b0 * e0[1] + b1 * e1[1]
        poly_z = cent[2] + b0 * e0[2] + b1 * e1[2]

    # cylindrical
    if gtype == 'cylindrical':
        iplan = np.isinf(curve_r).nonzero()[0][0]
        icurv = 1 - iplan
        rc = curve_r[icurv]

        centbis = cent + rc * nin
        ee = [e0, e1]

        if iplan == 0:
            bplan = extenthalf[iplan] * np.r_[-1, 1, add, 1, -1, -add]
            ang = extenthalf[icurv] * np.r_[-1, -1, ang_add, 1, 1, -ang_add]

        else:
            bplan = extenthalf[iplan] * np.r_[-1, -add, -1, 1, add, 1]
            ang = extenthalf[icurv] * np.r_[-1, ang_add, 1, 1, -ang_add, -1]

        vx = np.cos(ang) * (-nin[0]) + np.sin(ang) * ee[icurv][0]
        vy = np.cos(ang) * (-nin[1]) + np.sin(ang) * ee[icurv][1]
        vz = np.cos(ang) * (-nin[2]) + np.sin(ang) * ee[icurv][2]

        poly_x = centbis[0] + bplan * ee[iplan][0] + rc * vx
        poly_y = centbis[1] + bplan * ee[iplan][1] + rc * vy
        poly_z = centbis[2] + bplan * ee[iplan][2] + rc * vz

    # spherical
    elif gtype == 'spherical':

        rc = curve_r[0]
        centbis = cent + rc * nin
        dtheta = (
            extenthalf[0]
            * np.r_[-1, ang_add, 1, add, 1, -ang_add, -1, -add]
        )
        psi = (
            extenthalf[1]
            * np.r_[-1, -add, -1, ang_add, 1, add, 1, -ang_add]
        )[None, :]

        vpsi = np.cos(psi) * (-nin)[:, None] + np.sin(psi) * e0[:, None]
        vx = np.cos(dtheta) * vpsi[0, :] + np.sin(dtheta) * e1[0]
        vy = np.cos(dtheta) * vpsi[1, :] + np.sin(dtheta) * e1[1]
        vz = np.cos(dtheta) * vpsi[2, :] + np.sin(dtheta) * e1[2]

        poly_x = centbis[0] + rc * vx
        poly_y = centbis[1] + rc * vy
        poly_z = centbis[2] + rc * vz

    # toroidal
    elif gtype == 'toroidal':
        imax = np.argmax(curve_r)
        imin = 1 - imax
        rmax = np.max(curve_r)
        rmin = np.min(curve_r)

        cmax = cent + nin * (rmax + rmin)

        if imax == 0:
            bmax = np.r_[-1, ang_add, 1, add, 1, -ang_add, -1, -add]
            bmin = np.r_[-1, -add, -1, ang_add, 1, add, 1, -ang_add]
            emax, emin = e0, e1
        else:
            bmin = np.r_[-1, ang_add, 1, add, 1, -ang_add, -1, -add]
            bmax = np.r_[-1, -add, -1, ang_add, 1, add, 1, -ang_add]
            emax, emin = e1, e0

        bmax *= extenthalf[imax]
        bmin *= extenthalf[imin]

        vmax = np.cos(bmax) * (-nin)[:, None] + np.sin(bmax) * emax[:, None]
        cmin = cmax[:, None] + rmax * vmax
        vmin = np.cos(bmin) * vmax + np.sin(bmin) * emin[:, None]

        poly_x = cmin[0, :] + rmin * vmin[0, :]
        poly_y = cmin[1, :] + rmin * vmin[1, :]
        poly_z = cmin[2, :] + rmin * vmin[2, :]

    return poly_x, poly_y, poly_z


# #################################################################
# #################################################################
#               outline from poly 3d
# #################################################################


def _get_outline_from_poly(
    key=None,
    cent=None,
    poly_x=None,
    poly_y=None,
    poly_z=None,
    nin=None,
    e0=None,
    e1=None,
):

    # ----------
    # cent

    if cent is None:
        cent = np.r_[np.mean(poly_x), np.mean(poly_y), np.mean(poly_z)]

    # ----------
    # planar

    diff_x = poly_x[1:] - poly_x[0]
    diff_y = poly_y[1:] - poly_y[0]
    diff_z = poly_z[1:] - poly_z[0]
    norm = np.sqrt(diff_x**2 + diff_y**2 + diff_x**2)
    diff_x = diff_x / norm
    diff_y = diff_y / norm
    diff_z = diff_z / norm

    sca = np.abs(nin[0]*diff_x + nin[1]*diff_y + nin[2]*diff_z)

    if np.all(sca < 2.e-12) and e0 is not None:
        # all deviation smaller than 1.e-10 degree
        gtype = 'planar'

        # derive outline
        outline_x0 = (
            (poly_x - cent[0]) * e0[0]
            + (poly_y - cent[1]) * e0[1]
            + (poly_z - cent[2]) * e0[2]
        )
        outline_x1 = (
            (poly_x - cent[0]) * e1[0]
            + (poly_y - cent[1]) * e1[1]
            + (poly_z - cent[2]) * e1[2]
        )

        # check outline
        outline_x0, outline_x1, area = _check_polygon_2d(
            poly_x=outline_x0,
            poly_y=outline_x1,
            poly_name=f'{key}-outline',
            can_be_None=False,
            closed=False,
            counter_clockwise=True,
            return_area=True,
        )

    else:
        gtype = '3d'
        outline_x0, outline_x1 = None, None
        area = np.nan

    return gtype, outline_x0, outline_x1, area
