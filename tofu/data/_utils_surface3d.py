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
    make_planar=None,
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
        all([pp is not None for pp in [outline_x0, outline_x1, cent, e0]]),
        all([pp is not None for pp in [extenthalf, curve_r, cent, e0]]),
        all([pp is not None for pp in [poly_x, poly_y, poly_z]])
    ]
    if np.sum(lc) != 1:
        kwd = locals()

        lk = [
            'outline_x0', 'outline_x1', 'cent', 'e0',
            'extenthalf', 'curve_r', 'cent', 'e0',
            'poly_x', 'poly_y', 'poly_z',
        ]
        lstr = [f"\t- {ss}: {kwd[ss]}" for ss in lk]
        msg = (
            "Please provide either (xor):\n"
            "\t- planar: outline_x0, outline_x1 and cent, e0, e1\n"
            "xor\n"
            "\t- curved: extenthalf and cent, e0, e1\n"
            "xor\n"
            "\t- arbitrary 3d: poly_x, poly_y, poly_z\nn"
            "Provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------------
    # outline

    if lc[0]:

        # check outline
        outline_x0, outline_x1, area = _check_polygon_2d(
            poly_x=outline_x0,
            poly_y=outline_x1,
            poly_name=f'{key}_outline',
            can_be_None=False,
            closed=False,
            counter_clockwise=True,
            return_area=True,
        )

        # if rectangle, try to derive extenthalf and r_curve
        if np.unique(outline_x0).size == 2:
            if np.unique(outline_x1).size == 2:
                dx0 = outline_x0.max() - outline_x0.min()
                dx1 = outline_x1.max() - outline_x1.min()
                extenthalf = [dx0*0.5, dx1*0.5]
                curve_r = [np.inf, np.inf]

        # derive poly 3d
        # poly_x = cent[0] + outline_x0 * e0[0] + outline_x1 * e1[0]
        # poly_y = cent[1] + outline_x0 * e0[1] + outline_x1 * e1[1]
        # poly_z = cent[2] + outline_x0 * e0[2] + outline_x1 * e1[2]

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

        # outline in local coordinates
        outline_x0 = extenthalf[0]*np.r_[-1, 1, 1, -1]
        outline_x1 = extenthalf[1]*np.r_[-1, -1, 1, 1]

    # --------------------------------------------
    # try to get 2d outline from 3d poly if planar

    if lc[2]:

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

        gtype, cent, outline_x0, outline_x1, area = _get_outline_from_poly(
            key=key,
            cent=cent,
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            nin=nin,
            e0=e0,
            e1=e1,
            make_planar=make_planar,
        )

        if outline_x0 is not None:
            poly_x, poly_y, poly_z = None, None, None

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
    outline_x0=None,
    outline_x1=None,
    cent=None,
    nin=None,
    e0=None,
    e1=None,
):

    # ------------
    # compute

    # planar
    if gtype == 'planar':

        poly_x = cent[0] + outline_x0 * e0[0] + outline_x1 * e1[0]
        poly_y = cent[1] + outline_x0 * e0[1] + outline_x1 * e1[1]
        poly_z = cent[2] + outline_x0 * e0[2] + outline_x1 * e1[2]

    # cylindrical
    if gtype == 'cylindrical':
        iplan = np.isinf(curve_r).nonzero()[0][0]
        icurv = 1 - iplan
        rc = curve_r[icurv]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        centbis = cent + rc * nin
        ee = [e0, e1]
        outline = [outline_x0, outline_x1]

        ang = outline[icurv]
        xx = outline[iplan]

        vx = np.cos(ang) * (-rcs*nin[0]) + np.sin(ang) * ee[icurv][0]
        vy = np.cos(ang) * (-rcs*nin[1]) + np.sin(ang) * ee[icurv][1]
        vz = np.cos(ang) * (-rcs*nin[2]) + np.sin(ang) * ee[icurv][2]

        poly_x = centbis[0] + xx * ee[iplan][0] + rca * vx
        poly_y = centbis[1] + xx * ee[iplan][1] + rca * vy
        poly_z = centbis[2] + xx * ee[iplan][2] + rca * vz

    # spherical
    elif gtype == 'spherical':

        rc = curve_r[0]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        centbis = cent + rc * nin
        psi, dtheta = outline_x0, outline_x1

        vpsix = np.cos(psi) * (-rcs*nin[0]) + np.sin(psi) * e0[0]
        vpsiy = np.cos(psi) * (-rcs*nin[1]) + np.sin(psi) * e0[1]
        vpsiz = np.cos(psi) * (-rcs*nin[2]) + np.sin(psi) * e0[2]

        vx = np.cos(dtheta) * vpsix + np.sin(dtheta) * e1[0]
        vy = np.cos(dtheta) * vpsiy + np.sin(dtheta) * e1[1]
        vz = np.cos(dtheta) * vpsiz + np.sin(dtheta) * e1[2]

        poly_x = centbis[0] + rca * vx
        poly_y = centbis[1] + rca * vy
        poly_z = centbis[2] + rca * vz

    # toroidal
    elif gtype == 'toroidal':

        imax = np.argmax(np.abs(curve_r))
        imin = 1 - imax
        rmax = curve_r[imax]
        rmin = curve_r[imin]
        rmaxs = np.sign(rmax)
        rmins = np.sign(rmin)
        rmaxa = np.abs(rmax)
        rmina = np.abs(rmin)
        crosss = rmaxs*rmins

        cmax = cent + nin * (rmax + rmin)
        outline = [outline_x0, outline_x1]
        amax = outline[imax]
        amin = outline[imin]

        ee = [e0, e1]

        vmaxx = np.cos(amax) * (-rmaxs*nin)[0] + np.sin(amax) * ee[imax][0]
        vmaxy = np.cos(amax) * (-rmaxs*nin)[1] + np.sin(amax) * ee[imax][1]
        vmaxz = np.cos(amax) * (-rmaxs*nin)[2] + np.sin(amax) * ee[imax][2]

        cminx = cmax[0] + rmaxa * vmaxx
        cminy = cmax[1] + rmaxa * vmaxy
        cminz = cmax[2] + rmaxa * vmaxz

        vminx = np.cos(amin) * crosss * vmaxx + np.sin(amin) * ee[imin][0]
        vminy = np.cos(amin) * crosss * vmaxy + np.sin(amin) * ee[imin][1]
        vminz = np.cos(amin) * crosss * vmaxz + np.sin(amin) * ee[imin][2]

        poly_x = cminx + rmina * vminx
        poly_y = cminy + rmina * vminy
        poly_z = cminz + rmina * vminz

    return poly_x, poly_y, poly_z


# ###############################################################
# ###############################################################
#               outline from poly 3d
# ###############################################################


def _get_outline_from_poly(
    key=None,
    cent=None,
    poly_x=None,
    poly_y=None,
    poly_z=None,
    nin=None,
    e0=None,
    e1=None,
    make_planar=None,
):

    # --------
    # check input

    make_planar = ds._generic_check._check_var(
        make_planar, 'make_planar',
        types=bool,
        default=True,
    )

    if make_planar is True and e0 is None:
        msg = "Provide e0 to make planar!"
        raise Exception(msg)

    # ----------
    # cent

    if cent is None:
        cent = np.r_[np.mean(poly_x), np.mean(poly_y), np.mean(poly_z)]

    # ----------
    # planar

    if make_planar:
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

    return gtype, cent, outline_x0, outline_x1, area
