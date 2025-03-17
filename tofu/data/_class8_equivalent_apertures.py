# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


import Polygon as plg
import datastock as ds


from . import _class5_projections
from . import _class8_compute as _compute


# ##############################################################
# ##############################################################
#           Equivalent aperture
# ##############################################################


def equivalent_apertures(
    # resources
    coll=None,
    key=None,
    key_cam=None,
    pixel=None,
    # inital contour
    add_points=None,
    min_threshold=None,
    # options
    ind_ap_lim_spectral=None,
    convex=None,
    harmonize=None,     # Deprecated ?
    reshape=None,
    return_for_etendue=None,
    # plot
    verb=None,
    plot=None,
    store=None,
    # debug
    debug=None,
):

    # -------------
    # check inputs
    # -------------

    (
        key,
        key_cam,
        iref,
        pinhole,
        paths,
        spectro,
        ispectro,
        curved,
        concave,
        curve_mult,
        optics,
        optics_cls,
        is2d,
        # shape0,
        cx,
        cy,
        cz,
        pixel,
        add_points,
        min_threshold,
        ind_ap_lim_spectral,
        convex,
        harmonize,
        reshape,
        verb,
        plot,
        store,
        debug,
    ) = _check(**locals())

    if pixel is None:
        pixel = np.unravel_index(np.arange(0, cx.size), cx.shape)

    if return_for_etendue is None:
        return_for_etendue = False

    # ---------------
    # Prepare optics
    # ---------------

    # polygons of optics before the spectral element
    lpoly = [
        coll.get_optics_poly(
            key=oo,
            mode='thr',
            add_points=add_points,
            min_threshold=min_threshold,
            return_outline=False,
        )
        for oo in optics
    ]

    # polygons of optics after the spectral element
    lx01 = [
        coll.get_optics_outline(
            key=oo,
            mode='thr',
            add_points=add_points,
            min_threshold=min_threshold,
        )
        for oo in optics
    ]

    if pinhole is True:
        lpoly_pre = [pp for ii, pp in enumerate(lpoly) if ii < iref]
        lx01_post = [pp for ii, pp in enumerate(lx01) if ii > iref]

    # -------------------
    # prepare functions
    # -------------------

    # ---------------------
    # coordinates functions

    # coordinate func
    lcoord_x01toxyz_poly = [coll.get_optics_x01toxyz(key=oo) for oo in optics]
    if pinhole is True:
        coord_x01toxyz = coll.get_optics_x01toxyz(key=optics[iref])
        lcoord_x01toxyz_poly = [
            coll.get_optics_x01toxyz(key=optics[ii])
            for ii in range(iref+1, len(optics))
        ]

    # ------------
    # pts2pts func

    pts2pt = None
    dist_cryst2ap = 0.
    if spectro is True:
        if pinhole is True:
            pts2pt = coll.get_optics_reflect_pts2pt(key=optics[iref])
            if iref < len(optics) - 1:
                cls_ap_lim = optics_cls[ind_ap_lim_spectral]
                kap_lim = optics[ind_ap_lim_spectral]
                dist_cryst2ap = np.linalg.norm(
                    coll.dobj[cls_ap_lim][kap_lim]['dgeom']['cent']
                    - coll.dobj[optics_cls[iref]][optics[iref]]['dgeom']['cent']
                )

    # --------------
    # ptsvect func

    lptsvect = [coll.get_optics_reflect_ptsvect(key=oo) for oo in optics]
    if pinhole is True:
        ptsvect = coll.get_optics_reflect_ptsvect(key=optics[iref])
        lptsvect_post = [pp for ii, pp in enumerate(lptsvect) if ii > iref]

    # ------------------------
    # equivalent aperture func

    if spectro:
        func = _get_equivalent_aperture_spectro
    else:
        func = _get_equivalent_aperture

    # -------------------
    # prepare output
    # -------------------

    x0 = []
    x1 = []

    # ---------------
    # loop on pixels
    # ---------------

    # dt = np.zeros((14,), dtype=float)

    # -----------------------------------------
    # add pts to initial polygon only if curved

    if spectro:
        assert pinhole is True, "How to handle spectr with pinhole = False here ?"
        rcurv = np.r_[coll.dobj[optics_cls[iref]][optics[iref]]['dgeom']['curve_r']]
        ind = ~np.isinf(rcurv)
        if np.any(rcurv[ind] > 0):
            addp0 = add_points
        else:
            addp0 = False
    else:
        addp0 = False

    # ---------------
    # start loop

    npix = pixel[0].size
    shape0 = cx.shape
    iok = np.ones(shape0, dtype=bool)
    for ii, ij in enumerate(zip(*pixel)):

        # ----- DEBUG -------
        # if ij not in [1148]:
        #     iok[ii] = False
        #     x0.append(None)
        #     x1.append(None)
        #     continue
        # -------------------

        if verb is True:
            msg = f"\t- camera '{key_cam}': pixel {ii + 1} / {npix}"
            end = '\n' if ii == npix - 1 else '\r'
            print(msg, end=end, flush=True)

        # -----------------------
        # adjust optics if needed

        if pinhole is False:

            sli = tuple(list(ij) + [slice(None)])
            iop = np.nonzero(paths[sli])[0]

            lpoly_pre = [lpoly[jj] for jj in iop if jj < iref[ij]]
            lx01_post = [lx01[jj] for jj in iop if jj > iref[ij]]
            lptsvect_post = [lptsvect[jj] for jj in iop if jj > iref[ij]]

            # ref-specific
            coord_x01toxyz = coll.get_optics_x01toxyz(key=optics[iref[ij]])
            ptsvect = coll.get_optics_reflect_ptsvect(key=optics[iref[ij]])

            # initial polygon
            p_a = coll.get_optics_outline(
                key=optics[iref[ij]],
                add_points=addp0,
            )
            p_a = plg.Polygon(np.array([p_a[0], p_a[1]]).T)

            if spectro is True:
                pts2pt = coll.get_optics_reflect_pts2pt(key=optics[iref[ij]])

        else:
            # initial polygon
            if ii == 0:
                p_a = coll.get_optics_outline(
                    key=optics[iref],
                    add_points=addp0,
                )
                p_a = plg.Polygon(np.array([p_a[0], p_a[1]]).T)

        # -------
        # compute

        p0, p1 = func(
            p_a=p_a,
            pt=np.r_[cx[ij], cy[ij], cz[ij]],
            lpoly_pre=lpoly_pre,
            lx01_post=lx01_post,
            # functions
            coord_x01toxyz=coord_x01toxyz,
            lcoord_x01toxyz_poly=lcoord_x01toxyz_poly,
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            lptsvect_poly=lptsvect_post,
            # options
            add_points=add_points,
            convex=convex,
            # debug
            ii=ii,
            ij=ij,
            debug=debug,
            # timing
            # dt=dt,
        )

        # convex hull
        if p0 is None or p0.size == 0:
            iok[ij] = False

        elif convex:
            pass
            # p0, p1 = np.array(plgUtils.convexHull(
                # plg.Polygon(np.array([p0, p1]).T)
            # ).contour(0)).T
            # vert = ConvexHull(np.array([p0, p1]).T).vertices

            # --- DEBUG ---------
            # if ii in [97]:
                # _debug_plot(
                    # pa0=p0, pa1=p1,
                    # pb0=p0[vert], pb1=p1[vert],
                    # ii=ii, tit='local coords',
                # )
            # --------------------

            # p0c, p1c = _compute._interp_poly(
            #     lp=[
            #         p0[vert] * curve_mult[0],
            #         p1[vert] * curve_mult[1],
            #     ],
            #     add_points=1,
            #     mode='thr',
            #     isclosed=False,
            #     closed=False,
            #     ravel=True,
            #     min_threshold=min_threshold,
            #     debug=True,
            # )

            # --- DEBUG ---------
            # if ij in [104]:
            #     _debug_plot(
            #         pa0=p0, pa1=p1,
            #         pb0=p0[vert], pb1=p1[vert],
            #         pc0=p0c/curve_mult[0], pc1=p1c/curve_mult[1],
            #         ii=ii, tit='curve_mult',
            #     )
            # --------------------
            # p0, p1 = p0c / curve_mult[0], p1c / curve_mult[1]

        # append
        x0.append(p0)
        x1.append(p1)

        # --- DEBUG ---------
        # if ii in [7]:
        #     _debug_plot(pa0=p0, pa1=p1, ii=ii, tit='local coords')
        # --------------------

    # -------------------------------------------
    # harmonize if necessary the initial polygons
    # -------------------------------------------

    if harmonize:
        x0, x1 = _compute._harmonize_polygon_sizes(
            lp0=x0,
            lp1=x1,
            nmin=150 if curved else 0,
            shape=shape0,
        )

    # -------------
    # xyz
    # -------------

    if return_for_etendue is True:

        # --------------
        # get pts2plane

        if pinhole is True:
            pts2plane = coll.get_optics_reflect_ptsvect(
                key=optics[iref],
                asplane=True,
            )

        # ---------------
        # initialize

        px = np.full(x0.shape, np.nan)
        py = np.full(x0.shape, np.nan)
        pz = np.full(x0.shape, np.nan)

        cents0 = np.full(shape0, np.nan)
        cents1 = np.full(shape0, np.nan)

        if pinhole is False:
            centsx = np.full(shape0, np.nan)
            centsy = np.full(shape0, np.nan)
            centsz = np.full(shape0, np.nan)
            plane_nin = np.full(tuple(np.r_[3, shape0]), np.nan)

        area = np.full(shape0, np.nan)

        sli0 = np.array(list(shape0) + [slice(None)])
        for ii, ij in enumerate(zip(*pixel)):

            if not iok[ij]:
                continue

            # update slice
            sli0[:-1] = ij
            sli = tuple(sli0)

            # get coordinates in x, y, z
            if pinhole is False:
                # ref optics cls and key
                roc = optics_cls[iref[ij]]
                rok = optics[iref[ij]]

                coord_x01toxyz = coll.get_optics_x01toxyz(key=rok)
                pts2plane = coll.get_optics_reflect_ptsvect(
                    key=rok,
                    asplane=True,
                )

            # pts x, y, z
            pxi, pyi, pzi = coord_x01toxyz(x0=x0[sli], x1=x1[sli])

            # derive pts on plane
            (
                px[sli], py[sli], pz[sli],
                _, _, _, _, _, p0, p1,
            ) = pts2plane(
                pts_x=cx[ij],
                pts_y=cy[ij],
                pts_z=cz[ij],
                vect_x=pxi - cx[ij],
                vect_y=pyi - cy[ij],
                vect_z=pzi - cz[ij],
                strict=False,
                return_x01=True,
            )

            # area
            area[ij] = plg.Polygon(np.array([p0, p1]).T).area()

            # centroid in 3d
            polyi = plg.Polygon(np.array([x0[sli], x1[sli]]).T)
            cent = polyi.center()

            if not polyi.isInside(*cent):
                cent = _get_centroid(
                    x0[sli],
                    x1[sli],
                    cent,
                    debug=False,
                )

            cents0[ij], cents1[ij] = cent

            # x, y, z coordinates
            if pinhole is False:
                centsx[ij], centsy[ij], centsz[ij] = coord_x01toxyz(
                    x0=cents0[ij],
                    x1=cents1[ij],
                )
                sliF = tuple([slice(None)] + list(ij))
                plane_nin[sliF] = coll.dobj[roc][rok]['dgeom']['nin']

            # --- DEBUG ---------
            # if ii in [0]:
                # _debug_plot2(
                    # p0=p0, p1=p1,
                    # cents0=cents0, cents1=cents1,
                    # ii=ii,
                    # ip=0,
                    # coord_x01toxyz=coord_x01toxyz,
                    # cx=cx, cy=cy, cz=cz,
                    # area=area,
                # )
            # --------------------

        # -------------------
        # x, y, z coordinates

        if pinhole is True:
            centsx, centsy, centsz = coord_x01toxyz(
                x0=cents0,
                x1=cents1,
            )

            # add centroid to x0, x1
            roc_all = optics_cls[iref]
            rok_all = optics[iref]
            plane_nin = coll.dobj[roc_all][rok_all]['dgeom']['nin']

        # ------ DEBUG --------
        # if True:
            # plt.figure()
            # plt.plot(pixel, area, '.-k')
            # plt.gca().set_title('area')
        # --------------------

    else:
        cents0, cents1 = None, None

    # --------------------
    # reshape if necessary
    # --------------------

    shape0 = cx.shape
    ntot = np.prod(shape0)
    if is2d and harmonize and reshape and x0.shape[0] == ntot:
        shape = tuple(np.r_[shape0, x0.shape[-1]])
        x0 = x0.reshape(shape)
        x1 = x1.reshape(shape)

    # -------------
    # plot & return
    # -------------

    if plot is True:
        if pinhole is True:
            out0, out1 = coll.get_optics_outline(
                key=optics[iref],
                add_points=False,
            )
        else:
            raise NotImplementedError()
        _plot(
            poly_x0=out0,
            poly_x1=out1,
            p0=x0,
            p1=x1,
            cents0=cents0,
            cents1=cents1,
            # options
            tit=f"Reflections on {key}",
            xlab=None,
            ylab=None,
        )

    # timing
    # lt = [f"\t- dt{ii}: {dt[ii]}" for ii in range(len(dt))]
    # print(f"\nTiming for {key}:\n" + "\n".join(lt))

    if return_for_etendue:
        return (
            pinhole, optics, iref,
            x0, x1, iok,
            px, py, pz,
            cx, cy, cz,
            cents0, cents1,
            centsx, centsy, centsz,
            area, plane_nin,
            spectro, dist_cryst2ap
        )
    else:
        return x0, x1, optics[iref], iok


def _get_centroid(p0, p1, cent, debug=None):

    # ------------
    # compute

    # get unit vectors of lines going through centroid
    mid0 = np.r_[0.5*(p0[1:] + p0[:-1]), 0.5*(p0[-1] + p0[0]), p0]
    mid1 = np.r_[0.5*(p1[1:] + p1[:-1]), 0.5*(p1[-1] + p1[0]), p1]
    u0 = mid0 - cent[0]
    u1 = mid1 - cent[1]
    un = np.sqrt(u0**2 + u1**2)
    u0n = u0/un
    u1n = u1/un

    # get minimum of total algebraic distance from line
    dist = np.sum(
        u0n[:, None] * p1[None, :] - u1n[:, None] * p0[None, :],
        axis=1,
    )
    imin = np.argmin(np.abs(dist))

    # build unit vector
    vect = np.r_[u0n[imin], u1n[imin]]

    # get intersections
    AB0 = np.r_[p0[1:] - p0[:-1], p0[0] - p0[-1]]
    AB1 = np.r_[p1[1:] - p1[:-1], p1[0] - p1[-1]]
    detABu = AB0 * vect[1] - AB1 * vect[0]
    detACu = (cent[0] - p0) * vect[1] - (cent[1] - p1) * vect[0]
    kk = detACu / detABu
    ind = ((kk >= 0) & (kk < 1)).nonzero()[0]

    # pathological cases
    if ind.size != 2:
        # return the point is question
        cent2 = np.r_[mid0[imin], mid1[imin]]
    else:
        # normal case
        cent2 = np.r_[
            np.mean(p0[ind] + kk[ind] * AB0[ind]),
            np.mean(p1[ind] + kk[ind] * AB1[ind]),
        ]

    # --------------
    # debug

    if debug:

        indclose = np.r_[np.arange(p0.size), 0]
        plt.figure()
        plt.plot(dist, '.-')
        plt.gca().axvline(imin, c='k', ls='--')

        dim = 0.5 * max(np.max(p0) - np.min(p0), np.max(p1) - np.min(p1))
        plt.figure()
        plt.plot(
            p0[indclose],
            p1[indclose],
            '.-',
            cent[0] + np.r_[0, vect[0]*dim],
            cent[1] + np.r_[0, vect[1]*dim],
            'x-k',
            cent2[0],
            cent2[1],
            'ok',
            mid0[imin],
            mid1[imin],
            'sk',
            p0[ind],
            p1[ind],
            '.-r',
        )
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.gca().set_title(f"ind.size = {ind.size}")

    return cent2


# ##############################################################
# ##############################################################
#               check
# ##############################################################


def _check(
    coll=None,
    key=None,
    key_cam=None,
    pixel=None,
    add_points=None,
    min_threshold=None,
    ind_ap_lim_spectral=None,
    convex=None,
    harmonize=None,
    reshape=None,
    verb=None,
    plot=None,
    store=None,
    debug=None,
    **kwdargs,
):

    # --------
    # key
    # --------

    # allowed
    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([
            len(v1['optics']) > 0
            for v1 in v0['doptics'].values()
        ])
    ]

    # check
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    spectro = coll.dobj['diagnostic'][key]['spectro']

    # -----------
    # key_cam
    # --------

    lok = coll.dobj['diagnostic'][key]['camera']
    key_cam = ds._generic_check._check_var(
        key_cam, 'key_cam',
        types=str,
        allowed=lok,
    )

    # --------
    # doptics
    # --------

    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam]
    optics = doptics['optics']
    optics_cls = doptics['cls']
    ispectro = doptics.get('ispectro')

    # safety check
    pinhole = doptics['pinhole']
    paths = doptics.get('paths')

    # ----------
    # curvature
    # ----------

    if spectro:
        clssp = optics_cls[ispectro[0]]
        ksp = optics[ispectro[0]]
        rcurve = coll.dobj[clssp][ksp]['dgeom']['curve_r']
        iok = np.isfinite(rcurve)
        curved = np.any(iok)
        concave = np.any(np.r_[rcurve][iok] > 0.)

        curve_mult = np.copy(rcurve)
        curve_mult[~iok] = 1

    else:
        curved = False
        concave = False
        curve_mult = [1., 1.]

    # -------------------------------------------------
    # ldeti: list of individual camera dict (per pixel)
    # -------------------------------------------------

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)
    is2d = dgeom['nd'] == '2d'

    # ---------
    # pixel
    # ---------

    if pixel is not None:
        try:
            _ = np.empty(dgeom['shape'])[pixel]
        except Exception as err:
            msg = (
                "Arg pixel must be an index applicable to camera shape\n"
                f"\t- camera: '{key_cam}'\n"
                f"\t- shape: {dgeom['shape']}\n"
                f"\t- pixel: {pixel}\n"
            )
            raise Exception(msg) from err

    # ------------------------------------
    # compute equivalent optics if spectro
    # ------------------------------------

    if spectro and len(optics[ispectro[0]+1:]) > 0:

        c0 = (
            len(ispectro) == 1
            or (
                len(ispectro) == 2
                and len(optics[ispectro[1]+1:]) == 0
            )
        )

        # apertures after crystal => reflection
        if c0:
            iref = ispectro[0]

        else:
            raise NotImplementedError()

    else:

        if pinhole is True:
            iref = len(optics) - 1
        else:

            # get optics with smallest solid angle as ref
            cx, cy, cz = coll.get_camera_cents_xyz(key_cam)
            ap_cent = np.array([
                coll.dobj[ocls][kop]['dgeom']['cent']
                for ocls, kop in zip(optics_cls, optics)
            ])
            ap_area = np.array([
                coll.dobj[ocls][kop]['dgeom']['area']
                for ocls, kop in zip(optics_cls, optics)
            ])
            dist2 = (
                (ap_cent[:, 0][None, :] - cx.ravel()[:, None])**2
                + (ap_cent[:, 1][None, :] - cy.ravel()[:, None])**2
                + (ap_cent[:, 2][None, :] - cz.ravel()[:, None])**2
            )
            sang = ap_area[None, :] / dist2
            sang[~paths] = np.nan
            iref = np.nanargmin(sang, axis=-1)

    # -----------
    # add_points
    # -----------

    add_points = ds._generic_check._check_var(
        add_points, 'add_points',
        types=int,
        default=3,
        sign='>0',
    )

    # ----------------------------------------------
    # index of aperture limiting the spectral range

    if spectro is True:
        ind_ap_lim_spectral = int(ds._generic_check._check_var(
            ind_ap_lim_spectral, 'ind_ap_lim_spectral',
            types=(float, int),
            default=min(np.mean(iref)+1, len(optics)),
        ))
    else:
        ind_ap_lim_spectral = None

    # -----------
    # convex

    isconvex = any(coll.get_optics_isconvex(doptics['optics']))
    convex = ds._generic_check._check_var(
        convex, 'convex',
        types=bool,
        default=isconvex,
    )

    # -----------
    # min_threshold

    min_threshold = ds._generic_check._check_var(
        min_threshold, 'min_threshold',
        types=float,
        default=500e-6 if convex else 0.002,
        sign='>0',
    )

    # -----------
    # harmonnize

    harmonize = ds._generic_check._check_var(
        harmonize, 'harmonize',
        types=bool,
        default=True,
        allowed=[True],
    )

    # -----------
    # reshape

    reshape = ds._generic_check._check_var(
        reshape, 'reshape',
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

    plot = ds._generic_check._check_var(
        plot, 'plot',
        default=False,
        types=bool,
    )

    # -----------
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        default=False,
        types=bool,
    )

    # -----------
    # debug

    debug = ds._generic_check._check_var(
        debug, 'debug',
        default=False,
        allowed=['intersect', False, True]
    )

    return (
        key,
        key_cam,
        iref,
        pinhole,
        paths,
        spectro,
        ispectro,
        curved,
        concave,
        curve_mult,
        optics,
        optics_cls,
        is2d,
        cx,
        cy,
        cz,
        pixel,
        add_points,
        min_threshold,
        ind_ap_lim_spectral,
        convex,
        harmonize,
        reshape,
        verb,
        plot,
        store,
        debug,
    )


# ##############################################################
# ##############################################################
#           Equivalent aperture non-spectro
# ##############################################################


def _get_equivalent_aperture(
    p_a=None,
    pt=None,
    lpoly_pre=None,
    ptsvect=None,
    # debug
    ii=None,
    debug=None,
    **kwdargs,
):

    # --------------
    # loop on optics
    # --------------

    nop_pre = len(lpoly_pre)
    for jj in range(nop_pre):

        # --------------------------
        # project on reference frame

        p0, p1 = ptsvect(
            pts_x=pt[0],
            pts_y=pt[1],
            pts_z=pt[2],
            vect_x=lpoly_pre[jj][0] - pt[0],
            vect_y=lpoly_pre[jj][1] - pt[1],
            vect_z=lpoly_pre[jj][2] - pt[2],
            strict=False,
            return_x01=True,
        )[-2:]

        if p0 is None:
            return None, None

        # --- DEBUG ---------
        if debug is True:
            _debug_plot(p_a=p_a, pa0=p0, pa1=p1, ii=ii, tit='local coords')
        # --------------------

        # -------
        # inside

        if np.all([p_a.isInside(xx, yy) for xx, yy in zip(p0, p1)]):
            p_a = plg.Polygon(np.array([p0, p1]).T)
        else:
            # intersection
            p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
            if p_a.nPoints() < 3:
                return None, None

    # return
    return np.array(p_a.contour(0)).T


# ##############################################################
# ##############################################################
#           Equivalent aperture spectro
# ##############################################################


def _get_equivalent_aperture_spectro(
    p_a=None,
    pt=None,
    lpoly_pre=None,
    lx01_post=None,
    # functions
    coord_x01toxyz=None,
    lcoord_x01toxyz_poly=None,
    pts2pt=None,
    ptsvect=None,
    lptsvect_poly=None,
    # options
    add_points=None,
    convex=None,
    # timing
    dt=None,
    # debug
    ii=None,
    ij=None,
    debug=None,
):

    # -----------------------------
    # loop on optics before crystal
    # -----------------------------

    nop_pre = len(lpoly_pre)
    for jj in range(nop_pre):

        # --------------------------
        # project on reference frame

        p0, p1 = ptsvect(
            pts_x=pt[0],
            pts_y=pt[1],
            pts_z=pt[2],
            vect_x=lpoly_pre[jj][0] - pt[0],
            vect_y=lpoly_pre[jj][1] - pt[1],
            vect_z=lpoly_pre[jj][2] - pt[2],
            strict=False,
            return_x01=True,
        )[-2:]

        # ------
        # inside

        if np.all([p_a.isInside(xx, yy) for xx, yy in zip(p0, p1)]):
            p_a = plg.Polygon(np.array([p0, p1]).T)
        else:
            # intersection
            p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
            if p_a.nPoints() < 3:
                return None, None

    # --------------
    # extract p0, p1

    p0, p1 = np.array(p_a.contour(0)).T

    # -----------------------------
    # loop on optics after crystal
    # -----------------------------

    for jj in range(len(lx01_post)):
        # print(f'\t {jj} / {nop_post}')      # DB

        # ----------
        # reflection

        p0, p1 = _class5_projections._get_reflection(
            # inital contour
            x0=p0,
            x1=p1,
            # polygon observed
            poly_x0=lx01_post[jj][0],
            poly_x1=lx01_post[jj][1],
            # observation point
            pt=pt,
            add_points=add_points,
            # functions
            coord_x01toxyz=coord_x01toxyz,
            coord_x01toxyz_poly=lcoord_x01toxyz_poly[jj],
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            ptsvect_poly=lptsvect_poly[jj],
            # timing
            # dt=dt,
            # debug
            ii=ii,
            ij=ij,
            jj=jj,
        )

        if p0 is None:
            # print('\n\t \t None 0\n')       # DB
            return p0, p1

        if convex:
            p0, p1 = _check_self_intersect_rectify(
                p0=p0,
                p1=p1,
                debug=debug,
            )

        # --- DEBUG ---------
        # if debug:
        #     _debug_plot(
        #         p_a=p_a,
        #         # p_b=p_a & plg.Polygon(np.array([p0, p1]).T),
        #         pa0=p0,
        #         pa1=p1,
        #         ii=ii,
        #         tit='not all',
        #     )
        # ----------------------

        # ------------
        # intersection

        p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
        if p_a.nPoints() < 3:
            # print('\n\t \t None 1\n')       # DB
            return None, None

            # print(f'\t\t interp => {p0.size} pts')       # DB
            # print('inter: ', p0)

    return p0, p1


def _check_self_intersect_rectify(
    p0=None,
    p1=None,
    # debug
    debug=None,
):

    # ---------------
    # get segments

    npts = p0.size

    A0, A1 = p0, p1
    B0, B1 = np.r_[p0[1:], p0[0]], np.r_[p1[1:], p1[0]]

    s0 = B0 - A0
    s1 = B1 - A1

    # -----------------------
    # get intersection matrix

    # k horizontal
    kA = np.full((npts, npts), -1, dtype=float)

    det_up = (
        (A0[None, :] - A0[:, None]) * s1[None, :]
        - (A1[None, :] - A1[:, None]) * s0[None, :]
    )
    det_lo = s0[:, None] * s1[None, :] - s1[:, None] * s0[None, :]

    iok = np.abs(det_lo) > 0
    kA[iok] = det_up[iok] / det_lo[iok]
    iok[iok] = (kA[iok] > 0) & (kA[iok] < 1)

    if not np.any(iok):
        if debug is True:
            _debug_intersect(tit="No kA", **locals())
        return p0, p1

    kB = np.full((npts, npts), -1, dtype=float)
    A0f = np.repeat(A0[:, None], npts, axis=1)
    A1f = np.repeat(A1[:, None], npts, axis=1)
    s0f = np.repeat(s0[:, None], npts, axis=1)
    s1f = np.repeat(s1[:, None], npts, axis=1)

    M0 = A0f[iok] + kA[iok] * s0f[iok]
    M1 = A1f[iok] + kA[iok] * s1f[iok]

    kB[iok] = (
        (M0 - A0f.T[iok]) * s0f.T[iok] + (M1 - A1f.T[iok]) * s1f.T[iok]
    ) / (s0f.T[iok]**2 + s1f.T[iok]**2)
    iok[iok] = (kB[iok] > 0) & (kB[iok] < 1)

    # ---------------
    # trivial cases

    # no intersection
    if not np.any(iok):
        if debug is True:
            _debug_intersect(tit="No kB", **locals())
        return p0, p1

    # several intersections
    if np.sum(iok) != 2:
        msg = "Multiple intersections detected"
        if debug is True:
            _debug_intersect(tit=msg, **locals())
        raise Exception(msg)

    # --------------------
    # single intersection

    indA, indB = np.nonzero(iok)
    assert np.all(indA == indB[::-1]), (indA, indB)

    indpts = indA[:, None] + np.r_[0, 1][None, :]

    ind = np.r_[
        np.arange(0, indpts[0, 0] + 1),
        np.arange(indpts[1, 0], indpts[0, 1] - 1, -1),
        np.arange(indpts[1, 1], npts)
    ]

    # ----------------
    # DEBUG

    if debug is True or debug == 'intersect':
        _debug_intersect(tit='found', **locals())

    return p0[ind], p1[ind]


def _debug_intersect(
    p0=None,
    p1=None,
    kA=None,
    kB=None,
    iok=None,
    det_up=None,
    det_lo=None,
    ind=None,
    tit=None,
    **kwdargs,
):

    fig, axs = plt.subplots(figsize=(12, 10), nrows=2, ncols=4)
    fig.suptitle(tit, size=12, fontweight='bold')

    axs[0, 0].plot(np.r_[p0, p0[0]], np.r_[p1, p1[0]], '.-k')

    axs[0, 1].imshow(iok, origin='upper', interpolation='nearest')
    axs[0, 2].imshow(
        kA,
        origin='upper',
        interpolation='nearest',
        vmin=0,
        vmax=1,
    )
    axs[1, 1].imshow(
        det_up,
        origin='upper',
        interpolation='nearest',
        vmin=-np.max(np.abs(det_up)),
        vmax=np.max(np.abs(det_up)),
        cmap=plt.cm.seismic,
    )
    axs[1, 2].imshow(
        det_lo,
        origin='upper',
        interpolation='nearest',
        vmin=-np.max(np.abs(det_lo)),
        vmax=np.max(np.abs(det_lo)),
        cmap=plt.cm.seismic,
    )
    axs[0, 1].set_title("iok")
    axs[0, 2].set_title("kA")
    axs[1, 1].set_title("det_up")
    axs[1, 2].set_title("det_lo")

    print('kA\n',kA)
    print('det_up\n', det_up)
    print('det_lo\n', det_lo)

    if kB is not None:
        axs[0, 3].imshow(
            kB,
            origin='upper',
            interpolation='nearest',
            vmin=0,
            vmax=1,
        )

        print('kB\n', kB)

    if ind is not None:
        axs[0, 0].plot(
            np.r_[p0[ind], p0[ind[0]]],
            np.r_[p1[ind], p1[ind[0]]],
            '.-b',
        )

        print('ind', ind)

    raise Exception()

# ##############################################################
# ##############################################################
#           Plot
# ##############################################################


def _plot(
    poly_x0=None,
    poly_x1=None,
    p0=None,
    p1=None,
    cents0=None,
    cents1=None,
    # graph options
    fs=None,
    tit=None,
    xlab=None,
    ylab=None,
):

    fig = plt.figure(figsize=fs)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_title(tit)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    i0 = np.r_[np.arange(0, poly_x0.size), 0]
    ax.plot(poly_x0[i0], poly_x1[i0], '.-k')

    if p0.shape[1] > 0:
        i1 = np.r_[np.arange(0, p0.shape[1]), 0]
        for ii in range(p0.shape[0]):
            ax.plot(p0[ii, i1], p1[ii, i1], '.-', label=f'pix {ii}')

    if cents0 is not None:
        ax.plot(cents0, cents1, 'xr')

    ax.legend()
    return


# ##############################################################
# ##############################################################
#           Debug plots
# ##############################################################


def _debug_plot(
    p_a=None,
    p_b=None,
    pa0=None,
    pa1=None,
    pb0=None,
    pb1=None,
    pc0=None,
    pc1=None,
    ii=None,
    tit=None,
):

    plt.figure()

    # p_a
    if p_a is not None:
        p_a = np.array(p_a.contour(0))
        ind = np.r_[np.arange(0, p_a.shape[0]), 0]
        plt.plot(
            p_a[ind, 0],
            p_a[ind, 1],
            ls='-',
            lw=1.,
            marker='.',
            label=f'p_a ({p_a.shape[0]} pts)',
        )

    # p_b
    if p_b is not None:
        p_b = np.array(p_b.contour(0))
        ind = np.r_[np.arange(0, p_b.shape[0]), 0]
        plt.plot(
            p_b[ind, 0],
            p_b[ind, 1],
            ls='-',
            lw=1.,
            marker='.',
            label=f'p_b ({p_b.shape[0]} pts)',
        )

    # pa
    if pa0 is not None:
        ind = np.r_[np.arange(0, pa0.size), 0]
        plt.plot(
            pa0[ind],
            pa1[ind],
            ls='-',
            lw=1.,
            marker='x',
            label=f'pa ({pa0.size} pts)',
        )

    # pb
    if pb0 is not None:
        ind = np.r_[np.arange(0, pb0.size), 0]
        plt.plot(
            pb0[ind],
            pb1[ind],
            ls='-',
            lw=1.,
            marker='+',
            label=f'pb ({pb0.size} pts)',
        )

    # pc
    if pc0 is not None:
        ind = np.r_[np.arange(0, pc0.size), 0]
        plt.plot(
            pc0[ind],
            pc1[ind],
            ls='-',
            lw=1.,
            marker='+',
            label=f'pc ({pc0.size} pts)',
        )

    plt.legend()

    if ii is not None:
        tit0 = f'ii = {ii}'
        if tit is None:
            tit = tit0
        else:
            tit = tit0 + ', ' + tit
        plt.gca().set_title(tit, size=12)


def _debug_plot2(
    p0=None,
    p1=None,
    cents0=None,
    cents1=None,
    ii=None,
    ip=None,
    coord_x01toxyz=None,
    cx=None,
    cy=None,
    cz=None,
    area=None,
):

    plt.figure()
    plt.plot(
        np.r_[p0, p0[0]],
        np.r_[p1, p1[0]],
        c='k',
        ls='-',
        lw=1.,
        marker='.',
    )
    plt.plot([cents0[ii]], [cents1[ii]], 'xk')
    ppx, ppy, ppz = coord_x01toxyz(
        x0=np.r_[cents0[ii]],
        x1=np.r_[cents1[ii]],
    )
    ddd = np.linalg.norm(
        np.r_[ppx - cx[ip], ppy - cy[ip], ppz - cz[ip]]
    )
    plt.gca().text(
        np.mean(p0),
        np.mean(p1),
        f'area\n{area[ii]:.3e} m2\ndist\n{ddd:.6e} m',
        size=12,
        horizontalalignment='center',
        verticalalignment='center',
    )
    plt.gca().set_title(f'planar coordinates - {ii}', size=12)
    plt.gca().set_xlabel('x0 (m)', size=12)
    plt.gca().set_ylabel('x1 (m)', size=12)
    plt.gca().set_aspect('equal')
