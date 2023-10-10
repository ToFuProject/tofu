# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt


import Polygon as plg
# from Polygon import Utils as plgUtils
from scipy.spatial import ConvexHull
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
    harmonize=None,
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

    (
        key,
        key_cam,
        kref,
        cref,
        spectro,
        ispectro,
        curved,
        concave,
        curve_mult,
        lop_pre,
        lop_pre_cls,
        lop_post,
        lop_post_cls,
        is2d,
        shape0,
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
        pixel = np.arange(0, cx.size)

    if return_for_etendue is None:
        return_for_etendue = False

    # ---------------
    # Prepare optics

    lpoly_pre = [
        coll.get_optics_poly(
            key=k0,
            mode='thr',
            add_points=add_points,
            min_threshold=min_threshold,
            return_outline=False,
        )
        for k0 in lop_pre
    ]
    nop_pre = len(lop_pre)

    lx01_post = [
        coll.get_optics_outline(
            key=k0,
            mode='thr',
            add_points=add_points,
            min_threshold=min_threshold,
        )
        for c0, k0 in zip(lop_post_cls, lop_post)
    ]
    nop_post = len(lop_post)

    # -------------------
    # prepare functions

    # coordinate func
    coord_x01toxyz = coll.get_optics_x01toxyz(key=kref)
    lcoord_x01toxyz_poly = [
        coll.get_optics_x01toxyz(key=oo)
        for oo in lop_post
    ]

    # pts2pts func
    if spectro:
        pts2pt = coll.get_optics_reflect_pts2pt(key=kref)
        if len(lop_post) > 0:
            cls_ap_lim = lop_post_cls[ind_ap_lim_spectral]
            kap_lim = lop_post[ind_ap_lim_spectral]
            dist_cryst2ap = np.linalg.norm(
                coll.dobj[cls_ap_lim][kap_lim]['dgeom']['cent']
                - coll.dobj[cref][kref]['dgeom']['cent']
            )
        else:
            dist_cryst2ap = 0.
    else:
        pts2pt = None
        dist_cryst2ap = None

    # ptsvect func
    ptsvect = coll.get_optics_reflect_ptsvect(key=kref)
    lptsvect_poly = [
        coll.get_optics_reflect_ptsvect(key=oo)
        for oo in lop_post
    ]

    # equivalent aperture func
    if spectro:
        func = _get_equivalent_aperture_spectro
    else:
        func = _get_equivalent_aperture

    # -------------------
    # prepare output

    x0 = []
    x1 = []

    # ---------------
    # loop on pixels

    # dt = np.zeros((14,), dtype=float)

    # add pts to initial polygon only if curved
    if spectro:
        rcurv = np.r_[coll.dobj[cref][kref]['dgeom']['curve_r']]
        ind = ~np.isinf(rcurv)
        if np.any(rcurv[ind] > 0):
            addp0 = add_points
        else:
            addp0 = False
    else:
        addp0 = False

    # intial polygon
    p_a = coll.get_optics_outline(key=kref, add_points=addp0)
    p_a = plg.Polygon(np.array([p_a[0], p_a[1]]).T)

    iok = np.ones((pixel.size,), dtype=bool)

    for ii, ij in enumerate(pixel):

        # ----- DEBUG -------
        # if ij not in [1148]:
        #     iok[ii] = False
        #     x0.append(None)
        #     x1.append(None)
        #     continue
        # -------------------

        if verb is True:
            msg = f"\t- camera '{key_cam}': pixel {ii + 1} / {pixel.size}"
            end = '\n' if ii == len(pixel) - 1 else '\r'
            print(msg, end=end, flush=True)

        p0, p1 = func(
            p_a=p_a,
            pt=np.r_[cx[ij], cy[ij], cz[ij]],
            nop_pre=nop_pre,
            lpoly_pre=lpoly_pre,
            nop_post=nop_post,
            lx01_post=lx01_post,
            # functions
            coord_x01toxyz=coord_x01toxyz,
            lcoord_x01toxyz_poly=lcoord_x01toxyz_poly,
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            lptsvect_poly=lptsvect_poly,
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
            iok[ii] = False

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
        )

    # -------------
    # xyz
    # -------------

    if return_for_etendue is True:

        pts2plane = coll.get_optics_reflect_ptsvect(
            key=kref,
            asplane=True,
        )

        px = np.full(x0.shape, np.nan)
        py = np.full(x0.shape, np.nan)
        pz = np.full(x0.shape, np.nan)
        cents0 = np.full((x0.shape[0],), np.nan)
        cents1 = np.full((x0.shape[0],), np.nan)
        area = np.full((x0.shape[0],), np.nan)

        for ii, ip in enumerate(pixel):

            if not iok[ii]:
                continue

            pxi, pyi, pzi = coord_x01toxyz(x0=x0[ii, :], x1=x1[ii, :])

            (
                px[ii, :], py[ii, :], pz[ii, :],
                _, _, _, _, _, p0, p1,
            ) = pts2plane(
                pts_x=cx[ii],
                pts_y=cy[ii],
                pts_z=cz[ii],
                vect_x=pxi - cx[ii],
                vect_y=pyi - cy[ii],
                vect_z=pzi - cz[ii],
                strict=False,
                return_x01=True,
            )

            # area
            area[ii] = plg.Polygon(np.array([p0, p1]).T).area()

            # centroid in 3d
            polyi = plg.Polygon(np.array([x0[ii, :], x1[ii, :]]).T)
            cent = polyi.center()
            if not polyi.isInside(*cent):
                cent = _get_centroid(
                    x0[ii, :],
                    x1[ii, :],
                    cent,
                    debug=False,
                )

            cents0[ii], cents1[ii] = cent

            # --- DEBUG ---------
            # if ii in [1148]:
            #     _debug_plot2(
            #         p0=p0, p1=p1,
            #         cents0=cents0, cents1=cents1,
            #         ii=ii,
            #         ip=ip,
            #         coord_x01toxyz=coord_x01toxyz,
            #         cx=cx, cy=cy, cz=cz,
            #         area=area,
            #     )
            # --------------------

        centsx, centsy, centsz = coord_x01toxyz(
            x0=cents0,
            x1=cents1,
        )
        # add centroid to x0, x1


        plane_nin = coll.dobj[cref][kref]['dgeom']['nin']

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

    ntot = np.prod(shape0)
    if is2d and harmonize and reshape and x0.shape[0] == ntot:
        shape = tuple(np.r_[shape0, x0.shape[-1]])
        x0 = x0.reshape(shape)
        x1 = x1.reshape(shape)

    # -------------
    # plot & return
    # -------------

    if plot is True:
        out0, out1 = coll.get_optics_outline(key=kref, add_points=False)
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
            x0, x1, kref, iok,
            px, py, pz,
            cx, cy, cz,
            cents0, cents1,
            centsx, centsy, centsz,
            area, plane_nin,
            spectro, dist_cryst2ap
        )
    else:
        return x0, x1, kref, iok


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

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([
            v1['collimator']
            or len(v1['optics']) > 0
            for v1 in v0['doptics'].values()
        ])
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    spectro = coll.dobj['diagnostic'][key]['spectro']

    # -----------
    # key_cam

    lok =coll.dobj['diagnostic'][key]['camera']
    key_cam = ds._generic_check._check_var(
        key_cam, 'key_cam',
        types=str,
        allowed=lok,
    )

    # --------
    # doptics

    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam]
    collimator = doptics['collimator']

    if collimator:
        raise NotImplementedError("Collimator camera, TBF")

    else:
        optics = doptics['optics']
        optics_cls = doptics['cls']

    ispectro = doptics.get('ispectro')

    # --------
    # curvature

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

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)
    is2d = dgeom['nd'] == '2d'
    shape0 = cx.shape

    if is2d:
        cx = cx.ravel()
        cy = cy.ravel()
        cz = cz.ravel()

    # ---------
    # pixel

    if pixel is not None:
        pixel = np.atleast_1d(pixel).astype(int)

        if pixel.ndim == 2 and pixel.shape[1] == 2 and is2d:
            pixel = pixel[:, 0] * shape0[1]  + pixel[:, 1]

        if pixel.ndim != 1:
            msg = "pixel can only have ndim = 2 for 2d cameras!"
            raise Exception(msg)

    # ------------------------------------
    # compute equivalent optics if spectro

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
            kref = optics[ispectro[0]]
            cref = optics_cls[ispectro[0]]
            lop_pre = optics[:ispectro[0]]
            lop_pre_cls = optics_cls[:ispectro[0]]
            lop_post = optics[ispectro[0]+1:]
            lop_post_cls = optics_cls[ispectro[0]+1:]

        else:
            raise NotImplementedError()

    elif collimator:
        raise NotImplementedError()
        lop_post = []
        lop_post_cls = []

    else:
        kref = optics[-1]
        cref = optics_cls[-1]
        lop_pre = optics[:-1]
        lop_pre_cls = optics_cls[:-1]
        lop_post = []
        lop_post_cls = []

    # -----------
    # add_points

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
            default=0,
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
        kref,
        cref,
        spectro,
        ispectro,
        curved,
        concave,
        curve_mult,
        lop_pre,
        lop_pre_cls,
        lop_post,
        lop_post_cls,
        is2d,
        shape0,
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
    nop_pre=None,
    lpoly_pre=None,
    ptsvect=None,
    # debug
    ii=None,
    debug=None,
    **kwdargs,
):

    # loop on optics
    for jj in range(nop_pre):

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
    nop_pre=None,
    lpoly_pre=None,
    nop_post=None,
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

    # loop on optics before crystal
    for jj in range(nop_pre):

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

        # inside
        if np.all([p_a.isInside(xx, yy) for xx, yy in zip(p0, p1)]):
            p_a = plg.Polygon(np.array([p0, p1]).T)
        else:
            # intersection
            p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
            if p_a.nPoints() < 3:
                return None, None

    # extract p0, p1
    p0, p1 = np.array(p_a.contour(0)).T

    # loop on optics after crystal
    for jj in range(nop_post):
        # print(f'\t {jj} / {nop_post}')      # DB

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
        # if ij in [1148]:
        #     _debug_plot(
        #         p_a=p_a,
        #         # p_b=p_a & plg.Polygon(np.array([p0, p1]).T),
        #         pa0=p0,
        #         pa1=p1,
        #         ii=ii,
        #         tit='not all',
        #     )
        # ----------------------

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