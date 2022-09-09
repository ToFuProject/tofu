# -*- coding: utf-8 -*-


import warnings


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt


import Polygon as plg
from Polygon import Utils as plgUtils
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
    pixel=None,
    # inital contour
    add_points=None,
    # options
    convex=None,
    harmonize=None,
    reshape=None,
    # plot
    verb=None,
    plot=None,
    store=None,
):

    # -------------
    # check inputs

    (
        key,
        kref,
        ispectro,
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
        convex,
        harmonize,
        reshape,
        verb,
        plot,
        store,
    ) = _check(
        coll=coll,
        key=key,
        pixel=pixel,
        convex=convex,
        harmonize=harmonize,
        reshape=reshape,
        verb=verb,
        plot=plot,
        store=store,
    )

    if pixel is None:
        pixel = np.arange(0, cx.size)

    # ---------------
    # Prepare optics 

    lpoly_pre = [
        coll.get_optics_poly(
            key=k0,
            add_points=5,
            return_outline=False,
        )
        for k0 in lop_pre
    ]
    nop_pre = len(lop_pre)

    lout_post = [
        coll.dobj[c0][k0]['dgeom']['outline']
        for c0, k0 in zip(lop_post_cls, lop_post)
    ]
    lx0_post = [coll.ddata[pp[0]]['data'] for pp in lout_post]
    lx1_post = [coll.ddata[pp[1]]['data'] for pp in lout_post]
    nop_post = len(lop_post)

    # -------------------
    # prepare functions

    coord_x01toxyz = coll.get_optics_x01toxyz(key=kref)
    lcoord_x01toxyz_poly = [
        coll.get_optics_x01toxyz(key=oo)
        for oo in lop_post
    ]

    pts2pt = coll.get_optics_reflect_pts2pt(key=kref)
    ptsvect = coll.get_optics_reflect_ptsvect(key=kref)
    lptsvect_poly = [
        coll.get_optics_reflect_ptsvect(key=oo)
        for oo in lop_post
    ]

    if len(ispectro) == 0:
        func = _get_equivalent_aperture
    else:
        func = _get_equivalent_aperture_spectro

    # -------------------
    # prepare output

    x0 = []
    x1 = []

    # ---------------
    # loop on pixels

    # intial polygon
    p_a = coll.get_optics_outline(key=kref, add_points=False)
    p_a = plg.Polygon(np.array([p_a[0], p_a[1]]).T)

    for ii in pixel:
        p0, p1 = func(
            p_a=p_a,
            pt=np.r_[cx[ii], cy[ii], cz[ii]],
            nop_pre=nop_pre,
            lpoly_pre=lpoly_pre,
            nop_post=nop_post,
            lx0_post=lx0_post,
            lx1_post=lx1_post,
            # functions 
            coord_x01toxyz=coord_x01toxyz,
            lcoord_x01toxyz_poly=lcoord_x01toxyz_poly,
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            lptsvect_poly=lptsvect_poly,
            # options
            convex=convex,
        )

        # convex hull
        if p0 is not None and convex:
            p0, p1 = np.array(plgUtils.convexHull(
                plg.Polygon(np.array([p0, p1]).T)
            ).contour(0)).T

        # append
        x0.append(p0)
        x1.append(p1)

    # --------------------
    # harmonize if necessary
    # --------------------

    if harmonize:
        ln = [p0.size if p0 is not None else 0 for p0 in x0]
        nmax = np.max(ln)
        nan = np.full((nmax,), np.nan)
        for ii in range(pixel.size):
            if x0[ii] is None:
                x0[ii] = nan
            elif ln[ii] < nmax:
                imax = np.linspace(0, ln[ii]-1, nmax)
                x0[ii] = scpinterp.interp1d(
                    np.arange(0, ln[ii]),
                    x0[ii],
                    kind='linear',
                )(imax)
                x1[ii] = scpinterp.interp1d(
                    np.arange(0, ln[ii]),
                    x1[ii],
                    kind='linear',
                )(imax)

        x0 = np.array(x0)
        x1 = np.array(x1)

    # --------------------
    # reshape if necessary
    # --------------------

    ntot = shape0[0] * shape0[1]
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
            # options
            tit=f"Reflections on {key}",
            xlab=None,
            ylab=None,
        )

    return x0, x1


# ##############################################################
# ##############################################################
#               check
# ##############################################################


def _check(
    coll=None,
    key=None,
    pixel=None,
    convex=None,
    harmonize=None,
    reshape=None,
    verb=None,
    plot=None,
    store=None,
):

    # --------
    # key

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if len(v0['optics']) > 1
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    optics, optics_cls = coll.get_diagnostic_optics(key=key)
    ispectro = [
        ii for ii, cc in enumerate(optics_cls)
        if cc in ['grating', 'crystal']
    ]

    # -------------------------------------------------
    # ldeti: list of individual camera dict (per pixel)

    dgeom = coll.dobj['camera'][optics[0]]['dgeom']
    cx, cy, cz = coll.get_camera_cents_xyz(key=optics[0])
    is2d = dgeom['type'] == '2d'
    shape0 = cx.shape

    if is2d:
        cx = cx.ravel()
        cy = cy.ravel()
        cz = cz.ravel()

    # ---------
    # pixel

    if pixel is not None:
        pixel = np.atleast_1d(pixel).ravel().astype(int)

        if pixel.ndim == 2 and pixel.shape[1] == 1 and is2d:
            pixel = pixel[:, 0] + pixel[:, 1]

        if pixel.ndim != 1:
            msg = "pixel can only have ndim = 2 for 2d cameras!"
            raise Exception(msg)

    # ------------------------------------
    # compute equivalent optics if spectro

    if len(ispectro) > 0 and len(optics[ispectro[0]+1:]) > 0:

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
            lop_pre = optics[1:ispectro[0]]
            lop_pre_cls = optics_cls[1:ispectro[0]]
            lop_post = optics[ispectro[0]+1:]
            lop_post_cls = optics_cls[ispectro[0]+1:]

        else:
            raise NotImplementedError()

    else:
        kref = optics[-1]
        lop_pre = optics[1:-1]
        lop_pre_cls = optics_cls[1:-1]
        lop_post = []
        lop_post_cls = []

    # -----------
    # convex

    convex = ds._generic_check._check_var(
        convex, 'convex',
        types=bool,
        default=True,
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
    if pixel is None:
        plot = False

    # -----------
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        default=False,
        types=bool,
    )

    return (
        key,
        kref,
        ispectro,
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
        convex,
        harmonize,
        reshape,
        verb,
        plot,
        store,
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
    lx0_post=None,
    lx1_post=None,
    # functions 
    coord_x01toxyz=None,
    lcoord_x01toxyz_poly=None,
    pts2pt=None,
    ptsvect=None,
    lptsvect_poly=None,
    convex=None,
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

        import pdb; pdb.set_trace() # DB
        # intersection
        p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
        if p_a.nPoints() < 3:
            return None, None

    # extract p0, p1
    p0, p1 = np.array(p_a.contour(0)).T

    # loop on optics after crystal
    for jj in range(nop_post):

        # interpolate
        p0, p1 = _compute._interp_poly(
            lp=[p0, p1],
            add_points=5,
            mode='min',
            isclosed=False,
            closed=False,
            ravel=True,
        )

        p0, p1 = _class5_projections._get_reflection(
            # inital contour
            x0=p0,
            x1=p1,
            # polygon observed
            poly_x0=lx0_post[jj],
            poly_x1=lx1_post[jj],
            # observation point
            pt=pt,
            add_points=5,
            # functions
            coord_x01toxyz=coord_x01toxyz,
            coord_x01toxyz_poly=lcoord_x01toxyz_poly[jj],
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            ptsvect_poly=lptsvect_poly[jj],
        )

        if p0 is None:
            return p0, p1

        # convex hull
        if convex:
            p0, p1 = np.array(plgUtils.convexHull(
                plg.Polygon(np.array([p0, p1]).T)
            ).contour(0)).T

        # intersection
        p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
        if p_a.nPoints() < 3:
            return None, None

        # update
        p0, p1 = np.array(p_a.contour(0)).T

    return p0, p1


# ##############################################################
# ##############################################################
#           Plot
# ##############################################################


def _plot(
    poly_x0=None,
    poly_x1=None,
    p0=None,
    p1=None,
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

    ax.plot(poly_x0, poly_x1, '.-k')
    ax.plot(p0.ravel(), p1.ravel(), '.-r')
    return


