# -*- coding: utf-8 -*-


import warnings


import numpy as np
import matplotlib.pyplot as plt


import Polygon as plg
import datastock as ds


from . import _class5_projections


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
    plot=None,
):

    # -------------
    # check inputs

    # key
    lok = list(coll.dobj.get('diagnostic', {}))
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # optics
    optics, optics_cls = coll.get_diagnostic_optics(key=key)
    ispectro = [
        ii for ii, cc in enumerate(optics_cls)
        if cc in ['grating', 'crystal']
    ]

    # dgeom
    dgeom = coll.dobj['camera'][optics[0]]['dgeom']
    is2d = dgeom['type'] == '2d'

    # pixel
    if pixel is None:
        pixel = np.arange(0, cx.size)
    else:
        pixel = np.atleast_1d(pixel).ravel().astype(int)

        if pixel.ndim == 2 and pixel.shape[1] == 1 and is2d:
            pixel = pixel[:, 0] + pixel[:, 1]

        if pixel.ndim != 1:
            msg = "pixel can only have ndim = 2 for 2d cameras!"
            raise Exception(msg)

    # -------------
    # prepare

    cx, cy, cz = coll.get_camera_cents_xyz(key=optics[0])
    dvect = coll.get_camera_unit_vectors(key=optics[0])
    outline = dgeom['outline']
    out0 = coll.ddata[outline[0]]['data']
    out1 = coll.ddata[outline[1]]['data']
    par = dgeom['parallel']
    shape0 = cx.shape

    # pixel => pt
    if is2d:
        shape0 = cx.shape
        cx = cx.ravel()
        cy = cy.ravel()
        cz = cz.ravel()

    # -----------------------------------
    # get reference frame for projections

    if len(ispectro) == 0:
        kref = optics[-1]
    else:
        kref = ispectro[0]

    # ---------------
    # Prepare optics 

    lpoly_pre = [
        coll.dobj[c0][k0]['dgeom']['poly']
        for c0, k0 in zip(lop_pre_cls, lop_pre)
    ]
    lpoly_pre_x = [coll.ddata[pp[0]]['data'] for pp in lpoly_pre]
    lpoly_pre_y = [coll.ddata[pp[1]]['data'] for pp in lpoly_pre]
    lpoly_pre_z = [coll.ddata[pp[2]]['data'] for pp in lpoly_pre]

    lout_post = [
        coll.dobj[c0][k0]['dgeom']['outline']
        for c0, k0 in zip(lop_post_cls, lop_post)
    ]
    lx0_post = [coll.ddata[pp[0]]['data'] for pp in lout_post]
    lx1_post = [coll.ddata[pp[1]]['data'] for pp in lout_post]

    # -------------------
    # prepare functions

    coord_x01toxyz = coll.get_optics_x01toxyz(key=key)
    lcoord_x01toxyz_poly = [
        coll.get_optics_x01toxyz(key=oo)
        for oo in lop_post
    ]

    pts2pt = coll.get_optics_pts2pt(key=key)
    ptsvect = coll.get_optics_ptsvect(key=key)
    lptsvect_poly = [
        coll.get_optics_ptsvect(key=oo)
        for oo in lop_post
    ]

    # -------------------
    # prepare output

    x0 = []
    x1 = []

    # ---------------
    # loop on optics

    for ii in pixel:
        if len(ispectro):
            p0, p1 = _get_equivalent_aperture(
                p_a=p_a,
                pt=np.r_[cx[ii], cy[jj], cz[jj]],
                nap_pre=nap_pre,
                lpoly_pre_x=lpoly_pre_x,
                lpoly_pre_y=lpoly_pre_y,
                lpoly_pre_z=lpoly_pre_z,
            )

        else:
            p0, p1 = _get_equivalent_aperture_spectro(
                p_a=p_a,
                pt=np.r_[cx[ii], cy[jj], cz[jj]],
                nap_pre=nap_pre,
                lpoly_pre_x=lpoly_pre_x,
                lpoly_pre_y=lpoly_pre_y,
                lpoly_pre_z=lpoly_pre_z,
                nap_post=nap_post,
                lx0_post=lx0_post,
                lx1_post=lx1_post,
                # functions 
                coord_x01toxyz=coord_x01toxyz,
                lcoord_x01toxyz_poly=lcoord_x01toxyz_poly,
                pts2pt=pts2pt,
                ptsvect=ptsvect,
                lptsvect_poly=lptsvect_poly,
                # option
                convex=convex,
            )

        # convex hull
        if p0 is not None and convex:
            p0, p1 = np.array(plgUtils.ConvexHull(
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
        imax = np.arange(0, nmax)
        for ii in range(pixel.size):
            if x0[ii] is None:
                x0[ii] = nan
            elif ln[ii] < nmax:
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

    if is2d and harmonizei and reshape:
        shape = tuple(np.r_[shape0, x0.shape[-1]])
        x0 = x0.reshape(shape)
        x1 = x1.reshape(shape)

    # -------------
    # plot & return
    # -------------

    if plot is True:
        _plot(
            poly_x0=outline_x0,
            poly_x1=outline_x1,
            p0=p0,
            p1=p1,
            # options
            tit=f"Reflections on {key}",
            xlab=None,
            ylab=None,
        )

    return x0, x1


# ##############################################################
# ##############################################################
#           Equivalent aperture non-spectro
# ##############################################################


def _get_equivalent_aperture(
    p_a=None,
    pt=None,
    nap_pre=None,
    lpoly_pre_x=None,
    lpoly_pre_y=None,
    lpoly_pre_z=None,
):

    for jj in range(nap_pre):

        # project on reference frame
        p0, p1 = project_func(
            pt_x=pt[0],
            pt_y=pt[1],
            pt_z=pt[2],
            vect_x=lpoly_pre_x[jj] - pt[0],
            vect_y=lpoly_pre_y[jj] - pt[1],
            vect_z=lpoly_pre_z[jj] - pt[2],
        )

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
    nap_pre=None,
    lpoly_pre_x=None,
    lpoly_pre_y=None,
    lpoly_pre_z=None,
    nap_post=None,
    lx0_post=None,
    lx1_post=None,
    # functions 
    coord_x01toxyz=None,
    lcoord_x01toxyz_poly=None,
    pts2pt=None,
    ptsvect=None,
    lptsvect_poly=None,
):

    for jj in range(nap_pre):

        # project on reference frame
        p0, p1 = project_func(
            pt_x=pt[0],
            pt_y=pt[1],
            pt_z=pt[2],
            vect_x=lpoly_pre_x[jj] - pt[0],
            vect_y=lpoly_pre_y[jj] - pt[1],
            vect_z=lpoly_pre_z[jj] - pt[2],
        )

        # intersection
        p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
        if p_a.nPoints() < 3:
            return None, None

    # extract p0, p1
    p0, p1 = np.array(p_a.contour(0)).T

    # spectro part
    for jj in range(nap_post):

        p0, p1 = _class5_projection._get_reflection(
            # inital contour
            x0=p0,
            x1=p1,
            # polygon observed
            poly_x0=lx0_post[jj],
            poly_x1=lx1_post[jj],
            # observation point
            pt=pt,
            # functions
            coord_x01toxyz=coord_x01toxyz,
            coord_x01toxyz_poly=lcoord_x01toxyz_poly[jj],
            pts2pt=pts2pt,
            ptsvect=ptsvect,
            ptsvect_poly=lptsvect_poly[jj],
        )

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
    ax.plot(p0, p1, '.-r')
    return


