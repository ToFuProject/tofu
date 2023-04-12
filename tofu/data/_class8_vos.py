# -*- coding: utf-8 -*-


import numpy as np
from matplotlib.path import Path
import datastock as ds
import bsplines2d as bs2
from contourpy import contour_generator
from scipy.spatial import ConvexHull
import scipy.interpolate as scpinterp


import datetime as dtm      # DB


import Polygon as plg


from . import _class8_vos_broadband as _vos_broadband


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def compute_vos(
    coll=None,
    key_diag=None,
    key_mesh=None,
    config=None,
    # parameters
    res=None,
    margin_poly=None,
    margin_par=None,
    margin_perp=None,
    # options
    add_points=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    # bool
    visibility=None,
    convex=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
    timing=None,
):

    if timing:
        t0 = dtm.datetime.now()     # DB

    # ------------
    # check inputs

    (
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res,
        margin_par,
        margin_perp,
        visibility,
        verb,
        plot,
        store,
        timing,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_mesh=key_mesh,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        visibility=visibility,
        verb=verb,
        plot=plot,
        store=store,
        timing=timing,
    )

    if verb is True:
        msg = f"\nComputing vos for diag '{key_diag}':"
        print(msg)

    # ------------
    # sample mesh

    dsamp = coll.get_sample_mesh(
        key=key_mesh,
        res=res,
        mode='abs',
        grid=True,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=None,
        Dx1=None,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    sh = dsamp['x0']['data'].shape
    x0f = dsamp['x0']['data'].ravel()
    x1f = dsamp['x1']['data'].ravel()

    sh1 = tuple([ss + 2 for ss in sh])
    bool_cross = np.zeros(sh1, dtype=bool)

    x0u = dsamp['x0']['data'][:, 0]
    x1u = dsamp['x1']['data'][0, :]
    x0l = np.r_[x0u[0] - (x0u[1] - x0u[0]), x0u, x0u[-1] + (x0u[-1] - x0u[-2])]
    x1l = np.r_[x1u[0] - (x1u[1] - x1u[0]), x1u, x1u[-1] + (x1u[-1] - x1u[-2])]
    x0l = np.repeat(x0l[:, None], x1l.size, axis=1)
    x1l = np.repeat(x1l[None, :], x0l.shape[0], axis=0)

    # ------------
    # prepare output

    # dvos = _prepare_output(
        # coll=coll,
        # key_diag=key_diag,
        # shape_samp=sh,
        # spectro=spectro,
    # )

    # --------------
    # prepare optics

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    # timing
    if timing:
        t1 = dtm.datetime.now()     # DB
        dt1 = (t1 - t0).total_seconds()
        dt11, dt22 = 0, 0
        dt111, dt222, dt333 = 0, 0, 0
    dt1111, dt2222, dt3333, dt4444 = 0, 0, 0, 0

    # --------------
    # prepare optics

    dvos = {}
    for key_cam in dcompute.keys():


        if spectro:
            dvos[key_cam] = _vos_spectro(
                x0=x0,
                x1=x1,
                ind=ind,
                dphi=dphi[:, ii],
            )

        else:
            dvos[key_cam] = _vos_broadband._vos(

            )


    # timing
    if timing:
        t2 = dtm.datetime.now()     # DB
        print("\nTIMING\n--------")
        print(f'Prepare: {dt1} s')
        print(f'\tdt11 (pepare cam): {dt11} s')
        print(f'\t\tdt111 (prepare pix): {dt111} s')
        print(f"\t\t\tdt1111 (prepare): {dt1111} s")
        print(f"\t\t\tdt2222 (compute): {dt2222} s")
        print(f"\t\t\tdt3333 (format):  {dt3333} s")
        print(f"\t\t\tdt4444 (ind_bool):  {dt4444} s")
        print(f'\t\tdt222 (compute): {dt222} s')
        print(f'\t\tdt333 (get poly): {dt333} s')
        print(f'\tdt22 (interp poly): {dt22} s')
        print(f'loop total: {(t2-t1).total_seconds()} s')

    # -------------
    # replace

    if store is True:

        _store(
            coll=coll,
            key_diag=key_diag,
            dvos=dvos,
        )

    return dvos


# ###########################################################
# ###########################################################
#               check
# ###########################################################


def _check(
    coll=None,
    key_diag=None,
    key_mesh=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    visibility=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
    timing=None,
):

    # --------
    # key_diag

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([len(v1['optics']) > 0 for v1 in v0['doptics'].values()])
    ]
    key_diag = ds._generic_check._check_var(
        key_diag, 'key_diag',
        types=str,
        allowed=lok,
    )

    # spectro, is2d
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # doptics
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    dcompute = {
        k0: {'compute': len(v0['optics']) > 0}
        for k0, v0 in doptics.items()
    }

    # --------
    # key_mesh

    wm = coll._which_mesh
    lok = list(coll.dobj.get(wm, {}).keys())
    key_mesh = ds._generic_check._check_var(
        key_mesh, 'key_mesh',
        types=str,
        allowed=lok,
    )

    # -------------------------------------------------
    # ldeti: list of individual camera dict (per pixel)

    for k0, v0 in doptics.items():

        dgeom = coll.dobj['camera'][k0]['dgeom']
        cx, cy, cz = coll.get_camera_cents_xyz(key=k0)
        dvect = coll.get_camera_unit_vectors(key=k0)
        outline = dgeom['outline']
        out0 = coll.ddata[outline[0]]['data']
        out1 = coll.ddata[outline[1]]['data']
        is2d = dgeom['type'] == '2d'
        par = dgeom['parallel']
        dcompute[k0]['shape0'] = cx.shape

        if is2d:
            cx = cx.ravel()
            cy = cy.ravel()
            cz = cz.ravel()

        nd = cx.size

        dcompute[k0]['ldet'] = [
            {
                'cents_x': cx[ii],
                'cents_y': cy[ii],
                'cents_z': cz[ii],
                'outline_x0': out0,
                'outline_x1': out1,
                'nin_x': dvect['nin_x'] if par else dvect['nin_x'][ii],
                'nin_y': dvect['nin_y'] if par else dvect['nin_y'][ii],
                'nin_z': dvect['nin_z'] if par else dvect['nin_z'][ii],
                'e0_x': dvect['e0_x'] if par else dvect['e0_x'][ii],
                'e0_y': dvect['e0_y'] if par else dvect['e0_y'][ii],
                'e0_z': dvect['e0_z'] if par else dvect['e0_z'][ii],
                'e1_x': dvect['e1_x'] if par else dvect['e1_x'][ii],
                'e1_y': dvect['e1_y'] if par else dvect['e1_y'][ii],
                'e1_z': dvect['e1_z'] if par else dvect['e1_z'][ii],
            }
            for ii in range(nd)
        ]

    # -----------
    # res

    if res is None:
        res = 0.01

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
    # visibility

    visibility = ds._generic_check._check_var(
        visibility, 'visibility',
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

    # -----------
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # -----------
    # timing

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    return (
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res,
        margin_par,
        margin_perp,
        visibility,
        verb,
        plot,
        store,
        timing,
    )


# ###########################################################
# ###########################################################
#               Prepare ouput
# ###########################################################


def _prepare_output(
    coll=None,
    key_diag=None,
    shape_samp=None,
    spectro=None,
):

    # -------
    # spectro

    if spectro is True:
        pass

    # -----------
    # non-spectro

    else:

        dvos = {'solid_angle_int': {}}
        for k0 in coll.dobj['diagnostic'][key_diag]['doptics'].keys():
            dgeom = coll.dobj['camera'][k0]['dgeom']
            sh = dgeom['shape']
            shape = tuple(np.r_[shape_samp, sh])
            ref = tuple([None, None] + list(dgeom['ref']))

            # --------
            # slice

            if is2d:
                def sli(ir, iz, ii):
                    pass

            else:
                def sli(ir, iz, ii):
                    pass



            # --------
            # dvos

            dvos['solid_angle_int'][k0] = {
                'data': None,
                'units': 'sr.m',
                'dim': '',
                'quant': '',
                'name': '',
                'ref': ref,
            }

    return dvos


# ###########################################################
# ###########################################################
#               Get cross-section indices
# ###########################################################


def _get_cross_section_indices(
    dsamp=None,
    # polygon
    pcross0=None,
    pcross1=None,
    phor0=None,
    phor1=None,
    margin_poly=None,
    # points
    x0f=None,
    x1f=None,
    sh=None,
):

    # ----------
    # check

    margin_poly = ds._generic_check._check_var(
        margin_poly, 'margin_poly',
        types=float,
        default=0.2,
        sign='>0'
    )

    # ---------------------------
    # add extra margin to pcross

    # get centroid
    center = plg.Polygon(np.array([pcross0, pcross1]).T).center()

    # add margin
    pcross02 = center[0] + (1. + margin_poly) * (pcross0 - center[0])
    pcross12 = center[1] + (1. + margin_poly) * (pcross1 - center[1])

    # define path
    pcross = Path(np.array([pcross02, pcross12]).T)

    # ---------------------------
    # add extra margin to phor

    # get center
    center = plg.Polygon(np.array([phor0, phor1]).T).center()

    # add margin
    phor02 = center[0] + (1. + margin_poly) * (phor0 - center[0])
    phor12 = center[1] + (1. + margin_poly) * (phor1 - center[1])

    # define path
    phor = Path(np.array([phor02, phor12]).T)

    # get ind
    return (
        dsamp['ind']['data']
        & pcross.contains_points(np.array([x0f, x1f]).T).reshape(sh)
    ), phor


# ###########################################################
# ###########################################################
#               Detector
# ###########################################################


def _get_deti(
    coll=None,
    cxi=None,
    cyi=None,
    czi=None,
    dvect=None,
    par=None,
    out0=None,
    out1=None,
    ii=None,
):

    # ------------
    # detector

    if not par:
        msg = "Maybe dvect needs to be flattened?"
        raise Exception(msg)

    det = {
        'cents_x': cxi,
        'cents_y': cyi,
        'cents_z': czi,
        'outline_x0': out0,
        'outline_x1': out1,
        'nin_x': dvect['nin_x'] if par else dvect['nin_x'][ii],
        'nin_y': dvect['nin_y'] if par else dvect['nin_y'][ii],
        'nin_z': dvect['nin_z'] if par else dvect['nin_z'][ii],
        'e0_x': dvect['e0_x'] if par else dvect['e0_x'][ii],
        'e0_y': dvect['e0_y'] if par else dvect['e0_y'][ii],
        'e0_z': dvect['e0_z'] if par else dvect['e0_z'][ii],
        'e1_x': dvect['e1_x'] if par else dvect['e1_x'][ii],
        'e1_y': dvect['e1_y'] if par else dvect['e1_y'][ii],
        'e1_z': dvect['e1_z'] if par else dvect['e1_z'][ii],
    }

    return det


# ###########################################################
# ###########################################################
#               get polygons
# ###########################################################


def _get_polygons(
    x0=None,
    x1=None,
    bool_cross=None,
    res=None,
):

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

    sign = np.sign(cross)
    sign0 = np.mean(sign)
    assert np.all(cross * sign0 >= -1e-12)

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


# ###########################################################
# ###########################################################
#               store
# ###########################################################


def _store(
    coll=None,
    key_diag=None,
    dvos=None,
):

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    for k0, v0 in dvos.items():

        # re-use previous keys
        kpc0, kpc1 = doptics[k0]['vos_pcross']
        kr = coll.ddata[kpc0]['ref'][0]

        # safety check
        if coll.ddata[kpc0]['data'].shape[1:] != v0['pcross0'].shape[1:]:
            msg = "Something is wrong"
            raise Exception(msg)

        coll._dref[kr]['size'] = v0['pcross0'].shape[0]
        coll._ddata[kpc0]['data'] = v0['pcross0']
        coll._ddata[kpc1]['data'] = v0['pcross1']
