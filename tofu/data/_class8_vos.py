# -*- coding: utf-8 -*-


import numpy as np
from matplotlib.path import Path
import datastock as ds
import bsplines2d as bs2
from contourpy import contour_generator
from scipy.spatial import ConvexHull


import Polygon as plg


from ..geom import _comp_solidangles


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
    convex=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
):

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
        verb,
        plot,
        store,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_mesh=key_mesh,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        verb=verb,
        plot=plot,
        store=store,
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

    # --------------
    # prepare optics

    dvos = {}
    for key_cam, v0 in dcompute.items():

        # ---------------
        # prepare polygon

        # get temporary vos
        kpc0, kpc1 = doptics[key_cam]['vos_pcross']
        shape = coll.ddata[kpc0]['data'].shape
        pcross0 = coll.ddata[kpc0]['data'].reshape((shape[0], -1))
        pcross1 = coll.ddata[kpc1]['data'].reshape((shape[0], -1))
        kph0, kph1 = doptics[key_cam]['vos_phor']
        phor0 = coll.ddata[kpc0]['data'].reshape((shape[0], -1))
        phor1 = coll.ddata[kpc1]['data'].reshape((shape[0], -1))

        dphi = doptics[key_cam]['vos_dphi']

        # ---------------
        # prepare det

        dgeom = coll.dobj['camera'][key_cam]['dgeom']
        par = dgeom['parallel']
        is2d = dgeom['type'] == '2d'
        cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)
        dvect = coll.get_camera_unit_vectors(key=key_cam)
        outline = dgeom['outline']
        out0 = coll.ddata[outline[0]]['data']
        out1 = coll.ddata[outline[1]]['data']

        if is2d:
            cx = cx.ravel()
            cy = cy.ravel()
            cz = cz.ravel()

        # -----------
        # prepare lap

        if spectro is False:
            lap = coll.get_optics_as_input_solid_angle(
                keys=doptics[key_cam]['optics'],
            )

        # -----------
        # loop on pix

        lpcross = []
        for ii in range(pcross0.shape[1]):

            if verb is True:
                msg = f"\tcam '{key_cam}' pixel {ii+1} / {pcross0.shape[1]}"
                end = '\n' if ii == pcross0.shape[1] - 1 else '\r'
                print(msg, end=end, flush=True)

            # -----------------
            # get volume limits

            if np.isnan(pcross0[0, ii]):
                continue

            # get cross-section polygon
            ind, path_hor = _get_cross_section_indices(
                dsamp=dsamp,
                # polygon
                pcross0=pcross0[:, ii],
                pcross1=pcross1[:, ii],
                phor0=phor0[:, ii],
                phor1=phor1[:, ii],
                margin_poly=margin_poly,
                # points
                x0f=x0f,
                x1f=x1f,
                sh=sh,
            )

            # re-initialize
            bool_cross[...] = False

            # ---------------------
            # loop on volume points


            if spectro:
                dvos[key_cam] = _vos_spectro(
                    x0=x0,
                    x1=x1,
                    ind=ind,
                    dphi=dphi[:, ii],
                )

            else:
                # get detector / aperture
                deti = _get_deti(
                    coll=coll,
                    cxi=cx[ii],
                    cyi=cy[ii],
                    czi=cz[ii],
                    dvect=dvect,
                    par=par,
                    out0=out0,
                    out1=out1,
                    ii=ii,
                )

                # compute
                _vos_broadband(
                    x0=x0u,
                    x1=x1u,
                    ind=ind,
                    dphi=dphi[:, ii],
                    deti=deti,
                    lap=lap,
                    res=res,
                    config=config,
                    # output
                    key_cam=key_cam,
                    dvos=dvos,
                    sli=None,
                    ii=ii,
                    bool_cross=bool_cross,
                    path_hor=path_hor,
                )

            # -----------------------
            # get pcross and simplify

            if np.any(bool_cross):
                pc0, pc1 = _get_polygons(
                    bool_cross=bool_cross,
                    x0=x0l,
                    x1=x1l,
                    res=res,
                )
            else:
                pc0, pc1 = None, None

            # -----------
            # replace

            lpcross.append((pc0, pc1))

    # ----------------
    # harmonize pcross

    ln = [pp[0].size if pp[0] is not None else 0 for pp in pc0]
    nmax = np.max(ln)
    pcross0 = np.full((nmax, ), np.nan)
    pcross1 = np.full((nmax, ), np.nan)
    for ii, nn in enumerate(ln):

        if nn == 0:
            continue

        pcross0[:, ii] = scpinterp.interp1d(
            range(0, nn),
            lpcross[ii][0],
            kind='linear',
        )(ind)

        pcross1[:, ii] = scpinterp.interp1d(
            range(0, nn),
            lpcross[ii][1],
            kind='linear',
        )(ind)

    # -------------
    # replace

    if store is True:

        import pdb; pdb.set_trace() # DB
        kr = None
        coll.remove_ref()
        coll.add_ref()

        coll.add_data()
        coll.add_data()

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
    check=None,
    verb=None,
    plot=None,
    store=None,
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
        verb,
        plot,
        store,
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
#               Broadband
# ###########################################################


def _vos_broadband(
    x0=None,
    x1=None,
    ind=None,
    dphi=None,
    deti=None,
    lap=None,
    res=None,
    config=None,
    # output
    key_cam=None,
    dvos=None,
    sli=None,
    ii=None,
    bool_cross=None,
    path_hor=None,
):

    # -----------------
    # loop on (r, z) points

    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    for i0 in iru:

        nphi = int(np.ceil(x0[i0]*(dphi[1] - dphi[0]) / res))
        phi = np.linspace(dphi[0], dphi[1], nphi)
        xx = x0[i0] * np.cos(phi)
        yy = x0[i0] * np.sin(phi)

        # only if in phor
        ihor = path_hor.contains_points(np.array([xx, yy]).T)
        if not np.any(ihor):
            continue

        xx = xx[ihor]
        yy = yy[ihor]
        zz = np.full((ihor.sum(),), np.nan)

        for i1 in iz[ir == i0]:

            zz[:] = x1[i1]

            out = _comp_solidangles.calc_solidangle_apertures(
                # observation points
                pts_x=xx,
                pts_y=yy,
                pts_z=zz,
                # polygons
                apertures=lap,
                detectors=deti,
                # possible obstacles
                config=config,
                # parameters
                summed=False,
                visibility=True,
                return_vector=False,
                return_flat_pts=None,
                return_flat_det=None,
            )

            # dvos[key_cam]['sang_int'][ii] = out
            # dvos[key_cam]['ind'][ii] = sli(ir, iz, ii)
            bool_cross[i0 + 1, i1 + 1] = np.any(out > 0.)

    return

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

    # get hull
    convh = ConvexHull(np.array([c0, c1]).T)
    indh = convh.vertices
    ch0 = c0[indh]
    ch1 = c1[indh]
    nh = indh.size

    # keep egdes that match res
    lp = []
    for ii, ih in enumerate(indh):

        # ind of points in between
        i1 = indh[(ii + 1) % nh]
        if i1 > ih:
            ind = np.arange(ih + 1, i1)
        else:
            import pdb; pdb.set_trace()     # DB
            ind = np.arange(ih)

        # get distances
        x0 = c0[ind]
        x1 = c1[ind]
        dist =

        if np.all(dist < 0.8*res):
            lind.append((ih, i1))
        else:
            lind.append(ind)

    import pdb; pdb.set_trace()     # DB

    # ------------------------------------
    # point by point on remaining segments

    return c0[iok], c1[iok]
