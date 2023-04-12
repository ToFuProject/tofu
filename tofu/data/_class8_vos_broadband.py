# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np


from ..geom import _comp_solidangles


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos_main(
    # ressources
    coll=None,
    doptics=None,
    key_cam=None,
    dsamp=None,
    # inputs
    x0f=None,
    x1f=None,
    sh=None,
    res=None,
    bool_cross=None,
    shape=None,
    # parameters
    margin_poly=None,
    config=None,
    visibility=None,
    # timing
    timing=None,
    dt11=None,
    dt111=None,
    dt1111=None,
    dt2222=None,
    dt3333=None,
    dt4444=None,
    dt222=None,
    dt333=None,
    dt22=None,
):

    # ---------------
    # prepare polygon

    if timing:
        t00 = dtm.datetime.now()     # DB

    # get temporary vos
    kpc0, kpc1 = doptics[key_cam]['vos_pcross']
    shape = coll.ddata[kpc0]['data'].shape
    pcross0 = coll.ddata[kpc0]['data'].reshape((shape[0], -1))
    pcross1 = coll.ddata[kpc1]['data'].reshape((shape[0], -1))
    kph0, kph1 = doptics[key_cam]['vos_phor']
    shapeh = coll.ddata[kph0]['data'].shape
    phor0 = coll.ddata[kph0]['data'].reshape((shapeh[0], -1))
    phor1 = coll.ddata[kph1]['data'].reshape((shapeh[0], -1))

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

    lap = coll.get_optics_as_input_solid_angle(
        keys=doptics[key_cam]['optics'],
    )

    if timing:
        t11 = dtm.datetime.now()     # DB
        dt11 += (t11-t00).total_seconds()

    # -----------
    # loop on pix

    lpcross = []
    for ii in range(pcross0.shape[1]):

        # -----------------
        # get volume limits

        if timing:
            t000 = dtm.datetime.now()     # DB

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

        # verb
        if verb is True:
            msg = (
                f"\tcam '{key_cam}' pixel {ii+1} / {pcross0.shape[1]}\t"
                f"npts in cross_section = {ind.sum()}   "
            )
            end = '\n 'if ii == pcross0.shape[1] - 1 else '\r'
            print(msg, end=end, flush=True)

        # ---------------------
        # loop on volume points


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

        if timing:
            t111 = dtm.datetime.now()     # DB
            dt111 += (t111-t000).total_seconds()

        # compute
        out = _vos_pixel(
            x0=x0u,
            x1=x1u,
            ind=ind,
            dphi=dphi[:, ii],
            deti=deti,
            lap=lap,
            res=res,
            config=config,
            visibility=visibility,
            # output
            key_cam=key_cam,
            dvos=dvos,
            sli=None,
            ii=ii,
            bool_cross=bool_cross,
            path_hor=path_hor,
            # timing
            timing=timing,
            dt1111=dt1111,
            dt2222=dt2222,
            dt3333=dt3333,
            dt4444=dt4444,
        )

        if timing:
            dt1111, dt2222, dt3333, dt4444 = out
            t222 = dtm.datetime.now()     # DB
            dt222 += (t222-t111).total_seconds()

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

        if timing:
            t333 = dtm.datetime.now()     # DB
            dt333 += (t333-t222).total_seconds()

    # ----------------
    # harmonize pcross

    if timing:
        t22 = dtm.datetime.now()     # DB

    ln = [pp[0].size if pp[0] is not None else 0 for pp in lpcross]
    nmax = np.max(ln)
    sh2 = (nmax, pcross0.shape[1])
    pcross0 = np.full(sh2, np.nan)
    pcross1 = np.full(sh2, np.nan)
    for ii, nn in enumerate(ln):

        if nn == 0:
            continue

        if nmax > nn:
            ind = np.r_[0, np.linspace(0.1, 0.9, nmax-nn), np.arange(1, nn)]
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

        else:
            pcross0[:, ii] = lpcross[ii][0]
            pcross1[:, ii] = lpcross[ii][1]

    # -------------
    # reshape

    if is2d:
        newsh = tuple(np.r_[nmax, shape])
        pcross0 = pcross0.reshape(newsh)
        pcross1 = pcross1.reshape(newsh)

    dout = {
        'pcross0': pcross0,
        'pcross1': pcross1,
    }

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return dout


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos_pixel(
    x0=None,
    x1=None,
    ind=None,
    dphi=None,
    deti=None,
    lap=None,
    res=None,
    config=None,
    visibility=None,
    # output
    key_cam=None,
    dvos=None,
    sli=None,
    ii=None,
    bool_cross=None,
    path_hor=None,
    # timing
    timing=None,
    dt1111=None,
    dt2222=None,
    dt3333=None,
    dt4444=None,
):

    # --------------------------
    # prepare points and indices

    ir, iz = ind.nonzero()
    iru = np.unique(ir)
    izru = [iz[ir == i0] for i0 in iru]

    nphi = np.ceil(x0[ir]*(dphi[1] - dphi[0]) / res).astype(int)

    irf = np.repeat(ir, nphi)
    izf = np.repeat(iz, nphi)
    phi = np.concatenate(tuple([
        np.linspace(dphi[0], dphi[1], nn) for nn in nphi
    ]))

    xx = x0[irf] * np.cos(phi)
    yy = x0[irf] * np.sin(phi)
    zz = x1[izf]

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
        visibility=visibility,
        return_vector=False,
        return_flat_pts=None,
        return_flat_det=None,
        timing=timing,
    )

    # ------------
    # get indices

    if timing:
        t0 = dtm.datetime.now()     # DB
        out, dt1, dt2, dt3 = out

    for ii, i0 in enumerate(iru):
        ind0 = irf == i0
        for i1 in izru[ii]:
            ind = ind0 & (izf == i1)
            bool_cross[i0 + 1, i1 + 1] = np.any(out[0, ind] > 0.)

    # timing
    if timing:
        dt4444 += (dtm.datetime.now() - t0).total_seconds()
        dt1111 += dt1
        dt2222 += dt2
        dt3333 += dt3

        return dt1111, dt2222, dt3333, dt4444
    else:
        return
