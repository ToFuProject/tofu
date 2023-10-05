# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
from matplotlib.path import Path
import datastock as ds


from ..geom import _comp_solidangles
from . import _class8_vos_utilities as _utilities


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos(
    # ressources
    coll=None,
    doptics=None,
    key_cam=None,
    dsamp=None,
    # inputs
    x0u=None,
    x1u=None,
    x0f=None,
    x1f=None,
    x0l=None,
    x1l=None,
    dx0=None,
    dx1=None,
    sh=None,
    res_RZ=None,
    res_phi=None,
    bool_cross=None,
    # user-defined limits
    user_limits=None,
    # parameters
    margin_poly=None,
    config=None,
    visibility=None,
    verb=None,
    # debug
    debug=False,
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
    # unused
    **kwdargs,
):
    """ vos computation for broadband """

    # ---------------
    # prepare polygon


    if timing:
        t00 = dtm.datetime.now()     # DB

    # ----------------
    # user-defined vos

    if user_limits is not None:
        xx, yy, zz, dind, iz, iphi = _vos_points(
            # polygons
            pcross0=user_limits['pcross_user'][0, :],
            pcross1=user_limits['pcross_user'][1, :],
            phor0=user_limits['phor_user'][0, :],
            phor1=user_limits['phor_user'][1, :],
            margin_poly=margin_poly,
            dphi=np.r_[-np.pi, np.pi],
            # sampling
            dsamp=dsamp,
            x0f=x0f,
            x1f=x1f,
            x0u=x0u,
            x1u=x1u,
            res=res_phi,
            dx0=dx0,
            dx1=dx1,
            # shape
            sh=sh,
        )

        shape = coll.dobj['camera'][key_cam]['dgeom']['shape']
        shape = np.r_[0, shape]

    else:

        # get temporary vos
        kpc0, kpc1 = doptics[key_cam]['dvos']['pcross']
        shape = coll.ddata[kpc0]['data'].shape
        pcross0 = coll.ddata[kpc0]['data'].reshape((shape[0], -1))
        pcross1 = coll.ddata[kpc1]['data'].reshape((shape[0], -1))
        kph0, kph1 = doptics[key_cam]['dvos']['phor']
        shapeh = coll.ddata[kph0]['data'].shape
        phor0 = coll.ddata[kph0]['data'].reshape((shapeh[0], -1))
        phor1 = coll.ddata[kph1]['data'].reshape((shapeh[0], -1))

        dphi = doptics[key_cam]['dvos']['dphi']

    # ---------------
    # prepare det

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    par = dgeom['parallel']
    is2d = dgeom['nd'] == '2d'
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
    lphor = []
    lsang = []
    lindr = []
    lindz = []
    npix = coll.dobj['camera'][key_cam]['dgeom']['pix_nb']
    for ii in range(npix):

        # -----------------
        # get volume limits

        if timing:
            t000 = dtm.datetime.now()     # DB

        # get points
        if user_limits is None:

            if np.isnan(pcross0[0, ii]):
                continue

            xx, yy, zz, dind, iz, iphi = _vos_points(
                # polygons
                pcross0=pcross0[:, ii],
                pcross1=pcross1[:, ii],
                phor0=phor0[:, ii],
                phor1=phor1[:, ii],
                margin_poly=margin_poly,
                dphi=dphi[:, ii],
                # sampling
                dsamp=dsamp,
                x0f=x0f,
                x1f=x1f,
                x0u=x0u,
                x1u=x1u,
                res=res_phi,
                dx0=dx0,
                dx1=dx1,
                # shape
                sh=sh,
            )

        if xx is None:
            npts_cross = 0
            lpcross.append((None, None))
            lsang.append(np.zeros((npts_cross,), dtype=float))
            lindr.append(np.zeros((npts_cross,), dtype=float))
            lindz.append(np.zeros((npts_cross,), dtype=float))
            continue

        # re-initialize
        bool_cross[...] = False
        npts_tot = xx.size
        npts_cross = np.sum([v0['iz'].size for v0 in dind.values()])

        dsang_hor = {}
        sang = np.zeros((npts_cross,), dtype=float)
        indr = np.zeros((npts_cross,), dtype=int)
        indz = np.zeros((npts_cross,), dtype=int)

        if verb is True:
            msg = (
                f"\tcam '{key_cam}' pixel {ii+1} / {npix}"
                f"\tnpts in cross_section = {npts_cross}"
                f"\t({npts_tot} total)"
            )
            end = '\n 'if ii == npix - 1 else '\r'
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

        # update cross
        ipt = 0
        for i0, v0 in dind.items():
            for i1 in v0['iz']:
                ind1 = dind[i0]['indrz'] & (iz == i1)
                sang[ipt] = np.sum(out[0, ind1]) * v0['dV']
                indr[ipt] = i0
                indz[ipt] = i1
                bool_cross[i0 + 1, i1 + 1] = sang[ipt] > 0.
                ipt += 1

        # update hor
        for i0, v0 in dind.items():
            dsang_hor[i0] = np.zeros((v0['phi'].size,))
            for i1 in range(v0['phi'].size):
                ind1 = dind[i0]['indrz'] & (iphi == i1)
                dsang_hor[i0][i1] = np.sum(out[0, ind1]) * v0['dV']

        # ----- DEBUG --------
        if debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ipos = out[0, :] > 0
            # ax.scatter(
            #     xx[ipos], yy[ipos],
            #     c=out[0, ipos], s=6, marker='o', vmin=0,
            # )
            # ax.plot(xx[~ipos], yy[~ipos], c='r', marker='x')
            ax.scatter(np.arctan2(yy, xx), out[0, :], c=np.hypot(xx, yy), s=6, marker='.')
        # ----- END DEBUG ----


        # timing
        if timing:
            dt4444 += (dtm.datetime.now() - t0).total_seconds()
            dt1111 += dt1
            dt2222 += dt2
            dt3333 += dt3
            t222 = dtm.datetime.now()     # DB
            dt222 += (t222-t111).total_seconds()

        # -----------------------
        # get pcross and simplify

        if np.any(bool_cross):

            # pcross
            pc0, pc1 = _utilities._get_polygons(
                bool_cross=bool_cross,
                x0=x0l,
                x1=x1l,
                res=np.min(np.atleast_1d(res_RZ)),
            )

            # phor
            # ph0, ph1 = _get_phor(
            #     dind=dind,
            #     xx=xx,
            #     yy=yy,
            #     out=out[0, :],
            #     dsang_hor=dsang_hor,
            #     x0=x0l[:, 0],
            #     res=np.min(np.atleast_1d(res_RZ)),
            # )

            ph0, ph1 = _get_phor2(
                xx=xx,
                yy=yy,
                out=out[0, :],
                res=min(res_RZ[0], res_phi),
            )

        else:
            pc0, pc1 = None, None
            ph0, ph1 = None, None

        # -----------
        # replace

        lpcross.append((pc0, pc1))
        lphor.append((ph0, ph1))
        lsang.append(sang)
        lindr.append(indr)
        lindz.append(indz)

        if timing:
            t333 = dtm.datetime.now()     # DB
            dt333 += (t333-t222).total_seconds()

    # ----------------------------
    # harmonize and reshape pcross

    if timing:
        t22 = dtm.datetime.now()     # DB

    pcross0, pcross1 = _harmonize_reshape_pcross(
        lpcross=lpcross,
        npix=npix,
        is2d=is2d,
        shape=shape[1:],
    )

    phor0, phor1 = _harmonize_reshape_pcross(
        lpcross=lphor,
        npix=npix,
        is2d=is2d,
        shape=shape[1:],
    )

    # --------------------------------------
    # harmonize and reshape sang, indr, indz

    sang, indr, indz = _harmonize_reshape_others(
        lsang=lsang,
        lindr=lindr,
        lindz=lindz,
        npix=npix,
        is2d=is2d,
        shape=shape[1:],
    )

    # ----------------
    # prepare output

    knpts = f'{key_cam}_vos_npts'
    kir = f'{key_cam}_vos_ir'
    kiz = f'{key_cam}_vos_iz'
    ksa = f'{key_cam}_vos_sa'

    ref = tuple(list(coll.dobj['camera'][key_cam]['dgeom']['ref']) + [knpts])

    # ----------------
    # format output

    dref = {
        'npts': {
            'key': knpts,
            'size': indr.shape[-1],
        },
    }

    dout = {
        'pcross0': {
            'data': pcross0,
            'units': 'm',
            'dim': 'distance',
        },
        'pcross1': {
            'data': pcross1,
            'units': 'm',
            'dim': 'distance',
        },
        'phor0': {
            'data': phor0,
            'units': 'm',
            'dim': 'distance',
        },
        'phor1': {
            'data': phor1,
            'units': 'm',
            'dim': 'distance',
        },
        'indr': {
            'key': kir,
            'data': indr,
            'ref': ref,
            'units': '',
            'dim': 'index',
        },
        'indz': {
            'key': kiz,
            'data': indz,
            'ref': ref,
            'units': '',
            'dim': 'index',
        },
        'sang': {
            'key': ksa,
            'data': sang,
            'ref': ref,
            'units': 'sr.m3',
            'dim': 'sang',
        },
    }

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return (
        dout, dref,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


# ###########################################################
# ###########################################################
#               get points
# ###########################################################


def _vos_points(
    # polygons
    pcross0=None,
    pcross1=None,
    phor0=None,
    phor1=None,
    margin_poly=None,
    dphi=None,
    # sampling
    dsamp=None,
    x0f=None,
    x1f=None,
    x0u=None,
    x1u=None,
    res=None,
    dx0=None,
    dx1=None,
    # shape
    sh=None,
):

    # ---------------------
    # get polygons - cross

    # get cross-section polygon with margin
    pc0, pc1 = _utilities._get_poly_margin(
        # polygon
        p0=pcross0,
        p1=pcross1,
        # margin
        margin=margin_poly,
    )

    pcross = Path(np.array([pc0, pc1]).T)

    # ------------
    # get indices

    # indices
    ind = (
        dsamp['ind']['data']
        & pcross.contains_points(np.array([x0f, x1f]).T).reshape(sh)
    )

    # R and Z indices
    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    # ------------
    # get polygons - cross dphi_r

    # get cross-section polygon with margin
    ph0, ph1 = _utilities._get_poly_margin(
        # polygon
        p0=phor0,
        p1=phor1,
        # margin
        margin=margin_poly,
    )

    # ------------
    # get dphi_r

    # phi_r
    dphi_r = _utilities._get_dphi_from_R_phor(
        R=x0u[iru],
        phor0=ph0,
        phor1=ph1,
        phimin=dphi[0],
        phimax=dphi[1],
        res=res,
    )

    # get nphi
    iok = np.all(np.isfinite(dphi_r), axis=0)
    dphi_r = dphi_r[:, iok]
    iru = iru[iok]

    # ------------
    # safety check

    if iru.size == 0:
        return None, None, None, None, None

    # ------------
    # go on

    nphi_r = (
        np.ceil(x0u[iru]*(dphi_r[1, :] - dphi_r[0, :]) / res).astype(int)
        + 1
    )
    ddphi_r = np.diff(dphi_r, axis=0)[0, :] / (nphi_r - 1)

    # ------------
    # get indices

    # get indices
    lind = [ir == i0 for i0 in iru]
    ln = [i0.sum() for i0 in lind]

    indrz = np.concatenate([
        np.tile(i0.nonzero()[0], nphi_r[ii]) for ii, i0 in enumerate(lind)
    ])

    # get phi
    lphi = [
        np.linspace(dphi_r[0, ii], dphi_r[1, ii], nn)
        for ii, nn in enumerate(nphi_r)
    ]
    phi = np.concatenate(tuple([
        np.repeat(phii, ln[ii]) for ii, phii in enumerate(lphi)
    ]))
    iphi = np.concatenate(tuple([
        np.repeat(np.arange(0, nn), ln[ii]) for ii, nn in enumerate(nphi_r)
    ]))

    # -------------
    # derive coords

    # coordinates
    rr = x0u[ir[indrz]]
    xx = rr * np.cos(phi)
    yy = rr * np.sin(phi)
    zz = x1u[iz[indrz]]

    # ----------------
    # get indices dict

    dind = {
        i0: {
            'dV': dx0 * dx1 * x0u[i0] * ddphi_r[ii],
            'iz': np.unique(iz[lind[ii]]),
            'indrz': ir[indrz] == i0,
            'phi': lphi[ii],
        }
        for ii, i0 in enumerate(iru)
    }

    return xx, yy, zz, dind, iz[indrz], iphi


# ###########################################################
# ###########################################################
#               PHOR
# ###########################################################


def _get_phor2(xx=None, yy=None, out=None, res=None, debug=False):

    # boundaries
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()

    # grid
    nx = int(np.ceil((xmax - xmin) / (2 * res)))
    ny = int(np.ceil((ymax - ymin) / (2 * res)))

    xb = np.linspace(xmin - 4*res, xmax + 4*res, nx+4)
    yb = np.linspace(ymin - 4*res, ymax + 4*res, ny+4)

    binned = scpstats.binned_statistic_2d(
        xx,
        yy,
        out,
        bins=(xb, yb),
        statistic='mean',
    ).statistic

    # ------- DEBUG -------
    if debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(xx, yy, c=out, s=6, marker='.')

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(
            binned.T,
            extent=(xb[0], xb[-1], yb[0], yb[-1]),
            aspect='equal',
            interpolation='nearest',
            origin='lower',
        )
    # -----------------------

    # --------------------------
    # get phor in (r, phi) space

    phx, phy = _utilities._get_polygons(
        bool_cross=binned > 0,
        x0=np.repeat(0.5*(xb[1:] + xb[:-1])[:, None], ny+3, axis=1),
        x1=np.repeat(0.5*(yb[1:] + yb[:-1])[None, :], nx+3, axis=0),
        res=2*res,
    )

    return phx, phy


# def _get_phor(dind=None, dsang_hor=None, x0=None, res=None):

#     # ------------
#     # get phi map

#     dphi = np.min([
#         (v0['phi'][1] - v0['phi'][0]) for v0 in dind.values()
#         if v0['phi'].size >= 2
#     ])

#     phi_min = np.min([np.min(v0['phi']) for v0 in dind.values()])
#     phi_max = np.max([np.max(v0['phi']) for v0 in dind.values()])

#     nphi = int(np.ceil((phi_max - phi_min) / dphi))
#     phi = np.linspace(phi_min - dphi, phi_max + dphi, nphi + 2)

#     # --------------
#     # get sang map

#     nr = x0.size
#     bool_hor = np.zeros((nr, nphi + 2), dtype=float)

#     for ii, (i0, v0) in enumerate(dind.items()):

#         bool_hor[i0 + 1, :] = scpinterp.UnivariateSpline(
#             v0['phi'],
#             dsang_hor[i0],
#             w=None,
#             bbox=[None, None],
#             k=1,
#             s=None,
#             ext=0,
#             check_finite=False,
#         )(phi) > 0
#     bool_hor[:, 0] = 0.
#     bool_hor[:, -1] = 0.

#     # ----------------
#     # convert to x, y

#     rf = np.repeat(x0[:, None], nphi+2, axis=1)
#     phif = np.repeat(phi[None, :], nr, axis=0)

#     xf = rf * np.cos(phif)
#     yf = rf * np.sin(phif)

#     xmin, xmax = xf.min(), xf.max()
#     ymin, ymax = yf.min(), yf.max()

#     res = min(res, dphi*x0[0])
#     nx = int(np.ceil((xmax - xmin) / res))
#     ny = int(np.ceil((ymax - ymin) / res))

#     xx = np.linspace(np.min(xf), np.max(xf), nx)
#     yy = np.linspace(np.min(xf), np.max(xf), ny)
#     rr = np.hypot(xx[:, None], yy[None, :])
#     pp = np.arctan2(yy[None, :], xx[:, None])

#     bool_xy = np.zeros((nx, ny), dtype=float)

#     iok = (rr > x0[0]) & (rr < x0[-1])
#     bool_xy[iok] = scpinterp.RectBivariateSpline(
#         x0,
#         phi,
#         bool_hor,
#         kx=1,
#         ky=1,
#         s=0,
#     )(rr[iok], pp[iok], grid=False)

#     # --------------------------
#     # get phor in (r, phi) space

#     phx, phy = _utilities._get_polygons(
#         bool_cross=bool_xy,
#         x0=xx,
#         x1=yy,
#         res=res,
#     )

#     return phx, phy


# ###########################################################
# ###########################################################
#               Pixel
# ###########################################################


# def _vos_pixel(
#     x0=None,
#     x1=None,
#     ind=None,
#     npts=None,
#     dphi=None,
#     deti=None,
#     lap=None,
#     res=None,
#     config=None,
#     visibility=None,
#     # output
#     key_cam=None,
#     sli=None,
#     ii=None,
#     bool_cross=None,
#     sang=None,
#     indr=None,
#     indz=None,
#     # timing
#     timing=None,
#     dt1111=None,
#     dt2222=None,
#     dt3333=None,
#     dt4444=None,
# ):


#     out = _comp_solidangles.calc_solidangle_apertures(
#         # observation points
#         pts_x=xx,
#         pts_y=yy,
#         pts_z=zz,
#         # polygons
#         apertures=lap,
#         detectors=deti,
#         # possible obstacles
#         config=config,
#         # parameters
#         summed=False,
#         visibility=visibility,
#         return_vector=False,
#         return_flat_pts=None,
#         return_flat_det=None,
#         timing=timing,
#     )

#     # ------------
#     # get indices

#     if timing:
#         t0 = dtm.datetime.now()     # DB
#         out, dt1, dt2, dt3 = out

#     ipt = 0
#     for ii, i0 in enumerate(iru):
#         ind0 = irf == i0
#         for i1 in izru[ii]:
#             ind = ind0 & (izf == i1)
#             bool_cross[i0 + 1, i1 + 1] = np.any(out[0, ind] > 0.)
#             sang[ipt] = np.sum(out[0, ind])
#             indr[ipt] = i0
#             indz[ipt] = i1
#             ipt += 1
#     assert ipt == npts

#     # timing
#     if timing:
#         dt4444 += (dtm.datetime.now() - t0).total_seconds()
#         dt1111 += dt1
#         dt2222 += dt2
#         dt3333 += dt3

#         return dt1111, dt2222, dt3333, dt4444
#     else:
#         return


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
#               Harmonize and reshape pcross
# ###########################################################


def _harmonize_reshape_pcross(
    lpcross=None,
    npix=None,
    is2d=None,
    shape=None,
):

    ln = [pp[0].size if pp[0] is not None else 0 for pp in lpcross]
    nmax = np.max(ln)
    sh2 = (nmax, npix)
    pcross0 = np.full(sh2, np.nan)
    pcross1 = np.full(sh2, np.nan)
    for ii, nn in enumerate(ln):

        if nn == 0:
            continue

        if nmax > nn:
            ind = np.r_[0, np.linspace(0.1, 0.9, nmax - nn), np.arange(1, nn)]
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

    # -------
    # reshape

    if is2d:
        newsh = tuple(np.r_[nmax, shape])
        pcross0 = pcross0.reshape(newsh)
        pcross1 = pcross1.reshape(newsh)

    return pcross0, pcross1


# ###########################################################
# ###########################################################
#               Harmonize and reshape others
# ###########################################################


def _harmonize_reshape_others(
    lsang=None,
    lindr=None,
    lindz=None,
    npix=None,
    is2d=None,
    shape=None,
):

    lnpts = [sa.size for sa in lsang]
    nmax = np.max(lnpts)

    sang = np.full((npix, nmax), np.nan)
    indr = -np.ones((npix, nmax), dtype=int)
    indz = -np.ones((npix, nmax), dtype=int)
    for ii, sa in enumerate(lsang):
        sang[ii, :lnpts[ii]] = sa
        indr[ii, :lnpts[ii]] = lindr[ii]
        indz[ii, :lnpts[ii]] = lindz[ii]

    # -------
    # reshape

    if is2d:
        newsh = tuple(np.r_[shape, nmax])
        sang = sang.reshape(newsh)
        indr = indr.reshape(newsh)
        indz = indz.reshape(newsh)

    return sang, indr, indz