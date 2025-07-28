# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.stats as scpstats
import matplotlib.pyplot as plt


from ..geom import _comp_solidangles
from . import _class8_vos_utilities as _utilities


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos(
    func_RZphi_from_ind=None,
    func_ind_from_domain=None,
    # ressources
    coll=None,
    doptics=None,
    key_diag=None,
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
    # keep
    dkeep=None,
    return_vector=None,
    # parameters
    margin_poly=None,
    config=None,
    visibility=None,
    verb=None,
    # debug
    debug=None,
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
    # ----------------

    if user_limits is not None:

        if user_limits.get('pcross_user') is not None:

            # margin poly cross
            pc0, pc1 = _utilities._get_poly_margin(
                # polygon
                p0=user_limits['pcross_user'][0, :],
                p1=user_limits['pcross_user'][1, :],
                # margin
                margin=margin_poly,
            )

            # margin poly hor
            ph0, ph1 = _utilities._get_poly_margin(
                # polygon
                p0=user_limits['phor_user'][0, :],
                p1=user_limits['phor_user'][1, :],
                # margin
                margin=margin_poly,
            )

            # indices
            ind3dr, ind3dz, ind3dphi = func_ind_from_domain(
                pcross0=pc0,
                pcross1=pc1,
                phor0=ph0,
                phor1=ph1,
            )

            # coordinates
            rr, zz, pp, dV = func_RZphi_from_ind(ind3dr, ind3dz, ind3dphi)

            shape = coll.dobj['camera'][key_cam]['dgeom']['shape']
            shape = np.r_[0, shape]

        elif user_limits.get('Dphi') is not None:
            # get temporary vos
            kpc0, kpc1 = doptics[key_cam]['dvos']['pcross']
            shape = coll.ddata[kpc0]['data'].shape
            pcross0 = coll.ddata[kpc0]['data']
            pcross1 = coll.ddata[kpc1]['data']

            # phor
            phor0 = user_limits['phor0'][key_cam]
            phor1 = user_limits['phor1'][key_cam]

        else:
            msg = (
                "Something weird with pcross0:\n"
                f"user_limits: {user_limits}\n"
            )
            raise Exception(msg)

    else:

        # get temporary vos
        kpc0, kpc1 = doptics[key_cam]['dvos']['pcross']
        shape = coll.ddata[kpc0]['data'].shape
        pcross0 = coll.ddata[kpc0]['data']
        pcross1 = coll.ddata[kpc1]['data']

        # phor
        kph0, kph1 = doptics[key_cam]['dvos']['phor']
        phor0 = coll.ddata[kph0]['data']
        phor1 = coll.ddata[kph1]['data']

    # pinhole?
    pinhole = doptics[key_cam]['pinhole']

    # ---------------
    # prepare det
    # ---------------

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    par = dgeom['parallel']
    cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)
    dvect = coll.get_camera_unit_vectors(key=key_cam)
    outline = dgeom['outline']
    out0 = coll.ddata[outline[0]]['data']
    out1 = coll.ddata[outline[1]]['data']

    # -----------
    # prepare lap
    # -----------

    optics = doptics[key_cam]['optics']
    dap0 = coll.get_optics_as_input_solid_angle(keys=optics)
    if pinhole is False:
        paths = doptics[key_cam]['paths']
    else:
        dap = dap0

    if timing:
        t11 = dtm.datetime.now()     # DB
        dt11 += (t11-t00).total_seconds()

    # ----------------
    # loop on pixels
    # ----------------

    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']
    npix = int(np.prod(shape_cam))
    douti = {}
    indok = np.ones(shape_cam, dtype=bool)
    vectx, vecty, vectz = None, None, None
    for ii, ind in enumerate(np.ndindex(shape_cam)):

        debugi = debug if isinstance(debug, bool) else debug(ind)

        # -----------------
        # slices

        sli_poly = ind + (slice(None),)
        sli_poly0 = ind + (0,)

        # -----------------
        # get volume limits

        if timing:
            t000 = dtm.datetime.now()     # DB

        # get points
        if user_limits is None or user_limits.get('pcross_user') is None:

            if np.isnan(pcross0[sli_poly0]):
                indok[ind] = False
                continue

            # margin poly cross
            pc0, pc1 = _utilities._get_poly_margin(
                # polygon
                p0=pcross0[sli_poly],
                p1=pcross1[sli_poly],
                # margin
                margin=margin_poly,
            )

            # margin poly hor
            ph0, ph1 = _utilities._get_poly_margin(
                # polygon
                p0=phor0[sli_poly],
                p1=phor1[sli_poly],
                # margin
                margin=margin_poly,
            )

            ind3dr, ind3dz, ind3dphi = func_ind_from_domain(
                pcross0=pc0,
                pcross1=pc1,
                phor0=ph0,
                phor1=ph1,
                debug=debugi,
                debug_msg=f"kcam = {key_cam}, ind = {ind}",
            )

            rr, zz, pp, dV = func_RZphi_from_ind(ind3dr, ind3dz, ind3dphi)

        if rr is None:
            indok[ind] = False
            continue

        xx = rr * np.cos(pp)
        yy = rr * np.sin(pp)

        # --------------
        # re-initialize

        npts_tot = xx.size
        npts_cross = np.unique([ind3dr, ind3dz], axis=1).shape[1]

        # safety check
        assert npts_tot >= npts_cross

        if npts_cross == 0:
            assert npts_tot == 0
            indok[ind] = False
            continue

        # dsang_hor = {}
        # sang_cross = np.zeros((npts_cross,), dtype=float)
        # indr_cross = np.zeros((npts_cross,), dtype=int)
        # indz_cross = np.zeros((npts_cross,), dtype=int)
        # if return_vector is True:
            # vectx_cross = np.zeros((npts_cross,), dtype=float)
            # vecty_cross = np.zeros((npts_cross,), dtype=float)
            # vectz_cross = np.zeros((npts_cross,), dtype=float)

        if verb is True:
            msg = (
                f"\tcam '{key_cam}' pixel {ii+1} / {npix}"
                f"\tnpts in cross_section = {npts_cross}"
                f"\t({npts_tot} total)"
            )
            end = '\n 'if ii == npix - 1 else '\r'
            print(msg, end=end, flush=True)

        # --------------------------------
        # get formatted detector geometry

        # get detector / aperture
        deti = _get_deti(
            coll=coll,
            cxi=cx[ind],
            cyi=cy[ind],
            czi=cz[ind],
            dvect=dvect,
            par=par,
            out0=out0,
            out1=out1,
            ind=ind,
        )

        if timing:
            t111 = dtm.datetime.now()     # DB
            dt111 += (t111-t000).total_seconds()

        # --------------------------------
        # get pixel-specific apertures if not pinhole

        if pinhole is False:
            sli_path = tuple(list(ind) + [slice(None)])
            iop = np.nonzero(paths[sli_path])[0]
            dap = {optics[ii]: dap0[optics[ii]] for ii in iop}

        # -------------------
        # compute solid angle

        # compute
        out = _comp_solidangles.calc_solidangle_apertures(
            # observation points
            pts_x=xx,
            pts_y=yy,
            pts_z=zz,
            # polygons
            apertures=dap,
            detectors=deti,
            # possible obstacles
            config=config,
            # parameters
            summed=False,
            visibility=visibility,
            return_vector=return_vector,
            return_flat_pts=None,
            return_flat_det=None,
            timing=timing,
        )

        # ------------
        # extract

        if timing is True:
            t0 = dtm.datetime.now()     # DB
            if len(out) > 4:
                sang, vectx, vecty, vectz = out[:4]
                dt1, dt2, dt3 = out[4:]
            else:
                sang, dt1, dt2, dt3 = out
        else:
            if isinstance(out, tuple):
                sang, vectx, vecty, vectz = out
            else:
                sang = out

        # ----------------------
        # keep only valid points

        # only one detector
        sang = sang[0, ...]
        iok = np.isfinite(sang) & (sang > 0.)
        if not np.any(iok):
            indok[ind] = False
            continue

        dtemp_3d = {
            'indr_3d': ind3dr[iok],
            'indz_3d': ind3dz[iok],
            'indphi_3d': ind3dphi[iok],
            'sang_3d': sang[iok],
            'dV_3d': dV[iok],
        }

        if vectx is not None:
            dtemp_3d.update({
                'vectx_3d': vectx[0, iok],
                'vecty_3d': vecty[0, iok],
                'vectz_3d': vectz[0, iok],
            })

        # dtemp_cross
        dtemp_cross = _get_crosshor_from_3d_single_det(
            lind=[dtemp_3d['indr_3d'], dtemp_3d['indz_3d']],
            proj='cross',
            **dtemp_3d,
        )

        # dtemp_hor
        dtemp_hor = _get_crosshor_from_3d_single_det(
            lind=[dtemp_3d['indr_3d'], dtemp_3d['indphi_3d']],
            proj='hor',
            **dtemp_3d,
        )

        # ------------
        # get indices

        douti[ind] = {}
        if dkeep['3d'] is True:
            douti[ind].update(**dtemp_3d)

        if dkeep['cross'] is True:
            douti[ind].update(**dtemp_cross)

        if dkeep['hor'] is True:
            douti[ind].update(**dtemp_hor)

        # ----- DEBUG --------
        if debugi:
            fig = plt.figure()
            fig.suptitle(f"pixel ind = {ind}", size=14, fontweight='bold')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_xlabel("phi (deg)", size=12)
            ax.set_ylabel("solid angle (sr)", size=12)
            # ipos = out[0, :] > 0
            # ax.scatter(
            #     xx[ipos], yy[ipos],
            #     c=out[0, ipos], s=6, marker='o', vmin=0,
            # )
            # ax.plot(xx[~ipos], yy[~ipos], c='r', marker='x')
            ax.scatter(
                np.arctan2(yy, xx) * 180/np.pi,
                out[0, :],
                c=np.hypot(xx, yy),
                s=6,
                marker='.',
            )
            raise Exception()
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
        # get pcross and phor

        bool_cross[...] = False
        sli = (dtemp_cross['indr_cross'] + 1, dtemp_cross['indz_cross'] + 1)
        bool_cross[sli] = True

        # pcross
        (
            douti[ind]['pcross0'], douti[ind]['pcross1'],
        ) = _utilities._get_polygons(
            bool_cross=bool_cross,
            x0=x0l,
            x1=x1l,
            res=np.min(np.atleast_1d(res_RZ)),
        )

        rr_hor, _, pp_hor, _ = func_RZphi_from_ind(
            dtemp_hor['indr_hor'],
            None,
            dtemp_hor['indphi_hor'],
        )
        douti[ind]['phor0'], douti[ind]['phor1'] = _get_phor2(
            xx=rr_hor*np.cos(pp_hor),
            yy=rr_hor*np.sin(pp_hor),
            out=dtemp_hor['sang_hor'],
            res=min(res_RZ[0], res_phi),
        )

        if timing:
            t333 = dtm.datetime.now()     # DB
            dt333 += (t333-t222).total_seconds()

    # ----------------------------
    # harmonize and reshape pcross
    # ----------------------------

    if timing:
        t22 = dtm.datetime.now()     # DB

    ddata, dref = _utilities._harmonize_reshape(
        douti=douti,
        indok=indok,
        key_diag=key_diag,
        key_cam=key_cam,
        ref_cam=coll.dobj['camera'][key_cam]['dgeom']['ref'],
    )

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return (
        ddata, dref,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


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
    ind=None,
):

    # ------------
    # detector

    det = {
        'cents_x': cxi,
        'cents_y': cyi,
        'cents_z': czi,
        'outline_x0': out0,
        'outline_x1': out1,
        'nin_x': dvect['nin_x'] if par else dvect['nin_x'][ind],
        'nin_y': dvect['nin_y'] if par else dvect['nin_y'][ind],
        'nin_z': dvect['nin_z'] if par else dvect['nin_z'][ind],
        'e0_x': dvect['e0_x'] if par else dvect['e0_x'][ind],
        'e0_y': dvect['e0_y'] if par else dvect['e0_y'][ind],
        'e0_z': dvect['e0_z'] if par else dvect['e0_z'][ind],
        'e1_x': dvect['e1_x'] if par else dvect['e1_x'][ind],
        'e1_y': dvect['e1_y'] if par else dvect['e1_y'][ind],
        'e1_z': dvect['e1_z'] if par else dvect['e1_z'][ind],
    }

    return det


# ###########################################################
# ###########################################################
#           Get cross from 3d
# ###########################################################


def _get_crosshor_from_3d_single_det(
    lind=None,
    proj=None,
    # always
    sang_3d=None,
    dV_3d=None,
    # vect
    vectx_3d=None,
    vecty_3d=None,
    vectz_3d=None,
    # unused
    **kwdargs,
):

    # ------------
    # keys
    # -----------

    if vectx_3d is None:
        lkvect = []
    else:
        lkvect = [
            (vectx_3d, f'vectx_{proj}'),
            (vecty_3d, f'vecty_{proj}'),
            (vectz_3d, f'vectz_{proj}'),
        ]
    zphi = 'z' if proj == 'cross' else 'phi'

    # ------------
    # get cross
    # -----------

    iptsu = np.unique([lind[0], lind[1]], axis=1)
    npts = iptsu.shape[1]

    ksang = f'sang_{proj}'
    kdV = f'dV_{proj}'
    kndV = f"ndV_{proj}"
    dout = {
        f'indr_{proj}': iptsu[0, :],
        f'ind{zphi}_{proj}': iptsu[1, :],
        ksang: np.full((npts,), np.nan),
        kdV: np.full((npts,), np.nan),
        kndV: np.full((npts,), np.nan),
    }
    for (_, k1) in lkvect:
        dout[k1] = np.full((npts,), np.nan)

    # ------------
    # fill
    # -----------

    for ii, (ir, izphi) in enumerate(iptsu.T):
        ind = (lind[0] == ir) & (lind[1] == izphi)

        dout[ksang][ii] = np.sum(sang_3d[ind])
        dout[kdV][ii] = dV_3d[ind.nonzero()[0][0]]
        dout[kndV][ii] = np.sum(ind)

        for (vect, k1) in lkvect:
            dout[k1][ii] = np.sum(vect[ind] * sang_3d[ind]) / dout[ksang][ii]

    return dout
