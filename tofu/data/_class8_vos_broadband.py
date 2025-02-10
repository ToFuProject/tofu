# -*- coding: utf-8 -*-


import itertools as itt


import datetime as dtm      # DB
import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
from matplotlib.path import Path
import matplotlib.pyplot as plt
# import datastock as ds


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
    # keep
    keep3d=None,
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
            xx, yy, zz, dind, ir, iz, iphi, dV = _vos_points(
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

        elif user_limits.get('Dphi') is not None:
            # get temporary vos
            kpc0, kpc1 = doptics[key_cam]['dvos']['pcross']
            shape = coll.ddata[kpc0]['data'].shape
            pcross0 = coll.ddata[kpc0]['data']
            pcross1 = coll.ddata[kpc1]['data']

            # phor
            phor0 = user_limits['phor0'][key_cam]
            phor1 = user_limits['phor1'][key_cam]
            dphi = user_limits['dphi'][key_cam]

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
        dphi = doptics[key_cam]['dvos']['dphi']

    # pinhole?
    pinhole = doptics[key_cam]['pinhole']

    # ---------------
    # prepare det
    # ---------------

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    par = dgeom['parallel']
    is2d = dgeom['nd'] == '2d'
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

    # -----------------
    # initialize lists
    # -----------------

    (
        # common
        lpcross, lphor, lsang_cross, lindr_cross, lindz_cross,
        # keep3d
        lindr_3d, lindz_3d, lphi_3d, lsang_3d,
        # return_vector
        lang_tor_cross, lang_pol_cross, lvectx, lvecty, lvectz
    ) = _initialize_lists(
        return_vector=return_vector,
        keep3d=keep3d,
    )

    # ----------------
    # loop on pixels
    # ----------------

    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']
    npix = int(np.prod(shape_cam))
    linds = [range(ss) for ss in shape_cam]
    for ii, ind in enumerate(itt.product(*linds)):

        debugi = debug if isinstance(debug, bool) else debug(ind)

        # -----------------
        # slices

        sli_poly = tuple([slice(None)] + list(ind))
        sli_poly0 = tuple([0] + list(ind))

        # -----------------
        # get volume limits

        if timing:
            t000 = dtm.datetime.now()     # DB

        # get points
        if user_limits is None or user_limits.get('pcross_user') is None:

            if np.isnan(pcross0[sli_poly0]):
                pts_cross = np.zeros((0,), dtype=float)
                lpcross.append((None, None))
                lphor.append((None, None))
                lsang_cross.append(pts_cross)
                lindr_cross.append(pts_cross)
                lindz_cross.append(pts_cross)
                if return_vector is True:
                    lang_pol_cross.append(pts_cross)
                    lang_tor_cross.append(pts_cross)
                continue

            xx, yy, zz, dind, ir, iz, iphi, dV = _vos_points(
                # polygons
                pcross0=pcross0[sli_poly],
                pcross1=pcross1[sli_poly],
                phor0=phor0[sli_poly],
                phor1=phor1[sli_poly],
                margin_poly=margin_poly,
                dphi=dphi[sli_poly],
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
                # debug
                debug=debugi,
                ii=ii,
                ind=ind,
            )

        if xx is None:
            pts_cross = np.zeros((0,), dtype=float)
            lpcross.append((None, None))
            lphor.append((None, None))
            lsang_cross.append(pts_cross)
            lindr_cross.append(pts_cross)
            lindz_cross.append(pts_cross)
            if return_vector is True:
                lang_pol_cross.append(pts_cross)
                lang_tor_cross.append(pts_cross)
            continue

        # --------------
        # re-initialize

        bool_cross[...] = False
        npts_tot = xx.size
        npts_cross = np.sum([v0['iz'].size for v0 in dind.values()])

        dsang_hor = {}
        sang_cross = np.zeros((npts_cross,), dtype=float)
        indr_cross = np.zeros((npts_cross,), dtype=int)
        indz_cross = np.zeros((npts_cross,), dtype=int)
        if return_vector is True:
            ang_pol_cross = np.zeros((npts_cross,), dtype=float)
            ang_tor_cross = np.zeros((npts_cross,), dtype=float)

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

        if isinstance(out, tuple):
            out, vectx, vecty, vectz = out
        else:
            vectx, vecty, vectz = None, None, None

        # ------------
        # get indices

        if timing:
            t0 = dtm.datetime.now()     # DB
            out, dt1, dt2, dt3 = out

        # update cross-section
        ipt = 0
        for i0, v0 in dind.items():
            for i1 in v0['iz']:
                ind1 = dind[i0]['indrz'] & (iz == i1)
                totii = np.sum(out[0, ind1]) * v0['dV']
                sang_cross[ipt] = totii
                indr_cross[ipt] = i0
                indz_cross[ipt] = i1
                bool_cross[i0 + 1, i1 + 1] = sang_cross[ipt] > 0.
                if vectx is not None:
                    tor = np.arctan2(yy[ind1], xx[ind1])
                    ang_pol = np.arctan2(
                        vectz[0, ind1],
                        vectx[0, ind1]*np.cos(tor) + vecty[0, ind1]*np.sin(tor),
                    )
                    ang_tor = np.arccos(
                        vectx[0, ind1] * (-np.sin(tor))
                        + vecty[0, ind1] * np.cos(tor)
                    )
                    ang_pol_cross[ipt] = np.sum(out[0, ind1] * ang_pol) / totii
                    ang_tor_cross[ipt] = np.sum(out[0, ind1] * ang_tor) / totii
                ipt += 1

        # update horizontal
        for i0, v0 in dind.items():
            dsang_hor[i0] = np.zeros((v0['phi'].size,))
            for i1 in range(v0['phi'].size):
                ind1 = dind[i0]['indrz'] & (iphi == i1)
                dsang_hor[i0][i1] = np.sum(out[0, ind1]) * v0['dV']

        # update 3d
        if keep3d is True and np.any(bool_cross):
            indsa = out[0, :] > 0.

            indr_3d = ir[indsa]
            indz_3d = iz[indsa]
            phi_3d = np.arctan2(yy[indsa], xx[indsa])
            sang_3d = out[0, indsa] * dV[indsa]

            if vectx is not None:
                vx = vectx[0, indsa]
                vy = vecty[0, indsa]
                vz = vectz[0, indsa]

        # ----- DEBUG --------
        if debugi:
            import matplotlib.pyplot as plt
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
        # get pcross and simplify

        if np.any(bool_cross):

            # pcross
            pc0, pc1 = _utilities._get_polygons(
                bool_cross=bool_cross,
                x0=x0l,
                x1=x1l,
                res=np.min(np.atleast_1d(res_RZ)),
            )

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
        lsang_cross.append(sang_cross)
        lindr_cross.append(indr_cross)
        lindz_cross.append(indz_cross)
        if lang_pol_cross is not None:
            lang_pol_cross.append(ang_pol_cross)
            lang_tor_cross.append(ang_tor_cross)

        if keep3d is True:
            lsang_3d.append(sang_3d)
            lindr_3d.append(indr_3d)
            lindz_3d.append(indz_3d)
            lphi_3d.append(phi_3d)

            if lvectx is not None:
                lvectx.append(vx)
                lvecty.append(vy)
                lvectz.append(vz)

        if timing:
            t333 = dtm.datetime.now()     # DB
            dt333 += (t333-t222).total_seconds()

    # ----------------------------
    # harmonize and reshape pcross
    # ----------------------------

    if timing:
        t22 = dtm.datetime.now()     # DB

    pcross0, pcross1 = _harmonize_reshape_pcross(
        lpcross=lpcross,
        shape=shape_cam,
    )

    phor0, phor1 = _harmonize_reshape_pcross(
        lpcross=lphor,
        shape=shape_cam,
    )

    # --------------------------------------
    # harmonize and reshape sang, indr, indz
    # --------------------------------------

    # cross
    dout = _harmonize_reshape_others(
        # cross
        lsang_cross=lsang_cross,
        lindr_cross=lindr_cross,
        lindz_cross=lindz_cross,
        # params
        npix=npix,
        is2d=is2d,
        shape=shape_cam,
    )

    # extract
    lk = ['lsang_cross', 'lindr_cross', 'lindz_cross']
    sang_cross, indr_cross, indz_cross = [dout.get(k0) for k0 in lk]


    if lang_pol_cross is not None:
        dout = _harmonize_reshape_others(
            lang_pol_cross=lang_pol_cross,
            lang_tor_cross=lang_tor_cross,
            # params
            npix=npix,
            is2d=is2d,
            shape=shape_cam,
        )
        lk = ['lang_pol_cross', 'lang_tor_cross']
        ang_pol_cross, ang_tor_cross = [dout.get(k0) for k0 in lk]

    # -------
    # 3d

    dout = _harmonize_reshape_others(
        # 3d
        lsang_3d=lsang_3d,
        lindr_3d=lindr_3d,
        lindz_3d=lindz_3d,
        lphi_3d=lphi_3d,
        # vect
        lvectx=lvectx,
        lvecty=lvecty,
        lvectz=lvectz,
        # params
        npix=npix,
        is2d=is2d,
        shape=shape_cam,
    )

    # extract
    lk = [
        'lsang_3d', 'lindr_3d', 'lindz_3d', 'lphi_3d',
        'lvectx', 'lvecty', 'lvectz',
    ]

    (
     sang_3d, indr_3d, indz_3d, phi_3d,
     vectx, vecty, vectz,
     ) = [dout.get(k0) for k0 in lk]

    # --------------
    # prepare output
    # --------------

    knpts_cross = f'{key_cam}_vos_npts_cross'
    kir_cross = f'{key_cam}_vos_ir_cross'
    kiz_cross = f'{key_cam}_vos_iz_cross'
    ksa_cross = f'{key_cam}_vos_sa_cross'
    ref_cam = coll.dobj['camera'][key_cam]['dgeom']['ref']
    ref_cross = tuple(list(ref_cam) + [knpts_cross])

    if lang_pol_cross is not None:
        kap_cross = f'{key_cam}_vos_ang_pol_cross'
        kat_cross = f'{key_cam}_vos_ang_tor_cross'

    if keep3d:
        knpts_3d = f'{key_cam}_vos_npts_3d'
        kir_3d = f'{key_cam}_vos_ir_3d'
        kiz_3d = f'{key_cam}_vos_iz_3d'
        kphi_3d = f'{key_cam}_vos_phi_3d'
        ksa_3d = f'{key_cam}_vos_sa_3d'
        ref_3d = tuple(list(ref_cam) + [knpts_3d])

        if lvectx is not None:
            kvectx = f'{key_cam}_vos_vx'
            kvecty = f'{key_cam}_vos_vy'
            kvectz = f'{key_cam}_vos_vz'

    # ----------------
    # format output
    # ----------------

    # dref
    dref = {
        'npts_cross': {
            'key': knpts_cross,
            'size': indr_cross.shape[-1],
        },
    }

    if keep3d is True:
        dref['npts_3d'] = {
            'key': knpts_3d,
            'size': indr_3d.shape[1],
        }

    # ddata - polygons
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
    }

    # ddata - cross indices
    dout.update({
        'indr_cross': {
            'key': kir_cross,
            'data': indr_cross,
            'ref': ref_cross,
            'units': '',
            'dim': 'index',
        },
        'indz_cross': {
            'key': kiz_cross,
            'data': indz_cross,
            'ref': ref_cross,
            'units': '',
            'dim': 'index',
        },
        'sang_cross': {
            'key': ksa_cross,
            'data': sang_cross,
            'ref': ref_cross,
            'units': 'sr.m3',
            'dim': 'sang',
        },
    })

    # return_vect
    if lang_pol_cross is not None:
        dout.update({
            'ang_pol_cross': {
                'key': kap_cross,
                'data': ang_pol_cross,
                'ref': ref_cross,
                'units': 'rad',
                'dim': 'angle',
            },
            'ang_tor_cross': {
                'key': kat_cross,
                'data': ang_tor_cross,
                'ref': ref_cross,
                'units': 'rad',
                'dim': 'angle',
            },
        })

    # keep3d
    if keep3d is True:
        dout.update({
            'indr_3d': {
                'key': kir_3d,
                'data': indr_3d,
                'ref': ref_3d,
                'units': '',
                'dim': 'index',
            },
            'indz_3d': {
                'key': kiz_3d,
                'data': indz_3d,
                'ref': ref_3d,
                'units': '',
                'dim': 'index',
            },
            'phi_3d': {
                'key': kphi_3d,
                'data': phi_3d,
                'ref': ref_3d,
                'units': '',
                'dim': 'index',
            },
            'sang_3d': {
                'key': ksa_3d,
                'data': sang_3d,
                'ref': ref_3d,
                'units': 'sr.m3',
                'dim': 'sang',
            },
        })

        if lvectx is not None:
            dout.update({
                'vectx_3d': {
                    'key': kvectx,
                    'data': vectx,
                    'ref': ref_3d,
                    'units': '',
                },
                'vecty_3d': {
                    'key': kvecty,
                    'data': vecty,
                    'ref': ref_3d,
                    'units': '',
                },
                'vectz_3d': {
                    'key': kvectz,
                    'data': vectz,
                    'ref': ref_3d,
                    'units': '',
                },
            })

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
#               Iniialize lists
# ###########################################################


def _initialize_lists(
    return_vector=None,
    keep3d=None,
):

    lpcross = []
    lphor = []
    lsang_cross = []
    lindr_cross = []
    lindz_cross = []
    if keep3d is True:
        lindr_3d = []
        lindz_3d = []
        lphi_3d = []
        lsang_3d = []
    else:
        lindr_3d = None
        lindz_3d = None
        lphi_3d = None
        lsang_3d = None

    if return_vector is True:
        lang_tor_cross = []
        lang_pol_cross = []
        if keep3d is True:
            lvectx = []
            lvecty = []
            lvectz = []
        else:
            lvectx = None
            lvecty = None
            lvectz = None
    else:
        lang_tor_cross = None
        lang_pol_cross = None
        lvectx = None
        lvecty = None
        lvectz = None

    return (
        lpcross, lphor, lsang_cross, lindr_cross, lindz_cross,
        lindr_3d, lindz_3d, lphi_3d, lsang_3d,
        lang_tor_cross, lang_pol_cross, lvectx, lvecty, lvectz,
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
    # debug
    debug=None,
    ii=None,
    ind=None,
):

    # ---------------------
    # get polygons - cross
    # ---------------------

    # ---------------------
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
    # ------------

    # indices
    index = (
        dsamp['ind']['data']
        & pcross.contains_points(np.array([x0f, x1f]).T).reshape(sh)
    )

    # R and Z indices
    ir, iz = index.nonzero()
    iru = np.unique(ir)

    # ---------------------------
    # get polygons - cross dphi_r
    # ---------------------------

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
    # ------------

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
    # ------------

    if iru.size == 0:
        return None, None, None, None, None, None, None, None

    # ------------
    # go on
    # ------------

    nphi_r = (
        np.ceil(x0u[iru]*(dphi_r[1, :] - dphi_r[0, :]) / res).astype(int)
        + 1
    )
    ddphi_r = np.diff(dphi_r, axis=0)[0, :] / (nphi_r - 1)

    # ------------
    # get indices
    # ------------

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

    dV = np.concatenate(tuple([
        np.repeat(dx0 * dx1 * x0u[i0] * ddphi_r[ii], ln[ii] * nphi_r[ii])
        for ii, i0 in enumerate(iru)
    ]))

    # -------------
    # derive coords
    # -------------

    # coordinates
    rr = x0u[ir[indrz]]
    xx = rr * np.cos(phi)
    yy = rr * np.sin(phi)
    zz = x1u[iz[indrz]]

    # ----------------
    # get indices dict
    # ----------------

    dind = {
        i0: {
            'dV': dx0 * dx1 * x0u[i0] * ddphi_r[ii],
            'iz': np.unique(iz[lind[ii]]),
            'indrz': ir[indrz] == i0,
            'phi': lphi[ii],
        }
        for ii, i0 in enumerate(iru)
    }

    # ----------------
    # debug
    # ----------------

    if debug is True:

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f"pixel ind = {ind}", size=14, fontweight='bold')

        ax0 = fig.add_subplot(1, 2, 1, aspect='equal')
        ax0.set_xlabel("R (m)", size=12, fontweight='bold')
        ax0.set_ylabel("Z (m)", size=12, fontweight='bold')

        ax1 = fig.add_subplot(1, 2, 2, aspect='equal')
        ax1.set_xlabel("X (m)", size=12, fontweight='bold')
        ax1.set_ylabel("Y (m)", size=12, fontweight='bold')

        ax0.fill(pc0, pc1, fc=(0.5, 0.5, 0.5, 0.5))
        ax1.fill(ph0, ph1, fc=(0.5, 0.5, 0.5, 0.5))

        ax0.plot(np.hypot(xx, yy), zz, '.')
        ax1.plot(xx, yy, '.')

    return xx, yy, zz, dind, ir[indrz], iz[indrz], iphi, dV


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


# DEPRECATED ?
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
#               Harmonize and reshape pcross
# ###########################################################


def _harmonize_reshape_pcross(
    lpcross=None,
    shape=None,
):

    # ------------------
    # get max nb of pts
    # ------------------

    # list if nb of pts
    ln = [pp[0].size if pp[0] is not None else 0 for pp in lpcross]
    nmax = np.max(ln)

    # initialize
    sh2 = tuple([nmax] + list(shape))
    pcross0 = np.full(sh2, np.nan)
    pcross1 = np.full(sh2, np.nan)

    # ----------------
    # loop on pixels
    # ---------------

    linds = [range(ss) for ss in shape]
    for ii, ind in enumerate(itt.product(*linds)):

        nn = ln[ii]
        if nn == 0:
            continue

        sli = tuple([slice(None)] + list(ind))

        if nmax > nn:
            ind = np.r_[0, np.linspace(0.1, 0.9, nmax - nn), np.arange(1, nn)]
            pcross0[sli] = scpinterp.interp1d(
                range(0, nn),
                lpcross[ii][0],
                kind='linear',
            )(ind)

            pcross1[sli] = scpinterp.interp1d(
                range(0, nn),
                lpcross[ii][1],
                kind='linear',
            )(ind)

        else:
            pcross0[sli] = lpcross[ii][0]
            pcross1[sli] = lpcross[ii][1]

    return pcross0, pcross1


# ###########################################################
# ###########################################################
#               Harmonize and reshape others
# ###########################################################


def _harmonize_reshape_others(
    npix=None,
    is2d=None,
    shape=None,
    **kwdargs,
):

    # -----------------------------
    # check all input args have the same list of sizes
    # -----------------------------

    dnpts = {
        k0: [v1.size for v1 in v0] for k0, v0 in kwdargs.items()
        if v0 is not None
    }
    lkey = list(dnpts.keys())

    if not all([np.allclose(v0, dnpts[lkey[0]]) for v0 in dnpts.values()]):
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dnpts.items()]
        msg = (
            "All input args must have the same list of sizes!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ------
    # trival

    if len(lkey) == 0:
        return {}

    # list of sizes and max size
    lnpts = dnpts[lkey[0]]
    nmax = np.max(lnpts)
    shape_max = tuple(list(shape) + [nmax])

    # ------------------
    # prepare
    # ------------------

    dout = {}
    for k0 in lkey:

        # initialize
        if 'ind' in k0:
            dout[k0] = -np.ones(shape_max, dtype=int)
        else:
            dout[k0] = np.full(shape_max, np.nan)

        # fill
        linds = [range(ss) for ss in shape]
        for ii, ind in enumerate(itt.product(*linds)):
            sli = tuple(list(ind) + [np.arange(lnpts[ii])])
            dout[k0][sli] = kwdargs[k0][ii]

    return dout