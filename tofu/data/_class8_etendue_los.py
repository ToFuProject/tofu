# -*- coding: utf-8 -*-


import warnings


import numpy as np
import scipy.spatial as scpspat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import Polygon as plg
import datastock as ds


from ..geom import _comp_solidangles
from . import _class8_compute as _compute


__all__ = ['compute_etendue_los']


# ##################################################################
# ##################################################################
#                       Main
# ##################################################################


def compute_etendue_los(
    coll=None,
    key=None,
    # parameters
    analytical=None,
    numerical=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    # bool
    check=None,
    verb=None,
    plot=None,
    store=None,
):

    # ------------
    # check inputs

    (
        key,
        optics,
        optics_cls,
        ispectro,
        ldet,
        is2d,
        shape0,
        analytical,
        numerical,
        res,
        margin_par,
        margin_perp,
        check,
        verb,
        plot,
        store,
    ) = _diag_compute_etendue_check(
        coll=coll,
        key=key,
        analytical=analytical,
        numerical=numerical,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        check=check,
        verb=verb,
        plot=plot,
        store=store,
    )

    # prepare optics
    key_cam = optics[0]
    nd = len(ldet)

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

            lop_pre = optics[1:ispectro[0]+1]
            lop_pre_cls = optics_cls[1:ispectro[0]+1]
            lop_post = optics[ispectro[0]+1:]
            lop_post_cls = optics_cls[ispectro[0]+1:]
            iop_ref = ispectro[0]

            spec_key = optics[ispectro[0]]
            spec_cls = optics_cls[ispectro[0]]
            dg = coll.dobj[spec_cls][spec_key]['dgeom']
            spectro_pts2pt = coll.get_optics_spectro_pts2pt(spec_key)
            spectro_ptsvect = coll.get_optics_spectro_ptsvect(spec_key)
            spectro_x01toxyz = coll.get_optics_x01toxyz(spec_key)

        else:
            raise NotImplementedError()

    else:
        lop_pre = optics[1:]
        lop_pre_cls = optics_cls[1:]
        lop_post = []
        lop_post_cls = []
        iop_ref = 1

        spectro_pts2pt = None
        spectro_ptsvect = None
        spectro_x01toxyz = None

    cref = optics_cls[iop_ref]
    kref = optics[iop_ref]

    # get plane-projection functions
    func_to_plane_pre, func_to_3d_pre = _get_project_plane(
        plane_pt=coll.dobj[cref][kref]['dgeom']['cent'],
        plane_nin=coll.dobj[cref][kref]['dgeom']['nin'],
        plane_e0=coll.dobj[cref][kref]['dgeom']['e0'],
        plane_e1=coll.dobj[cref][kref]['dgeom']['e1'],
    )

    lfunc_post = [
        _get_project_plane(
            plane_pt=coll.dobj[cc][oo]['dgeom']['cent'],
            plane_nin=coll.dobj[cc][oo]['dgeom']['nin'],
            plane_e0=coll.dobj[cc][oo]['dgeom']['e0'],
            plane_e1=coll.dobj[cc][oo]['dgeom']['e1'],
        )
        for cc, oo in zip(lop_post_cls, lop_post)
    ]

    # ------------------------
    # loop on pixels to get:
    # analytical etendue
    # equivalent unique aperture
    # los

    (
        det_area, ap_area, distances,
        los_x, los_y, los_z,
        dlos_x, dlos_y, dlos_z,
        cos_los_det, cos_los_ap, solid_angles, res, pix_ap,
    ) = _loop_on_pix(
        coll=coll,
        ldet=ldet,
        # optics
        lop_pre=lop_pre,
        lop_pre_cls=lop_pre_cls,
        spectro_pts2pt=spectro_pts2pt,
        spectro_ptsvect=spectro_ptsvect,
        spectro_x01toxyz=spectro_x01toxyz,
        lop_post=lop_post,
        lop_post_cls=lop_post_cls,
        # projections
        plane_nin=coll.dobj[cref][kref]['dgeom']['nin'],
        func_to_plane_pre=func_to_plane_pre,
        func_to_3d_pre=func_to_3d_pre,
        lfunc_post=lfunc_post,
    )

    import pdb; pdb.set_trace() # DB

    # --------------------
    # compute analytically

    if analytical is True:
        etend0 = np.full(tuple(np.r_[3, nd]), np.nan)

        # 0th order
        etend0[0, :] = ap_area * det_area / distances**2

        # 1st order
        etend0[1, :] = (
            cos_los_ap * ap_area
            * cos_los_det * det_area / distances**2
        )

        # 2nd order
        etend0[2, :] = cos_los_ap * ap_area * solid_angles

    else:
        etend0 = None

    # --------------------
    # compute numerically

    if numerical is True:
        etend1 = _compute_etendue_numerical(
            ldeti=ldeti,
            aperture=aperture,
            pix_ap=pix_ap,
            res=res,
            los_x=los_x,
            los_y=los_y,
            los_z=los_z,
            margin_par=margin_par,
            margin_perp=margin_perp,
            check=check,
            verb=verb,
        )

    else:
        etend1 = None

    # --------------------
    # optional plotting

    if plot is True:
        dax = _plot_etendues(
            etend0=etend0,
            etend1=etend1,
            res=res,
        )

    # --------
    # reshape

    # etend0
    if etend0 is not None and is2d:
        etend0 = etend0.reshape(tuple(np.r_[3, shape0]))

    # etend1
    if etend1 is not None and is2d:
        etend1 = etend1.reshape(tuple(np.r_[res.size, shape0]))

    # los
    if los_x.shape != shape0:
        los_x = los_x.reshape(shape0)
        los_y = los_y.reshape(shape0)
        los_z = los_z.reshape(shape0)

    # --------------------
    # return dict

    dout = {
        'analytical': etend0,
        'numerical': etend1,
        'res': res,
        'los_x': los_x,
        'los_y': los_y,
        'los_z': los_z,
        'dlos_x': dlos_x,
        'dlos_y': dlos_y,
        'dlos_z': dlos_z,
    }

    # ----------
    # store

    if store is not False:

        # ref
        ref = coll.dobj['camera'][key_cam]['dgeom']['ref']

        # data
        etendue = dout[store][-1, :]

        if store == 'analytical':
            etend_type = store
        else:
            etend_type = res[-1]

        # keys
        ketendue = f'{key}-etend'
        ddata = {
            ketendue: {
                'data': etendue,
                'ref': ref,
                'dim': 'etendue',
                'quant': 'etendue',
                'name': 'etendue',
                'units': 'm2.sr'
            },
        }
        coll.update(ddata=ddata)

        coll.set_param(
            which='diagnostic',
            key=key,
            param='etendue',
            value=ketendue,
        )
        coll.set_param(
            which='diagnostic',
            key=key,
            param='etend_type',
            value=etend_type,
        )

    return dout, store


# ##################################################################
# ##################################################################
#                       Check
# ##################################################################


def _diag_compute_etendue_check(
    coll=None,
    key=None,
    analytical=None,
    numerical=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    check=None,
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
    dvect = coll.get_camera_unit_vectors(key=optics[0])
    outline = dgeom['outline']
    out0 = coll.ddata[outline[0]]['data']
    out1 = coll.ddata[outline[1]]['data']
    is2d = dgeom['type'] == '2d'
    par = dgeom['parallel']
    shape0 = cx.shape


    if is2d:
        cx = cx.ravel()
        cy = cy.ravel()
        cz = cz.ravel()
    nd = cx.size

    ldet = [
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
    # analytical

    analytical = ds._generic_check._check_var(
        analytical, 'analytical',
        types=bool,
        default=True,
    )

    # -----------
    # numerical

    numerical = ds._generic_check._check_var(
        numerical, 'numerical',
        types=bool,
        default=False,
    )

    # -----------
    # res

    if res is not None:
        res = np.atleast_1d(res).ravel()

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
    # check

    check = ds._generic_check._check_var(
        check, 'check',
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

    lok = [False]
    if analytical is True:
        lok.append('analytical')
    if numerical is True:
        lok.append('numerical')
    store = ds._generic_check._check_var(
        store, 'store',
        default=lok[-1],
        allowed=lok,
    )

    return (
        key,
        optics,
        optics_cls,
        ispectro,
        ldet,
        is2d,
        shape0,
        analytical,
        numerical,
        res,
        margin_par,
        margin_perp,
        check,
        verb,
        plot,
        store,
    )


# ##################################################################
# ##################################################################
#                    Loop on camera pixels
# ##################################################################


def _loop_on_pix(
    coll=None,
    # detectors
    ldet=None,
    # optics before spectro
    lop_pre=None,
    lop_pre_cls=None,
    # spectro optics
    spectro_planar=None,
    spectro_pts2pt=None,
    spectro_ptsvect=None,
    spectro_x01toxyz=None,
    # optics after spectro
    lop_post=None,
    lop_post_cls=None,
    # projection plane
    plane_nin=None,
    func_to_plane_pre=None,
    func_to_3d_pre=None,
    lfunc_post=None,
    # extra
    res=None,
):

    # apertures before a cryst / grating
    nap_pre = len(lop_pre)
    nap_post = len(lop_post)
    nd = len(ldet)

    ap01 = np.r_[np.nan, np.nan]
    ap_cent = np.r_[np.nan, np.nan, np.nan]

    # -------------------------
    # intersection of apertures

    lpoly_pre = [
        coll.dobj[c0][k0]['dgeom']['poly']
        for c0, k0 in zip(lop_pre_cls, lop_pre)
    ]
    lpoly_pre_x = [coll.ddata[pp[0]]['data'] for pp in lpoly_pre]
    lpoly_pre_y = [coll.ddata[pp[1]]['data'] for pp in lpoly_pre]
    lpoly_pre_z = [coll.ddata[pp[2]]['data'] for pp in lpoly_pre]

    lpoly_post = _get_lpoly_post(
        coll=coll,
        lop_post_cls=lop_post_cls,
        lop_post=lop_post,
    )

    # prepare data
    nd = len(ldet)
    ap_area = np.zeros((nd,), dtype=float)
    los_x = np.full((nd,), np.nan)
    los_y = np.full((nd,), np.nan)
    los_z = np.full((nd,), np.nan)
    solid_angles = np.zeros((nd,), dtype=float)
    cos_los_det = np.full((nd,), np.nan)
    distances = np.full((nd,), np.nan)
    mindiff = np.full((nd,), np.nan)

    # extra los for spectro only
    # if len(nap_post) > 0:
        # dlos_x = np.full((nd), np.nan)
        # dlos_y = np.full((nd), np.nan)
        # dlos_z = np.full((nd), np.nan)
    # else:
    dlos_x = None
    dlos_y = None
    dlos_z = None

    # store projected intersection of apertures (3d), per pix
    # useful later for estimating the plane to be sample (numerical)
    pix_ap = []

    for ii in range(nd):

        isok = True
        p_a = None

        # loop on pre-crystal apertures
        for jj in range(nap_pre):

            # ap
            p0, p1 = func_to_plane_pre(
                pt_x=ldet[ii]['cents_x'],
                pt_y=ldet[ii]['cents_y'],
                pt_z=ldet[ii]['cents_z'],
                poly_x=lpoly_pre_x[jj],
                poly_y=lpoly_pre_y[jj],
                poly_z=lpoly_pre_z[jj],
            )

            if p_a is None:
                p_a = plg.Polygon(np.array([p0, p1]).T)
            else:
                p_a = p_a & plg.Polygon(np.array([p0, p1]).T)
                if p_a.nPoints() < 3:
                    p_a = None
                    isok = False
                    break

        # loop on post-crystal apertures
        if isok is True and nap_post > 0:

            # det cent to contour of intersection
            p0, p1 = np.array(p_a.contour(0)).T
            p0, p1 = _compute._interp_poly(
                p0=p0,
                p1=p1,
                add_points=5,
                mode='min',
                isclosed=False,
                closed=False,
                ravel=False,
            )[:2]
            px, py, pz = func_to_3d_pre(p0, p1)

            vx = px - ldet[ii]['cents_x']
            vy = py - ldet[ii]['cents_y']
            vz = pz - ldet[ii]['cents_z']
            vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
            vx = vx / vnorm
            vy = vy / vnorm
            vz = vz / vnorm

            # project contours of crystal onto post-crytal plane
            Dx, Dy, Dz, vx, vy, vz = spectro_ptsvect(
                pts_x=ldet[ii]['cents_x'],
                pts_y=ldet[ii]['cents_y'],
                pts_z=ldet[ii]['cents_z'],
                vect_x=vx,
                vect_y=vy,
                vect_z=vz,
            )[:6]

            for jj in range(nap_post):

                # project on post plane
                p0, p1 = lfunc_post[jj][0](
                    pt_x=Dx,
                    pt_y=Dy,
                    pt_z=Dz,
                    vx=vx,
                    vy=vy,
                    vz=vz,
                )
                p_a2 = plg.Polygon(np.array([p0, p1]).T)

                # ap
                if len(lpoly_post[jj]) == 2:
                    p0 = lpoly_post[jj][0]
                    p1 = lpoly_post[jj][1]
                else:
                    centroid = None
                    p0, p1 = lfunc_post[jj][0](
                        pt_x=centroid[0],
                        pt_y=centroid[1],
                        pt_z=centroid[2],
                        poly_x=lpoly_post[jj][0],
                        poly_y=lpoly_post[jj][1],
                        poly_z=lpoly_post[jj][2],
                    )

                # intersection
                p_a2 = p_a2 & plg.Polygon(np.array([p0, p1]).T)
                if p_a2.nPoints() < 3:
                    p_a2 = None
                    isok = False
                    break

                # shrink for safety
                p_a2.scale(1.-1e-5, 1-1e-5)

                # add points
                p0, p1 = np.array(p_a2.contour(0)).T
                cent01 = p_a2.center()
                p0, p1 = _compute._interp_poly(
                    p0=p0,
                    p1=p1,
                    add_points=10,
                    mode='min',
                    isclosed=False,
                    closed=False,
                    ravel=False,
                )[:2]

                # back to 3d
                px, py, pz = lfunc_post[jj][1](x0=p0, x1=p1)

                # get reflected aperture
                px, py, pz, x0, x1 = spectro_pts2pt(
                    pt_x=ldet[ii]['cents_x'],
                    pt_y=ldet[ii]['cents_y'],
                    pt_z=ldet[ii]['cents_z'],
                    # poly
                    pts_x=px,
                    pts_y=py,
                    pts_z=pz,
                    # surface
                    return_xyz=True,
                    returnx01=True,
                )

                if jj < nap_post - 1:
                    # update
                    Dx, Dy, Dz, vx, vy, vz = spectro_ptsvect(
                        pts_x=ldet[ii]['cents_x'],
                        pts_y=ldet[ii]['cents_y'],
                        pts_z=ldet[ii]['cents_z'],
                        vect_x=px - ldet[ii]['cents_x'],
                        vect_y=py - ldet[ii]['cents_y'],
                        vect_z=pz - ldet[ii]['cents_z'],
                    )[:6]

                else:
                    # project on plane
                    p0, p1 = func_to_plane_pre(
                        pt_x=ldet[ii]['cents_x'],
                        pt_y=ldet[ii]['cents_y'],
                        pt_z=ldet[ii]['cents_z'],
                        poly_x=px,
                        poly_y=py,
                        poly_z=pz,
                    )
                    centroid = plg.Polygon(np.array([x0, x1])).center()
                    centroid = spectro_x01toxyz(centroid)
                    centroid = func_to_plane_pre(
                        pt_x=ldet[ii]['cents_x'],
                        pt_y=ldet[ii]['cents_y'],
                        pt_z=ldet[ii]['cents_z'],
                        poly_x=centroid[0],
                        poly_y=centtoid[1],
                        poly_z=centroid[2],
                    )


                    p_a = p_a & plg.Polygon(np.array([p02, p12]).T)
                    if p_a.nPoints() < 3:
                        isok = False

                    if lop_post[0] == 'cryst1-slit':
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.plot(
                            np.array(p_a.contour(0))[:, 0],
                            np.array(p_a.contour(0))[:, 1],
                            '.-k',
                            p_a.center()[0:1],
                            p_a.center()[1:2],
                            'or',
                        )
                        import pdb; pdb.set_trace()     # DB


        # -------------------------
        # compute solid angle + los

        if isok is False:
            pix_ap.append(None)
            continue

        else:

            # area
            ap_area[ii] = p_a.area()

            # ap_cent
            ap01[:] = p_a.center()
            ap_cent[:] = func_to_3d_pre(x0=ap01[0], x1=ap01[1])
            mindiff[ii] = np.sqrt(np.min(np.diff(p0)**2 + np.diff(p1)**2))

            # ----------------------------------
            # los, distances, cosines

            los_x[ii] = ap_cent[0] - ldet[ii]['cents_x']
            los_y[ii] = ap_cent[1] - ldet[ii]['cents_y']
            los_z[ii] = ap_cent[2] - ldet[ii]['cents_z']

            if dlos_x is not None:
                pa01 = np.array(p_a.contour(0)).T
                import pdb; pdb.set_trace()     # DB
                dlos_x[ii, ...] = pax - ldet[ii]['cents_x']
                dlos_y[ii, ...] = pay - ldet[ii]['cents_y']
                dlos_z[ii, ...] = paz - ldet[ii]['cents_z']

            # ------------
            # solid angles

            solid_angles[ii] = _comp_solidangles.calc_solidangle_apertures(
                # observation points
                pts_x=ap_cent[0],
                pts_y=ap_cent[1],
                pts_z=ap_cent[2],
                # polygons
                apertures=None,
                detectors=ldet[ii],
                # possible obstacles
                config=None,
                # parameters
                visibility=False,
                return_vector=False,
            )[0, 0]

            # 2d polygon
            p0, p1 = np.array(p_a.contour(0)).T

            # equivalent ap as seen from pixel
            pix_ap.append(func_to_3d_pre(x0=p0, x1=p1))

    # -------------
    # normalize los

    distances = np.sqrt(los_x**2 + los_y**2 + los_z**2)

    los_x = los_x / distances
    los_y = los_y / distances
    los_z = los_z / distances

    if dlos_x is not None:
        ddist = np.sqrt(dlos_x**2 + dlos_y**2 + dlos_z**2)
        dlos_x = dlos_x / ddist
        dlos_y = dlos_y / ddist
        dlos_z = dlos_z / ddist

    # ------
    # angles

    for ii in range(nd):
        cos_los_det[ii] = (
            los_x[ii] * ldet[ii]['nin_x']
            + los_y[ii] * ldet[ii]['nin_y']
            + los_z[ii] * ldet[ii]['nin_z']
        )

    # abs() because for spectro nin is the other way around
    cos_los_ap = np.abs(
        los_x * plane_nin[0]
        + los_y * plane_nin[1]
        + los_z * plane_nin[2]
    )

    # -----------
    # surfaces

    # det
    if ldet[0].get('pix_area') is None:
        det_area = plg.Polygon(np.array([
            ldet[0]['outline_x0'],
            ldet[0]['outline_x1'],
        ]).T).area()
    else:
        det_area = ldet[0]['pix_area']

    # -------------------------------------
    # det outline discretization resolution

    if res is None:

        res = min(np.sqrt(det_area), np.nanmin(mindiff))
        if np.any(ap_area > 0.):
            res = min(res, np.sqrt(np.min(ap_area[ap_area > 0.])))

        res = res * np.r_[1., 0.5, 0.1]

    iok = np.isfinite(res)
    iok[iok] = res[iok] > 0
    if not np.any(iok):
        res = np.r_[0.001]
    else:
        res = res[iok]

    return (
        det_area, ap_area, distances,
        los_x, los_y, los_z,
        dlos_x, dlos_y, dlos_z,
        cos_los_det, cos_los_ap, solid_angles, res, pix_ap,
    )


# ##################################################################
# ##################################################################
#                   op_post interpolation
# ##################################################################


def _get_lpoly_post(coll=None, lop_post_cls=None, lop_post=None):

    if len(lop_post) == 0:
        return

    lpoly_post = []
    for cc, oo in zip(lop_post_cls, lop_post):
        dgeom = coll.dobj[cc][oo]['dgeom']
        if dgeom['type'] == 'planar':
            p0, p1 = dgeom['outline']
            lpoly_post.append((
                coll.ddata[p0]['data'],
                coll.ddata[p1]['data'],
            ))
        else:
            px, py, pz = dgeom['poly']
            lpoly_post.append((
                coll.ddata[px]['data'],
                coll.ddata[py]['data'],
                coll.ddata[pz]['data'],
            ))
    return lpoly_post


# ##################################################################
# ##################################################################
#                   preparation routine
# ##################################################################


def _get_project_plane(
    plane_pt=None,
    plane_nin=None,
    plane_e0=None,
    plane_e1=None,
):

    def _project_poly_on_plane_from_pt(
        pt_x=None,
        pt_y=None,
        pt_z=None,
        poly_x=None,
        poly_y=None,
        poly_z=None,
        vx=None,
        vy=None,
        vz=None,
        plane_pt=plane_pt,
        plane_nin=plane_nin,
        plane_e0=plane_e0,
        plane_e1=plane_e1,
    ):

        sca0 = (
            (plane_pt[0] - pt_x)*plane_nin[0]
            + (plane_pt[1] - pt_y)*plane_nin[1]
            + (plane_pt[2] - pt_z)*plane_nin[2]
        )

        if vx is None:
            vx = poly_x - pt_x
            vy = poly_y - pt_y
            vz = poly_z - pt_z

        sca1 = vx*plane_nin[0] + vy*plane_nin[1] + vz*plane_nin[2]

        k = sca0 / sca1

        px = pt_x + k * vx
        py = pt_y + k * vy
        pz = pt_z + k * vz

        p0 = (
            (px - plane_pt[0])*plane_e0[0]
            + (py - plane_pt[1])*plane_e0[1]
            + (pz - plane_pt[2])*plane_e0[2]
        )
        p1 = (
            (px - plane_pt[0])*plane_e1[0]
            + (py - plane_pt[1])*plane_e1[1]
            + (pz - plane_pt[2])*plane_e1[2]
        )

        return p0, p1

    def _back_to_3d(
        x0=None,
        x1=None,
        plane_pt=plane_pt,
        plane_e0=plane_e0,
        plane_e1=plane_e1,
    ):

        return (
            plane_pt[0] + x0*plane_e0[0] + x1*plane_e1[0],
            plane_pt[1] + x0*plane_e0[1] + x1*plane_e1[1],
            plane_pt[2] + x0*plane_e0[2] + x1*plane_e1[2],
        )

    return _project_poly_on_plane_from_pt, _back_to_3d

# ##################################################################
# ##################################################################
#           Numerical etendue estimation routine
# ##################################################################


def _compute_etendue_numerical(
    ldeti=None,
    aperture=None,
    pix_ap=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    los_x=None,
    los_y=None,
    los_z=None,
    check=None,
    verb=None,
):

    # shape0 = det['cents_x'].shape
    nd = len(ldeti)

    ap_ind = np.cumsum([v0['poly_x'].size for v0 in aperture.values()][:-1])

    ap_tot_px = np.concatenate(tuple(
        [v0['poly_x'] for v0 in aperture.values()]
    ))
    ap_tot_py = np.concatenate(tuple(
        [v0['poly_y'] for v0 in aperture.values()]
    ))
    ap_tot_pz = np.concatenate(tuple(
        [v0['poly_z'] for v0 in aperture.values()]
    ))

    # ------------------------------
    # Get plane perpendicular to los

    etendue = np.full((res.size, nd), np.nan)
    for ii in range(nd):

        if verb is True:
            msg = f"Numerical etendue for det {ii+1} / {nd}"
            print(msg)

        if np.isnan(los_x[ii]):
            continue

        # get det corners to aperture corners vectors
        out_c_x0 = np.r_[0, ldeti[ii]['outline_x0']]
        out_c_x1 = np.r_[0, ldeti[ii]['outline_x1']]

        # det poly 3d
        det_Px = (
            ldeti[ii]['cents_x']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_x']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_x']
        )
        det_Py = (
            ldeti[ii]['cents_y']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_y']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_y']
        )
        det_Pz = (
            ldeti[ii]['cents_z']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_z']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_z']
        )

        # det to ap vectors
        PA_x = ap_tot_px[:, None] - det_Px[None, :]
        PA_y = ap_tot_py[:, None] - det_Py[None, :]
        PA_z = ap_tot_pz[:, None] - det_Pz[None, :]

        sca1 = PA_x * los_x[ii] + PA_y * los_y[ii] + PA_z * los_z[ii]
        # get length along los
        k_los = (1. + margin_par) * np.max(sca1)

        # get center of plane perpendicular to los
        c_los_x = ldeti[ii]['cents_x'] + k_los * los_x[ii]
        c_los_y = ldeti[ii]['cents_y'] + k_los * los_y[ii]
        c_los_z = ldeti[ii]['cents_z'] + k_los * los_z[ii]

        # get projections of corners on plane perp. to los
        sca0 = (
            (c_los_x - det_Px[None, :]) * los_x[ii]
            + (c_los_y - det_Py[None, :]) * los_y[ii]
            + (c_los_z - det_Pz[None, :]) * los_z[ii]
        )
        k_plane = sca0 / sca1

        # get LOS-specific unit vectors

        e0_xi = (
            los_y[ii] * ldeti[ii]['e1_z'] - los_z[ii] * ldeti[ii]['e1_y']
        )
        e0_yi = (
            los_z[ii] * ldeti[ii]['e1_x'] - los_x[ii] * ldeti[ii]['e1_z']
        )
        e0_zi = (
            los_x[ii] * ldeti[ii]['e1_y'] - los_y[ii] * ldeti[ii]['e1_x']
        )

        e0_normi = np.sqrt(e0_xi**2 + e0_yi**2 + e0_zi**2)
        e0_xi = e0_xi / e0_normi
        e0_yi = e0_yi / e0_normi
        e0_zi = e0_zi / e0_normi

        e1_xi = los_y[ii] * e0_zi - los_z[ii] * e0_yi
        e1_yi = los_z[ii] * e0_xi - los_x[ii] * e0_zi
        e1_zi = los_x[ii] * e0_yi - los_y[ii] * e0_xi

        # get projections on det_e0 and det_e1 in plane

        x0 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e0_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e0_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e0_zi,
            ap_ind,
            axis=0,
        )
        x1 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e1_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e1_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e1_zi,
            ap_ind,
            axis=0,
        )

        x0_min = np.max([np.min(x0s) for x0s in x0])
        x0_max = np.min([np.max(x0s) for x0s in x0])
        x1_min = np.max([np.min(x1s) for x1s in x1])
        x1_max = np.min([np.max(x1s) for x1s in x1])

        w0 = x0_max - x0_min
        w1 = x1_max - x1_min

        min_res = min(2*margin_perp*w0, 2*margin_perp*w1)
        too_large = res >= min_res
        if np.any(too_large):
            msg = (
                f"Minimum etendue resolution for det {ii} / {nd}: {min_res}\n"
                "The following res values may lead to errors:\n"
                f"\t- res values = {res}\n"
                f"\t- too large  = {too_large}\n"
            )
            warnings.warn(msg)

        # -------------------
        # Discretize aperture

        for jj in range(res.size):

            coef = 1. + 2.*margin_perp
            n0 = int(np.ceil(coef*w0 / res[jj]))
            n1 = int(np.ceil(coef*w1 / res[jj]))

            d0 = coef*w0 / n0
            d1 = coef*w1 / n1

            ds = d0 * d1

            pts_0 = np.linspace(
                x0_min - margin_perp*w0,
                x0_max + margin_perp*w0,
                n0 + 1,
            )
            pts_1 = np.linspace(
                x1_min - margin_perp*w1,
                x1_max + margin_perp*w1,
                n1 + 1,
            )
            pts_0 = 0.5 * (pts_0[1:] + pts_0[:-1])
            pts_1 = 0.5 * (pts_1[1:] + pts_1[:-1])

            # debug
            # n0, n1 = 2, 2
            # pts_0 = np.r_[pts_0[0], pts_0[0]]
            # pts_1 = np.r_[0, 0]

            pts_x = (
                c_los_x + pts_0[:, None] * e0_xi + pts_1[None, :] * e1_xi
            ).ravel()
            pts_y = (
                c_los_y + pts_0[:, None] * e0_yi + pts_1[None, :] * e1_yi
            ).ravel()
            pts_z = (
                c_los_z + pts_0[:, None] * e0_zi + pts_1[None, :] * e1_zi
            ).ravel()

            if verb is True:
                msg = (
                    f"\tres = {res[jj]} ({jj+1} / {res.size})"
                    f"    nb. of points = {pts_x.size}"
                )
                print(msg)

            # ----------------------------------
            # compute solid angle for each pixel

            if check is True:
                solid_angle = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                )

                sar = solid_angle.reshape((n0, n1))
                c0 = (
                    ((pts_0[0] < x0_min) == np.all(sar[0, :] == 0))
                    and ((pts_0[-1] > x0_max) == np.all(sar[-1, :] == 0))
                    and ((pts_1[0] < x1_min) == np.all(sar[:, 0] == 0))
                    and ((pts_1[-1] > x1_max) == np.all(sar[:, -1] == 0))
                )
                if not c0 and not too_large[jj]:
                    # debug
                    plt.figure()
                    plt.imshow(
                        sar.T,
                        extent=(
                            x0_min - margin_perp*w0, x0_max + margin_perp*w0,
                            x1_min - margin_perp*w1, x1_max + margin_perp*w1,
                        ),
                        interpolation='nearest',
                        origin='lower',
                        aspect='equal',
                    )

                    lc = ['r', 'm', 'c', 'y']
                    for ss in range(len(x0)):
                        iss = np.r_[np.arange(0, x0[ss].shape[0]), 0]
                        plt.plot(
                            x0[ss][iss, :],
                            x1[ss][iss, :],
                            c=lc[ss%len(lc)],
                            marker='o',
                            ls='-',
                        )

                    plt.plot(
                        pts_0, np.mean(pts_1)*np.ones((n0,)),
                        c='k', marker='.', ls='None',
                    )
                    plt.plot(
                        np.mean(pts_0)*np.ones((n1,)), pts_1,
                        c='k', marker='.', ls='None',
                    )
                    plt.gca().set_xlabel('x0')
                    plt.gca().set_xlabel('x1')
                    # import pdb; pdb.set_trace()
                    msg = "Something is wrong with solid_angle or sampling"
                    raise Exception(msg)
                else:
                    etendue[jj, ii] = np.sum(solid_angle) * ds

            else:
                etendue[jj, ii] = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    summed=True,
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                ) * ds

    return etendue


# ##################################################################
# ##################################################################
#                   Plotting routine
# ##################################################################


def _plot_etendues(
    etend0=None,
    etend1=None,
    res=None,
):

    # -------------
    # prepare data

    nmax = 0
    if etend0 is not None:
        if etend0.ndim > 2:
            etend0 = etend0.reshape((etend0.shape[0], -1))
        nmax = max(nmax, etend0.shape[0])
    if etend1 is not None:
        if etend1.ndim > 2:
            etend1 = etend1.reshape((etend1.shape[0], -1))
        nmax = max(nmax, etend1.shape[0])

    x0 = None
    if etend0 is not None:
        x0 = [
            f'order {ii}' if ii < 3 else '' for ii in range(nmax)
        ]
    if etend1 is not None:
        x1 = [f'{res[ii]}' if ii < res.size-1 else '' for ii in range(nmax)]
        if x0 is None:
            x0 = x1
        else:
            x0 = [f'{x0[ii]}\n{x1[ii]}' for ii in range(nmax)]

    # -------------
    # prepare axes

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_ylabel('Etendue ' + r'($m^2.sr$)', size=12, fontweight='bold')
    ax.set_xlabel('order of approximation', size=12, fontweight='bold')

    ax.set_xticks(range(0, nmax))
    ax.set_xticklabels(x0)

    # -------------
    # plot

    if etend0 is not None:
        lines = ax.plot(
            etend0,
            ls='-',
            marker='o',
            ms=6,
        )
        lcol = [ll.get_color() for ll in lines]
    else:
        lcol = [None for ii in range(etend1.shape[1])]

    if etend1 is not None:
        for ii in range(etend1.shape[1]):
            ax.plot(
                etend1[:, ii],
                ls='--',
                marker='*',
                ms=6,
                color=lcol[ii],
            )

    # -------------
    # legend

    handles = [
        mlines.Line2D(
            [], [],
            c='k', marker='o', ls='-', ms=6,
            label='analytical',
        ),
        mlines.Line2D(
            [], [],
            c='k', marker='*', ls='--', ms=6,
            label='numerical',
        ),
    ]
    ax.legend(handles=handles)

    return ax
