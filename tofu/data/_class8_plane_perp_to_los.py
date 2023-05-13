# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


# tofu
from ..geom import _comp_solidangles


# ###############################################################
# ###############################################################
#                   Main
# ###############################################################


def main(
    coll=None,
    key_diag=None,
    key_cam=None,
    indch=None,
    indref=None,
    # parameters
    res=None,
    margin_par=None,
    margin_perp=None,
    config=None,
    # bool
    verb=None,
    plot=None,
):

    # ------------
    # check inputs

    (
        key_diag, key_cam, indch, indref,
        parallel, is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        verb, plot,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        indch=indch,
        indref=indref,
        # parameters
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        # bool
        verb=verb,
        plot=plot,
    )

    # ------------------
    # get indref for los

    if indref is None:
        if spectro or is2d or indch is None:
            etend = coll.ddata[doptics['etendue']]['data']
            indref = np.nanargmax(etend)

            if is2d:
                n0, n1 = etend.shape
                indref = np.r_[indref // n1, indref % n1].astype(int)

        else:
            indref = indch

    # ----------
    # los_ref

    klos = doptics['los']

    # start_ref
    ptsx, ptsy, ptsz = coll.get_rays_pts(
        key=klos,
    )
    ipts = tuple(np.r_[-2, indref])
    pt_ref = np.r_[ptsx[ipts], ptsy[ipts], ptsz[ipts]]

    # los_ref
    vx, vy, vz = coll.get_rays_vect(
        key=klos,
        norm=True,
    )
    iv = tuple(np.r_[-1, indref])
    los_ref = np.r_[vx[iv], vy[iv], vz[iv]]

    # -----------------
    # furthest aperture

    if spectro:
        if len(lop_post) > 0:
            poly = lop_post[-1]
        else:
            kmax = 0.
    else:
        if len(doptics['optics']) > 0:
            poly = doptics['optics'][-1]
        else:
            kmax = 0.

    # poly => sca
    if poly is not None:
        px, py, pz = coll.get_optics_poly(poly)
        poly = np.array([px, py, pz])
        kmax = np.max(np.sum(
            (poly - pt_ref[:, None])*los_ref[:, None],
            axis=0,
        ))

    # get length along los
    klos = kmax + margin_par

    # get pt plane
    pt_plane = pt_ref + klos * los_ref

    # -------------------------------------
    # create plane perpendicular to los_ref

    if parallel:
        e0_cam = coll.dobj['camera'][key_cam]['dgeom']['e0']
        e1_cam = coll.dobj['camera'][key_cam]['dgeom']['e1']
    else:
        ke0 = coll.dobj['camera'][key_cam]['dgeom']['e0']
        ke1 = coll.dobj['camera'][key_cam]['dgeom']['e1']
        e0_cam = coll.ddata[ke0]['data'][indref]
        e1_cam = coll.ddata[ke1]['data'][indref]

    e0 = np.cross(e1_cam, los_ref)
    e0 = e0 / np.linalg.norm(e0)

    e1 = np.cross(los_ref, e0)
    e1 = e1 / np.linalg.norm(e1)

    # -------------------------------------
    # create plane perpendicular to los_ref

    # get limits of plane
    if indch is None:
        kk = vx[-1, ...]

        x0, x1, iok = _get_intersect_los_plane(
            cent=pt_plane,
            nin=los_ref,
            e0=e0,
            e1=e1,
            ptx=ptsx[-2, ...],
            pty=ptsy[-2, ...],
            ptz=ptsz[-2, ...],
            vx=vx[-1, ...],
            vy=vy[-1, ...],
            vz=vz[-1, ...],
        )

    else:
        x0 = np.r_[0]
        x1 = np.r_[0]

    # dx0, dx1
    dx0 = [np.nanmin(x0) - margin_perp, np.nanmax(x0) + margin_perp]
    dx1 = [np.nanmin(x1) - margin_perp, np.nanmax(x1) + margin_perp]

    # cerate 2d grid
    n0 = int(np.ceil((dx0[1] - dx0[0]) / res[0])) + 2
    n1 = int(np.ceil((dx1[1] - dx1[0]) / res[1])) + 2

    x0 = np.linspace(dx0[0], dx0[1], n0)
    x1 = np.linspace(dx1[0], dx1[1], n1)

    ds = (x0[1] - x0[0]) * (x1[1] - x1[0])

    x0f = np.repeat(x0[:, None], n1, axis=1)
    x1f = np.repeat(x1[None, :], n0, axis=0)

    # derive 3d pts
    ptsx = pt_ref[0] + klos*los_ref[0] + x0f*e0[0] + x1f*e1[0]
    ptsy = pt_ref[1] + klos*los_ref[1] + x0f*e0[1] + x1f*e1[1]
    ptsz = pt_ref[1] + klos*los_ref[2] + x0f*e0[2] + x1f*e1[2]

    # ----------
    # compute

    if spectro:
        dout = _spectro(
            coll=coll,
            doptics=doptics,
            indch=indch,
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
        )

    else:
        dout = _nonspectro(
            coll=coll,
            key_cam=key_cam,
            doptics=doptics,
            par=parallel,
            is2d=is2d,
            # points
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            config=config,
        )

    # ------------
    # format output

    dout.update({
        'key_diag': key_diag,
        'key_cam': key_cam,
        'indch': indch,
        'indref': indref,
        'los_ref': los_ref,
        'pt_ref': pt_ref,
        'klos': klos,
        'e0': e0,
        'e1': e1,
        'ptsx': ptsx,
        'ptsy': ptsy,
        'ptsz': ptsz,
        'x0': x0,
        'x1': x1,
        'ds': ds,
    })

    # -------
    # plot

    if plot is True:
        _plot(**dout)

    return dout


# ###############################################################
# ###############################################################
#                   Check
# ###############################################################


def _check(
    coll=None,
    key_diag=None,
    key_cam=None,
    indch=None,
    indref=None,
    # parameters
    res=None,
    margin_par=None,
    margin_perp=None,
    # bool
    verb=None,
    plot=None,
):

    # ----------
    # keys

    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key_diag,
        key_cam=key_cam,
    )

    if len(key_cam) > 1:
        msg = f"Please select a key_cam!\n\tkey_cam: {key_cam}"
        raise Exception(msg)

    key_cam = key_cam[0]
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']
    parallel = coll.dobj['camera'][key_cam]['dgeom']['parallel']
    doptics = coll.dobj['diagnostic'][key_diag]['doptics'][key_cam]

    if spectro and not is2d:
        msg = "Only implemented for 2d spectro"
        raise Exception(msg)

    if is2d:
        n0, n1 = coll.dobj['camera'][key_cam]['dgeom']['shape']
    else:
        nch = coll.dobj['camera'][key_cam]['dgeom']['shape'][0]

    # -----------------
    # loptics

    if spectro:
        optics, cls_optics = coll.get_optics_cls(doptics['optics'])
        ispectro = cls_optics.index('crystal')
        lop_pre = doptics['optics'][:ispectro]
        lop_post = doptics['optics'][ispectro+1:]

    else:
        lop_pre = doptics['optics']
        lop_post = []

    # -----------------
    # indch

    if indch is not None:
        if spectro or is2d:
            indch = ds._generic_check._check_flat1darray(
                indch, 'indch',
                dtype=int,
                size=2,
                sign='>0',
            )
            indch[0] = indch[0] % n0
            indch[1] = indch[1] % n1

        else:
            indch = int(ds._generic_check._check_var(
                indch, 'indch',
                types=(float, int),
                allowed=['los', 'vos'],
            )) % nch

    # -----------------
    # indref

    if indref is not None:
        if spectro or is2d:
            indref = ds._generic_check._check_flat1darray(
                indref, 'indref',
                dtype=int,
                size=2,
                sign='>0',
            )
            indref[0] = indref[0] % n0
            indref[1] = indref[1] % n1

        else:
            indref = int(ds._generic_check._check_var(
                indref, 'indref',
                types=(float, int),
                allowed=['los', 'vos'],
            )) % nch

            if indch is not None:
                indref = indch

    # ----------
    # res

    if res is None:
        res = 0.001

    if isinstance(res, (int, float)):
        res = np.r_[res, res]

    res = ds._generic_check._check_flat1darray(
        res, 'res',
        dtype=float,
        size=2,
        sign='>0.',
    )

    # -----------
    # margin_par

    margin_par = ds._generic_check._check_var(
        margin_par, 'margin_par',
        types=float,
        default=0.5,
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

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    return (
        key_diag, key_cam, indch, indref,
        parallel, is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        verb, plot,
    )


# ###############################################################
# ###############################################################
#              get intersection los vs plane
# ###############################################################


def _get_intersect_los_plane(
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    ptx=None,
    pty=None,
    ptz=None,
    vx=None,
    vy=None,
    vz=None,
):

    # ---------------
    # prepare output
    
    shape = vx.shape
    kk = np.full(shape, np.nan)
    
    # ------------
    # compute kk
    
    sca0 = vx * nin[0] + vy * nin[1] + vz * nin[2]
    sca1 = (
        (cent[0] - ptx) * nin[0]
        + (cent[1] - pty) * nin[1]
        + (cent[2] - ptz) * nin[2]
    )
    
    iok = np.abs(sca1) > 1e-6
    kk[iok] = sca1[iok] / sca0[iok]
    
    # --------------
    # 3d coordinates
    
    xx = ptx + kk * vx
    yy = pty + kk * vy
    zz = ptz + kk * vz
    
    # -----------------
    # local coordinates
    
    dx = xx - cent[0]
    dy = yy - cent[1]
    dz = zz - cent[2]
    
    x0 = dx * e0[0] + dy * e0[1] + dz * e0[2]
    x1 = dx * e1[0] + dy * e1[1] + dz * e1[2]
    
    return x0, x1, iok
    

# ###############################################################
# ###############################################################
#                   non-spectro
# ###############################################################


def _nonspectro(
    coll=None,
    key_cam=None,
    doptics=None,
    par=None,
    is2d=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    config=None,
):

    # -----------------
    # prepare apertures
    
    apertures = coll.get_optics_as_input_solid_angle(doptics['optics'])
    
    # -----------
    # prepare det
    
    k0, k1 = coll.dobj['camera'][key_cam]['dgeom']['outline']
    cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)
    dvect = coll.get_camera_unit_vectors(key=key_cam)
    
    det = {
        'cents_x': cx,
        'cents_y': cy,
        'cents_z': cz,
        'outline_x0': coll.ddata[k0]['data'],
        'outline_x1': coll.ddata[k1]['data'],
        'nin_x': np.full(cx.shape, dvect['nin_x']) if par else dvect['nin_x'],
        'nin_y': np.full(cx.shape, dvect['nin_y']) if par else dvect['nin_y'],
        'nin_z': np.full(cx.shape, dvect['nin_z']) if par else dvect['nin_z'],
        'e0_x': np.full(cx.shape, dvect['e0_x']) if par else dvect['e0_x'],
        'e0_y': np.full(cx.shape, dvect['e0_y']) if par else dvect['e0_y'],
        'e0_z': np.full(cx.shape, dvect['e0_z']) if par else dvect['e0_z'],
        'e1_x': np.full(cx.shape, dvect['e1_x']) if par else dvect['e1_x'],
        'e1_y': np.full(cx.shape, dvect['e1_y']) if par else dvect['e1_y'],
        'e1_z': np.full(cx.shape, dvect['e1_z']) if par else dvect['e1_z'],
    }
    
    # -------------
    # compute

    sang = _comp_solidangles.calc_solidangle_apertures(
        # observation points
        pts_x=ptsx,
        pts_y=ptsy,
        pts_z=ptsz,
        # polygons
        apertures=apertures,
        detectors=det,
        # possible obstacles
        config=config,
        # parameters
        visibility=False,
        return_vector=False,
        return_flat_pts=False,
        return_flat_det=False,
    )
    
    return {
        'sang': {
            'data': sang,
            'ref': None,
            'units': 'sr',
        },
    }



# ###############################################################
# ###############################################################
#                   Spectro
# ###############################################################


def _spectro(
    coll=None,
    doptics=None,
    indref=None,
    indch=None,
):


    # los_ref = np.r_[vx[]]

    # kapmax =

    return


# ###############################################################
# ###############################################################
#                   Plot
# ###############################################################


def _plot(**kwdargs):
    
    pass