# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import datastock as ds


# tofu
from ..geom import _comp_solidangles
from . import _class8_plot
from . import _generic_check


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
    # plotting
    indplot=None,
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    vmin_cam=None,
    vmax_cam=None,
    vmin_plane=None,
    vmax_plane=None,
):

    # ------------
    # check inputs

    (
        key_diag, key_cam, indch, indref,
        parallel, is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        verb, plot,
        indref, indplot,
        plot_config,
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
        # plotting
        indplot=indplot,
        plot_config=plot_config,
        config=config,
    )

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
    dx0 = [
        np.nanmin(x0) - margin_perp[0],
        np.nanmax(x0) + margin_perp[0],
    ]
    dx1 = [
        np.nanmin(x1) - margin_perp[1],
        np.nanmax(x1) + margin_perp[1],
    ]

    # cerate 2d grid
    n0 = int(np.ceil((dx0[1] - dx0[0]) / res[0])) + 2
    n1 = int(np.ceil((dx1[1] - dx1[0]) / res[1])) + 2

    x0 = np.linspace(dx0[0], dx0[1], n0)
    x1 = np.linspace(dx1[0], dx1[1], n1)

    ds = (x0[1] - x0[0]) * (x1[1] - x1[0])

    x0f = np.repeat(x0[:, None], n1, axis=1)
    x1f = np.repeat(x1[None, :], n0, axis=0)

    # derive 3d pts
    ptsx = pt_plane[0] + x0f*e0[0] + x1f*e1[0]
    ptsy = pt_plane[1] + x0f*e0[1] + x1f*e1[1]
    ptsz = pt_plane[2] + x0f*e0[2] + x1f*e1[2]

    # ----------
    # compute

    if spectro:
        dout = _spectro(
            coll=coll,
            key_diag=key_diag,
            key_cam=key_cam,
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
        _plot(
            coll=coll,
            is2d=is2d,
            # extra
            indplot=indplot,
            dax=dax,
            plot_config=plot_config,
            fs=fs,
            dmargin=dmargin,
            vmin_cam=vmin_cam,
            vmax_cam=vmax_cam,
            vmin_plane=vmin_plane,
            vmax_plane=vmax_plane,
            # dout
            **dout,
        )

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
    # plotting
    indplot=None,
    plot_config=None,
    config=None,
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
                sign='>=0',
            )
            indref[0] = indref[0] % n0
            indref[1] = indref[1] % n1

        else:
            indref = int(ds._generic_check._check_var(
                indref, 'indref',
                types=(float, int),
                sign='>=0',
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

    if margin_perp is None:
        margin_perp = 0.02

    if isinstance(margin_perp, (float, int)):
        margin_perp = [margin_perp, margin_perp]

    margin_perp = ds._generic_check._check_flat1darray(
        margin_perp, 'margin_perp',
        dtype=float,
        size=2,
        sign='>=0',
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

    # -----------
    # indplot

    if indplot is None:
        indplot = indref

    # -----------
    # plot_config

    if plot is True and plot_config is None:
        plot_config = config

    return (
        key_diag, key_cam, indch, indref,
        parallel, is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        verb, plot,
        indref, indplot,
        plot_config,
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
        summed=False,
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
    key_diag=None,
    key_cam=None,
    is2d=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
):

    # ------------
    # prepare

    dout = coll.get_raytracing_from_pts(
        key=key_diag,
        key_cam=key_cam,
        # mesh
        key_mesh=None,
        res_RZ=None,
        res_phi=None,
        # points
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        res_rock_curve=None,
        n0=None,
        n1=None,
        lamb=None,
        plot=False,
        plot_pixels=None,
        plot_config=None,
        vmin=None,
        vmax=None,
        append=False,
    )

    dout['sang'] = {
        'data': dout['sang'],
        'units': 'sr',
    }

    # counts

    return dout


# ###############################################################
# ###############################################################
#                   Plot
# ###############################################################


def _plot(
    coll=None,
    is2d=None,
    # dout
    key_diag=None,
    key_cam=None,
    indch=None,
    indref=None,
    los_ref=None,
    pt_ref=None,
    klos=None,
    e0=None,
    e1=None,
    ptsx=None,
    ptsy=None,
    ptsz=None,
    x0=None,
    x1=None,
    ds=None,
    sang=None,
    # extra
    indplot=None,
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    vmin_cam=None,
    vmax_cam=None,
    vmin_plane=None,
    vmax_plane=None,
    **kwdargs,
):

    # ------------
    # prepare

    (
        etend0, etend, sli,
        extent_cam, extent_plane,
        vmin_cam, vmax_cam,
        vmin_plane, vmax_plane,
        los_refr, los_refz,
    ) = _check_plot(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        is2d=is2d,
        sang=sang,
        x0=x0,
        x1=x1,
        ds=ds,
        pt_ref=pt_ref,
        los_ref=los_ref,
        indref=indref,
        indplot=indplot,
        vmin_cam=vmin_cam,
        vmax_cam=vmax_cam,
        vmin_plane=vmin_plane,
        vmax_plane=vmax_plane,
    )

    sang_plot = sang['data'][sli].ravel()

    # --------------
    # prepare

    dplot = coll.get_diagnostic_dplot(
        key=key_diag,
        key_cam=key_cam,
        optics=None,
        elements='o',
    )

    # --------------
    # pepare dax

    if dax is None:
        dax = _get_dax(
            is2d=is2d,
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # ----------
    # points

    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        ax.scatter(
            np.hypot(ptsx, ptsy).ravel(),
            ptsz.ravel(),
            c=sang_plot,
            s=4,
            marker='.',
            vmin=vmin_plane,
            vmax=vmax_plane,
        )

        # los_ref
        ax.plot(
            los_refr,
            los_refz,
            c='k',
            ls='-',
            lw=1.,
        )

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        ax.scatter(
            ptsx.ravel(),
            ptsy.ravel(),
            c=sang_plot,
            s=4,
            marker='.',
            vmin=vmin_plane,
            vmax=vmax_plane,
        )

        # los_ref
        ax.plot(
            pt_ref[0] + np.r_[0, los_ref[0]],
            pt_ref[1] + np.r_[0, los_ref[1]],
            c='k',
            ls='-',
            lw=1.,
        )

    kax = '3d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        ax.scatter(
            ptsx.ravel(),
            ptsy.ravel(),
            ptsz.ravel(),
            c=sang['data'][sli].ravel(),
            s=4,
            marker='.',
            vmin=vmin_plane,
            vmax=vmax_plane,
        )

        # los_ref
        ax.plot(
            pt_ref[0] + np.r_[0, los_ref[0]],
            pt_ref[1] + np.r_[0, los_ref[1]],
            pt_ref[2] + np.r_[0, los_ref[2]],
            c='k',
            ls='-',
            lw=1.,
        )

    # ------------------
    # integral per pixel

    kax = 'cam'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            im = ax.imshow(
                etend.T,
                extent=extent_cam,
                origin='lower',
                interpolation='nearest',
                vmin=vmin_cam,
                vmax=vmax_cam,
            )

            # ref pixel
            ax.plot(
                [indref[0]],
                [indref[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='k',
                ms=4,
            )

            # plot pixel
            ax.plot(
                [indplot[0]],
                [indplot[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='g',
                ms=4,
            )

        else:
            # per-pixel etendue
            ax.plot(
                etend0,
                ls='-',
                lw=1.,
                c='k',
                marker='.',
            )

            # plane-specific etendue
            ax.plot(
                etend,
                ls='-',
                lw=1.,
                c='b',
                marker='.',
            )

            # ref pixel
            ax.axvline(
                indref,
                c='k',
                ls='--',
                lw=1,
            )

            # plot pixel
            ax.axvline(
                indplot,
                c='g',
                ls='--',
                lw=1,
            )

    kax = 'cam0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            im0 = ax.imshow(
                etend0.T,
                extent=extent_cam,
                origin='lower',
                interpolation='nearest',
                vmin=vmin_cam,
                vmax=vmax_cam,
            )

            # ref pixel
            ax.plot(
                [indref[0]],
                [indref[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='k',
                ms=4,
            )

            # plot pixel
            ax.plot(
                [indplot[0]],
                [indplot[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='g',
                ms=4,
            )

            plt.colorbar(im, ax=[ax, dax['cam']['handle']])

    kax = 'camdiff'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            diff = (etend - etend0).T
            err = np.nanmax(np.abs(diff))
            imdiff = ax.imshow(
                diff,
                extent=extent_cam,
                origin='lower',
                interpolation='nearest',
                vmin=-err,
                vmax=err,
                cmap=plt.cm.seismic,
            )

            # ref pixel
            ax.plot(
                [indref[0]],
                [indref[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='k',
                ms=4,
            )

            # plot pixel
            ax.plot(
                [indplot[0]],
                [indplot[1]],
                marker='s',
                markerfacecolor='None',
                markeredgecolor='g',
                ms=4,
            )

            plt.colorbar(imdiff, ax=ax)

    # ------------------
    # integral per pixel

    kax = 'plane'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang['data'][sli].T,
            extent=extent_plane,
            origin='lower',
            interpolation='nearest',
            vmin=vmin_plane,
            vmax=vmax_plane,
        )

        plt.colorbar(im, ax=ax, label='solid angle (sr)')

    # ----------
    # diag geom

    _class8_plot._plot_diag_geom(
        dax=dax,
        key_cam=key_cam,
        dplot=dplot,
        is2d=coll.dobj['diagnostic'][key_diag]['is2d'],
    )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

    return dax


def _check_plot(
    coll=None,
    key_diag=None,
    key_cam=None,
    is2d=None,
    sang=None,
    x0=None,
    x1=None,
    ds=None,
    pt_ref=None,
    los_ref=None,
    indref=None,
    indplot=None,
    vmin_cam=None,
    vmax_cam=None,
    vmin_plane=None,
    vmax_plane=None,
):

    # ---------
    # data

    if is2d is True:
        # sang['data']
        pass
    else:
        pass

    # ---------
    # integral

    ketend = coll.dobj['diagnostic'][key_diag]['doptics'][key_cam]['etendue']
    etend0 = coll.ddata[ketend]['data']
    etend = np.nansum(np.nansum(sang['data'], axis=-1), axis=-1) * ds

    # ---------
    # extent

    # cam
    extent_cam = None
    if is2d:
        extent_cam = None
        sli = (indplot[0], indplot[1], slice(None), slice(None))
    else:
        sli = (indplot, slice(None), slice(None))

    # extent_plane
    dx0 = 0.5*(x0[1] - x0[0])
    dx1 = 0.5*(x1[1] - x1[0])
    extent_plane = (
        x0.min() - dx0,
        x0.max() + dx0,
        x1.min() - dx1,
        x1.max() + dx1,
    )

    # ----------
    # vmin, vmax

    if vmin_cam is None:
        vmin_cam = min(np.nanmin(etend0), np.nanmin(etend))
    if vmax_cam is None:
        vmax_cam = max(np.nanmax(etend0), np.nanmax(etend))

    if vmin_plane is None:
        vmin_plane = np.nanmin(sang['data'])
    if vmax_plane is None:
        vmax_plane = np.nanmax(sang['data'])

    # --------
    # los_ref

    add = np.linspace(0, 1, 20)
    los_refr = np.hypot(
        pt_ref[0] + add * los_ref[0],
        pt_ref[1] + add * los_ref[1],
    )
    los_refz = pt_ref[2] + add * los_ref[2]

    return (
        etend0, etend, sli,
        extent_cam, extent_plane,
        vmin_cam, vmax_cam,
        vmin_plane, vmax_plane,
        los_refr, los_refz,
    )


def _get_dax(
    is2d=None,
    fs=None,
    dmargin=None,
):

    # ----------
    # check

    if fs is None:
        fs = (15, 9)

    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.95,
            'wspace': 0.4, 'hspace': 0.5,
        },
    )

    # --------
    # prepare

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=5, nrows=6, **dmargin)

    # --------
    # create

    ax0 = fig.add_subplot(gs[:3, :2], aspect='equal', adjustable='datalim')
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')

    ax1 = fig.add_subplot(gs[3:, :2], aspect='equal', adjustable='datalim')
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')

    ax2 = fig.add_subplot(gs[:2, 2:], projection='3d')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_ylabel('Z (m)')

    if is2d is True:
        ax30 = fig.add_subplot(gs[2:4, 2], aspect='equal')
        ax30.set_ylabel('x1 (m)')
        ax30.set_xlabel('x0 (m)')
        ax30.set_title('intregral', size=12, fontweight='bold')

        ax31 = fig.add_subplot(gs[2:4, 3], sharex=ax30, sharey=ax30)
        ax31.set_xlabel('x0 (m)')
        ax31.set_title('etendue', size=12, fontweight='bold')

        ax32 = fig.add_subplot(gs[2:4, 4], sharex=ax30, sharey=ax30)
        ax32.set_xlabel('x0 (m)')
        ax32.set_title('difference', size=12, fontweight='bold')

    else:
        ax30 = fig.add_subplot(gs[2:4, 2:])
        ax30.set_ylabel('x1 (m)')

    ax4 = fig.add_subplot(gs[4:, 2:], aspect='equal', adjustable='datalim')
    ax4.set_ylabel('x1 (m)')
    ax4.set_xlabel('x0 (m)')

    # cax = fig.add_subplot(gs[3, -1])

    dax = {
        'cross': {'handle': ax0, 'type': 'cross'},
        'hor': {'handle': ax1, 'type': 'hor'},
        '3d': {'handle': ax2, 'type': '3d'},
        'cam': {'handle': ax30, 'type': 'camera'},
        'plane': {'handle': ax4, 'type': 'misc'},
    }
    if is2d is True:
        dax['cam0'] = {'handle': ax31, 'type': 'camera'}
        dax['camdiff'] = {'handle': ax32, 'type': 'camera'}

    return dax  # , cax
