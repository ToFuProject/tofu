# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# tofu
from ..geom import _comp_solidangles
from . import _class8_plot
from . import _generic_check
from . import _class8_plot_coverage_slice_utils as _utils


# #####################################################
# #####################################################
#                   Compute
# #####################################################


def _compute(
    coll=None,
    key_cam=None,
    doptics=None,
    is2d=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    config=None,
    visibility=None,
    # unused
    **kwdargs,
):

    dout = {}
    for kcam in key_cam:

        # -----------------
        # prepare apertures

        pinhole = doptics[kcam]['pinhole']
        if pinhole is False:
            paths = doptics[kcam]['paths']

        apertures = coll.get_optics_as_input_solid_angle(
            doptics[kcam]['optics']
        )

        # -----------
        # prepare det

        k0, k1 = coll.dobj['camera'][kcam]['dgeom']['outline']
        cx, cy, cz = coll.get_camera_cents_xyz(key=kcam)
        dvect = coll.get_camera_unit_vectors(key=kcam)

        # -------------
        # compute
        # -------------

        ref = (
            coll.dobj['camera'][kcam]['dgeom']['ref']
            + tuple([None for ii in ptsx.shape])
        )
        par = coll.dobj['camera'][kcam]['dgeom']['parallel']

        # --------
        # pinhole

        if pinhole is True:

            sh = cx.shape
            det = {
                'cents_x': cx,
                'cents_y': cy,
                'cents_z': cz,
                'outline_x0': coll.ddata[k0]['data'],
                'outline_x1': coll.ddata[k1]['data'],
                'nin_x': np.full(sh, dvect['nin_x']) if par else dvect['nin_x'],
                'nin_y': np.full(sh, dvect['nin_y']) if par else dvect['nin_y'],
                'nin_z': np.full(sh, dvect['nin_z']) if par else dvect['nin_z'],
                'e0_x': np.full(sh, dvect['e0_x']) if par else dvect['e0_x'],
                'e0_y': np.full(sh, dvect['e0_y']) if par else dvect['e0_y'],
                'e0_z': np.full(sh, dvect['e0_z']) if par else dvect['e0_z'],
                'e1_x': np.full(sh, dvect['e1_x']) if par else dvect['e1_x'],
                'e1_y': np.full(sh, dvect['e1_y']) if par else dvect['e1_y'],
                'e1_z': np.full(sh, dvect['e1_z']) if par else dvect['e1_z'],
            }

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
                visibility=visibility,
                return_vector=False,
                return_flat_pts=False,
                return_flat_det=False,
            )

        # -----------
        # collimator

        else:

            sang = np.full(cx.shape + ptsx.shape, np.nan)
            for ii, indch in enumerate(np.ndindex(cx.shape)):

                det = {
                    'cents_x': cx[indch],
                    'cents_y': cy[indch],
                    'cents_z': cz[indch],
                    'outline_x0': coll.ddata[k0]['data'],
                    'outline_x1': coll.ddata[k1]['data'],
                    'nin_x': dvect['nin_x'] if par else dvect['nin_x'][indch],
                    'nin_y': dvect['nin_y'] if par else dvect['nin_y'][indch],
                    'nin_z': dvect['nin_z'] if par else dvect['nin_z'][indch],
                    'e0_x': dvect['e0_x'] if par else dvect['e0_x'][indch],
                    'e0_y': dvect['e0_y'] if par else dvect['e0_y'][indch],
                    'e0_z': dvect['e0_z'] if par else dvect['e0_z'][indch],
                    'e1_x': dvect['e1_x'] if par else dvect['e1_x'][indch],
                    'e1_y': dvect['e1_y'] if par else dvect['e1_y'][indch],
                    'e1_z': dvect['e1_z'] if par else dvect['e1_z'][indch],
                }

                sliap = indch + (slice(None),)
                lap = [
                    doptics[kcam]['optics'][ii]
                    for ii in paths[sliap].nonzero()[0]
                ]
                api = {kap: apertures[kap] for kap in lap}

                sli = (indch, slice(None), slice(None))

                sang[sli] = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=ptsx,
                    pts_y=ptsy,
                    pts_z=ptsz,
                    # polygons
                    apertures=api,
                    detectors=det,
                    # possible obstacles
                    config=config,
                    # parameters
                    summed=False,
                    visibility=visibility,
                    return_vector=False,
                    return_flat_pts=False,
                    return_flat_det=False,
                )

        axis = tuple([ii for ii in range(len(cx.shape))])

        dout[kcam] = {
            'sang0': {
                'data': sang,
                'ref': ref,
                'units': 'sr',
            },
            'ndet': {
                'data': np.sum(sang > 0., axis=axis),
                'units': '',
                'ref': ref[-2:],
            },
            'axis': axis,
        }

    return dout


# #######################################################
# #######################################################
#       Check plot args
# #######################################################


def _check_plot(
    coll=None,
    key_diag=None,
    key_cam=None,
    vect=None,
    is2d=None,
    spectro=None,
    sang0=None,
    sang=None,
    ndet=None,
    sang_lamb=None,
    x0=None,
    x1=None,
    dS=None,
    pt_ref=None,
    los_ref=None,
    indref=None,
    indplot=None,
    dvminmax=None,
):

    # ---------
    # prepare
    # ---------

    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']

    # ---------
    # integral
    # ---------

    if sang0 is not None and dS is not None:
        etend_plane0 = np.nansum(
            np.nansum(sang0['data'], axis=-1),
            axis=-1,
        ) * dS
    else:
        etend_plane0 = None

    etend_plane = None
    etend = None
    if vect == 'nin':
        ketend = doptics['etendue']
        etend0 = coll.ddata[ketend]['data']
    else:
        etend0 = None

    etend_lamb = None
    etend_plane_lamb = None

    # ---------
    # extent
    # ---------

    # cam
    extent_cam = None
    if is2d:
        extent_cam = None
        sli = (indplot[0], indplot[1], slice(None), slice(None))
    else:
        sli = (indplot, slice(None), slice(None))

    axis = tuple([ii for ii in range(len(shape_cam))])

    # extent_plane
    if x0 is not None:
        dx0 = 0.5*(x0[1] - x0[0])
        dx1 = 0.5*(x1[1] - x1[0])
        extent_plane = (
            x0.min() - dx0,
            x0.max() + dx0,
            x1.min() - dx1,
            x1.max() + dx1,
        )
    else:
        extent_plane = None

    # --------
    # vminmax
    # --------

    dvminmax = _utils._check_dvminmax(
        dvminmax=dvminmax,
        etend0=etend0,
        etend_plane0=etend_plane0,
        etend=etend,
        etend_plane=etend_plane,
        etend_lamb=etend_lamb,
        etend_plane_lamb=etend_plane_lamb,
        sang0=sang0,
        sang=sang,
        ndet=ndet,
    )

    # --------
    # los_ref

    if pt_ref is not None:
        add = np.linspace(0, 1, 20)
        los_refr = np.hypot(
            pt_ref[0] + add * los_ref[0],
            pt_ref[1] + add * los_ref[1],
        )
        los_refz = pt_ref[2] + add * los_ref[2]
    else:
        los_refr, los_refz = None, None

    return (
        etend0, etend, etend_lamb,
        etend_plane0, etend_plane, etend_plane_lamb,
        sli, axis,
        extent_cam, extent_plane,
        dvminmax,
        los_refr, los_refz,
    )


# #########################################################
# #########################################################
#             Plot from LOS
# #########################################################


def _plot_from_los(
    coll=None,
    key_diag=None,
    key_cam=None,
    # dout
    is2d=None,
    spectro=None,
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
    dS=None,
    vect=None,
    sang0=None,
    sang=None,
    sang_lamb=None,
    # extra
    indplot=None,
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    dvminmax=None,
    **kwdargs,
):

    # ------------
    # prepare

    (
        etend0, etend, etend_lamb,
        etend_plane0, etend_plane, etend_plane_lamb,
        sli, axis,
        extent_cam, extent_plane,
        dvminmax,
        los_refr, los_refz,
    ) = _check_plot(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        vect=vect,
        is2d=is2d,
        spectro=spectro,
        sang0=sang0,
        sang=sang,
        sang_lamb=sang_lamb,
        x0=x0,
        x1=x1,
        dS=dS,
        pt_ref=pt_ref,
        los_ref=los_ref,
        indref=indref,
        indplot=indplot,
        dvminmax=dvminmax,
    )

    # --------------
    # prepare
    # --------------

    ix1 = int(x1.size/2)
    # sang_plot = sang0['data'][sli].ravel()
    sang_tot_plot = np.sum(sang0['data'], axis=axis)

    # tit
    tit = (
        "Solid angle subtended from points in the plasma by:\n"
        f"{key_diag} - {key_cam}"
    )

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
        dax = _get_dax_los(
            is2d=is2d,
            spectro=spectro,
            fs=fs,
            dmargin=dmargin,
            vect=vect,
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
            c=sang_tot_plot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
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
            c=sang_tot_plot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
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
            c=sang_tot_plot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
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

    kax = 'cam_plane0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            im = ax.imshow(
                etend_plane0.T,
                extent=extent_cam,
                origin='lower',
                interpolation='nearest',
                vmin=dvminmax.get('cam0', {}).get('min'),
                vmax=dvminmax.get('cam0', {}).get('max'),
            )

            _utils._add_marker(ax, indref, indplot)

        else:
            # per-pixel etendue
            ax.plot(
                etend0,
                ls='-',
                lw=1.,
                c='k',
                marker='.',
                label='Analytical',
            )

            # plane-specific etendue
            ax.plot(
                etend_plane0,
                ls='-',
                lw=1.,
                c='b',
                marker='.',
                label='Numerical',
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

            ax.legend()

    kax = 'cam_etend0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            im0 = ax.imshow(
                etend0.T,
                extent=extent_cam,
                origin='lower',
                interpolation='nearest',
                vmin=dvminmax.get('cam0', {}).get('min'),
                vmax=dvminmax.get('cam0', {}).get('max'),
            )

            _utils._add_marker(ax, indref, indplot)
            plt.colorbar(im, ax=[ax, dax['cam_plane0']['handle']])

    kax = 'cam_diff0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if is2d is True:

            # plane-specific etendue
            diff = (etend_plane0 - etend0).T
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
            _utils._add_marker(ax, indref, indplot)
            plt.colorbar(imdiff, ax=ax)

    # -------------------------------------
    # integral per pixel with rocking curve

    kax = 'cam_plane'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        im = ax.imshow(
            etend_plane.T,
            extent=extent_cam,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('cam', {}).get('min'),
            vmax=dvminmax.get('cam', {}).get('max'),
        )

        # ref pixel
        _utils._add_marker(ax, indref, indplot)

    kax = 'cam_etend'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        im0 = ax.imshow(
            etend.T,
            extent=extent_cam,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('cam', {}).get('min'),
            vmax=dvminmax.get('cam', {}).get('max'),
        )

        # ref pixel
        _utils._add_marker(ax, indref, indplot)
        plt.colorbar(im, ax=[ax, dax['cam_plane']['handle']])

    kax = 'cam_diff'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        diff = (etend_plane - etend).T
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
        _utils._add_marker(ax, indref, indplot)
        plt.colorbar(imdiff, ax=ax)

    # -----------------------------------------------
    # integral per pixel with rocking curve and dlamb

    kax = 'cam_plane_lamb'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        im = ax.imshow(
            etend_plane_lamb.T,
            extent=extent_cam,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('cam_lamb', {}).get('min'),
            vmax=dvminmax.get('cam_lamb', {}).get('max'),
        )

        # ref pixel
        _utils._add_marker(ax, indref, indplot)

    kax = 'cam_etend_lamb'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        im0 = ax.imshow(
            etend_lamb.T,
            extent=extent_cam,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('cam_lamb', {}).get('min'),
            vmax=dvminmax.get('cam_lamb', {}).get('max'),
        )

        # ref pixel
        _utils._add_marker(ax, indref, indplot)
        plt.colorbar(im0, ax=[ax, dax['cam_plane_lamb']['handle']])

    kax = 'cam_diff_lamb'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        diff = (etend_plane_lamb - etend_lamb).T
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
        _utils._add_marker(ax, indref, indplot)
        plt.colorbar(imdiff, ax=ax)

    # -------------------------------------------
    # sang per plane point for selected pixel

    kax = 'plane0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang_tot_plot.T,
            # sang0['data'][sli].T,
            extent=extent_plane,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('plane0', {}).get('min'),
            vmax=dvminmax.get('plane0', {}).get('max'),
        )

        ax.axhline(x1[ix1], c='w', ls='--', lw=0.5)

        plt.colorbar(
            im,
            ax=ax,
            cax=dax.get('colorbar', {}).get('handle'),
            label='solid angle (sr)',
        )

    kax = 'plane'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang['data'][sli].T,
            extent=extent_plane,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
        )

        plt.colorbar(im, ax=ax, label='solid angle (sr)')

    # -------------------------------------------
    # slice through sang 2d map

    kax = 'slice'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.plot(
            x0,
            sang0['data'][..., ix1].T.reshape((x0.size, -1)),
        )

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

    # ------------
    # figure title

    list(dax.values())[0]['handle'].figure.suptitle(
        tit,
        size=14,
        fontweight='bold',
    )

    return dax


# #########################################################
# #########################################################
#             Plot from mesh
# #########################################################


def _plot_from_mesh(
    coll=None,
    key_diag=None,
    key_cam=None,
    # params
    phi=None,
    Z=None,
    # dout
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # dout
    dout=None,
    # extra
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    dvminmax=None,
    **kwdargs,
):

    # ------------
    # prepare

    (
        sang_tot, ndet_tot, dvminmax
    ) = _check_plot_mesh(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        dout=dout,
        dvminmax=dvminmax,
    )

    # --------------
    # prepare
    # --------------

    # tit
    tit = (
        "Solid angle subtended from points in the plasma by:\n"
        f"{key_diag} - {key_cam}"
    )

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
    # --------------

    if dax is None:
        dax = _get_dax_mesh(
            phi=phi,
            Z=Z,
            # options
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # --------------
    # points - hor
    # --------------

    # ----
    # sang

    kax = 'hor_sang'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        im = ax.scatter(
            ptsx,
            ptsy,
            c=sang_tot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
        )

        kaxc = f'{kax}_cbar'
        if dax.get(kaxc) is not None:
            plt.colorbar(im, cax=dax[kaxc]['handle'])

    # -----
    # ndet

    kax = 'hor_ndet'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        im = ax.scatter(
            ptsx,
            ptsy,
            c=ndet_tot,
            s=4,
            marker='.',
            vmin=dvminmax.get('ndet', {}).get('min'),
            vmax=dvminmax.get('ndet', {}).get('max'),
        )

        kaxc = f'{kax}_cbar'
        if dax.get(kaxc) is not None:
            plt.colorbar(im, cax=dax[kaxc]['handle'])

    # --------------
    # points - cross
    # --------------

    # ----
    # sang

    kax = 'cross_sang'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        im = ax.scatter(
            np.hypot(ptsx, ptsy),
            ptsz,
            c=sang_tot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
        )

        kaxc = f'{kax}_cbar'
        if dax.get(kaxc) is not None:
            plt.colorbar(im, cax=dax[kaxc]['handle'])

    # -----
    # ndet

    kax = 'cross_ndet'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        im = ax.scatter(
            np.hypot(ptsx, ptsy),
            ptsz,
            c=ndet_tot,
            s=4,
            marker='.',
            vmin=dvminmax.get('ndet', {}).get('min'),
            vmax=dvminmax.get('ndet', {}).get('max'),
        )

        kaxc = f'{kax}_cbar'
        if dax.get(kaxc) is not None:
            plt.colorbar(im, cax=dax[kaxc]['handle'])

    # --------------
    # points - 3d
    # --------------

    kax = '3d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # pts
        im = ax.scatter(
            ptsx,
            ptsy,
            ptsz,
            c=sang_tot,
            s=4,
            marker='.',
            vmin=dvminmax.get('plane', {}).get('min'),
            vmax=dvminmax.get('plane', {}).get('max'),
        )

    # ----------
    # diag geom
    # ----------

    _class8_plot._plot_diag_geom(
        dax=dax,
        key_cam=key_cam,
        dplot=dplot,
        is2d=coll.dobj['diagnostic'][key_diag]['is2d'],
    )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        axtype = 'cross'
        lax = [kax for kax, vax in dax.items() if vax['type'] == axtype]
        for kax in lax:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=axtype, dLeg=False)

        axtype = 'hor'
        lax = [kax for kax, vax in dax.items() if vax['type'] == axtype]
        for kax in lax:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=axtype, dLeg=False)

    # ------------
    # figure title

    list(dax.values())[0]['handle'].figure.suptitle(
        tit,
        size=14,
        fontweight='bold',
    )

    return dax


# #######################################################
# #######################################################
#       Check plot mesh
# #######################################################


def _check_plot_mesh(
    coll=None,
    key_diag=None,
    key_cam=None,
    dout=None,
    dvminmax=None,
):

    # ---------
    # prepare
    # ---------

    for ii, kcam in enumerate(key_cam):
        axis = dout[kcam]['axis']
        if ii == 0:
            sang_tot = np.sum(dout[kcam]['sang0']['data'], axis=axis)
            ndet_tot = dout[kcam]['ndet']['data']
        else:
            sang_tot += np.sum(dout[kcam]['sang0']['data'], axis=axis)
            ndet_tot += dout[kcam]['ndet']['data']

    ndet_tot[ndet_tot == 0] = np.nan

    # --------
    # vminmax
    # --------

    dvminmax = _utils._check_dvminmax(
        dvminmax=dvminmax,
        sang0={'data': sang_tot},
        ndet={'data': ndet_tot},
    )

    return sang_tot, ndet_tot, dvminmax


# ############################################################
# ############################################################
#                 dax for LOS
# ############################################################


def _get_dax_los(
    is2d=None,
    spectro=None,
    fs=None,
    dmargin=None,
    vect=None,
):

    # ----------
    # check

    if fs is None:
        fs = (16, 10)

    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.92,
            'left': 0.05, 'right': 0.95,
            'wspace': 0.4, 'hspace': 0.5,
        },
    )

    # --------
    # prepare

    fig = plt.figure(figsize=fs)

    nn = 4
    gs = gridspec.GridSpec(ncols=5*nn+1, nrows=6, **dmargin)

    # --------
    # create

    ax0 = fig.add_subplot(gs[:3, :2*nn], aspect='equal', adjustable='datalim')
    ax0.set_xlabel('X (m)', size=12, fontweight='bold')
    ax0.set_ylabel('Y (m)', size=12, fontweight='bold')

    ax1 = fig.add_subplot(gs[3:, :2*nn], aspect='equal', adjustable='datalim')
    ax1.set_xlabel('R (m)', size=12, fontweight='bold')
    ax1.set_ylabel('Z (m)', size=12, fontweight='bold')

    ax2 = fig.add_subplot(gs[:2, 2*nn+1:3*nn+1], projection='3d')
    ax2.set_xlabel('X (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Z (m)', size=12, fontweight='bold')

    # etendue
    if vect == 'nin':
        if is2d is True:
            ax30 = fig.add_subplot(gs[2:4, 2*nn:3*nn], aspect='equal')
            ax30.set_ylabel('x1 (m)')
            ax30.set_xlabel('x0 (m)')
            ax30.set_title('integral', size=12, fontweight='bold')

            ax31 = fig.add_subplot(
                gs[2:4, 3*nn:4*nn],
                sharex=ax30,
                sharey=ax30,
            )
            ax31.set_xlabel('x0 (m)')
            ax31.set_title('etendue', size=12, fontweight='bold')

            ax32 = fig.add_subplot(
                gs[2:4, 4*nn:-1],
                sharex=ax30,
                sharey=ax30,
            )
            ax32.set_xlabel('x0 (m)')
            ax32.set_title('difference', size=12, fontweight='bold')

        else:
            ax30 = fig.add_subplot(gs[:2, 3*nn+2:])
            ax30.set_xlabel('channel', size=12, fontweight='bold')
            ax30.set_ylabel('Etendue (m2.sr)', size=12, fontweight='bold')

    ax4 = fig.add_subplot(
        gs[2:4, 2*nn+1:-1],
        aspect='equal',
        adjustable='datalim',
    )
    ax4.set_ylabel('x1 (m)', size=12, fontweight='bold')
    ax4.set_xlabel('x0 (m)', size=12, fontweight='bold')

    ax4c = fig.add_subplot(gs[2:4, -1])

    ax5 = fig.add_subplot(gs[4:, 2*nn+1:-1], sharex=ax4)
    ax5.set_ylabel('sang (sr)', size=12, fontweight='bold')
    ax5.set_xlabel('x0 (m)', size=12, fontweight='bold')

    # dict
    dax = {
        'cross': {'handle': ax0, 'type': 'cross'},
        'hor': {'handle': ax1, 'type': 'hor'},
        '3d': {'handle': ax2, 'type': '3d'},
        'plane0': {'handle': ax4, 'type': 'misc'},
        'colorbar': {'handle': ax4c, 'type': 'misc'},
        'slice': {'handle': ax5, 'type': 'misc'},
    }
    if vect == 'nin':
        dax['cam_plane0'] = {'handle': ax30, 'type': 'camera'}
        if is2d is True:
            dax['cam_etend0'] = {'handle': ax31, 'type': 'camera'}
            dax['cam_diff0'] = {'handle': ax32, 'type': 'camera'}

    return dax  # , cax


# ############################################################
# ############################################################
#                 dax for mesh
# ############################################################


def _get_dax_mesh(
    phi=None,
    Z=None,
    # options
    fs=None,
    dmargin=None,
):

    # ----------
    # check
    # ----------

    if fs is None:
        fs = (18, 8)

    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.90,
            'left': 0.05, 'right': 0.95,
            'wspace': 0.40, 'hspace': 0.5,
        },
    )

    # ------------
    # prepare data
    # ------------

    if phi is None:
        xlab = 'X (m)'
        ylab = 'Y (m)'
        xlab2 = 'R (m)'
        ylab2 = 'Z (m)'
        tit_sang = f"sang at Z = {round(Z, ndigits=2)} m"
        tit_ndet = f"ndet at Z = {round(Z, ndigits=2)} m"
    else:
        xlab = 'R (m)'
        ylab = 'Z (m)'
        xlab2 = 'X (m)'
        ylab2 = 'Y (m)'
        tit_sang = f"sang at phi = {round(phi*180./np.pi, ndigits=0)} deg"
        tit_ndet = f"ndet at phi = {round(phi*180./np.pi, ndigits=0)} deg"

    # ------------
    # prepare fig
    # ------------

    fig = plt.figure(figsize=fs)

    na, ni, nc = 4, 2, 1
    gs = gridspec.GridSpec(ncols=na*3+ni*2+nc*2, nrows=2, **dmargin)

    # ------------
    # create axes
    # ------------

    # ax0
    ax0 = fig.add_subplot(
        gs[:, na+ni:2*na+ni],
        aspect='equal',
        adjustable='datalim',
    )
    ax0.set_xlabel(xlab, size=12, fontweight='bold')
    ax0.set_ylabel(ylab, size=12, fontweight='bold')
    ax0.set_title(tit_sang, size=12, fontweight='bold')

    # ax0c
    ax0c = fig.add_subplot(gs[:, 2*na+ni])

    # ax1
    ax1 = fig.add_subplot(
        gs[:, 2*na+2*ni+nc:3*na+2*ni+nc],
        aspect='equal',
        adjustable='datalim',
    )
    ax1.set_xlabel(xlab, size=12, fontweight='bold')
    ax1.set_ylabel(ylab, size=12, fontweight='bold')
    ax1.set_title(tit_ndet, size=12, fontweight='bold')

    # ax1c
    ax1c = fig.add_subplot(gs[:, 3*na+2*ni+nc])

    # ax2
    ax2 = fig.add_subplot(gs[0, :na], aspect='equal', projection='3d')
    ax2.set_xlabel('X (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', size=12, fontweight='bold')
    ax2.set_zlabel('Z (m)', size=12, fontweight='bold')

    # ax3
    ax3 = fig.add_subplot(gs[1, :na], aspect='equal')
    ax3.set_xlabel(xlab2, size=12, fontweight='bold')
    ax3.set_ylabel(ylab2, size=12, fontweight='bold')

    # dict
    dax = {
        '3d': {'handle': ax2, 'type': '3d'},
    }
    if phi is None:
        dax['hor_sang'] = {'handle': ax0, 'type': 'hor'}
        dax['hor_ndet'] = {'handle': ax1, 'type': 'hor'}
        dax['hor_sang_cbar'] = {'handle': ax0c, 'type': 'cbar'}
        dax['hor_ndet_cbar'] = {'handle': ax1c, 'type': 'cbar'}
        dax['cross_sang'] = {'handle': ax3, 'type': 'cross'}
    else:
        dax['cross_sang'] = {'handle': ax0, 'type': 'cross'}
        dax['cross_ndet'] = {'handle': ax1, 'type': 'cross'}
        dax['cross_sang_cbar'] = {'handle': ax0c, 'type': 'cbar'}
        dax['cross_ndet_cbar'] = {'handle': ax1c, 'type': 'cbar'}
        dax['hor_sang'] = {'handle': ax3, 'type': 'hor'}

    return dax
