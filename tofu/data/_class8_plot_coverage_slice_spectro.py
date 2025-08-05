# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# tofu
from . import _class8_plot
from . import _generic_check
from . import _class8_plot_coverage_slice_utils as _utils


# ########################################################
# ########################################################
#                   Spectro
# ########################################################


def _compute(
    coll=None,
    key_diag=None,
    key_cam=None,
    is2d=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # solid angle
    n0=None,
    n1=None,
    # res
    res_lamb=None,
    # unused
    **kwdargs,
):

    key_cam = key_cam[0]

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
        # solid angle
        n0=n0,
        n1=n1,
        # others
        lamb0=None,
        res_lamb=res_lamb,
        plot=False,
        plot_pixels=None,
        plot_config=None,
        vmin=None,
        vmax=None,
        append=False,
    )

    dout['sang0'] = {
        'data': dout['sang0'],
        'units': 'sr',
    }
    dout['sang'] = {
        'data': dout['sang'],
        'units': 'sr',
    }
    dout['sang_lamb'] = {
        'data': dout['sang_lamb'],
        'units': 'sr.m',
    }

    return dout


# #######################################################
# #######################################################
#           Check plot args
# #######################################################


def _check_plot(
    coll=None,
    key_diag=None,
    key_cam=None,
    is2d=None,
    spectro=None,
    sang0=None,
    sang=None,
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

    doptics = coll.dobj['diagnostic'][key_diag]['doptics'][key_cam]

    # ---------
    # integral
    # ---------

    etend_plane0 = np.nansum(
        np.nansum(sang0['data'], axis=-1),
        axis=-1
    ) * dS
    etend_plane = np.nansum(
        np.nansum(sang['data'], axis=-1),
        axis=-1,
    ) * dS

    ketend = doptics['etendue0']
    etend0 = coll.ddata[ketend]['data']
    ketend = doptics['etendue']
    etend = coll.ddata[ketend]['data']

    dlamb = coll.get_diagnostic_data(
        key_diag,
        key_cam=key_cam,
        data='dlamb',
    )[0][key_cam]

    etend_lamb = etend * dlamb
    etend_plane_lamb = sang_lamb['data'] * dS

    # ---------
    # extent
    # ---------

    # cam
    extent_cam = None
    if is2d:
        extent_cam = None
        sli = (indplot[0], indplot[1], slice(None), slice(None))
        axis = (0, 1)
    else:
        sli = (indplot, slice(None), slice(None))
        axis = 0

    # extent_plane
    dx0 = 0.5*(x0[1] - x0[0])
    dx1 = 0.5*(x1[1] - x1[0])
    extent_plane = (
        x0.min() - dx0,
        x0.max() + dx0,
        x1.min() - dx1,
        x1.max() + dx1,
    )

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
    )

    # --------
    # los_ref
    # --------

    add = np.linspace(0, 1, 20)
    los_refr = np.hypot(
        pt_ref[0] + add * los_ref[0],
        pt_ref[1] + add * los_ref[1],
    )
    los_refz = pt_ref[2] + add * los_ref[2]

    return (
        etend0, etend, etend_lamb,
        etend_plane0, etend_plane, etend_plane_lamb,
        sli, axis,
        extent_cam, extent_plane,
        dvminmax,
        los_refr, los_refz,
    )


# ###############################################################
# ###############################################################
#                   Plot from LOS
# ###############################################################


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
        # minmax
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
        dax = _get_dax(
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


# ############################################################
# ############################################################
#                 dax
# ############################################################


def _get_dax(
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

    gs = gridspec.GridSpec(ncols=8, nrows=8, **dmargin)

    # --------
    # create

    # geometry
    ax0 = fig.add_subplot(
        gs[:2, :2],
        aspect='equal',
        adjustable='datalim',
    )
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')

    ax1 = fig.add_subplot(
        gs[2:4, :2],
        aspect='equal',
        adjustable='datalim',
    )
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')

    ax2 = fig.add_subplot(gs[4:, :2], projection='3d')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_ylabel('Z (m)')

    # images for etend0
    ax30 = fig.add_subplot(gs[:2, 2:4], aspect='equal')
    ax30.set_ylabel('x1 (m)')
    ax30.set_xlabel('x0 (m)')
    ax30.set_title(
        'etendue\nw/o rock. curve',
        size=12,
        fontweight='bold',
    )

    ax31 = fig.add_subplot(gs[:2, 4:6], sharex=ax30, sharey=ax30)
    ax31.set_xlabel('x0 (m)')
    ax31.set_title(
        'integral\nw/o rock. curve',
        size=12,
        fontweight='bold',
    )

    ax32 = fig.add_subplot(gs[:2, 6:], sharex=ax30, sharey=ax30)
    ax32.set_xlabel('x0 (m)')
    ax32.set_title('difference', size=12, fontweight='bold')

    # images for etend
    ax40 = fig.add_subplot(gs[2:4, 2:4], aspect='equal')
    ax40.set_ylabel('x1 (m)')
    ax40.set_xlabel('x0 (m)')
    ax40.set_title(
        'etendue\nwith rock. curve',
        size=12,
        fontweight='bold',
    )

    ax41 = fig.add_subplot(gs[2:4, 4:6], sharex=ax30, sharey=ax30)
    ax41.set_xlabel('x0 (m)')
    ax41.set_title(
        'integral\nwith rock. curve',
        size=12,
        fontweight='bold',
    )

    ax42 = fig.add_subplot(gs[2:4, 6:], sharex=ax30, sharey=ax30)
    ax42.set_xlabel('x0 (m)')
    ax42.set_title('difference', size=12, fontweight='bold')

    # images for etend * dlamb
    ax50 = fig.add_subplot(gs[4:6, 2:4], aspect='equal')
    ax50.set_ylabel('x1 (m)')
    ax50.set_xlabel('x0 (m)')
    ax50.set_title(
        'etendue * dlamb\nwith rock. curve',
        size=12,
        fontweight='bold',
    )

    ax51 = fig.add_subplot(gs[4:6, 4:6], sharex=ax30, sharey=ax30)
    ax51.set_xlabel('x0 (m)')
    ax51.set_title(
        'integral * dlamb\nwith rock. curve',
        size=12,
        fontweight='bold',
    )

    ax52 = fig.add_subplot(gs[4:6, 6:], sharex=ax30, sharey=ax30)
    ax52.set_xlabel('x0 (m)')
    ax52.set_title('difference', size=12, fontweight='bold')

    # plane
    ax60 = fig.add_subplot(
        gs[-2:, 2:5],
        aspect='equal',
        adjustable='datalim',
    )
    ax60.set_ylabel('x1 (m)')
    ax60.set_xlabel('x0 (m)')

    ax61 = fig.add_subplot(
        gs[-2:, 5:],
        aspect='equal',
        adjustable='datalim',
    )
    ax61.set_ylabel('x1 (m)')
    ax61.set_xlabel('x0 (m)')

    # dict
    dax = {
        'cross': {'handle': ax0, 'type': 'cross'},
        'hor': {'handle': ax1, 'type': 'hor'},
        '3d': {'handle': ax2, 'type': '3d'},
        'cam_etend0': {'handle': ax30, 'type': 'camera'},
        'cam_plane0': {'handle': ax31, 'type': 'camera'},
        'cam_diff0': {'handle': ax32, 'type': 'camera'},
        'cam_etend': {'handle': ax40, 'type': 'camera'},
        'cam_plane': {'handle': ax41, 'type': 'camera'},
        'cam_diff': {'handle': ax42, 'type': 'camera'},
        'cam_etend_lamb': {'handle': ax50, 'type': 'camera'},
        'cam_plane_lamb': {'handle': ax51, 'type': 'camera'},
        'cam_diff_lamb': {'handle': ax52, 'type': 'camera'},
        'plane0': {'handle': ax60, 'type': 'misc'},
        'plane': {'handle': ax61, 'type': 'misc'},
    }

    return dax  # , cax
