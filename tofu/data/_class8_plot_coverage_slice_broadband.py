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


# #####################################################
# #####################################################
#                   Compute
# #####################################################


def _compute(
    coll=None,
    key_diag=None,
    key_cam=None,
    doptics=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    config=None,
    visibility=None,
    # unused
    **kwdargs,
):

    ddata, din = coll.compute_diagnostic_sang_vect_from_pts(
        key_diag=key_diag,
        key_cam=key_cam,
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        visibility=visibility,
        config=config,
        return_vect=False,
    )

    return ddata


# #######################################################
# #######################################################
#       Check plot args
# #######################################################


def _check_plot_los(
    coll=None,
    key_diag=None,
    key_cam=None,
    vect=None,
    dout=None,
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
    is2d = coll.dobj['camera'][key_cam[0]]['dgeom']['nd'] == '2d'

    shape = (-1,) + dout[key_cam[0]]['sang']['data'].shape[-2:]
    for ii, kcam in enumerate(key_cam):
        axis_cam = dout[kcam]['axis_cam']
        if ii == 0:
            sang = np.reshape(dout[kcam]['sang']['data'], shape)
            sang_tot = np.sum(dout[kcam]['sang']['data'], axis=axis_cam)
            ndet = dout[kcam]['ndet']['data'].astype(float)
        else:
            sang = np.concatenate(
                (
                    sang,
                    np.reshape(dout[kcam]['sang']['data'], shape),
                ),
                axis=0,
            )
            sang_tot += np.sum(dout[kcam]['sang']['data'], axis=axis_cam)
            ndet += dout[kcam]['ndet']['data']
    del kcam

    ndet[ndet == 0] = np.nan

    # ---------
    # etendue
    # ---------

    if vect == 'nin':
        ketend = doptics[key_cam[0]]['etendue']
        etend = coll.ddata[ketend]['data']

        etend_plane = np.nansum(
            dout[key_cam[0]]['sang']['data'],
            axis=dout[key_cam[0]]['axis_plane'],
        ) * dS
    else:
        etend = None
        etend_plane = None

    # ---------
    # extent
    # ---------

    # cam
    extent_cam = coll.get_camera_extent(key_cam[0])

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
        etend=etend,
        etend_plane=etend_plane,
        sang={'data': sang_tot},
        ndet={'data': ndet},
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
        is2d,
        sang_tot, sang, ndet,
        etend, etend_plane,
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
    # indices
    indch=None,
    indref=None,
    los_ref=None,
    pt_ref=None,
    klos=None,
    # vect
    e0=None,
    e1=None,
    vect=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # plane
    x0=None,
    x1=None,
    dS=None,
    # dout
    dout=None,
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
        is2d,
        sang_tot, sang, ndet,
        etend0, etend_plane0,
        extent_cam, extent_plane,
        dvminmax,
        los_refr, los_refz,
    ) = _check_plot_los(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        vect=vect,
        x0=x0,
        x1=x1,
        dS=dS,
        dout=dout,
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
            c=sang_tot,
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
            c=sang_tot,
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
            c=sang_tot,
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

    # --------------------
    # etendue cam vs plane

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

    # ------------
    # camera etend0

    kax = 'cam_etend0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plane-specific etendue
        imdiff = ax.imshow(
            etend0.T,
            extent=extent_cam,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('cam0', {}).get('min'),
            vmax=dvminmax.get('cam0', {}).get('max'),
            cmap=plt.cm.seismic,
        )

        # ref pixel
        _utils._add_marker(ax, indref, indplot)
        plt.colorbar(imdiff, ax=ax)

    # ------------
    # camera diff

    kax = 'cam_diff0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

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

    # -------------------------------------------
    # sang per plane point for selected pixel

    kax = 'sang'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang_tot.T,
            extent=extent_plane,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('sang', {}).get('min'),
            vmax=dvminmax.get('sang', {}).get('max'),
        )

        ax.axhline(x1[ix1], c='w', ls='--', lw=0.5)

        plt.colorbar(
            im,
            cax=dax.get('sang_cbar', {}).get('handle'),
            label='solid angle (sr)',
        )

    kax = 'ndet'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            ndet.T,
            extent=extent_plane,
            origin='lower',
            interpolation='nearest',
            vmin=dvminmax.get('ndet', {}).get('min'),
            vmax=dvminmax.get('ndet', {}).get('max'),
        )

        plt.colorbar(
            im,
            cax=dax.get('ndet_cbar', {}).get('handle'),
            label='ndet',
        )

    # -------------------------------------------
    # slice through sang 2d map

    kax = 'slice'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.plot(
            x0,
            sang[..., ix1].T.reshape((x0.size, -1)),
        )
        ax.plot(
            x0,
            sang_tot[..., ix1],
            lw=2,
            c='k',
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
    markersize=None,
    **kwdargs,
):

    # ------------
    # prepare

    (
        sang_tot, ndet_tot, dvminmax, markersize,
    ) = _check_plot_mesh(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        dout=dout,
        dvminmax=dvminmax,
        markersize=markersize,
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
            s=markersize,
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
            s=markersize,
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
            s=markersize,
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
            s=markersize,
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
            s=markersize,
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
    markersize=None,
):

    # ---------
    # prepare
    # ---------

    for ii, kcam in enumerate(key_cam):
        axis_cam = dout[kcam]['axis_cam']
        if ii == 0:
            sang_tot = np.sum(dout[kcam]['sang']['data'], axis=axis_cam)
            ndet_tot = dout[kcam]['ndet']['data'].astype(float)
        else:
            sang_tot += np.sum(dout[kcam]['sang']['data'], axis=axis_cam)
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

    # --------
    # markersize
    # --------

    markersize = float(ds._generic_check._check_var(
        markersize, 'markersize',
        types=(float, int),
        default=6,
        sign='>=0',
    ))

    return sang_tot, ndet_tot, dvminmax, markersize


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
        fs = (18, 13)

    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.94,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.4, 'hspace': 0.7,
        },
    )

    # --------
    # prepare

    fig = plt.figure(figsize=fs)

    nca_left = 4
    nca_right = 3
    ncb = 1
    nci = 1
    ncols = nca_left + nci*(2+2) + nca_right*3 + ncb

    gs = gridspec.GridSpec(ncols=ncols, nrows=12, **dmargin)

    # ------------
    # create axes
    # ------------

    # hor
    sli = (slice(8, 12), slice(0, nca_left))
    ax0 = fig.add_subplot(gs[sli], aspect='equal', adjustable='datalim')
    ax0.set_xlabel('X (m)', size=12, fontweight='bold')
    ax0.set_ylabel('Y (m)', size=12, fontweight='bold')

    # cross
    sli = (slice(4, 8), slice(0, nca_left))
    ax1 = fig.add_subplot(gs[sli], aspect='equal', adjustable='datalim')
    ax1.set_xlabel('R (m)', size=12, fontweight='bold')
    ax1.set_ylabel('Z (m)', size=12, fontweight='bold')

    # 3d
    sli = (slice(0, 4), slice(0, nca_left))
    ax2 = fig.add_subplot(gs[sli], projection='3d')
    ax2.set_xlabel('X (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Z (m)', size=12, fontweight='bold')

    # etendue
    i0 = nca_left + 2*nci
    if vect == 'nin':
        if is2d is True:

            # etend0
            sli = (slice(0, 3), slice(i0, i0 + nca_right))
            ax30 = fig.add_subplot(gs[sli], aspect='equal')
            ax30.set_ylabel('x1 (m)')
            ax30.set_xlabel('x0 (m)')
            ax30.set_title('integral', size=12, fontweight='bold')

            # etend
            i0 = nca_left + 3*nci + nca_right
            sli = (slice(0, 3), slice(i0, i0 + nca_right))
            ax31 = fig.add_subplot(gs[sli], sharex=ax30, sharey=ax30)
            ax31.set_xlabel('x0 (m)')
            ax31.set_title('etendue', size=12, fontweight='bold')

            # diff
            i0 = nca_left + 4*nci + 2*nca_right
            sli = (slice(0, 3), slice(i0, i0 + nca_right))
            ax32 = fig.add_subplot(gs[sli], sharex=ax30, sharey=ax30)
            ax32.set_xlabel('x0 (m)')
            ax32.set_title('difference', size=12, fontweight='bold')

        else:
            # etend
            sli = (slice(0, 3), slice(i0, None))
            ax30 = fig.add_subplot(gs[sli])
            ax30.set_xlabel('channel', size=12, fontweight='bold')
            ax30.set_ylabel('Etendue (m2.sr)', size=12, fontweight='bold')

    # reinitialize i0
    i0 = nca_left + 2*nci

    # sang
    sli = (slice(3, 6), slice(i0, -1))
    ax4 = fig.add_subplot(gs[sli], aspect='equal', adjustable='box')
    ax4.set_ylabel('x1 (m)', size=12, fontweight='bold')

    # sang cbar
    sli = (slice(3, 6), -1)
    ax4c = fig.add_subplot(gs[sli])

    # slice
    sli = (slice(6, 9), slice(i0, -1))
    ax5 = fig.add_subplot(gs[sli], sharex=ax4)
    ax5.set_ylabel('sang (sr)', size=12, fontweight='bold')

    # ndet
    sli = (slice(9, None), slice(i0, -1))
    ax6 = fig.add_subplot(
        gs[sli],
        sharex=ax4,
        sharey=ax4,
    )
    ax6.set_ylabel('x1 (m)', size=12, fontweight='bold')
    ax6.set_xlabel('x0 (m)', size=12, fontweight='bold')

    # ndet cbar
    sli = (slice(9, None), -1)
    ax6c = fig.add_subplot(gs[sli])

    # dict
    dax = {
        'cross': {'handle': ax0, 'type': 'cross'},
        'hor': {'handle': ax1, 'type': 'hor'},
        '3d': {'handle': ax2, 'type': '3d'},
        'sang': {'handle': ax4, 'type': 'misc'},
        'sang_cbar': {'handle': ax4c, 'type': 'cbar'},
        'slice': {'handle': ax5, 'type': 'misc'},
        'ndet': {'handle': ax6, 'type': 'misc'},
        'ndet_cbar': {'handle': ax6c, 'type': 'cbar'},
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
        adjustable='box',
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
        adjustable='box',
        sharex=ax0,
        sharey=ax0,
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
