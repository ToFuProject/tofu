# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot
from . import _class8_plot as _plot


# ##################################################################
# ##################################################################
#                           plot check
# ##################################################################


# ##################################################################
# ##################################################################
#                           plot main
# ##################################################################


def _plot_diagnostic_vos(
    coll=None,
    key=None,
    key_cam=None,
    optics=None,
    elements=None,
    proj=None,
    los_res=None,
    indch=None,
    # data plot
    dvos=None,
    units=None,
    cmap=None,
    vmin=None,
    vmax=None,
    alpha=None,
    # config
    plot_config=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    color_dict=None,
):

    # ------------
    # check inputs

    (
        key,
        key_cam,
        is2d,
        proj,
        ddata,
        dref,
        static,
        daxis,
        _,
        vmin,
        vmax,
        alpha,
        units,
        los_res,
        color_dict,
        _,
        ylab,
        _,
    ) = _plot._plot_diagnostic_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # parameters
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        # figure
        proj=proj,
        units=units,
        los_res=los_res,
        # interactivity
        color_dict=color_dict,
    )

    # single camera + get dvos
    key_cam = key_cam[:1]
    if dvos is None:
        dvos = coll.dobj['diagnostic'][key]['doptics'][key_cam]['dvos']

    spectro = coll.dobj['diagnostic'][key]['spectro']
    if spectro:
        msg = "plot_vos not implemented yet for spectral diagnostic"
        raise NotImplementedError(msg)

    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam[0]]

    # indch
    if indch is None:
        if is2d:
            indch = [0, 0]
        else:
            indch = 0

    # ------------
    # prepare data

    dplot = coll.get_diagnostic_dplot(
        key=key,
        key_cam=key_cam,
        optics=optics,
        elements=elements,
    )

    # -------------------------
    # prepare los interactivity

    # ---------------------
    # prepare los and ddata

    # dcamref
    dcamref, drefx, drefy = _plot._prepare_dcamref(
        coll=coll,
        key_cam=key_cam,
        is2d=is2d,
    )

    # los
    los_x, los_y, los_z = coll.sample_rays(
        key=doptics['los'],
        res=los_res,
        mode='rel',
        concatenate=False,
    )
    los_r = np.hypot(los_x, los_y)

    # vos
    kpc = doptics['dvos']['pcross']
    pc0 = coll.ddata[kpc[0]]['data']
    pc1 = coll.ddata[kpc[1]]['data']
    kph = doptics['dvos']['phor']
    ph0 = coll.ddata[kph[0]]['data']
    ph1 = coll.ddata[kph[1]]['data']

    # dsamp from mesh
    sang_tot, sang_integ, sang, extent = _prepare_sang(
        coll=coll,
        dvos=dvos[key_cam[0]],
        key_cam=key_cam,
        indch=indch,
        spectro=spectro,
        is2d=is2d,
    )

    # ddatax, ddatay
    # _, dkeyx, dkeyy, ddatax, ddatay, dextent = _prepare_datarefxy(
        # coll=coll,
        # coll2=coll2,
        # dcamref=dcamref,
        # drefx=drefx,
        # drefy=drefy,
        # ddata=ddata,
        # static=static,
        # is2d=is2d,
    # )

    # -----------------
    # prepare figure

    if dax is None:

        dax = _get_dax(
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key,
            is2d=is2d,
            key_cam=key_cam,
            indch=indch,
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot diag elements

    for k0, v0 in dplot.items():

        for k1, v1 in v0.items():

            # cross
            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['r'],
                        v1['z'],
                        v1['ur'],
                        v1['uz'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['r'],
                        v1['z'],
                        **v1.get('props', {}),
                    )

            # hor
            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['x'],
                        v1['y'],
                        v1['ux'],
                        v1['uy'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['x'],
                        v1['y'],
                        **v1.get('props', {}),
                    )

            # plotting of 2d camera contour
            kax = f"{k0}_sig"
            if is2d and k0 in key_cam and dax.get(kax) is not None:
                ax = dax[kax]['handle']
                if k1 == 'o':
                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1.get('props', {}),
                    )

    # plot data
    for k0 in key_cam:
        kax = f'{k0}_sig'
        if dax.get(kax) is not None:
            if ddata is None or ddata.get(k0) is None:
                continue

            ax = dax[kax]['handle']

            if is2d:
                im = ax.imshow(
                    ddata[k0].T,
                    extent=dextent[k0],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin='lower',
                    interpolation='nearest',
                )
                plt.colorbar(im, ax=ax)

            else:
                ax.plot(
                    ddata[k0],
                    c='k',
                    ls='-',
                    lw=1.,
                    marker='.',
                    ms=6,
                )
                ax.set_xlim(-1, ddata[k0].size)
                ax.set_ylabel(ylab)
                ax.set_title(k0, size=12, fontweight='bold')

                if vmin is not None:
                    ax.set_ylim(bottom=vmin)
                if vmax is not None:
                    ax.set_ylim(top=vmax)

    # ------------------
    # plot sang

    # crosstot
    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(
            sang_tot.T,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
        )

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(
            sang.T,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
        )

    # ------------------
    # plot los / vos

    # crosstot
    # kax = 'crosstot'
    # if dax.get(kax) is not None:
        # ax = dax[kax]['handle']

        # _add_camera_los_cross(
            # ax=ax,
            # los_r=los_r,
            # los_z=los_z,
            # vos_r=vos_r,
            # vos_z=vos_z,
            # alpha=alpha,
            # color_dict=color_dict,
        # )

    # # cross
    # kax = 'cross'
    # if dax.get(kax) is not None:
        # ax = dax[kax]['handle']

        # _add_camera_los_cross(
            # ax=ax,
            # los_r=los_r,
            # los_z=los_z,
            # vos_r=vos_r,
            # vos_z=vos_z,
            # alpha=alpha,
            # color_dict=color_dict,
        # )

    # # hor
    # kax = 'hor'
    # if dax.get(kax) is not None:
        # ax = dax[kax]['handle']

        # _add_camera_los_hor(
            # ax=ax,
            # los_x=los_x,
            # los_y=los_y,
            # vos_x=vos_x,
            # vos_y=vos_y,
            # alpha=alpha,
            # color_dict=color_dict,
        # )

    # camera
    # kax = f'{k0}_sig'
    # if dax.get(kax) is not None:
        # ax = dax[kax]['handle']

        # _add_camera_vlines_marker(
            # coll2=coll2,
            # dax=dax,
            # ax=ax,
            # kax=kax,
            # is2d=is2d,
            # k0=k0,
            # nlos=nlos,
            # ddatax=ddatax,
            # ddatay=ddatay,
            # drefx=drefx,
            # drefy=drefy,
            # dkeyx=dkeyx,
            # dkeyy=dkeyy,
            # color_dict=color_dict,
        # )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

        kax = 'crosstot'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj='cross', dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

    # -------
    # connect

    return dax


# ##################################################################
# ##################################################################
#                       Prepare sang
# ##################################################################


def _prepare_sang(
    coll=None,
    dvos=None,
    key_cam=None,
    indch=None,
    spectro=None,
    is2d=None,
):

    # -----------------
    # get mesh sampling

    dsamp = coll.get_sample_mesh(
        key=dvos['keym'],
        res=dvos['res'],
        mode='abs',
        grid=False,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=None,
        Dx1=None,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    # -----------------
    # prepare image

    n0, n1 = dsamp['x0']['data'].size, dsamp['x1']['data'].size
    shape = (n0, n1)
    sang = np.full(shape, np.nan)
    sang_tot = np.full(shape, 0.)

    if is2d:
        for ii in range(dvos['indr'].shape[1]):
            for jj in range(dvos['indr'].shape[2]):
                iok = dvos['indr'][:, ii, jj] >= 0
                indr = dvos['indr'][:, ii, jj]
                indz = dvos['indz'][:, ii, jj]
                sang_tot[indr, indz] += dvos['sang'][iok, ii, jj]

                # sang
                if ii == indch[0] and jj == indch[1]:
                    sang[indr, indz] = dvos['sang'][iok, ii, jj]

    else:
        for ii in range(dvos['indr'].shape[1]):
            iok = dvos['indr'][:, ii] >= 0
            indr = dvos['indr'][iok, ii]
            indz = dvos['indz'][iok, ii]
            sang_tot[indr, indz] += dvos['sang'][iok, ii]

            # sang
            if ii == indch:
                sang[indr, indz] = dvos['sang'][iok, ii]

    sang_tot[sang_tot == 0.] = np.nan
    sang[sang == 0.] = np.nan

    # -------------------
    # get integrated vos

    sang_integ = np.nansum(dvos['sang'], axis=0)

    # extent
    x0 = dsamp['x0']['data']
    dx0 = x0[1] - x0[0]
    x1 = dsamp['x1']['data']
    dx1 = x1[1] - x1[0]

    extent = (
        x0[0] - 0.5*dx0,
        x0[-1] + 0.5*dx0,
        x1[0] - 0.5*dx1,
        x1[-1] + 0.5*dx1,
    )

    return sang_tot, sang_integ, sang, extent


# ##################################################################
# ##################################################################
#                       Prepare
# ##################################################################


def _prepare_datarefxy(
    coll=None,
    dcamref=None,
    drefx=None,
    drefy=None,
    ddata=None,
    static=None,
    is2d=None,
):
    # prepare dict

    # loop on cams
    for k0, v0 in dcamref.items():

        # datax, datay
        if ddata is not None:
            if is2d:
                dkeyx[k0], dkeyy[k0] = coll.dobj['camera'][k0]['dgeom']['cents']

                ddatax[k0] = coll.ddata[dkeyx[k0]]['data']
                ddatay[k0] = coll.ddata[dkeyy[k0]]['data']

                coll2.add_data(key=dkeyx[k0], data=ddatax[k0], ref=drefx[k0])
                coll2.add_data(key=dkeyy[k0], data=ddatay[k0], ref=drefy[k0])
            else:
                dkeyx[k0] = f'{k0}_i0'
                ddatax[k0] = np.arange(0, coll.dref[drefx[k0]]['size'])
                coll2.add_data(key=dkeyx[k0], data=ddatax[k0], ref=drefx[k0])

            # -------------------------
            # extent

            reft = None
            if is2d:
                if ddatax[k0].size == 1:
                    ddx = coll.ddata[coll.dobj['camera'][k0]['dgeom']['outline'][0]]['data']
                    ddx = np.max(ddx) - np.min(ddx)
                else:
                    ddx = ddatax[k0][1] - ddatax[k0][0]
                if ddatay[k0].size == 1:
                    ddy = coll.ddata[coll.dobj['camera'][k0]['dgeom']['outline'][1]]['data']
                    ddy = np.max(ddy) - np.min(ddy)
                else:
                    ddy = ddatay[k0][1] - ddatay[k0][0]

                dextent[k0] = (
                    ddatax[k0][0] - 0.5*ddx,
                    ddatax[k0][-1] + 0.5*ddx,
                    ddatay[k0][0] - 0.5*ddy,
                    ddatay[k0][-1] + 0.5*ddy,
                )

    return reft, dkeyx, dkeyy, ddatax, ddatay, dextent


# ##################################################################
# ##################################################################
#                       add mobile
# ##################################################################


def _add_camera_los_cross(
    ax=None,
    color_dict=None,
    los_r=None,
    los_z=None,
    vos_r=None,
    vos_z=None,
    alpha=None,
):

    # ------
    # los

    l0, = ax.plot(
        los_r,
        los_z,
        c=color_dict['x'][ii],
        ls='-',
        lw=1.,
    )

    # ------
    # vos

    l0, = ax.fill(
        vos_r,
        vos_z,
        fc=color_dict['x'][ii],
        alpha=alpha,
        ls='None',
        lw=0.,
    )


def _add_camera_los_hor(
    coll2=None,
    k0=None,
    ax=None,
    kax=None,
    nlos=None,
    dref_los=None,
    dref_vos=None,
    color_dict=None,
    nan_los=None,
    nan_vos=None,
    alpha=None,
):

    for ii in range(nlos):

        # ------
        # los

        l0, = ax.plot(
            nan_los,
            nan_los,
            c=color_dict['x'][ii],
            ls='-',
            lw=1.,
        )

        # add mobile
        kl0 = f'{k0}_los_hor{ii}'
        coll2.add_mobile(
            key=kl0,
            handle=l0,
            refs=dref_los[k0],
            data=[f'{k0}_los_x', f'{k0}_los_y'],
            dtype=['xdata', 'ydata'],
            axes=kax,
            ind=ii,
        )

        # ------
        # vos

        if f'{k0}_vos_hor' in coll2.ddata.keys():

            l0, = ax.fill(
                nan_vos,
                nan_vos,
                fc=color_dict['x'][ii],
                alpha=alpha,
                ls='None',
                lw=0.,
            )

            # add mobile
            kl0 = f'{k0}_vos_hor{ii}'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=dref_vos[k0],
                data=[f'{k0}_vos_hor'],
                dtype=['xy'],
                axes=kax,
                ind=ii,
            )


def _add_camera_vlines_marker(
    coll2=None,
    dax=None,
    ax=None,
    kax=None,
    is2d=None,
    k0=None,
    nlos=None,
    ddatax=None,
    ddatay=None,
    drefx=None,
    drefy=None,
    dkeyx=None,
    dkeyy=None,
    color_dict=None,
    suffix=None,
):

    if suffix is None:
        suffix = ''

    if is2d:
        for ii in range(nlos):
            mi, = ax.plot(
                ddatax[k0][0:1],
                ddatay[k0][0:1],
                marker='s',
                ms=6,
                markeredgecolor=color_dict['x'][ii],
                markerfacecolor='None',
            )

            km = f'{k0}_m{ii:02.0f}{suffix}'
            coll2.add_mobile(
                key=km,
                handle=mi,
                refs=[drefx[k0], drefy[k0]],
                data=[dkeyx[k0], dkeyy[k0]],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[drefx[k0]],
            refy=[drefy[k0]],
            datax=[dkeyx[k0]],
            datay=[dkeyy[k0]],
        )

    else:

        for ii in range(nlos):
            lv = ax.axvline(
                ddatax[k0][0], c=color_dict['y'][ii], lw=1., ls='-',
            )
            kv = f'{k0}_v{ii:02.0f}{suffix}'
            coll2.add_mobile(
                key=kv,
                handle=lv,
                refs=drefx[k0],
                data=dkeyx[k0],
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[drefx[k0]], datax=[dkeyx[k0]])


# ################################################################
# ################################################################
#                   figure
# ################################################################


def _get_dax(
    proj=None,
    dmargin=None,
    fs=None,
    tit=None,
    wintit=None,
    is2d=None,
    key_cam=None,
    indch=None,
):

    # ------------
    # check inputs

    # fs
    if fs is None:
        fs = (14, 10)

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.30, 'hspace': 0.40,
            # 'width_ratios': [0.6, 0.4],
            # 'height_ratios': [0.4, 0.6],
        },
    )

    # wintit
    wintit = ds._generic_check._check_var(
        wintit, 'wintit',
        types=str,
        default=_generic_plot._WINDEF,
    )

    # tit
    if tit is None:
        tit = f"{key_cam} - {indch}\nVOS"

    # -------------
    # Create figure

    fig = plt.figure(figsize=fs)
    fig.canvas.set_window_title(wintit)
    fig.suptitle(tit, size=12, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)

    # -------------
    # hor

    ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
    ax0.set_xlabel(r'X (m)', size=12)
    ax0.set_ylabel(r'Y (m)', size=12)

    # -------------
    # cross

    ax1 = fig.add_subplot(gs[1, 0], aspect='equal')
    ax1.set_xlabel(r'R (m)', size=12)
    ax1.set_ylabel(r'Z (m)', size=12)

    # -------------
    # cross

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    ax2.set_xlabel(r'R (m)', size=12)
    ax2.set_ylabel(r'Z (m)', size=12)

    # -------------
    # camera

    if is2d:
        ax3 = fig.add_subplot(gs[0, 1], aspect='equal')
        ax3.set_xlabel(r'index', size=12)
        ax3.set_ylabel(r'index', size=12)
    else:
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set_xlabel(r'index', size=12)
        ax3.set_ylabel(r'data', size=12)

    # ---------
    # dict

    dax = {
        'hor': {'handle': ax0, 'type': 'hor'},
        'cross': {'handle': ax1, 'type': 'cross'},
        'crosstot': {'handle': ax2, 'type': 'cross'},
        'camera': {'handle': ax3, 'type': 'camera'},
    }

    return dax
