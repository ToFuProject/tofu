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

    # ---------------------
    # prepare los and ddata

    # dcamref
    dcamref, drefx, drefy = _plot._prepare_dcamref(
        coll=coll,
        key_cam=key_cam,
        is2d=is2d,
    )

    # ---------------------
    # prepare los and vos

    (
        los_x, los_y, los_z, los_r,
        los_xi, los_yi, los_zi, los_ri,
    ) = _prepare_los(
        coll=coll,
        doptics=doptics,
        los_res=los_res,
        is2d=is2d,
        indch=indch,
    )

    (
        pc0, pc1, ph0, ph1, pc0i, pc1i, ph0i, ph1i
    ) = _prepare_vos(
        coll=coll,
        doptics=doptics,
        los_res=los_res,
        is2d=is2d,
        indch=indch,
    )

    # dsamp from mesh
    sang_tot, sang_integ, sang, extent = _prepare_sang(
        coll=coll,
        dvos=dvos[key_cam[0]],
        key_cam=key_cam,
        indch=indch,
        spectro=spectro,
        is2d=is2d,
    )

    # etendue and length
    etendue, length = _get_etendue_length(
        coll=coll,
        doptics=doptics,
    )

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
    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_cross(
            ax=ax,
            los_r=los_r,
            los_z=los_z,
            pc0=pc0,
            pc1=pc1,
            alpha=alpha,
            color_dict=color_dict,
        )

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_cross(
            ax=ax,
            los_r=los_ri,
            los_z=los_zi,
            pc0=pc0i,
            pc1=pc1i,
            alpha=alpha,
            color_dict=color_dict,
        )

    # hor
    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_hor(
            ax=ax,
            los_x=los_x,
            los_y=los_y,
            ph0=ph0,
            ph1=ph1,
            alpha=alpha,
            color_dict=color_dict,
        )

    # camera
    kax = key_cam[0]
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_data(
            coll=coll,
            ax=ax,
            sang_integ=sang_integ,
            is2d=is2d,
            etendue=etendue,
            length=length,
            indch=indch,
        )

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
#                       Prepare los and vos
# ##################################################################


def _prepare_los(
    coll=None,
    doptics=None,
    los_res=None,
    is2d=None,
    indch=None,
):

    # los
    los_x, los_y, los_z = coll.sample_rays(
        key=doptics['los'],
        res=los_res,
        mode='rel',
        concatenate=False,
    )
    los_r = np.hypot(los_x, los_y)

    # for chosen index
    if is2d:
        los_xi = los_x[:, indch[0], indch[1]]
        los_yi = los_y[:, indch[0], indch[1]]
        los_zi = los_z[:, indch[0], indch[1]]
        los_ri = los_r[:, indch[0], indch[1]]
    else:
        los_xi = los_x[:, indch]
        los_yi = los_y[:, indch]
        los_zi = los_z[:, indch]
        los_ri = los_r[:, indch]

    # concatenate
    sh = tuple(np.r_[1, los_x.shape[1:]])
    los_x = np.append(los_x, np.full(sh, np.nan), axis=0).T.ravel()
    los_y = np.append(los_y, np.full(sh, np.nan), axis=0).T.ravel()
    los_z = np.append(los_z, np.full(sh, np.nan), axis=0).T.ravel()
    los_r = np.append(los_r, np.full(sh, np.nan), axis=0).T.ravel()

    return los_x, los_y, los_z, los_r, los_xi, los_yi, los_zi, los_ri


def _prepare_vos(
    coll=None,
    doptics=None,
    los_res=None,
    is2d=None,
    indch=None,
):

    # vos
    kpc = doptics['dvos']['pcross']
    pc0 = coll.ddata[kpc[0]]['data']
    pc1 = coll.ddata[kpc[1]]['data']
    kph = doptics['dvos']['phor']
    ph0 = coll.ddata[kph[0]]['data']
    ph1 = coll.ddata[kph[1]]['data']

    if is2d:
        pc0i = pc0[:, indch[0], indch[1]]
        pc1i = pc1[:, indch[0], indch[1]]
        ph0i = ph0[:, indch[0], indch[1]]
        ph1i = ph1[:, indch[0], indch[1]]

    else:
        pc0i = pc0[:, indch]
        pc1i = pc1[:, indch]
        ph0i = ph0[:, indch]
        ph1i = ph1[:, indch]

    return pc0, pc1, ph0, ph1, pc0i, pc1i, ph0i, ph1i



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


def _get_etendue_length(
    coll=None,
    doptics=None,
):

    # ----------------------
    # get etendue and length

    ketend = doptics['etendue']
    etendue = coll.ddata[ketend]['data']

    # -------------------------
    # los through mesh envelopp

    length = np.ones(etendue.shape, dtype=float)

    return etendue, length


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
    pc0=None,
    pc1=None,
    alpha=None,
):

    # ------
    # los

    l0, = ax.plot(
        los_r,
        los_z,
        c='k',
        ls='-',
        lw=1.,
    )

    # ------
    # vos
    if pc0 is not None:
        if pc0.ndim == 2:
            for ii in range(pc0.shape[1]):
                l0, = ax.fill(
                    pc0[:, ii],
                    pc1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif pc0.ndim == 3:
            for ii in range(pc0.shape[1]):
                for jj in range(pc0.shape[2]):
                    l0, = ax.fill(
                        pc0[:, ii, jj],
                        pc1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                pc0,
                pc1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_los_hor(
    ax=None,
    los_x=None,
    los_y=None,
    ph0=None,
    ph1=None,
    alpha=None,
    color_dict=None,
):

    # ------
    # los

    l0, = ax.plot(
        los_x,
        los_y,
        c=color_dict['x'][0],
        ls='-',
        lw=1.,
    )

    # ------
    # vos

    if ph0 is not None:

        if ph0.ndim == 2:
            for ii in range(ph0.shape[1]):
                l0, = ax.fill(
                    ph0[:, ii],
                    ph1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif ph0.ndim == 3:
            for ii in range(ph0.shape[1]):
                for jj in range(ph0.shape[2]):
                    l0, = ax.fill(
                        ph0[:, ii, jj],
                        ph1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                ph0,
                ph1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_data(
    coll=None,
    ax=None,
    is2d=None,
    sang_integ=None,
    etendue=None,
    length=None,
    indch=None,
    color_dict=None,
):


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

    else:

        nlos = sang_integ.shape[0]
        ind = np.arange(0, nlos)

        # vos
        ax.plot(
            ind,
            sang_integ,
            c='b',
            marker='.',
            ls='-',
            lw=1.,
            label="vos",
        )

        # los
        ax.plot(
            ind,
            etendue * length,
            c='k',
            marker='.',
            ls='-',
            lw=1.,
            label="etendue * length",
        )

        # indch
        ax.axvline(
            indch,
            c='k',
            lw=1.,
            ls='--',
        )

        ax.set_ylim(bottom=0)

    plt.legend()


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

    ax0 = fig.add_subplot(gs[1, 1], aspect='equal')
    ax0.set_xlabel(r'X (m)', size=12)
    ax0.set_ylabel(r'Y (m)', size=12)

    # -------------
    # cross

    ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
    ax1.set_xlabel(r'R (m)', size=12)
    ax1.set_ylabel(r'Z (m)', size=12)

    # -------------
    # cross

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
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
        key_cam[0]: {'handle': ax3, 'type': 'camera'},
    }

    return dax
