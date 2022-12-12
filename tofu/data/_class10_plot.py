# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# tofu
# from tofu import __version__ as __version__
from . import _generic_check
from . import _class8_plot


# #############################################################################
# #############################################################################
#                           inversions
# #############################################################################


def _plot_inversion_check(
    coll=None,
    key=None,
    indbf=None,
    indchan=None,
    cmap=None,
    dcolorbar=None,
    dleg=None,
    # los sampling
    los_res=None,
    # interactivity
    color_dict=None,
    nlos=None,
    dax=None,
    connect=None,
):

    # key
    if 'inversions' not in coll.dobj.keys():
        msg = 'No inversions available!'
        raise Exception(msg)

    lk = list(coll.dobj['inversions'].keys())
    keyinv = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lk,
    )
    keymat = coll.dobj['inversions'][keyinv]['matrix']
    key_data = coll.dobj['inversions'][keyinv]['data_in']
    key_retro = coll.dobj['inversions'][keyinv]['retrofit']
    keybs = coll.dobj['geom matrix'][keymat]['bsplines']
    key_diag = coll.dobj['geom matrix'][keymat]['diagnostic']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']
    key_cam = coll.dobj['geom matrix'][keymat]['camera']
    key_retro = coll.dobj['diagnostic'][key_diag]['dsignal'][key_retro]['data']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    # refbs = coll.dobj['bsplines'][keybs]['ref']

    crop = coll.dobj['geom matrix'][keymat]['crop']
    if crop is True:
        cropbs = coll.dobj['bsplines'][keybs]['crop']
        cropbs = coll.ddata[cropbs]['data']
    else:
        cropbs = None

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = ds._generic_check._check_var(
        dcolorbar, 'dcolorbar',
        default=defdcolorbar,
        types=dict,
    )

    # los_res
    los_res = ds._generic_check._check_var(
        los_res, 'los_res',
        types=float,
        default=0.05,
        sign='> 0.',
    )

    # color_dict
    color_dict = _class8_plot._check_color_dict(color_dict)

    # nlos
    nlos = ds._generic_check._check_var(
        nlos, 'nlos',
        types=int,
        default=5,
    )

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # connect
    connect = ds._generic_check._check_var(
        connect, 'connect',
        default=True,
        types=bool,
    )

    return (
        keyinv, keymat,
        key_diag, key_cam, keybs, key_data, key_retro,
        is2d, mtype,
        cropbs, cmap, dcolorbar,
        nlos, los_res, color_dict,
        dleg, connect,
    )


def _plot_inversion_prepare(
    coll=None,
    coll2=None,
    is2d=None,
    mtype=None,
    keyinv=None,
    key_matrix=None,
    key_diag=None,
    key_cam=None,
    key_data=None,
    key_retro=None,
    los_res=None,
):

    # -----------------
    # add nearest-neighbourg interpolated data

    # just for preparation
    ddata = {
        k0: {'data': coll.ddata[key_data[ii]]} for ii, k0 in enumerate(key_cam)
    }

    # dcamref
    dcamref, drefx, drefy = _class8_plot._prepare_dcamref(
        coll=coll,
        key_cam=key_cam,
        is2d=is2d,
    )

    # los
    dlos, dref_los = _class8_plot._prepare_los(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        key_diag=key_diag,
        key_cam=key_cam,
        los_res=los_res,
    )

    # ddatax, ddatay
    (
        reft, dkeyx, dkeyy, ddatax, ddatay, dextent,
    ) = _class8_plot._prepare_datarefxy(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        drefx=drefx,
        drefy=drefy,
        ddata=ddata,
        is2d=is2d,
    )

    # -----------------
    # add nearest-neighbourg interpolated data

    reft, keyt, time = coll.get_time(key=keyinv)[2:5]
    lkmat = coll.dobj['geom matrix'][key_matrix]['data']

    dind = None
    if coll.get_time(key=lkmat[0])[0]:
        keyt_data = coll.get_time(key=key_data[0])[3]
        if keyt_data != keyt:
            dind = coll.get_time(
                key=key_data[0],
                t=keyt,
            )[-1]

    if reft not in coll2.dref.keys():
        coll2.add_ref(key=reft, size=coll.dref[reft]['size'])

    # --------------------
    # add data, retro, err

    datamin, datamax = np.inf, -np.inf
    errmin, errmax = np.inf, -np.inf
    for ii, k0 in enumerate(key_cam):

        # ref
        for rr in coll.dobj['camera'][k0]['dgeom']['ref']:
            if rr not in coll2.dref.keys():
                coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

        refi = coll.ddata[key_data[ii]]['ref']
        datai = coll.ddata[key_data[ii]]['data']
        if dind is not None:
            nd = len(refi)
            refti = coll.get_time(key=key_data[ii])[2]
            axis = refi.index(refti)
            refi = tuple([reft if jj == axis else refi[jj] for jj in range(nd)])
            sli = tuple([
                dind['ind'] if jj == axis else slice(None) for jj in range(nd)
            ])
            datai = datai[sli]

        # transpose for 2d
        retroi = coll.ddata[key_retro[ii]]['data']
        if is2d:
            ndim = datai.ndim
            datai = np.swapaxes(datai, ndim-1, ndim-2)
            retroi = np.swapaxes(retroi, ndim-1, ndim-2)
            refi = list(refi)
            refi[-2], refi[-1] = refi[-1], refi[-2]
            refi = tuple(refi)

        # data min max
        datamin = min(datamin, np.nanmin(datai))
        datamax = max(datamax, np.nanmax(datai))

        # data
        coll2.add_data(
            key=key_data[ii],
            data=datai,
            ref=refi,
            units=coll.ddata[key_data[ii]]['units'],
        )

        # retro
        coll2.add_data(
            key=key_retro[ii],
            data=retroi,
            ref=refi,
        )

        # err min max
        erri = retroi - datai
        errmin = min(errmin, np.nanmin(erri))
        errmax = max(errmax, np.nanmax(erri))

        # data
        coll2.add_data(
            key=f"{key_data[ii]}_err",
            data=erri,
            ref=refi,
            units=coll.ddata[key_data[ii]]['units'],
        )

        # cross-section los

    # errmax
    errmax = max(np.abs(errmin), np.abs(errmax))

    # ----------------
    # inversion parameters

    if reft is not None:
        chi2n = coll.ddata[f'{keyinv}-chi2n']['data']
        mu = coll.ddata[f'{keyinv}-mu']['data']
        reg = coll.ddata[f'{keyinv}-reg']['data']
        niter = coll.ddata[f'{keyinv}-niter']['data']
    else:
        chi2n = None    # coll.dobj['inversions'][keyinv]['chi2n']
        mu = None       # coll.dobj['inversions'][keyinv]['mu']
        reg = None      # coll.dobj['inversions'][keyinv]['reg']
        niter = None    # coll.dobj['inversions'][keyinv]['niter']

    return (
        dlos, dref_los,
        drefx, drefy, dkeyx, dkeyy, ddatax, ddatay, dextent,
        time, reft,
        chi2n, mu, reg, niter,
        datamin, datamax, errmax,
    )


def plot_inversion(
    coll=None,
    key=None,
    indt=None,
    res=None,
    vmin=None,
    vmax=None,
    cmap=None,
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    # los sampling
    los_res=None,
    # interactivity
    color_dict=None,
    nlos=None,
    dinc=None,
    connect=None,
):

    # ------------
    # check inputs

    (
        keyinv, keymat,
        key_diag, key_cam, keybs, key_data, key_retro,
        is2d, mtype,
        cropbs, cmap, dcolorbar,
        nlos, los_res, color_dict,
        dleg, connect,
    ) = _plot_inversion_check(
        coll=coll,
        key=key,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
        # los sampling
        los_res=los_res,
        # interactivity
        color_dict=color_dict,
        nlos=nlos,
        connect=connect,
    )

    # --------------
    # plot - prepare

    if dax is None:

        dax = _plot_inversion_create_axes(
            fs=fs,
            dmargin=dmargin,
            mtype=mtype,
            key_cam=key_cam,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # --------------
    # plot profile2d

    coll2, dgroup = coll.plot_profile2d(
        key=keyinv,
        res=res,
        # figure
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        dax=dax,
        dmargin=dmargin,
        fs=fs,
        dcolorbar=dcolorbar,
        dleg=dleg,
        # interactivity
        dinc=dinc,
        connect=False,
    )
    dax = coll2.dax

    # ------------
    # prepare data

    (
        dlos, dref_los,
        drefx, drefy, dkeyx, dkeyy, ddatax, ddatay, dextent,
        time, reft,
        chi2n, mu, reg, niter,
        datamin, datamax, errmax,
    ) = _plot_inversion_prepare(
        coll=coll,
        coll2=coll2,
        is2d=is2d,
        keyinv=keyinv,
        key_matrix=keymat,
        key_cam=key_cam,
        key_diag=key_diag,
        key_data=key_data,
        key_retro=key_retro,
        los_res=los_res,
    )

    # ----------------
    # define and set dgroup

    dgroup.update({
        f'{k0}_i0': {
            'ref': [drefx[k0]],
            'data': ['index'],
            'nmax': nlos,
        }
        for k0 in key_cam
    })

    if is2d:
        dgroup.update({
            f'{k0}_i1': {
                'ref': [drefy[k0]],
                'data': ['index'],
                'nmax': nlos,
            }
            for k0 in key_cam
        })

    if reft is not None and reft not in dgroup['Z']['ref']:
        dgroup['Z']['ref'].append(reft)

    # ---------
    # plot data

    for ii, k0 in enumerate(key_cam):


        # data vs retro
        kax = k0
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            if is2d:

                if reft is None:
                    ax.imshow(
                        coll2.ddata[key_data[ii]]['data'],
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                        aspect='equal',
                    )

                else:

                    im = ax.imshow(
                        coll2.ddata[key_data[ii]]['data'][0, ...],
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                        aspect='equal',
                    )

                    ki = key_data[ii]
                    coll2.add_mobile(
                        key=ki,
                        handle=im,
                        refs=(reft,),
                        data=[key_data[ii]],
                        dtype=['data'],
                        axes=kax,
                        ind=0,
                    )

            else:

                nch = coll.dobj['camera'][k0]['dgeom']['pix_nb']
                chani = np.arange(0, nch)
                if reft is None:
                    ax.plot(
                        chani,
                        coll2.ddata[key_data[ii]]['data'].ravel(),
                        c=(0.8, 0.8, 0.8),
                        marker='o',
                        ls='-',
                    )

                    ax.plot(
                        chani,
                        coll2.ddata[key_retro[ii]]['data'].ravel(),
                        c='k',
                        marker='None',
                        ls='-',
                        lw=2,
                    )

                else:
                    nani = np.nan * chani
                    l0, = ax.plot(
                        chani,
                        nani,
                        c=(0.8, 0.8, 0.8),
                        marker='o',
                        ls='-',
                    )

                    l1, = ax.plot(
                        chani,
                        nani,
                        c='k',
                        marker='None',
                        ls='-',
                        lw=2,
                    )

                    kl0 = key_data[ii]
                    coll2.add_mobile(
                        key=kl0,
                        handle=l0,
                        refs=(reft,),
                        data=[key_data[ii]],
                        dtype=['ydata'],
                        axes=kax,
                        ind=0,
                    )

                    kl1 = key_retro[ii]
                    coll2.add_mobile(
                        key=kl1,
                        handle=l1,
                        refs=(reft,),
                        data=[key_retro[ii]],
                        dtype=['ydata'],
                        axes=kax,
                        ind=0,
                    )
                    ax.set_ylim(min(0, datamin), datamax)
                    ax.set_xlim(0, nch + 1)

            # add vlines / markers
            _class8_plot._add_camera_vlines_marker(
                coll2=coll2,
                dax=dax,
                ax=ax,
                kax=kax,
                is2d=is2d,
                k0=k0,
                nlos=nlos,
                ddatax=ddatax,
                ddatay=ddatay,
                drefx=drefx,
                drefy=drefy,
                dkeyx=dkeyx,
                dkeyy=dkeyy,
                color_dict=color_dict,
            )

        # add los
        kax = 'matrix'
        if dlos[k0]['rays'] is not None:
            ax = dax[kax]['handle']

            nan = np.full((dlos[k0]['x'].shape[0],), np.nan)

            _class8_plot._add_camera_los_cross(
                coll2=coll2,
                k0=k0,
                ax=ax,
                kax=kax,
                nlos=nlos,
                dref_los=dref_los,
                color_dict=color_dict,
                nan=nan,
            )

        # err
        kax = f"{k0}_err"
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            kerr = f"{key_data[ii]}_err"
            if is2d:

                if reft is None:
                    ax.imshow(
                        coll2.ddata[kerr]['data'],
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                        aspect='equal',
                    )

                else:

                    im = ax.imshow(
                        coll2.ddata[kerr]['data'][0, ...],
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                        aspect='equal',
                    )

                    ki = kerr
                    coll2.add_mobile(
                        key=ki,
                        handle=im,
                        refs=(reft,),
                        data=[kerr],
                        dtype=['data'],
                        axes=kax,
                        ind=0,
                    )

            else:

                nch = coll.dobj['camera'][k0]['dgeom']['pix_nb']
                chani = np.arange(0, nch)
                if reft is None:
                    ax.plot(
                        chani,
                        coll2.ddata[kerr]['data'].ravel(),
                        c=(0.8, 0.8, 0.8),
                        marker='o',
                        ls='-',
                    )

                else:
                    nani = np.nan * chani
                    l0, = ax.plot(
                        chani,
                        nani,
                        c=(0.8, 0.8, 0.8),
                        marker='o',
                        ls='-',
                    )

                    kl0 = kerr
                    coll2.add_mobile(
                        key=kl0,
                        handle=l0,
                        refs=(reft,),
                        data=[kerr],
                        dtype=['ydata'],
                        axes=kax,
                        ind=0,
                    )

                    ax.set_ylim(-errmax, errmax)
                    ax.set_xlim(0, nch + 1)

            # add vlines / markers
            _class8_plot._add_camera_vlines_marker(
                coll2=coll2,
                dax=dax,
                ax=ax,
                kax=kax,
                is2d=is2d,
                k0=k0,
                nlos=nlos,
                ddatax=ddatax,
                ddatay=ddatay,
                drefx=drefx,
                drefy=drefy,
                dkeyx=dkeyx,
                dkeyy=dkeyy,
                color_dict=color_dict,
                suffix='_err'
            )

    # # -------------------------
    # # plot inversion parameters

    if reft is not None:
        kax = 'inv-param'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            ax.plot(
                # time,
                chi2n / np.nanmax(chi2n),
                c='k',
                ls='-',
                lw=1.,
                marker='.',
                label='nchi2n',
            )

            ax.plot(
                # time,
                reg / np.nanmax(reg),
                c='b',
                ls='-',
                lw=1.,
                marker='.',
                label='mu*reg',
            )

            # add mobile
            l0 = ax.axvline(time[0], c='k', ls='-', lw=1.)

            # add mobile
            kl0 = 't-par'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=(reft,),
                data=['index'],
                dtype=['xdata'],
                axes=kax,
                ind=0,
            )
            ax.set_ylim(bottom=0)

        kax = 'niter'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            ax.plot(
                # time,
                niter,
                c='k',
                ls='-',
                lw=1.,
                marker='.',
            )

            ax.set_ylim(bottom=0)

    # connect
    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

        coll2.show_commands()
        return coll2
    else:
        return coll2, dgroup


def _plot_inversion_create_axes(
    fs=None,
    dmargin=None,
    mtype=None,
    key_cam=None,
):

    if fs is None:
        fs = (16, 10)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.98,
            'bottom': 0.05, 'top': 0.95,
            'hspace': 1.5, 'wspace': 0.5,
        }

    ncam = len(key_cam)
    nblock = 3*ncam
    nrows = 3*nblock

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=8, nrows=nrows, **dmargin)

    # ------------------
    # axes for profile2d

    # axes for image
    ax0 = fig.add_subplot(gs[:2*nblock, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:2*nblock, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[2*nblock:, 2:4], sharex=ax0)

    # axes for radius
    if mtype == 'polar':
        ax7 = fig.add_subplot(gs[:nblock, :2], sharey=ax2)
    else:
        ax7 = None

    # axes for traces
    ax3 = fig.add_subplot(gs[nblock:2*nblock, :2])

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0, 'type': 'matrix'},
        'vertical': {'handle': ax1, 'type': 'misc'},
        'horizontal': {'handle': ax2, 'type': 'misc'},
        'traces': {'handle': ax3, 'type': 'misc'},
    }
    # axes for text
    # ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
    # ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
    # ax6 = fig.add_subplot(gs[4:, :2], frameon=False)

    # ------------------
    # axes for inversion

    npc = nrows / ncam
    for ii, k0 in enumerate(key_cam):
        # retrofit
        dax[k0] = fig.add_subplot(gs[9*ii:9*ii+6, 5:])

        # error
        dax[f'{k0}_err'] = fig.add_subplot(gs[9*ii+6:9*(ii+1), 5:], sharex=dax[k0])

    # parameters (chi2, ...)
    dax['inv-param'] = fig.add_subplot(gs[2*nblock:2*nblock+ncam, :2], sharex=ax3)

    # nb of iterations
    dax['niter'] = fig.add_subplot(gs[2*nblock+ncam:, :2], sharex=ax3)

    if ax7 is not None:
        dax['radial'] = {'handle': ax7, 'type': 'misc'}

    return dax

