# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# tofu
# from tofu import __version__ as __version__
from . import _generic_check


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
    dax=None,
    connect=None,
):

    # key
    if 'inversions' not in coll.dobj.keys():
        msg = 'No inversions available!'
        raise Exception(msg)

    lk = list(coll.dobj['inversions'].keys())
    keyinv = _generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lk,
    )
    keymat = coll.dobj['inversions'][keyinv]['matrix']
    keydata = coll.dobj['inversions'][keyinv]['data_in']
    keyretro = coll.dobj['inversions'][keyinv]['retrofit']
    keybs = coll.dobj['matrix'][keymat]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    # refbs = coll.dobj['bsplines'][keybs]['ref']

    crop = coll.dobj['matrix'][keymat]['crop']
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
    dcolorbar = _generic_check._check_var(
        dcolorbar, 'dcolorbar',
        default=defdcolorbar,
        types=dict,
    )

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=True,
        types=bool,
    )

    return (
        keyinv, keymat, keybs, keydata, keyretro, mtype,
        cropbs, cmap, dcolorbar, dleg, connect,
    )


def _plot_inversion_prepare(
    coll=None,
    coll2=None,
    mtype=None,
    keyinv=None,
    key_matrix=None,
    key_data=None,
    key_retro=None,
):



    # -----------------
    # add nearest-neighbourg interpolated data

    reft, keyt, time = coll.get_time(key=keyinv)[2:5]
    if coll.get_time(key=key_matrix)[0]:
        keyt_data = coll.get_time(key=key_data)[3]
        if keyt_data != keyt:
            dind = coll.get_time(
                key=key_data,
                t=keyt,
            )[-1]
            data = coll.ddata[key_data]['data'][dind['ind'], :]
        else:
            data = coll.ddata[key_data]['data']
    else:
        data = coll.ddata[key_data]['data']

    # ----------------
    # add data + retro

    # ref
    ref_retro = coll.ddata[key_retro]['ref']

    # chan vector
    chan = coll.get_ref_vector(key=key_data, ref=ref_retro[-1])[4]
    if chan is None:
        chan = np.arange(0, coll.dref[ref_retro[-1]]['size'])

    # add ref
    for rr in ref_retro:
        if rr not in coll2.dref.keys():
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

    # data
    coll2.add_data(
        key=key_data,
        data=data,
        ref=ref_retro,
    )

    # retro
    if key_retro not in coll2.ddata.keys():
        coll2.add_data(
            key=key_retro,
            data=coll.ddata[key_retro]['data'],
            ref=ref_retro,
        )

    # err
    key_err = f'{key_data}-err'
    err = coll.ddata[key_retro]['data'] - data
    coll2.add_data(
        key=key_err,
        data=err,
        ref=ref_retro,
    )

    # chi2n, nu, reg, niter


    # # data
    # refchan = coll.ddata[keydata]['ref'][-1]
    # chan = coll.ddata[keychan]['data']
    # hastime = coll.ddata[keydata]['data'].ndim == 2

    # if hastime:
        # data = coll.ddata[keydata]['data']
        # time = np.arange(coll.ddata[keydata]['data'].shape[0])
    # else:
        # data = coll.ddata[keydata]['data'][None, :]
        # time = [0]
        # indt = 0

    # reconstructed data
    # matrix = coll.ddata[keymat]['data']
    # sol = coll.ddata[keyinv]['data']
    # shapebs = coll.dobj['bsplines'][keybs]['shape']
    # nbs = int(np.prod(shapebs))

    # if cropbs is not None:
        # cropbs_flat = cropbs.ravel(order='F')

    # if hastime:
        # nt = sol.shape[0]
        # if sol.ndim == 3:
            # sol_flat = sol.reshape((nt, nbs), order='F')
        # else:
            # sol_flat = sol
        # if cropbs is not None:
            # data_re = matrix.dot(sol_flat[:, cropbs_flat].T)
        # else:
            # data_re = matrix.dot(sol_flat[:, ...].T)
    # else:
        # if sol.ndim == 2:
            # sol_flat = sol.ravel(order='F')
        # else:
            # sol_flat = sol
        # if cropbs is not None:
            # data_re = matrix.dot(sol_flat[cropbs_flat].T)[:, None]
        # else:
            # data_re = matrix.dot(sol_flat.T)[:, None]

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

    datamin = min([np.nanmin(data), np.nanmin(coll2.ddata[key_retro]['data'])])
    datamax = max([np.nanmax(data), np.nanmax(coll2.ddata[key_retro]['data'])])
    errmin = np.nanmin(coll2.ddata[key_err]['data'])
    errmax = np.nanmax(coll2.ddata[key_err]['data'])
    errmax = max(np.abs(errmin), np.abs(errmax))

    return (
        chan, time, reft,
        key_err, chi2n, mu, reg, niter,
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
    # interactivity
    dinc=None,
    connect=None,
):

    # ------------
    # check inputs

    (
        keyinv, keymat, keybs, key_data, key_retro, mtype,
        cropbs, cmap, dcolorbar, dleg, connect,
    ) = _plot_inversion_check(
        coll=coll,
        key=key,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
        connect=connect,
    )

    # --------------
    # plot - prepare

    if dax is None:

        dax = _plot_inversion_create_axes(
            fs=fs,
            dmargin=dmargin,
            mtype=mtype,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # --------------
    # plot profile2d

    coll2, dgroup = coll.plot_profile2d(
        key=key,
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
        chan, time, reft,
        key_err, chi2n, mu, reg, niter,
        datamin, datamax, errmax,
    ) = _plot_inversion_prepare(
        coll=coll,
        coll2=coll2,
        keyinv=keyinv,
        key_matrix=keymat,
        key_data=key_data,
        key_retro=key_retro,
    )

    # ---------
    # plot data

    kax = 'retrofit'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if reft is None:
            retro = coll2.ddata[key_retro]['data']
            data = coll2.ddata[key_data]['data']
        else:
            retro = coll2.ddata[key_retro]['data'][0, :]
            data = coll2.ddata[key_data]['data'][0, :]

        # plot
        l1, = ax.plot(
            chan,
            retro,
            c=(0.8, 0.8, 0.8),
            ls='-',
            lw=1.,
        )

        l0, = ax.plot(
            chan,
            data,
            c='k',
            ls='None',
            lw=2.,
            marker='.',
        )

        # add mobiles
        if reft is not None:
            kl0 = 'data'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=(reft,),
                data=[key_data],
                dtype=['ydata'],
                axes=kax,
                ind=0,
            )
            kl1 = 'retrofit'
            coll2.add_mobile(
                key=kl1,
                handle=l1,
                refs=(reft,),
                data=[key_retro],
                dtype=['ydata'],
                axes=kax,
                ind=0,
            )

        ax.set_ylim(min(0, datamin), datamax)
        ax.set_xlim(0, chan.size + 1)

    kax = 'err'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if reft is None:
            err = coll2.ddata[key_err]['data']
        else:
            err = coll2.ddata[key_err]['data'][0, :]

        l0, = ax.plot(
            chan,
            err,
            c='k',
            ls='-',
            lw=1.,
            marker='.',
        )
        ax.axhline(0, color='k', ls='--')

        # add mobile
        if reft is not None:
            kl0 = 'err'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=(reft,),
                data=[key_err],
                dtype=['ydata'],
                axes=kax,
                ind=0,
            )

        if np.isfinite(errmax):
            ax.set_ylim(-errmax, errmax)

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
):

    if fs is None:
        fs = (16, 10)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.05, 'top': 0.95,
            'hspace': 0.4, 'wspace': 0.3,
        }

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=8, nrows=6, **dmargin)

    # ------------------
    # axes for profile2d

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    if mtype == 'polar':
        # axes for traces
        ax7 = fig.add_subplot(gs[:2, :2], sharey=ax2)
        # axes for traces
        ax3 = fig.add_subplot(gs[2:4, :2])
    else:
        ax7 = None
        # axes for traces
        ax3 = fig.add_subplot(gs[:3, :2])


    # axes for text
    ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax6 = fig.add_subplot(gs[4:, :2], frameon=False)

    # ------------------
    # axes for inversion

    # retrofit
    ax8 = fig.add_subplot(gs[:2, 6:])

    # error
    ax9 = fig.add_subplot(gs[2:4, 6:], sharex=ax8)

    # parameters (chi2, ...)
    ax10 = fig.add_subplot(gs[4, 6:], sharex=ax3)

    # nb of iterations
    ax11 = fig.add_subplot(gs[5, 6:], sharex=ax3)

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0, 'type': 'matrix'},
        'vertical': {'handle': ax1, 'type': 'misc'},
        'horizontal': {'handle': ax2, 'type': 'misc'},
        'traces': {'handle': ax3, 'type': 'misc'},
        # inversion
        'retrofit': {'handle': ax8, 'type': 'misc'},
        'err': {'handle': ax9, 'type': 'misc'},
        'inv-param': {'handle': ax10, 'type': 'misc'},
        'niter': {'handle': ax11, 'type': 'misc'},
        # text
        'textX': {'handle': ax4, 'type': 'text'},
        'textY': {'handle': ax5, 'type': 'text'},
        'textZ': {'handle': ax6, 'type': 'text'},
    }

    if ax7 is not None:
        dax['radial'] = {'handle': ax7, 'type': 'misc'}
    return dax

