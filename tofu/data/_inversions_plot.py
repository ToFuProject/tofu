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
    keybs = coll.dobj['matrix'][keymat]['bsplines']
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

    return keyinv, keymat, keybs, keydata, cropbs, cmap, dcolorbar, dleg


def _plot_inversion_prepare():

    # data
    keychan = coll.ddata[keydata]['ref'][-1]
    chan = coll.ddata[keychan]['data']
    hastime = coll.ddata[keydata]['data'].ndim == 2

    if hastime:
        data = coll.ddata[keydata]['data']
        time = np.arange(coll.ddata[keydata]['data'].shape[0])
    else:
        data = coll.ddata[keydata]['data'][None, :]
        time = [0]
        indt = 0

    # reconstructed data
    matrix = coll.ddata[keymat]['data']
    sol = coll.ddata[keyinv]['data']
    shapebs = coll.dobj['bsplines'][keybs]['shape']
    nbs = int(np.prod(shapebs))

    if cropbs is not None:
        cropbs_flat = cropbs.ravel(order='F')
    if hastime:
        nt = sol.shape[0]
        if sol.ndim == 3:
            sol_flat = sol.reshape((nt, nbs), order='F')
        else:
            sol_flat = sol
        if cropbs is not None:
            data_re = matrix.dot(sol_flat[:, cropbs_flat].T)
        else:
            data_re = matrix.dot(sol_flat[:, ...].T)
    else:
        if sol.ndim == 2:
            sol_flat = sol.ravel(order='F')
        else:
            sol_flat = sol
        if cropbs is not None:
            data_re = matrix.dot(sol_flat[cropbs_flat].T)[:, None]
        else:
            data_re = matrix.dot(sol_flat.T)[:, None]

    # inversion parameters
    if hastime:
        nchi2n = chan.size * coll.ddata[f'{keyinv}-niter']['data']
        mu = coll.ddata[f'{keyinv}-mu']['data']
        reg = coll.ddata[f'{keyinv}-reg']['data']
        niter = coll.ddata[f'{keyinv}-niter']['data']
    else:
        nchi2n = chan.size * coll.dobj['inversions'][keyinv]['chi2n']
        mu = coll.dobj['inversions'][keyinv]['mu']
        reg = coll.dobj['inversions'][keyinv]['reg']
        niter = coll.dobj['inversions'][keyinv]['niter']


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
):

    # ------------
    # check inputs

    (
        keyinv, keymat, keybs, keydata, cropbs, cmap, dcolorbar, dleg,
    ) = _plot_inversion_check(
        coll=coll,
        key=key,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
    )

    # ------------
    # prepare data

    # _plot_inversion_prepare()

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

    coll2 = coll.plot_profile2d(
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

    # ---------
    # plot data

    kax = 'retrofit'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.plot(
            chan,
            data[indt, :],
            c='k',
            ls='-',
            lw=1.,
            marker='.',
        )
        ax.plot(
            chan,
            data_re[:, indt],
            c='k',
            ls='--',
            lw=1.,
            marker='.',
        )

    kax = 'data-err'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.plot(
            chan,
            data_re[:, indt] - data[indt, :],
            c='k',
            ls='-',
            lw=1.,
            marker='.',
        )
        ax.axhline(
            0.,
            c='k',
            ls='-',
            lw=1.,
        )

    # -------------------------
    # plot inversion parameters

    kax = 'inv-param'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.plot(
            time,
            nchi2n + mu*reg,
            c='k',
            ls='-',
            lw=1.,
            marker='.',
            label='n*chi2n + mu*reg',
        )
        ax.plot(
            time,
            nchi2n,
            c='r',
            ls='-',
            lw=1.,
            marker='.',
            label='nchi2n',
        )
        ax.plot(
            time,
            mu*reg,
            c='b',
            ls='-',
            lw=1.,
            marker='.',
            label='mu*reg',
        )
        ax.axvline(time[indt], c='k', ls='-', lw=1.)

    kax = 'niter'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.plot(
            time,
            niter,
            c='k',
            ls='-',
            lw=1.,
            marker='.',
        )
        ax.axvline(time[indt], c='k', ls='-', lw=1.)

    return dax


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
    ax8 = fig.add_subplot(gs[:2, 6:], frameon=False)

    # error
    ax9 = fig.add_subplot(gs[2:4, 6:], frameon=False)

    # parameters (chi2, ...)
    ax10 = fig.add_subplot(gs[4, 6:], frameon=False)

    # nb of iterations
    ax11 = fig.add_subplot(gs[5, 6:], frameon=False)

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0, 'type': 'matrix'},
        'vertical': {'handle': ax1, 'type': 'misc'},
        'horizontal': {'handle': ax2, 'type': 'misc'},
        'traces': {'handle': ax3, 'type': 'misc'},
        # inversion
        'retrofit': {'handle': ax8, 'type': 'misc'}
        'err': {'handle': ax9, 'type': 'misc'}
        'inv-param': {'handle': ax10, 'type': 'misc'}
        'niter': {'handle': ax11, 'type': 'misc'}
        # text
        'textX': {'handle': ax4, 'type': 'text'},
        'textY': {'handle': ax5, 'type': 'text'},
        'textZ': {'handle': ax6, 'type': 'text'},
    }

    if ax7 is not None:
        dax['radial'] = {'handle': ax7, 'type': 'misc'}
    return dax

