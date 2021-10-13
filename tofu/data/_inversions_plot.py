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


def _plot_geometry_matrix_check(
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
    ) = _plot_geometry_matrix_check(
        coll=coll,
        key=key,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
    )

    # ------------
    # prepare data

    # synthetic ?
    synthetic = keydata in coll.dobj.get('synthetic', {}).keys()

    # data
    keychan = coll.ddata[keydata]['ref'][-1]
    chan = coll.ddata[keychan]['data']
    if 'time' in coll.ddata[keyinv]['group']:
        data = coll.ddata[keydata]['data']
        keyt = coll.ddata[keydata]['ref'][0]
        time = coll.ddata[keyt]['data']
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
    if 'time' in coll.ddata[keyinv]['group']:
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
    if 'time' in coll.ddata[keyinv]['group']:
        nchi2n = chan.size * coll.ddata[f'{keyinv}-niter']['data']
        mu = coll.ddata[f'{keyinv}-mu']['data']
        reg = coll.ddata[f'{keyinv}-reg']['data']
        niter = coll.ddata[f'{keyinv}-niter']['data']
    else:
        nchi2n = chan.size * coll.dobj['inversions'][keyinv]['chi2n']
        mu = coll.dobj['inversions'][keyinv]['mu']
        reg = coll.dobj['inversions'][keyinv]['reg']
        niter = coll.dobj['inversions'][keyinv]['niter']

    # indt
    if indt is None:
        indt = 0

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (16, 9)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.98,
                'bottom': 0.05, 'top': 0.95,
                'hspace': 0.15, 'wspace': 0.15,
            }

        fig = plt.figure(figsize=fs)
        if synthetic:
            gs = gridspec.GridSpec(ncols=4, nrows=2, **dmargin)
            ax0 = fig.add_subplot(gs[1, 1])
            ax1 = fig.add_subplot(gs[:2, 2:])
            ax2 = fig.add_subplot(gs[2, 2:], sharex=ax1)
            ax3 = fig.add_subplot(gs[3, 2:])
            ax4 = fig.add_subplot(gs[4, 2:], sharex=ax3)

            ax5 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
            ax6 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)

        else:
            gs = gridspec.GridSpec(ncols=2, nrows=5, **dmargin)
            ax0 = fig.add_subplot(gs[:, 0])
            ax1 = fig.add_subplot(gs[:2, 1])
            ax2 = fig.add_subplot(gs[2, 1], sharex=ax1)
            ax3 = fig.add_subplot(gs[3, 1])
            ax4 = fig.add_subplot(gs[4, 1], sharex=ax3)

        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')
        ax0.set_title('reconstruction', size=14)

        ax1.set_xlabel(f'chan')
        ax1.set_ylabel(f'data')
        ax1.set_title('data', size=14)

        ax2.set_xlabel(f'chan')
        ax2.set_ylabel(f'error')

        ax3.set_xlabel(f'time')
        ax3.set_ylabel(f'potential')

        ax4.set_xlabel(f'time')
        ax4.set_ylabel(f'niter')

        dax = {
            'reconstruction': {'ax': ax0, 'type': 'cross'},
            'data': {'ax': ax1, 'type': 'misc'},
            'data-err': {'ax': ax2, 'type': 'misc'},
            'inv-param': {'ax': ax3, 'type': 'misc'},
            'niter': {'ax': ax4, 'type': 'misc'},
        }

        if synthetic:
            ax5.set_xlabel(f'R (m)')
            ax5.set_ylabel(f'Z (m)')
            ax5.set_title('original', size=14)

            ax6.set_xlabel(f'R (m)')
            ax6.set_ylabel(f'Z (m)')
            ax6.set_title('error', size=14)

            dax.update({
                'original': {'ax': ax5, 'type': 'cross'},
                'error': {'ax': ax6, 'type': 'cross'},
            })

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # ---------
    # plot inv

    kax = 'reconstruction'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']
        coll.plot_profile2d(
            key=keyinv,
            indt=indt,
            res=res,
            dax={'cross': ax},
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dleg=False,
        )

    # ---------
    # plot data

    kax = 'data'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']
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
        ax = dax[kax]['ax']
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
        ax = dax[kax]['ax']
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
        ax = dax[kax]['ax']
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
