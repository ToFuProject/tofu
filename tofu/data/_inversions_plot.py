# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


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
    aspect=None,
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
    refbs = coll.dobj['bsplines'][keybs]['ref']

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='auto',
        types=str,
        allowed=['auto', 'equal'],
    )

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

    return keyinv, keymat, keybs, keym, cmap, aspect, dcolorbar, dleg


def plot_inversion(
    coll=None,
    key=None,
    indt=None,
    res=None,
    vmin=None,
    vmax=None,
    res=None,
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
        keyinv, keymat, keybs, keydata, cmap, aspect, dcolorbar, dleg,
    ) = _plot_geometry_matrix_check(
        key=key,
        cmap=cmap,
        aspect=aspect,
        dcolorbar=dcolorbar,
        dleg=dleg,
    )

    # ------------
    # prepare data

    data = coll.ddata[keydata]['data']
    keychan = coll.ddata[keydata]['ref'][-1]
    chan = coll.ddata[keychan]['data']
    matrix = coll.ddata[keymat]['data']
    data_re = matrix.dot(coll.ddata[keyinv]['data'].T).T

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
        gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.set_ylabel(f'channels')
        ax01.set_xlabel(f'basis functions')
        ax01.set_title(key, size=14)
        ax01.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax01.xaxis.set_label_position('top')
        ax00 = fig.add_subplot(gs[0, 0], sharex=ax01)
        ax00.set_xlabel(f'basis functions (m)')
        ax00.set_ylabel(f'data')
        ax10 = fig.add_subplot(gs[1, 0], aspect='equal')
        ax10.set_xlabel(f'R (m)')
        ax10.set_ylabel(f'Z (m)')
        ax11 = fig.add_subplot(gs[1, 1], aspect='equal')
        ax11.set_ylabel(f'R (m)')
        ax11.set_xlabel(f'Z (m)')
        ax02 = fig.add_subplot(gs[0, 2], sharey=ax01)
        ax02.set_xlabel(f'channels')
        ax02.set_ylabel(f'data')
        ax02.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax02.xaxis.set_label_position('top')
        ax02.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax02.yaxis.set_label_position('right')
        ax12 = fig.add_subplot(gs[1, 2], aspect='equal')
        ax12.set_xlabel(f'R (m)')
        ax12.set_ylabel(f'Z (m)')

        dax = {
            'cross': ax0,
            'data': {'ax': ax10, 'type': 'cross'},
            'trace0': {'ax': ax12, 'type': 'cross'},
            'trace1': {'ax': ax11, 'type': 'cross'},
            'trace2': {'ax': ax02, 'type': 'misc'},
        }

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # ---------
    # plot inv

    dax = coll.plot_profile2d(
        key=keyinv,
        indt=indt,
        res=res,
        dax=dax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # ---------
    # plot data

    kax = 'data'
    if dax.get('data') is not None:
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
            data_re[indt, :],
            c='k',
            ls='--',
            lw=1.,
            marker='.',
        )

    kax = 'trace0'
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

    return dax
