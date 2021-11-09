# -*- coding: utf-8 -*-


# Built-in
import datetime as dtm


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


# specific
from . import _generic_check


# #############################################################################
# #############################################################################
#                           plot matrix
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
    lk = list(coll.dobj['matrix'].keys())
    key = _generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lk,
    )
    keybs = coll.dobj['matrix'][key]['bsplines']
    refbs = coll.dobj['bsplines'][keybs]['ref']
    keym = coll.dobj['bsplines'][keybs]['mesh']

    # indbf
    if indbf is None:
        indbf = 0
    try:
        assert np.isscalar(indbf)
        indbf = int(indbf)
    except Exception as err:
        msg = (
            f"Arg indbf should be a int!\nProvided: {indt}"
        )
        raise Exception(msg)

    # indchan
    if indchan is None:
        indchan = 0
    try:
        assert np.isscalar(indchan)
        indchan = int(indchan)
    except Exception as err:
        msg = (
            f"Arg indchan should be a int!\nProvided: {indt}"
        )
        raise Exception(msg)

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

    return key, keybs, keym, indbf, indchan, cmap, aspect, dcolorbar, dleg


def _plot_geometry_matrix_prepare(
    cam=None,
    coll=None,
    key=None,
    keybs=None,
    keym=None,
    indbf=None,
    indchan=None,
    res=None,
):

    # res
    deg = coll.dobj['bsplines'][keybs]['deg']
    km = coll.dobj['bsplines'][keybs]['mesh']
    kR, kZ = coll.dobj['mesh'][km]['knots']
    Rk = coll.ddata[kR]['data']
    Zk = coll.ddata[kZ]['data']
    dR = np.min(np.diff(Rk))
    dZ = np.min(np.diff(Zk))
    if res is None:
        res_coef = 0.025
        res = [res_coef*dR, res_coef*dZ]

    # crop
    crop = coll.dobj['matrix'][key]['crop']

    # indchan => indchan_bf
    ich_bf_tup = coll.select_ind(key=keybs, returnas='tuple-flat', crop=crop)
    ich_bf = coll.select_ind(key=keybs, returnas=np.ndarray, crop=crop)
    indbf_full = coll.select_ind(
        key=keybs, returnas='array-flat', crop=crop,
    )[indbf]
    indbf_tup = coll.select_ind(
        key=keybs, ind=indbf_full, returnas=tuple, crop=crop,
    )

    # mesh sampling
    km = coll.dobj['bsplines'][keybs]['mesh']
    R, Z = coll.get_sample_mesh(
        key=km, res=res, mode='abs', grid=True, imshow=True,
    )

    # bsplinetot
    shapebs = coll.dobj['bsplines'][keybs]['shape']
    coefs = np.zeros(tuple(np.r_[1, shapebs]), dtype=float)
    coefs[0, ich_bf_tup[0], ich_bf_tup[1]] = np.nansum(
        coll.ddata[key]['data'],
        axis=0,
    )

    bsplinetot = coll.interp2d(
        key=keybs,
        R=R,
        Z=Z,
        coefs=coefs,
        crop=crop,
        nan0=True,
        details=False,
    )[0, ...]

    # bspline1
    coefs[0, ich_bf_tup[0], ich_bf_tup[1]] = (
        coll.ddata[key]['data'][indchan, :]
    )
    bspline1 = coll.interp2d(
        key=keybs,
        R=R,
        Z=Z,
        coefs=coefs,
        crop=crop,
        nan0=True,
        details=False,
    )[0, ...]

    # los
    ptslos, coefslines, indlosok = None, None, None
    if cam is not None:
        ptslos = cam._get_plotL(return_pts=True, proj='cross', Lplot='tot')
        indsep = np.nonzero(np.isnan(ptslos[0, :]))[0]
        ptslos = np.split(ptslos, indsep, axis=1)
        coefslines = coll.ddata[key]['data'][:, indbf]
        indlosok = np.nonzero(coefslines > 0)[0]
        # normalize for line width
        coefslines = (
            (3. - 0.5) * (coefslines - coefslines.min())
            / (coefslines.max() - coefslines.min()) + 0.5
        )

    # extent and interp
    extent = (
        Rk[0] - 0.*dR, Rk[-1] + 0.*dR,
        Zk[0] - 0.*dZ, Zk[-1] + 0.*dZ,
    )

    if deg == 0:
        interp = 'nearest'
    elif deg == 1:
        interp = 'bilinear'
    elif deg >= 2:
        interp = 'bicubic'

    return (
        bsplinetot, bspline1, extent, interp,
        ptslos, coefslines, indlosok, indbf_tup,
    )


def plot_geometry_matrix(
    cam=None,
    coll=None,
    key=None,
    indbf=None,
    indchan=None,
    vmin=None,
    vmax=None,
    res=None,
    cmap=None,
    aspect=None,
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
):

    # --------------
    # check input

    (
        key, keybs, keym,
        indbf, indchan,
        cmap, aspect, dcolorbar, dleg,
    ) = _plot_geometry_matrix_check(
        coll=coll,
        key=key,
        indbf=indbf,
        indchan=indchan,
        cmap=cmap,
        aspect=aspect,
        dcolorbar=dcolorbar,
        dleg=dleg,
        dax=dax,
    )

    # --------------
    #  Prepare data

    (
        bsplinetot, bspline1,
        extent, interp,
        ptslos, coefslines, indlosok,
        ich_bf_tup,
    ) = _plot_geometry_matrix_prepare(
        cam=cam,
        coll=coll,
        key=key,
        keybs=keybs,
        keym=keym,
        indbf=indbf,
        indchan=indchan,
        res=res,
    )
    nchan, nbs = coll.ddata[key]['data'].shape

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
            'matrix': ax01,
            'cross1': {'ax': ax10, 'type': 'cross'},
            'cross2': {'ax': ax12, 'type': 'cross'},
            'crosstot': {'ax': ax11, 'type': 'cross'},
            'misc1': {'ax': ax02, 'type': 'misc'},
            'misc2': {'ax': ax00, 'type': 'misc'},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # --------------
    # plot mesh

    dax = coll.plot_mesh(
        key=keym, dax=dax, crop=True, dleg=False,
    )

    # --------------
    # plot matrix

    dax = coll.plot_as_matrix(
        key=key, dax=dax, ind=[indchan, indbf],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
    )

    kax = 'cross1'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        im = ax.imshow(
            bspline1,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=0,
            vmax=None,
        )

        if ptslos is not None:
            ax.plot(
                ptslos[indchan][0, :],
                ptslos[indchan][1, :],
                ls='-',
                lw=1.,
                color='k',
            )

    kax = 'cross2'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        coll.plot_bsplines(
            key=keybs,
            ind=ich_bf_tup,
            knots=False,
            cents=False,
            plot_mesh=False,
            dax={'cross': dax[kax]},
            dleg=False,
        )

        if ptslos is not None:
            for ii in indlosok:
                ax.plot(
                    ptslos[ii][0, :],
                    ptslos[ii][1, :],
                    ls='-',
                    lw=coefslines[ii],
                    color='k',
                )

    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        im = ax.imshow(
            bsplinetot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=0,
            vmax=None,
        )

    # --------------
    # dleg

    # if dleg is not False:
        # dax['cross'].legend(**dleg)

    return dax
