# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


_LALLOWED_AXESTYPES = [
    'cross', 'hor',
    'matrix',
    'timetrace',
    'profile1d',
    'image',
    'misc'
]


# #############################################################################
# #############################################################################
#                           utility
# #############################################################################


def _check_var(var, varname, default=None, types=None, allowed=None):

    if var is None:
        var = default

    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be a {types}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    if allowed is not None:
        if var not in allowed:
            msg = (
                f"Arg {varname} must be in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    return var


def _check_dax(dax=None, main=None):

    # None
    if dax is None:
        return dax

    # Axes
    if issubclass(dax.__class__, plt.Axes):
        if main is None:
            msg = (
            )
            raise Exception(msg)
        else:
            return {main: dax}

    # dict
    c0 = (
        isinstance(dax, dict)
        and all([
            isinstance(k0, str)
            and (
                (
                    k0 in _LALLOWED_AXESTYPES
                    and issubclass(v0.__class__, plt.Axes)
                )
                or (
                    isinstance(v0, dict)
                    and issubclass(v0.get('ax').__class__, plt.Axes)
                    and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
        )
        raise Exception(msg)

    for k0, v0 in dax.items():
        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'ax': v0, 'type': k0}

    return dax

# #############################################################################
# #############################################################################
#                           plot matrix
# #############################################################################


def _plot_matrix_check(
    matrix=None,
    key=None,
    indbf=None,
    indchan=None,
    cmap=None,
    dcolorbar=None,
    dleg=None,
    dax=None,
):

    # key
    lk = list(matrix.dobj['matrix'].keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    key = _check_var(key, 'key', default=None, types=str, allowed=lk)
    keybs = matrix.dobj['matrix'][key]['bsplines']
    refbs = matrix.dobj['bsplines'][keybs]['ref']
    keym = matrix.dobj['bsplines'][keybs]['mesh']

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

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = _check_var(
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
    dleg = _check_var(dleg, 'dleg', default=defdleg, types=(bool, dict))

    return key, keybs, keym, indbf, indchan, cmap, dcolorbar, dleg


def _plot_matrix_prepare(
    cam=None,
    matrix=None,
    key=None,
    keybs=None,
    keym=None,
    indbf=None,
    indchan=None,
    res=None,
):

    # res
    deg = matrix.dobj['bsplines'][keybs]['deg']
    km = matrix.dobj['bsplines'][keybs]['mesh']
    kR, kZ = matrix.dobj['mesh'][km]['knots']
    Rk = matrix.ddata[kR]['data']
    Zk = matrix.ddata[kZ]['data']
    dR = np.min(np.diff(Rk))
    dZ = np.min(np.diff(Zk))
    if res is None:
        res_coef = 0.025
        res = [res_coef*dR, res_coef*dZ]

    # grid
    vert = np.array([
        np.repeat(Rk, 3),
        np.tile((Zk[0], Zk[-1], np.nan), Rk.size),
    ])
    hor = np.array([
        np.tile((Rk[0], Rk[-1], np.nan), Zk.size),
        np.repeat(Zk, 3),
    ])
    grid = np.concatenate((vert, hor), axis=1)

    # indchan => indchan_bf
    indchan_bf = matrix.select_ind(key=keybs, returnas=np.ndarray)
    indbf_tup = matrix.select_ind(key=keybs, ind=indbf, returnas=tuple)

    # bspline1
    km = matrix.dobj['bsplines'][keybs]['mesh']
    R, Z = matrix.get_sample_mesh(
        key=km, res=res, mode='abs', grid=True, imshow=True,
    )

    shapebs = matrix.dobj['bsplines'][keybs]['shape']
    coefs = np.zeros(tuple(np.r_[1, shapebs]), dtype=float)
    coefs[0, :, :] = np.nansum(matrix.ddata[key]['data'], axis=0)[indchan_bf]
    bsplinetot = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs,
    )[0, ...]
    bsplinetot[bsplinetot == 0] = np.nan
    coefs[0, :, :] = matrix.ddata[key]['data'][indchan, indchan_bf]
    bspline1 = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs,
    )[0, ...]
    bspline1[bspline1 == 0] = np.nan

    # bspline2
    coefs[...] = 0.
    coefs[0, indbf_tup[0], indbf_tup[1]] = 1.
    bspline2 = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs,
    )[0, ...]
    bspline2[bspline2 == 0] = np.nan

    # los
    ptslos, coefslines, indlosok = None, None, None
    if cam is not None:
        ptslos = cam._get_plotL(return_pts=True, proj='cross', Lplot='tot')
        indsep = np.nonzero(np.isnan(ptslos[0, :]))[0]
        ptslos = np.split(ptslos, indsep, axis=1)
        coefslines = matrix.ddata[key]['data'][:, indbf]
        indlosok = np.nonzero(coefslines > 0)[0]
        # normalize for line width
        coefslines =  (
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
        bsplinetot, bspline1, bspline2, extent, interp,
        grid, ptslos, coefslines, indlosok
    )


def plot_matrix(
    cam=None,
    matrix=None,
    key=None,
    indbf=None,
    indchan=None,
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

    # --------------
    # check input

    (
        key, keybs, keym,
        indbf, indchan,
        cmap, dcolorbar, dleg,
    ) = _plot_matrix_check(
        matrix=matrix,
        key=key,
        indbf=indbf,
        indchan=indchan,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
        dax=dax,
    )

    # --------------
    #  Prepare data

    (
        bsplinetot, bspline1, bspline2,
        extent, interp, grid,
        ptslos, coefslines, indlosok,
    ) = _plot_matrix_prepare(
        cam=cam,
        matrix=matrix,
        key=key,
        keybs=keybs,
        keym=keym,
        indbf=indbf,
        indchan=indchan,
        res=res,
    )
    nchan, nbs = matrix.ddata[key]['data'].shape

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (16, 9)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.98,
                'bottom': 0.05, 'top': 0.95,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.set_xlabel(f'basis functions (m)')
        ax00.set_ylabel(f'matrix')
        ax10 = fig.add_subplot(gs[1, 0], aspect='equal')
        ax10.set_xlabel(f'R (m)')
        ax10.set_ylabel(f'Z (m)')
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.set_ylabel(f'channels')
        ax01.set_xlabel(f'basis functions')
        ax01.set_title(key, size=14)
        ax11 = fig.add_subplot(gs[1, 1], aspect='equal')
        ax11.set_ylabel(f'R (m)')
        ax11.set_xlabel(f'Z (m)')
        ax02 = fig.add_subplot(gs[0, 2])
        ax02.set_xlabel(f'channels')
        ax02.set_ylabel(f'matrix')
        ax12 = fig.add_subplot(gs[1, 2], aspect='equal')
        ax12.set_xlabel(f'R (m)')
        ax12.set_ylabel(f'Z (m)')

        dax = {
            'matrix': ax01,
            'cross1': {'ax': ax10, 'type': 'cross'},
            'cross2': {'ax': ax12, 'type': 'cross'},
            'crosstot': {'ax': ax11, 'type': 'cross'},
            'misc1': {'ax': ax00, 'type': 'misc'},
            'misc2': {'ax': ax02, 'type': 'misc'},
        }

    dax = _check_dax(dax=dax, main='matrix')

    # --------------
    # plot

    kax = 'matrix'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        # matrix
        im = ax.imshow(
            matrix.ddata[key]['data'],
            interpolation='nearest',
            origin='upper',
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(im, ax=ax, **dcolorbar)

        # indbf, indchan
        ax.axhline(indchan, c='r', lw=1., ls='-')
        ax.axvline(indbf, c='r', lw=1., ls='-')


    kax = 'misc1'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        ax.plot(
            np.arange(0, nbs),
            matrix.ddata[key]['data'][indchan, :],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
        )

    kax = 'misc2'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        ax.plot(
            np.arange(0, nchan),
            matrix.ddata[key]['data'][:, indbf],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
        )

    kax = 'cross1'
    if dax.get(kax) is not None:
        ax = dax[kax]['ax']

        ax.plot(
            grid[0, :],
            grid[1, :],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
            label=keym,
        )

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

        ax.plot(
            grid[0, :],
            grid[1, :],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
            label=keym,
        )

        if bspline2 is not None:
            im = ax.imshow(
                bspline2,
                extent=extent,
                interpolation=interp,
                origin='lower',
                aspect='equal',
                cmap=cmap,
                vmin=0,
                vmax=1,
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

        ax.plot(
            grid[0, :],
            grid[1, :],
            ls='-',
            marker='None',
            lw=1.,
            color='k',
            label=keym,
        )

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
