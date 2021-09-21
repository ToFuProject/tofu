# -*- coding: utf-8 -*-


# Built-in
import datetime as dtm

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


    t0 = dtm.datetime.now()     # DB

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

    # crop
    crop = matrix.dobj['matrix'][key]['crop']

    t1 = dtm.datetime.now()     # DB
    print('\tres and knots', t1-t0)     # DB

    # indchan => indchan_bf
    ich_bf_tup = matrix.select_ind(key=keybs, returnas='tuple-flat', crop=crop)
    ich_bf = matrix.select_ind(key=keybs, returnas=np.ndarray, crop=crop)
    indbf_full = matrix.select_ind(
        key=keybs, returnas='array-flat', crop=crop,
    )[indbf]
    indbf_tup = matrix.select_ind(
        key=keybs, ind=indbf_full, returnas=tuple, crop=crop,
    )

    t2 = dtm.datetime.now()     # DB
    print('\tindices', t2-t1)     # DB

    # mesh sampling
    km = matrix.dobj['bsplines'][keybs]['mesh']
    R, Z = matrix.get_sample_mesh(
        key=km, res=res, mode='abs', grid=True, imshow=True,
    )

    t3 = dtm.datetime.now()     # DB
    print('\tmesh sampling', t3-t2)     # DB

    # bsplinetot
    shapebs = matrix.dobj['bsplines'][keybs]['shape']
    coefs = np.zeros(tuple(np.r_[1, shapebs]), dtype=float)
    coefs[0, ich_bf_tup[0], ich_bf_tup[1]] = np.nansum(
        matrix.ddata[key]['data'],
        axis=0,
    )

    bsplinetot = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs, crop=crop,
    )[0, ...]
    bsplinetot[bsplinetot == 0] = np.nan

    t4 = dtm.datetime.now()     # DB
    print('\tbsplinetot:', t4-t3)     # DB

    # bspline1
    coefs[0, ich_bf_tup[0], ich_bf_tup[1]] = matrix.ddata[key]['data'][indchan, :]
    bspline1 = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs, crop=crop,
    )[0, ...]
    bspline1[bspline1 == 0] = np.nan

    t5 = dtm.datetime.now()     # DB
    print('\tbspline1: ', t5-t4)     # DB

    # bspline2
    coefs[...] = 0.
    coefs[0, indbf_tup[0], indbf_tup[1]] = 1.
    bspline2 = matrix.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs,
    )[0, ...]
    bspline2[bspline2 == 0] = np.nan

    t6 = dtm.datetime.now()     # DB
    print('\tbspline2', t6-t5)     # DB

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

    t7 = dtm.datetime.now()     # DB
    print('\tlos', t7-t6)     # DB

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
        ptslos, coefslines, indlosok
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

    t0 = dtm.datetime.now()     # DB

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

    t1 = dtm.datetime.now()     # DB
    print('checks: ', t1-t0)     # DB

    # --------------
    #  Prepare data

    (
        bsplinetot, bspline1, bspline2,
        extent, interp,
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

    t2 = dtm.datetime.now()     # DB
    print('prepare: ', t2-t1)     # DB

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

    t3 = dtm.datetime.now()     # DB
    print('prepare dax: ', t3-t2)     # DB

    # --------------
    # plot mesh

    dax = matrix.plot_mesh(
        key=keym, dax=dax, crop=True, dleg=False,
    )

    t4 = dtm.datetime.now()     # DB
    print('plot mesh: ', t4-t3)     # DB

    # --------------
    # plot matrix

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

    t5 = dtm.datetime.now()     # DB
    print('plot matrix and misc: ', t5-t4)     # DB


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

    t6 = dtm.datetime.now()     # DB
    print('plot 3 cross: ', t6-t5)     # DB


    # --------------
    # dleg

    # if dleg is not False:
        # dax['cross'].legend(**dleg)

    return dax
