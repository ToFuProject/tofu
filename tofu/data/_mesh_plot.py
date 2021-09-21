# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


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


# #############################################################################
# #############################################################################
#                           plot mesh
# #############################################################################


def _plot_mesh_check(
    mesh=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    color=None,
    dleg=None,
):

    # key
    lk = list(mesh.dobj[mesh._groupmesh].keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    key = _check_var(key, 'key', default=None, types=str, allowed=lk)

    # ind_knot
    if ind_knot is not None:
        ind_knot = mesh.select_mesh_elements(
            key=key, ind=ind_knot, elements='knots',
            returnas='data', return_neighbours=True,
        )

    # ind_cent
    if ind_cent is not None:
        ind_cent = mesh.select_mesh_elements(
            key=key, ind=ind_cent, elements='cent',
            returnas='data', return_neighbours=True,
        )

    # color
    if color is None:
        color = 'k'
    if not mcolors.is_color_like(color):
        msg = (
            "Arg color must be a valid matplotlib color identifier!\n"
            f"Provided: {color}"
        )
        raise Exception(msg)

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _check_var(dleg, 'dleg', default=defdleg, types=(bool, dict))

    return key, ind_knot, ind_cent, color, dleg


def _plot_mesh_prepare(
    mesh=None,
    key=None,
):

    Rk = mesh.dobj[mesh._groupmesh][key]['R-knots']
    Zk = mesh.dobj[mesh._groupmesh][key]['Z-knots']
    R = mesh.ddata[Rk]['data']
    Z = mesh.ddata[Zk]['data']

    vert = np.array([
        np.repeat(R, 3),
        np.tile((Z[0], Z[-1], np.nan), R.size),
    ])
    hor = np.array([
        np.tile((R[0], R[-1], np.nan), Z.size),
        np.repeat(Z, 3),
    ])
    grid = np.concatenate((vert, hor), axis=1)

    return grid


def plot_mesh(
    mesh=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    color=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
):

    # --------------
    # check input

    key, ind_knot, ind_cent, color, dleg = _plot_mesh_check(
        mesh=mesh,
        key=key,
        ind_knot=ind_knot,
        ind_cent=ind_cent,
        color=color,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    grid = _plot_mesh_prepare(
        mesh=mesh,
        key=key,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if dmargin is None:
            dmargin = {
                'left': 0.1, 'right': 0.9,
                'bottom': 0.1, 'top': 0.9,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')

        dax = {'cross': ax0}

    # --------------
    # plot

    kax = 'cross'
    if dax.get(kax) is not None:
        dax[kax].plot(
            grid[0, :],
            grid[1, :],
            color=color,
            ls='-',
            label=key,
        )

        if ind_knot is not None:
            dax[kax].plot(
                ind_knot[0][0],
                ind_knot[0][1],
                marker='o',
                ms=8,
                ls='None',
                color=color,
                label='knots',
            )
            dax[kax].plot(
                ind_knot[1][0, :, :],
                ind_knot[1][1, :, :],
                marker='x',
                ms=4,
                ls='None',
                color=color,
            )

        if ind_cent is not None:
            dax[kax].plot(
                ind_cent[0][0],
                ind_cent[0][1],
                marker='x',
                ms=8,
                ls='None',
                color=color,
                label='cents',
            )
            dax[kax].plot(
                ind_cent[1][0, :, :],
                ind_cent[1][1, :, :],
                marker='o',
                ms=4,
                ls='None',
                color=color,
            )

    # --------------
    # dleg

    if dleg is not False:
        dax['cross'].legend(**dleg)

    return dax


# #############################################################################
# #############################################################################
#                           plot bspline
# #############################################################################


def _plot_bspline_check(
    mesh=None,
    key=None,
    ind=None,
    knots=None,
    cents=None,
    cmap=None,
    dleg=None,
):

    # key
    lk = list(mesh.dobj['bsplines'].keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    key = _check_var(key, 'key', default=None, types=str, allowed=lk)

    # knots, cents
    knots = _check_var(knots, 'knots', default=True, types=bool)
    cents = _check_var(cents, 'cents', default=True, types=bool)

    # ind_bspline
    ind = mesh.select_bsplines(
        key=key,
        ind=ind,
        returnas='ind',
        return_knots=False,
        return_cents=False,
    )

    _, knotsi, centsi = mesh.select_bsplines(
        key=key,
        ind=ind,
        returnas='data',
        return_knots=True,
        return_cents=True,
    )

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _check_var(dleg, 'dleg', default=defdleg, types=(bool, dict))

    return key, ind, knotsi, centsi, cmap, dleg


def _plot_bspline_prepare(
    mesh=None,
    key=None,
    ind=None,
    res=None,
    knots=None,
    cents=None,
):

    # check input
    deg = mesh.dobj['bsplines'][key]['deg']
    km = mesh.dobj['bsplines'][key]['mesh']
    kR = mesh.dobj['mesh'][km]['R-knots']
    kZ = mesh.dobj['mesh'][km]['Z-knots']
    Rk = mesh.ddata[kR]['data']
    Zk = mesh.ddata[kZ]['data']
    dR = np.min(np.diff(Rk))
    dZ = np.min(np.diff(Zk))
    if res is None:
        res_coef = 0.05
        res = [res_coef*dR, res_coef*dZ]

    # bspline
    km = mesh.dobj['bsplines'][key]['mesh']
    R, Z = mesh.get_sample_mesh(key=km, res=res, mode='abs', grid=True)

    shapebs = mesh.dobj['bsplines'][key]['shapebs']
    coefs = np.zeros((1, shapebs[0], shapebs[1]), dtype=float)
    coefs[0, ind[0], ind[1]] = 1.

    bspline = mesh.dobj['bsplines'][key]['func_sum'](R, Z, coefs=coefs)[0, ...]
    bspline[bspline == 0] = np.nan

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

    # knots and cents

    return bspline, extent, interp, knots, cents


def plot_bspline(
    mesh=None,
    key=None,
    ind=None,
    knots=None,
    cents=None,
    res=None,
    cmap=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
):

    # --------------
    # check input

    key, ind, knotsi, centsi, cmap, dleg = _plot_bspline_check(
        mesh=mesh,
        key=key,
        ind=ind,
        knots=knots,
        cents=cents,
        cmap=cmap,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    bspline, extent, interp, knotsi, centsi = _plot_bspline_prepare(
        mesh=mesh,
        key=key,
        ind=ind,
        knots=knotsi,
        cents=centsi,
        res=res,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if dmargin is None:
            dmargin = {
                'left': 0.1, 'right': 0.9,
                'bottom': 0.1, 'top': 0.9,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')

        dax = {'cross': ax0}

    # --------------
    # plot

    kax = 'cross'
    if dax.get(kax) is not None:

        dax[kax].imshow(
            bspline,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=0.,
            vmax=1.,
        )

        if knots is not False:
            dax[kax].plot(
                knotsi[0].ravel(),
                knotsi[1].ravel(),
                marker='x',
                ms=6,
                ls='None',
                color='k',
            )

        if cents is not False:
            dax[kax].plot(
                centsi[0].ravel(),
                centsi[1].ravel(),
                marker='o',
                ms=6,
                ls='None',
                color='k',
            )

    # --------------
    # dleg

    if dleg is not False:
        dax['cross'].legend(**dleg)

    return dax


# #############################################################################
# #############################################################################
#                           plot profile2d
# #############################################################################


def _plot_profile2d_check(
    mesh=None,
    key=None,
    indt=None,
    cmap=None,
    dcolorbar=None,
    dleg=None,
):

    # key
    dk = mesh.get_profiles2d()
    if key is None and len(dk) == 1:
        key = list(dk.keys())[0]
    key = _check_var(
        key, 'key', default=None, types=str, allowed=list(dk.keys())
    )
    keybs = dk[key]
    refbs = mesh.dobj['bsplines'][keybs]['ref']

    # indt
    if len(mesh.ddata[key]['ref']) > len(refbs):
        if indt is None and mesh.ddata[key]['data'].shape[0] == 1:
            indt = 0
        try:
            assert np.isscalar(indt)
            indt = int(indt)
        except Exception as err:
            msg = (
                f"Arg indt should be a int!\nProvided: {indt}"
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

    return key, keybs, indt, cmap, dcolorbar, dleg


def _plot_profiles2d_prepare(
    mesh=None,
    key=None,
    keybs=None,
    indt=None,
    res=None,
):

    # check input
    deg = mesh.dobj['bsplines'][keybs]['deg']
    km = mesh.dobj['bsplines'][keybs]['mesh']
    kR = mesh.dobj['mesh'][km]['R-knots']
    kZ = mesh.dobj['mesh'][km]['Z-knots']
    Rk = mesh.ddata[kR]['data']
    Zk = mesh.ddata[kZ]['data']
    dR = np.min(np.diff(Rk))
    dZ = np.min(np.diff(Zk))
    if res is None:
        res_coef = 0.05
        res = [res_coef*dR, res_coef*dZ]

    # bspline
    km = mesh.dobj['bsplines'][keybs]['mesh']
    R, Z = mesh.get_sample_mesh(key=km, res=res, mode='abs', grid=True)

    shapebs = mesh.dobj['bsplines'][keybs]['shapebs']
    coefs = mesh.ddata[key]['data']

    if len(coefs.shape) > len(shapebs):
        coefs = coefs[indt:indt+1, ...]

    bspline = mesh.dobj['bsplines'][keybs]['func_sum'](
        R, Z, coefs=coefs,
    )[0, ...]
    bspline[bspline == 0] = np.nan

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

    # knots and cents

    return bspline, extent, interp


def plot_profile2d(
    mesh=None,
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

    # --------------
    # check input

    key, keybs, indt, cmap, dcolorbar, dleg = _plot_profile2d_check(
        mesh=mesh,
        key=key,
        indt=indt,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    bspline, extent, interp = _plot_profiles2d_prepare(
        mesh=mesh,
        key=key,
        keybs=keybs,
        indt=indt,
        res=res,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if dmargin is None:
            dmargin = {
                'left': 0.1, 'right': 0.9,
                'bottom': 0.1, 'top': 0.9,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')

        dax = {'cross': ax0}

    # --------------
    # plot

    kax = 'cross'
    if dax.get(kax) is not None:

        im = dax[kax].imshow(
            bspline,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(im, ax=dax[kax], **dcolorbar)

    # --------------
    # dleg

    if dleg is not False:
        dax['cross'].legend(**dleg)

    return dax
