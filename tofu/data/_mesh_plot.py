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
        ind_knot = mesh.select_elements(
            key=key, ind=ind_knot, elements='knots',
            returnas='data', return_neighbours=True,
        )

    # ind_knot
    if ind_cent is not None:
        ind_cent = mesh.select_elements(
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
    dleg = _check_var(dleg, 'dleg', default=defleg, types=(bool, dict))

    return key, ind_knot, ind_cent, color, cmap, dleg


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
        ind_knot=ind_knot,
        ind_cent=ind_cent,
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
                ind_knot[0][0, :],
                ind_knot[0][1, :],
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
                label='knots - neigh',
            )

        if ind_cent is not None:
            dax[kax].plot(
                ind_cent[0][0, :],
                ind_cent[0][1, :],
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
                label='cent - neigh',
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
    ind_bspline=None,
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

    # ind_bspline
    if ind_bspline is not None:
        ind_bspline = mesh.select_elements(
            key=key, ind=ind_knot, elements='knots',
            returnas='data', return_neighbours=True,
        )

    # knots, cents
    knots = _check_var(knots, 'knots', default=True, types=bool)
    cents = _check_var(cents, 'cents', default=True, types=bool)

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _check_var(dleg, 'dleg', default=defleg, types=(bool, dict))

    return key, ind_bspline, knots, cents, cmap, dleg


def _plot_bspline_prepare(
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

    return bspline, extent, interp, knots, cents


def plot_bspline(
    mesh=None,
    key=None,
    ind_bspline=None,
    knots=None,
    cents=None,
    cmap=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
):

    # --------------
    # check input

    key, ind_bspline, knots, cents, cmap, dleg = _plot_bspline_check(
        mesh=mesh,
        key=key,
        ind_bspline=ind_bspline,
        knots=knots,
        cents=cents,
        cmap=cmap,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    bspline, extent, interp, knots, cents = _plot_bspline_prepare(
        mesh=mesh,
        key=key,
        ind_bspline=ind_bspline,
        knots=knots,
        cents=cents,
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

        if ind_bspline is not None:
            dax[kax].imshow(
                bspline,
                extent=extent,
                interpolation=interpolation,
                origin='lower',
                aspect='equal',
                cmap=cmap,
            )

        if knots is not False:
            dax[kax].plot(
                knots[0, :],
                knots[1, :],
                marker='x',
                ms=6,
                ls='None',
                color='k',
            )

        if cents is not False:
            dax[kax].plot(
                cents[0, :],
                cents[1, :],
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
