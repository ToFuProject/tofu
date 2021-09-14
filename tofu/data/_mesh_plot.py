# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


# #############################################################################
# #############################################################################
#                           plot
# #############################################################################


def _plot_basic_check(
    mesh=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    color=None,
    dleg=None,
):

    # key
    if key is None and len(mesh.dobj[mesh._groupmesh]) == 1:
        key = list(mesh.dobj[mesh._groupmesh].keys())[0]

    c0 = isinstance(key, str) and key in mesh.dobj[mesh._groupmesh].keys()
    if not c0:
        msg = (
            "Arg key must be a valid mesh identifier\n"
            f"\t- Available: {mesh.dobj[mesh._groupmesh].keys()}\n"
            f"\t- Provided: {key}"
        )
        raise Exception(msg)

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
    if dleg is None:
        dleg = {
            'bbox_to_anchor': (1.1, 1.),
            'loc': 'upper left',
            'frameon': True,
        }
    c0 = dleg is False or isinstance(dleg, dict)
    if not c0:
        msg = (
            "Arg dleg must be:\n"
            "\t- False: no legend\n"
            "\t- dict: dict of legend properties\n"
        )
        raise Exception(msg)

    return key, ind_knot, ind_cent, color, dleg


def _plot_basic_prepare(
    mesh=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
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

    # if ind_knot is not None:

    return grid


def plot_basic(
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

    key, ind_knot, ind_cent, color, dleg = _plot_basic_check(
        mesh=mesh,
        key=key,
        ind_knot=ind_knot,
        ind_cent=ind_cent,
        color=color,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    grid = _plot_basic_prepare(
        mesh=mesh, key=key, ind_knot=ind_knot, ind_cent=ind_cent,
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
