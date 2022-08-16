# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds

# specific
from . import _mesh_comp


# ##################################################################
# ##################################################################
#                           plot check
# ##################################################################


def _plot_diagnostic_check(
    # figure
    proj=None,
    # interactivity
    connect=None,
):

    # -----
    # proj

    proj = _proj(
        proj=proj,
        pall=['cross', 'hor', '3d', 'camera'],
    )

    # -------
    # connect

    connect = ds._generic_check._check_var(
        connect, 'connect',
        types=bool,
        default=True,
    )

    return (
        proj,
        connect,
    )


# ##################################################################
# ##################################################################
#                           plot main
# ##################################################################


def _plot_diagnostic(
    coll=None,
    key=None,
    optics=None,
    element=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    # interactivity
    connect=None,
):

    # ------------
    # check inputs

    (
        proj,
        connect,
    ) = _plot_diagnostic_check(
        # figure
        proj=proj,
        # interactivity
        connect=connect,
    )

    # ------------
    # prepare data

    dplot = coll.get_diagnostic_dplot(
        key=key,
        optics=optics,
        elements=elements,
    )

    cam = coll.dobj['diagnostic'][key]['optics'][0]

    # -----------------
    # prepare figure

    if dax is None:

        dax = _generic_plot.get_dax_diag(
            proj=proj,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot static parts

    for k0, v0 in dplot.items():

        for k1, v1 in v0.items():

            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    v1['r'],
                    v1['z'],
                    c='k',
                    ls='-',
                    lw=1.,
                    label=v1['label'],
                )

            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    v1['x'],
                    v1['y'],
                    c='k',
                    ls='-',
                    lw=1.,
                    label=v1['label'],
                )


            kax = '3d'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    v1['x'],
                    v1['y'],
                    v1['z'],
                    c='k',
                    ls='-',
                    lw=1.,
                    label=v1['label'],
                )

            if k0 == cam:
                kax = 'camera'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        c='k',
                        ls='-',
                        lw=1.,
                        label=v1['label'],
                    )

    return dax
