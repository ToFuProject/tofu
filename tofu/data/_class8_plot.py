# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


# specific
from . import _mesh_comp
from . import _generic_check
from . import _generic_plot


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

    proj = _generic_plot._proj(
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
    elements=None,
    proj=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
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
            tit=key,
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
                    **v1.get('props', {}),
                )

            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    v1['x'],
                    v1['y'],
                    **v1.get('props', {}),
                )

            kax = '3d'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    v1['x'],
                    v1['y'],
                    v1['z'],
                    **v1.get('props', {}),
                )

            # plotting of 2d camera contour
            c0 = (
                k0 == cam
                and k1 == 'o'
                and coll.dobj['camera'][k0]['dgeom']['type'] == '2d'
            )

            if c0:
                kax = 'camera'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1.get('props', {}),
                    )

    return dax
