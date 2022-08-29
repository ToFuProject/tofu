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
    coll=None,
    key=None,
    # figure
    proj=None,
    data=None,
    # interactivity
    connect=None,
):

    # -------
    # key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # -----
    # proj

    proj = _generic_plot._proj(
        proj=proj,
        pall=['cross', 'hor', '3d', 'camera'],
    )

    # -------
    # data

    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['diag'] == key
    ]
    data = ds._generic_check._check_var(
        data, 'data',
        types=str,
        default=lok,
    )

    # -------
    # nlos

    nlos = ds._generic_check._check_var(
        nlos, 'nlos',
        types=int,
        default=4,
    )

    # -------
    # connect

    connect = ds._generic_check._check_var(
        connect, 'connect',
        types=bool,
        default=True,
    )

    return (
        key,
        proj,
        data,
        nlos,
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
    data=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    nlos=None,
    dinc=None,
    connect=None,
):

    # ------------
    # check inputs

    (
        key,
        proj,
        data,
        nlos,
        connect,
    ) = _plot_diagnostic_check(
        coll=coll,
        key=key,
        # figure
        proj=proj,
        data=data,
        # interactivity
        nlos=nlos,
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
    is2d = coll.dobj['camera'][cam]['type'] == '2d'

    # -------------------------
    # prepare los interactivity

    los = coll.dobj['diagnostic'][key]['los']
    if los is not None:
        los_x, los_y, los_z = coll.sample_rays(
            key=los,
            res=los_res,
            mode='rel',
        )
        los_r = np.hypot(los_x, los_y)
        reflos = coll.dobj['rays'][los]['ref'][1:]

        if is2d:
            refx, refy = reflos
        else:
            refx = reflos[0]

        coll2 = Diagnostic()
        coll2.add_ref(size=los_x.shape[0])
        coll2.add_ref(key=refx, size=los_x.shape[1])
        if is2d:
            col2.add_ref(key=refy, size=los.shape[2])

        coll2.add_data(key='los_x', data=los_x, ref=reflos)
        coll2.add_data(key='los_y', data=los_y, ref=reflos)
        coll2.add_data(key='los_z', data=los_z, ref=reflos)

    # -------------------------
    # prepare data interactivity

    has3d = False

    if has3d is True:
        reft = None

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

            # cross
            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['r'],
                        v1['z'],
                        v1['ur'],
                        v1['uz'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['r'],
                        v1['z'],
                        **v1.get('props', {}),
                    )

            # hor
            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['x'],
                        v1['y'],
                        v1['ux'],
                        v1['uy'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['x'],
                        v1['y'],
                        **v1.get('props', {}),
                    )

            # 3d
            kax = '3d'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['x'],
                        v1['y'],
                        v1['z'],
                        v1['ux'],
                        v1['uy'],
                        v1['uz'],
                        **v1.get('props', {}),
                    )

                else:
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

    # ----------------
    # define and set dgroup


    dgroup = {
        'x': {
            'ref': [refx],
            'data': ['index'],
            'nmax': nlos,
        },
    }

    if is2d:
        dgroup.update({
            'y': {
                'ref': [refx],
                'data': ['index'],
                'nmax': nlos,
            },
        })

    if has3d:
        dgroup.update({
            't': {
                'ref': [reft],
                'data': ['index'],
                'nmax': 1,
            },
        })

    # ------------------
    # plot mobile parts

    if los is not None:

        nan = np.full((los_x.shape[0],), np.nan)

        # cross
        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            for ii in range(nlos):
                l0, = ax.plot(
                    nan,
                    nan,
                    c='k',
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-cross-{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=reflos,
                    data=['index', 'index'],
                    dtype=['xdata', 'ydata'],
                    axes=kax,
                    ind=ii,
                )

        # hor
        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            for ii in range(nlos):
                l0, = ax.plot(
                    nan,
                    nan,
                    c='k',
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-hor-{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=reflos,
                    data=['index', 'index'],
                    dtype=['xdata', 'ydata'],
                    axes=kax,
                    ind=ii,
                )

        # 3d
        kax = '3d'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            for ii in range(nlos):
                l0, = ax.plot(
                    nan,
                    nan,
                    nan,
                    c='k',
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-3d-{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=reflos,
                    data=['index', 'index', 'index'],
                    dtype=['xdata', 'ydata', 'zdata'],
                    axes=kax,
                    ind=ii,
                )

        # camera
        if is2d:
            pass

        else:
            pass


    # -------
    # connect

    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

        coll2.show_commands()
        return coll2
    else:
        return coll2, dgroup
