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
    rocking_curve=None,
    # interactivity
    color_dict=None,
    nlos=None,
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
    optics, optics_cls = coll.get_diagnostic_optics(key)
    if 'crystal' in optics_cls:
        kcryst = optics[optics_cls.index('crystal')]
    else:
        kcryst = None

    # -----
    # proj

    proj = _generic_plot._proj(
        proj=proj,
        pall=['cross', 'hor', '3d', 'camera'],
    )

    # -------
    # data

    defdata = 'etendue'
    c0 = data is None and coll.dobj['diagnostic'][key].get(defdata) is None

    ref = None
    if not c0:
        lok = [
            k0 for k0, v0 in coll.ddata.items()
            if v0.get('camera') == coll.dobj['diagnostic'][key]['optics'][0]
        ]
        ldiag = [
            k0 for k0, v0 in coll.dobj['diagnostic'][key].items()
            if isinstance(v0, str) and v0 in lok
        ]
        klos = coll.dobj['diagnostic'][key].get('los')
        if klos is None:
            lrays = []
        else:
            lrays = [
                k0 for k0, v0 in coll.dobj['rays'][klos].items()
                if isinstance(v0, str) and v0 in lok
            ]

        if kcryst is None:
            llamb = []
        else:
            llamb = ['lamb', 'lambmin', 'lambmax', 'res']

        data = ds._generic_check._check_var(
            data, 'data',
            types=str,
            allowed=lok + ldiag + lrays + llamb,
            default=defdata,
        )

        if data in ldiag:
            data = coll.dobj['diagnostic'][key][data]
        elif data in lrays:
            data = coll.dobj['rays'][klos][data]
        elif data in llamb:
            data, ref = coll.get_diagnostic_lamb(
                key=key,
                rocking_curve=rocking_curve,
                lamb=data,
            )

    # -------
    # color_dict

    if color_dict is None:
        lc = ['r', 'g', 'b', 'm', 'c', 'y']
        color_dict = {
            'x': lc,
            'y': lc,
        }

    # -------
    # nlos

    nlos = ds._generic_check._check_var(
        nlos, 'nlos',
        types=int,
        default=5,
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
        ref,
        color_dict,
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
    los_res=None,
    # data plot
    data=None,
    cmap=None,
    vmin=None,
    vmax=None,
    # config
    plot_config=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    color_dict=None,
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
        dataref,
        color_dict,
        nlos,
        connect,
    ) = _plot_diagnostic_check(
        coll=coll,
        key=key,
        # figure
        proj=proj,
        data=data,
        # interactivity
        color_dict=color_dict,
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
    camref = coll.dobj['camera'][cam]['dgeom']['ref']
    is2d = coll.dobj['camera'][cam]['dgeom']['type'] == '2d'
    if is2d:
        refx, refy = camref
    else:
        refx = camref[0]

    # -------------------------
    # prepare los interactivity

    los = coll.dobj['diagnostic'][key]['los']

    coll2 = coll.__class__()
    for rr in camref:
        coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

    if los is not None:
        los_x, los_y, los_z = coll.sample_rays(
            key=los,
            res=los_res,
            mode='rel',
        )
        los_r = np.hypot(los_x, los_y)
        reflos = coll.dobj['rays'][los]['ref']
        ref_los = (reflos[1:], reflos[1:])

        coll2.add_ref(key=reflos[0], size=los_x.shape[0])
        coll2.add_data(key='los_x', data=los_x, ref=reflos)
        coll2.add_data(key='los_y', data=los_y, ref=reflos)
        coll2.add_data(key='los_z', data=los_z, ref=reflos)
        coll2.add_data(key='los_r', data=los_r, ref=reflos)

    if data is not None:
        if is2d:
            keyx, keyy = coll.dobj['camera'][cam]['dgeom']['cents']

            datax = coll.ddata[keyx]['data']
            datay = coll.ddata[keyy]['data']

            coll2.add_data(key=keyx, data=datax, ref=refx)
            coll2.add_data(key=keyy, data=datay, ref=refy)
        else:
            keyx = 'i0'
            datax = np.arange(0, coll.dref[refx]['size'])
            coll2.add_data(key=keyx, data=datax, ref=refx)

    # -------------------------
    # prepare data interactivity

    reft = None
    if data is not None:
        if dataref is None:
            dataref = coll.ddata[data]['ref']
        if dataref == camref:
            if isinstance(data, str):
                datamap = coll.ddata[data]['data'].T
            else:
                datamap = data.T
        elif len(dataref) == len(camref) + 1:
            dataref == camref
            datamap = coll.ddata[data]['data'][0, ...].T
            # reft = [rr for rr in dataref if rr not in camref][0]

        else:
            raise NotImplementedError()

        if is2d:
            extent = (
                datax[0] - 0.5*(datax[1] - datax[0]),
                datax[-1] + 0.5*(datax[-1] - datax[-2]),
                datay[0] - 0.5*(datay[1] - datay[0]),
                datay[-1] + 0.5*(datay[-1] - datay[-2]),
            )
        else:
            ylab = f"{coll.ddata[data]['quant']} ({coll.ddata[data]['units']})"

    # -----------------
    # prepare figure

    if dax is None:

        dax = _generic_plot.get_dax_diag(
            proj=proj,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key,
            is2d=is2d,
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
            kax = 'camera'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k0 == cam and is2d and k1 == 'o':
                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1.get('props', {}),
                    )

    # plot data
    kax = 'camera'
    if dax.get(kax) is not None and data is not None:
        ax = dax[kax]['handle']

        if is2d and reft is None:
            im = ax.imshow(
                datamap,
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin='lower',
                interpolation='nearest',
            )
            plt.colorbar(im, ax=ax)

        elif reft is None:
            ax.plot(
                datamap,
                c='k',
                ls='-',
                lw=1.,
                marker='.',
                ms=6,
            )
            ax.set_xlim(-1, datax[-1] + 1)
            ax.set_ylabel(ylab)

    # ----------------
    # define and set dgroup

    if coll2 is not None:
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
                    'ref': [refy],
                    'data': ['index'],
                    'nmax': nlos,
                },
            })

        if reft is not None:
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
                    c=color_dict['x'][ii],
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-cross-{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=ref_los,
                    data=['los_r', 'los_z'],
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
                    c=color_dict['x'][ii],
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-hor-{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=ref_los,
                    data=['los_x', 'los_y'],
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
                    c=color_dict['x'][ii],
                    ls='-',
                    lw=1.,
                )

                # add mobile
                kl0 = f'los-3d-{ii}'
                # coll2.add_mobile(
                # key=kl0,
                # handle=l0,
                # refs=reflos,
                # data=['index', 'index', 'index'],
                # dtype=['xdata', 'ydata', 'zdata'],
                # axes=kax,
                # ind=ii,
                # )

        # camera
        kax = 'camera'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            if is2d:
                for ii in range(nlos):
                    mi, = ax.plot(
                        datax[0:1],
                        datay[0:1],
                        marker='s',
                        ms=6,
                        markeredgecolor=color_dict['x'][ii],
                        markerfacecolor='None',
                    )

                    km = f'm{ii:02.0f}'
                    coll2.add_mobile(
                        key=km,
                        handle=mi,
                        refs=[refx, refy],
                        data=[keyx, keyy],
                        dtype=['xdata', 'ydata'],
                        axes=kax,
                        ind=ii,
                    )

                dax[kax].update(
                    refx=[refx],
                    refy=[refy],
                    datax=keyx,
                    datay=keyy,
                )

            else:
                for ii in range(nlos):
                    lv = ax.axvline(
                        datax[0], c=color_dict['y'][ii], lw=1., ls='-',
                    )
                    kv = f'v{ii:02.0f}'
                    coll2.add_mobile(
                        key=kv,
                        handle=lv,
                        refs=refx,
                        data=keyx,
                        dtype='xdata',
                        axes=kax,
                        ind=ii,
                    )

                dax[kax].update(refx=[refx], datax=keyx)

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax)

    # -------
    # connect

    if coll2.dobj.get('mobile') is not None:
        # add axes
        for ii, kax in enumerate(dax.keys()):
            harmonize = ii == len(dax.keys()) - 1
            coll2.add_axes(key=kax, harmonize=harmonize, **dax[kax])

        # connect
        if connect is True:
            coll2.setup_interactivity(
                kinter='inter0',
                dgroup=dgroup,
                dinc=dinc,
            )
            coll2.disconnect_old()
            coll2.connect()

            coll2.show_commands()
            return coll2
        else:
            return coll2, dgroup
    else:
        return dax
