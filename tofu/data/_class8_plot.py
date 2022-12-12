# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot


# ##################################################################
# ##################################################################
#                           plot check
# ##################################################################


def _plot_diagnostic_check(
    coll=None,
    key=None,
    key_cam=None,
    # figure
    proj=None,
    data=None,
    rocking_curve=None,
    los_res=None,
    # interactivity
    color_dict=None,
    nlos=None,
    connect=None,
):

    # -------
    # key

    # key
    key, key_cam = coll.get_diagnostic_cam(key, key_cam)
    is2d = coll.dobj['diagnostic'][key]['is2d']
    spectro = coll.dobj['diagnostic'][key]['spectro']

    if spectro:
        assert len(key_cam) == 1
        doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam[0]]
        kcryst = doptics['optics'][doptics['ispectro'][0]]
    else:
        kcryst = None

    # -------
    # data

    defdata = 'etendue'
    c0 = data is None and coll.dobj['diagnostic'][key].get(defdata) is not None
    if c0:
        data = defdata

    ddata, dref = coll.get_diagnostic_data(
        key=key,
        key_cam=key_cam,
        data=data,
    )

    ylab = None # f"{ddata[key_cam[0]]['quant']} ({ddata[key_cam[0]]['units']})"

    # -----
    # proj

    proj = _generic_plot._proj(
        proj=proj,
        pall=['cross', 'hor', '3d', 'camera'],
    )

    # ----------
    # los_res

    los_res = ds._generic_check._check_var(
        los_res, 'los_res',
        types=float,
        default=0.05,
        sign='> 0.',
    )

    # -------
    # color_dict

    color_dict = _check_color_dict(color_dict)

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
        key_cam,
        is2d,
        proj,
        ddata,
        dref,
        los_res,
        color_dict,
        nlos,
        ylab,
        connect,
    )


def _check_color_dict(color_dict=None):
    if color_dict is None:
        lc = ['r', 'g', 'b', 'm', 'c', 'y']
        color_dict = {
            'x': lc,
            'y': lc,
        }
    return color_dict


# ##################################################################
# ##################################################################
#                           plot main
# ##################################################################


def _plot_diagnostic(
    coll=None,
    key=None,
    key_cam=None,
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
        key_cam,
        is2d,
        proj,
        ddata,
        dref,
        los_res,
        color_dict,
        nlos,
        ylab,
        connect,
    ) = _plot_diagnostic_check(
        coll=coll,
        key=key,
        # figure
        proj=proj,
        data=data,
        los_res=los_res,
        # interactivity
        color_dict=color_dict,
        nlos=nlos,
        connect=connect,
    )

    # ------------
    # prepare data

    dplot = coll.get_diagnostic_dplot(
        key=key,
        key_cam=key_cam,
        optics=optics,
        elements=elements,
    )

    # -------------------------
    # prepare los interactivity

    # instanciate new Datastock
    coll2 = coll.__class__()

    # ---------------------
    # prepare los and ddata

    # dcamref
    dcamref, drefx, drefy = _prepare_dcamref(
        coll=coll,
        key_cam=key_cam,
        is2d=is2d,
    )

    # los
    dlos, dref_los = _prepare_los(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        key_diag=key,
        key_cam=key_cam,
        los_res=los_res,
    )

    # ddatax, ddatay
    reft, dkeyx, dkeyy, ddatax, ddatay, dextent = _prepare_datarefxy(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        drefx=drefx,
        drefy=drefy,
        ddata=ddata,
        is2d=is2d,
    )

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
            key_cam=key_cam,
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
            kax = k0
            if is2d and k0 in key_cam and dax.get(kax) is not None:
                ax = dax[kax]['handle']
                if k1 == 'o':
                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1.get('props', {}),
                    )

    # plot data
    for k0 in key_cam:
        kax = k0
        if dax.get(kax) is not None and ddata is not None:
            ax = dax[kax]['handle']

            if is2d and reft is None:
                im = ax.imshow(
                    ddata[k0].T,
                    extent=dextent[k0],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin='lower',
                    interpolation='nearest',
                )
                plt.colorbar(im, ax=ax)

            elif reft is None:
                ax.plot(
                    ddata[k0],
                    c='k',
                    ls='-',
                    lw=1.,
                    marker='.',
                    ms=6,
                )
                ax.set_xlim(-1, ddata[k0].size)
                ax.set_ylabel(ylab)

                if vmin is not None:
                    ax.set_ylim(bottom=vmin)
                if vmax is not None:
                    ax.set_ylim(top=vmax)

    # ----------------
    # define and set dgroup

    if coll2 is not None:
        dgroup = {
            f'{k0}_x': {
                'ref': [drefx[k0]],
                'data': ['index'],
                'nmax': nlos,
            }
            for k0 in key_cam
        }

        if is2d:
            dgroup.update({
                'y': {
                    'ref': list(drefy.values()),
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

    for k0 in key_cam:

        if dlos[k0]['rays'] is not None:

            nan = np.full((dlos[k0]['x'].shape[0],), np.nan)

            # cross
            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                _add_camera_los_cross(
                    coll2=coll2,
                    k0=k0,
                    ax=ax,
                    kax=kax,
                    nlos=nlos,
                    dref_los=dref_los,
                    color_dict=color_dict,
                    nan=nan,
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
                    kl0 = f'{k0}-los-hor-{ii}'
                    coll2.add_mobile(
                        key=kl0,
                        handle=l0,
                        refs=dref_los[k0],
                        data=[f'{k0}_los_x', f'{k0}_los_y'],
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
                    kl0 = f'{k0}_los-3d-{ii}'
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
            kax = k0
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                _add_camera_vlines_marker(
                    coll2=coll2,
                    dax=dax,
                    ax=ax,
                    kax=kax,
                    is2d=is2d,
                    k0=k0,
                    nlos=nlos,
                    ddatax=ddatax,
                    ddatay=ddatay,
                    drefx=drefx,
                    drefy=drefy,
                    dkeyx=dkeyx,
                    dkeyy=dkeyy,
                    color_dict=color_dict,
                )

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


# ##################################################################
# ##################################################################
#                       Prepare
# ##################################################################


def _prepare_dcamref(
    coll=None,
    key_cam=None,
    is2d=None,
):
    dcamref = {
        k0: coll.dobj['camera'][k0]['dgeom']['ref']
        for k0 in key_cam
    }
    drefx = {k0: v0[0] for k0, v0 in dcamref.items()}

    if is2d:
        drefy = {k0: v0[1] for k0, v0 in dcamref.items()}
    else:
        drefy = None

    return dcamref, drefx, drefy


def _prepare_los(
    coll=None,
    coll2=None,
    dcamref=None,
    key_diag=None,
    key_cam=None,
    los_res=None,
):

    # create dlos
    dlos = {
        k0: {'rays': coll.dobj['diagnostic'][key_diag]['doptics'][k0]['los']}
        for k0 in key_cam
    }
    dref_los = {}

    # los on cams
    for k0, v0 in dcamref.items():
        for rr in v0:
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

        # los
        if dlos[k0]['rays'] is not None:
            los_x, los_y, los_z = coll.sample_rays(
                key=dlos[k0]['rays'],
                res=los_res,
                mode='rel',
                concatenate=False,
            )
            los_r = np.hypot(los_x, los_y)
            reflos = coll.dobj['rays'][dlos[k0]['rays']]['ref']
            dref_los[k0] = (reflos[1:], reflos[1:])

            if reflos[0] not in coll2.dref.keys():
                coll2.add_ref(key=reflos[0], size=los_x.shape[0])

            coll2.add_data(key=f'{k0}_los_x', data=los_x, ref=reflos)
            coll2.add_data(key=f'{k0}_los_y', data=los_y, ref=reflos)
            coll2.add_data(key=f'{k0}_los_z', data=los_z, ref=reflos)
            coll2.add_data(key=f'{k0}_los_r', data=los_r, ref=reflos)

            # store x, y, z
            dlos[k0]['x'] = los_x
            dlos[k0]['y'] = los_y
            dlos[k0]['z'] = los_z

    return dlos, dref_los


def _prepare_datarefxy(
    coll=None,
    coll2=None,
    dcamref=None,
    drefx=None,
    drefy=None,
    ddata=None,
    is2d=None,
):
    # prepare dict
    dkeyx, ddatax = {}, {}
    if is2d:
        dkeyy, ddatay, dextent = {}, {}, {}
    else:
        dkeyy, ddatay, dextent = None, None, None

    # loop on cams
    for k0, v0 in dcamref.items():

        # datax, datay
        if ddata is not None:
            if is2d:
                dkeyx[k0], dkeyy[k0] = coll.dobj['camera'][k0]['dgeom']['cents']

                ddatax[k0] = coll.ddata[dkeyx[k0]]['data']
                ddatay[k0] = coll.ddata[dkeyy[k0]]['data']

                coll2.add_data(key=dkeyx[k0], data=ddatax[k0], ref=drefx[k0])
                coll2.add_data(key=dkeyy[k0], data=ddatay[k0], ref=drefy[k0])
            else:
                dkeyx[k0] = f'{k0}_i0'
                ddatax[k0] = np.arange(0, coll.dref[drefx[k0]]['size'])
                coll2.add_data(key=dkeyx[k0], data=ddatax[k0], ref=drefx[k0])

            # -------------------------
            # extent

            reft = None
            if is2d:
                if ddatax[k0].size == 1:
                    ddx = coll.ddata[coll.dobj['camera'][k0]['dgeom']['outline'][0]]['data']
                    ddx = np.max(ddx) - np.min(ddx)
                else:
                    ddx = ddatax[k0][1] - ddatax[k0][0]
                if ddatay[k0].size == 1:
                    ddy = coll.ddata[coll.dobj['camera'][k0]['dgeom']['outline'][1]]['data']
                    ddy = np.max(ddy) - np.min(ddy)
                else:
                    ddy = ddatay[k0][1] - ddatay[k0][0]

                dextent[k0] = (
                    ddatax[k0][0] - 0.5*ddx,
                    ddatax[k0][-1] + 0.5*ddx,
                    ddatay[k0][0] - 0.5*ddy,
                    ddatay[k0][-1] + 0.5*ddy,
                )

    return reft, dkeyx, dkeyy, ddatax, ddatay, dextent


# ##################################################################
# ##################################################################
#                       add mobile
# ##################################################################


def _add_camera_los_cross(
    coll2=None,
    k0=None,
    ax=None,
    kax=None,
    nlos=None,
    dref_los=None,
    color_dict=None,
    nan=None,
):

    for ii in range(nlos):
        l0, = ax.plot(
            nan,
            nan,
            c=color_dict['x'][ii],
            ls='-',
            lw=1.,
        )

        # add mobile
        kl0 = f'{k0}-los-cross-{ii}'
        coll2.add_mobile(
            key=kl0,
            handle=l0,
            refs=dref_los[k0],
            data=[f'{k0}_los_r', f'{k0}_los_z'],
            dtype=['xdata', 'ydata'],
            axes=kax,
            ind=ii,
        )


def _add_camera_vlines_marker(
    coll2=None,
    dax=None,
    ax=None,
    kax=None,
    is2d=None,
    k0=None,
    nlos=None,
    ddatax=None,
    ddatay=None,
    drefx=None,
    drefy=None,
    dkeyx=None,
    dkeyy=None,
    color_dict=None,
    suffix=None,
):

    if suffix is None:
        suffix = ''

    if is2d:
        for ii in range(nlos):
            mi, = ax.plot(
                ddatax[k0][0:1],
                ddatay[k0][0:1],
                marker='s',
                ms=6,
                markeredgecolor=color_dict['x'][ii],
                markerfacecolor='None',
            )

            km = f'{k0}_m{ii:02.0f}{suffix}'
            coll2.add_mobile(
                key=km,
                handle=mi,
                refs=[drefx[k0], drefy[k0]],
                data=[dkeyx[k0], dkeyy[k0]],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[drefx[k0]],
            refy=[drefy[k0]],
            datax=[dkeyx[k0]],
            datay=[dkeyy[k0]],
        )

    else:

        for ii in range(nlos):
            lv = ax.axvline(
                ddatax[k0][0], c=color_dict['y'][ii], lw=1., ls='-',
            )
            kv = f'{k0}_v{ii:02.0f}{suffix}'
            coll2.add_mobile(
                key=kv,
                handle=lv,
                refs=drefx[k0],
                data=dkeyx[k0],
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[drefx[k0]], datax=[dkeyx[k0]])
