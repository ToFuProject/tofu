# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot


# ################################################################
# ################################################################
#                           plot check
# ################################################################


def _plot_diagnostic_check(
    coll=None,
    key=None,
    key_cam=None,
    # parameters
    vmin=None,
    vmax=None,
    alpha=None,
    dx0=None,
    dx1=None,
    # figure
    plot_colorbar=None,
    proj=None,
    data=None,
    units=None,
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
    # spectro = coll.dobj['diagnostic'][key]['spectro']

    # if spectro:
    #     assert len(key_cam) == 1
    #     doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam[0]]
    #     kcryst = doptics['optics'][doptics['ispectro'][0]]
    # else:
    #    pass
    #    kcryst = None

    # -------
    # data

    defdata = 'etendue'
    c0 = (
        data is None
        and all([
            v0.get(defdata) is not None
            for v0 in coll.dobj['diagnostic'][key]['doptics'].values()
        ])
    )
    if c0:
        data = defdata

    ddata, dref, units, static, daxis = coll.get_diagnostic_data(
        key=key,
        key_cam=key_cam,
        data=data,
        units=units,
    )

    refz = None
    if static is False:
        lnr = [len(v0) for v0 in dref.values()]
        refz = [v0[daxis[k0]] for k0, v0 in dref.items()]

        if len(set(lnr)) != 1:
            msg = f"data '{data}' shall have the same ndims for all cameras!"
            raise Exception(msg)

        if len(set(refz)) != 1:
            msg = f"data '{data}' shall have the same extra ref for all cameras"
            raise Exception(msg)

        refz = refz[0]

    ylab = f"{data} ({units})"

    # ----------
    # vmin, vmax

    if vmin is None and len(ddata) > 0:
        vmin = np.nanmin([np.nanmin(v0) for v0 in ddata.values()])

    if vmax is None and len(ddata) > 0:
        vmax = np.nanmax([np.nanmax(v0) for v0 in ddata.values()])

    # -----
    # alpha

    alpha = ds._generic_check._check_var(
        alpha, 'alpha',
        types=float,
        default=0.2,
        sign='> 0.',
    )

    # -----
    # proj

    pall = ['cross', 'hor', '3d', 'camera', 'traces']
    proj = _generic_plot._proj(
        proj=proj,
        pall=pall,
    )

    if static is True:
        proj = [pp for pp in proj if pp != 'traces']

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

    # ---------------
    # dx0, dx1

    # dx0
    dx0 = float(ds._generic_check._check_var(
        dx0, 'dx0',
        types=(int, float),
        default=0.,
    ))

    # dx1
    dx1 = float(ds._generic_check._check_var(
        dx1, 'dx1',
        types=(int, float),
        default=0.,
    ))

    # -------
    # plot_colorbar

    plot_colorbar = ds._generic_check._check_var(
        plot_colorbar, 'plot_colorbar',
        types=bool,
        default=True,
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
        static,
        daxis,
        refz,
        vmin,
        vmax,
        alpha,
        units,
        los_res,
        color_dict,
        nlos,
        dx0,
        dx1,
        ylab,
        plot_colorbar,
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
    units=None,
    cmap=None,
    vmin=None,
    vmax=None,
    keyZ=None,
    alpha=None,
    dx0=None,
    dx1=None,
    # config
    plot_config=None,
    plot_colorbar=None,
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
        static,
        daxis,
        refz,
        vmin,
        vmax,
        alpha,
        units,
        los_res,
        color_dict,
        nlos,
        dx0,
        dx1,
        ylab,
        plot_colorbar,
        connect,
    ) = _plot_diagnostic_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # parameters
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        dx0=dx0,
        dx1=dx1,
        # figure
        plot_colorbar=plot_colorbar,
        proj=proj,
        data=data,
        units=units,
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
        dx0=dx0,
        dx1=dx1,
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
    dlos_n, dref_los = _prepare_los(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        key_diag=key,
        key_cam=key_cam,
        los_res=los_res,
    )

    # vos
    dvos_n, dref_vos = _prepare_vos(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        key_diag=key,
        key_cam=key_cam,
        los_res=los_res,
    )

    # ddatax, ddatay
    _, dkeyx, dkeyy, ddatax, ddatay, dextent = _prepare_datarefxy(
        coll=coll,
        coll2=coll2,
        dcamref=dcamref,
        drefx=drefx,
        drefy=drefy,
        ddata=ddata,
        static=static,
        dx0=dx0,
        dx1=dx1,
        is2d=is2d,
    )

    # ---------------------
    # prepare non-static

    if static is False and len(ddata) > 0:

        k0 = key_cam[0]
        keyz = coll.get_ref_vector(ref=refz)[3]
        nz = ddata[k0].shape[daxis[k0]]

        keyz, zstr, dataz, dz2, labz = ds._plot_as_array._get_str_datadlab(
            keyX=keyz, nx=nz, islogX=False, coll=coll,
        )

        npts = 0
        for k0 in key_cam:
            npts = max(npts, ddata[k0].size)

        bck = 'envelop' if npts > 10000 else 'lines'

        coll2.add_ref(key=refz, size=nz)
        coll2.add_data(key=keyz, data=dataz, ref=refz)

        # add camera data
        for k0 in key_cam:
            coll2.add_data(
                key=f'{k0}_{data}',
                data=ddata[k0].T,
                ref=dref[k0][::-1],
                units=units,
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

    _plot_diag_geom(
        dax=dax,
        key_cam=key_cam,
        dplot=dplot,
        is2d=is2d,
    )

    # plot data
    if static is True:

        for k0 in key_cam:
            kax = f'{k0}_sig'
            if dax.get(kax) is not None:
                if ddata is None or ddata.get(k0) is None:
                    continue

                ax = dax[kax]['handle']

                if is2d:
                    im = ax.imshow(
                        ddata[k0].T,
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                    )
                    if plot_colorbar is True:
                        plt.colorbar(im, ax=ax)

                else:
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
                    ax.set_title(k0, size=12, fontweight='bold')

                    if vmin is not None:
                        ax.set_ylim(bottom=vmin)
                    if vmax is not None:
                        ax.set_ylim(top=vmax)

    else:
        # plot traces envelop
        for k0 in key_cam:

            kax = f'{k0}_trace'
            if dax.get(kax) is None or ddata.get(k0) is None:
                continue

            ax = dax[kax]['handle']

            if bck == 'lines':
                shap = list(ddata[k0].shape)
                shap[daxis[k0]] = 1
                bckl = np.concatenate(
                    (ddata[k0], np.full(shap, np.nan)),
                    axis=daxis[k0],
                )
                bckl = np.swapaxes(bckl, daxis[k0], -1).ravel()

                ax.plot(
                    np.tile(np.r_[dataz, np.nan], int(np.prod(shap))),
                    bckl,
                    c=(0.8, 0.8, 0.8),
                    ls='-',
                    lw=1.,
                    marker='None',
                )

            else:
                tax = tuple([
                    ii for ii in range(ddata[k0].ndim) if ii != daxis[k0]
                ])

                ax.fill_between(
                    dataz,
                    np.nanmin(ddata[k0], axis=tax),
                    np.nanmax(ddata[k0], axis=tax),
                    facecolor=(0.8, 0.8, 0.8, 0.8),
                    edgecolor='None',
                )

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

        if static is False is not None:
            dgroup.update({
                't': {
                    'ref': [refz],
                    'data': ['index'],
                    'nmax': 1,
                },
            })

    # ------------------
    # plot mobile parts

    for k0 in key_cam:

        if dlos_n[k0] is not None:
            nan_los = np.full((dlos_n[k0],), np.nan)
        else:
            nan_los = None

        if dvos_n is None:
            nan_vos = None
        else:
            nan_vos = np.full((dvos_n[k0],), np.nan)

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
                dref_vos=dref_vos,
                color_dict=color_dict,
                nan_los=nan_los,
                nan_vos=nan_vos,
                alpha=alpha,
            )

        # hor
        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            _add_camera_los_hor(
                coll2=coll2,
                k0=k0,
                ax=ax,
                kax=kax,
                nlos=nlos,
                dref_los=dref_los,
                dref_vos=dref_vos,
                color_dict=color_dict,
                nan_los=nan_los,
                nan_vos=nan_vos,
                alpha=alpha,
            )

        # 3d
        kax = '3d'
        if dax.get(kax) is not None and nan_los is not None:
            ax = dax[kax]['handle']

            for ii in range(nlos):
                l0, = ax.plot(
                    nan_los,
                    nan_los,
                    nan_los,
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
        kax = f'{k0}_sig'
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

        # vline on traces
        if static is False:

            kax = f'{k0}_trace'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                lv = ax.axvline(
                    dataz[0],
                    c='k',
                    lw=1.,
                    ls='-',
                )

                kv = f'{k0}_zline'
                coll2.add_mobile(
                    key=kv,
                    handle=lv,
                    refs=(refz,),
                    data=[keyz],
                    dtype=['xdata'],
                    axes=kax,
                    ind=0,
                )

                dax[kax].update(refx=[refz], datax=[keyz])

    # -------------------
    # data if not static

    if static is False:

        for k0 in key_cam:

            # line/im plot on data
            kax = f'{k0}_sig'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                tax = tuple([
                    ii for ii in range(ddata[k0].ndim) if ii != daxis[k0]
                ])

                if is2d:
                    im = ax.imshow(
                        np.take(ddata[k0], 0, axis=daxis[k0]).T,
                        extent=dextent[k0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        origin='lower',
                        interpolation='nearest',
                    )
                    if plot_colorbar is True:
                        plt.colorbar(im, ax=ax)

                    km = f'{k0}_{data}'
                    coll2.add_mobile(
                        key=km,
                        handle=im,
                        refs=((refz,),),
                        data=[f'{k0}_{data}'],
                        dtype='data',
                        axes=kax,
                        ind=0,
                    )

                else:

                    l0, = ax.plot(
                        np.take(ddata[k0], 0, axis=daxis[k0]),
                        c='k',
                        ls='-',
                        lw=1.,
                        marker='.',
                        ms=6,
                    )
                    ax.set_xlim(-1, ddata[k0].size / nz)
                    ax.set_ylabel(ylab)
                    ax.set_title(k0, size=12, fontweight='bold')

                    if vmin is not None:
                        ax.set_ylim(bottom=vmin)
                    if vmax is not None:
                        ax.set_ylim(top=vmax)

                    km = f'{k0}_{data}'
                    coll2.add_mobile(
                        key=km,
                        handle=l0,
                        refs=((refz,),),
                        data=[f'{k0}_{data}'],
                        dtype='ydata',
                        axes=kax,
                        ind=0,
                    )

                    if vmin is not None:
                        ax.set_ylim(bottom=vmin)
                    if vmax is not None:
                        ax.set_ylim(top=vmax)

            # line plot on traces
            kax = f'{k0}_trace'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                sli = tuple([
                    slice(None) if ii == daxis[k0] else 0
                    for ii in range(ddata[k0].ndim)
                ])

                for ii in range(nlos):
                    l0, = ax.plot(
                        dataz,
                        ddata[k0][sli],
                        c=color_dict['x'][ii],
                        lw=1.,
                        ls='-',
                    )

                    refi = dref_los[k0][0] if is2d else dref_los[k0][0]
                    kv = f'{k0}_trace{ii}'
                    coll2.add_mobile(
                        key=kv,
                        handle=l0,
                        refs=(refi,),
                        data=[f'{k0}_{data}'],
                        dtype=['ydata'],
                        axes=kax,
                        ind=ii,
                    )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

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


# ################################################################
# ################################################################
#                       Prepare
# ################################################################


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

    # -----------------
    # create dlos, dvos

    # dlos
    dlos_n = {
        k0: coll.dobj['diagnostic'][key_diag]['doptics'][k0]['los']
        for k0 in key_cam
    }
    dref_los = {}

    # -------------
    # los

    # los on cams
    for k0, v0 in dcamref.items():
        for rr in v0:
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])

        # los
        if dlos_n[k0] is not None:
            los_x, los_y, los_z = coll.sample_rays(
                key=dlos_n[k0],
                res=los_res,
                mode='rel',
                concatenate=False,
            )
            los_r = np.hypot(los_x, los_y)
            reflos = coll.dobj['rays'][dlos_n[k0]]['ref']
            dref_los[k0] = (reflos[1:], reflos[1:])

            if reflos[0] not in coll2.dref.keys():
                coll2.add_ref(key=reflos[0], size=los_x.shape[0])

            coll2.add_data(key=f'{k0}_los_x', data=los_x, ref=reflos)
            coll2.add_data(key=f'{k0}_los_y', data=los_y, ref=reflos)
            coll2.add_data(key=f'{k0}_los_z', data=los_z, ref=reflos)
            coll2.add_data(key=f'{k0}_los_r', data=los_r, ref=reflos)

            # store x, y, z
            dlos_n[k0] = los_x.shape[0]

    return dlos_n, dref_los


def _prepare_vos(
    coll=None,
    coll2=None,
    dcamref=None,
    key_diag=None,
    key_cam=None,
    los_res=None,
):

    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    if doptics[key_cam[0]].get('dvos') is None:
        return None, None

    # -----------------
    # create dlos, dvos

    # dvos
    dvos_n = {
        k0: {'pc': doptics[k0]['dvos']['pcross']}
        for k0 in key_cam
    }
    dref_vos = {}

    # ----
    # vos

    # vos on cams
    for k0, v0 in dcamref.items():

        krxy = f'{k0}_xy2'
        coll2.add_ref(key=krxy, size=2)

        # vos
        if dvos_n[k0]['pc'] is not None:

            pc0 = coll.ddata[dvos_n[k0]['pc'][0]]['data']
            pc1 = coll.ddata[dvos_n[k0]['pc'][1]]['data']
            pcref = coll.ddata[dvos_n[k0]['pc'][0]]['ref']

            ph0, ph1, phref = None, None, None
            if doptics[k0]['dvos'].get('phor') is not None:
                ph0 = coll.ddata[doptics[k0]['dvos']['phor'][0]]['data']
                ph1 = coll.ddata[doptics[k0]['dvos']['phor'][1]]['data']
                phref = coll.ddata[doptics[k0]['dvos']['phor'][0]]['ref']

            if pcref[0] not in coll2.dref.keys():
                coll2.add_ref(key=pcref[0], size=pc0.shape[0])

            dref_vos[k0] = (pcref[1:],)

            ref = tuple(list(pcref[::-1]) + [krxy])
            pcxy = np.array([pc0, pc1]).T
            coll2.add_data(key=f'{k0}_vos_cross', data=pcxy, ref=ref)
            if ph0 is not None:
                ref = tuple(list(phref[::-1]) + [krxy])
                phxy = np.array([ph0, ph1]).T
                coll2.add_data(key=f'{k0}_vos_hor', data=phxy, ref=ref)

            # store
            dvos_n[k0] = pc0.shape[0]

    return dvos_n, dref_vos


def _prepare_datarefxy(
    coll=None,
    coll2=None,
    dcamref=None,
    drefx=None,
    drefy=None,
    ddata=None,
    static=None,
    dx0=None,
    dx1=None,
    is2d=None,
):

    # ---------------
    # dx0, dx1

    # dx0
    dx0 = float(ds._generic_check._check_var(
        dx0, 'dx0',
        types=(int, float),
        default=0.,
    ))

    # dx1
    dx1 = float(ds._generic_check._check_var(
        dx1, 'dx1',
        types=(int, float),
        default=0.,
    ))

    # -----------------
    # prepare

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

                ddatax[k0] = coll.ddata[dkeyx[k0]]['data'] + dx0
                ddatay[k0] = coll.ddata[dkeyy[k0]]['data'] + dx1

                coll2.add_data(key=dkeyx[k0], data=ddatax[k0], ref=drefx[k0])
                coll2.add_data(key=dkeyy[k0], data=ddatay[k0], ref=drefy[k0])
            else:
                dkeyx[k0] = f'{k0}_i0'
                ddatax[k0] = np.arange(0, coll.dref[drefx[k0]]['size']) + dx0
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


# ################################################################
# ################################################################
#                       add diag geom
# ################################################################


def _plot_diag_geom(
    dax=None,
    key_cam=None,
    dplot=None,
    is2d=None,
):

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
            kax = f"{k0}_sig"
            if is2d and k0 in key_cam and dax.get(kax) is not None:
                ax = dax[kax]['handle']
                if k1 == 'o':
                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1.get('props', {}),
                    )


# ################################################################
# ################################################################
#                       add mobile
# ################################################################


def _add_camera_los_cross(
    coll2=None,
    k0=None,
    ax=None,
    kax=None,
    nlos=None,
    dref_los=None,
    dref_vos=None,
    color_dict=None,
    nan_los=None,
    nan_vos=None,
    alpha=None,
):

    for ii in range(nlos):

        # ------
        # los

        if nan_los is not None:
            l0, = ax.plot(
                nan_los,
                nan_los,
                c=color_dict['x'][ii],
                ls='-',
                lw=1.,
            )

            # add mobile
            kl0 = f'{k0}_los_cross{ii}'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=dref_los[k0],
                data=[f'{k0}_los_r', f'{k0}_los_z'],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        # ------
        # vos

        if nan_vos is not None:
            l0, = ax.fill(
                nan_vos,
                nan_vos,
                fc=color_dict['x'][ii],
                alpha=alpha,
                ls='None',
                lw=0.,
            )

            # add mobile
            kl0 = f'{k0}_vos_cross{ii}'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=dref_vos[k0],
                data=[f'{k0}_vos_cross'],
                dtype=['xy'],
                axes=kax,
                ind=ii,
            )


def _add_camera_los_hor(
    coll2=None,
    k0=None,
    ax=None,
    kax=None,
    nlos=None,
    dref_los=None,
    dref_vos=None,
    color_dict=None,
    nan_los=None,
    nan_vos=None,
    alpha=None,
):

    for ii in range(nlos):

        # ------
        # los

        if nan_los is not None:
            l0, = ax.plot(
                nan_los,
                nan_los,
                c=color_dict['x'][ii],
                ls='-',
                lw=1.,
            )

            # add mobile
            kl0 = f'{k0}_los_hor{ii}'
            coll2.add_mobile(
                key=kl0,
                handle=l0,
                refs=dref_los[k0],
                data=[f'{k0}_los_x', f'{k0}_los_y'],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        # ------
        # vos

        if f'{k0}_vos_hor' in coll2.ddata.keys():

            if nan_vos is not None:
                l0, = ax.fill(
                    nan_vos,
                    nan_vos,
                    fc=color_dict['x'][ii],
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

                # add mobile
                kl0 = f'{k0}_vos_hor{ii}'
                coll2.add_mobile(
                    key=kl0,
                    handle=l0,
                    refs=dref_vos[k0],
                    data=[f'{k0}_vos_hor'],
                    dtype=['xy'],
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