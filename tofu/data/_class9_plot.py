# -*- coding: utf-8 -*-


# Built-in
import datetime as dtm


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import datastock as ds
import bsplines2d as bs2


# specific
from . import _generic_check


# #############################################################################
# #############################################################################
#                           plot matrix
# #############################################################################


def _plot_geometry_matrix_check(
    coll=None,
    key=None,
    indbf=None,
    indchan=None,
    indt=None,
    plot_mesh=None,
    cmap=None,
    vmin=None,
    vmax=None,
    aspect=None,
    dcolorbar=None,
    dleg=None,
    dax=None,
):

    # key
    lk = list(coll.dobj['geom matrix'].keys())
    key = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lk,
    )

    keybs = coll.dobj['geom matrix'][key]['bsplines']
    refbs = coll.dobj['bsplines'][keybs]['ref']
    keym = coll.dobj['bsplines'][keybs]['mesh']

    key_diag = coll.dobj['geom matrix'][key]['diagnostic']
    key_cam = coll.dobj['geom matrix'][key]['camera']
    key_data = coll.dobj['geom matrix'][key]['data']
    shape = coll.dobj['geom matrix'][key]['shape']
    axis_chan = coll.dobj['geom matrix'][key]['axis_chan']
    axis_bs = coll.dobj['geom matrix'][key]['axis_bs']
    axis_other = coll.dobj['geom matrix'][key]['axis_other']

    # indbfi
    indbf = ds._generic_check._check_var(
        indbf, 'indbf',
        types=int,
        default=0,
    )

    # indchan
    indchan = ds._generic_check._check_var(
        indchan, 'indchan',
        types=int,
        default=0,
    )

    # ind
    indt = None
    if axis_other is not None:
        indt = 0

    # plot_mesh
    wm = coll._which_mesh
    plot_mesh = ds._generic_check._check_var(
        plot_mesh, 'plot_mesh',
        # default=coll.dobj[wm][keym]['type'] != 'polar',
        default=False,
        types=bool,
    )

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # vmin, vmax
    if vmax is None:
        lmax = [np.nanmax(coll.ddata[kk]['data']) for kk in key_data]
        vmax = max(lmax)
    if vmin is None:
        vmin = 0

    # aspect
    aspect = ds._generic_check._check_var(
        aspect, 'aspect',
        default='auto',
        types=str,
        allowed=['auto', 'equal'],
    )

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = ds._generic_check._check_var(
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
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    return (
        key,
        keybs, keym,
        key_diag, key_cam, key_data,
        shape, axis_chan, axis_bs, axis_other,
        indbf, indchan, indt,
        plot_mesh,
        cmap, vmin, vmax,
        aspect, dcolorbar, dleg,
    )


def _plot_geometry_matrix_prepare(
    coll=None,
    key=None,
    key_data=None,
    keybs=None,
    keym=None,
    axis_chan=None,
    axis_bs=None,
    axis_other=None,
    indbf=None,
    indchan=None,
    indt=None,
    res=None,
):

    # --------
    # prepare

    # res
    deg = coll.dobj['bsplines'][keybs]['deg']
    km = coll.dobj['bsplines'][keybs]['mesh']
    nd = coll.dobj['mesh'][km]['nd']
    mtype = coll.dobj['mesh'][km]['type']

    # if polar => submesh
    km0 = km
    mtype0 = mtype
    submesh = coll.dobj[coll._which_mesh][km0]['submesh']
    if submesh is not None:
        km = submesh
        mtype = coll.dobj[coll._which_mesh][km]['type']

    # R, Z
    kR, kZ = coll.dobj['mesh'][km]['knots']
    Rk = coll.ddata[kR]['data']
    Zk = coll.ddata[kZ]['data']

    # get dR, dZ
    dR, dZ, _, _ = bs2._class02_plot_as_profile2d._plot_bsplines_get_dx01(
        coll=coll, km=km,
    )
    if res is None:
        if mtype == 'rect':
            res_coef = 0.1
        else:
            res_coef = 0.25
        res = res_coef*dR

    # crop
    nchan, nbs = coll.dobj['geom matrix'][key]['shape'][-2:]
    crop = coll.dobj['geom matrix'][key]['crop']

    # --------
    # indices

    # indchan => indchan_bf
    if nd == '1d':
        ich_bf_tup = coll.select_ind(
            key=keybs,
            returnas=int,
            crop=crop,
        )
        nbf = ich_bf_tup.size

        indbf_bool = coll.select_ind(
            key=keybs,
            ind=ich_bf_tup[indbf],
            returnas=bool,
            crop=crop,
        )

    else:
        if mtype0 == 'rect':
            ich_bf_tup = coll.select_ind(
                key=keybs,
                returnas='tuple-flat',
                crop=crop,
            )
            nbf = ich_bf_tup[0].size

            # indbf_bool
            indbf_bool = coll.select_ind(
                key=keybs,
                ind=(ich_bf_tup[0][indbf], ich_bf_tup[1][indbf]),
                returnas=bool,
                crop=crop,
            )

        elif mtype0 == 'tri':
            ich_bf_tup = coll.select_ind(
                key=keybs,
                returnas=int,
                crop=crop,
            )
            nbf = ich_bf_tup.size

            indbf_bool = coll.select_ind(
                key=keybs,
                ind=ich_bf_tup[indbf],
                returnas=bool,
                crop=crop,
            )

    assert nbs == nbf

    # -------------
    # mesh sampling

    # mesh sampling
    dout = coll.get_sample_mesh(
        key=km, res=res, mode='abs', grid=True, imshow=True,
    )
    R = dout['x0']['data']
    Z = dout['x1']['data']

    # -------------
    # interpolation

    # bspline details
    bsplinebase = coll.interpolate(
        keys=None,
        ref_key=keybs,
        x0=R,
        x1=Z,
        grid=False,
        submesh=True,
        crop=crop,
        nan0=True,
        details=True,
        return_params=False,
        store=False,
    )[f'{keybs}_details']['data']

    gmat0 = coll.ddata[key_data[0]]['data']
    if axis_other is not None:
        diffdim = len(bsplinebase.shape) - 1 - len(gmat0.shape)
        axo = axis_other if axis_other < axis_bs else axis_other + diffdim
        bsplinebase = bsplinebase.take(indt, axis=axo)

    # bsplinetot
    coefstot = np.nansum(
        [
            np.nansum(coll.ddata[kk]['data'], axis=axis_chan)
            for kk in key_data
        ],
        axis=0,
    )

    if axis_other is not None:
        coefstot = coefstot.take(indt, axis=axis_other)

    bsplinetot = np.full(bsplinebase.shape[:-1], np.nan)
    for ii in range(bsplinebase.shape[0]):
        for jj in range(bsplinebase.shape[1]):
            bsplinetot[ii, jj] = np.sum(bsplinebase[ii, jj, :] * coefstot)

    # bsplinedet
    gmat0 = gmat0.take(indchan, axis=axis_chan)
    if axis_other is not None:
        axo = axis_other - 1 if axis_other > axis_chan else axis_other
        gmat0 = gmat0.take(indt, axis=axo)

    bsplinedet = np.full(bsplinebase.shape[:-1], np.nan)
    for ii in range(bsplinebase.shape[0]):
        for jj in range(bsplinebase.shape[1]):
            bsplinedet[ii, jj] = np.sum(bsplinebase[ii, jj, :] * gmat0)

    # --------
    # LOS

    # los
    ptslos = None
    indlosok = None
    # ptslos = cam._get_plotL(return_pts=True, proj='cross', Lplot='tot')
    # indsep = np.nonzero(np.isnan(ptslos[0, :]))[0]
    # ptslos = np.split(ptslos, indsep, axis=1)
    # indlosok = np.nonzero(coll.ddata[key]['data'][:, indbf] > 0)[0]

    # ---------------
    # extent / interp

    # extent and interp
    extent = (
        np.nanmin(Rk) - 0.*dR, np.nanmax(Rk) + 0.*dR,
        np.nanmin(Zk) - 0.*dZ, np.nanmax(Zk) + 0.*dZ,
    )

    if deg == 0:
        interp = 'nearest'
    elif deg == 1:
        interp = 'bilinear'
    elif deg >= 2:
        interp = 'bicubic'

    # -------------
    # coll2

    coll2 = coll.__class__()

    gmat, ref, dind = coll.get_geometry_matrix_concatenated(key=key)
    npix = gmat.shape[ref.index(None)]

    for ii, rr in enumerate(ref):
        if rr is None:
            coll2.add_ref(key='npix', size=npix)
            ref[ii] = 'npix'
        else:
            coll2.add_ref(key=rr, size=coll.dref[rr]['size'])
    ref = tuple(ref)

    coll2.add_data(
        key=key,
        data=gmat,
        ref=ref,
        units=coll.ddata[key_data[0]]['units'],
    )

    return (
        bsplinetot, bsplinedet, extent, interp,
        ptslos, indlosok, indbf_bool,
        coll2, ref,
    )


def plot_geometry_matrix(
    # resources
    coll=None,
    # parameters
    key=None,
    indbf=None,
    indchan=None,
    indt=None,
    # options
    plot_mesh=None,
    plot_config=None,
    # plotting
    vmin=None,
    vmax=None,
    res=None,
    cmap=None,
    aspect=None,
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
):

    # --------------
    # check input

    (
        key,
        keybs, keym,
        key_diag, key_cam, key_data,
        shape, axis_chan, axis_bs, axis_other,
        indbf, indchan, indt,
        plot_mesh,
        cmap, vmin, vmax,
        aspect, dcolorbar, dleg,
    ) = _plot_geometry_matrix_check(
        coll=coll,
        key=key,
        indbf=indbf,
        indchan=indchan,
        indt=indt,
        plot_mesh=plot_mesh,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        dcolorbar=dcolorbar,
        dleg=dleg,
        dax=dax,
    )

    # --------------
    #  Prepare data

    (
        bsplinetot, bspline1,
        extent, interp,
        ptslos, indlosoki,
        ich_bf,
        coll1, refs,
    ) = _plot_geometry_matrix_prepare(
        coll=coll,
        key=key,
        key_data=key_data,
        keybs=keybs,
        keym=keym,
        axis_chan=axis_chan,
        axis_bs=axis_bs,
        axis_other=axis_other,
        indbf=indbf,
        indchan=indchan,
        indt=indt,
        res=res,
    )
    nchan, nbs = shape[-2:]

    # --------------
    # plot - prepare

    if dax is None:
        dax = _create_dax(
            fs=fs,
            dmargin=dmargin,
            indt=indt,
            key=key,
            vmin=vmin,
            vmax=vmax,
        )

    dax = ds._generic_check._check_dax(dax=dax, main='matrix')

    # --------------
    # plot mesh

    if plot_mesh is True:
        _ = coll.plot_mesh(
            key=keym, dax=dax, crop=True, dleg=False,
        )

    # --------------
    # plot matrix

    if indt is None:
        ind = [indbf, indchan]
        keyX = refs[1]
        keyY = refs[0]
        keyZ = None
    else:
        ind = [indbf, indchan, indt]
        keyX = refs[2]
        keyY = refs[1]
        keyZ = refs[0]

    coll2, dgroup = coll1.plot_as_array(
        key=key,
        keyX=keyX,
        keyY=keyY,
        keyZ=keyZ,
        dax=dax,
        ind=ind,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        connect=False,
    )

    kax = 'cross1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

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
        ax = dax[kax]['handle']

        # coll.plot_bsplines(
            # key=keybs,
            # indbs=ich_bf,
            # indt=indt,
            # knots=False,
            # cents=False,
            # plot_mesh=False,
            # dax={'cross': dax[kax]},
            # dleg=False,
        # )

        if ptslos is not None:
            for ii in indlosok:
                ax.plot(
                    ptslos[ii][0, :],
                    ptslos[ii][1, :],
                    ls='-',
                    lw=1,
                    color='k',
                )

    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

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

    # --------------
    # dleg

    # if dleg is not False:
        # dax['cross'].legend(**dleg)

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        for kax in ['cross1', 'cross2', 'crosstot']:
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']
                plot_config.plot(lax=ax, proj='cross', dLeg=False)

    # -------
    # connect

    coll2.setup_interactivity(kinter='inter0', dgroup=dgroup)
    coll2.disconnect_old()
    coll2.connect()
    coll2.show_commands()

    return coll2


# ############################################################
# ############################################################
#           Create axes
# ############################################################


def _create_dax(
    fs=None,
    dmargin=None,
    indt=None,
    key=None,
    vmin=None,
    vmax=None,
):
    if fs is None:
        fs = (16, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.98,
            'bottom': 0.05, 'top': 0.95,
            'hspace': 0.20, 'wspace': 0.25,
        }

    fig = plt.figure(figsize=fs)
    ncols = 4 + (indt is not None)
    gs = gridspec.GridSpec(ncols=ncols, nrows=2, **dmargin)

    # ax01 = matrix
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_ylabel(f'channels')
    ax01.set_xlabel(f'basis functions')
    ax01.set_title(key, size=14)
    ax01.tick_params(
        axis="x",
        bottom=False, top=True,
        labelbottom=False, labeltop=True,
    )
    ax01.xaxis.set_label_position('top')

    # ax00 = horizontal
    ax00 = fig.add_subplot(gs[0, 0], sharex=ax01)
    ax00.set_xlabel(f'basis functions')
    ax00.set_ylabel(f'data')
    ax00.set_ylim(vmin, vmax)

    # ax02 = vertical
    ax02 = fig.add_subplot(gs[0, 2], sharey=ax01)
    ax02.set_xlabel(f'channels')
    ax02.set_ylabel(f'data')
    ax02.tick_params(
        axis="x",
        bottom=False, top=True,
        labelbottom=False, labeltop=True,
    )
    ax02.xaxis.set_label_position('top')
    ax02.tick_params(
        axis="y",
        left=False, right=True,
        labelleft=False, labelright=True,
    )
    ax02.yaxis.set_label_position('right')
    ax02.set_xlim(vmin, vmax)

    if indt is not None:
        axt = fig.add_subplot(gs[0, 3], sharey=ax00)
        axt.set_xlabel(f'time')
        axt.set_ylabel(f'data')


    # ax10 = cross1
    ax10 = fig.add_subplot(gs[1, 0], aspect='equal')
    ax10.set_xlabel(f'R (m)')
    ax10.set_ylabel(f'Z (m)')

    # ax11 = crosstot
    ax11 = fig.add_subplot(
        gs[1, 1],
        aspect='equal',
        sharex=ax10,
        sharey=ax10,
    )
    ax11.set_xlabel(f'R (m)')
    ax11.set_ylabel(f'Z (m)')

    # ax12 = cross2
    ax12 = fig.add_subplot(
        gs[1, 2],
        aspect='equal',
        sharex=ax10,
        sharey=ax10,
    )
    ax12.set_xlabel(f'R (m)')
    ax12.set_ylabel(f'Z (m)')

    # text
    axt0 = fig.add_subplot(gs[0, -1], frameon=False)
    axt0.set_xticks([])
    axt0.set_yticks([])
    axt1 = fig.add_subplot(gs[1, -1], frameon=False)
    axt1.set_xticks([])
    axt1.set_yticks([])

    # define dax
    dax = {
        # matrix
        'matrix': {'handle': ax01, 'inverty': True},
        'vertical': {'handle': ax02},
        'horizontal': {'handle': ax00},
        # cross-section
        'cross1': {'handle': ax10, 'type': 'cross'},
        'cross2': {'handle': ax12, 'type': 'cross'},
        'crosstot': {'handle': ax11, 'type': 'cross'},
        # text
        'text0': {'handle': axt0, 'type': 'text'},
        'text1': {'handle': axt1, 'type': 'text'},
    }
    if indt is not None:
        dax['tracesZ'] = {'handle': axt}

    return dax
