# -*- coding: utf-8 -*-


# Built-in
import itertools as itt


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot
from . import _class8_plot as _plot


# ###############################################################
# ###############################################################
#                           plot main
# ###############################################################


def _plot(
    coll=None,
    key=None,
    key_cam=None,
    doptics=None,
    is2d=None,
    spectro=None,
    elements=None,
    proj=None,
    los_res=None,
    indch=None,
    indlamb=None,
    # data plot
    dvos=None,
    dplot=None,
    units=None,
    cmap=None,
    dvminmax=None,
    alpha=None,
    # camera
    x0=None,
    x1=None,
    out0=None,
    out1=None,
    extent_cam=None,
    # los, vos
    los_x=None,
    los_y=None,
    los_z=None,
    los_r=None,
    los_xi=None,
    los_yi=None,
    los_zi=None,
    los_ri=None,
    pc0=None,
    pc1=None,
    ph0=None,
    ph1=None,
    pc0i=None,
    pc1i=None,
    ph0i=None,
    ph1i=None,
    # mesh
    p0=None,
    p1=None,
    etendue=None,
    length=None,
    # config
    plot_config=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    color_dict=None,
    **kwdargs,
):

    # ---------------------
    # prepare ph_counts, lamb, cos...

    ddata, extent, dhor = _prepare_ph(
        coll=coll,
        dvos=dvos[key_cam[0]],
        key=key,
        key_cam=key_cam,
        indch=indch,
        indlamb=indlamb,
        spectro=spectro,
        is2d=is2d,
        etendue=etendue,
        length=length,
    )

    # vmin, vmax
    dvminmax = _get_dvminmax(
        ddata=ddata,
        dvminmax=dvminmax,
    )

    # -----------------
    # prepare figure

    if dax is None:

        dax = _get_dax(
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key,
            is2d=is2d,
            key_cam=key_cam,
            indch=indch,
            ddata=ddata,
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot diag elements

    for k0, v0 in dplot.items():

        for k1, v1 in v0.items():

            # cross
            typ = 'cross'
            lkax = [kk for kk, vv in dax.items() if vv['type'] == typ]
            for kax in lkax:
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

    # ------------------
    # plot ddata

    for k0, v0 in ddata.items():

        kax = [kk for kk, vv in dax.items() if vv['k0'] == k0][0]
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            typ = dax[kax]['type']
            if typ == 'camera':
                im = ax.imshow(
                    v0['data'].T,
                    extent=extent_cam,
                    origin='lower',
                    aspect='equal',
                    interpolation='nearest',
                    vmin=dvminmax[k0][0],
                    vmax=dvminmax[k0][1],
                )

                # camera outline
                ax.plot(
                    np.r_[out0, out0[0]],
                    np.r_[out1, out1[0]],
                    c='k',
                    ls='-',
                    lw=1,
                )

                # pixel
                ax.plot(
                    [x0[indch[0]]],
                    [x1[indch[1]]],
                    c='k',
                    ls='None',
                    marker='s',
                    ms=6,
                )

            elif typ == 'cross':
                im = ax.imshow(
                    v0['data'].T,
                    extent=extent,
                    origin='lower',
                    aspect='equal',
                    interpolation='nearest',
                    vmin=dvminmax[k0][0],
                    vmax=dvminmax[k0][1],
                )

                # ------------------
                # plot los / vos

                _add_camera_los_cross(
                    ax=ax,
                    is2d=is2d,
                    los_r=los_r,
                    los_z=los_z,
                    pc0=pc0,
                    pc1=pc1,
                    alpha=alpha,
                    color_dict=color_dict,
                )

                _add_camera_los_cross(
                    ax=ax,
                    is2d=is2d,
                    los_r=los_ri,
                    los_z=los_zi,
                    pc0=pc0i,
                    pc1=pc1i,
                    alpha=alpha,
                    color_dict=color_dict,
                )

                # mesh outline
                if dax[kax]['jj'] == 1:
                    ax.plot(
                        np.r_[p0, p0[0]],
                        np.r_[p1, p1[0]],
                        c='k',
                        ls='-',
                        lw=1,
                    )

            # labels
            if dax[kax]['ii'] > 0:
                ax.tick_params(labelleft=False)
            if dax[kax]['jj'] == 1:
                ax.tick_params(labelbottom=False)

            plt.colorbar(im, ax=ax)

    # hor
    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_hor(
            ax=ax,
            is2d=is2d,
            los_x=los_xi,
            los_y=los_yi,
            ph0=ph0,
            ph1=ph1,
            alpha=alpha,
            color_dict=color_dict,
        )

        ax.plot(
            dhor['envelopx'],
            dhor['envelopy'],
            ls='-',
            c='k',
            marker='.',
            ms=4,
            lw=1.
        )

        ax.fill(
            dhor['minx'],
            dhor['miny'],
            ls='None',
            lw=1.,
            fc='r',
            alpha=0.5,
        )

        ax.plot(
            dhor['mainx'],
            dhor['mainy'],
            ls='-',
            c='r',
            marker='.',
            ms=4,
            lw=1.
        )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        typ = 'cross'
        lkax = [kk for kk, vv in dax.items() if vv['type'] == typ]
        for kax in lkax:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj='cross', dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

    # -------
    # connect

    return dax


# ##################################################################
# ##################################################################
#                       Prepare sang
# ##################################################################


def _prepare_ph(
    coll=None,
    dvos=None,
    key=None,
    key_cam=None,
    indch=None,
    indlamb=None,
    spectro=None,
    is2d=None,
    etendue=None,
    length=None,
):

    # -----------------
    # check up
    # -----------------

    kph = 'ph2'

    if indlamb is None:
        indlamb = int(dvos['lamb']['data'].size / 2)

    # -----------------
    # get mesh sampling
    # -----------------

    dsamp = coll.get_sample_mesh(
        key=dvos['keym'],
        res=dvos['res_RZ'],
        mode='abs',
        grid=False,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=None,
        Dx1=None,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    # ------------------------------
    # dhor
    # ------------------------------

    x0u = dsamp['x0']['data']
    x1u = dsamp['x1']['data']

    iru = np.unique(dvos['indr_cross']['data'])
    phi_env = np.full((iru.size, 2), np.nan)
    phi_mean = np.full((iru.size,), np.nan)
    phi_minmax = np.full((iru.size, 2), np.nan)
    for ii, i0 in enumerate(iru):
        ind = dvos['indr_cross']['data'] == i0
        phi_env[ii, 0] = np.nanmin(dvos['phi_min']['data'][..., ind])
        phi_env[ii, 1] = np.nanmax(dvos['phi_max']['data'][..., ind])

        phi_mean[ii] = np.nanmean(dvos['phi_mean']['data'][indch[0], indch[1], ind])
        phi_minmax[ii, 0] = np.nanmin(dvos['phi_min']['data'][indch[0], indch[1], ind])
        phi_minmax[ii, 1] = np.nanmax(dvos['phi_max']['data'][indch[0], indch[1], ind])

    dhor = {
        'envelopx': np.r_[
            x0u[iru] * np.cos(phi_env[:, 0]),
            x0u[iru[::-1]] * np.cos(phi_env[::-1, 1]),
            x0u[iru[0]] * np.cos(phi_env[0, 0]),
        ],
        'envelopy': np.r_[
            x0u[iru] * np.sin(phi_env[:, 0]),
            x0u[iru[::-1]] * np.sin(phi_env[::-1, 1]),
            x0u[iru[0]] * np.sin(phi_env[0, 0]),
        ],
        'mainx': x0u[iru] * np.cos(phi_mean),
        'mainy': x0u[iru] * np.sin(phi_mean),
        'minx': np.r_[
            x0u[iru] * np.cos(phi_minmax[:, 0]),
            x0u[iru[::-1]] * np.cos(phi_minmax[::-1, 1]),
            x0u[iru[0]] * np.cos(phi_minmax[0, 0]),
        ],
        'miny': np.r_[
            x0u[iru] * np.sin(phi_minmax[:, 0]),
            x0u[iru[::-1]] * np.sin(phi_minmax[::-1, 1]),
            x0u[iru[0]] * np.sin(phi_minmax[0, 0]),
        ],
    }

    # ------------------------------
    # prepare image in cross-section
    # ------------------------------

    # -----------------
    # get shapes and initialize

    n0, n1 = x0u.size, x1u.size
    shape = (n0, n1)

    nc_tot = np.full(shape, np.nan)
    nc_toti = np.full(shape, np.nan)

    dV = np.full(shape, np.nan)

    ph_tot = np.full(shape, np.nan)
    ph_toti = np.full(shape, np.nan)
    ph_tot_lamb = np.full(shape, np.nan)
    ph_toti_lamb = np.full(shape, np.nan)

    lamb = np.full(shape, np.nan)
    lambi = np.full(shape, np.nan)
    lamb0 = np.full(shape, np.nan)
    lambi0 = np.full(shape, np.nan)
    dlamb = np.full(shape, np.nan)
    dlambi = np.full(shape, np.nan)
    cos = np.full(shape, np.nan)
    cosi = np.full(shape, np.nan)

    shape_cam = dvos[kph]['data'].shape[:-2]
    nc_cam = np.full(shape_cam, np.nan)
    ph_cam = np.full(shape_cam, np.nan)
    cos_cam = np.full(shape_cam, np.nan)
    lamb_cam = np.full(shape_cam, np.nan)
    dlamb_cam = np.full(shape_cam, np.nan)

    indr = dvos['indr_cross']['data']
    indz = dvos['indz_cross']['data']
    ilambr = dvos['ilambr']['data']
    npts = dvos[kph]['data'].shape[-2]

    # -----------------
    # multiply by dlamb

    indch0 = indch
    ph0 = dvos[kph]['data'] * np.mean(np.diff(dvos['lamb']['data']))
    if is2d:
        nc = dvos['ncounts']['data'].reshape((-1, npts))
        ph = ph0.reshape(tuple([-1] + list(ph0.shape[-2:])))
        if kph == 'ph2':
            ilambr = dvos['ilambr']['data'].reshape(tuple([-1] + list(ph0.shape[-2:])))
        coss = dvos['cos']['data'].reshape((-1, npts))
        lambc = dvos['lamb']['data'][None, None, None, :]
        indch = indch[0] * shape_cam[1] + indch[1]

    else:
        nc = dvos['ncounts']['data']
        ph = ph0
        if kph == 'ph2':
            ilambr = dvos['ilambr']['data']
        coss = dvos['cos']['data']
        lambc = dvos['lamb']['data'][None, None, :]

    nci = nc[indch, :]
    phi = ph[indch, :, :]
    cossi = coss[indch, :]

    # -----------------
    # lambmax, lambmin

    shapel = ph0.shape[:-1]
    lambmin = np.full(shapel, np.nan)
    lambmax = np.full(shapel, np.nan)
    if kph == 'ph2':
        for ind in itt.product(*[range(ss) for ss in shapel]):
            sli = tuple(list(ind) + [slice(None)])
            iok = ph0[sli] > 0.
            if np.any(iok):
                if kph == 'ph2':
                    iok = dvos['ilambr']['data'][sli][iok]
                lambmin[ind] = np.min(dvos['lamb']['data'][iok])
                lambmax[ind] = np.max(dvos['lamb']['data'][iok])

    lambmini = lambmin.reshape((-1, npts))[indch]
    lambmaxi = lambmax.reshape((-1, npts))[indch]

    # -----------------
    # photon counts

    nc_tot[indr, indz] = np.sum(nc, axis=0)
    nc_toti[indr, indz] = np.sum(nci, axis=0)

    # -----------------
    # update indr, indz

    iok = nc_tot[indr, indz] > 0.
    iokn = np.nonzero(iok)[0]
    ir, iz = indr[iok], indz[iok]

    ioki = nc_toti[indr, indz] > 0.
    iri, izi = indr[ioki], indz[ioki]

    nc_tot[nc_tot == 0] = np.nan
    nc_toti[nc_toti == 0] = np.nan

    # -----------------
    # dV

    dV[ir, iz] =  dvos['dV']['data'][iok]

    # -----------------
    # cos

    cos[ir, iz] = np.nansum(nc[:, iok] * coss[:, iok], axis=0) / nc_tot[ir, iz]
    cosi[iri, izi] = cossi[ioki]

    # -----------------
    # ph_tot

    if np.any(np.isnan(ph)):
        msg = "ph_count should not contain nans! (nansum copies)"
        raise Exception(msg)

    ph_tot[ir, iz] = np.sum(np.sum(ph[:, iok, :], axis=0), axis=-1)
    ph_toti[iri, izi] = np.sum(phi[ioki, :], axis=-1)

    # ------------------
    # average wavelength

    for ii, (irii, izii) in enumerate(zip(ir, iz)):
        if kph == 'ph2':
            sli = (None, ilambr[:, iokn[ii], :])
        else:
            sli = (None, slice(None))
        lamb[irii, izii] = (
            np.sum(dvos['lamb']['data'][sli] * ph[:, iokn[ii], :])
            / ph_tot[irii, izii]
        )

    if kph == 'ph2':
        ll = dvos['lamb']['data'][ilambr[indch, ioki, :]]
    else:
        ll = dvos['lamb']['data'][None, :]

    lambi[iri, izi] = (
        np.sum(phi[ioki, :] * ll, axis=-1)
        / ph_toti[iri, izi]
    )

    if kph == 'ph':
        lamb0[ir, iz] = np.sum(
            np.sum(ph[:, iok, :] * dvos['lamb']['data'][None, None, :], axis=0),
            axis=-1,
        ) / ph_tot[ir, iz]

        lambi0[iri, izi] = (
            np.sum(phi[ioki, :] * dvos['lamb']['data'][None, :], axis=-1)
            / ph_toti[iri, izi]
        )

        assert np.allclose(lamb0, lamb, equal_nan=True), ('lamb\n', lamb, '\n', lamb0)
        assert np.allclose(lambi0, lambi, equal_nan=True), ('lambi\n', lambi, '\n', lambi0)

    # -----------------
    # delta wavelength

    dlamb[ir, iz] = (
        np.nanmax(lambmax.reshape((-1, npts))[:, iok], axis=0)
        - np.nanmin(lambmin.reshape((-1, npts))[:, iok], axis=0)
    )
    dlambi[iri, izi] = lambmaxi[ioki] - lambmini[ioki]

    # -----------------
    # adjust

    ph_tot[ph_tot == 0.] = np.nan
    ph_toti[ph_toti == 0.] = np.nan

    # -----------------------
    # prepare image on camera
    # -----------------------

    nc_cam[...] = np.nansum(dvos['ncounts']['data'], axis=-1)
    iok = nc_cam > 0.
    nc_cam[~iok] = np.nan

    ph_cam[iok] = np.nansum(np.nansum(ph0, axis=-1), axis=-1)[iok]

    if kph == 'ph2':
        lamb_aa = dvos['lamb']['data'][dvos['ilambr']['data']]
    else:
        lamb_aa = lambc

    lamb_cam[iok] = (
        np.nansum(np.nansum(ph0 * lamb_aa, axis=-1), axis=-1)[iok]
        / ph_cam[iok]
    )

    cos_cam[iok] = (
        np.nansum(dvos['ncounts']['data'] * dvos['cos']['data'], axis=-1)[iok]
        / nc_cam[iok]
    )

    iok = np.any(np.isfinite(lambmax), axis=-1)
    dlamb_cam[iok] = (
        np.nanmax(lambmax[iok, :], axis=-1)
        - np.nanmin(lambmin[iok, :], axis=-1)
    )

    # ----------------------
    # prepare per wavelength
    # ----------------------

    if kph == 'ph2':
        indlamb2 = dvos['ilambr']['data'] == indlamb
        ii2 = np.any(indlamb2, axis=-2)
        print(ii2.shape, ii2.sum(), ph.shape)
        print()
        print(np.sum(np.any(ii2, axis=-1)), np.sum(~np.any(ii2, axis=-1)))
        ii2[~np.any(ii2, axis=-1), 0] = True
        print(ii2.sum())
        assert np.all(np.sum(indlamb2, axis=-2) <= 1)
        assert ii2.sum() == ph.shape[0], (ii2.shape, ii2.size, ii2.sum())
        print(indlamb2.shape, indlamb2.sum(), ii2.shape)
        print(ph0.shape, ph.shape, phi.shape)
        ph_cam_lamb = np.nansum(ph0, axis=-1, where=indlamb2)
        ph_cam_lamb2 = np.nansum(ph0, axis=-2, where=indlamb2)[ii2]
        print(ph_cam_lamb.shape)
        print(ph_cam_lamb2.shape)
        indlamb2 = ilambr == indlamb
        assert np.all(np.sum(indlamb2, axis=-1) <= 1)
        print(indlamb2.shape)
        ph_tot_lamb[indr, indz] = np.nansum(ph, axis=0, where=indlamb2)
        ph_toti_lamb[indr, indz] = np.nansum(phi, axis=0, where=indlamb2)
        print(ph_cam_lamb.shape, ph_tot_lamb.shape, ph_toti_lamb.shape)

    else:
        ph_cam_lamb = np.nansum(ph0[..., indlamb], axis=-1)
        ph_tot_lamb[indr, indz] = np.nansum(ph[..., indlamb], axis=0)
        ph_toti_lamb[indr, indz] = np.nansum(phi[..., indlamb], axis=0)

    ph_cam_lamb[ph_cam_lamb == 0] = np.nan

    # ----------------------
    # delta_lamb for etendue
    # ----------------------

    delta_lamb = coll.get_diagnostic_data(key, data='dlamb')[0][key_cam[0]]

    # -------------------
    # extent
    # -------------------

    x0 = dsamp['x0']['data']
    dx0 = x0[1] - x0[0]
    x1 = dsamp['x1']['data']
    dx1 = x1[1] - x1[0]

    extent = (
        x0[0] - 0.5*dx0,
        x0[-1] + 0.5*dx0,
        x1[0] - 0.5*dx1,
        x1[-1] + 0.5*dx1,
    )

    return (
        {
            'dV': {'data': dV},
            'nc_tot': {'data': nc_tot},
            'nc_toti': {'data': nc_toti},
            'ph_tot': {'data': ph_tot},
            'ph_toti': {'data': ph_toti},
            'cos': {'data': cos},
            'lamb': {'data': lamb},
            'dlamb': {'data': dlamb},
            'cosi': {'data': cosi},
            'lambi': {'data': lambi},
            'dlambi': {'data': dlambi},
            'nc_cam': {'data': nc_cam},
            'ph_cam': {'data': ph_cam},
            'cos_cam': {'data': cos_cam},
            'lamb_cam': {'data': lamb_cam},
            'dlamb_cam': {'data': dlamb_cam},
            'ph_cam_lamb': {'data': ph_cam_lamb},
            'ph_tot_lamb': {'data': ph_tot_lamb},
            'ph_toti_lamb': {'data': ph_toti_lamb},
            'etendue*length*dlamb': {'data': etendue * length * delta_lamb},
            # 'etendue*length': {'data': etendue * length},
        },
        extent,
        dhor,
    )


# ###############################################################
# ###############################################################
#                       vminmax
# ###############################################################


def _get_dvminmax(ddata=None, dvminmax=None):

    if dvminmax is None:
        dvminmax = {}

    # safety check
    lkout = [k0 for k0 in dvminmax.keys() if k0 not in ddata.keys()]
    if len(lkout) > 0:
        lstr_in = [f"\t{k0}" for k0 in ddata.keys()]
        lstr_out = [f"\t{k0}" for k0 in lkout]
        msg = (
            "Arg dvminmax must be a dict with the following keys only:\n"
            + '\n'.join(lstr_in)
            + "\nHence, the following ones are not valid:\n"
            + '\n'.join(lstr_out)
        )
        raise Exception(msg)

    # fill with default
    for k0, v0 in ddata.items():
        if dvminmax.get(k0) is None:
            dvminmax[k0] = [None, None]

        if dvminmax[k0][0] is None:
            dvminmax[k0][0] = np.nanmin(v0['data'])
        if dvminmax[k0][1] is None:
            dvminmax[k0][1] = np.nanmax(v0['data'])

    return dvminmax


# ##################################################################
# ##################################################################
#                       add mobile
# ##################################################################


def _add_camera_los_cross(
    ax=None,
    is2d=None,
    color_dict=None,
    los_r=None,
    los_z=None,
    pc0=None,
    pc1=None,
    alpha=None,
):

    # ------
    # los

    if los_r is not None:
        l0, = ax.plot(
            los_r,
            los_z,
            c='k',
            ls='-',
            lw=1.,
        )

    # ------
    # vos
    if pc0 is not None:
        if pc0.ndim == 2:
            for ii in range(pc0.shape[1]):
                l0, = ax.fill(
                    pc0[:, ii],
                    pc1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif pc0.ndim == 3:
            for ii in range(pc0.shape[1]):
                for jj in range(pc0.shape[2]):
                    l0, = ax.fill(
                        pc0[:, ii, jj],
                        pc1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                pc0,
                pc1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_los_hor(
    ax=None,
    is2d=None,
    los_x=None,
    los_y=None,
    ph0=None,
    ph1=None,
    alpha=None,
    color_dict=None,
):

    # ------
    # los

    if los_x is not None:
        l0, = ax.plot(
            los_x,
            los_y,
            c=color_dict['x'][0],
            ls='-',
            lw=1.,
        )

    # ------
    # vos

    if ph0 is not None:

        if ph0.ndim == 2:
            for ii in range(ph0.shape[1]):
                l0, = ax.fill(
                    ph0[:, ii],
                    ph1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif ph0.ndim == 3:
            for ii in range(ph0.shape[1]):
                for jj in range(ph0.shape[2]):
                    l0, = ax.fill(
                        ph0[:, ii, jj],
                        ph1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                ph0,
                ph1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_data(
    coll=None,
    ax=None,
    sang_integ=None,
    etendue=None,
    length=None,
    indch=None,
    color_dict=None,
    # vmin, vmax
    vmin_cam=None,
    vmax_cam=None,
    # 2d only
    ax_diff=None,
    ax_etend=None,
    is2d=None,
    x0=None,
    x1=None,
    extent_cam=None,
):


    if is2d:

        # sang
        mi = ax.imshow(
            sang_integ.T,
            origin='lower',
            extent=extent_cam,
            interpolation='nearest',
            vmin=vmin_cam,
            vmax=vmax_cam,
        )

        # etendue
        etendle = etendue * length
        mi = ax_etend.imshow(
            etendle.T,
            origin='lower',
            extent=extent_cam,
            interpolation='nearest',
            vmin=vmin_cam,
            vmax=vmax_cam,
        )
        plt.colorbar(mi, ax=[ax, ax_etend])

        # diff
        diff = (sang_integ - etendle).T
        dmax = np.abs(max(np.nanmin(diff), np.nanmax(diff)))
        imd = ax_diff.imshow(
            (sang_integ - etendle).T,
            origin='lower',
            extent=extent_cam,
            interpolation='nearest',
            cmap=plt.cm.seismic,
            vmin=-dmax,
            vmax=dmax,
        )

        plt.colorbar(imd, ax=ax_diff)

        # marker
        for aa in [ax, ax_etend, ax_diff]:
            aa.plot(
                [x0[indch[0]]],
                [x1[indch[1]]],
                c='k',
                marker='s',
                ms=6,
                ls='None',
                lw=1.,
            )

    else:

        nlos = sang_integ.shape[0]
        ind = np.arange(0, nlos)

        # vos
        ax.plot(
            ind,
            sang_integ,
            c='b',
            marker='.',
            ls='-',
            lw=1.,
            label="sang * dV (sr.m3)",
        )

        # los
        ax.plot(
            ind,
            etendue * length,
            c='k',
            marker='.',
            ls='-',
            lw=1.,
            label="etendue * length (sr.m3)",
        )

        # indch
        ax.axvline(
            indch,
            c='k',
            lw=1.,
            ls='--',
        )

        ax.set_ylim(vmin_cam, vmax_cam)
        plt.legend()


# ################################################################
# ################################################################
#                   figure
# ################################################################


def _get_dax(
    proj=None,
    dmargin=None,
    fs=None,
    tit=None,
    wintit=None,
    is2d=None,
    key_cam=None,
    indch=None,
    ddata=None,
):

    # ------------
    # check inputs

    # fs
    if fs is None:
        fs = (16, 9)

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.90,
            'left': 0.04, 'right': 0.98,
            'wspace': 0.20, 'hspace': 0.40,
            # 'width_ratios': [0.6, 0.4],
            # 'height_ratios': [0.4, 0.6],
        },
    )

    # wintit
    wintit = ds._generic_check._check_var(
        wintit, 'wintit',
        types=str,
        default=_generic_plot._WINDEF,
    )

    # tit
    if tit is None:
        tit = f"{key_cam} - {indch}\nVOS"

    # -------------
    # Prepare

    lcross = [
        'dV',
        'ph_tot', 'ph_toti',
        'nc_tot', 'nc_toti',
        'cos', 'cosi',
        'lamb', 'lambi',
        'dlamb', 'dlambi',
        'ph_tot_lamb', 'ph_toti_lamb',
    ]
    lcross = [k0 for k0 in lcross if k0 in ddata.keys()]
    lcross.insert(1, None)

    lcam = [
        'ph_cam', 'nc_cam', 'cos_cam', 'lamb_cam', 'dlamb_cam',
        'ph_cam_lamb',
    ]
    lcam = [k0 for k0 in lcam if k0 in ddata.keys()]

    ncross = len(lcross)
    ncam = len(lcam)

    # -------------
    # Create figure

    fig = plt.figure(figsize=fs)
    fig.canvas.manager.set_window_title(wintit)
    fig.suptitle(tit, size=12, fontweight='bold')

    if is2d is False:
        raise NotImplementedError()

    gs = gridspec.GridSpec(ncols=ncam + 1, nrows=3, **dmargin)

    dax = {}
    shxca, shyca = None, None
    shxcr, shycr = None, None
    for ii in range(0, ncam + 1):
        for jj in range(3):

            # prepare share options
            if ii == 0 and jj == 2:
                shx = None
                shy = None
            elif jj == 0:
                shx = shxca
                shy = shyca
            else:
                shx = shxcr
                shy = shycr

            # create axes
            ax = fig.add_subplot(
                gs[jj, ii],
                aspect='equal',
                sharex=shx,
                sharey=shy,
            )

            # update share options
            if jj == 0 and shxca is None:
                shxca = ax
                shyca = ax

            if jj > 0 and shxcr is None:
                shxcr = ax
                shycr = ax

            # labels
            if jj == 0:
                if ii == 0:
                    name = 'etendue*length*dlamb'
                else:
                    name = lcam[ii -1].split('_')[0]
                    if name == 'ph':
                        name = 'photon flux'
                    if '_lamb' in lcam[ii - 1]:
                        name = f'{name}\n' + r'$\lambda$ = '

                ax.set_title(name, size=12, fontweight='bold')
                ax.set_xlabel('x0 (m)', size=12)
                if ii == 0:
                    ax.set_ylabel('x1 (m)', size=12)

            elif jj == 2:
                if ii == 0:
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                else:
                    ax.set_xlabel('R (m)')
                    if ii == 1:
                        ax.set_ylabel('Z (m)')
            elif jj == 1 and ii == 0:
                name = 'dV'
                ax.set_title(name, size=12, fontweight='bold')
                ax.set_ylabel('Z (m)')

            # update dict
            if ii == 0:
                if jj == 0:
                    kax = f'{name}_cam'
                    typ = 'camera'
                    k0 = name
                elif jj == 1:
                    kax = f'{name}_cross'
                    typ = 'cross'
                    k0 = 'dV'
                elif jj == 2:
                    kax = 'hor'
                    typ = 'hor'
                    k0 = None
            elif jj == 0:
                kax = f'{name}_cam'
                typ = 'camera'
                k0 = lcam[ii - 1]
            else:
                kax = f'{name}_cross'
                typ = 'cross'
                if jj == 2:
                    kax = f'{kax}_pix'

                k0 = lcross[ii*2 + (jj - 1)]

            dax[kax] = {
                'handle': ax, 'type': typ, 'k0': k0, 'ii': ii, 'jj': jj,
            }

    return dax