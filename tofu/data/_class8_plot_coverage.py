# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:12:28 2024

@author: dvezinet
"""


# Built-in
import itertools as itt


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _class8_vos_utilities as _vos_utils


# ################################################################
# ################################################################
#                           Main
# ################################################################


def main(
    coll=None,
    key=None,
    # observation directions
    observation_directions=None,
    # mesh sampling
    key_mesh=None,
    res_RZ=None,
    nan0=None,
    # plotting options
    config=None,
    dcolor=None,
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    cmap=None,
    vmin=None,
    vmax=None,
):

    # -------------
    # check inputs
    # -------------

    key, is_vos, observation_directions, keym, res_RZ, nan0, dcolor = _check(
        coll=coll,
        key=key,
        observation_directions=observation_directions,
        key_mesh=key_mesh,
        res_RZ=res_RZ,
        nan0=nan0,
        dcolor=dcolor,
    )

    # -----------
    # compute
    # -----------

    dpoly, ndet, extent = _compute(
        coll=coll,
        key=key,
        is_vos=is_vos,
        keym=keym,
        res_RZ=res_RZ,
        dcolor=dcolor,
    )

    # ------------
    # nan0

    if nan0 is True:
        ndet[ndet == 0] = np.nan

    # -----------
    # plot
    # -----------

    dax = _plot(
        coll=coll,
        key=key,
        # data
        is_vos=is_vos,
        ndet=ndet,
        extent=extent,
        dpoly=dpoly,
        # plotting options
        config=config,
        dax=dax,
        fs=fs,
        dmargin=dmargin,
        tit=tit,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    return dpoly, ndet, dax


# ################################################################
# ################################################################
#                           Check
# ################################################################


def _check(
    coll=None,
    key=None,
    observation_directions=None,
    key_mesh=None,
    res_RZ=None,
    nan0=None,
    dcolor=None,
):

    # -----------
    # key
    # -----------

    lok_vos = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if all([
            v1.get('dvos', {}).get('keym') is not None
            for v1 in v0['doptics'].values()
        ])
    ]
    lok = [
        k0 for k0 in coll.dobj.get('diagnostic', {}).keys()
        if k0 not in lok_vos
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok + lok_vos,
    )

    lcam = coll.dobj['diagnostic'][key]['camera']
    is_vos = key in lok_vos

    # is 3d ?
    doptics = coll.dobj['diagnostic'][key]['doptics']
    # is_3d = doptics[lcam[0]]['dvos'].get('indr_3d') is not None

    # ------------------------
    # observation_directions
    # ------------------------

    obsdef = False
    observation_directions = ds._generic_check._check_var(
        observation_directions, 'observation_directions',
        types=bool,
        default=obsdef,
    )

    # -------------
    # key mesh
    # -------------

    doptics = coll.dobj['diagnostic'][key]['doptics']
    if is_vos:

        # key_mesh unicity
        lmesh = set([v0['dvos']['keym'] for v0 in doptics.values()])

        # res unicity
        lres_RZ = set([tuple(v0['dvos']['res_RZ']) for v0 in doptics.values()])

        if len(lmesh) != 1 or len(lres_RZ) != 1:
            msg = (
                "Non-unique mesh or res_RZ for vos of diag '{key}':\n"

            )
            raise Exception(msg)

        keym = list(lmesh)[0]
        res_RZ = list(list(lres_RZ)[0])

    else:
        pass

    # -----------
    # nan0 => set 0 to nan
    # -----------

    nan0 = ds._generic_check._check_var(
        nan0, 'nan0',
        types=bool,
        default=True,
    )

    # -----------
    # dcolor
    # -----------

    if dcolor is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        dcolor = {k0: colors[ii%len(colors)] for ii, k0 in enumerate(lcam)}

    if mcolors.is_color_like(dcolor):
        dcolor = {k0: dcolor for k0 in lcam}

    c0 = (
        isinstance(dcolor, dict)
        and all([
            mcolors.is_color_like(dcolor.get(k0))
            for k0 in lcam
        ])
    )
    if not c0:
        lstr = [f"\t- {k0}: rgb or rgba color" for k0 in lcam]
        msg = (
            f"Arg dcolor must be a dict of (camera: color) for diag '{key}':\n"
            + "\n".join(lstr)
            + f"\nProvided:\n{dcolor}\n"
        )
        raise Exception(msg)

    # ----------
    # add alpha

    for k0, v0 in dcolor.items():
        if isinstance(v0, tuple) and len(v0) == 4:
            alpha = v0[-1]
        else:
            alpha = 0.5
        dcolor[k0] = mcolors.to_rgba(v0, alpha=alpha)

    return key, is_vos, observation_directions, keym, res_RZ, nan0, dcolor


# ################################################################
# ################################################################
#                           Compute
# ################################################################


def _compute(
    coll=None,
    key=None,
    is_vos=None,
    keym=None,
    res_RZ=None,
    dcolor=None,
):

    # ---------------
    # prepare
    # ---------------

    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ---------------
    # mesh sampling
    # ---------------

    # ----------------
    # initial sampling

    dsamp = coll.get_sample_mesh(
        key=keym,
        res=res_RZ,
        mode='abs',
        grid=False,
        store=False,
    )

    R = dsamp['x0']['data']
    Z = dsamp['x1']['data']

    # --------------------
    # derive shape, extent

    # shape
    shape = (R.size, Z.size)

    # extent
    extent = (
        R[0] - 0.5*(R[1] - R[0]),
        R[-1] + 0.5*(R[-1] - R[-2]),
        Z[0] - 0.5*(Z[1] - Z[0]),
        Z[-1] + 0.5*(Z[-1] - Z[-2]),
    )

    # ---------------
    # ndet for vos
    # ---------------

    ndet = np.zeros(shape, dtype=float)

    if is_vos:
        for kcam, v0 in doptics.items():

            kindr, kindz = v0['dvos']['ind_cross']

            shape_poly = coll.ddata[kindr]['data'].shape
            lind = [range(ss) for ss in shape_poly[:-1]]

            sli0 = np.asarray(list(shape_poly[:-1]) + [slice(None)])
            sli = np.asarray(list(shape_poly[:-1]) + [slice(None)])

            for ii, ij in enumerate(itt.product(*lind)):

                # get rid of -1
                sli0[:-1] = ij
                sli[-1] = coll.ddata[kindr]['data'][tuple(sli0)] >= 0

                # get indices to incremente
                sli[:-1] = ij
                ind = (
                    coll.ddata[kindr]['data'][tuple(sli)],
                    coll.ddata[kindz]['data'][tuple(sli)],
                )

                ndet[ind] += 1

    # -----------------
    # ndet for non-vos
    # -----------------

    else:
        raise NotImplementedError()

    # ---------------
    # dpoly
    # ---------------

    dpoly = {}
    for ii, (kcam, v0) in enumerate(doptics.items()):

        # concatenate all vos
        pr, pz = _vos_utils._get_overall_polygons(
            coll=coll,
            doptics=doptics,
            key_cam=kcam,
            poly='pcross',
            convexHull=False,
        )

        # store
        dpoly[kcam] = {
            'pr': pr,
            'pz': pz,
            'color': dcolor[kcam],
        }

    return dpoly, ndet, extent


# ################################################################
# ################################################################
#                           plot
# ################################################################


def _plot(
    coll=None,
    key=None,
    # data
    ndet=None,
    extent=None,
    dpoly=None,
    # plotting options
    config=None,
    is_vos=None,
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    cmap=None,
    vmin=None,
    vmax=None,
):

    # ----------------
    # check input
    # ----------------

    # tit
    if tit is not False:
        titdef = f"geometrical coverage of diag '{key}'"
        tit = ds._generic_check._check_var(
            tit, 'tit',
            types=str,
            default=titdef,
        )

    if cmap is None:
        cmap = plt.cm.viridis   # Greys

    if vmin is None:
        vmin = 0

    if vmax is None:
        vmax = np.nanmax(ndet)

    # ----------------
    # prepare data
    # ----------------

    # directions of observation



    # ----------------
    # prepare figure
    # ----------------

    if dax is None:
        if fs is None:
            fs = (14, 7)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.20, 'wspace': 0.40,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=16, nrows=1, **dmargin)

        # ax0 = spans
        ax0 = fig.add_subplot(gs[0, :6], aspect='equal', adjustable='box')
        ax0.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax0.set_xlabel('R (m)', size=12, fontweight='bold')
        ax0.set_title("spans", size=14, fontweight='bold')

        # ax1 = nb of detectors
        ax1 = fig.add_subplot(gs[0, 9:-1], sharex=ax0, sharey=ax0)
        ax1.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax1.set_xlabel('R (m)', size=12, fontweight='bold')
        ax1.set_title("nb. of detectors", size=14, fontweight='bold')

        # colorbar
        cax = fig.add_subplot(gs[0, -1], frameon=False)

        dax = {
            'span': {'handle': ax0, 'type': 'span'},
            'ndet': {'handle': ax1, 'type': 'ndet'},
            'cax': {'handle': cax, 'type': 'span_colorbar'},
        }

    # --------------------
    # check / format dax

    dax = ds._generic_check._check_dax(dax)
    fig = list(dax.values())[0]['handle'].figure

    # ---------------
    # plot ndet
    # ---------------

    ktype = 'ndet'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.imshow(
            ndet.T,
            extent=extent,
            origin='lower',
            interpolation='bilinear',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # colorbar
        ktype2 = 'span_colorbar'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # cax.set_ylabel("nb. of detectors", size=14, fontweight='bold')

    # ---------------
    # plot spans
    # ---------------

    ktype = 'span'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        for k0, v0 in dpoly.items():
            ax.fill(
                v0['pr'],
                v0['pz'],
                fc=v0['color'],
                label=k0,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # --------------
    # add config
    # --------------

    if config is not None:
        for ktype in ['span', 'ndet']:
            lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
            for ax in lax:
                config.plot(lax=ax, proj='cross', dLeg=False)

    # --------------
    # add tit
    # --------------

    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    return dax