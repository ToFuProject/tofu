# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:12:28 2024

@author: dvezinet
"""

import itertools as itt


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _class8_plot_coverage_broadband as _broadband
from . import _class8_plot_coverage_spectro as _spectro


# ################################################################
# ################################################################
#                           Main
# ################################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    # what to plot
    plot_cross=None,
    plot_hor=None,
    plot_rank=None,
    # observation directions
    observation_directions=None,
    # mesh sampling
    key_mesh=None,
    res_RZ=None,
    nan0=None,
    # plotting options
    marker=None,
    markersize=None,
    config=None,
    dcolor=None,
    dax=None,
    fs=None,
    dmargin=None,
    tit=None,
    cmap=None,
    dvminmax=None,
):

    # -------------
    # check inputs
    # -------------

    (
        key, is_vos, lcam, dvos, spectro,
        plot_cross, plot_hor, plot_rank,
        observation_directions, keym,
        res_RZ, res_phi,
        nan0, dcolor,
    ) = _check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        dvos=None,
        # plot
        plot_cross=plot_cross,
        plot_hor=plot_hor,
        plot_rank=plot_rank,
        observation_directions=observation_directions,
        key_mesh=key_mesh,
        res_RZ=res_RZ,
        nan0=nan0,
        dcolor=dcolor,
    )

    # ---------------
    # mesh func
    # ---------------

    (
        func_RZphi_from_ind,
        func_ind_from_domain,
    ) = coll.get_sample_mesh_3d_func(
        key=keym,
        res_RZ=res_RZ,
        mode='abs',
        res_phi=res_phi,
    )

    if dax is None:
        dax = {}

    # -----------------
    # select func
    # -----------------

    if spectro is True:

        # safety check
        wdiag = coll._which_diagnostic
        doptics = coll.dobj[wdiag][key]['doptics']
        compact_lamb_cross = any([
            v0['dvos'].get('indlamb_cross') is not None
            for k0, v0 in doptics.items()
        ])
        compact_lamb_hor = any([
            v0['dvos'].get('indlamb_hor') is not None
            for k0, v0 in doptics.items()
        ])
        compact_lamb = compact_lamb_cross or compact_lamb_hor
        if compact_lamb is True:
            msg = (
                "plot_diagnostic_geometrical_coverage() not implemented "
                "for spectro with compact_lamb = True\n"
            )
            raise NotImplementedError(msg)

        _compute_cross = _spectro._compute_cross
        _compute_hor = _spectro._compute_hor
        _plot_cross = _spectro._plot
        _plot_hor = _spectro._plot

    else:
        _compute_cross = _broadband._compute_cross
        _compute_hor = _broadband._compute_hor
        _plot_cross = _broadband._plot_cross
        _plot_hor = _broadband._plot_hor

    # -----------------
    # compute cross
    # -----------------

    if plot_cross is True:

        dcomp_cross = _compute_cross(
            coll=coll,
            key=key,
            lcam=lcam,
            is_vos=is_vos,
            func_RZphi_from_ind=func_RZphi_from_ind,
            keym=keym,
            nan0=nan0,
            res_RZ=res_RZ,
            dcolor=dcolor,
        )

        # -----------
        # plot

        dax = _plot_cross(
            coll=coll,
            key=key,
            lcam=lcam,
            # data
            is_vos=is_vos,
            # plotting options
            config=config,
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
            cmap=cmap,
            dvminmax=dvminmax,
            # proj
            proj='cross',
            # compute output
            **dcomp_cross,
        )

    # ------------
    # plot_hor
    # ------------

    if plot_hor is True:

        # -----------
        # compute

        dcomp_hor = _compute_hor(
            coll=coll,
            key=key,
            lcam=lcam,
            is_vos=is_vos,
            keym=keym,
            func_RZphi_from_ind=func_RZphi_from_ind,
            nan0=nan0,
            res_RZ=res_RZ,
            res_phi=res_phi,
            dcolor=dcolor,
        )

        # --------
        # plot

        dax = _plot_hor(
            coll=coll,
            key=key,
            lcam=lcam,
            # data
            is_vos=is_vos,
            # plotting options
            config=config,
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
            cmap=cmap,
            dvminmax=dvminmax,
            # proj
            proj='hor',
            # compute output
            **dcomp_hor,
        )

    return dax


# ################################################################
# ################################################################
#                           Check
# ################################################################


def _check(
    coll=None,
    key=None,
    key_cam=None,
    dvos=None,
    plot_cross=None,
    plot_hor=None,
    plot_rank=None,
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
        if any([
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

    # spectro
    spectro = coll.dobj['diagnostic'][key]['spectro']

    # ------------------
    # key_cam
    # ------------------

    is_vos = key in lok_vos
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']
    if is_vos:
        lcam = [
            kk for kk in lcam
            if doptics[kk]['dvos'].get('keym') is not None
        ]

    if isinstance(key_cam, str):
        key_cam = [key_cam]

    key_cam = ds._generic_check._check_var_iter(
        key_cam, 'key_cam',
        types=(list, tuple),
        types_iter=str,
        allowed=coll.dobj['diagnostic'][key]['camera'],
    )

    lcam = [kcam for kcam in lcam if kcam in key_cam]

    # ------------------
    # is_3d vs plot_hor
    # ------------------

    # cross vs hor vs 3d
    dvosproj = coll.check_diagnostic_vos_proj(
        key=key,
        key_cam=key_cam,
        logic='all',
    )

    # plot_cross
    plot_cross = ds._generic_check._check_var(
        plot_cross, 'plot_cross',
        types=bool,
        default=True if dvosproj['cross'] else False,
        allowed=[True, False] if dvosproj['cross'] else [False],
        extra_msg=f"diag '{key}' needs vos and 'ind_cross'",
    )

    # plot_hor
    plot_hor = ds._generic_check._check_var(
        plot_hor, 'plot_hor',
        types=bool,
        default=True if dvosproj['hor'] else False,
        allowed=[True, False] if dvosproj['hor'] else [False],
        extra_msg=f"diag '{key}' needs vos and 'ind_hor'",
    )

    # plot_rank
    plot_rank = ds._generic_check._check_var(
        plot_rank, 'plot_rank',
        types=bool,
        default=True,
    )

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

    if is_vos:

        # key_mesh unicity
        lmesh = set([doptics[kcam]['dvos']['keym'] for kcam in lcam])

        # res unicity
        lres_RZ = set([
            tuple(doptics[kcam]['dvos']['res_RZ']) for kcam in lcam
        ])

        # res_phi unicity
        lres_phi = set([
            doptics[kcam]['dvos']['res_phi'] for kcam in lcam
        ])

        if len(lmesh) != 1 or len(lres_RZ) != 1 or len(lres_phi) != 1:
            msg = (
                f"Non-unique mesh or res_RZ for vos of diag '{key}':\n"
                f"For lcam: {lcam}\n"
            )
            raise Exception(msg)

        keym = list(lmesh)[0]
        res_RZ = list(list(lres_RZ)[0])
        res_phi = list(lres_phi)[0]

    else:
        msg = "plot_diagnostic_coverage() only work if vos has been computed!"
        raise Exception(msg)

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
        dcolor = {k0: colors[ii % len(colors)] for ii, k0 in enumerate(lcam)}

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

    return (
        key, is_vos, lcam, dvos, spectro,
        plot_cross, plot_hor, plot_rank,
        observation_directions,
        keym, res_RZ, res_phi,
        nan0, dcolor,
    )


# ################################################################
# ################################################################
#                   Compute_rank
# ################################################################


# DEPRECATED ?
def _compute_rank(
    coll=None,
    key=None,
    lcam=None,
):

    # ---------------
    # prepare
    # ---------------

    wcam = coll._which_cam
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # get all sensors
    sensors = list(itt.chain.from_iterable([
        [
            f'{key}_[kcam]_{idet}'
            for idet in np.ndindex(coll.dobj[wcam][kcam]['dgeom']['shape'])
        ]
        for kcam in lcam
    ]))
    nsensors = len(sensors)
    isensors = np.arange(0, nsensors)

    # get all points
    dipts = {}
    for kcam in lcam:
        v0 = doptics[kcam]

        is_3d = (
            v0['dvos'].get('ind_3d') is not None
            and not any([ii is None for ii in v0['dvos']['ind_3d']])
        )
        if not is_3d:
            continue

        kindr, kindz, kindphi = v0['dvos']['ind_3d']
        indr = coll.ddata[kindr]['data']
        indz = coll.ddata[kindz]['data']
        indphi = coll.ddata[kindphi]['data']

        dipts[kcam] = {}
        for idet in np.ndindex(coll.dobj[wcam][kcam]['dgeom']['shape']):
            sli = idet + (slice(None),)
            iok = np.isfinite(indr[sli]) & (indr[sli] >= 0)
            sli = idet + (iok,)
            ind = np.unique(
                np.array([indr[sli], indz[sli], indphi[sli]]),
                axis=1,
            )
            dipts[kcam][idet] = ind

    ipts = np.unique(
        np.concatenate(
            tuple([
                np.unique(
                    np.concatenate(
                        tuple([v0 for v0 in v1.values()]),
                        axis=1,
                    ),
                    axis=1,
                )
                for v1 in dipts.values()
            ]),
            axis=1,
        ),
        axis=1,
    )

    npts = ipts.shape[1]

    # ---------------------
    # unique R, phi indices
    # ---------------------

    matrix_ndet = np.zeros((nsensors, npts), dtype=bool)
    idet_tot = 0
    for kcam in lcam:
        shape_cam = coll.dobj[wcam][kcam]['dgeom']['shape']
        for i0, idet in enumerate(np.ndindex(shape_cam)):
            for i1, ipt in enumerate(dipts[kcam][idet].T):
                ind = np.all(ipt[:, None] == ipts, axis=0)
                assert ind.sum() == 1
                ipt_tot = ind.nonzero()[0][0]
                matrix_ndet[idet_tot, ipt_tot] = True
            idet_tot += 1

    # ---------------------------
    # extract unique combinations
    # ---------------------------

    rank, ncounts = np.unique(
        matrix_ndet,
        axis=1,
        return_counts=True,
    )

    n_per_rank = np.sum(rank, axis=0)

    rank_x = np.unique(n_per_rank)
    rank_y = np.zeros(rank_x.shape)
    rank_z = np.zeros(rank_x.shape)
    for ir, rr in enumerate(rank_x):
        ind = n_per_rank == rr
        rank_y[ir] = np.sum(ncounts[ind])
        rank_z[ir] = ind.sum()

    assert np.sum(rank_y) == np.sum(ncounts)

    # ------------
    # output
    # ------------

    drank = {
        'matrix_det': matrix_ndet,
        'sensors': sensors,
        'isensors': isensors,
        'ipts': ipts,
        'dipts': dipts,
        'rank': rank,
        'ncounts': ncounts,
        'n_per_rank': n_per_rank,
        'rank_x': rank_x,
        'rank_y': rank_y,
        'rank_z': rank_z,
    }

    return drank


# ################################################################
# ################################################################
#                   plot_rank
# ################################################################


def _plot_rank(
    coll=None,
    key=None,
    lcam=None,
    # data
    drank=None,
    # plotting options
    marker=None,
    markersize=None,
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
        titdef = (
            f"Resolution info on diag '{key}' "
            "- full volume"
        )
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
        vmax = None

    if marker is None:
        marker = 's'
    if markersize is None:
        markersize = 8

    # ----------------
    # prepare data
    # ----------------

    # directions of observation

    # ----------------
    # prepare figure
    # ----------------

    if dax.get('rank') is None:
        if fs is None:
            fs = (16, 7)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.20, 'wspace': 0.40,
            }

        Na, Ni, Nc = 7, 3, 1
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=(4*Na+3*Ni+3*Nc), nrows=8, **dmargin)

        # ax0 = spans
        ax0 = fig.add_subplot(gs[:, :Na], aspect='equal', adjustable='box')
        ax0.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax0.set_xlabel('R (m)', size=12, fontweight='bold')
        ax0.set_title("spans", size=14, fontweight='bold')

        # ax1 = nb of detectors
        ax1 = fig.add_subplot(
            gs[:, Na+Ni:2*Na+Ni],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax1.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax1.set_xlabel('R (m)', size=12, fontweight='bold')
        ax1.set_title("nb. of detectors", size=14, fontweight='bold')

        # colorbar ndet
        cax_ndet = fig.add_subplot(gs[1:-1, 2*Na+Ni], frameon=False)

        # ax2 = dz
        ax2 = fig.add_subplot(
            gs[:, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax2.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax2.set_xlabel('R (m)', size=12, fontweight='bold')
        ax2.set_title(
            "Vertical depth (m)",
            size=14,
            fontweight='bold',
        )

        # colorbar
        cax_dz = fig.add_subplot(gs[1:-1, 3*Na+2*Ni+Nc], frameon=False)
        cax_dz.set_title('m', size=12)

        # ax3 = sang
        ax3 = fig.add_subplot(
            gs[:, 3*Na+3*Ni+2*Nc:4*Na+3*Ni+2*Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax3.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax3.set_xlabel('R (m)', size=12, fontweight='bold')
        ax3.set_title(
            "Integrated solid angle (sr)",
            size=14,
            fontweight='bold',
        )

        # colorbar
        cax_sang = fig.add_subplot(gs[1:-1, -1], frameon=False)

        dax.update({
            'span_hor': {'handle': ax0, 'type': 'span_hor'},
            'ndet_hor': {'handle': ax1, 'type': 'ndet_hor'},
            'cax_ndet_hor': {'handle': cax_ndet, 'type': 'cbar_ndet_hor'},
            'dz_hor': {'handle': ax2, 'type': 'dz_hor'},
            'cax_dz_hor': {'handle': cax_dz, 'type': 'cbar_dz_hor'},
            'sang_hor': {'handle': ax3, 'type': 'sang_hor'},
            'cax_sang_hor': {'handle': cax_sang, 'type': 'cbar_sang_hor'},
        })

    # --------------------
    # check / format dax

    dax = ds._generic_check._check_dax(dax)
    fig = dax['ndet_hor']['handle'].figure

    # ---------------
    # plot spans
    # ---------------
