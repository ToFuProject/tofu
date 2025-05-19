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


from . import _class8_vos_utilities as _vos_utils


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
    vmin=None,
    vmax=None,
):

    # -------------
    # check inputs
    # -------------

    (
        key, is_vos, lcam, dvos,
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

    # -----------
    # compute cross
    # -----------

    if plot_cross is True:

        dpoly, ndet, dphi, sang, extent = _compute_cross(
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

        # ------------
        # nan0

        if nan0 is True:
            ndet[ndet == 0] = np.nan

        # -----------
        # plot

        dax = _plot_cross(
            coll=coll,
            key=key,
            lcam=lcam,
            # data
            is_vos=is_vos,
            ndet=ndet,
            dphi=dphi,
            sang=sang,
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

    # ------------
    # plot_hor
    # ------------

    if plot_hor is True:

        # -----------
        # compute

        dpoly, ndet, dz, sang, xx, yy = _compute_hor(
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
            ndet=ndet,
            dz=dz,
            sang=sang,
            xx=xx,
            yy=yy,
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

    # ------------
    # plot_rank
    # ------------

    if plot_rank is True:

        # -----------
        # compute
        drank = None
        # drank = _compute_rank(
            # coll=coll,
            # key=key,
            # lcam=lcam,
        # )

        # --------
        # plot

        # dax = _plot_rank(
            # coll=coll,
            # key=key,
            # lcam=lcam,
            # # data
            # drank=drank,
            # # plotting options
            # config=config,
            # dax=dax,
            # fs=fs,
            # dmargin=dmargin,
            # tit=tit,
            # cmap=cmap,
            # vmin=vmin,
            # vmax=vmax,
        # )

    return dpoly, ndet, drank, dax


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

    # is 3d ?
    is_3d = (
        doptics[lcam[0]]['dvos'].get('ind_3d') is not None
        and all([kk is not None for kk in doptics[lcam[0]]['dvos']['ind_3d']])
    )

    # plot_cross
    plot_cross = ds._generic_check._check_var(
        plot_cross, 'plot_cross',
        types=bool,
        default=True,
    )

    # plot_hor
    plot_hor = ds._generic_check._check_var(
        plot_hor, 'plot_hor',
        types=bool,
        default=is_3d,
        allowed=[False, True] if is_3d else [False],
    )

    # plot_hor
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
        key, is_vos, lcam, dvos,
        plot_cross, plot_hor, plot_rank,
        observation_directions,
        keym, res_RZ, res_phi,
        nan0, dcolor,
    )


# ################################################################
# ################################################################
#                           Compute
# ################################################################


def _compute_cross(
    coll=None,
    key=None,
    lcam=None,
    is_vos=None,
    keym=None,
    func_RZphi_from_ind=None,
    res_RZ=None,
    nan0=None,
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

    sang = np.zeros(shape, dtype=float)
    ndet = np.zeros(shape, dtype=float)

    d3d = {
        kcam: doptics[kcam]['dvos'].get('ind_3d', [None])[0] is not None
        for kcam in lcam
    }
    if any([vv is True for vv in d3d.values()]):
        dphi = np.zeros(shape, dtype=float)
    else:
        dphi = None

    if is_vos:
        for kcam in lcam:

            v0 = doptics[kcam]

            kindr, kindz = v0['dvos']['ind_cross']
            ksang = v0['dvos']['sang_cross']
            indr = coll.ddata[kindr]['data']
            indz = coll.ddata[kindz]['data']
            sangi = coll.ddata[ksang]['data']
            res_phi = v0['dvos']['res_phi']

            for ii, ind in enumerate(np.ndindex(sangi.shape[:-1])):
                iok = np.isfinite(sangi[ind + (slice(None),)])
                sli = ind + (iok,)
                sang[indr[sli], indz[sli]] += sangi[sli]
                ndet[indr[sli], indz[sli]] += 1

                if d3d[kcam] is True:
                    dphi[indr[sli], indz[sli]] += res_phi

        if nan0 is True:
            iout = ndet == 0
            sang[iout] = np.nan
            ndet[iout] = np.nan
            if dphi is not None:
                dphi[iout] = np.nan
        else:
            ndet = ndet.astype(int)

    # -----------------
    # ndet for non-vos
    # -----------------

    else:
        raise NotImplementedError()

    # ---------------
    # dpoly
    # ---------------

    dpoly = {}
    for kcam in lcam:

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

    return dpoly, ndet, dphi, sang, extent


# ################################################################
# ################################################################
#                   Compute_rank
# ################################################################


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
#                   Compute_hor
# ################################################################


def _compute_hor(
    coll=None,
    key=None,
    lcam=None,
    is_vos=None,
    keym=None,
    func_RZphi_from_ind=None,
    res_RZ=None,
    res_phi=None,
    nan0=None,
    dcolor=None,
):

    # ---------------
    # prepare
    # ---------------

    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ---------------------
    # unique R, phi indices
    # ---------------------

    indrpu = []
    for kcam in lcam:
        v0 = doptics[kcam]
        kindr, _, kindphi = v0['dvos']['ind_3d']
        indr = coll.ddata[kindr]['data']
        indphi = coll.ddata[kindphi]['data']

        iok = np.isfinite(indr) & (indr >= 0)
        indrpu.append(
            np.unique([indr[iok].ravel(), indphi[iok].ravel()], axis=1)
        )

    indrpuT = np.unique(np.concatenate(tuple(indrpu), axis=1), axis=1).T
    indrpuT = indrpuT.astype(int)

    # ------------------------
    # Get solid angle and ndet
    # ------------------------

    sang = np.zeros((indrpuT.shape[0],), dtype=float)
    ndet = np.zeros((indrpuT.shape[0],), dtype=float)
    dz = np.zeros((indrpuT.shape[0],), dtype=float)
    dindz = {}

    for kcam in lcam:
        v0 = doptics[kcam]

        # keys
        kindr, kindz, kindphi = v0['dvos']['ind_3d']
        ksang = v0['dvos']['sang_3d']
        kdV = v0['dvos']['dV_3d']
        res_Z = v0['dvos']['res_RZ'][1]

        # data
        indr = coll.ddata[kindr]['data']
        indz = coll.ddata[kindz]['data']
        indphi = coll.ddata[kindphi]['data']
        sangi = coll.ddata[ksang]['data']
        dV = coll.ddata[kdV]['data']

        iok = (indr >= 0)
        for ii, (ir, ip) in enumerate(indrpuT):
            ind = (indr == ir) & (indphi == ip) & iok
            sang[ii] += np.sum(sangi[ind] / dV[ind])
            ndet[ii] += np.sum(np.any(ind, axis=-1))

            # vertical depth without duplicates
            if (ir, ip) not in dindz.keys():
                dindz[(ir, ip)] = {'iz': np.empty((0,)), 'ii': ii}

            dindz[(ir, ip)]['iz'] = np.append(
                dindz[(ir, ip)]['iz'],
                indz[ind],
            )

    for irp, vv in dindz.items():
        dz[vv['ii']] = np.unique(vv['iz']).size * res_Z

    # ------------------
    # nan0
    # ------------------

    if nan0 is True:
        iout = ndet == 0
        sang[iout] = np.nan
        ndet[iout] = np.nan
    else:
        ndet = ndet.astype(int)

    # ------------------
    # R, phi coordinates
    # ------------------

    rr, zz, pp, dV = func_RZphi_from_ind(
        indr=indrpuT[:, 0],
        indz=indrpuT[:, 0],
        indphi=indrpuT[:, 1],
    )

    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)

    # ---------------
    # dpoly
    # ---------------

    dpoly = {}
    for kcam in lcam:

        # concatenate all vos
        px, py = _vos_utils._get_overall_polygons(
            coll=coll,
            doptics=doptics,
            key_cam=kcam,
            poly='phor',
            convexHull=False,
        )

        # store
        dpoly[kcam] = {
            'px': px,
            'py': py,
            'color': dcolor[kcam],
        }

    return dpoly, ndet, dz, sang, xx, yy


# ################################################################
# ################################################################
#                           plot
# ################################################################


def _plot_cross(
    coll=None,
    key=None,
    lcam=None,
    # data
    ndet=None,
    dphi=None,
    sang=None,
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
        titdef = (
            f"geometrical coverage of diag '{key}' "
            "- integrated cross-section"
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
        vmax = np.nanmax(ndet)

    # ----------------
    # prepare data
    # ----------------

    # directions of observation

    # ----------------
    # prepare figure
    # ----------------

    if dax.get('ndet_cross') is None:

        if fs is None:
            fs = (16, 7)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.10, 'top': 0.88,
                'hspace': 0.20, 'wspace': 0.20,
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

        # ax2 = dphi
        ax2 = fig.add_subplot(
            gs[:, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax2.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax2.set_xlabel('R (m)', size=12, fontweight='bold')
        ax2.set_title("toroidal depth", size=14, fontweight='bold')

        # colorbar ndet
        cax_dphi = fig.add_subplot(gs[1:-1, 3*Na+2*Ni+Nc], frameon=False)

        # ax3 = sang
        ax3 = fig.add_subplot(
            gs[:, 3*Na+3*Ni+2*Nc: 4*Na+3*Ni+2*Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax3.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax3.set_xlabel('R (m)', size=12, fontweight='bold')
        ax3.set_title(
            "Cumulated integrated solid angle (sr.m3)",
            size=14,
            fontweight='bold',
        )

        # colorbar
        cax_sang = fig.add_subplot(gs[1:-1, -1], frameon=False)

        dax.update({
            'span_cross': {'handle': ax0, 'type': 'span_cross'},
            'ndet_cross': {'handle': ax1, 'type': 'ndet_cross'},
            'cax_ndet_cross': {'handle': cax_ndet, 'type': 'cbar_ndet_cross'},
            'dphi_cross': {'handle': ax2, 'type': 'dphi_cross'},
            'cax_dphi_cross': {'handle': cax_dphi, 'type': 'cbar_dphi_cross'},
            'sang_cross': {'handle': ax3, 'type': 'sang_cross'},
            'cax_sang_cross': {'handle': cax_sang, 'type': 'cbar_sang_cross'},
        })

    # --------------------
    # check / format dax

    dax = ds._generic_check._check_dax(dax)
    fig = dax['ndet_cross']['handle'].figure

    # ---------------
    # plot spans
    # ---------------

    ktype = 'span_cross'
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

    # ---------------
    # plot ndet
    # ---------------

    ktype = 'ndet_cross'
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
        ktype2 = 'cbar_ndet_cross'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # ylab = "nb. of detectors"
                # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # ---------------
    # plot dphi
    # ---------------

    if dphi is not None:
        ktype = 'dphi_cross'
        lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
        for ax in lax:

            # plot
            im = ax.imshow(
                dphi.T,
                extent=extent,
                origin='lower',
                interpolation='bilinear',
                cmap=cmap,
                vmin=0,
                vmax=None,
            )

            # colorbar
            ktype2 = 'cbar_dphi_cross'
            lcax = [
                v0['handle'] for v0 in dax.values()
                if ktype2 in v0['type']
            ]
            if len(lcax) == 0:
                plt.colorbar(im)
            else:
                for cax in lcax:
                    plt.colorbar(im, cax=cax)
                    # ylab = "nb. of detectors"
                    # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # ---------------
    # plot sang
    # ---------------

    ktype = 'sang_cross'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.imshow(
            sang.T,
            extent=extent,
            origin='lower',
            interpolation='bilinear',
            cmap=cmap,
            vmin=0,
            vmax=None,
        )

        # colorbar
        ktype2 = 'cbar_sang_cross'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # ylab = "nb. of detectors"
                # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # --------------
    # add config
    # --------------

    if config is not None:
        for ktype in ['span_cross', 'ndet_cross', 'dphi_cross', 'sang_cross']:
            lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
            for ax in lax:
                config.plot(lax=ax, proj='cross', dLeg=False)

    # --------------
    # add tit
    # --------------

    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    return dax


# ################################################################
# ################################################################
#                   plot_hor
# ################################################################


def _plot_hor(
    coll=None,
    key=None,
    lcam=None,
    # data
    ndet=None,
    dz=None,
    sang=None,
    xx=None,
    yy=None,
    dpoly=None,
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
            f"geometrical coverage of diag '{key}' "
            "- integrated vertically"
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
        vmax = np.nanmax(ndet)

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

    if dax.get('ndet_hor') is None:
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

    ktype = 'span_hor'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        for k0, v0 in dpoly.items():
            ax.fill(
                v0['px'],
                v0['py'],
                fc=v0['color'],
                label=k0,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # ---------------
    # plot ndet
    # ---------------

    ktype = 'ndet_hor'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.scatter(
            xx,
            yy,
            c=ndet,
            s=markersize,
            marker=marker,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # colorbar
        ktype2 = 'cbar_ndet_hor'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # ylab = "nb. of detectors"
                # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # ---------------
    # plot dz
    # ---------------

    ktype = 'dz_hor'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.scatter(
            xx,
            yy,
            c=dz,
            s=markersize,
            marker=marker,
            cmap=cmap,
            vmin=0,
            vmax=None,
        )

        # colorbar
        ktype2 = 'cbar_dz_hor'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # ylab = "nb. of detectors"
                # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # ---------------
    # plot sang
    # ---------------

    ktype = 'sang_hor'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.scatter(
            xx,
            yy,
            c=sang,
            s=markersize,
            marker=marker,
            cmap=cmap,
            vmin=0,
            vmax=None,
        )

        # colorbar
        ktype2 = 'cbar_sang_hor'
        lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
        if len(lcax) == 0:
            plt.colorbar(im)
        else:
            for cax in lcax:
                plt.colorbar(im, cax=cax)
                # ylab = "nb. of detectors"
                # cax.set_ylabel(ylab, size=14, fontweight='bold')

    # --------------
    # add config
    # --------------

    if config is not None:
        for ktype in ['span_hor', 'ndet_hor', 'dz_hor', 'sang_hor']:
            lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
            for ax in lax:
                config.plot(lax=ax, proj='hor', dLeg=False)

    # --------------
    # add tit
    # --------------

    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    return dax


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
