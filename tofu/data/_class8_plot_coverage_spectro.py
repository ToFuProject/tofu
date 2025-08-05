

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _class8_vos_utilities as _vos_utils


# ################################################################
# ################################################################
#                           Compute cross
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

    # -----------
    # initialize

    # common with broadband
    sangdV = np.zeros(shape, dtype=float)
    dVndV = np.zeros(shape, dtype=float)
    ndet = np.zeros(shape, dtype=float)

    # specific to spectro
    lamb = np.zeros(shape, dtype=float)
    lamb_min = np.full(shape, np.inf, dtype=float)
    lamb_max = np.zeros(shape, dtype=float)

    # ---------------
    # loop on cameras

    if is_vos:

        wcam = coll._which_cam
        for kcam in lcam:

            v0 = doptics[kcam]

            # keys
            kindr, kindz = v0['dvos']['ind_cross']
            ksang = v0['dvos']['sang_cross']
            kdV = v0['dvos']['dV_cross']
            kndV = v0['dvos']['ndV_cross']
            klamb = v0['dvos']['lamb']
            kph = v0['dvos']['ph_cross']

            # data
            indr = coll.ddata[kindr]['data']
            indz = coll.ddata[kindz]['data']
            sangi = coll.ddata[ksang]['data']
            dVi = coll.ddata[kdV]['data']
            ndVi = coll.ddata[kndV]['data']
            lambi = coll.ddata[klamb]['data']
            ph = coll.ddata[kph]['data']

            # axis
            ref_cam = coll.dobj[wcam][kcam]['dgeom']['ref']
            ref_sang = coll.ddata[ksang]['ref']
            axis_cam = tuple([ref_sang.index(rr) for rr in ref_cam])
            axis_camlamb = axis_cam + (axis_cam[-1]+2,)
            sli_lamb = (None,)*len(ref_cam) + (None, slice(None))

            # safety check: for spectro, cameras must not overlap
            if np.any(ndet[indr, indz] > 0):
                msg = (
                    "plot coverage for spectro only handles multiple cameras"
                    " if they don't have overlapping FOV!\n"
                    f"\t- diag: {key}\n"
                    f"\t- cam: {kcam}\n"
                )
                raise Exception(msg)

            # compute - dVndV, sangdV, ndet
            sangdV[indr, indz] = np.sum(sangi * dVi, axis=axis_cam)
            dVndV[indr, indz] = ndVi * dVi
            ndet[indr, indz] = np.sum(sangi > 0., axis=axis_cam)

            # compute lamb, dlamb
            sum_ph = np.sum(ph, axis=axis_camlamb)
            iok_sum_ph = sum_ph > 0.
            lamb[indr[iok_sum_ph], indz[iok_sum_ph]] = (
                np.sum(ph * lambi[sli_lamb], axis=axis_camlamb)[iok_sum_ph]
                / sum_ph[iok_sum_ph]
            )

            iok = ph > 0.
            lamb_min[indr, indz] = np.minimum(
                np.min(
                    lambi[sli_lamb]*iok,
                    axis=axis_camlamb,
                    where=iok,
                    initial=np.inf,
                ),
                lamb_min[indr, indz],
            )
            lamb_max[indr, indz] = np.maximum(
                np.max(
                    lambi[sli_lamb]*iok,
                    axis=axis_camlamb,
                    where=iok,
                    initial=0,
                ),
                lamb_max[indr, indz],
            )

        # derive dlamb
        dlamb = lamb_max - lamb_min
        dlamb[np.isinf(lamb_min)] = 0.

        # nan
        if nan0 is True:
            iout = ndet == 0
            sangdV[iout] = np.nan
            dVndV[iout] = np.nan
            ndet[iout] = np.nan
            lamb[iout] = np.nan
            dlamb[iout] = np.nan
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

    return {
        'dpoly': dpoly,
        'ndet': ndet,
        'dVndV': dVndV,
        'sangdV': sangdV,
        'lamb': lamb,
        'dlamb': dlamb,
        'extent': extent,
    }


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
        kindr, kindphi = v0['dvos']['ind_hor']
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

    # -----------
    # initialize

    shape = (indrpuT.shape[0],)
    sangdV = np.zeros(shape, dtype=float)
    dVndV = np.zeros(shape, dtype=float)
    ndet = np.zeros(shape, dtype=float)

    # specific to spectro
    lamb = np.zeros(shape, dtype=float)
    lamb_min = np.full(shape, np.inf, dtype=float)
    lamb_max = np.zeros(shape, dtype=float)

    # ---------------
    # loop on cameras

    wcam = coll._which_cam
    for kcam in lcam:
        v0 = doptics[kcam]

        # keys
        kindr, kindphi = v0['dvos']['ind_hor']
        ksang = v0['dvos']['sang_hor']
        kdV = v0['dvos']['dV_hor']
        kndV = v0['dvos']['ndV_hor']
        klamb = v0['dvos']['lamb']
        kph = v0['dvos']['ph_hor']

        # axis
        ref_cam = coll.dobj[wcam][kcam]['dgeom']['ref']
        ref_sang = coll.ddata[ksang]['ref']
        axis_cam = tuple([ref_sang.index(rr) for rr in ref_cam])
        axis_camlamb = axis_cam + (axis_cam[-1]+2,)
        sli_lamb = (None,)*len(ref_cam) + (None, slice(None))

        # data
        indr = coll.ddata[kindr]['data']
        indphi = coll.ddata[kindphi]['data']
        sangi = coll.ddata[ksang]['data']
        dVi = coll.ddata[kdV]['data']
        ndVi = coll.ddata[kndV]['data']
        lambi = coll.ddata[klamb]['data']
        ph = coll.ddata[kph]['data']

        # inds
        ind = np.array([
            np.nonzero(
                np.all(
                    indrpuT == np.r_[indr[ii], indphi[ii]][None, :],
                    axis=1,
                )
            )[0][0]
            for ii in range(0, indr.size)
        ]).astype(int)

        # compute
        sangdV[ind] = np.sum(sangi * dVi, axis=axis_cam)
        dVndV[ind] = ndVi * dVi
        ndet[ind] = np.sum(sangi > 0., axis=axis_cam)

        # compute lamb, dlamb
        sum_ph = np.sum(ph, axis=axis_camlamb)
        iok_sum_ph = sum_ph > 0.
        lamb[ind[iok_sum_ph]] = (
            np.sum(ph * lambi[sli_lamb], axis=axis_camlamb)[iok_sum_ph]
            / sum_ph[iok_sum_ph]
        )

        iok = ph > 0.
        lamb_min[ind] = np.minimum(
            np.min(
                lambi[sli_lamb]*iok,
                axis=axis_camlamb,
                where=iok,
                initial=np.inf,
            ),
            lamb_min[ind],
        )
        lamb_max[ind] = np.maximum(
            np.max(
                lambi[sli_lamb]*iok,
                axis=axis_camlamb,
                where=iok,
                initial=0,
            ),
            lamb_max[ind],
        )

    # derive dlamb
    dlamb = lamb_max - lamb_min
    dlamb[np.isinf(lamb_min)] = 0.

    # ------------------
    # nan0
    # ------------------

    if nan0 is True:
        iout = ndet == 0
        sangdV[iout] = np.nan
        dVndV[iout] = np.nan
        ndet[iout] = np.nan
        lamb[iout] = np.nan
        dlamb[iout] = np.nan
    else:
        ndet = ndet.astype(int)

    # ------------------
    # R, phi coordinates
    # ------------------

    rr, zz, pp, _ = func_RZphi_from_ind(
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

    return {
        'dpoly': dpoly,
        'ndet': ndet,
        'dVndV': dVndV,
        'sangdV': sangdV,
        'lamb': lamb,
        'dlamb': dlamb,
        'xx': xx,
        'yy': yy,
    }


# ################################################################
# ################################################################
#                       plot cross
# ################################################################


def _plot(
    coll=None,
    key=None,
    lcam=None,
    # data
    ndet=None,
    dVndV=None,
    sangdV=None,
    dpoly=None,
    lamb=None,
    dlamb=None,
    # proj
    proj=None,
    # proj-specific
    extent=None,
    xx=None,
    yy=None,
    marker=None,
    markersize=None,
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
    # unused
    **kwdargs,
):

    # ----------------
    # check input
    # ----------------

    # tit
    if tit is not False:
        titdef = f"geometrical coverage of diag '{key}' "
        if proj == 'cross':
            titdef += "- integrated cross-section"
        else:
            titdef += "- integrated vertically"

        tit = ds._generic_check._check_var(
            tit, 'tit',
            types=str,
            default=titdef,
        )

    # cmap
    if cmap is None:
        cmap = plt.cm.viridis   # Greys

    # vmin
    if vmin is None:
        vmin = 0

    # vmax
    if vmax is None:
        vmax = np.nanmax(ndet)

    # plot_func
    pfunc = _get_plot_func(
        proj=proj,
        # set
        xx=xx,
        yy=yy,
        extent=extent,
        marker=marker,
        markersize=markersize,
        cmap=cmap,
    )

    # ----------------
    # prepare data
    # ----------------

    # directions of observation

    # ----------------
    # prepare figure
    # ----------------

    if dax.get(f'ndet_{proj}') is None:
        dax = _get_dax(
            fs=fs,
            dmargin=dmargin,
            dax=dax,
            proj=proj,
        )

    # --------------------
    # check / format dax

    dax = ds._generic_check._check_dax(dax)
    fig = dax[f'ndet_{proj}']['handle'].figure

    # ---------------
    # plot spans
    # ---------------

    ktype = f'span_{proj}'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        for k0, v0 in dpoly.items():
            ax.fill(
                v0['pr'] if proj == 'cross' else v0['px'],
                v0['pz'] if proj == 'cross' else v0['py'],
                fc=v0['color'],
                label=k0,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # ---------------
    # plot ndet
    # ---------------

    ktype = f'ndet_{proj}'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = pfunc(
            ax,
            ndet,
            vmin=vmin,
            vmax=vmax,
        )

        # colorbar
        _add_colorbar(im, ktype, dax)

    # ---------------
    # plot dphi
    # ---------------

    if dVndV is not None:
        ktype = f'dV_{proj}'
        lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
        for ax in lax:

            # plot
            im = pfunc(
                ax,
                dVndV,
                vmin=0,
                vmax=None,
            )

            # colorbar
            _add_colorbar(im, ktype, dax)

    # ---------------
    # plot sang
    # ---------------

    ktype = f'sang_{proj}'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = pfunc(
            ax,
            sangdV,
            vmin=0,
            vmax=None,
        )

        # colorbar
        _add_colorbar(im, ktype, dax)

    # ---------------
    # plot lamb
    # ---------------

    ktype = f'lamb_{proj}'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = pfunc(
            ax,
            lamb,
            vmin=np.min(lamb[lamb > 0]),
            vmax=np.max(lamb),
        )

        # colorbar
        _add_colorbar(im, ktype, dax)

    # ---------------
    # plot dlamb
    # ---------------

    ktype = f'dlamb_{proj}'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        iok = dlamb > 0

        # plot
        im = pfunc(
            ax,
            lamb,
            vmin=np.min(dlamb[iok]) if np.any(iok) else 0.,
            vmax=np.max(dlamb),
        )

        # colorbar
        _add_colorbar(im, ktype, dax)

    # --------------
    # add config
    # --------------

    if config is not None:
        for kax, vax in dax.items():
            typ = vax['type'][0]
            if not typ.startswith('cbar') and (proj in typ):
                config.plot(lax=vax['handle'], proj=proj, dLeg=False)

    # --------------
    # add tit
    # --------------

    if tit is not False:
        fig.suptitle(tit, size=14, fontweight='bold')

    return dax


# #########################################################
# #########################################################
#                       dax
# #########################################################


def _get_dax(
    fs=None,
    dmargin=None,
    dax=None,
    proj=None,
):

    # ---------
    # inputs
    # ---------

    if fs is None:
        fs = (16, 7)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.20, 'wspace': 0.40,
        }

    if proj == 'cross':
        xlab = 'R (m)'
        ylab = 'Z (m)'
    elif proj == 'hor':
        xlab = 'X (m)'
        ylab = 'Y (m)'

    # --------------
    # initialize
    # --------------

    Na, Ni, Nc = 7, 3, 1
    Ca, Ci = 4, 1
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(
        ncols=(4*Na+3*Ni+3*Nc),
        nrows=2*Ca+Ci,
        **dmargin,
    )

    # --------------
    # ax0 = spans
    # --------------

    ax0 = fig.add_subplot(gs[:Ca, :Na], aspect='equal', adjustable='box')
    ax0.set_title("spans", size=14, fontweight='bold')

    # --------------
    # ax1 = nb of detectors
    # --------------

    ax1 = fig.add_subplot(
        gs[:Ca, Na+Ni:2*Na+Ni],
        aspect='equal',
        sharex=ax0,
        sharey=ax0,
        adjustable='box',
    )
    tstr = r"$\sum_{det_i}$"
    ax1.set_title(f"nb. of detectors {tstr}", size=14, fontweight='bold')

    # colorbar ndet
    cax_ndet = fig.add_subplot(gs[:Ca, 2*Na+Ni], frameon=False)

    # --------------
    # ax2 = dV
    # --------------

    ax2 = fig.add_subplot(
        gs[:Ca, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
        aspect='equal',
        sharex=ax0,
        sharey=ax0,
        adjustable='box',
    )
    tstr = r"$\sum_{det_i} \sum_{V_i} dV$"
    ax2.set_title(
        f"Observed volume {tstr} (m3)",
        size=14,
        fontweight='bold',
    )

    # colorbar
    cax_dz = fig.add_subplot(gs[:Ca, 3*Na+2*Ni+Nc], frameon=False)
    cax_dz.set_title('m', size=12)

    # --------------
    # ax3 = sang
    # --------------

    ax3 = fig.add_subplot(
        gs[:Ca, 3*Na+3*Ni+2*Nc:4*Na+3*Ni+2*Nc],
        aspect='equal',
        sharex=ax0,
        sharey=ax0,
        adjustable='box',
    )
    tstr = r"$\sum_{det_i} \sum_{V_i} \Omega dV$"
    ax3.set_title(
        f"Integrated solid angle {tstr} (sr.m3)",
        size=14,
        fontweight='bold',
    )

    # colorbar
    cax_sang = fig.add_subplot(gs[:Ca, -1], frameon=False)

    # --------------
    # ax4 = lamb
    # --------------

    ax4 = fig.add_subplot(
        gs[Ca+Ci:, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
        aspect='equal',
        sharex=ax0,
        sharey=ax0,
        adjustable='box',
    )
    tstr = r"$<\lambda>$"
    ax4.set_title(
        f"Average wavelength {tstr} (m)",
        size=14,
        fontweight='bold',
    )

    # colorbar
    cax_lamb = fig.add_subplot(gs[Ca+Ci:, 3*Na+2*Ni+Nc], frameon=False)

    # --------------
    # ax5 = dlamb
    # --------------

    ax5 = fig.add_subplot(
        gs[Ca+Ci:, 3*Na+3*Ni+2*Nc: 4*Na+3*Ni+2*Nc],
        aspect='equal',
        sharex=ax0,
        sharey=ax0,
        adjustable='box',
    )
    tstr = r"$\delta\lambda$"
    ax5.set_title(
        f"Wavelength span {tstr} (m)",
        size=14,
        fontweight='bold',
    )

    # colorbar
    cax_dlamb = fig.add_subplot(gs[Ca+Ci:, -1], frameon=False)

    # --------------
    # dict
    # --------------

    dax.update({
        f'span_{proj}': {'handle': ax0, 'type': f'span_{proj}'},
        f'ndet_{proj}': {'handle': ax1, 'type': f'ndet_{proj}'},
        f'cax_ndet_{proj}': {'handle': cax_ndet, 'type': f'cbar_ndet_{proj}'},
        f'dV_{proj}': {'handle': ax2, 'type': f'dV_{proj}'},
        f'cax_dV_{proj}': {'handle': cax_dz, 'type': f'cbar_dV_{proj}'},
        f'sang_{proj}': {'handle': ax3, 'type': f'sang_{proj}'},
        f'cax_sang_{proj}': {'handle': cax_sang, 'type': f'cbar_sang_{proj}'},
        f'lamb_{proj}': {'handle': ax4, 'type': f'lamb_{proj}'},
        f'cax_lamb_{proj}': {'handle': cax_lamb, 'type': f'cbar_lamb_{proj}'},
        f'dlamb_{proj}': {'handle': ax5, 'type': f'dlamb_{proj}'},
        f'cax_dlamb_{proj}': {
            'handle': cax_dlamb,
            'type': f'cbar_dlamb_{proj}',
        },
    })

    # -------------
    # add labels
    # -------------

    for kax, vax in dax.items():
        if proj in kax and not kax.startswith('cbar'):
            vax['handle'].set_xlabel(xlab, size=12, fontweight='bold')
            vax['handle'].set_ylabel(ylab, size=12, fontweight='bold')

    return dax


# #####################################################
# #####################################################
#               Get plot func
# #####################################################


def _get_plot_func(
    proj=None,
    # set
    xx=None,
    yy=None,
    extent=None,
    marker=None,
    markersize=None,
    cmap=None,
):
    if proj == 'cross':
        def func(
            # needed
            ax, data,
            vmin=None,
            vmax=None,
            # set
            extent=extent,
            cmap=cmap,
        ):
            im = ax.imshow(
                data.T,
                extent=extent,
                origin='lower',
                interpolation='bilinear',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            return im
    else:
        def func(
            # needed
            ax, data,
            vmin=None,
            vmax=None,
            # set
            xx=xx,
            yy=yy,
            marker=marker,
            markersize=markersize,
            cmap=cmap,
        ):
            im = ax.scatter(
                xx,
                yy,
                c=data,
                s=markersize,
                marker=marker,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            return im

    return func


# ##################################################
# ##################################################
#               add colorbar
# ##################################################


def _add_colorbar(
    im,
    ktype,
    dax,
):
    # colorbar
    ktype2 = f'cbar_{ktype}'
    lcax = [v0['handle'] for v0 in dax.values() if ktype2 in v0['type']]
    if len(lcax) == 0:
        plt.colorbar(im)
    else:
        for cax in lcax:
            plt.colorbar(im, cax=cax)

    return
