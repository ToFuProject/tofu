

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

    sangdV = np.zeros(shape, dtype=float)
    dVndV = np.zeros(shape, dtype=float)
    ndet = np.zeros(shape, dtype=float)

    # ---------------
    # loop on cameras

    if is_vos:
        for kcam in lcam:

            v0 = doptics[kcam]

            # keys
            kindr, kindz = v0['dvos']['ind_cross']
            ksang = v0['dvos']['sang_cross']
            kdV = v0['dvos']['dV_cross']
            kndV = v0['dvos']['ndV_cross']

            # data
            indr = coll.ddata[kindr]['data']
            indz = coll.ddata[kindz]['data']
            sangi = coll.ddata[ksang]['data']
            dVi = coll.ddata[kdV]['data']
            ndVi = coll.ddata[kndV]['data']

            # compute
            for ii, ind in enumerate(np.ndindex(sangi.shape[:-1])):
                iok = np.isfinite(sangi[ind + (slice(None),)])
                sli = ind + (iok,)
                sangdV[indr[sli], indz[sli]] += sangi[sli] * dVi[sli]
                dVndV[indr[sli], indz[sli]] += ndVi[sli] * dVi[sli]
                ndet[indr[sli], indz[sli]] += 1

        # nan
        if nan0 is True:
            iout = ndet == 0
            sangdV[iout] = np.nan
            dVndV[iout] = np.nan
            ndet[iout] = np.nan
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

    sangdV = np.zeros((indrpuT.shape[0],), dtype=float)
    dVndV = np.zeros((indrpuT.shape[0],), dtype=float)
    ndet = np.zeros((indrpuT.shape[0],), dtype=float)

    # ---------------
    # loop on cameras

    for kcam in lcam:
        v0 = doptics[kcam]

        # keys
        kindr, kindphi = v0['dvos']['ind_hor']
        ksang = v0['dvos']['sang_hor']
        kdV = v0['dvos']['dV_hor']
        kndV = v0['dvos']['ndV_hor']

        # data
        indr = coll.ddata[kindr]['data']
        indphi = coll.ddata[kindphi]['data']
        sangi = coll.ddata[ksang]['data']
        dVi = coll.ddata[kdV]['data']
        ndVi = coll.ddata[kndV]['data']

        iok = (indr >= 0)
        for ii, (ir, ip) in enumerate(indrpuT):
            ind = (indr == ir) & (indphi == ip) & iok
            if np.any(ind):
                sangdV[ii] += np.sum(sangi[ind] * dVi[ind])
                dVndV[ii] += np.sum(dVi[ind] * ndVi[ind])
                ndet[ii] += np.sum(ind)

    # ------------------
    # nan0
    # ------------------

    if nan0 is True:
        iout = ndet == 0
        sangdV[iout] = np.nan
        dVndV[iout] = np.nan
        ndet[iout] = np.nan
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
        'xx': xx,
        'yy': yy,
    }


# ################################################################
# ################################################################
#                       plot cross
# ################################################################


def _plot_cross(
    coll=None,
    key=None,
    lcam=None,
    # data
    ndet=None,
    dVndV=None,
    sangdV=None,
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
    dvminmax=None,
    # unused
    **kwdargs,
):

    # ----------------
    # check input
    # ----------------

    tit, cmap, _, _, dvminmax = _check_plot(
        proj='cross',
        key=key,
        tit=tit,
        cmap=cmap,
        dvminmax=dvminmax,
    )

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
        tstr = r"$\sum_{det_i}$"
        ax1.set_title(f"nb. of detectors {tstr}", size=14, fontweight='bold')

        # colorbar ndet
        cax_ndet = fig.add_subplot(gs[1:-1, 2*Na+Ni], frameon=False)

        # ax2 = dV
        ax2 = fig.add_subplot(
            gs[:, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax2.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax2.set_xlabel('R (m)', size=12, fontweight='bold')
        tstr = r"$\sum_{det_i} \sum_{V_i} dV$"
        ax2.set_title(
            f"Volume observed {tstr} (m3)",
            size=14,
            fontweight='bold',
        )

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
        tstr = r"$\sum_{det_i} \sum_{V_i} \Omega dV$"
        ax3.set_title(
            f"Integrated solid angle {tstr} (sr.m3)",
            size=14,
            fontweight='bold',
        )

        # colorbar
        cax_sang = fig.add_subplot(gs[1:-1, -1], frameon=False)

        dax.update({
            'span_cross': {'handle': ax0, 'type': 'span_cross'},
            'ndet_cross': {'handle': ax1, 'type': 'ndet_cross'},
            'cax_ndet_cross': {'handle': cax_ndet, 'type': 'cbar_ndet_cross'},
            'dV_cross': {'handle': ax2, 'type': 'dV_cross'},
            'cax_dV_cross': {'handle': cax_dphi, 'type': 'cbar_dV_cross'},
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
            vmin=dvminmax['ndet']['min'],
            vmax=dvminmax['ndet']['max'],
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

    if dVndV is not None:
        ktype = 'dV_cross'
        lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
        for ax in lax:

            # plot
            im = ax.imshow(
                dVndV.T,
                extent=extent,
                origin='lower',
                interpolation='bilinear',
                cmap=cmap,
                vmin=dvminmax['dV']['min'],
                vmax=dvminmax['dV']['max'],
            )

            # colorbar
            ktype2 = 'cbar_dV_cross'
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
            sangdV.T,
            extent=extent,
            origin='lower',
            interpolation='bilinear',
            cmap=cmap,
            vmin=dvminmax['sang']['min'],
            vmax=dvminmax['sang']['max'],
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
        for ktype in ['span_cross', 'ndet_cross', 'dV_cross', 'sang_cross']:
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
    dVndV=None,
    sangdV=None,
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
    dvminmax=None,
    # unused
    **kwdargs,
):

    # ----------------
    # check input
    # ----------------

    tit, cmap, marker, markersize, dvminmax = _check_plot(
        proj='hor',
        key=key,
        tit=tit,
        cmap=cmap,
        marker=marker,
        markersize=markersize,
        dvminmax=dvminmax,
    )

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
        tstr = r"$\sum_{det_i}$"
        ax1.set_title(f"nb. of detectors {tstr}", size=14, fontweight='bold')

        # colorbar ndet
        cax_ndet = fig.add_subplot(gs[1:-1, 2*Na+Ni], frameon=False)

        # ax2 = dV
        ax2 = fig.add_subplot(
            gs[:, 2*Na+2*Ni+Nc:3*Na+2*Ni+Nc],
            aspect='equal',
            sharex=ax0,
            sharey=ax0,
            adjustable='box',
        )
        ax2.set_ylabel('Z (m)', size=12, fontweight='bold')
        ax2.set_xlabel('R (m)', size=12, fontweight='bold')
        tstr = r"$\sum_{det_i} \sum_{V_i} dV$"
        ax2.set_title(
            f"Observed volume {tstr} (m3)",
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
        tstr = r"$\sum_{det_i} \sum_{V_i} \Omega dV$"
        ax3.set_title(
            f"Integrated solid angle {tstr} (sr.m3)",
            size=14,
            fontweight='bold',
        )

        # colorbar
        cax_sang = fig.add_subplot(gs[1:-1, -1], frameon=False)

        dax.update({
            'span_hor': {'handle': ax0, 'type': 'span_hor'},
            'ndet_hor': {'handle': ax1, 'type': 'ndet_hor'},
            'cax_ndet_hor': {'handle': cax_ndet, 'type': 'cbar_ndet_hor'},
            'dV_hor': {'handle': ax2, 'type': 'dV_hor'},
            'cax_dV_hor': {'handle': cax_dz, 'type': 'cbar_dV_hor'},
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
            vmin=dvminmax['ndet']['min'],
            vmax=dvminmax['ndet']['max'],
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

    ktype = 'dV_hor'
    lax = [v0['handle'] for v0 in dax.values() if ktype in v0['type']]
    for ax in lax:

        # plot
        im = ax.scatter(
            xx,
            yy,
            c=dVndV,
            s=markersize,
            marker=marker,
            cmap=cmap,
            vmin=dvminmax['dV']['min'],
            vmax=dvminmax['dV']['max'],
        )

        # colorbar
        ktype2 = 'cbar_dV_hor'
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
            c=sangdV,
            s=markersize,
            marker=marker,
            cmap=cmap,
            vmin=dvminmax['sang']['min'],
            vmax=dvminmax['sang']['max'],
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
        for ktype in ['span_hor', 'ndet_hor', 'dV_hor', 'sang_hor']:
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
#                       _check
# ################################################################


def _check_plot(
    proj=None,
    key=None,
    tit=None,
    cmap=None,
    dvminmax=None,
    # hor
    marker=None,
    markersize=None,
    # unused
    **kwdargs,
):

    # ---------------
    # tit
    # ---------------

    if tit is not False:
        if proj == 'cross':
            titdef = (
                f"geometrical coverage of diag '{key}' "
                "- integrated cross-section"
            )
        else:
            titdef = (
                f"geometrical coverage of diag '{key}' "
                "- integrated vertically"
            )
        tit = ds._generic_check._check_var(
            tit, 'tit',
            types=str,
            default=titdef,
        )

    # ---------------
    # cmap
    # ---------------

    if cmap is None:
        cmap = plt.cm.viridis   # Greys

    # -------------
    # markers
    # -------------

    if marker is None:
        marker = 's'

    if markersize is None:
        markersize = 8

    # ---------------
    # dvminmax
    # ---------------

    # -------
    # default

    if dvminmax is None:
        dvminmax = {}

    # ------------
    # sanity check

    lk = ['ndet', 'dV', 'sang']
    c0 = (
        isinstance(dvminmax, dict)
        and all([
            isinstance(v0, dict)
            and k0 in lk
            and (v0.get('min') is None or np.isscalar(v0['min']))
            and (v0.get('max') is None or np.isscalar(v0['max']))
            for k0, v0 in dvminmax.items()
        ])
    )
    if not c0:
        msg = (
            "Arg 'dvminmax' must be a dict with a "
            "{'min': float, 'max': float} subdict for each keys:\n"
            f"\t- 'ndet': {'min': float, 'max': float}\n"
            f"\t- 'dV': {'min': float, 'max': float}\n"
            f"\t- 'sang': {'min': float, 'max': float}\n"
            f"Provided:\n{dvminmax}\n"
        )
        raise Exception(msg)

    # ----------
    # completing

    for k0 in lk:
        if dvminmax.get(k0) is None:
            dvminmax[k0] = {}
        dvminmax[k0]['min'] = dvminmax[k0].get('min', 0.)
        dvminmax[k0]['max'] = dvminmax[k0].get('max')

    return (
        tit,
        cmap,
        marker,
        markersize,
        dvminmax,
    )
