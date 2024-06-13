# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:12:43 2024

@author: dvezinet
"""


import warnings


import numpy as np
import astropy.units as asunits
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import datastock as ds


# ##################################################################
# ##################################################################
#             interpolate along los
# ##################################################################


def main(
    coll=None,
    key_diag=None,
    key_cam=None,
    key_integrand=None,
    key_coords=None,
    # sampling
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
    concatenate=None,
    # interpolating
    domain=None,
    val_out=None,
    # plotting
    vmin=None,
    vmax=None,
    plot=None,
    dcolor=None,
    dax=None,
):

    # ------------
    # check inputs
    # ------------

    (
        key_diag, key_cam, key_los,
        key_integrand, key_coords,
        key_bs_integrand, key_bs_coords,
        concatenate,
        lok_coords, segment, mode, radius_max,
        plot, dcolor,
    ) = _integrate_along_los_check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        key_integrand=key_integrand,
        key_coords=key_coords,
        # sampling
        segment=segment,
        mode=mode,
        radius_max=radius_max,
        concatenate=concatenate,
        # plotting
        plot=plot,
        dcolor=dcolor,
    )

    # --------------
    # prepare output
    # --------------

    dx = {}
    dy = {}
    dout = {}

    # key_los
    if key_los is not None:
        doptics = {k0: {'los': k0} for k0 in key_los}
    else:
        doptics = coll.dobj['diagnostic'][key_diag]['doptics']

    # ---------------
    # loop on cameras
    # ---------------

    axis_los = 0

    # -----------------------
    # only static coordinates

    if key_integrand in lok_coords and key_coords in lok_coords:

        for ii, kk in enumerate(key_cam):

            klos = doptics[kk]['los']
            if klos is None:
                continue

            # sample
            dx[kk], dy[kk] = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=concatenate,
                return_coords=[key_coords, key_integrand],
            )

    # -------------------------------
    # at least one static coordinate

    elif key_coords in lok_coords or key_integrand in lok_coords:

        # dispatch keys
        if key_coords in lok_coords:
            cll = key_coords
            c2d = key_integrand
            key_bs = key_bs_integrand
        else:
            cll = key_integrand
            c2d = key_coords
            key_bs = key_bs_coords

        for ii, kk in enumerate(key_cam):

            klos = doptics[kk]['los']
            if klos is None:
                continue

            # sample
            pts_x, pts_y, pts_z, pts_ll = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=concatenate,
                return_coords=['x', 'y', 'z', cll],
            )

            Ri = np.hypot(pts_x, pts_y)

            # interpolate
            q2d = coll.interpolate(
                keys=c2d,
                x0=Ri,
                x1=pts_z,
                ref_key=key_bs,
                grid=False,
                domain=domain,
                crop=True,
                nan0=True,
                val_out=val_out,
                return_params=None,
                store=False,
                inplace=False,
            )[c2d]

            # check shapes
            pts_ll, q2d, axis_los = _interpolate_along_los_reshape(
                xdata=pts_ll,
                ydata=q2d['data'],
                yref=q2d['ref'],
            )

            # initialize and fill
            dx[kk] = np.full(q2d.shape, np.nan)
            dy[kk] = np.full(q2d.shape, np.nan)

            isok = np.isfinite(q2d) & np.isfinite(Ri)
            if key_coords in lok_coords:
                dx[kk][isok] = pts_ll[isok]
                dy[kk][isok] = q2d[isok]
            else:
                dx[kk][isok] = q2d[isok]
                dy[kk][isok] = pts_ll[isok]

    # --------------------------------
    # both potentially dynamic 2d data

    else:
        for ii, kk in enumerate(key_cam):

            klos = doptics[kk]['los']
            if klos is None:
                continue

            # sample
            pts_x, pts_y, pts_z = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=concatenate,
                return_coords=['x', 'y', 'z'],
            )

            Ri = np.hypot(pts_x, pts_y)

            # interpolate
            q2dx = coll.interpolate(
                keys=key_coords,
                x0=Ri,
                x1=pts_z,
                ref_key=key_bs_coords,
                grid=False,
                domain=domain,
                crop=True,
                nan0=True,
                val_out=val_out,
                return_params=None,
                store=False,
                inplace=False,
            )[key_coords]['data']

            # interpolate
            q2dy = coll.interpolate_profile2d(
                keys=key_integrand,
                x0=Ri,
                x1=pts_z,
                ref_key=key_bs_integrand,
                grid=False,
                domain=domain,
                crop=True,
                nan0=True,
                val_out=val_out,
                return_params=None,
                store=False,
                inplace=False,
            )[key_integrand]['data']

            # check shape
            if q2dx['data'].shape != q2dy['data'].shape:
                msg = "The two interpolated quantities must have same shape!"
                raise Exception(msg)

            # prepare
            dx[kk] = np.full(q2d.shape, np.nan)
            dy[kk] = np.full(q2d.shape, np.nan)

            # isok
            isok = np.isfinite(q2dx) & np.isfinite(q2dy) & np.isfinite(Ri)
            dx[kk][isok] = q2dx[isok]
            dy[kk][isok] = q2dy[isok]

    # -----------------
    # safety check
    # -----------------

    lout = [k0 for k0, v0 in dx.items() if v0 is None]
    if len(lout) > 0:
        lstr = [f"\t- {k0}" for k0 in lout]
        msg = (
            "The following rays seem to have no existence in desired range:\n"
            + "\n".join(lstr)
            + "\nDetails:\n"
            f"\t- segment: {segment}\n"
            f"\t- radius_max: {radius_max}\n"
        )
        raise Exception(msg)

    # ------------
    # units
    # ------------

    ldist = ['x', 'y', 'z', 'R', 'l', 'ltot']
    lang = ['phi', 'ang_vs_ephi']

    # ---------
    # coords

    if key_coords in lok_coords:
        if key_coords in ldist:
            units_coords = asunits.Unit('m')
        elif key_coords in lang:
            units_coords = asunits.Unit('rad')
        else:
            units_coords = key_coords
    else:
        units_coords = coll.ddata[key_coords]['units']

    # ----------
    # integrand

    if key_integrand in lok_coords:
        if key_integrand in ldist:
            units_integrand = asunits.Unit('m')
        elif key_integrand in lang:
            units_integrand = asunits.Unit('rad')
        else:
            units_integrand = key_integrand
    else:
        units_integrand = coll.ddata[key_integrand]['units']

    # ---------------------
    # add indices
    # ---------------------

    dind = _get_dind(
        coll=coll,
        doptics=doptics,
        dx=dx,
        dy=dy,
    )

    # ------------
    # format
    # ------------

    dout['integrand'] = {
        'key': key_integrand,
        'ddata': dy,
        'units': units_integrand,
    }
    dout['coords'] = {
        'key': key_coords,
        'ddata': dx,
        'units': units_coords,
    }
    dout['ind'] = dind

    # ------------
    # plot
    # ------------

    if plot is True:
        return interpolate_along_los_plot(
            dout=dout,
            axis_los=axis_los,
            # plotting
            vmin=vmin,
            vmax=vmax,
            plot=plot,
            dcolor=dcolor,
            dax=dax,
        )
    else:
        return dout


# ################################################################
#                     Check
# ################################################################


def _integrate_along_los_check(
    coll=None,
    key_diag=None,
    key_cam=None,
    key_integrand=None,
    key_coords=None,
    # sampling
    segment=None,
    mode=None,
    radius_max=None,
    concatenate=None,
    # plotting
    plot=None,
    dcolor=None,
):

    # -----------------
    # keys of diag, cam
    # -----------------

    lrays = list(coll.dobj.get('rays', {}).keys())
    ldiag = list(coll.dobj.get('diagnostic', {}).keys())
    lc = [
            isinstance(key_diag, str)
            and key_diag in lrays
            and key_diag not in ldiag,
            isinstance(key_diag, list)
            and all([
                isinstance(kd, str)
                and kd in lrays
                and kd not in ldiag
                for kd in key_diag
            ]),
    ]

    if any(lc):
        if lc[0]:
            key_los = [key_diag]
        else:
            key_los = key_diag
        key_cam = key_los
    else:
        # key_cam
        key_diag, key_cam = coll.get_diagnostic_cam(
            key=key_diag,
            key_cam=key_cam,
            default='all',
        )
        key_los = None

    # -------------------------
    # keys of coords, integrand
    # -------------------------

    # ---------------------
    # key_data: coordinates

    lok_coords = [
        'x', 'y', 'z', 'R', 'phi', 'ang_vs_ephi',
        'k', 'l', 'ltot', 'itot',
    ]

    # -------------------
    # key_data: integrand

    dp2d = coll.get_profiles2d()
    lok_2d = list(dp2d.keys())
    key_integrand = ds._generic_check._check_var(
        key_integrand, 'key_integrand',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )

    # --------------------------
    # key_integrand: 2d or not ?

    if key_integrand in lok_2d:
        key_bs_integrand = dp2d[key_integrand]
    else:
        key_bs_integrand = None

    # ------------------
    # key_coords: check

    key_coords = ds._generic_check._check_var(
        key_coords, 'key_coords',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )

    # -----------------------
    # key_coords: 2d or not ?

    if key_coords in lok_2d:
        key_bs_coords = dp2d[key_coords]
    else:
        key_bs_coords = None

    # -----------------
    # Sampling parameters
    # -----------------

    # segment
    segment = ds._generic_check._check_var(
        segment, 'segment',
        types=int,
        default=-1,
    )

    # mode
    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='abs',
        allowed=['abs', 'rel'],
    )

    # radius_max
    if radius_max is None and mode == 'abs':

        if key_bs_integrand is None and key_bs_coords is None:
            pass

        else:
            wm = coll._which_mesh
            wbs = coll._which_bsplines

            rmax0, rmax1 = 0, 0
            if key_bs_integrand is not None:
                keym = coll.dobj[wbs][key_bs_integrand][wm]
                submesh = coll.dobj[wm][keym]['submesh']
                if submesh is not None:
                    keym = submesh
                knotsR = coll.dobj[wm][keym]['knots'][0]
                rmax0 = np.max(coll.ddata[knotsR]['data'])

            if key_bs_coords is not None:
                keym = coll.dobj[wbs][key_bs_coords][wm]
                submesh = coll.dobj[wm][keym]['submesh']
                if submesh is not None:
                    keym = submesh
                knotsR = coll.dobj[wm][keym]['knots'][0]
                rmax1 = np.max(coll.ddata[knotsR]['data'])

            radius_max = max(rmax0, rmax1)

    # -----------------
    # Plotting parameters
    # -----------------

    # ------
    # plot

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # -----------------
    # concatenate
    # -----------------

    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=plot,
    )

    if plot is True and concatenate is False:
        msg = "Arg concatenate must be True if plot is True"
        raise Exception(msg)

    # --------
    # dcolor
    # --------

    dcolor = _plot_check(
        dcolor=dcolor,
        key_diag=key_diag,
        key_cam=key_cam,
    )

    return (
        key_diag, key_cam, key_los,
        key_integrand, key_coords,
        key_bs_integrand, key_bs_coords,
        concatenate,
        lok_coords, segment, mode, radius_max,
        plot, dcolor,
    )


# ################################################################
#                     reshape
# ################################################################


def _interpolate_along_los_reshape(
    xdata=None,
    ydata=None,
    yref=None,
):

    axis_los = 0
    if xdata.shape != ydata.shape:

        ref = [ss if ss is None else 1 for ss in yref]

        # None matching xdata
        i0 = 0
        for ii in range(ydata.ndim):
            if ref[ii] is None:
                if ydata.shape[ii] == xdata.shape[i0]:
                    ref[ii] = xdata.shape[i0]
                    axis_los = ii
                    i0 += 1
                else:
                    ref[ii] = 1

        # None not macthing xdata (e.g.: domain)
        xdata = xdata.reshape(ref) * np.ones(ydata.shape)

    return xdata, ydata, axis_los


# ################################################################
#                     ind
# ################################################################


def _get_dind(
    coll=None,
    doptics=None,
    dx=None,
    dy=None,
):

    # ----------------
    #  loop on cameras

    dind = {}
    for kcam in doptics.keys():

        klos = doptics[kcam]['los']

        # ------------------
        # preliminary check

        inan = (
            np.isnan(dx[kcam])
            & np.isnan(dy[kcam])
        )

        # ------------------
        # preliminary check

        shape = coll.dobj['rays'][klos]['shape'][1:]
        nnan = np.prod(shape)
        if inan.sum() < nnan:
            msg = (
                f"cam '{kcam}' has unconsistent nb of nans:\n"
                f"\t- shape: {shape}\n"
                f"\t- inan.sum(): {inan.sum()}\n"
                f"\t- nnan: {nnan}\n"
            )
            warnings.warn(msg)
            return None

        # ------------------
        # preliminary check

        i10 = 0
        ind = np.full(dy[kcam].shape, -1, dtype=int)
        for i0, i1 in enumerate(inan.nonzero()[0]):
            ii = np.arange(i10, i1)
            ind[ii] = i0
            i10 = i1 + 1

        # -----------
        # safety check

        lc = [np.any(inan[ind>=0]), (ind == -1).sum() < nnan]
        if  any(lc):
            msg = (
                "Inconsistent nans!\n"
                f"\t- lc: {lc}\n"
                f"\t- ind: {ind}\n"
                f"\t- inan: {inan}\n"
                f"\t- isnan: {inan[ind>=0].sum()}\n"
                f"\t- nnan vs -1: {nnan} vs {(ind == -1).sum()}\n"
            )
            raise Exception(msg)

        dind[kcam] = ind

    return dind


# ################################################################
#                     PLOT
# ################################################################


def interpolate_along_los_plot(
    dout=None,
    axis_los=None,
    # plotting
    vmin=None,
    vmax=None,
    plot=None,
    dcolor=None,
    dax=None,
):


    # -------------
    # check inputs
    # -------------

    dcolor = _plot_check(
        dcolor=dcolor,
        key_diag=None,
        key_cam=list(dout['coords']['ddata'].keys()),
    )

    # ------------
    # prepare
    # -------------

    xlab = f"{dout['coords']['key']} ({dout['coords']['units']})"
    ylab = f"{dout['integrand']['key']} ({dout['integrand']['units']})"

    # -----------
    # make figure
    # -----------

    if dax is None:

        fig = plt.figure()

        ax = fig.add_axes([0.15, 0.1, 0.80, 0.8])

        tit = "LOS-interpolated"
        ax.set_title(tit, size=12, fontweight='bold')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        dax = {'main': ax}

    elif isinstance(dax, plt.Axes) or issubclass(dax.__class__, plt.Axes):
        dax = {'main': dax}

    # -------------
    # main
    # -------------

    kax = 'main'
    if dax.get(kax) is not None:
        ax = dax[kax]

        for k0 in dout['coords']['ddata'].keys():

            xx = dout['coords']['ddata'][k0]
            yy = dout['integrand']['ddata'][k0]
            if axis_los != 0:
                xx = xx.swapaxes(0, axis_los)
                yy = yy.swapaxes(0, axis_los)

            ax.plot(
                xx,
                yy,
                label=k0,
                **dcolor[k0],
            )

        ax.legend()

        if vmin is not None:
            ax.set_ylim(bottom=vmin)
        if vmax is not None:
            ax.set_ylim(top=vmax)

    return dout, dax


# #####################
# check inputs
# #####################


def _plot_check(
    dcolor=None,
    key_diag=None,
    key_cam=None,
):

    # ------------------
    # preliminary checks
    # ------------------

    if mcolors.is_color_like(dcolor):
        lc = [dcolor]
    else:
        lc = ['k', 'r', 'g', 'b', 'm', 'c']

    if not isinstance(dcolor, dict):
        dcolor = {
            kk: {
                'color': lc[ii % len(lc)],
                'ls': '-',
                'marker': '.',
                'lw': 1,
            }
            for ii, kk in enumerate(key_cam)
        }

    elif not any([kk in dcolor.keys() for kk in key_cam]):
        dcolor = {
            kk: {
                'color': dcolor.get('color', lc[ii % len(lc)]),
                'ls': dcolor.get('ls', '-'),
                'marker': dcolor.get('marker', '.'),
                'lw': dcolor.get('lw', 1),
                'ms': dcolor.get('ms', 6),
            }
            for ii, kk in enumerate(key_cam)
        }

    # ------------
    # check
    # ------------

    try:
        for ii, kk in enumerate(key_cam):
            dcolor[kk] = {
                'color': dcolor.get(kk, {}).get('color', lc[ii % len(lc)]),
                'ls': dcolor.get(kk, {}).get('ls', '-'),
                'lw': dcolor.get(kk, {}).get('lw', 1),
                'marker': dcolor.get(kk, {}).get('marker', '.'),
                'ms': dcolor.get(kk, {}).get('ms', 6),
            }

    except Exception as err:

        msg = (
            "Arg dcolor must be a dict with camera keys specifying plotting:\n"
            f"\t- camera keys for diag '{key_diag}': {key_cam}\n"
            "\t- expected values: dict of 'color', 'ls', 'lw', 'marker', ...\n"
            "\t{\n"
            "\t\t'color': ...,\n"
            "\t\t'ls': ...,\n"
            "\t\t'lw': ...,\n"
            "\t\t'marker': ...,\n"
            "\t\t'ms': ...,\n"
            "\t}\n"
            "\nProvided:\n{dcolor}\n\n\n"
        ) + str(err)
        raise Exception(msg)

    return dcolor