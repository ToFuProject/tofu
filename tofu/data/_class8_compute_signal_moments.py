# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import scipy.stats as scpstats
import datastock as ds


# ###############################################################
# ###############################################################
#                   Binning of 2d spectro data
# ###############################################################


def binned(
    coll=None,
    key_diag=None,
    key_cam=None,
    # data to be binned
    data=None,
    statistic=None,
    # binning dimension
    bins0=None,
    bins1=None,
    bin_data0=None,
    bin_data1=None,
    # store
    store=None,
    # plotting
    plot=None,
    ax=None,
):

    # ------------
    # check

    (
        key_diag, key_cam, spectro, is2d,
        ddata, dref, units,
        dbins0, dbins1,
        statistic, store, plot,
    ) = _binned_check(**locals())

    # ----------
    # compute

    dout = {}
    for ii, kcam in enumerate(ddata.keys()):

        dati = ddata[kcam]
        bind = dbins0['ddata'][kcam]

        # binning index
        if dbins1 is None:

            for kcam in key_cam:
                dout[kcam] = coll.binning(
                    keys=data,
                    ref_key=dbins0['key'][''],
                    bins=dbins0['dedges'][kcam],
                    verb=False,
                    store=False,
                    returnas=None,
                    key_store=None,
                )

                import pdb; pdb.et_trace()      # DB

            iok = np.isfinite(dati) & np.isfinite(bind)
            out = scpstats.binned_statistic(
                bind[iok],
                dati[iok],
                statistic=statistic,
                bins=dbins0['dedges'][kcam],
            )

        else:
            out = scpstats.binned_statistic_2d(
                bind[iok],
                dati[iok],
                statistic=statistic,
                bins=[dbins0['dedges'][kcam], dbins1['dedges'][kcam]],
            )

        import pdb; pdb.et_trace()      # DB

        # fill dict
        # dout[kcam] = {
        #     'data': out,
        #     'units': units,
        #     'bins': 0.5*(dbin_edges[kcam][1:] + dbin_edges[kcam][:-1]),
        #     'bin_units': bin_dd[kcam]['units'],
        # }

    # ----------
    # plot

    if plot is True:
        ax = _spectro2d_binned_plot(
            key_diag=key_diag,
            # data to be binned
            data=None,
            # binning dimension
            bin_data=None,
            dout=dout,
            # plotting
            ax=ax,
        )

    # ----------
    # return

    if plot is True:
        return dout, ax
    else:
        return dout


def _binned_check(
    coll=None,
    key_diag=None,
    key_cam=None,
    # data to be binned
    data=None,
    # binning dimension
    bins0=None,
    bins1=None,
    bin_data0=None,
    bin_data1=None,
    statistic=None,
    # store
    store=None,
    # plotting
    plot=None,
    # others
    **kwdargs,
):

    # --------
    # key_diag

    key_diag, key_cam = coll.get_diagnostic_cam(key=key_diag, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # ----------
    # data

    (
        ddata, dref,
        units, _, _,
    ) = coll.get_diagnostic_data(key_diag=key_diag, key_cam=key_cam, data=data)
    key_cam = list(ddata.keys())

    dref_cam = {k0: coll.dobj['camera'][k0]['dgeom']['ref'] for k0 in key_cam}

    # --------------
    # bin_data, bins

    dbins0 = _check_bins(
        coll=coll,
        dref=dref,
        key_diag=key_diag,
        dref_cam=dref_cam,
        key_cam=key_cam,
        bin_data=bin_data0,
        bins=bins0,
    )

    if bin_data1 is not None:
        dbins1 = _check_bins(
            coll=coll,
            dref=dref,
            dref_cam=dref_cam,
            key_diag=key_diag,
            key_cam=key_cam,
            bin_data=bin_data1,
            bins=bins1,
        )
    else:
        dbins1 = None

    # --------
    # statistic

    statistic = ds._generic_check._check_var(
        statistic, 'statistic',
        types=str,
        default='sum',
    )

    # --------
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
    )

    # --------
    # plot

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    return (
        key_diag, key_cam, spectro, is2d,
        ddata, dref, units,
        dbins0, dbins1,
        statistic, store, plot,
    )



def _check_bins(
    coll=None,
    ddata=None,
    bin_data=None,
    bins=None,
    bin_units=None,
    
    # dref=None,
    # dref_cam=None,
    # key_diag=None,
    # key_cam=None,
    # bin_data=None,
    # bins=None,
    # bin_units=None,
    # # if bsplines
    # safety_ratio=None,
    # strict=None,
    # deg=None,
):

    # --------------
    # options

    # check
    strict = _generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=True,
    )

    # check
    safety_ratio = _generic_check._check_var(
        safety_ratio, 'safety_ratio',
        types=(int, float),
        default=1.5,
        sign='>0.'
    )

    # --------------
    # bins

    if bins is None:
        bins = 100

    if np.isscalar(bins):
        bins = int(bins)

    else:
        bins = _generic_check._check_flat1d_array(
            bins, 'bins',
            dtype=float,
            unique=True,
            can_be_None=False,
        )

    # -------------
    # bin data

    if isinstance(bin_data, str):

        # lquant = ['etendue', 'amin', 'amax']  # 'los'
        # lcomp = ['length', 'tangency radius', 'alpha']
        # llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
        # lok_fixed = ['x0', 'x1'] + lquant + lcomp + llamb

        lok_static = []
        lok_var = []
        for k0 in coll.dobj['diagnostic'][key_diag]['signal']:
            cams = coll.dobj['synth sig'][k0]['camera']
            ref = coll.ddata[coll.dobj['synth sig'][k0]['data'][0]]['ref']
            if ref == dref_cam[cams[0]]:
                lok_sig_static.append(k0)
            elif ref == dref[cams[0]]:
                lok_sig_var.append(k0)

        bin_key = ds._generic_check._check_var(
            bin_data, 'bin_data',
            types=str,
            allowed=lok_fixed + lok_sig_static + lok_sig_var,
        )
        bin_data = coll.ddata[bin_key]['data']
        bin_ref = coll.ddata[bin_key]['ref']
        bin_units = coll.ddata[bin_key]['units']

        variable = bin_data in lok_sig_var

    elif isinstance(bin_data, np.ndarray):
        bin_key = None
        shape = tuple([ss for ss in data_shape if ss in bin_data.shape])
        if bin_data.shape != shape:
            msg = "Arg bin_data must have "
            raise Exception(msg)

    elif bin_data is None:
        hasref, ref, bin_data, val, dkeys = coll.get_ref_vector_common(
            keys=keys,
            strategy=ref_vector_strategy,
        )
        if bin_data is None:
            msg = (
                f"No matching ref vector found for:\n"
                f"\t- keys: {keys}\n"
                f"\t- hasref: {hasref}\n"
                f"\t- ref: {ref}\n"
                f"\t- ddata['{keys[0]}']['ref'] = {coll.ddata[keys[0]]['ref']} "
            )
            raise Exception(msg)

    else:
        msg = f"Invalid bin_data:\n{bin_data}"
        raise Exception(msg)

    # ----------------
    # get axis

    if axis is None:
        if bin_key is None:
            axis = np.array([
                ii for ii, ss in enumerate(data_shape)
                if ss in bin_data.shape
            ])
        else:
            axis = np.array([
                ii for ii, rr in enumerate(data_ref)
                if rr in bin_ref
            ])

    axis = ds._generic_check._check_flat1d_array(
        axis, 'axis',
        dtype=int,
        unique=True,
        can_be_None=False,
        sign='>=0',
    )

    if np.any(axis > len(data_shape)-1):
        msg = f"axis too large\n{axis}"
        raise Exception(msg)

    if np.any(np.diff(axis) > 1):
        msg = f"axis must be adjacent indices!\n{axis}"
        raise Exception(msg)

    # --------------
    # bins

    # bins
    if isinstance(bins, int):
        bin_min = np.nanmin(bin_data)
        bin_max = np.nanmax(bin_data)
        bin_edges = np.linspace(bin_min, bin_max, bins + 1)

    else:
        bin_edges = np.r_[
            bins[0] - 0.5*(bins[1] - bins[0]),
            0.5*(bins[1:] + bins[:-1]),
            bins[-1] + 0.5*(bins[-1] - bins[-2]),
        ]

    # ----------
    # bin method

    if bin_data.ndim == 1:
        dv = np.abs(np.diff(bin_data))
        dv = np.append(dv, dv[-1])
        dvmean = np.mean(dv) + np.std(dv)

        if strict is True:
            lim = safety_ratio*dvmean
            if not np.mean(np.diff(bin_edges)) > lim:
                msg = (
                    f"Uncertain binning for '{sorted(keys)}', ref vect '{ref_key}':\n"
                    f"Binning steps ({db}) are < {safety_ratio}*ref ({lim}) vector step"
                )
                raise Exception(msg)

            else:
                npts = None

        else:
            npts = (deg + 3) * max(1, dvmean / db)

    # ----------
    # dbins

    dbins = {
        'key': bin_key,
        'bins': bins,
        'edges': edges,
        'data': bin_data,
        'ref': bin_ref,
        'units': bin_units,
        'axis': axis,
    }

    return dbins, npts




def _spectro2d_binned_plot(
    key=None,
    key_cam=None,
    # data to be binned
    data=None,
    # binning dimension
    bin_data=None,
    dout=None,
    # plotting
    ax=None,
):

    # ------------
    #



    return ax
