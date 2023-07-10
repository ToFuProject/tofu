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
        dout[kcam] = {
            'data': out,
            'units': units,
            'bins': 0.5*(dbin_edges[kcam][1:] + dbin_edges[kcam][:-1]),
            'bin_units': bin_dd[kcam]['units'],
        }

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
