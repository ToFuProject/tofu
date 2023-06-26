# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# ###############################################################
# ###############################################################
#                   Binning of 2d spectro data
# ###############################################################


def spectro2d_binned(
    key=None,
    key_cam=None,
    # data to be binned
    data=None,
    statistic=None,
    # binning dimension
    bin_data0=None,
    bins0=None,
    bin_data1=None,
    bins1=None,
    # plotting
    plot=None,
    ax=None,
):

    # ------------
    # check

    (
        key,
        ddata,
        dbins0,
        dbins1,
        plot,
    ) = _spectro2d_binned_check(**locals())

    # ----------
    # prepare

    dbins0, dbins1 = _spectro2d_binned_prepare(
        coll=coll,
        key=key,
        # data
        ddata=ddata,
        # bins
        bin_data0=bin_data0,
        bins0=bins0,
        bin_data1=bin_data1,
        bins1=bins1,
    )

    # ----------
    # compute

    dout = {}
    for ii, kcam in enumerate(ddata.keys()):

        dati = ddata[kcam]['data']
        bind = bin_dd[kcam]['data']

        # binning index
        out = scpstats.binned_statistic(
            bind[iok],
            dati[iok],
            statistic=statistic,
            bins=dbin_edges[kcam],
        )[0]

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
            key=key,
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


def _spectro2d_binned_check(
    key=None,
    key_cam=None,
    # data to be binned
    data=None,
    # binning dimension
    bin_data0=None,
    bins0=None,
    bin_data1=None,
    bins1=None,
    # plotting
    plot=None,
):

    # --------
    # key

    lok = [k0 for k0, v0 in coll.dobj.get('diagnostic', {})]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']

    # ----------
    # data

    (
        ddata, dref,
        units, _, _,
    ) = coll.get_diagnostic_data(key=key, key_cam=key_cam, data=data)
    key_cam = list(ddata.keys())

    dref_cam = {k0: coll.dobj['camera'][k0]['dgeom']['ref'] for k0 in key_cam}

    # --------------
    # bin_data, bins

    bin_data0, bins0 = _check_bins(
        dref=dref,
        dref_cam=dref_cam,
        key_cam=key_cam,
        bin_data=bin_data0,
        bins=bins0,
    )

    if bin_data1 is not None:
        bin_data1, bins1 = _check_bins(
            dref=dref,
            dref_cam=dref_cam,
            key_cam=key_cam,
            bin_data=bin_data1,
            bins=bins1,
        )
        bin2 = True
    else:
        bin2d = False

    # --------
    # plot

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    return key, spectro, is2d, dbins0, dbins1, plot


def _check_bins(
    dref=None,
    dref_cam=None,
    key_cam=None,
    bin_data=None,
    bins=None,
):

    # --------------
    # bins

    if bins is None:
        bins = 100

    if np.isscalar(bins):
        bins = int(bins)

    else:
        bins = ds._generic_check._check_flat1d_array(
            bins, 'bins',
            unique=True,
        )

    # -------------
    # bin data

    lquant = ['etendue', 'amin', 'amax']  # 'los'
    lcomp = ['length', 'tangency radius', 'alpha']
    llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
    lok_fixed = ['x0', 'x1'] + lquant + lcomp + llamb

    lok_var = [
        k0 for k0, v0 in coll.dobj['diagnostic'][key]['signal']
        if
    ]

    if isinstance(bins, int):
        lok += []
    else:
        lok =

    bin_data = ds._generic_check._check_var(
        bin_data, 'bin_data',
        types=str,
        allowed=lok,
    )

    coll.dobj['camera'][key_cam[0]]['dgeom']['ref']
    return dbins


def _spectro2d_binned_prepare(
    coll=None,
    key=None,
    key_cam=None,
    # data
    data=None,
    # bins
    bin_data0=None,
    bins0=None,
    bin_data1=None,
    bins1=None,
):

    # --------
    # data

    (
        ddata, dref,
        units, _, _,
    ) = coll.get_diagnostic_data(key=key, key_cam=key_cam, data=data)

    # --------------
    # binning data 0

    dbins0 = _prepare_bins(
        coll=coll,
        key=key,
        key_cam=key_cam,
        bin_data=bin_data,
        bins=bins,
    )

    # --------------
    # binning data 1

    if bin_data1 is not None:
        bin_dd1, bin_dref1, bin_units1, dbin_edges1 = _prepare_bins(
            coll=coll,
            key=key,
            key_cam=key_cam,
            bin_data=bin_data1,
            bins=bins1,
        )

    else:
        dbins1 = None


    return dbins0, dbins1


def _prepare_bins(
    coll=None,
    key=None,
    key_cam=None,
    bin_data=None,
    bins=None,
):

    (
        bin_dd, bin_dref,
        bin_units, _, _,
    ) = coll.get_diagnostic_data(key=key, key_cam=key_cam, data=bin_data)

    # bins
    if isinstance(bins, int):
        dbin_edges = {}
        for kcam, v0 in bin_dd.items():
            bin_min = np.nanmin(v0['data'])
            bin_max = np.nanmax(v0['data'])
            dbin_edges[kcam] = np.linspace(bin_min, bin_max, bins + 1)

    else:
        dbin_edges = {
            kcam: np.r_[
                bins[0] - 0.5*(bins[1] - bins[0]),
                0.5*(bins[1:] + bins[:-1]),
                bins[-1] + 0.5*(bins[-1] - bins[-2]),
            ]
            for kcam in ddata.keys()
        }

    # ------------
    # format

    dbin = {
        ''
    }

    return dbins


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
