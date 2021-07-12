
# Common
import numpy as np
import scipy.signal as scpsig
import scipy.interpolate as scpinterp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tofu.version import __version__


__all__ = [
    'get_localextrema_1d',
    'peak_analysis_spect1d',
    'plot_peak_analysis_spect1d',
]


_LTYPES = [int, float, np.int_, np.float_]

_GITHUB = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = f'tofu-{__version__}\treport issues / requests at {_GITHUB}'


###########################################################
###########################################################
#           hidden utilities
###########################################################


def _check_bool(var, vardef=None, varname=None):
    if var is None:
        var = vardef
    if not isinstance(var, bool):
        msg = (
            "Arg {} must be a bool\n".format(varname)
            + "  You provided: {}".format(type(var))
        )
        raise Exception(msg)
    return var


###########################################################
###########################################################
#
#           Preliminary
#       utility tools for 1d spectral analysis
#
###########################################################
###########################################################


def _get_localextrema_1d_check(
    data=None, lamb=None,
    weights=None, width=None,
    prom_rel=None, rel_height=None,
    method=None, returnas=None,
    return_minima=None,
    return_prominence=None,
    return_width=None,
):
    # data
    c0 = (
        isinstance(data, np.ndarray)
        and data.ndim in [1, 2]
        and data.size > 0
        and np.all(np.isfinite(data))
    )
    if not c0:
        msg = (
            "Arg data must be a (nlamb,) or (nt, nlamb) finite np.array!\n"
            + "\t- provided: {}\n".format(data)
        )
        raise Exception(msg)
    if data.ndim == 1:
        data = data[None, :]

    # lamb
    if lamb is None:
        lamb = np.arange(data.shape[1])
    c0 = (
        isinstance(lamb, np.ndarray)
        and lamb.shape == (data.shape[1],)
        and np.all(np.isfinite(lamb))
        and np.all(np.diff(lamb) > 0)
    )
    if not c0:
        msg = (
            "Arg lamb must be a finite increasing (data.shape[0],) np.array!\n"
            + "\t- provided: {}".format(lamb)
        )
        raise Exception(msg)

    # method
    if method is None:
        method = 'find_peaks'
    c0 = method in ['find_peaks', 'bspline']
    if not c0:
        msg = (
            "Arg method must be either:\n"
            + "\t- 'find_peaks': uses scipy.signal.find_peaks()\n"
            + "\t- 'bspline': uses bspline-fitting to find extrema\n"
            + "You provided:\n{}".format(method)
        )
        raise Exception(msg)

    # weights (for fitting, optional)
    c0 = (
        weights is None
        or (
            isinstance(weights, np.ndarray)
            and weights.shape == (data.shape[1],)
        )
    )
    if not c0:
        msg = (
            "Arg weights must be either None or a (nlamb,) np.array!\n"
            + "Fed to scipy.interpolate.UnivariateSpline(w=weights)\n"
            + "\t- provided: {}".format(weights)
        )
        raise Exception(msg)

    # width
    if width is None:
        if method == 'find_peaks':
            width = 0.
        else:
            width = False
    c0 = width is False or (type(width) in _LTYPES and width >= 0.)
    if not c0:
        msg = (
            "Arg width must be a float\n"
            + "width = estimate of the minimum line width\n"
            + "If method == 'find_peaks':\n"
            + "=>  used as minimum peak width\n"
            + "If method == 'bspline':\n"
            + "=>  Used to smooth the bsplines fitting\n"
            + "\t- False: spline fits all points\n"
            + "\t- float > 0: spline is smoothed\n"
            + "You provided: {}".format(width)
        )
        raise Exception(msg)
    if width is not False and method == 'find_peaks':
        width = int(np.ceil(width / np.nanmean(np.diff(lamb))))

    if rel_height is None:
        rel_height = 0.8
    if not (type(rel_height) in _LTYPES and 0 <= rel_height <= 1.):
        msg = (
            "Arg rel_height must be positive float in [0, 1]!\n"
            + "Provided: {}".format(rel_height)
        )
        raise Exception(msg)

    # returnas
    if returnas is None:
        returnas = float
    c0 = returnas in [bool, float]
    if not c0:
        msg = (
            "Arg returnas must be:\n"
            + "\t- bool: return 2 (nt, nlamb) bool arrays, True at extrema\n"
            + "\t- float: return 2 (nt, nn) float arrays, of extrema values\n"
            + "  You provided:\n{}".format(returnas)
        )
        raise Exception(msg)

    # return_minima, prominence, width
    return_minima = _check_bool(
        return_minima, vardef=False, varname='return_minima',
    )
    return_prominence = _check_bool(
        return_prominence, vardef=True, varname='return_prominence',
    )
    return_width = _check_bool(
        return_width, vardef=True, varname='return_width',
    )
    prom_rel = _check_bool(
        prom_rel, vardef=True, varname='prom_rel',
    )

    return (
        data, lamb, weights, width, prom_rel, rel_height, method,
        returnas, return_minima, return_prominence, return_width,
    )


def get_localextrema_1d(
    data=None, lamb=None,
    width=None, weights=None,
    prom_rel=None, rel_height=None,
    distance=None,
    method=None, returnas=None,
    return_minima=None,
    return_prominence=None,
    return_width=None,
):
    """ Automatically find peaks in spectrum """

    # ------------------
    #   check inputs
    (
        data, lamb, weights,
        width, prom_rel, rel_height, method,
        returnas, return_minima,
        return_prominence, return_width
    ) = _get_localextrema_1d_check(
        data=data, lamb=lamb,
        weights=weights, width=width,
        prom_rel=prom_rel, rel_height=rel_height,
        method=method, returnas=returnas,
        return_minima=return_minima,
        return_prominence=return_prominence,
        return_width=return_width,
    )

    # -----------------
    #   fit and extract extrema
    mini, prom, widt = None, None, None
    minima, prominence, widths = None, None, None
    if method == 'find_peaks':

        # find_peaks
        maxi = np.zeros(data.shape, dtype=bool)
        if return_prominence is True:
            prom = np.full(data.shape, np.nan)
        if return_width is True:
            widt = np.full(data.shape, np.nan)

        for ii in range(data.shape[0]):
            peaks, prop = scpsig.find_peaks(
                data[ii, :], height=None, threshold=None,
                distance=distance, prominence=0,
                width=width, wlen=None, rel_height=rel_height,
                plateau_size=None,
            )
            maxi[ii, peaks] = True
            if return_prominence is True:
                prom[ii, peaks] = prop['prominences']
            if return_width is True:
                widt[ii, peaks] = prop['widths']

        if return_minima is True:
            mini = np.zeros(data.shape, dtype=bool)
            for ii in range(data.shape[0]):
                peaks, prop = scpsig.find_peaks(
                    -data[ii, :], height=None, threshold=None,
                    distance=distance, prominence=None,
                    width=width, wlen=None, rel_height=rel_height,
                    plateau_size=None,
                )
                mini[ii, peaks] = True

        #   reshape
        if returnas is float:
            nmax = np.sum(maxi, axis=1)
            if return_prominence is True:
                prominence = np.full((data.shape[0], np.max(nmax)), np.nan)
                for ii in range(data.shape[0]):
                    prominence[ii, :nmax[ii]] = prom[ii, maxi[ii, :]]
            if return_width is True:
                widths = np.full((data.shape[0], np.max(nmax)), np.nan)
                for ii in range(data.shape[0]):
                    widths[ii, :nmax[ii]] = widt[ii, maxi[ii, :]]

            maxima = np.full((data.shape[0], np.max(nmax)), np.nan)
            for ii in range(data.shape[0]):
                maxima[ii, :nmax[ii]] = lamb[maxi[ii, :]]
            if return_minima is True:
                nmax = np.sum(mini, axis=1)
                minima = np.full((data.shape[0], np.max(nmax)), np.nan)
                for ii in range(data.shape[0]):
                    minima[ii, :nmax[ii]] = lamb[mini[ii, :]]

        else:
            maxima = maxi
            minima = mini
            prominence = prom
            widths = widt

        if prominence is not None and prom_rel is True:
            prominence = prominence / np.nanmax(data, axis=1)[:, None]

    else:

        # bspline
        prominence, widths = None, None
        bbox = [lamb.min(), lamb.max()]
        mini, maxi = [], []
        if width is False:
            for ii in range(data.shape[0]):
                bs = scpinterp.UnivariateSpline(
                    lamb, data[ii, :], w=weights,
                    bbox=bbox, k=4,
                    s=0, ext=2,
                    check_finite=False,
                )
                extrema = bs.derivative(1).roots()
                indmin = bs.derivative(2)(extrema) > 0.
                indmax = bs.derivative(2)(extrema) < 0.
                mini.append(extrema[indmin])
                maxi.append(extrema[indmax])
        else:
            nint = int(np.ceil((bbox[1]-bbox[0]) / (1.1*width)))
            delta = (bbox[1]-bbox[0]) / nint
            nknots = nint - 1
            knots = np.linspace(bbox[0]+delta, bbox[1]-delta, nknots)
            for ii in range(data.shape[0]):
                bs = scpinterp.LSQUnivariateSpline(
                    lamb, data[ii, :], t=knots,
                    w=weights,
                    bbox=bbox, k=4, ext=2,
                    check_finite=False,
                )
                extrema = bs.derivative(1).roots()
                indmin = bs.derivative(2)(extrema) > 0.
                indmax = bs.derivative(2)(extrema) < 0.
                mini.append(extrema[indmin])
                maxi.append(extrema[indmax])

        #   reshape
        if returnas is bool:
            bins = 0.5*(lamb[1:] + lamb[:-1])
            bins = np.r_[
                bins[0]-(lamb[1]-lamb[0]),
                bins,
                bins[-1]+(lamb[-1]-lamb[-2]),
            ]
            minima = np.zeros(data.shape, dtype=bool)
            maxima = np.zeros(data.shape, dtype=bool)
            for ii in range(data.shape[0]):
                if len(mini[ii]) > 0:
                    minima[
                        ii,
                        np.digitize(mini[ii], bins, right=False)-1
                    ] = True
                if len(maxi[ii]) > 0:
                    maxima[
                        ii,
                        np.digitize(maxi[ii], bins, right=False)-1
                    ] = True
        else:
            nmin = np.max([len(mm) for mm in mini])
            nmax = np.max([len(mm) for mm in maxi])
            minima = np.full((data.shape[0], nmin), np.nan)
            maxima = np.full((data.shape[0], nmax), np.nan)
            for ii in range(data.shape[0]):
                minima[ii, :len(mini[ii])] = mini[ii]
                maxima[ii, :len(maxi[ii])] = maxi[ii]

    # ---------------
    # return
    lout = [
        (minima, return_minima),
        (prominence, return_prominence),
        (widths, return_width),
    ]
    out = [maxima] + [aa for aa, vv in lout if vv is True]
    if len(out) == 1:
        out = out[0]
    return out


def peak_analysis_spect1d(
    intervals=None,
    data=None, lamb=None, t=None,
    thresh_prom=None,
    thresh_time=None,
    thresh_time_higher=None,
    thresh_lines_prom=None,
    groupby=None,
    distance=None,
    width=None, weights=None,
    prom_rel=None, rel_height=None,
    method=None, returnas=None,
    return_minima=None,
    return_prominence=None,
    return_width=None,
):

    # ------------------
    #   check inputs
    (
        data, lamb, weights,
        width, prom_rel, rel_height, method,
        returnas, return_minima,
        return_prominence, return_width
    ) = _get_localextrema_1d_check(
        data=data, lamb=lamb,
        weights=weights, width=width,
        prom_rel=prom_rel, rel_height=rel_height,
        method=method, returnas=returnas,
        return_minima=return_minima,
        return_prominence=return_prominence,
        return_width=return_width,
    )

    # time
    if t is None:
        t = np.arange(0, data.shape[0])

    # intervals
    if intervals is None:
        intervals = [(-np.inf, np.inf)]

    # thresholds
    if thresh_prom is None:
        thresh_prom = 0.
    if thresh_time is None:
        thresh_time = 0.
    if thresh_time_higher is None:
        thresh_time_higher = True
    if thresh_lines_prom is None:
        thresh_lines_prom = 0.
    if groupby is None:
        groupby = False

    # Get peaks and prominences
    maxi, prom = get_localextrema_1d(
        data=data,
        lamb=lamb,
        distance=distance,
        prom_rel=True,
        returnas=bool,
        return_minima=False,
        return_prominence=True,
        return_width=False,
    )

    # by-interval
    prom0 = np.copy(prom)
    dt = np.mean(np.diff(t))

    prom0[prom0 < thresh_prom] = np.nan
    # if thresh_prom is not None:
    # for ii, (i0, i1) in enumerate(intervals):
    # ind = (t >= i0) & (t <= i1)
    # prom0[ind[:, None] & np.all(prom0[ind, :]< thresh_prom, axis=0)] = np.nan

    if thresh_time is not None:
        fract_time = (~np.isnan(prom0))
        for ii, (i0, i1) in enumerate(intervals):
            deltai = (intervals[ii][1]-intervals[ii][0])/dt
            ind = (t >= i0) & (t <= i1)
            indi = np.sum(fract_time[ind, :], axis=0) > thresh_time*ind.sum()
            if thresh_time_higher is False:
                indi = ~indi
            fract_time[ind, :] = fract_time[ind, :] & indi[None, :]
        prom0[~fract_time] = np.nan

    hist = np.zeros((len(intervals), data.shape[1]), dtype=float)
    for ii, (i0, i1) in enumerate(intervals):
        ind = (t >= i0) & (t <= i1)
        hist[ii, :] = np.nansum(prom0[ind, :], axis=0)

    if groupby is not False:
        hist0 = np.copy(hist)
        for ii in range(hist0.shape[0]):
            cont = True
            ic = 0
            while cont:
                inds = np.argsort(hist0[ii, :])[::-1]
                for kk, jj in enumerate(inds):
                    i0 = np.arange(max(0, jj-groupby), jj)
                    i1 = np.arange(jj+1, min(jj+groupby+1, hist0.shape[1]))
                    c0 = (
                        hist0[ii, jj] > 0.
                        and (
                            np.any(hist0[ii, i0] > 0.)
                            or np.any(hist0[ii, i1] > 0.)
                        )
                    )
                    if c0:
                        hist0[ii, i0] = 0.
                        hist0[ii, i1] = 0.
                        ic += 1
                        break
                    else:
                        pass
                    if jj == inds[-1]:
                        cont = False
        lines = [
            lamb[hist0[ii, :] > thresh_lines_prom]
            for ii in range(hist0.shape[0])
        ]
    else:
        lines = [
            lamb[hist[ii, :] > thresh_lines_prom]
            for ii in range(hist.shape[0])
        ]

    # Return
    danalysis = {
        'data': data, 'lamb': lamb, 't': t,
        'maxi': maxi, 'prom': prom,
        'intervals': intervals, 'hist': hist, 'lines': lines,
        'thresh_prom': thresh_prom,
        'thresh_time': thresh_time,
        'thresh_lines_prom': thresh_lines_prom,
    }
    return danalysis


def plot_peak_analysis_spect1d(
    danalysis,
    sharey=None,
    dax=None,
    cmap=None, vmin=None, vmax=None,
    fs=None, dmargin=None,
    tit=None, wintit=None,
):

    # Prepare data
    # ------------
    nint = len(danalysis['intervals'])

    # Check plot inputs
    # ------------------

    if fs is None:
        fs = (14, 8)
    if tit is None:
        tit = False
    if wintit is None:
        wintit = _WINTIT
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.05, 'right': 0.99,
                   'bottom': 0.07, 'top': 0.92,
                   'wspace': 0.2, 'hspace': 0.3}
    if sharey is None:
        sharey = len(danalysis['lines']) > 1

    if dax is None:
        fig = plt.figure(figsize=fs)
        if tit is not False:
            fig.suptitle(tit, size=14, fontweight='bold')

        gs = gridspec.GridSpec(1, 2, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0])
        sharey = ax0 if sharey else None
        ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=sharey)

        ax0.set_xlabel(r'$\lambda$')
        ax0.set_ylabel(r'$t$')
        ax1.set_xlabel(r'$\lambda$')
        ax0.set_title('Peaks prominence', size=14, fontweight='bold')
        ax1.set_title('Peaks per interval', size=14, fontweight='bold')

        dax = {
            'peaks': ax0,
            'hist': ax1,
        }

    k0 = 'peaks'
    if dax.get(k0) is not None:
        prom0 = np.copy(danalysis['prom'])
        prom0[np.isnan(prom0)] = 0
        extent = (
            danalysis['lamb'].min(), danalysis['lamb'].max(),
            danalysis['t'].min(), danalysis['t'].max(),
        )
        dax[k0].imshow(
            prom0,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            extent=extent,
            interpolation='nearest',
            origin='lower',
            aspect='auto',
        )

    k0 = 'hist'
    if dax.get(k0) is not None:
        for ii, (i0, i1) in enumerate(danalysis['intervals']):
            indt = (danalysis['t'] >= i0) & (danalysis['t'] <= i1)
            if sharey is not None:
                intmin = np.min(danalysis['t'][indt])
                intdelta = (
                    danalysis['t'][indt].max() - danalysis['t'][indt].min()
                )
                ymax = np.nanmax(danalysis['hist'][ii, :])
                y = danalysis['hist'][ii, :] * (intdelta/ymax) + intmin
                ythr = (
                    danalysis['thresh_lines_prom'] * (intdelta/ymax) + intmin
                )
            else:
                intmin = 0.
                y = danalysis['hist'][ii, :]
                ythr = danalysis['thresh_lines_prom']
            dax[k0].plot(danalysis['lamb'], y, c='k', ls='-')
            dax[k0].axhline(intmin, c='k', ls='--')
            dax[k0].axhline(ythr, c='k', ls='--')

    return dax
