
# Common
import numpy as np
import scipy.signal as scpsig


__all__ = ['get_localextrema_1d']


_LTYPES = [int, float, np.int_, np.float_]


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

    return (
        data, lamb, weights, width, method,
        returnas, return_minima, return_prominence, return_width,
    )


def get_localextrema_1d(
    data=None, lamb=None,
    width=None, weights=None,
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
        width, method,
        returnas, return_minima,
        return_prominence, return_width
    ) = _get_localextrema_1d_check(
        data=data, lamb=lamb,
        weights=weights, width=width,
        method=method, returnas=returnas,
        return_minima=return_minima,
        return_prominence=return_prominence,
        return_width=return_width,
    )

    # -----------------
    #   fit and extract extrema
    if method == 'find_peaks':

        # find_peaks
        maxi = np.zeros(data.shape, dtype=bool)
        if return_prominence is True:
            prom = [None for ii in range(data.shape[0])]
        if return_width is True:
            widt = [None for ii in range(data.shape[0])]

        for ii in range(data.shape[0]):
            peaks, prop = scpsig.find_peaks(
                data[ii, :], height=None, threshold=None,
                distance=None, prominence=0,
                width=width, wlen=None, rel_height=0.5,
                plateau_size=None,
            )
            maxi[ii, peaks] = True
            if return_prominence is True:
                prom[ii] = prop['prominences']
            if return_width is True:
                widt[ii] = prop['widths']

        nmax = np.sum(maxi, axis=1)
        if return_prominence is True:
            prominence = np.full((data.shape[0], np.max(nmax)), np.nan)
            for ii in range(data.shape[0]):
                prominence[ii, :nmax[ii]] = prom[ii]
        if return_width is True:
            widths = np.full((data.shape[0], np.max(nmax)), np.nan)
            for ii in range(data.shape[0]):
                widths[ii, :nmax[ii]] = widt[ii]

        if return_minima is True:
            mini = np.zeros(data.shape, dtype=bool)
            for ii in range(data.shape[0]):
                peaks, prop = scpsig.find_peaks(
                    -data[ii, :], height=None, threshold=None,
                    distance=None, prominence=None,
                    width=width, wlen=None, rel_height=0.5,
                    plateau_size=None,
                )
                mini[ii, peaks] = True

        #   reshape
        if returnas is float:
            nmax = np.sum(maxi, axis=1)
            maxima = np.full((data.shape[0], np.max(nmax)), np.nan)
            for ii in range(data.shape[0]):
                maxima[ii, :nmax[ii]] = lamb[maxi[ii, :]]
            if return_minima is True:
                nmax = np.sum(mini, axis=1)
                minima = np.full((data.shape[0], np.max(nmax)), np.nan)
                for ii in range(data.shape[0]):
                    minima[ii, :nmax[ii]] = lamb[mini[ii, :]]

    else:

        # bspline
        prominence, widths = None, None
        bbox = [lamb.min(), lamb.max()]
        mini, maxi = [], []
        if smooth is False:
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
            nint = int(np.ceil((bbox[1]-bbox[0]) / (1.5*smooth)))
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
                    minima[ii, np.digitize(mini[ii], bins, right=False)-1] = True
                if len(maxi[ii]) > 0:
                    maxima[ii, np.digitize(maxi[ii], bins, right=False)-1] = True
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
