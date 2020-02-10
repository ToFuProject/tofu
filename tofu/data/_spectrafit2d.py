
# Built-in
import os
import warnings
import itertools as itt
import datetime as dtm      # DB

# Common
import numpy as np
import scipy.optimize as scpopt
import scipy.constants as scpct
import scipy.sparse as sparse
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


#                   --------------
#       TO BE MOVED TO tofu.data WHEN FINISHED !!!!
#                   --------------


_NPEAKMAX = 12

###########################################################
###########################################################
#
#           Preliminary
#       utility tools for 1d spectral fitting
#
###########################################################
###########################################################


def remove_bck(x, y):
    # opt = np.polyfit(x, y, deg=0)
    opt = [np.nanmin(y)]
    return y-opt[0], opt[0]


def get_peaks(x, y, nmax=None):

    if nmax is None:
        nmax = _NPEAKMAX

    # Prepare
    ybis = np.copy(y)
    A = np.empty((nmax,), dtype=y.dtype)
    x0 = np.empty((nmax,), dtype=x.dtype)
    sigma = np.empty((nmax,), dtype=y.dtype)
    def gauss(xx, A, x0, sigma): return A*np.exp(-(xx-x0)**2/sigma**2)
    def gauss_jac(xx, A, x0, sigma):
        jac = np.empty((xx.size, 3), dtype=float)
        jac[:, 0] = np.exp(-(xx-x0)**2/sigma**2)
        jac[:, 1] = A*2*(xx-x0)/sigma**2 * np.exp(-(xx-x0)**2/sigma**2)
        jac[:, 2] = A*2*(xx-x0)**2/sigma**3 * np.exp(-(xx-x0)**2/sigma**2)
        return jac

    dx = np.nanmin(np.diff(x))

    # Loop
    nn = 0
    while nn < nmax:
        ind = np.nanargmax(ybis)
        x00 = x[ind]
        if np.any(np.diff(ybis[ind:], n=2) >= 0.):
            wp = min(x.size-1,
                     ind + np.nonzero(np.diff(ybis[ind:],n=2)>=0.)[0][0] + 1)
        else:
            wp = ybis.size-1
        if np.any(np.diff(ybis[:ind+1], n=2) >= 0.):
            wn = max(0, np.nonzero(np.diff(ybis[:ind+1],n=2)>=0.)[0][-1] - 1)
        else:
            wn = 0
        width = x[wp]-x[wn]
        assert width>0.
        indl = np.arange(wn, wp+1)
        sig = np.ones((indl.size,))
        if (np.abs(np.mean(np.diff(ybis[ind:wp+1])))
            > np.abs(np.mean(np.diff(ybis[wn:ind+1])))):
            sig[indl < ind] = 1.5
            sig[indl > ind] = 0.5
        else:
            sig[indl < ind] = 0.5
            sig[indl > ind] = 1.5
        p0 = (ybis[ind], x00, width)#,0.)
        bounds = (np.r_[0., x[wn], dx/2.],
                  np.r_[5.*ybis[ind], x[wp], 5.*width])
        try:
            (Ai, x0i, sigi) = scpopt.curve_fit(gauss, x[indl], ybis[indl],
                                               p0=p0, bounds=bounds, jac=gauss_jac,
                                               sigma=sig, x_scale='jac')[0]
        except Exception as err:
            print(str(err))
            import ipdb
            ipdb.set_trace()
            pass

        ybis = ybis - gauss(x, Ai, x0i, sigi)
        A[nn] = Ai
        x0[nn] = x0i
        sigma[nn] = sigi


        nn += 1
    return A, x0, sigma

def get_p0bounds_all(x, y, nmax=None, lamb0=None):

    yflat, bck = remove_bck(x, y)
    amp, x0, sigma = get_peaks(x, yflat, nmax=nmax)
    lamb0 = x0
    nmax = lamb0.size

    p0 = amp.tolist() + [0 for ii in range(nmax)] + sigma.tolist() + [bck]

    lx = [np.nanmin(x), np.nanmax(x)]
    Dx = np.diff(lx)
    dx = np.nanmin(np.diff(x))

    bamp = (np.zeros(nmax,), np.full((nmax,),3.*np.nanmax(y)))
    bdlamb = (np.full((nmax,), -Dx/2.), np.full((nmax,), Dx/2.))
    bsigma = (np.full((nmax,), dx/2.), np.full((nmax,), Dx/2.))
    bbck0 = (0., np.nanmax(y))

    bounds = (np.r_[bamp[0], bdlamb[0], bsigma[0], bbck0[0]],
              np.r_[bamp[1], bdlamb[1], bsigma[1], bbck0[1]])
    if not np.all(bounds[0]<bounds[1]):
        msg = "Lower bounds must be < upper bounds !\n"
        msg += "    lower :  %s\n"+str(bounds[0])
        msg += "    upper :  %s\n"+str(bounds[1])
        raise Exception(msg)
    return p0, bounds, lamb0

def get_p0bounds_lambfix(x, y, nmax=None, lamb0=None):

    nmax = lamb0.size
    # get typical x units
    Dx = x.nanmax()-x.nanmin()
    dx = np.nanmin(np.diff(x))

    # Get background and background-subtracted y
    yflat, bck = remove_bck(x, y)

    # get initial guesses
    amp = [yflat[np.nanargmin(np.abs(x-lamb))] for lamb in lamb0]
    sigma = [Dx/nmax for ii in range(nmax)]
    p0 = A + sigma + [bck]

    # Get bounding boxes
    bamp = (np.zeros(nmax,), np.full((nmax,),3.*np.nanmax(y)))
    bsigma = (np.full((nmax,), dx/2.), np.full((nmax,), Dx/2.))
    bbck0 = (0., np.nanmax(y))

    bounds = (np.r_[bamp[0], bsigma[0], bbck0[0]],
              np.r_[bamp[1], bsigma[1], bbck0[1]])
    if not np.all(bounds[0]<bounds[1]):
        msg = "Lower bounds must be < upper bounds !\n"
        msg += "    lower :  %s\n"+str(bounds[0])
        msg += "    upper :  %s\n"+str(bounds[1])
        raise Exception(msg)
    return p0, bounds, lamb0


def get_func1d_all(n=5, lamb0=None):
    if lamb0 is None:
        lamb0 = np.zeros((n,), dtype=float)
    assert lamb0.size == n

    def func_vect(x, amp, dlamp, sigma, bck0, lamb0=lamb0, n=n):
        y = np.full((n+1, x.size), np.nan)
        y[:-1, :] = amp[:, None]*np.exp(-(x[None, :]-(lamb0+dlamb)[:, None])**2
                                        /sigma[:, None]**2)
        y[-1, :] = bck0
        return y

    def func_sca(x, *args, lamb0=lamb0, n=n):
        amp = np.r_[args[0:n]][:, None]
        dlamb = np.r_[args[n:2*n]][:, None]
        sigma = np.r_[args[2*n:3*n]][:, None]
        bck0 = np.r_[args[3*n]]
        gaus = amp * np.exp(-(x[None, :]-(lamb0[:, None] + dlamb))**2/sigma**2)
        back = bck0
        return np.sum(gaus, axis=0) + back

    def func_sca_jac(x, *args, lamb0=lamb0, n=n):
        amp = np.r_[args[0:n]][None, :]
        dlamb = np.r_[args[n:2*n]][None, :]
        sigma = np.r_[args[2*n:3*n]][None, :]
        bck0 = np.r_[args[3*n]]
        lamb0 = lamb0[None, :]
        x = x[:, None]
        jac = np.full((x.size, 3*n+1,), np.nan)
        jac[:, :n] = np.exp(-(x - (lamb0+dlamb))**2/sigma**2)
        jac[:, n:2*n] = amp*2*((x - (lamb0+dlamb))/(sigma**2)
                               * np.exp(-(x - (lamb0+dlamb))**2/sigma**2))
        jac[:, 2*n:3*n] = amp*2*((x - (lamb0+dlamb))**2/sigma**3
                                 * np.exp(-(x - (lamb0+dlamb))**2/sigma**2))
        jac[:, -1] = 1.
        return jac

    return func_vect, func_sca, func_sca_jac


def get_func1d_lamb0fix(n=5, lamb0=None):
    if lamb0 is None:
        lamb0 = np.zeros((n,), dtype=float)
    assert lamb0.size == n

    def func_vect(x, amp, sigma, bck0, lamb0=lamb0, n=n):
        y = np.full((n+1, x.size), np.nan)
        for ii in range(n):
            y[ii, :] = amp[ii]*np.exp(-(x-lamb0[ii])**2/sigma[ii]**2)
        y[-1, :] = bck0
        return y

    def func_sca(x, *args, lamb0=lamb0, n=n):
        amp = np.r_[args[0:n]][:, None]
        sigma = np.r_[args[2*n:3*n]][:, None]
        bck0 = np.r_[args[3*n]]
        gaus = amp * np.exp(-(x[None, :]-lamb0[:, None])**2/sigma**2)
        back = bck0
        return np.sum(gaus, axis=0) + back

    def func_sca_jac(x, *args, lamb0=lamb0, n=n):
        amp = np.r_[args[0:n]][None, :]
        sigma = np.r_[args[2*n:3*n]][None, :]
        bck0 = np.r_[args[3*n]]
        lamb0 = lamb0[None, :]
        x = x[:, None]
        jac = np.full((x.size, 2*n+1,), np.nan)
        jac[:, :n] = np.exp(-(x - lamb0)**2/sigma**2)
        jac[:, n:2*n] = amp*2*((x - lamb0)**2/sigma**3
                                 * np.exp(-(x-lamb0)**2/sigma**2))
        jac[:, -1] = 1.
        return jac

    return func_vect, func_sca, func_sca_jac


def multiplegaussianfit1d(x, spectra, nmax=None,
                          lamb0=None, forcelamb=None,
                          p0=None, bounds=None,
                          max_nfev=None, xtol=None, verbose=0,
                          percent=None, plot_debug=False):
    # Check inputs
    if xtol is None:
        xtol = 1.e-8
    if percent is None:
        percent = 20


    # Prepare
    if spectra.ndim == 1:
        spectra = spectra.reshape((1,spectra.size))
    nt = spectra.shape[0]

    # Prepare info
    if verbose is not None:
        print("----- Fitting spectra with {0} gaussians -----".format(nmax))
    nspect = spectra.shape[0]
    nstr = max(nspect//max(int(100/percent), 1), 1)

    # get initial guess function
    if forcelamb is True:
        get_p0bounds = get_p0bounds_lambfix
    else:
        get_p0bounds = get_p0bounds_all

    # lamb0
    if p0 is None or bounds is None or lamb0 is None:
        p00, bounds0, lamb00 = get_p0bounds_all(x, spectra[0,:],
                                              nmax=nmax, lamb0=lamb0)
        if lamb0 is None:
            lamb0 = lamb00
        assert lamb0 is not None
        if forcelamb is True:
            p00 = p00[:nmax] + p00[2*nmax:]
            bounds0 = bounds0[:nmax] + bounds0[2*nmax:]
        if p0 is None:
            p0 = p00
        if bounds is None:
            bounds = bounds0
    if nmax is None:
        nmax = lamb0.size
    assert nmax == lamb0.size

    # Get fit vector, scalar and jacobian functions
    if forcelamb is True:
        func_vect, func_sca, func_sca_jac = get_func1d_lambfix(n=nmax,
                                                               lamb0=lamb0)
    else:
        func_vect, func_sca, func_sca_jac = get_func1d_all(n=nmax,
                                                           lamb0=lamb0)

    # Prepare index for splitting p0
    if forcelamb is True:
        indsplit = nmax*np.r_[1, 2]
    else:
        indsplit = nmax*np.r_[1, 2, 3]

    # Prepare output
    fit = np.full(spectra.shape, np.nan)
    amp = np.full((nt, nmax), np.nan)
    sigma = np.full((nt, nmax), np.nan)
    bck = np.full((nt,), np.nan)
    ampstd = np.full((nt, nmax), np.nan)
    sigmastd = np.full((nt, nmax), np.nan)
    bckstd = np.full((nt,), np.nan)
    if not forcelamb is True:
        dlamb = np.full((nt, nmax), np.nan)
        dlambstd = np.full((nt, nmax), np.nan)
    else:
        dlamb, dlambstd = None, None

    # Loop on spectra
    lch = []
    for ii in range(0, nspect):

        if verbose is not None and ii%nstr==0:
            print("=> spectrum {0} / {1}".format(ii+1, nspect))

        try:
            popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                          jac=func_sca_jac,
                                          p0=p0, bounds=bounds,
                                          max_nfev=max_nfev, xtol=xtol,
                                          x_scale='jac',
                                          verbose=verbose)
        except Exception as err:
            msg = "    Convergence issue for {0} / {1}\n".format(ii+1, nspect)
            msg += "    => %s\n"%str(err)
            msg += "    => Resetting initial guess and bounds..."
            print(msg)
            try:
                p0, bounds, _ = get_p0bounds(x, spectra[ii,:],
                                             nmax=nmax, lamb0=lamb0)
                popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                              jac=func_sca_jac,
                                              p0=p0, bounds=bounds,
                                              max_nfev=max_nfev, xtol=xtol,
                                              x_scale='jac',
                                              verbose=verbose)
                p0 = popt
                popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                              jac=func_sca_jac,
                                              p0=p0, bounds=bounds,
                                              max_nfev=max_nfev, xtol=xtol,
                                              x_scale='jac',
                                              verbose=verbose)
                lch.append(ii)
            except Exception as err:
                print(str(err))
                import ipdb
                ipdb.set_trace()
                raise err

        out = np.split(popt, indsplit)
        outstd = np.split(np.sqrt(np.diag(pcov)), indsplit)
        if forcelamb is True:
            amp[ii, :], sigma[ii, :], bck[ii] = out
            ampstd[ii, :], sigmastd[ii, :], bckstd[ii] = outstd
        else:
            amp[ii, :], dlamb[ii, :], sigma[ii, :], bck[ii] = out
            ampstd[ii,:], dlambstd[ii,:], sigmastd[ii,:], bckstd[ii] = outstd
        fit[ii, :] = func_sca(x, *popt)
        p0[:] = popt[:]

        if plot_debug and ii in [0,1]:
            fit = func_vect(x, amp[ii,:], x0[ii,:], sigma[ii,:], bck0[ii])

            plt.figure()
            ax0 = plt.subplot(2,1,1)
            ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)
            ax0.plot(x,spectra[ii,:], '.k',
                     x, np.sum(fit, axis=0), '-r')
            ax1.plot(x, fit.T)

    std = np.sqrt(np.sum((spectra-fit)**2, axis=1))

    dout = {'fit': fit, 'lamb0': lamb0, 'std': std, 'lch': lch,
            'amp': amp, 'ampstd': ampstd,
            'sigma': sigma, 'sigmastd': sigmastd,
            'bck': bck, 'bckstd': bckstd,
            'dlamb': dlamb, 'dlambstd': dlambstd}

    return dout


###########################################################
###########################################################
#
#           1d spectral fitting from dlines
#
###########################################################
###########################################################


def multigausfit1d_from_dlines_ind(dions=None,
                                   double=None,
                                   Ti=None, vi=None):
    """ Return the indices of quantities in x to compute y """

    # Prepare lines concatenation
    nions = len(dions)
    llines = [v0['lamb'] for v0 in dions.values()]
    mz = np.array([v0['m'][0] for v0 in dions.values()])
    lkeys = [v0['key'] for v0 in dions.values()]
    lines = np.concatenate(llines)
    keys = list(itt.chain.from_iterable(lkeys))
    lnlines = np.array([ll.size for ll in llines])
    nlines = lines.size

    # indices
    # General shape: [bck, widths, shifts, amp]
    # If double [..., double_shift, double_ratio]
    # Excpet for bck, all indices should render nlines (2*nlines if double)
    indbck = np.r_[0]
    inddratio, inddshift = None, None
    if Ti is False and vi is False:
        indw = 1 + np.arange(0, nlines)
        indw_lines = indw
        inds = indw + nlines
        inds_lines = inds
        inda = inds + nlines
        inda_lines = inda
        sizex = 1 + 3*nlines
    elif Ti is True and vi is False:
        indw = 1 + np.arange(0, nions)
        indw_lines = np.repeat(indw, lnlines)
        inds = 1 + nions + np.arange(0, nlines)
        inds_lines = inds
        inda = inds + nlines
        inda_lines = inda
        sizex = 1 + nions + 2*nlines
    elif Ti is False and vi is True:
        indw = 1 + np.arange(0, nlines)
        indw_lines = indw
        inds = 1 + nlines + np.arange(0, nions)
        inds_lines = np.repeat(inds, lnlines)
        inda = indw + nlines + nions
        inda_lines = inda
        sizex = 1 + nions + 2*nlines
    else:
        indw = 1 + np.arange(0, nions)
        indw_lines = np.repeat(indw, lnlines)
        inds = indw + nions
        inds_lines = indw_lines + nions
        inda = 1 + 2*nions + np.arange(0, nlines)
        inda_lines = inda
        sizex = 1 + 2*nions + nlines
    shapey0 = 1 + nlines

    # index to get back unique ions values from width and shift
    indions = np.repeat(np.arange(0, nions), lnlines)
    indions_back = np.r_[0, np.cumsum(lnlines[:-1])]

    if double:
        inddshift = -2
        inddratio = -1
        sizex += 2

    # Indices for jacobian
    if Ti is True:
        indw_jac = indions_back
    else:
        indw_jac = np.arange(0, nlines)

    if vi is True:
        inds_jac = indions_back
    else:
        inds_jac = np.arange(0, nlines)

    dind = {'bck': indbck,
            'width': indw, 'shift': inds, 'amp': inda,
            'width_lines': indw_lines, 'shift_lines': inds_lines,
            'amp_lines': inda_lines, 'width_jac': indw_jac,
            'shift_jac': inds_jac,
            'dratio': inddratio, 'dshift': inddshift,
            'ions': indions, 'ions_back': indions_back,
            }
    return dind, lines, mz, keys, sizex, shapey0

def multigausfit1d_from_dlines_scale(data, lamb):
    Dlamb = lamb[-1]-lamb[0]
    lambm = np.nanmin(lamb)
    dscale = {'bck': np.nanmean(data)/2., 'amp': np.nanmax(data),
              'width': (Dlamb/(20*lambm))**2, 'shift': Dlamb/(10*lambm)}
    return dscale

def multigausfit1d_from_dlines_x0(sizex, dind,
                                  lines=None, data=None, lamb=None,
                                  dscale=None, double=None):
    # Each x0 should be understood as x0*scale
    x0_scale = np.full((sizex,), np.nan)
    x0_scale[dind['bck']] = 0.2
    amp0 = data[np.searchsorted(lamb, lines)]
    x0_scale[dind['amp']] = amp0 / dscale['amp']
    x0_scale[dind['width']] = 0.4
    x0_scale[dind['shift']] = 0.
    if double is True:
        x0_scale[dind['dratio']] = 1.
        x0_scale[dind['dshift']] = 0.
    return x0_scale

def multigausfit1d_from_dlines_bounds(sizex=None, dind=None, double=None):
    # Each x0 should be understood as x0*scale
    xup = np.full((sizex,), np.nan)
    xlo = np.full((sizex,), np.nan)
    xup[dind['bck']] = 1.
    xlo[dind['bck']] = 0.
    xup[dind['amp']] = 1
    xlo[dind['amp']] = 0.
    xup[dind['width']] = 1.
    xlo[dind['width']] = 0.01
    xup[dind['shift']] = 1.
    xlo[dind['shift']] = -1.
    if double is True:
        xup[dind['dratio']] = 1.6
        xlo[dind['dratio']] = 0.4
        xup[dind['dshift']] = 1.
        xlo[dind['dshift']] = -1.
    bounds_scale = (xlo, xup)
    return bounds_scale

def multigausfit1d_from_dlines_funccostjac(data, lamb,
                                           dind=None,
                                           dscale=None,
                                           lines=None,
                                           shapey0=None,
                                           double=None,
                                           jac=None):
    if dscale is None:
        bckscale = 1.
        ampscale = 1.
        shscale = 1.
        wscale = 1.
    else:
        bckscale = dscale['bck']
        ampscale = dscale['amp']
        shscale = dscale['shift']
        wscale = dscale['width']

    indbck = dind['bck']
    inda = dind['amp']
    indw = dind['width']
    inds = dind['shift']
    inddratio = dind['dratio']
    inddshift = dind['dshift']

    indal = dind['amp_lines']
    indwl = dind['width_lines']
    indsl = dind['shift_lines']

    indwj = dind['width_jac']
    indsj = dind['shift_jac']

    lines = lines[None, :]
    lamb = lamb[:, None]
    shape = (lamb.size, shapey0)
    # w2 with Ti in eV -> J
    # w2 = 2*scpct.k/scpct.c**2

    def func_detail(x, lamb=lamb, lines=lines, shape=shape,
                    indbck=indbck, indal=indal, indwl=indwl, indsl=indsl,
                    inddratio=inddratio, inddshift=inddshift,
                    bckscale=bckscale, ampscale=ampscale,
                    shscale=shscale, wscale=wscale, double=double):
        y = np.full(shape, np.nan)
        y[:, 0] = x[indbck] * bckscale

        # lines
        amp = ampscale*x[indal][None, :]
        shifti = shscale*x[indsl][None, :]
        wi2 = wscale*x[indwl][None, :]
        y[:, 1:] = amp * np.exp(-(lamb/lines - (1 + shifti))**2 / (2*wi2))

        if double is True:
            ampd = ampscale*(x[indal]*x[inddratio])[None, :]
            shiftid = shscale*(x[indsl]+x[inddshift])[None, :]
            y[:, 1:] += (ampd
                         * np.exp(-(lamb/lines - (1 + shiftid))**2 / (2*wi2)))
        return y

    def func(x):
        return np.sum(func_detail(x), axis=1)

    def cost_scale(x, data=data,
                   bckscale=bckscale, ampscale=ampscale,
                   wscale=wscale, shscale=shscale):
        return (np.sum(func_detail(x,
                                   bckscale=bckscale,
                                   ampscale=ampscale,
                                   wscale=wscale,
                                   shscale=shscale), axis=1) - data)

    if jac == 'call':
        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)
        def jac_scale(x, lamb=lamb, lines=lines,
                      indbck=indbck, indal=indal, indwl=indwl,
                      indsl=indsl, inddratio=inddratio, inddshift=inddshift,
                      inda=inda, indw=indw, inds=inds,
                      indwj=indwj, indsj=indsj,
                      bckscale=bckscale, ampscale=ampscale,
                      shscale=shscale, wscale=wscale, double=double):
            jac = np.full((lamb.size, x.size), np.nan)
            jac[:, 0] = bckscale

            # Assuming Ti = False and vi = False
            amp = ampscale*x[indal][None, :]
            wi2 = x[indwl][None, :] * wscale
            shifti = x[indsl][None, :] * shscale
            beta = (lamb/lines - (1 + shifti)) / (2*wi2)
            alpha = -beta**2 * (2*wi2)
            exp = np.exp(alpha)

            jac[:, inda] = ampscale * exp
            jac[:, indw] = np.add.reduceat(
                amp * (-alpha/x[indwl][None, :]) * exp, indwj, axis=1)
            jac[:, inds] = np.add.reduceat(amp * 2.*beta*shscale * exp,
                                            indsj, axis=1)
            if double is True:
                # Assuming Ti = False and vi = False
                ampd = ampscale*x[indal][None, :]*x[inddratio]
                shiftid = (x[indsl][None, :] + x[inddshift]) * shscale
                betad = (lamb/lines - (1 + shiftid)) / (2*wi2)
                alphad = -betad**2 * (2*wi2)
                expd = np.exp(alphad)

                jac[:, inda] += ampscale * x[inddratio] * expd
                jac[:, indw] += np.add.reduceat(
                    ampd * (-alphad/x[indwl][None, :]) * expd, indwj, axis=1)
                jac[:, inds] += np.add.reduceat(
                    ampd * 2.*betad*shscale * expd, indsj, axis=1)
                jac[:, inddratio] = np.sum(ampd * expd, axis=1)
                jac[:, inddshift] = np.sum(ampd * 2.*betad*shscale * expd,
                                           axis=1)
            return jac
    else:
        if jac not in ['2-point', '3-point']:
            msg = "jac should be None or in ['2-point', '3-point']"
            raise Exception(msg)
        jac_scale = jac

    return func_detail, func, cost_scale, jac_scale


def multigausfit1d_from_dlines(data, lamb,
                               lambmin=None, lambmax=None,
                               dlines=None, ratio=None,
                               dscale=None, x0_scale=None, bounds_scale=None,
                               method=None, max_nfev=None,
                               xtol=None, ftol=None, gtol=None,
                               Ti=None, vi=None, double=None,
                               verbose=None,
                               loss=None, jac=None):
    """ Solve multi_gaussian fit in 1d from dlines

    If Ti is True, all lines from the same ion have the same width
    If vi is True, all lines from the same ion have the same normalised shift
    If double is True, all lines are double with common shift and ratio

    Unknowns are:
        x = [bck, w0, v0, c00, c01, ..., c0n, w1, v1, c10, c11, ..., c1N, ...]

        - bck : constant background
        - wi  : spectral width of a group of lines (ion): wi^2 = 2kTi / m*c**2
                This way, it is dimensionless
        - vni : normalised velicity of the ion: vni = vi / c
        - cij : normalised coef (intensity) of line: cij = Aij

    Scaling is done so each quantity is close to unity:
        - bck: np.mean(data[data < mean(data)/2])
        - wi : Dlamb / 20
        - vni: 10 km/s
        - cij: np.mean(data)

    """

    # Check format
    if Ti is None:
        Ti = False
    if vi is None:
        vi = False
    if double is None:
        double = False
    assert isinstance(double, bool)
    if jac is None:
        jac = 'call'
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox'], method
    if xtol is None:
        xtol = 1.e-12
    if ftol is None:
        ftol = 1.e-12
    if gtol is None:
        gtol = 1.e-12
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None

    # Use valid data only and optionally restrict lamb
    if lambmin is not None:
        data[lamb<lambmin] = np.nan
    if lambmax is not None:
        data[lamb>lambmax] = np.nan
    indok = ~np.isnan(data)
    data = data[indok]
    lamb = lamb[indok]

    # Prepare
    assert np.allclose(np.unique(lamb), lamb)
    DLamb = lamb[-1] - lamb[0]
    dlines = {k0: v0 for k0, v0 in dlines.items()
              if v0['lambda'] >= lamb[0] and v0['lambda'] <= lamb[-1]}
    lions = sorted(set([dlines[k0]['ION'] for k0 in dlines.keys()]))
    nions = len(lions)
    dions = {k0: [k1 for k1 in dlines.keys() if dlines[k1]['ION'] == k0]
             for k0 in lions}
    dions = {k0: {'lamb': np.array([dlines[k1]['lambda']
                                    for k1 in dions[k0]]),
                  'key': [k1 for k1 in dions[k0]],
                  'symbol': [dlines[k1]['symbol'] for k1 in dions[k0]],
                  'm': np.array([dlines[k1]['m'] for k1 in dions[k0]])}
             for k0 in lions}
    nlines = np.sum([v0['lamb'].size for v0 in dions.values()])

    # Get indices
    dind, lines, mz, keys, sizex, shapey0 = multigausfit1d_from_dlines_ind(
        dions=dions, double=double, Ti=Ti, vi=vi)

    # Get scaling
    if dscale is None:
        dscale = multigausfit1d_from_dlines_scale(data, lamb)

    # Get initial guess
    if x0_scale is None:
        x0_scale = multigausfit1d_from_dlines_x0(sizex, dind,
                                                 lines=lines, data=data,
                                                 lamb=lamb, dscale=dscale,
                                                 double=double)

    # get bounds
    if bounds_scale is None:
        bounds_scale = multigausfit1d_from_dlines_bounds(sizex, dind, double)

    # Get function, cost function and jacobian
    (func_detail, func,
     cost_scale, jac_scale) = multigausfit1d_from_dlines_funccostjac(
        data, lamb,
        dind=dind, dscale=dscale,
         lines=lines, shapey0=shapey0, double=double, jac=jac)

    # Minimize
    t0 = dtm.datetime.now()     # DB
    res = scpopt.least_squares(cost_scale, x0_scale,
                               jac=jac_scale, bounds=bounds_scale,
                               method=method, ftol=ftol, xtol=xtol,
                               gtol=gtol, x_scale=1.0, f_scale=1.0, loss=loss,
                               diff_step=None, tr_solver=None,
                               tr_options={}, jac_sparsity=None,
                               max_nfev=max_nfev, verbose=verbose,
                               args=(), kwargs={})
    msg = ("{}:   time (s)    cost   nfev   njev   term\n\t  ".format(jac)
           + str(round((dtm.datetime.now()-t0).total_seconds(), ndigits=2))
           + "    {}    {}   {}".format(round(res.cost), res.nfev, res.njev)
           + "   "+res.message)
    print(msg)

    # Separate and reshape output
    sol_detail = func_detail(res.x,
                             bckscale=dscale['bck'], ampscale=dscale['amp'],
                             wscale=dscale['width'], shscale=dscale['shift']).T

    # Get result in physical units
    bck = res.x[dind['bck']] * dscale['bck']
    amp = res.x[dind['amp_lines']] * dscale['amp']
    width2 = res.x[dind['width_lines']] * dscale['width']
    shift = res.x[dind['shift_lines']] * dscale['shift']*lines
    coefs = amp*lines*np.sqrt(2*np.pi*width2)
    if double is True:
        dratio = res.x[dind['dratio']]
        dshift = res.x[dind['dshift']] * dscale['shift']*lines
    else:
        dratio, dshift = None, None

    # Derive plasma quantities
    kTiev, vims = None, None
    if Ti is True:
        # Get Ti in eV and vi in m/s
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        kTiev = conv * width2[dind['ions_back']] * mz * scpct.c**2
    if vi is True:
        vims = (res.x[dind['shift_lines'][dind['ions_back']]]
                * dscale['shift'] * scpct.c)

    if ratio is not None:
        # Te can only be obtained as a proxy, units don't matter at this point
        indup = keys.index(ratio['up'])
        indlow = keys.index(ratio['low'])
        ratio['value'] = coefs[indup] / coefs[indlow]
        ratio['str'] = "{}/{}".format(dlines[ratio['up']]['symbol'],
                                      dlines[ratio['low']]['symbol'])

    # Create output dict
    dout = {'data': data, 'lamb': lamb,
            'lines': lines, 'nlines': nlines,
            'sol_detail': sol_detail,
            'sol': np.sum(sol_detail, axis=0),
            'Ti': Ti, 'vi': vi, 'double': double,
            'bck': bck, 'width2': width2, 'shift': shift, 'amp': amp,
            'dratio': dratio, 'dshift': dshift, 'coefs': coefs,
            'dions': dions, 'kTiev': kTiev, 'vims': vims, 'ratio': ratio,
            'cost': res.cost, 'fun': res.fun, 'active_mask': res.active_mask,
            'nfev': res.nfev, 'njev': res.njev, 'status': res.status,
            'msg': res.message, 'success': res.success}
    return dout


###########################################################
###########################################################
#
#           2d spectral fitting
#
###########################################################
###########################################################

def get_knots_nbs_for_bsplines(knots_unique, deg):
    if deg > 0:
        knots = np.r_[[knots_unique[0]]*deg, knots_unique,
                      [knots_unique[-1]]*deg]
    else:
        knots = knots_unique
    nbknotsperbs = 2 + deg
    nbs = knots_unique.size - 1 + deg
    assert nbs == knots.size - 1 - deg
    return knots, nbknotsperbs, nbs


def get_2dspectralfit_func(lamb0, forcelamb=False,
                           deg=None, knots=None):

    lamb0 = np.atleast_1d(lamb0).ravel()
    nlamb = lamb0.size
    knots = np.atleast_1d(knots).ravel()
    nknots = knots.size
    nbsplines = np.unique(knots).size - 1 + deg

    # Define function
    def func(lamb, phi,
             camp=None, cwidth=None, cshift=None,
             lamb0=lamb0, nlamb=nlamb,
             knots=knots, deg=deg, forcelamb=forcelamb,
             nbsplines=nbsplines, mesh=True):
        assert phi.ndim in [1, 2]
        if camp is not None:
            assert camp.shape[0] == nbsplines
            bsamp = BSpline(knots, camp, deg,
                            extrapolate=False, axis=0)
        if csigma is not None:
            assert csigma.shape[0] == nbsplines
            bssigma = BSpline(knots, csigma, deg,
                              extrapolate=False, axis=0)
        if mesh or phi.ndim == 2:
            lamb0 = lamb0[None, None, :]
        else:
            lamb0 = lamb0[None, :]
        if forcelamb:
            if mesh:
                assert angle.ndim == lamb.ndim == 1
                # shape (lamb, angle, lines)
                return np.sum(bsamp(phi)[None,:,:]
                              * np.exp(-(lamb[:,None,None]
                                         - lamb0)**2
                                       /(bssigma(phi)[None,:,:]**2)), axis=-1)
            else:
                assert phi.shape == lamb.shape
                lamb = lamb[..., None]
                # shape (lamb/angle, lines)
                return np.sum(bsamp(phi)
                              * np.exp(-(lamb
                                         - lamb0)**2
                                       /(bssigma(phi)**2)), axis=-1)
        else:
            if cdlamb is not None:
                assert cdlamb.shape[0] == nbsplines
                bsdlamb = BSpline(knots, cdlamb, deg,
                                  extrapolate=False, axis=0)

    return func


def get_multigaussianfit2d_costfunc(lamb=None, phi=None, data=None, std=None,
                                    lamb0=None, forcelamb=None,
                                    deg=None, knots=None,
                                    nlamb0=None, nkperbs=None, nbs=None,
                                    nc=None, debug=None):
    assert lamb.shape == phi.shape == data.shape
    assert lamb.ndim == 1
    assert nc == nbs*nlamb0

    if forcelamb is None:
        forcelamb = False
    if debug is None:
        debug = False

    # Define func assuming all inpus properly formatted
    if forcelamb:
        # x = [camp[1-nbs,...,nbs*(nlamb0-1)-nc}, csigma[1-nc]]
        def func(x,
                 lamb=lamb, phi=phi, data=data, std=std,
                 lamb0=lamb0, knots=knots, deg=deg, nc=nc):
            amp = BSpline(knots, x[:nc], deg,
                          extrapolate=False, axis=0)(phi)
            sigma = BSpline(knots, x[nc:], deg,
                            extrapolate=False, axis=0)(phi)
            val = np.sum(amp[:, None]
                         * np.exp(-(lamb[:, None] - lamb0[None, :])**2
                                  /(sigma[:, None]**2)), axis=-1)
            return (val-data)/(std*data.size)

        def jac(x,
                lamb=lamb, phi=phi, std=std,
                lamb0=lamb0, knots=knots, deg=deg,
                nlamb0=nlamb0, nkperbs=nkperbs, nbs=nbs, nc=nc):
            amp = BSpline(knots, x[:nc], deg,
                          extrapolate=False, axis=0)(phi)
            sigma = BSpline(knots, x[nc:], deg,
                            extrapolate=False, axis=0)(phi)
            jacx = sparse.csr_matrix((phi.size, 2*nc), dtype=float)
            #jacx = np.zeros((phi.size, 2*nc), dtype=float)
            for ii in range(nlamb0):
                expi = np.exp(-(lamb-lamb0[ii])**2/sigma**2)
                for jj in range(nbs):
                    ind = ii*nbs + jj
                    indk = np.r_[jj*nkperbs:(jj+1)*nkperbs]
                    # all bsplines are the same, only coefs (x) are changing
                    bj = BSpline.basis_element(knots[indk],
                                               extrapolate=False)(phi)
                    #bj[np.isnan(bj)] = 0.
                    indok = ~np.isnan(bj)
                    # Differentiate wrt camp
                    jacx[indok, ind] = (bj * expi)[indok]
                    # Differentiate wrt csigma
                    jacx[indok, nc+ind] = (
                        amp * (2*(lamb-lamb0[ii])**2*bj/sigma**3) * expi
                    )[indok]
            return jacx/(std*phi.size)
    else:
        # x = [camp1-nbs*nlamb, csigma1-nbs*nlamb, cdlamb1-nbs*nlamb]
        def func(x,
                 lamb=lamb, phi=phi, data=data, std=std,
                 lamb0=lamb0, knots=knots, deg=deg,
                 nbs=nbs, nlamb0=nlamb0, nc=nc, debug=debug):
            amp = BSpline(knots, x[:nc].reshape((nbs, nlamb0), order='F'),
                          deg, extrapolate=False, axis=0)(phi)
            sigma = BSpline(knots, x[nc:2*nc].reshape((nbs, nlamb0), order='F'),
                            deg, extrapolate=False, axis=0)(phi)
            dlamb = BSpline(knots, x[2*nc:-1].reshape((nbs, nlamb0), order='F'),
                            deg, extrapolate=False, axis=0)(phi)
            val = np.nansum(amp
                            * np.exp(-(lamb[:, None] - (lamb0[None, :]+dlamb))**2
                                  / sigma**2),
                            axis=-1) + x[-1]
            if debug:
                vmin, vmax = 0, np.nanmax(data)
                fig = plt.figure(figsize=(14, 10));
                ax0 = fig.add_axes([0.05,0.55,0.25,0.4])
                ax1 = fig.add_axes([0.35,0.55,0.25,0.4], sharex=ax0, sharey=ax0)
                ax2 = fig.add_axes([0.65,0.55,0.25,0.4], sharex=ax0, sharey=ax0)
                ax3 = fig.add_axes([0.05,0.05,0.25,0.4], sharex=ax0, sharey=ax0)
                ax4 = fig.add_axes([0.35,0.05,0.25,0.4], sharex=ax0, sharey=ax0)
                ax5 = fig.add_axes([0.65,0.05,0.25,0.4], sharex=ax0, sharey=ax0)
                ax0.scatter(lamb, phi, c=data, s=2, marker='s', edgecolors='None',
                           vmin=vmin, vmax=vmax)  # DB
                ax1.scatter(lamb, phi, c=val, s=2, marker='s', edgecolors='None',  # DB
                           vmin=vmin, vmax=vmax)  # DB
                errmax = np.nanmax(np.abs((val-data) / (std*data.size)))
                ax2.scatter(lamb, phi, c=(val-data) / (std*data.size),
                            s=2, marker='s', edgecolors='None',  # DB
                            vmin=-errmax, vmax=errmax, cmap=plt.cm.seismic)  # DB
                dlamb0_amp = np.max(np.diff(lamb0))/np.nanmax(np.abs(amp))
                dlamb0_sigma = np.max(np.diff(lamb0))/np.nanmax(np.abs(sigma))
                dlamb0_dlamb = np.max(np.diff(lamb0))/np.nanmax(np.abs(dlamb))
                for ii in range(nlamb0):
                    ax3.axvline(lamb0[ii], ls='--', c='k')
                    ax4.axvline(lamb0[ii], ls='--', c='k')
                    ax5.axvline(lamb0[ii], ls='--', c='k')
                    ax3.plot(lamb0[ii] + dlamb0_amp*amp[:, ii], phi, '.', ms=4)
                    ax4.plot(lamb0[ii] + dlamb0_sigma*sigma[:, ii], phi, '.', ms=4)
                    ax5.plot(lamb0[ii] + dlamb0_dlamb*dlamb[:, ii], phi, '.', ms=4)
                import ipdb         # DB
                ipdb.set_trace()    # DB
            return (val-data) / (std*data.size)

        def jac(x,
                lamb=lamb, phi=phi, std=std,
                lamb0=lamb0, knots=knots, deg=deg,
                nlamb0=nlamb0, nkperbs=nkperbs, nbs=nbs, nc=nc):
            amp = BSpline(knots, x[:nc], deg,
                          extrapolate=False, axis=0)(phi)
            sigma = BSpline(knots, x[nc:2*nc], deg,
                            extrapolate=False, axis=0)(phi)
            dlamb = BSpline(knots, x[2*nc:], deg,
                            extrapolate=False, axis=0)(phi)
            #jacx = sparse.csr_matrix((phi.size, 2*nc), dtype=float)
            jacx = np.zeros((phi.size, 3*nc+1), dtype=float)
            for ii in range(nlamb0):
                expi = np.exp(-(lamb-(lamb0[ii]+dlamb))**2/sigma**2)
                indlamb = expi > 0.001
                for jj in range(nbs):
                    kk = ii*nbs + jj
                    indk = jj + np.r_[:nkperbs]
                    # all bsplines are the same, only coefs (x) are changing
                    bj = BSpline.basis_element(knots[indk],
                                               extrapolate=False)(phi)
                    # bj[np.isnan(bj)] = 0.
                    indok = (~np.isnan(bj)) & indlamb
                    # Differentiate wrt camp
                    jacx[indok, kk] = (bj[indok] * expi[indok])
                    # Differentiate wrt csigma
                    jacx[indok, nc+kk] = (
                        amp * 2*(lamb-(lamb0[ii]+dlamb))**2*bj/sigma**3 * expi
                    )[indok]
                    # Differentiate wrt dlamb
                    jacx[indok, 2*nc+kk] = (
                         amp * 2*(lamb-(lamb0[ii]+dlamb))*bj/sigma**2 * expi
                    )[indok]
            jacx[:, -1] = 1.
            return jacx/(std*phi.size)
    return func, jac

def multigaussianfit2d(lamb, phi, data, std=None,
                       lamb0=None, forcelamb=None,
                       knots=None, deg=None, nbsplines=None,
                       x0=None, bounds=None,
                       method=None, max_nfev=None,
                       xtol=None, ftol=None, gtol=None,
                       loss=None, verbose=0, debug=None):
    """ Solve multi_gaussian fit in 2d from dlines and bsplines

    If Ti is True, all lines from the same ion have the same width
    If vi is True, all lines from the same ion have the same normalised shift
    If double is True, all lines are double with common shift and ratio

    Unknowns are:
        x = [bck, w0, v0, c00, c01, ..., c0n, w1, v1, c10, c11, ..., c1N, ...]

        - bck : constant background
        - wi  : spectral width of a group of lines (ion): wi^2 = 2kTi / m*c**2
                This way, it is dimensionless
        - vni : normalised velicity of the ion: vni = vi / c
        - cij : normalised coef (intensity) of line: cij = Aij

    Scaling is done so each quantity is close to unity:
        - bck: np.mean(data[data < mean(data)/2])
        - wi : Dlamb / 20
        - vni: 10 km/s
        - cij: np.mean(data)

    """

    # Check format
    if Ti is None:
        Ti = False
    if vi is None:
        vi = False
    if double is None:
        double = False
    assert isinstance(double, bool)
    if jac is None:
        jac = 'call'
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox'], method
    if xtol is None:
        xtol = 1.e-12
    if ftol is None:
        ftol = 1.e-12
    if gtol is None:
        gtol = 1.e-12
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None

    if deg is None:
        deg = 3
    if nbsplines is None:
        nbsplines = 5

    # Use valid data only and optionally restrict lamb
    if lambmin is not None:
        data[lamb<lambmin] = np.nan
    if lambmax is not None:
        data[lamb>lambmax] = np.nan
    indok = ~np.isnan(data)
    data = data[indok]
    lamb = lamb[indok]

    # Prepare
    assert np.allclose(np.unique(lamb), lamb)
    DLamb = lamb[-1] - lamb[0]
    dlines = {k0: v0 for k0, v0 in dlines.items()
              if v0['lambda'] >= lamb[0] and v0['lambda'] <= lamb[-1]}
    lions = sorted(set([dlines[k0]['ION'] for k0 in dlines.keys()]))
    nions = len(lions)
    dions = {k0: [k1 for k1 in dlines.keys() if dlines[k1]['ION'] == k0]
             for k0 in lions}
    dions = {k0: {'lamb': np.array([dlines[k1]['lambda']
                                    for k1 in dions[k0]]),
                  'key': [k1 for k1 in dions[k0]],
                  'symbol': [dlines[k1]['symbol'] for k1 in dions[k0]],
                  'm': np.array([dlines[k1]['m'] for k1 in dions[k0]])}
             for k0 in lions}
    nlines = np.sum([v0['lamb'].size for v0 in dions.values()])

    # Get knots
    if knots is None:
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)
        knots = np.linspace(phimin, phimax, nbsplines+1-deg)
    knots, nkperbs, nbs = get_knots_nbs_for_bsplines(np.unique(knots), deg)

    # Get indices
    dind, lines, mz, keys, sizex, shapey0 = multigausfit1d_from_dlines_ind(
        dions=dions, double=double, Ti=Ti, vi=vi)




    # -------------------------------------
    # ------ Back-up ----------------------
    # -------------------------------------


    # Check inputs
    if std is None:
        std = 0.1*np.nanmean(data)

    # Scaling
    lambmin = np.nanmin(lamb)
    lamb0Delta = np.max(lamb0) - np.min(lamb0)
    nlamb0 = np.size(lamb0)
    nc = nbs*nlamb0

    dlambscale = lamb0Delta / nlamb0
    ampscale = np.nanmean(data) + np.nanstd(data)

    datascale = data / ampscale
    lambscale = lamb / dlambscale
    lamb0scale = lamb0 / dlambscale
    stdscale = std / ampscale

    # Get cost function and jacobian
    func, jac = get_multigaussianfit2d_costfunc(lamb=lambscale,
                                                phi=phi,
                                                data=datascale,
                                                std=stdscale,
                                                lamb0=lamb0scale,
                                                forcelamb=forcelamb,
                                                deg=deg, knots=knots,
                                                nlamb0=nlamb0, nbs=nbs,
                                                nkperbs=nkperbs, nc=nc,
                                                debug=debug)

    # Get initial guess
    if x0 is None:
        x0 = np.r_[np.ones((nc,)), np.ones((nc,))]
        if not forcelamb:
            x0 = np.r_[x0, np.zeros((nc,))]
            x0 = np.r_[x0, 0.]

    # Get bounds
    if bounds is None:
        bounds = (np.r_[np.zeros((nc,)),
                        np.full((nc,), nlamb0/100)],
                  np.r_[np.full((nc,), np.nanmax(data)/ampscale),
                        np.full((nc,), 3.)])
        if not forcelamb:
            bounds = (np.r_[bounds[0], -np.full((nc,), 2.)],
                      np.r_[bounds[1], np.full((nc,), 2.)])
        bounds = (np.r_[bounds[0], 0.],
                  np.r_[bounds[1], 0.1*np.nanmax(data)/ampscale])

    # Minimize
    res = scpopt.least_squares(func, x0, jac=jac, bounds=bounds,
                               method=method, ftol=ftol, xtol=xtol,
                               gtol=gtol, x_scale=1.0, f_scale=1.0, loss=loss,
                               diff_step=None, tr_solver=None,
                               tr_options={}, jac_sparsity=None,
                               max_nfev=max_nfev, verbose=verbose,
                               args=(), kwargs={})

    # Separate and reshape output
    camp = res.x[:nc].reshape((nlamb0, nbs)) * ampscale
    csigma = res.x[nc:2*nc].reshape((nlamb0, nbs)) * dlambscale
    if forcelamb:
        cdlamb = None
    else:
        cdlamb = res.x[2*nc:3*nc].reshape((nlamb0, nbs)) * dlambscale

    # Create output dict
    dout = {'camp': camp, 'csigma': csigma, 'cdlamb': cdlamb, 'bck':res.x[-1],
            'fit':(func(res.x)*stdscale*data.size + datascale) * ampscale,
            'lamb0':lamb0, 'knots': knots, 'deg':deg, 'nbsplines': nbsplines,
            'cost': res.cost, 'fun': res.fun, 'active_mask': res.active_mask,
            'nfev': res.nfev, 'njev': res.njev, 'status': res.status}
    return dout
