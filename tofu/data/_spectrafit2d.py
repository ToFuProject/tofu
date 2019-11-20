
# Built-in
import os
import warnings

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
#           1d spectral fitting with physics parameters
#
###########################################################
###########################################################


def get_lamb0_from_dlines(dlines):
    lamb0, ions = zip(*[(vv['lamb0'],
                         np.full((len(vv['lamb0']),), kk))
                       for kk, vv in dlines.items()])
    lamb0 = np.r_[lamb0]
    ind = np.argsort(lamb0)
    return lamb0[ind], np.concatenate(ions)[ind]


def get_dindx(bckdeg=None, dlines=None, nbs=None):

    nbck = bckdeg + 1
    if nbs is None:
        # 1d spectral fit
        nbs = 1

    i0 = nbck
    lk = ['sigma', 'dlamb', 'amp', 'ntot', 'nlamb']
    dindx= {'bck': np.r_[:nbck],
            'ions':dict.fromkeys(sorted(dlines.keys())),
            'nbs': nbs}
    for kk in dindx['ions'].keys():
        dindx['ions'][kk] = dict.fromkeys(lk)
        dindx['ions'][kk]['sigma'] = i0 + np.r_[:nbs]
        dindx['ions'][kk]['dlamb'] = i0+nbs + np.r_[:nbs]
        nlamb = len(dlines[kk]['lamb0'])
        dindx['ions'][kk]['amp'] = i0+2*nbs + np.r_[:nlamb*nbs]
        dindx['ions'][kk]['nlamb'] = nlamb
        dindx['ions'][kk]['ntot'] = (2 + nlamb)*nbs
        i0 += dindx['ions'][kk]['ntot']
    dindx['nall'] = i0
    return dindx


def get_x0_bounds(x01d=None, dlines=None, dindx=None,
                  lamb=None, data=None):

    x0 = np.zeros((dindx['nall'],), dtype=float)
    if x01d is None:
        # Get average spectral width and separation
        lamb0_Delta = lamb0.max() - lamb0.min()
        nlamb0 = lamb0.size
        lamb0_delta = lamb0_Delta / nlamb0

        nbs = dindx['nbs']

        x0[dindx['bck']] = np.zeros((dindx['bck'].size,))
        for kk in dindx['ions'].keys():
            # sigma
            x0[dindx[kk]['sigma']] = lamb0_delta
            # dlamb
            x0[dindx[kk]['dlamb']] = 0.
            # amp
            x0[dindx[kk]['amp']] = ampmean

    else:
        x0[dindx['bck']] = x01d[dindx['bck']]
        i0 = dindx['bck'].size
        for kk in dindx['ions'].keys():
            # TBF
            # x0[dindx[kk]['sigma']] = x01d[]
            pass

    # Get bounds
    lamb_delta = np.mean(np.abs(np.diff(np.unique(lamb))))
    datamax = np.nanmax(data)
    bampup = min(datamax, np.nanmean(data) + np.nanstd(data))

    bounds0 = np.zeros((dindx['nall'],), dtype=float)
    bounds1 = np.zeros((dindx['nall'],), dtype=float)
    if dindx['bck'].size == 1:
        bounds0[dindx['bck']] = 0.
        bounds1[dindx['bck']] = bampup
    elif dindx['bck'].size == 2:
        bounds0[dindx['bck'][0]] = 0.
        bounds1[dindx['bck'][0]] = bampup
        bounds0[dindx['bck'][0]] = 0.           # TBC
        bounds1[dindx['bck'][0]] = bampup       # TBC
    for kk in dindx['ions'].keys():
        bounds0[dindx[kk]['sigma']] = 2.*lamb_delta
        bounds1[dindx[kk]['sigma']] = lamb0_delta*5.
        bounds0[dindx[kk]['dlamb']] = -3.*lamb0_delta
        bounds1[dindx[kk]['dlamb']] = 3.*lamb0_delta
        bounds0[dindx[kk]['amp']] = 0.
        bounds1[dindx[kk]['amp']] = datamax

    return x0, bounds




def get_funccostjac():

    def func():
        pass

    def cost():
        pass

    def jac():
        pass

    return func, cost, jac










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

    # Check inputs
    if deg is None:
        deg = 3
    if nbsplines is None:
        nbsplines = 5
    if method is None:
        method = 'trf'
    # Only 2 method for pb. with bounds
    assert method in ['trf', 'dogbox'], method
    if xtol is None:
        xtol = 1.e-6
    if ftol is None:
        ftol = 1.e-6
    if gtol is None:
        gtol = 1.e-8
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None
    if std is None:
        std = 0.1*np.nanmean(data)
    assert lamb0 is not None

    # Get knots
    if knots is None:
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)
        knots = np.linspace(phimin, phimax, nbsplines+1-deg)
    knots, nkperbs, nbs = get_knots_nbs_for_bsplines(np.unique(knots), deg)

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




###########################################################
#
#           From DataCam2D
#
###########################################################
###########################################################


# DEPRECATED
def fit_spectra2d_x0_per_row():

    # Loop from centre to edges
    for jj in range(ny):
        out = multiplegaussianfit(x, datat[jj,:], nmax=nmax, p0=p0u, bounds=None,
                                  max_nfev=None, xtol=1.e-8, verbose=0,
                                  percent=20, plot_debug=False)
        x0[jj,:], x0_std[jj,:] = out[:2]

        for jj in range(nybis):
            # Upper half
            ju = indy1[jj]
            out = multiplegaussianfit(x, spect, nmax=nmax, p0=p0u, bounds=None,
                                      max_nfev=None, xtol=1.e-8, verbose=0,
                                      percent=20, plot_debug=False)
            x0[ju,:], x0_std[ju,:] = out[:2]
            p0u[:nmax], p0u[nmax:2*nmax], = amp[ii,ju,:], x0[ii,ju,:]
            p0u[2*nmax:3*nmax], p0u[3*nmax:] = sigma[ii,ju,:], bck0

            # Lower half
            jl = indy2[jj]
    return x0



def get_func2d(y0, y1, x0_y, bspl_n, bspl_deg):
    knots = np.linspace(y0,y1, 6)
    bspliney = scpinterp.LSQUnivariateSpline()
    def func(x, y, ampy_coefs, sigy_coefs, bcky_coefs):
        amp_bs = BSpline(knots, ampy_coefs, k=bspl_deg,
                         extrapolate=False, axis=0)
        amp = amp_bs(y)
        x0y = x0_y(y)
        return np.sum(amp*np.exp((x-xoy)**2/sig**2) + bck0, axis=-1)
    return func



def fit_spectra_2d(data2d, indt=None, nbin_init=None,
                   nmax=None, bck=None, nbsplines=None):
    """ Return fitted spectra

    Can handle unique or multiple time
    Takes already formatted 2d data:
        - (nx, ny)
        - (nt, nx, ny)
    x being the horizontal / spectral direction (lambda)

    """

    #####################
    # Check / format input
    #####################

    # Check data
    assert isinstance(data, np.ndarray)
    assert data.ndim in [2,3]
    if data.ndim == 2:
        data = np.reshape((1,data.shape[0],data.shape[1]))
    if indt is not None:
        data = data[indt,...]

    # Set bck type
    if bck is None:
        bck = 0
    assert type(bck) in [int, str]
    if type(bck) is int:
        nbck = bck + 1
    elif bck == 'exp':
        nbck = 3

    # Extract shape
    nt = data.shape[0]
    nlamb, ny = data.shape[1:]
    x = np.arange(0,nlamb)

    # max number of spectral lines (gaussians)
    if nmax is None:
        nmax = 10

    # Check nbin_init vs ny
    if nbin_init is None:
        nbin_init = 100
    if ny % 2 != nbin_init % 2:
        nbin_init += 1

    # get indybin
    indybin = np.arange(0, nbin_init)
    if ny % 2 == 0:
        indybin += int(ny/2 - nbin_init/2)
    else:
        indybin += int((ny-1)/2 - (nbin_init-1)/2)

    # get indybis
    if ny % 2 == 0:
        indy1 = np.arange(ny/2-1, -1, -1)
        indy2 = np.arange(ny/2, ny, 1)
    else:
        indy1 = np.arange((ny-1)/2-1, -1, -1)
        indy2 = np.arange((ny-1)/2+1, ny, 1)
    nybis = indy1.size
    assert nybis == indy2.size

    #####################
    # initialize output
    #####################

    # results
    amp = np.full((nt, ny, nmax), np.nan)
    x0 = np.full((ny, nmax), np.nan)
    sigma = np.full((nt, ny, nmax), np.nan)
    bck0 = np.full((nt, ny, nbck), np.nan)
    p0u = np.full((nmax*3 + nbck,), np.nan)
    p0l = np.full((nmax*3 + nbck,), np.nan)

    # results std
    amp_std = np.full((nt, ny, nmax), np.nan)
    x0_std = np.full((nt, ny, nmax), np.nan)
    sigma_std = np.full((nt, ny, nmax), np.nan)
    bck0_std = np.full((nt, ny, nbck), np.nan)
    std = np.full((nt,), np.nan)


    #####################
    # Determine x0 for each row
    #####################

    datat = np.sum(data, axis=0)

    # Start with central binned (initial guess)

    # if ny is odd => use binned result for central line


    x0 = None

    # Deduce ellipses

    # Derive x0_func(y, x0)


    #####################
    # Define 2d multi-gaussian
    #####################

    # Define x0(y) law




    #####################
    # compute fits, with fixed x0
    #####################

    # Get binned spectra to deduce initial guess
    databin =  None

    # loop over y to fit spectra, start from middle, extent to edges
    for ii in range(nt):

            out = fit2dspectra(x, y, data[ii,:,:])

            p0u[:nmax] = amp[ii-1,indy1[0],:]
            p0u[nmax:2*nmax] = x0[ii-1,indy1[0],:]
            p0u[2*nmax:3*nmax] = sigma[ii,indy1[0],:]
            p0u[3*nmax:] = bck0[ii,indy1[0]]

            p0l[:nmax] = amp[ii-1,indy2[0],:]
            p0l[nmax:2*nmax] = x0[ii-1,indy2[0],:]
            p0l[2*nmax:3*nmax] = sigma[ii,indy2[0],:]
            p0l[3*nmax:] = bck0[ii,indy2[0]]


    return (x0, x0_std), ( )


###########################################################
###########################################################
#
#           For inspiration
#
###########################################################
###########################################################



# _indymod = {0: np.arange(0,195),
            # 1: np.arange(212,407),
            # 2: np.arange(424,619),
            # 3: np.arange(636,831),
            # 4: np.arange(848,1043),
            # 5: np.arange(1060,1255),
            # 6: np.arange(1272,1467)}

# def get_binnedspectra(imp='FeXXV', mod=3, path='./',
                      # dshots=_dshots, indymod=_indymod, save=True):

    # # Prepare
    # indy = indymod[mod]
    # dimp = dict(dshots[imp])

    # # Loop on shots
    # lextra = ['Te0','ne0','dg16']
    # ls = sorted(dimp.keys())
    # for shot in ls:

        # # Load data
        # if 'tlim' in dimp[shot].keys():
            # tlim = dimp[shot]['tlim']
        # else:
            # tlim = None
        # try:
            # xics, kh = tfw.SpectroX2D.load_data(shot,
                                                # tlim=tlim, plot=False)
        # except Exception as err:
            # dimp[shot]['err'] = str(err)
            # continue

        # # Bin data
        # data, t = xics._Ref['data'], xics._Ref['t']
        # dimp[shot]['nt'] = t.size
        # spectra = np.empty((dimp[shot]['nt'], 487), dtype=float)
        # for ii in range(0,dimp[shot]['nt']):
            # spectra[ii,:] = np.nanmean(data[ii,:].reshape(1467,487)[indy,:],
                                       # axis=0)
        # dimp[shot]['spectra'] = spectra
        # dimp[shot]['t'] = t
        # #dimp[shot]['date'] = IRFMtb.

        # # Dextra
        # for ss in lextra:
            # try:
                # indt = np.digitize(xics.dextra[ss]['t'], 0.5*(t[1:]+t[:-1]))
                # val = np.empty((dimp[shot]['nt'],),dtype=float)
                # std = np.empty((dimp[shot]['nt'],),dtype=float)
                # ssum = np.empty((dimp[shot]['nt'],),dtype=float)
                # for ii in range(0,dimp[shot]['nt']):
                    # val[ii] = np.nanmean(xics.dextra[ss]['data'][indt==ii])
                    # std[ii] = np.nanstd(xics.dextra[ss]['data'][indt==ii])
                    # ssum[ii] = np.nansum(xics.dextra[ss]['data'][indt==ii])
                # dimp[shot][ss] = {'mean':val, 'std':std, 'sum':ssum}
            # except Exception as err:
                # dimp[shot][ss] = {'mean':np.nan, 'std':np.nan, 'sum':np.nan}

    # # Reshape dict for np.savez and pandas DataFrame
    # nt = np.array([dimp[shot]['nt'] for shot in ls])
    # nttot = np.sum(nt)
    # ntcum = np.cumsum(nt)
    # lk = ['shot','angle','spectra','t',
          # 'Te0-mean','Te0-std','ne0-mean','ne0-std',
          # 'dg16-sum']
    # dk = {}
    # for k in lk:
        # shape = (nttot,487) if k=='spectra' else (nttot,)
        # dk[k] = np.full(shape, np.nan)

    # i0 = 0
    # for ii in range(0,len(ls)):
        # ind = np.arange(i0,i0+nt[ii])
        # dk['shot'][ind] = ls[ii]
        # dk['angle'][ind] = dimp[ls[ii]]['ang']
        # dk['spectra'][ind,:] = dimp[ls[ii]]['spectra']
        # dk['t'][ind] = dimp[ls[ii]]['t']
        # dk['Te0-mean'][ind] = dimp[ls[ii]]['Te0']['mean']
        # dk['Te0-std'][ind] = dimp[ls[ii]]['Te0']['std']
        # dk['ne0-mean'][ind] = dimp[ls[ii]]['ne0']['mean']
        # dk['ne0-std'][ind] = dimp[ls[ii]]['ne0']['std']
        # dk['dg16-sum'][ind] = dimp[ls[ii]]['dg16']['sum']
        # i0 = ntcum[ii]

    # # Saving
    # if save:
        # name = '%s_spectra'%imp
        # path = os.path.abspath(path)
        # pfe = os.path.join(path,name+'.npz')
        # try:
            # np.savez(pfe, **dk)
            # print("Saved in :", pfe)
        # except:
            # import ipdb
            # ipdb.set_trace()
            # pass
    # return dk



# ####################################################################
# #       spectral fit
# ####################################################################


# def remove_bck(x, y):
    # #opt = np.polyfit(x, y, deg=0)
    # opt = [np.nanmin(y)]
    # ybis = y - opt[0]
    # return ybis, opt[0]


# def get_peaks(x, y, nmax=10):

    # # Prepare
    # ybis = np.copy(y)
    # A = np.empty((nmax,),dtype=y.dtype)
    # x0 = np.empty((nmax,),dtype=x.dtype)
    # sigma = np.empty((nmax,),dtype=y.dtype)
    # gauss = lambda xx, A, x0, sigma: A*np.exp(-(xx-x0)**2/sigma**2)
    # def gauss_jac(xx, A, x0, sigma):
        # jac = np.empty((xx.size,3),dtype=float)
        # jac[:,0] = np.exp(-(xx-x0)**2/sigma**2)
        # jac[:,1] = A*2*(xx-x0)/sigma**2 * np.exp(-(xx-x0)**2/sigma**2)
        # jac[:,2] = A*2*(xx-x0)**2/sigma**3 * np.exp(-(xx-x0)**2/sigma**2)
        # return jac

    # dx = np.nanmin(np.diff(x))

    # # Loop
    # nn = 0
    # while nn<nmax:
        # ind = np.nanargmax(ybis)
        # x00 = x[ind]
        # if np.any(np.diff(ybis[ind:],n=2)>=0.):
            # wp = min(x.size-1,
                     # ind + np.nonzero(np.diff(ybis[ind:],n=2)>=0.)[0][0] + 1)
        # else:
            # wp = ybis.size-1
        # if np.any(np.diff(ybis[:ind+1],n=2)>=0.):
            # wn = max(0, np.nonzero(np.diff(ybis[:ind+1],n=2)>=0.)[0][-1] - 1)
        # else:
            # wn = 0
        # width = x[wp]-x[wn]
        # assert width>0.
        # indl = np.arange(wn,wp+1)
        # sig = np.ones((indl.size,))
        # if (np.abs(np.mean(np.diff(ybis[ind:wp+1])))
            # > np.abs(np.mean(np.diff(ybis[wn:ind+1])))):
            # sig[indl<ind] = 1.5
            # sig[indl>ind] = 0.5
        # else:
            # sig[indl<ind] = 0.5
            # sig[indl>ind] = 1.5
        # p0 = (ybis[ind],x00,width)#,0.)
        # bounds = (np.r_[0.,x[wn],dx/2.],
                  # np.r_[5.*ybis[ind],x[wp],5.*width])
        # try:
            # (Ai, x0i, sigi) = scpopt.curve_fit(gauss, x[indl], ybis[indl],
                                               # p0=p0, bounds=bounds, jac=gauss_jac,
                                               # sigma=sig, x_scale='jac')[0]
        # except Exception as err:
            # print(str(err))
            # import ipdb
            # ipdb.set_trace()
            # pass

        # ybis = ybis - gauss(x, Ai, x0i, sigi)
        # A[nn] = Ai
        # x0[nn] = x0i
        # sigma[nn] = sigi


        # nn += 1
    # return A, x0, sigma

# def get_p0bounds(x, y, nmax=10):

    # yflat, bck = remove_bck(x,y)
    # A, x0, sigma = get_peaks(x, yflat, nmax=nmax)

    # p0 = A.tolist() + x0.tolist() + sigma.tolist() + [bck]

    # lx = [np.nanmin(x), np.nanmax(x)]
    # Dx = np.diff(lx)
    # dx = np.nanmin(np.diff(x))

    # bA = (np.zeros(nmax,), np.full((nmax,),3.*np.nanmax(y)))
    # bx0 = (np.full((nmax,),lx[0]-Dx/2.), np.full((nmax,),lx[1]+Dx/2.))
    # bsigma = (np.full((nmax,),dx/2.), np.full((nmax,),Dx/2.))
    # bbck0 = (0., np.nanmax(y))

    # bounds = (np.r_[bA[0],bx0[0],bsigma[0], bbck0[0]],
              # np.r_[bA[1],bx0[1],bsigma[1], bbck0[1]])
    # if not np.all(bounds[0]<bounds[1]):
        # msg = "Lower bounds must be < upper bounds !\n"
        # msg += "    lower :  %s\n"+str(bounds[0])
        # msg += "    upper :  %s\n"+str(bounds[1])
        # raise Exception(msg)
    # return p0, bounds


# def get_func(n=5):
    # def func_vect(x, A, x0, sigma, bck0):
        # y = np.full((A.size+1, x.size), np.nan)
        # for ii in range(A.size):
            # y[ii,:] = A[ii]*np.exp(-(x-x0[ii])**2/sigma[ii]**2)
        # y[-1,:] = bck0
        # return y

    # def func_sca(x, *args, n=n):
        # A = np.r_[args[0:n]]
        # x0 = np.r_[args[n:2*n]]
        # sigma = np.r_[args[2*n:3*n]]
        # bck0 = np.r_[args[3*n]]
        # gaus = A[:,np.newaxis]*np.exp(-(x[np.newaxis,:]-x0[:,np.newaxis])**2/sigma[:,np.newaxis]**2)
        # back = bck0
        # return np.sum( gaus, axis=0) + back

    # def func_sca_jac(x, *args, n=n):
        # A = np.r_[args[0:n]][np.newaxis,:]
        # x0 = np.r_[args[n:2*n]][np.newaxis,:]
        # sigma = np.r_[args[2*n:3*n]][np.newaxis,:]
        # bck0 = np.r_[args[3*n]]
        # jac = np.full((x.size,3*n+1,), np.nan)
        # jac[:,:n] = np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        # jac[:,n:2*n] = A*2*(x[:,np.newaxis]-x0)/(sigma**2) * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        # jac[:,2*n:3*n] = A*2*(x[:,np.newaxis]-x0)**2/sigma**3 * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        # jac[:,-1] = 1.
        # return jac

    # return func_vect, func_sca, func_sca_jac


# def multiplegaussianfit(x, spectra, nmax=10, p0=None, bounds=None,
                        # max_nfev=None, xtol=1.e-8, verbose=0,
                        # percent=20, plot_debug=False):

    # # Prepare
    # if spectra.ndim==1:
        # spectra = spectra.reshape((1,spectra.size))
    # nt = spectra.shape[0]

    # A = np.full((nt,nmax),np.nan)
    # x0 = np.full((nt,nmax),np.nan)
    # sigma = np.full((nt,nmax),np.nan)
    # bck0 = np.full((nt,),np.nan)
    # Astd = np.full((nt,nmax),np.nan)
    # x0std = np.full((nt,nmax),np.nan)
    # sigmastd = np.full((nt,nmax),np.nan)
    # bck0std = np.full((nt,),np.nan)
    # std = np.full((nt,),np.nan)

    # # Prepare info
    # if verbose is not None:
        # print("----- Fitting spectra with {0} gaussians -----".format(nmax))
    # nspect = spectra.shape[0]
    # nstr = max(nspect//max(int(100/percent),1),1)

    # # bounds and initial guess
    # if p0 is None or bounds is None:
        # p00, bounds0 = get_p0bounds(x, spectra[0,:], nmax=nmax)
    # if p0 is None:
        # p0 = p00
    # if bounds is None:
        # bounds = bounds0

    # # Get fit
    # func_vect, func_sca, func_sca_jac = get_func(nmax)
    # lch = []
    # for ii in range(0,nspect):

        # if verbose is not None and ii%nstr==0:
            # print("=> spectrum {0} / {1}".format(ii,nspect))

        # try:
            # popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                          # jac=func_sca_jac,
                                          # p0=p0, bounds=bounds,
                                          # max_nfev=max_nfev, xtol=xtol,
                                          # x_scale='jac',
                                          # verbose=verbose)
        # except Exception as err:
            # msg = "    Convergence issue for {0} / {1}\n".format(ii,nspect)
            # msg += "    => %s\n"%str(err)
            # msg += "    => Resetting initial guess and bounds..."
            # print(msg)
            # try:
                # p0, bounds = get_p0bounds(x, spectra[ii,:], nmax=nmax)
                # popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                              # jac=func_sca_jac,
                                              # p0=p0, bounds=bounds,
                                              # max_nfev=max_nfev, xtol=xtol,
                                              # x_scale='jac',
                                              # verbose=verbose)
                # p0 = popt
                # popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                              # jac=func_sca_jac,
                                              # p0=p0, bounds=bounds,
                                              # max_nfev=max_nfev, xtol=xtol,
                                              # x_scale='jac',
                                              # verbose=verbose)
                # lch.append(ii)
            # except Exception as err:
                # print(str(err))
                # import ipdb
                # ipdb.set_trace()
                # continue


        # A[ii,:] = popt[:nmax]
        # x0[ii,:] = popt[nmax:2*nmax]
        # sigma[ii,:] = popt[2*nmax:3*nmax]
        # bck0[ii] = popt[3*nmax]

        # stdi = np.sqrt(np.diag(pcov))
        # Astd[ii,:] = stdi[:nmax]
        # x0std[ii,:] = stdi[nmax:2*nmax]
        # sigmastd[ii,:] = stdi[2*nmax:3*nmax]
        # bck0std[ii] = stdi[3*nmax]
        # std[ii] = np.sqrt(np.sum((spectra[ii,:]-func_sca(x,*popt))**2))

        # p0[:] = popt[:]

        # if plot_debug and ii in [0,1]:
            # fit = func_vect(x, A[ii,:], x0[ii,:], sigma[ii,:], bck0[ii])

            # plt.figure()
            # ax0 = plt.subplot(2,1,1)
            # ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)
            # ax0.plot(x,spectra[ii,:], '.k',
                     # x, np.sum(fit,axis=0), '-r')
            # ax1.plot(x, fit.T)

    # return (A,Astd), (x0,x0std), (sigma,sigmastd), (bck0,bck0std), std, lch



# def add_gaussianfits(dimp, nmax=10, verbose=0, percent=20,
                     # path='./', save=False):
    # assert type(dimp) in [str,dict]

    # # Prepare
    # if type(dimp) is str:
        # imp = str(dimp)
        # if '_' in imp:
            # imp = imp.split('_')[0]
        # dimp = dict(np.load(dimp+'_spectra.npz'))
        # inds = np.argsort(dimp['angle'])
        # for k in dimp.keys():
            # if inds.size in dimp[k].shape:
                # dimp[k] = dimp[k][inds] if dimp[k].ndim==1 else dimp[k][inds,:]
    # else:
        # imp = None
        # save = False


    # # Compute
    # ind = np.arange(3,437)
    # x = np.arange(0,dimp['spectra'].shape[1])
    # spectra = dimp['spectra'][:,ind]
    # A, x0, sigma, bck0, std, lch = multiplegaussianfit(x[ind], spectra,
                                                       # nmax=nmax, percent=percent,
                                                       # verbose=verbose)
    # # Store
    # dimp['nmax'] = nmax
    # dimp['indch'] = np.r_[lch]
    # dimp['x'] = x
    # dimp['ind'] = ind
    # dimp['A'] = A[0]
    # dimp['A-std'] = A[1]
    # dimp['x0'] = x0[0]
    # dimp['x0-std'] = x0[1]
    # dimp['sigma'] = sigma[0]
    # dimp['sigma-std'] = sigma[1]
    # dimp['bck0'] = bck0[0]
    # dimp['bck0-std'] = bck0[1]
    # dimp['std'] = std
    # if imp is not None:
        # dimp['imp'] = imp

    # if save:
        # name = '{0}_fitted{1}'.format(imp,nmax)
        # path = os.path.abspath(path)
        # pfe = os.path.join(path,name+'.npz')
        # np.savez(pfe, **dimp)
        # print("Saved in :", pfe)

    # return dimp



# def plot_gaussianfits(dimp, ind):

    # # Prepare
    # x = dimp['x']
    # spectra = dimp['spectra'][ind,:]
    # func_vect, func_sca, func_sca_jac = get_func(dimp['nmax'])
    # fit = func_vect(x, dimp['A'][ind,:], dimp['x0'][ind,:],
                    # dimp['sigma'][ind,:], dimp['bck0'][ind])

    # # Plot
    # plt.figure();
    # ax0 = plt.subplot(2,1,1)
    # ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)

    # ax0.plot(x, spectra, '.k', label='spectrum')
    # ax0.plot(x, np.sum(fit,axis=0), '-r', label='fit')
    # ax1.plot(x, fit.T)

    # ax1.set_xlabel(r'x')
    # ax0.set_ylabel(r'data')
    # ax1.set_ylabel(r'data')


# def plot_allraw(dimp):

    # # Prepare
    # x = dimp['x']
    # spectra = dimp['spectra']
    # nspect = spectra.shape[0]
    # spectranorm = spectra / np.nanmax(spectra,axis=1)[:,np.newaxis]

    # # Plot
    # plt.figure();
    # ax0 = plt.subplot(2,1,1)
    # ax1 = plt.subplot(2,1,2, sharex=ax0)

    # ax0.imshow(spectranorm, cmap=plt.cm.viridis,
               # extent=(x.min(),x.max(),0,nspect),
               # origin='lower',
               # interpolation='bilinear')


# def extract_lines(dimp):

    # nspect, nx = dimp['spectra'].shape
    # if dimp['imp'] == 'ArXVII':
        # dlines = {'w': {'range':np.arange(280,nx)},
                  # 'z': {'range':np.arange(0,200)}}
        # for k in dlines.keys():
            # spect = np.copy(dimp['spectra'])
            # spect[:,dlines[k]['range']] = 0.
            # dlines[k].update({'x':np.full((nspect,),np.nan),
                              # 'x0':np.full((nspect,),np.nan),
                              # 'A':np.full((nspect,),np.nan),
                              # 'sigma':np.full((nspect,),np.nan)})
            # for ii in range(0,dimp['spectra'].shape[0]):
                # xl = dimp['x'][np.argmax(spect[ii,:])]
                # ind = np.argmin(np.abs(dimp['x0'][ii,:]-xl))
                # dlines[k]['x'][ii] = xl
                # dlines[k]['x0'][ii] = dimp['x0'][ii,ind]
                # dlines[k]['A'][ii] = dimp['A'][ii,ind]
                # dlines[k]['sigma'][ii] = dimp['sigma'][ii,ind]

    # dimp.update(**dlines)
    # return dimp
