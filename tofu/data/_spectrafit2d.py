
# Built-in
import os
import warnings
import itertools as itt
import copy
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
_DCONSTRAINTS = {'amp': False,
                 'width': False,
                 'shift': False,
                 'double': False}
_SAME_SPECTRUM = False
_DEG = 3
_NBSPLINES = 6
_TOL1D = {'x': 1e-12, 'f': 1.e-12, 'g': 1.e-12}
_TOL2D = {'x': 1e-6, 'f': 1.e-6, 'g': 1.e-6}

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

def _width_shift_amp(indict, keys=None, dlines=None, nlines=None):

    # ------------------------
    # Prepare error message
    msg = ''

    # ------------------------
    # Check case
    c0 = indict is False
    c1 = isinstance(indict, str)
    c2 = (isinstance(indict, dict)
          and all([isinstance(k0, str)
                   and (isinstance(v0, list) or isinstance(v0, str))
                   for k0, v0 in indict.items()]))
    c3 = (isinstance(indict, dict)
          and all([(ss in keys
                    and isinstance(vv, dict)
                    and all([s1 in ['key', 'coef', 'offset']
                             for s1 in vv.keys()])
                    and isinstance(vv['key'], str))
                   for ss, vv in indict.items()]))
    c4 = (isinstance(indict, dict)
          and isinstance(indict.get('keys'), list)
          and isinstance(indict.get('ind'), np.ndarray))
    if not any([c0, c1, c2, c3, c4]):
        msg = ("Wrong input dict!\n"
               + "\t- lc = {}\n".format([c0, c1, c2, c3, c4])
               + "\t- indict =\n{}".format(indict))
        raise Exception(msg)

    # ------------------------
    # str key to be taken from dlines as criterion
    if c0:
        lk = keys
        ind = np.eye(nlines)
        outdict = {'keys': np.r_[lk], 'ind': ind,
                   'coefs': np.ones((nlines,)),
                   'offset': np.zeros((nlines,))}

    if c1:
        lk = sorted(set([dlines[k0].get(indict, k0)
                         for k0 in keys]))
        ind = np.array([[dlines[k1].get(indict, k1) == k0
                         for k1 in keys] for k0 in lk])
        outdict = {'keys': np.r_[lk], 'ind': ind,
                   'coefs': np.ones((nlines,)),
                   'offset': np.zeros((nlines,))}

    elif c2:
        lkl = []
        for k0, v0 in indict.items():
            if isinstance(v0, str):
                v0 = [v0]
            if not (len(set(v0)) == len(v0)
                    and all([k1 in keys and k1 not in lkl for k1 in v0])):
                msg = ("Inconsistency in indict[{}], either:\n".format(k0)
                       + "\t- v0 not unique: {}\n".format(v0)
                       + "\t- some v0 not in keys: {}\n".format(keys)
                       + "\t- some v0 in lkl:      {}".format(lkl))
                raise Exception(msg)
            indict[k0] = v0
            lkl += v0
        for k0 in set(keys).difference(lkl):
            indict[k0] = [k0]
        lk = sorted(set(indict.keys()))
        ind = np.array([[k1 in indict[k0] for k1 in keys] for k0 in lk])
        outdict = {'keys': np.r_[lk], 'ind': ind,
                   'coefs': np.ones((nlines,)),
                   'offset': np.zeros((nlines,))}

    elif c3:
        lk = sorted(set([v0['key'] for v0 in indict.values()]))
        lk += sorted(set(keys).difference(indict.keys()))
        ind = np.array([[indict.get(k1, {'key': k1})['key'] == k0
                         for k1 in keys]
                        for k0 in lk])
        coefs = np.array([indict.get(k1, {'coef': 1.}).get('coef', 1.)
                          for k1 in keys])
        offset = np.array([indict.get(k1, {'offset': 0.}).get('offset', 0.)
                           for k1 in keys])
        outdict = {'keys': np.r_[lk], 'ind': ind,
                   'coefs': coefs,
                   'offset': offset}

    elif c4:
        outdict = indict
        if 'coefs' not in indict.keys():
            outdict['coefs'] = np.ones((nlines,))
        if 'offset' not in indict.keys():
            outdict['offset'] = np.zeros((nlines,))

    # ------------------------
    # Ultimate conformity checks
    if not c0:
        assert sorted(outdict.keys()) == ['coefs', 'ind', 'keys', 'offset']
        assert isinstance(outdict['ind'], np.ndarray)
        assert outdict['ind'].dtype == np.bool_
        assert outdict['ind'].shape == (outdict['keys'].size, nlines)
        assert np.all(np.sum(outdict['ind'], axis=0) == 1)
        assert outdict['coefs'].shape == (nlines,)
        assert outdict['offset'].shape == (nlines,)
    return outdict


def multigausfit1d_from_dlines_dinput(dlines=None,
                                      dconstraints=None,
                                      lambmin=None, lambmax=None,
                                      same_spectrum=None,
                                      nspect=None, dlamb=None,
                                      defconst=_DCONSTRAINTS):

    # ------------------------
    # Check / format basics
    # ------------------------

    # Select relevant lines (keys, lamb)
    keys = np.array([k0 for k0 in dlines.keys()])
    lamb = np.array([dlines[k0]['lambda'] for k0 in keys])
    if lambmin is not None:
        keys = keys[lamb >= lambmin]
        lamb = lamb[lamb >= lambmin]
    else:
        lambmin = lamb.min()
    if lambmax is not None:
        keys = keys[lamb <= lambmax]
        lamb = lamb[lamb <= lambmax]
    else:
        lambmax = lamb.max()
    inds = np.argsort(lamb)
    keys, lamb = keys[inds], lamb[inds]
    nlines = lamb.size

    # Check constraints
    if dconstraints is None:
        dconstraints =  defconst

    # Check same_spectrum
    if same_spectrum is None:
        same_spectrum = _SAME_SPECTRUM
    if same_spectrum is True:
        if type(nspect) not in [int, np.int]:
            msg = "Please provide nspect if same_spectrum = True"
            raise Exception(msg)
        if dlamb is None:
            dlamb = [np.nanmin(lamb) if lambmin is None else lambmin,
                     np.nanmax(lamb) if lambmax is None else lambmax]
            dlamb = min(2*np.diff(dlamb), np.nanmin(lamb))

    # ------------------------
    # Check keys
    # ------------------------

    # Check dconstraints keys
    lk = sorted(_DCONSTRAINTS.keys())
    c0= (isinstance(dconstraints, dict)
         and all([k0 in lk for k0 in dconstraints.keys()]))
    if not c0:
        raise Exception(msg)

    # copy to avoid modifying reference
    dconstraints = copy.deepcopy(dconstraints)

    dinput = {}
    # ------------------------
    # Check / format double
    # ------------------------
    dinput['double'] = dconstraints.get('double', defconst['double'])
    if type(dinput['double']) is not bool:
        raise Exception(msg)

    # ------------------------
    # Check / format width, shift, amp (groups with posssible ratio)
    # ------------------------
    for k0 in ['amp', 'width', 'shift']:
        dinput[k0] = _width_shift_amp(dconstraints.get(k0, defconst[k0]),
                                      keys=keys, nlines=nlines, dlines=dlines)

    # ------------------------
    # mz, symb, ion
    # ------------------------
    mz = np.array([dlines[k0].get('m', np.nan) for k0 in keys])
    symb = np.array([dlines[k0].get('symbol', k0) for k0 in keys])
    ion = np.array([dlines[k0].get('ION', '?') for k0 in keys])

    # ------------------------
    # same_spectrum
    # ------------------------
    if same_spectrum is True:
        keysadd = np.array([[kk+'_bis{:04.0f}'.format(ii) for kk in keys]
                            for ii in range(1, nspect)]).ravel()
        lamb = (dlamb*np.arange(0, nspect)[:, None] + lamb[None, :])
        keys = np.r_[keys, keysadd]

        for k0 in ['amp', 'width', 'shift']:
            # Add other lines to original group
            keyk = dinput[k0]['keys']
            offset = np.tile(dinput[k0]['offset'], nspect)
            if k0 == 'shift':
                ind = np.tile(dinput[k0]['ind'], (1, nspect))
                coefs = (dinput[k0]['coefs'] * lamb[0, :] / lamb).ravel()
            else:
                coefs = np.tile(dinput[k0]['coefs'], nspect)
                keysadd = np.array([[kk+'_bis{:04.0f}'.format(ii)
                                     for kk in keyk]
                                    for ii in range(1, nspect)]).ravel()
                ind = np.zeros((keyk.size*nspect, nlines*nspect))
                for ii in range(nspect):
                    i0, i1 = ii*keyk.size, (ii+1)*keyk.size
                    j0, j1 = ii*nlines, (ii+1)*nlines
                    ind[i0:i1, j0:j1] = dinput[k0]['ind']
                keyk = np.r_[keyk, keysadd]
            dinput[k0]['keys'] = keyk
            dinput[k0]['ind'] = ind
            dinput[k0]['coefs'] = coefs
            dinput[k0]['offset'] = offset
        nlines *= nspect
        lamb = lamb.ravel()

        # update mz, symb, ion
        mz = np.tile(mz, nspect)
        symb = np.tile(symb, nspect)
        ion = np.tile(ion, nspect)

    # ------------------------
    # add to dinput
    # ------------------------
    dinput['keys'] = keys
    dinput['lines'] = lamb
    dinput['nlines'] = nlines

    dinput['mz'] = mz
    dinput['symb'] = symb
    dinput['ion'] = ion

    dinput['same_spectrum'] = same_spectrum
    dinput['same_spectrum_nspect'] = nspect
    dinput['same_spectrum_dlamb'] = dlamb

    dinput['Ti'] = dinput['width']['ind'].shape[0] < nlines
    dinput['vi'] = dinput['shift']['ind'].shape[0] < nlines

    # Add boundaries
    dinput['lambminmax'] = (lambmin, lambmax)
    return dinput


def multigausfit1d_from_dlines_ind(dinput=None):
    """ Return the indices of quantities in x to compute y """

    # indices
    # General shape: [bck, amp, widths, shifts]
    # If double [..., double_shift, double_ratio]
    # Excpet for bck, all indices should render nlines (2*nlines if double)
    dind = {'bck': {'x': np.r_[0]}, 'dshift': None, 'dratio': None}
    nn = dind['bck']['x'].size
    inddratio, inddshift = None, None
    for k0 in ['amp', 'width', 'shift']:
        lnl = np.sum(dinput[k0]['ind'], axis=1).astype(int)
        dind[k0] = {'x': nn + np.arange(0, dinput[k0]['ind'].shape[0]),
                    'lines': nn + np.argmax(dinput[k0]['ind'], axis=0),
                    'jac': [tuple(dinput[k0]['ind'][ii, :].nonzero()[0])
                            for ii in range(dinput[k0]['ind'].shape[0])]}
        nn += dind[k0]['x'].size

    sizex = dind['shift']['x'][-1] + 1
    indx = np.r_[dind['bck']['x'], dind['amp']['x'],
                 dind['width']['x'], dind['shift']['x']]
    assert np.all(np.arange(0, sizex) == indx)

    # check if double
    if dinput['double']:
        dind['dshift'] = -2
        dind['dratio'] = -1
        sizex += 2

    dind['sizex'] = sizex
    dind['shapey1'] = dind['bck']['x'].size + dinput['nlines']

    # Ref line for amp (for x0)
    amp_x0 = np.zeros((dinput['amp']['ind'].shape[0],), dtype=int)
    for ii in range(dinput['amp']['ind'].shape[0]):
        indi = dinput['amp']['ind'][ii, :].nonzero()[0]
        amp_x0[ii] = indi[np.argmin(np.abs(dinput['amp']['coefs'][indi]-1.))]
    dind['amp_x0'] = amp_x0

    return dind


def multigausfit1d_from_dlines_scale(data, lamb,
                                     dscales=None,
                                     dinput=None,
                                     nspect=None,
                                     dind=None):
    scales = np.full((nspect, dind['sizex']), np.nan)
    Dlamb = dinput['lambminmax'][1] - dinput['lambminmax'][0]
    scales[:, dind['bck']['x'][0]] = np.maximum(np.nanmin(data, axis=1),
                                                np.nanmax(data, axis=1)/20)
    # amp
    for ii, ij in enumerate(dind['amp_x0']):
        indi = np.abs(lamb-dinput['lines'][ij]) < Dlamb/20.
        scales[:, dind['amp']['x'][ii]] = np.nanmax(data[:, indi], axis=1)

    # width and shift
    lambm = dinput['lambminmax'][0]
    if dinput['same_spectrum'] is True:
        lambm2 = (lambm
                 + dinput['same_spectrum_dlamb']
                 * np.arange(0, dinput['same_spectrum_nspect']))
        nw0 = dind['width']['x'].size / dinput['same_spectrum_nspect']
        lambmw = np.repeat(lambm2, nw0)
        scales[:, dind['width']['x']] = (Dlamb/(20*lambmw))**2
    else:
        scales[:, dind['width']['x']] = (Dlamb/(20*lambm))**2
    scales[:, dind['shift']['x']] = Dlamb/(20*lambm)

    # Double
    if dinput['double'] is True:
        scales[:, dind['dratio']] = 1.
        scales[:, dind['dshift']] = Dlamb/(20*lambm)
    assert scales.ndim in [1, 2]
    assert scales.shape == (nspect, dind['sizex'])

    # Adjust with user-provided dscales
    lk = ['bck', 'amp', 'width', 'shift', 'dratio', 'dshift']
    if dscales is None:
        dscales = dict.fromkeys(lk, 1.)
    ltypes = [int, float, np.int, np.float]
    c0 = (isinstance(dscales, dict)
          and all([type(dscales.get(ss, 1.)) in ltypes for ss in lk]))
    if not c0:
        msg = ("Arg dscales must be a dict of the form (1. is default):\n"
               + "\t- {}\n".format(dict.fromkeys(lk, 1.))
               + "\t- provided: {}".format(dscales))
        raise Exception(msg)

    for kk in lk:
        if kk in ['dratio', 'dshift']:
            scales[:, dind[kk]] *= dscales.get(kk, 1.)
        else:
            scales[:, dind[kk]['x']] *= dscales.get(kk, 1.)
    return scales


def _checkformat_dx0(amp_x0=None, keys=None, dx0=None):
    # Check
    c0 = dx0 is None
    c1 = (isinstance(dx0, dict)
          and all([k0 in keys for k0 in dx0.keys()]))
    c2 = (isinstance(dx0, dict)
          and sorted(dx0.keys()) == ['amp']
          and isinstance(dx0['amp'], dict)
          and all([kk in keys for kk in dx0['amp'].keys()]))
    c3 = (isinstance(dx0, dict)
          and sorted(dx0.keys()) == ['amp']
          and isinstance(dx0['amp'], np.ndarray))
    if not any([c0, c1, c2]):
        msg = ("dx0 must be a dict of the form:\n"
               + "\t{k0: {'amp': float},\n"
               + "\t k1: {'amp': float},\n"
               + "\t ...,\n"
               + "\t kn: {'amp': float}\n"
               + "where [k0, k1, ..., kn] are keys of spectral lines")
        raise Exception(msg)

    # Build
    if c0:
        dx0 = {'amp': np.ones((amp_x0.size,))}
    elif c1:
        coefs = np.array([dx0.get(keys[ii], {'amp': 1.}).get('amp', 1.)
                          for ii in amp_x0])
        dx0 = {'amp': coefs}
    elif c2:
        coefs = np.array([dx0['amp'].get(keys[ii], 1.)
                          for ii in amp_x0])
        dx0 = {'amp': coefs}
    elif c3:
        assert dx0['amp'].shape == (amp_x0.size,)
    return dx0


def multigausfit1d_from_dlines_x0(dind=None,
                                  lines=None, data=None, lamb=None,
                                  scales=None, double=None, dx0=None,
                                  chain=None, nspect=None, keys=None):
    # user-defined coefs on amplitude
    dx0 = _checkformat_dx0(amp_x0=dind['amp_x0'], keys=keys, dx0=dx0)

    # Each x0 should be understood as x0*scale
    x0_scale = np.full((nspect, dind['sizex']), np.nan)
    if chain is True:
        x0_scale[0, dind['amp']['x']] = dx0['amp']
        x0_scale[0, dind['bck']['x']] = 1.
        x0_scale[0, dind['width']['x']] = 0.4
        x0_scale[0, dind['shift']['x']] = 0.
        if double is True:
            x0_scale[0, dind['dratio']] = 0.7
            x0_scale[0, dind['dshift']] = 0.7
    else:
        x0_scale[:, dind['amp']['x']] = dx0['amp']
        x0_scale[:, dind['bck']['x']] = 1.
        x0_scale[:, dind['width']['x']] = 0.4
        x0_scale[:, dind['shift']['x']] = 0.
        if double is True:
            x0_scale[:, dind['dratio']] = 0.7
            x0_scale[:, dind['dshift']] = 0.7
    return x0_scale


def multigausfit1d_from_dlines_bounds(sizex=None, dind=None, double=None):
    # Each x0 should be understood as x0*scale
    xup = np.full((sizex,), np.nan)
    xlo = np.full((sizex,), np.nan)
    xup[dind['bck']['x']] = 2.
    xlo[dind['bck']['x']] = 0.
    xup[dind['amp']['x']] = 1
    xlo[dind['amp']['x']] = 0.
    xup[dind['width']['x']] = 1.
    xlo[dind['width']['x']] = 0.01
    xup[dind['shift']['x']] = 2.
    xlo[dind['shift']['x']] = -2.
    if double is True:
        xup[dind['dratio']] = 1.6
        xlo[dind['dratio']] = 0.4
        xup[dind['dshift']] = 2.
        xlo[dind['dshift']] = -2.
    bounds_scale = (xlo, xup)
    return bounds_scale


def multigausfit1d_from_dlines_funccostjac(lamb,
                                           dinput=None,
                                           dind=None,
                                           scales=None,
                                           jac=None):
    ibckx = dind['bck']['x']
    iax = dind['amp']['x']
    iwx = dind['width']['x']
    ishx = dind['shift']['x']
    idratiox = dind['dratio']
    idshx = dind['dshift']

    ial = dind['amp']['lines']
    iwl = dind['width']['lines']
    ishl = dind['shift']['lines']

    iaj = dind['amp']['jac']
    iwj = dind['width']['jac']
    ishj = dind['shift']['jac']

    coefsal = dinput['amp']['coefs']
    coefswl = dinput['width']['coefs']
    coefssl = dinput['shift']['coefs']

    offsetal = dinput['amp']['offset']
    offsetwl = dinput['width']['offset']
    offsetsl = dinput['shift']['offset']

    shape = (lamb.size, dind['shapey1'])

    def func_detail(x, lamb=lamb[:, None],
                    lines=dinput['lines'][None, :],
                    double=dinput['double'],
                    shape=shape,
                    ibckx=ibckx, ial=ial, iwl=iwl, ishl=ishl,
                    idratiox=idratiox, idshx=idshx,
                    coefsal=coefsal, coefswl=coefswl, coefssl=coefssl,
                    offsetal=offsetal, offsetwl=offsetwl, offsetsl=offsetsl,
                    scales=scales):
        y = np.full(shape, np.nan)
        xscale = x*scales
        y[:, ibckx] = xscale[ibckx]

        # lines
        amp = (xscale[ial]*coefsal + offsetal)[None, :]
        wi2 = (xscale[iwl]*coefswl + offsetwl)[None, :]
        shifti = (xscale[ishl]*coefssl + offsetsl)[None, :]
        y[:, 1:] = amp * np.exp(-(lamb/lines - (1 + shifti))**2 / (2*wi2))

        if double is True:
            ampd = amp*x[idratiox]
            shiftid = shifti + scales[ishl]*x[idshx]
            y[:, 1:] += (ampd
                         * np.exp(-(lamb/lines - (1 + shiftid))**2 / (2*wi2)))
        return y

    def cost(x, lamb=lamb[:, None],
             lines=dinput['lines'][None, :],
             double=dinput['double'],
             shape=shape,
             ibckx=ibckx, ial=ial, iwl=iwl, ishl=ishl,
             idratiox=idratiox, idshx=idshx,
             coefsal=coefsal, coefswl=coefswl, coefssl=coefssl,
             offsetal=offsetal, offsetwl=offsetwl, offsetsl=offsetsl,
             scales=scales, data=None):
        xscale = x*scales

        # lines & bck
        amp = (xscale[ial]*coefsal + offsetal)[None, :]
        wi2 = (xscale[iwl]*coefswl + offsetwl)[None, :]
        shifti = (xscale[ishl]*coefssl + offsetsl)[None, :]
        y = np.sum(amp * np.exp(-(lamb/lines - (1 + shifti))**2 / (2*wi2)),
                   axis=1) + xscale[ibckx]

        if double is True:
            shiftid = shifti + scales[ishl]*x[idshx]
            y += np.sum((amp*x[idratiox]
                         * np.exp(-(lamb/lines - (1 + shiftid))**2 / (2*wi2))),
                        axis=1)
        # ravel in case of multiple times same_spectrum
        return y - data

    if jac == 'call':
        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)
        def jac(x,
                lamb=lamb[:, None],
                lines=dinput['lines'][None, :],
                ibckx=ibckx,
                iax=iax, iaj=iaj, ial=ial,
                iwx=iwx, iwj=iwj, iwl=iwl,
                ishx=ishx, ishj=ishj, ishl=ishl,
                idratiox=idratiox, idshx=idshx,
                coefsal=coefsal[None, :],
                coefswl=coefswl[None, :],
                coefssl=coefssl[None, :],
                offsetal=offsetal[None, :],
                offsetwl=offsetwl[None, :],
                offsetsl=offsetsl[None, :],
                scales=None, double=dinput['double'], data=None):
            xscale = x*scales
            jac = np.full((lamb.size, x.size), np.nan)
            jac[:, ibckx] = scales[ibckx]

            # Assuming Ti = False and vi = False
            amp = (xscale[ial]*coefsal + offsetal)
            wi2 = (xscale[iwl]*coefswl + offsetwl)
            shifti = (xscale[ishl]*coefssl + offsetsl)
            beta = (lamb/lines - (1 + shifti)) / (2*wi2)
            alpha = -beta**2 * (2*wi2)
            exp = np.exp(alpha)

            quant = scales[ial] * coefsal * exp
            for ii in range(iax.size):
                jac[:, iax[ii]] = np.sum(quant[:, iaj[ii]], axis=1)
            quant = amp * (-alpha) * (scales[iwl]*coefswl / wi2) * exp
            for ii in range(iwx.size):
                jac[:, iwx[ii]] = np.sum(quant[:, iwj[ii]], axis=1)
            quant = amp * 2.*beta*scales[ishl]*coefssl * exp
            for ii in range(ishx.size):
                jac[:, ishx[ii]] = np.sum(quant[:, ishj[ii]], axis=1)
            if double is True:
                # Assuming Ti = False and vi = False
                ampd = amp*x[idratiox]*scales[idratiox]
                shiftid = shifti + scales[idshx]*x[idshx]
                betad = (lamb/lines - (1 + shiftid)) / (2*wi2)
                alphad = -betad**2 * (2*wi2)
                expd = np.exp(alphad)

                quant = scales[ial] * coefsal * expd
                for ii in range(iax.size):
                    jac[:, iax[ii]] += np.sum(quant[:, iaj[ii]], axis=1)
                quant = ampd * (-alphad) * (scales[iwl]*coefswl / wi2) * expd
                for ii in range(iwx.size):
                    jac[:, iwx[ii]] += np.sum(quant[:, iwj[ii]], axis=1)
                quant = ampd * 2.*betad*scales[ishl]*coefssl * expd
                for ii in range(ishx.size):
                    jac[:, ishx[ii]] += np.sum(quant[:, ishj[ii]], axis=1)

                jac[:, idratiox] = np.sum(amp * scales[idratiox] * expd,
                                          axis=1)
                # * coefssl => NO, line-specific
                jac[:, idshx] = np.sum(ampd * 2.*betad*scales[idshx] * expd,
                                       axis=1)
            return jac
    else:
        if jac not in ['2-point', '3-point']:
            msg = "jac should be in ['call', '2-point', '3-point']"
            raise Exception(msg)
        jac = jac

    return func_detail, cost, jac


def multigausfit1d_from_dlines(data, lamb,
                               lambmin=None, lambmax=None,
                               dinput=None, dx0=None, ratio=None,
                               dscales=None, x0_scale=None, bounds_scale=None,
                               method=None, max_nfev=None,
                               xtol=None, ftol=None, gtol=None,
                               chain=None, verbose=None,
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
    if chain is None:
        chain = True
    if jac is None:
        jac = 'call'
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox'], method
    if xtol is None:
        xtol = _TOL1D['x']
    if ftol is None:
        ftol = _TOL1D['f']
    if gtol is None:
        gtol = _TOL1D['g']
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None
    if verbose is None:
        verbose = 1
    if verbose == 2:
        verbscp = 2
    else:
        verbscp = 0

    c0 = lamb.ndim == 1 and np.all(np.argsort(lamb) == np.arange(0, lamb.size))
    if not c0:
        msg = ("lamb must be a 1d sorted array!\n"
               + "\t- provided: {}".format(lamb))
        raise Exception(msg)

    assert data.ndim in [1, 2] and lamb.size in data.shape
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != lamb.size:
        data = data.T
    nspect = data.shape[0]

    # ---------------------------
    # Prepare
    assert np.allclose(np.unique(lamb), lamb)
    nlines = dinput['nlines']

    # If same spectrum => consider a single data set
    if dinput['same_spectrum'] is True:
        lamb = (dinput['same_spectrum_dlamb']*np.arange(0, nspect)[:, None]
                + lamb[None, :]).ravel()
        data = data.ravel()[None, :]
        nspect = data.shape[0]
        chain = False

    # Get indices dict
    dind = multigausfit1d_from_dlines_ind(dinput)

    # Get scaling
    scales = multigausfit1d_from_dlines_scale(data, lamb,
                                              dscales=dscales,
                                              dinput=dinput,
                                              dind=dind,
                                              nspect=nspect)

    # Get initial guess
    x0_scale = multigausfit1d_from_dlines_x0(dind=dind,
                                             lines=dinput['lines'],
                                             data=data,
                                             lamb=lamb,
                                             scales=scales,
                                             double=dinput['double'],
                                             nspect=nspect,
                                             chain=chain,
                                             dx0=dx0, keys=dinput['keys'])

    # get bounds
    bounds_scale = multigausfit1d_from_dlines_bounds(dind['sizex'],
                                                     dind,
                                                     dinput['double'])

    # Get function, cost function and jacobian
    (func_detail,
     func_cost, jacob) = multigausfit1d_from_dlines_funccostjac(lamb,
                                                                dinput=dinput,
                                                                dind=dind,
                                                                jac=jac)

    # ---------------------------
    # Optimize

    # Initialize
    nlines = dinput['nlines']
    sol_detail = np.full((nspect, dind['shapey1'], lamb.size), np.nan)
    amp = np.full((nspect, nlines), np.nan)
    width2 = np.full((nspect, nlines), np.nan)
    shift = np.full((nspect, nlines), np.nan)
    coefs = np.full((nspect, nlines), np.nan)
    if dinput['double'] is True:
        dratio = np.full((nspect,), np.nan)
        dshift = np.full((nspect,), np.nan)
    else:
        dratio, dshift = None, None
    kTiev, vims = None, None
    if dinput['Ti'] is True:
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        kTiev = np.full((nspect, dinput['width']['ind'].shape[0]), np.nan)
    if dinput['vi'] is True:
        indvi = np.array([iit[0] for iit in dind['shift']['jac']])
        vims = np.full((nspect, dinput['shift']['ind'].shape[0]), np.nan)

    # Prepare msg
    if verbose > 0:
        msg = ("Loop in {} spectra with jac = {}\n".format(nspect, jac)
               + "time (s)    chin   nfev   njev   term"
               + "-"*20)
        print(msg)

    # Minimize
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    x = np.full((nspect, dind['sizex']), np.nan)
    for ii in range(nspect):
        t0 = dtm.datetime.now()     # DB
        res = scpopt.least_squares(func_cost, x0_scale[ii, :],
                                   jac=jacob, bounds=bounds_scale,
                                   method=method, ftol=ftol, xtol=xtol,
                                   gtol=gtol, x_scale='jac', f_scale=1.0,
                                   loss=loss, diff_step=None,
                                   tr_solver=None, tr_options={},
                                   jac_sparsity=None, max_nfev=max_nfev,
                                   verbose=verbscp, args=(),
                                   kwargs={'data': data[ii, :],
                                           'scales': scales[ii, :]})
        time[ii] = (dtm.datetime.now()-t0).total_seconds()
        cost[ii] = res.cost
        nfev[ii] = res.nfev

        if chain is True and ii < nspect-1:
            x0_scale[ii+1, :] = res.x
        if verbose > 0:
            dt = round(time[ii], ndigits=3)
            msg = " {}    {:4.2e}    {}   {}   {}".format(
                dt, np.sqrt(res.cost)/lamb.size, res.nfev,
                res.njev, res.message)
            print(msg)

        # Separate and reshape output
        x[ii, :] = res.x
        sol_detail[ii, ...] = func_detail(res.x, scales=scales[ii, :]).T

        # Get result in physical units: TBC !!!
        xscales = res.x*scales[ii, :]
        amp[ii, :] = xscales[dind['amp']['lines']] * dinput['amp']['coefs']
        width2[ii, :] = (xscales[dind['width']['lines']]
                         * dinput['width']['coefs'])
        shift[ii, :] = (xscales[dind['shift']['lines']]
                        * dinput['shift']['coefs']
                        * dinput['lines'])
        if dinput['double'] is True:
            dratio[ii] = res.x[dind['dratio']]
            dshift[ii] = xscales[dind['dshift']] # * lines
        if dinput['vi'] is True:
            vims[ii, :] = xscales[dind['shift']['lines'][indvi]] * scpct.c

    coefs = amp*dinput['lines']*np.sqrt(2*np.pi*width2)
    if dinput['Ti'] is True:
        # Get Ti in eV and vi in m/s
        ind = np.array([iit[0] for iit in dind['width']['jac']])
        kTiev = conv * width2[:, ind] * dinput['mz'][ind] * scpct.c**2

    # import pdb; pdb.set_trace()     # DB
    # Reshape in case of same_spectrum
    if dinput['same_spectrum'] is True:
        nspect0 = dinput['same_spectrum_nspect']
        def reshape_custom(aa, nspect0=nspect0):
            return aa.reshape((nspect0, int(aa.size/nspect0)))
        nlamb = int(lamb.size / nspect0)
        nlines = int((sol_detail.shape[1]-1)/nspect0)
        lamb = lamb[:nlamb]
        (data, amp, width2,
         coefs, shift) = [reshape_custom(aa)
                          for aa in [data, amp, width2, coefs, shift]]
        if dinput['double'] is True:
            dshift = np.full((nspect0,), dshift[0])
            dratio = np.full((nspect0,), dratio[0])
        if dinput['vi'] is True:
            vims = np.tile(vims, (nspect0, 1))
        if dinput['Ti'] is True:
            kTiev = reshape_custom(kTiev)

        nxbis = int(dind['bck']['x'].size
                    + (dind['amp']['x'].size + dind['width']['x'].size)/nspect0
                    + dind['shift']['x'].size)
        if dinput['double'] is True:
            nxbis += 2
        nb = dind['bck']['x'].size
        na = int(dind['amp']['x'].size/nspect0)
        nw = int(dind['width']['x'].size/nspect0)
        ns = dind['shift']['x'].size
        x2 = np.full((nspect0, nxbis), np.nan)
        x2[:, :nb] = x[0, dind['bck']['x']][None, :]
        x2[:, nb:nb+na] = reshape_custom(x[0, dind['amp']['x']])
        x2[:, nb+na:nb+na+nw] = reshape_custom(x[0, dind['width']['x']])
        x2[:, nb+na+nw:nb+na+nw+ns] = x[:, dind['shift']['x']]
        if dinput['double'] is True:
            x2[:, dind['dratio']] = x[:, dind['dratio']]
            x2[:, dind['dshift']] = x[:, dind['dshift']]
        x = x2
        sol_detail2 = np.full((nspect0, 1+nlines, nlamb), np.nan)
        # sol_detail.split(np.arange(1, nspect0)*nlamb, axis=-1)
        sol_detail2[:, 0, :] = sol_detail[0, 0, :nlamb]
        for ii in range(nspect0):
            ili0, ili1 = 1 + ii*nlines, 1 + (ii+1)*nlines
            ila0, ila1 = ii*nlamb, (ii+1)*nlamb
            sol_detail2[ii, 1:, :] = sol_detail[0, ili0:ili1, ila0:ila1]
        sol_detail = sol_detail2

    # Extract ratio of lines
    if ratio is not None:
        # Te can only be obtained as a proxy, units don't matter at this point
        if isinstance(ratio['up'], str):
            ratio['up'] = [ratio['up']]
        if isinstance(ratio['low'], str):
            ratio['low'] = [ratio['low']]
        assert len(ratio['up']) == len(ratio['low'])
        indup = np.array([(dinput['keys'] == uu).nonzero()[0][0]
                          for uu in ratio['up']])
        indlow = np.array([(dinput['keys'] == ll).nonzero()[0][0]
                           for ll in ratio['low']])
        ratio['value'] = coefs[:, indup] / coefs[:, indlow]
        ratio['str'] = ["{}/{}".format(dinput['symb'][indup[ii]],
                                       dinput['symb'][indlow[ii]])
                        for ii in range(len(ratio['up']))]

    # Create output dict
    dout = {'data': data, 'lamb': lamb,
            'x': x,
            'sol_detail': sol_detail,
            'sol': np.sum(sol_detail, axis=1),
            'Ti': dinput['Ti'], 'vi': dinput['vi'], 'double': dinput['double'],
            'width2': width2, 'shift': shift, 'amp': amp,
            'dratio': dratio, 'dshift': dshift, 'coefs': coefs,
            'kTiev': kTiev, 'vims': vims, 'ratio': ratio,
            'cost': cost, 'fun': res.fun, 'active_mask': res.active_mask,
            'time': time, 'nfev': nfev, 'njev': res.njev, 'status': res.status,
            'msg': res.message, 'success': res.success}
    return dout


###########################################################
###########################################################
#
#           2d spectral fitting from dlines
#
###########################################################
###########################################################


def multigausfit2d_from_dlines_dbsplines(knots=None, deg=None, nbsplines=None,
                                         phimin=None, phimax=None):
    # Check / format input
    if deg is None:
        deg = _DEG
    if not (isinstance(deg, int) and deg <= 3):
        msg = "deg must be a int <= 3 (the degree of the bsplines to be used!)"
        raise Exception(msg)

    if nbsplines is None:
        nbsplines = _NBSPLINES
    if not isinstance(nbsplines, int):
        msg = "nbsplines must be a int (the degree of the bsplines to be used!)"
        raise Exception(msg)

    if knots is None:
        if phimin is None or phimax is None:
            msg = "Please provide phimin and phimax if knots is not provided!"
            raise Exception(msg)
        knots = np.linspace(phimin, phimax, nbsplines+1-deg)

    if not np.allclose(knots, np.unique(knots)):
        msg = "knots must be a vector of unique values!"
        raise Exception(msg)

    # Get knots for scipy (i.e.: with multiplicity)
    if deg > 0:
        knots_mult = np.r_[[knots[0]]*deg, knots,
                      [knots[-1]]*deg]
    else:
        knots_mult = knots
    nknotsperbs = 2 + deg
    nbs = knots.size - 1 + deg
    assert nbs == knots_mult.size - 1 - deg

    if deg == 0:
        ptsx0 = 0.5*(knots[:-1] + knots[1:])
    elif deg == 1:
        ptsx0 = knots
    elif deg == 2:
        num = (knots_mult[3:]*knots_mult[2:-1]
               - knots_mult[1:-2]*knots_mult[:-3])
        denom = (knots_mult[3:] + knots_mult[2:-1]
                 - knots_mult[1:-2] - knots_mult[:-3])
        ptsx0 = num / denom
    else:
        # To be derived analytically for more accuracy
        ptsx0 = np.r_[knots[0],
                      np.mean(knots[:2]),
                      knots[1:-1],
                      np.mean(knots[-2:]),
                      knots[-1]]
        msg = ("degree 3 not fully implemented yet!"
               + "Approximate values for maxima positions")
        warnings.warn(msg)
    assert ptsx0.size == nbs
    dbsplines = {'knots': knots, 'knots_mult': knots_mult,
                 'nknotsperbs': nknotsperbs, 'ptsx0': ptsx0,
                 'nbs': nbs, 'deg': deg}
    return dbsplines


def multigausfit2d_from_dlines_dinput(dlines=None,
                                      dconstraints=None,
                                      deg=None, nbsplines=None, knots=None,
                                      lambmin=None, lambmax=None,
                                      phimin=None, phimax=None,
                                      defconst=_DCONSTRAINTS):

    # ------------------------
    # Check / format basics
    # ------------------------

    # Select relevant lines (keys, lamb)
    keys = np.array([k0 for k0 in dlines.keys()])
    lamb = np.array([dlines[k0]['lambda'] for k0 in keys])
    if lambmin is not None:
        keys = keys[lamb >= lambmin]
        lamb = lamb[lamb >= lambmin]
    if lambmax is not None:
        keys = keys[lamb <= lambmax]
        lamb = lamb[lamb <= lambmax]
    inds = np.argsort(lamb)
    keys, lamb = keys[inds], lamb[inds]
    nlines = lamb.size

    # Check constraints
    if dconstraints is None:
        dconstraints =  defconst


    # ------------------------
    # Check keys
    # ------------------------

    # Check dconstraints keys
    lk = sorted(_DCONSTRAINTS.keys())
    c0= (isinstance(dconstraints, dict)
         and all([k0 in lk for k0 in dconstraints.keys()]))
    if not c0:
        raise Exception(msg)

    # copy to avoid modifying reference
    dconstraints = copy.deepcopy(dconstraints)

    dinput = {}

    # ------------------------
    # Check / format double
    # ------------------------

    dinput['double'] = dconstraints.get('double', defconst['double'])
    if type(dinput['double']) is not bool:
        raise Exception(msg)

    # ------------------------
    # Check / format width, shift, amp (groups with posssible ratio)
    # ------------------------
    for k0 in ['amp', 'width', 'shift']:
        dinput[k0] = _width_shift_amp(dconstraints.get(k0, defconst[k0]),
                                      keys=keys, nlines=nlines, dlines=dlines)

    # ------------------------
    # add mz, symb, ION, keys, lamb
    # ------------------------
    dinput['mz'] = np.array([dlines[k0].get('m', np.nan) for k0 in keys])
    dinput['symb'] = np.array([dlines[k0].get('symbol', k0) for k0 in keys])
    dinput['ion'] = np.array([dlines[k0].get('ION', '?') for k0 in keys])

    dinput['keys'] = keys
    dinput['lines'] = lamb
    dinput['nlines'] = nlines

    dinput['Ti'] = dinput['width']['ind'].shape[0] < nlines
    dinput['vi'] = dinput['shift']['ind'].shape[0] < nlines

    # Get dict of bsplines
    dinput.update(multigausfit2d_from_dlines_dbsplines(knots=knots,
                                                       deg=deg,
                                                       nbsplines=nbsplines,
                                                       phimin=phimin,
                                                       phimax=phimax))
    # Add boundaries
    dinput['phiminmax'] = (phimin, phimax)
    dinput['lambminmax'] = (lambmin, lambmax)
    return dinput


def multigausfit2d_from_dlines_ind(dinput=None):
    """ Return the indices of quantities in x to compute y """

    # indices
    # General shape: [bck, amp, widths, shifts]
    # If double [..., double_shift, double_ratio]
    # Excpet for bck, all indices should render nlines (2*nlines if double)
    nbs = dinput['nbs']
    dind = {'bck': {'x': np.arange(0, nbs)},
            'dshift': None, 'dratio': None}
    nn = dind['bck']['x'].size
    inddratio, inddshift = None, None
    for k0 in ['amp', 'width', 'shift']:
        # l0bs0, l0bs1, ..., l0bsN, l1bs0, ...., lnbsN
        ind = dinput[k0]['ind']
        lnl = np.sum(dinput[k0]['ind'], axis=1).astype(int)
        dind[k0] = {'x': (nn
                          + nbs*np.arange(0, ind.shape[0])[None, :]
                          + np.arange(0, nbs)[:, None]),
                    'lines': (nn
                              + nbs*np.argmax(ind, axis=0)[None, :]
                              + np.arange(0, nbs)[:, None]),
                    # TBF
                    'jac': [dinput[k0]['ind'][ii, :].nonzero()[0]
                            for ii in range(dinput[k0]['ind'].shape[0])]}
        nn += dind[k0]['x'].size

    sizex = dind['shift']['x'][-1, -1] + 1
    indx = np.r_[dind['bck']['x'], dind['amp']['x'].T.ravel(),
                 dind['width']['x'].T.ravel(), dind['shift']['x'].T.ravel()]
    assert np.allclose(np.arange(0, sizex), indx)

    # check if double
    if dinput['double'] is True:
        dind['dshift'] = -2
        dind['dratio'] = -1
        sizex += 2

    dind['sizex'] = sizex
    dind['nbck'] = 1

    # Ref line for amp (for x0)
    # TBC !!!
    amp_x0 = np.zeros((dinput['amp']['ind'].shape[0],), dtype=int)
    for ii in range(dinput['amp']['ind'].shape[0]):
        indi = dinput['amp']['ind'][ii, :].nonzero()[0]
        amp_x0[ii] = indi[np.argmin(np.abs(dinput['amp']['coefs'][indi]-1.))]
    dind['amp_x0'] = amp_x0

    return dind


def multigausfit2d_from_dlines_scale(data, lamb, phi,
                                     scales=None,
                                     dinput=None,
                                     dind=None, nspect=None):
    if scales is None:
        scales = np.full((nspect, dind['sizex']), np.nan)
        Dphi = dinput['phiminmax'][1] - dinput['phiminmax'][0]
        Dlamb = dinput['lambminmax'][1] - dinput['lambminmax'][0]
        lambm = dinput['lambminmax'][0]
        ibckx, iax = dind['bck']['x'], dind['amp']['x']
        iwx, isx = dind['width']['x'].ravel(), dind['shift']['x'].ravel()
        # Perform by sector
        nbs, nlines = dinput['nbs'], dinput['nlines']
        na = dinput['amp']['ind'].shape[0]
        for ii in range(nbs):
            ind = np.abs(phi-dinput['ptsx0'][ii]) < Dphi/20.
            for jj in range(nspect):
                indbck = data[jj, ind] < np.nanmean(data[jj, ind])
                scales[jj, ibckx[ii]] = np.nanmean(data[jj, ind][indbck])
            for jj in range(na):
                indl = dind['amp_x0'][jj]
                indj = ind & (np.abs(lamb-dinput['lines'][indl])<Dlamb/20.)
                if not np.any(indj):
                    lamb0 = dinput['lines'][indl]
                    msg = ("All nan in region scanned for scale:\n"
                           + "\t- amp[{}]\n".format(jj)
                           + "\t- bspline[{}]\n".format(ii)
                           + "\t- phi approx {}\n".format(dinput['ptsx0'][ii])
                           + "\t- lamb approx {}".format(lamb0))
                    raise Exception(msg)
                scales[:, iax[ii, jj]] = np.nanmax(data[:, indj], axis=1)
        scales[:, iwx] = (Dlamb/(20*lambm))**2
        scales[:, isx] = Dlamb/(50*lambm)
        if dinput['double'] is True:
            scales[:, dind['dratio']] = 1.
            scales[:, dind['dshift']] = Dlamb/(50*lambm)
    assert scales.ndim in [1, 2]
    if scales.ndim == 1:
        scales = np.tile(scales, (nspect, scales.size))
    assert scales.shape == (nspect, dind['sizex'])
    return scales


def multigausfit2d_from_dlines_x0(dind=None, nbs=None,
                                  double=None, dx0=None,
                                  nspect=None, keys=None):
    # user-defined coefs on amplitude
    dx0 = _checkformat_dx0(amp_x0=dind['amp_x0'], keys=keys, dx0=dx0)
    dx0['amp'] = np.repeat(dx0['amp'], nbs)

    # Each x0 should be understood as x0*scale
    x0_scale = np.full((nspect, dind['sizex']), np.nan)
    x0_scale[:, dind['amp']['x'].T.ravel()] = dx0['amp']
    x0_scale[:, dind['bck']['x']] = 1.
    x0_scale[:, dind['width']['x']] = 0.4
    x0_scale[:, dind['shift']['x']] = 0.
    if double is True:
        x0_scale[:, dind['dratio']] = 0.7
        x0_scale[:, dind['dshift']] = 0.7
    return x0_scale


def multigausfit2d_from_dlines_bounds(sizex=None, dind=None, double=None):
    # Each x0 should be understood as x0*scale
    xup = np.full((sizex,), np.nan)
    xlo = np.full((sizex,), np.nan)
    xup[dind['bck']['x']] = 10.
    xlo[dind['bck']['x']] = 0.
    xup[dind['amp']['x']] = 2.
    xlo[dind['amp']['x']] = 0.
    xup[dind['width']['x']] = 2.
    xlo[dind['width']['x']] = 0.01
    xup[dind['shift']['x']] = 1.
    xlo[dind['shift']['x']] = -1.
    if double is True:
        xup[dind['dratio']] = 1.6
        xlo[dind['dratio']] = 0.4
        xup[dind['dshift']] = 10.
        xlo[dind['dshift']] = -10.
    bounds_scale = (xlo, xup)
    return bounds_scale


def multigausfit2d_from_dlines_funccostjac(lamb, phi,
                                           dinput=None,
                                           dind=None,
                                           scales=None,
                                           jac=None):
    ibckx = dind['bck']['x']
    iax = dind['amp']['x']
    iwx = dind['width']['x']
    ishx = dind['shift']['x']
    idratiox = dind['dratio']
    idshx = dind['dshift']

    ial = dind['amp']['lines']
    iwl = dind['width']['lines']
    ishl = dind['shift']['lines']

    iaj = dind['amp']['jac']
    iwj = dind['width']['jac']
    ishj = dind['shift']['jac']

    coefsal = dinput['amp']['coefs']
    coefswl = dinput['width']['coefs']
    coefssl = dinput['shift']['coefs']

    def func_detail(x, phi=phi, lamb=lamb[:, None],
                    ibckx=ibckx,
                    ial=ial,
                    iwl=iwl,
                    ishl=ishl,
                    idratiox=idratiox,
                    idshx=idshx,
                    lines=dinput['lines'][None, :],
                    nlines=dinput['nlines'],
                    km=dinput['knots_mult'],
                    kpb=dinput['nknotsperbs'],
                    deg=dinput['deg'],
                    nbs=dinput['nbs'],
                    nbck=int(ibckx.size/dinput['nbs']),
                    scales=scales,
                    coefsal=coefsal[None, :],
                    coefswl=coefswl[None, :],
                    coefssl=coefssl[None, :],
                    double=dinput['double']):
        y = np.full((lamb.size, nbck+nlines, nbs), np.nan)
        xscale = x*scales

        # make sure iwl is 2D to get all lines at once
        shift = BSpline(km, xscale[ishl] * coefssl, deg,
                        extrapolate=False, axis=0)(phi)
        wi2 = BSpline(km, xscale[iwl] * coefswl, deg,
                      extrapolate=False, axis=0)(phi)
        exp = np.exp(-(lamb/lines - (1 + shift))**2 / (2*wi2))
        if double is True:
            # coefssl are line-specific, they do not affect dshift
            shiftd = shift + x[idshx]*scales[idshx]  # *coefssl
            expd = np.exp(-(lamb/lines - (1 + shiftd))**2 / (2*wi2))

        # Loop on individual bsplines for amp
        for ii in range(nbs):
            bs = BSpline.basis_element(km[ii:ii+kpb],
                                       extrapolate=False)(phi)
            indok = ~np.isnan(bs)
            bs = bs[indok]
            y[indok, 0, ii] = xscale[ibckx[ii]]*bs
            for jj in range(nlines):
                amp = bs * xscale[ial[ii, jj]] * coefsal[0, jj]
                y[indok, nbck+jj, ii] = amp * exp[indok, jj]
                if double is True:
                    y[indok, nbck+jj, ii] += (amp * x[idratiox]
                                              * expd[indok, jj])
        return y

    def cost(x, phi=phi, lamb=lamb[:, None],
             ibckx=ibckx,
             ial=ial,
             iwl=iwl,
             ishl=ishl,
             idratiox=idratiox,
             idshx=idshx,
             lines=dinput['lines'][None, :],
             km=dinput['knots_mult'],
             kpb=dinput['nknotsperbs'],
             deg=dinput['deg'],
             scales=scales,
             coefsal=coefsal[None, :],
             coefswl=coefswl[None, :],
             coefssl=coefssl[None, :],
             double=dinput['double'],
             data=None):

        xscale = x*scales

        # Background
        y = BSpline(km, xscale[ibckx], deg,
                    extrapolate=False, axis=0)(phi)

        # make sure iwl is 2D to get all lines at once
        amp = BSpline(km, xscale[ial] * coefsal, deg,
                      extrapolate=False, axis=0)(phi)
        wi2 = BSpline(km, xscale[iwl] * coefswl, deg,
                      extrapolate=False, axis=0)(phi)
        csh = xscale[ishl] * coefssl
        y += np.nansum((amp
                        * np.exp(-(lamb/lines
                                   - (1 + BSpline(km, csh, deg,
                                                  extrapolate=False,
                                                  axis=0)(phi)))**2
                                 / (2*wi2))),
                       axis=1)
        if double is True:
            csh = csh + x[idshx]*scales[ishl]*coefssl
            y += np.nansum((amp * x[idratiox]
                            * np.exp(-(lamb/lines
                                       - (1 + BSpline(km, csh, deg,
                                                      extrapolate=False,
                                                      axis=0)(phi)))**2
                                     / (2*wi2))),
                           axis=1)
        return y - data

    if jac == 'call':
        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)
        def jac(x, lamb=lamb[:, None], phi=phi,
                ibckx=ibckx,
                iax=iax, iaj=iaj, ial=ial,
                iwx=iwx, iwj=iwj, iwl=iwl,
                ishx=ishx, ishj=ishj, ishl=ishl,
                idratiox=idratiox, idshx=idshx,
                lines=dinput['lines'][None, :],
                km=dinput['knots_mult'],
                kpb=dinput['nknotsperbs'],
                deg=dinput['deg'],
                nbs=dinput['nbs'],
                coefsal=coefsal[None, :],
                coefswl=coefswl[None, :],
                coefssl=coefssl[None, :],
                scales=scales,
                double=dinput['double'],
                data=None):
            xscale = x*scales

            jac = np.zeros((lamb.size, x.size), dtype=float)

            # Intermediates

            # Loop on bs
            for ii in range(nbs):
                bs = BSpline.basis_element(km[ii:ii+kpb],
                                           extrapolate=False)(phi)
                indok = ~np.isnan(bs)
                bs = bs[indok][:, None]

                # Intermediates
                amp = BSpline(km, xscale[ial] * coefsal, deg,
                              extrapolate=False, axis=0)(phi[indok])
                wi2 = BSpline(km, xscale[iwl] * coefswl, deg,
                              extrapolate=False, axis=0)(phi[indok])
                shift = BSpline(km, xscale[ishl] * coefssl, deg,
                                extrapolate=False, axis=0)(phi[indok])
                beta = (lamb[indok]/lines - (1 + shift)) / (2*wi2)
                alpha = -beta**2 * (2*wi2)
                exp = np.exp(alpha)

                # Background
                jac[indok, ibckx[ii]] = bs[:, 0] * scales[ibckx[ii]]

                # amp
                for jj in range(len(iaj)):
                    ix = iax[ii, jj]
                    jac[indok, ix] = np.nansum(
                        (bs * scales[ix] * coefsal[0:1, iaj[jj]]
                         * exp[:, iaj[jj]]),
                        axis=1)

                # width2
                for jj in range(len(iwj)):
                    ix = iwx[ii, jj]
                    jac[indok, ix] = np.nansum(
                        (amp[:, iwj[jj]]
                         * (-alpha[:, iwj[jj]]
                            * bs * scales[ix] * coefswl[0:1, iwj[jj]]
                         / wi2[:, iwj[jj]])
                         * exp[:, iwj[jj]]),
                        axis=1)

                # shift
                for jj in range(len(ishj)):
                    ix = ishx[ii, jj]
                    jac[indok, ix] = np.nansum(
                        (amp[:, ishj[jj]]
                         * 2.*beta[:, ishj[jj]]
                         * bs * scales[ix] * coefssl[0:1, ishj[jj]]
                         * exp[:, ishj[jj]]),
                        axis=1)

                # double
                if double is True:
                    ampd = amp*x[idratiox]
                    # coefssl are line-specific, they do not affect dshift
                    shiftd = shift + x[idshx]*scales[idshx]  # *coefssl
                    betad = (lamb[indok]/lines - (1 + shiftd)) / (2*wi2)
                    alphad = -betad**2 * (2*wi2)
                    expd = np.exp(alphad)

                    # amp
                    for jj in range(len(iaj)):
                        ix = iax[ii, jj]
                        jac[indok, ix] += x[idratiox]*np.nansum(
                            (bs * scales[ix] * coefsal[0:1, iaj[jj]]
                             * expd[:, iaj[jj]]),
                            axis=1)

                    # width2
                    for jj in range(len(iwj)):
                        ix = iwx[ii, jj]
                        jac[indok, ix] += np.nansum(
                            (ampd[:, iwj[jj]]
                             * (-alphad[:, iwj[jj]]
                                * bs * scales[ix] * coefswl[0:1, iwj[jj]]
                             / wi2[:, iwj[jj]])
                             * expd[:, iwj[jj]]),
                            axis=1)

                    # shift
                    for jj in range(len(ishj)):
                        ix = ishx[ii, jj]
                        jac[indok, ix] += np.nansum(
                            (ampd[:, ishj[jj]]
                             * 2.*betad[:, ishj[jj]]
                             * bs * scales[ix] * coefssl[0:1, ishj[jj]]
                             * expd[:, ishj[jj]]),
                            axis=1)

                    # dratio
                    jac[indok, idratiox] = np.nansum(amp*expd, axis=1)

                    # dshift
                    jac[indok, idshx] = np.nansum(
                        ampd * 2.*betad*scales[idshx] * expd, axis=1)
            return jac
    else:
        if jac not in ['2-point', '3-point']:
            msg = "jac should be in ['call', '2-point', '3-point']"
            raise Exception(msg)
        jac = jac

    return func_detail, cost, jac


def multigausfit2d_from_dlines(data, lamb, phi,
                               dinput=None, dx0=None, ratio=None,
                               scales=None, x0_scale=None, bounds_scale=None,
                               method=None, max_nfev=None,
                               xtol=None, ftol=None, gtol=None,
                               chain=None, verbose=None,
                               loss=None, jac=None, npts=None):
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
    if chain is None:
        chain = True
    if jac is None:
        jac = 'call'
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox'], method
    if xtol is None:
        xtol = _TOL2D['x']
    if ftol is None:
        ftol = _TOL2D['f']
    if gtol is None:
        gtol = _TOL2D['g']
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None
    if verbose is None:
        verbose = 1
    if verbose == 2:
        verbscp = 2
    else:
        verbscp = 0
    if npts is None:
        npts = (2*dinput['deg']-1)*(dinput['knots'].size-1) + 1

    nspect = data.shape[0]

    # ---------------------------
    # Prepare
    nlines = dinput['nlines']

    # Get indices dict
    dind = multigausfit2d_from_dlines_ind(dinput)

    # Get scaling
    scales = multigausfit2d_from_dlines_scale(data, lamb, phi,
                                              dinput=dinput,
                                              dind=dind,
                                              scales=scales, nspect=nspect)

    # Get initial guess
    x0_scale = multigausfit2d_from_dlines_x0(dind=dind,
                                             double=dinput['double'],
                                             nspect=nspect, nbs=dinput['nbs'],
                                             dx0=dx0, keys=dinput['keys'])

    # get bounds
    bounds_scale = multigausfit2d_from_dlines_bounds(dind['sizex'],
                                                     dind,
                                                     dinput['double'])

    # Get function, cost function and jacobian
    (func_detail,
     func_cost, jacob) = multigausfit2d_from_dlines_funccostjac(lamb, phi,
                                                                dinput=dinput,
                                                                dind=dind,
                                                                jac=jac)


    # ---------------------------
    # Prepare output
    nlines = dinput['nlines']
    sol_x = np.full((nspect, dind['sizex']), np.nan)
    sol_tot = np.full((nspect, lamb.size), np.nan)
    success = np.full((nspect,), np.nan)
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    validity = np.zeros((nspect,), dtype=int)
    message = np.array(['' for ss in range(nspect)])
    if dinput['double'] is True:
        dratio = np.full((nspect,), np.nan)
        dshift_norm = np.full((nspect,), np.nan)
    else:
        dratio, dshift_norm = None, None
    pts = np.linspace(dinput['phiminmax'][0],
                      dinput['phiminmax'][1], npts, endpoint=True)
    kTiev, vims = None, None
    if dinput['Ti'] is True:
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        indTi = np.array([iit[0] for iit in dind['width']['jac']])
        kTiev = np.full((nspect, dinput['width']['ind'].shape[0], npts),
                        np.nan)
    if dinput['vi'] is True:
        # indvi = np.array([iit[0] for iit in dind['shift']['jac']])
        vims = np.full((nspect, dinput['shift']['ind'].shape[0], npts),
                       np.nan)
    if ratio is not None:
        # Te can only be obtained as a proxy, units don't matter at this point
        if isinstance(ratio['up'], str):
            ratio['up'] = [ratio['up']]
        if isinstance(ratio['low'], str):
            ratio['low'] = [ratio['low']]
        assert len(ratio['up']) == len(ratio['low'])
        ratio['indup'] = np.array([(dinput['keys'] == uu).nonzero()[0][0]
                                   for uu in ratio['up']])
        ratio['indlow'] = np.array([(dinput['keys'] == ll).nonzero()[0][0]
                                    for ll in ratio['low']])
        ratio['str'] = ["{}/{}".format(dinput['symb'][ratio['indup'][ii]],
                                       dinput['symb'][ratio['indlow'][ii]])
                        for ii in range(len(ratio['up']))]
        ratio['value'] = np.full((nspect, len(ratio['up']), npts), np.nan)


    # Prepare msg
    if verbose > 0:
        msg = ("Loop in {} spectra with jac = {}\n".format(nspect, jac)
               + "time (s)    cost   nfev   njev   term"
               + "-"*20)
        print(msg)

    # ---------------------------
    # Minimize
    t0 = dtm.datetime.now()     # DB
    for ii in range(nspect):
        if verbose > 0:
            msg = "Iteration {} / {}".format(ii+1, nspect)
            print(msg)
        try:
            t0i = dtm.datetime.now()     # DB
            res = scpopt.least_squares(func_cost, x0_scale[ii, :],
                                       jac=jacob, bounds=bounds_scale,
                                       method=method, ftol=ftol, xtol=xtol,
                                       gtol=gtol, x_scale=1.0, f_scale=1.0,
                                       loss=loss, diff_step=None,
                                       tr_solver=None, tr_options={},
                                       jac_sparsity=None, max_nfev=max_nfev,
                                       verbose=verbscp, args=(),
                                       kwargs={'data': data[ii, :],
                                               'scales': scales[ii, :]})
            if chain is True and ii < nspect-1:
                x0_scale[ii+1, :] = res.x

            # cost, message, time
            success[ii] = res.success
            cost[ii] = res.cost
            nfev[ii] = res.nfev
            message[ii] = res.message
            time[ii] = round((dtm.datetime.now()-t0i).total_seconds(),
                             ndigits=3)
            if verbose > 0:
                msg = " {}    {}    {}   {}   {}".format(time[ii],
                                                         round(res.cost),
                                                         res.nfev, res.njev,
                                                         res.message)
                print(msg)

            # Separate and reshape output
            sol_x[ii, :] = res.x
            sol_tot[ii, :] = func_cost(res.x, scales=scales[ii, :], data=0.)

            # Get result in physical units: TBC !!!
            if dinput['double'] is True:
                dratio[ii] = res.x[dind['dratio']]
                dshift_norm[ii] = (res.x[dind['dshift']]
                                   * scales[ii, dind['dshift']])
            if dinput['Ti'] is True:
                # Get Ti in eV
                width2 = BSpline(dinput['knots_mult'],
                                 (res.x[dind['width']['x']]
                                  * scales[ii, dind['width']['x']]),
                                 dinput['deg'],
                                 extrapolate=False, axis=0)(pts).T
                kTiev[ii, ...] = (conv * width2 * dinput['mz'][indTi][:, None]
                                  * scpct.c**2)
            if dinput['vi'] is True:
                # Get vi in m/s
                vims[ii, ...] = BSpline(dinput['knots_mult'],
                                        (res.x[dind['shift']['x']]
                                         * scales[ii, dind['shift']['x']]),
                                        dinput['deg'],
                                        extrapolate=False,
                                        axis=0)(pts).T * scpct.c
            if ratio is not None:
                # Te can only be obtained as a proxy, units don't matter
                cup = BSpline(dinput['knots_mult'],
                              (res.x[dind['amp']['lines']]
                               * scales[ii, dind['amp']['lines']]),
                              dinput['deg'],
                              extrapolate=False, axis=0)(pts)[:, ratio['indup']]
                clow = BSpline(dinput['knots_mult'],
                               (res.x[dind['amp']['lines']]
                                * scales[ii, dind['amp']['lines']]),
                               dinput['deg'],
                               extrapolate=False, axis=0)(pts)[:, ratio['indlow']]
                ratio['value'][ii, ...] = (cup / clow).T
        except Exception as err:
            validity[ii] = -1

    if verbose > 0:
        dt = round((dtm.datetime.now()-t0).total_seconds(), ndigits=3)
        msg = "Total computation time: {}".format(dt)


    # ---------------------------
    # Format output as dict
    dout = {'data': data, 'lamb': lamb, 'phi': phi,
            'sol_x': sol_x, 'sol_tot': sol_tot,
            'dinput': dinput, 'dind': dind, 'jac': jac,
            'pts_phi': pts, 'kTiev': kTiev, 'vims': vims, 'ratio': ratio,
            'time': time, 'success': success, 'validity': validity,
            'cost': cost, 'nfev': nfev, 'msg': message}
    return dout
