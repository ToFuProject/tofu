


# Built-in
import os
import warnings

# Common
import numpy as np
import scipy.optimize as scpopt
import scipy.constants as scpct
import matplotlib.pyplot as plt




_NPEAKMAX = 10



###########################################################
###########################################################
#
#           Preliminary
#       utility tools for spectral fitting
#
###########################################################
###########################################################



def remove_bck(x, y):
    #opt = np.polyfit(x, y, deg=0)
    opt = [np.nanmin(y)]
    return y-opt[0], opt[0]


def get_peaks(x, y, nmax=None):

    if nmax is None:
        nmax = _NPEAKMAX

    # Prepare
    ybis = np.copy(y)
    A = np.empty((nmax,),dtype=y.dtype)
    x0 = np.empty((nmax,),dtype=x.dtype)
    sigma = np.empty((nmax,),dtype=y.dtype)
    gauss = lambda xx, A, x0, sigma: A*np.exp(-(xx-x0)**2/sigma**2)
    def gauss_jac(xx, A, x0, sigma):
        jac = np.empty((xx.size,3),dtype=float)
        jac[:,0] = np.exp(-(xx-x0)**2/sigma**2)
        jac[:,1] = A*2*(xx-x0)/sigma**2 * np.exp(-(xx-x0)**2/sigma**2)
        jac[:,2] = A*2*(xx-x0)**2/sigma**3 * np.exp(-(xx-x0)**2/sigma**2)
        return jac

    dx = np.nanmin(np.diff(x))

    # Loop
    nn = 0
    while nn < nmax:
        ind = np.nanargmax(ybis)
        x00 = x[ind]
        if np.any(np.diff(ybis[ind:],n=2)>=0.):
            wp = min(x.size-1,
                     ind + np.nonzero(np.diff(ybis[ind:],n=2)>=0.)[0][0] + 1)
        else:
            wp = ybis.size-1
        if np.any(np.diff(ybis[:ind+1],n=2)>=0.):
            wn = max(0, np.nonzero(np.diff(ybis[:ind+1],n=2)>=0.)[0][-1] - 1)
        else:
            wn = 0
        width = x[wp]-x[wn]
        assert width>0.
        indl = np.arange(wn,wp+1)
        sig = np.ones((indl.size,))
        if (np.abs(np.mean(np.diff(ybis[ind:wp+1])))
            > np.abs(np.mean(np.diff(ybis[wn:ind+1])))):
            sig[indl<ind] = 1.5
            sig[indl>ind] = 0.5
        else:
            sig[indl<ind] = 0.5
            sig[indl>ind] = 1.5
        p0 = (ybis[ind],x00,width)#,0.)
        bounds = (np.r_[0.,x[wn],dx/2.],
                  np.r_[5.*ybis[ind],x[wp],5.*width])
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

def get_p0bounds(x, y, nmax=None):

    yflat, bck = remove_bck(x,y)
    A, x0, sigma = get_peaks(x, yflat, nmax=nmax)

    p0 = A.tolist() + x0.tolist() + sigma.tolist() + [bck]

    lx = [np.nanmin(x), np.nanmax(x)]
    Dx = np.diff(lx)
    dx = np.nanmin(np.diff(x))

    bA = (np.zeros(nmax,), np.full((nmax,),3.*np.nanmax(y)))
    bx0 = (np.full((nmax,),lx[0]-Dx/2.), np.full((nmax,),lx[1]+Dx/2.))
    bsigma = (np.full((nmax,),dx/2.), np.full((nmax,),Dx/2.))
    bbck0 = (0., np.nanmax(y))

    bounds = (np.r_[bA[0],bx0[0],bsigma[0], bbck0[0]],
              np.r_[bA[1],bx0[1],bsigma[1], bbck0[1]])
    if not np.all(bounds[0]<bounds[1]):
        msg = "Lower bounds must be < upper bounds !\n"
        msg += "    lower :  %s\n"+str(bounds[0])
        msg += "    upper :  %s\n"+str(bounds[1])
        raise Exception(msg)
    return p0, bounds


def get_func1d_all(n=5):
    def func_vect(x, A, x0, sigma, bck0):
        y = np.full((A.size+1, x.size), np.nan)
        for ii in range(A.size):
            y[ii,:] = A[ii]*np.exp(-(x-x0[ii])**2/sigma[ii]**2)
        y[-1,:] = bck0
        return y

    def func_sca(x, *args, n=n):
        A = np.r_[args[0:n]]
        x0 = np.r_[args[n:2*n]]
        sigma = np.r_[args[2*n:3*n]]
        bck0 = np.r_[args[3*n]]
        gaus = A[:,np.newaxis]*np.exp(-(x[np.newaxis,:]-x0[:,np.newaxis])**2/sigma[:,np.newaxis]**2)
        back = bck0
        return np.sum( gaus, axis=0) + back

    def func_sca_jac(x, *args, n=n):
        A = np.r_[args[0:n]][np.newaxis,:]
        x0 = np.r_[args[n:2*n]][np.newaxis,:]
        sigma = np.r_[args[2*n:3*n]][np.newaxis,:]
        bck0 = np.r_[args[3*n]]
        jac = np.full((x.size,3*n+1,), np.nan)
        jac[:,:n] = np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,n:2*n] = A*2*(x[:,np.newaxis]-x0)/(sigma**2) * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,2*n:3*n] = A*2*(x[:,np.newaxis]-x0)**2/sigma**3 * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,-1] = 1.
        return jac

    return func_vect, func_sca, func_sca_jac


def multiplegaussianfit(x, spectra, nmax=None, p0=None, bounds=None,
                        max_nfev=None, xtol=1.e-8, verbose=0,
                        percent=20, plot_debug=False):

    # Prepare
    if spectra.ndim==1:
        spectra = spectra.reshape((1,spectra.size))
    nt = spectra.shape[0]

    A = np.full((nt,nmax),np.nan)
    x0 = np.full((nt,nmax),np.nan)
    sigma = np.full((nt,nmax),np.nan)
    bck0 = np.full((nt,),np.nan)
    Astd = np.full((nt,nmax),np.nan)
    x0std = np.full((nt,nmax),np.nan)
    sigmastd = np.full((nt,nmax),np.nan)
    bck0std = np.full((nt,),np.nan)
    std = np.full((nt,),np.nan)

    # Prepare info
    if verbose is not None:
        print("----- Fitting spectra with {0} gaussians -----".format(nmax))
    nspect = spectra.shape[0]
    nstr = max(nspect//max(int(100/percent),1),1)

    # bounds and initial guess
    if p0 is None or bounds is None:
        p00, bounds0 = get_p0bounds(x, spectra[0,:], nmax=nmax)
    if p0 is None:
        p0 = p00
    if bounds is None:
        bounds = bounds0

    # Get fit
    func_vect, func_sca, func_sca_jac = get_func(nmax)
    lch = []
    for ii in range(0,nspect):

        if verbose is not None and ii%nstr==0:
            print("=> spectrum {0} / {1}".format(ii,nspect))

        try:
            popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                          jac=func_sca_jac,
                                          p0=p0, bounds=bounds,
                                          max_nfev=max_nfev, xtol=xtol,
                                          x_scale='jac',
                                          verbose=verbose)
        except Exception as err:
            msg = "    Convergence issue for {0} / {1}\n".format(ii,nspect)
            msg += "    => %s\n"%str(err)
            msg += "    => Resetting initial guess and bounds..."
            print(msg)
            try:
                p0, bounds = get_p0bounds(x, spectra[ii,:], nmax=nmax)
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
                continue


        A[ii,:] = popt[:nmax]
        x0[ii,:] = popt[nmax:2*nmax]
        sigma[ii,:] = popt[2*nmax:3*nmax]
        bck0[ii] = popt[3*nmax]

        stdi = np.sqrt(np.diag(pcov))
        Astd[ii,:] = stdi[:nmax]
        x0std[ii,:] = stdi[nmax:2*nmax]
        sigmastd[ii,:] = stdi[2*nmax:3*nmax]
        bck0std[ii] = stdi[3*nmax]
        std[ii] = np.sqrt(np.sum((spectra[ii,:]-func_sca(x,*popt))**2))

        p0[:] = popt[:]

        if plot_debug and ii in [0,1]:
            fit = func_vect(x, A[ii,:], x0[ii,:], sigma[ii,:], bck0[ii])

            plt.figure()
            ax0 = plt.subplot(2,1,1)
            ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)
            ax0.plot(x,spectra[ii,:], '.k',
                     x, np.sum(fit,axis=0), '-r')
            ax1.plot(x, fit.T)

            import ipdb         # DB
            ipdb.set_trace()    # DB

    return (A,Astd), (x0,x0std), (sigma,sigmastd), (bck0,bck0std), std, lch







###########################################################
###########################################################
#
#           From DataCam2D
#
###########################################################
###########################################################


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

def fit_spectra2d_fit_ellipses(x0):
    # Each lamb is associated to a cone
    # All cones have the same axis, but
    #   - different summits
    #   - different opening
    # All cones are intersected by the same unique plane (detector)
    #   => all have the same center and rotation, but
    #   - different minor and major radius
    ellip_C = [None, None]
    ellip_rot = None
    ellip_radii = np.full((2, x0.shape[1]), np.nan)

    return ellip_C, ellip_rot, ellip_radii

def get_func_x0_from_y():

    # General ellipse equation in cartesian coordinates
    ((x-x0)cos(theta) + (y-y0)*sin(theta))**2 / rx**2
    + ((x-x0)sin(theta) - (y-y0)*cos(theta))**2 / ry**2

    # Hence x = f(y) for x > x0


    return


def coord_transform(ang_in, ang_cone, xc, yc, rot, ang_plane, ang_rot, Z):
    x1 = None
    X2 = None
    return


def coord_transform(x, y, xc, yc, ang_rot, ang_plane, Z):
    """ Return the coordinates transform of (x,y) in (ang_cone, ang_in) """
    x1 = (x-xc)*np.cos(ang_rot)
    X2 = None
    X2 = None
    return







def get_func2d(y0, y1, x0_y, bspl_n, bspl_deg):
    knots = np.linspace(y0,y1, 6)
    bspliney = scpinterp.LSQUnivariateSpline()
    def func(x, y, ampy_coefs, sigy_coefs, bcky_coefs):
        amp_bs = scpinterp.BSpline(knots, ampy_coefs, k=bspl_deg,
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
    import ipdb             # DB
    ipdb.set_trace()        # DB

    func = get_func2d()




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

_indymod = {0: np.arange(0,195),
            1: np.arange(212,407),
            2: np.arange(424,619),
            3: np.arange(636,831),
            4: np.arange(848,1043),
            5: np.arange(1060,1255),
            6: np.arange(1272,1467)}

def get_binnedspectra(imp='FeXXV', mod=3, path='./',
                      dshots=_dshots, indymod=_indymod, save=True):

    # Prepare
    indy = indymod[mod]
    dimp = dict(dshots[imp])

    # Loop on shots
    lextra = ['Te0','ne0','dg16']
    ls = sorted(dimp.keys())
    for shot in ls:

        # Load data
        if 'tlim' in dimp[shot].keys():
            tlim = dimp[shot]['tlim']
        else:
            tlim = None
        try:
            xics, kh = tfw.SpectroX2D.load_data(shot,
                                                tlim=tlim, plot=False)
        except Exception as err:
            dimp[shot]['err'] = str(err)
            continue

        # Bin data
        data, t = xics._Ref['data'], xics._Ref['t']
        dimp[shot]['nt'] = t.size
        spectra = np.empty((dimp[shot]['nt'], 487), dtype=float)
        for ii in range(0,dimp[shot]['nt']):
            spectra[ii,:] = np.nanmean(data[ii,:].reshape(1467,487)[indy,:],
                                       axis=0)
        dimp[shot]['spectra'] = spectra
        dimp[shot]['t'] = t
        #dimp[shot]['date'] = IRFMtb.

        # Dextra
        for ss in lextra:
            try:
                indt = np.digitize(xics.dextra[ss]['t'], 0.5*(t[1:]+t[:-1]))
                val = np.empty((dimp[shot]['nt'],),dtype=float)
                std = np.empty((dimp[shot]['nt'],),dtype=float)
                ssum = np.empty((dimp[shot]['nt'],),dtype=float)
                for ii in range(0,dimp[shot]['nt']):
                    val[ii] = np.nanmean(xics.dextra[ss]['data'][indt==ii])
                    std[ii] = np.nanstd(xics.dextra[ss]['data'][indt==ii])
                    ssum[ii] = np.nansum(xics.dextra[ss]['data'][indt==ii])
                dimp[shot][ss] = {'mean':val, 'std':std, 'sum':ssum}
            except Exception as err:
                dimp[shot][ss] = {'mean':np.nan, 'std':np.nan, 'sum':np.nan}

    # Reshape dict for np.savez and pandas DataFrame
    nt = np.array([dimp[shot]['nt'] for shot in ls])
    nttot = np.sum(nt)
    ntcum = np.cumsum(nt)
    lk = ['shot','angle','spectra','t',
          'Te0-mean','Te0-std','ne0-mean','ne0-std',
          'dg16-sum']
    dk = {}
    for k in lk:
        shape = (nttot,487) if k=='spectra' else (nttot,)
        dk[k] = np.full(shape, np.nan)

    i0 = 0
    for ii in range(0,len(ls)):
        ind = np.arange(i0,i0+nt[ii])
        dk['shot'][ind] = ls[ii]
        dk['angle'][ind] = dimp[ls[ii]]['ang']
        dk['spectra'][ind,:] = dimp[ls[ii]]['spectra']
        dk['t'][ind] = dimp[ls[ii]]['t']
        dk['Te0-mean'][ind] = dimp[ls[ii]]['Te0']['mean']
        dk['Te0-std'][ind] = dimp[ls[ii]]['Te0']['std']
        dk['ne0-mean'][ind] = dimp[ls[ii]]['ne0']['mean']
        dk['ne0-std'][ind] = dimp[ls[ii]]['ne0']['std']
        dk['dg16-sum'][ind] = dimp[ls[ii]]['dg16']['sum']
        i0 = ntcum[ii]

    # Saving
    if save:
        name = '%s_spectra'%imp
        path = os.path.abspath(path)
        pfe = os.path.join(path,name+'.npz')
        try:
            np.savez(pfe, **dk)
            print("Saved in :", pfe)
        except:
            import ipdb
            ipdb.set_trace()
            pass
    return dk



####################################################################
#       spectral fit
####################################################################


def remove_bck(x, y):
    #opt = np.polyfit(x, y, deg=0)
    opt = [np.nanmin(y)]
    ybis = y - opt[0]
    return ybis, opt[0]


def get_peaks(x, y, nmax=10):

    # Prepare
    ybis = np.copy(y)
    A = np.empty((nmax,),dtype=y.dtype)
    x0 = np.empty((nmax,),dtype=x.dtype)
    sigma = np.empty((nmax,),dtype=y.dtype)
    gauss = lambda xx, A, x0, sigma: A*np.exp(-(xx-x0)**2/sigma**2)
    def gauss_jac(xx, A, x0, sigma):
        jac = np.empty((xx.size,3),dtype=float)
        jac[:,0] = np.exp(-(xx-x0)**2/sigma**2)
        jac[:,1] = A*2*(xx-x0)/sigma**2 * np.exp(-(xx-x0)**2/sigma**2)
        jac[:,2] = A*2*(xx-x0)**2/sigma**3 * np.exp(-(xx-x0)**2/sigma**2)
        return jac

    dx = np.nanmin(np.diff(x))

    # Loop
    nn = 0
    while nn<nmax:
        ind = np.nanargmax(ybis)
        x00 = x[ind]
        if np.any(np.diff(ybis[ind:],n=2)>=0.):
            wp = min(x.size-1,
                     ind + np.nonzero(np.diff(ybis[ind:],n=2)>=0.)[0][0] + 1)
        else:
            wp = ybis.size-1
        if np.any(np.diff(ybis[:ind+1],n=2)>=0.):
            wn = max(0, np.nonzero(np.diff(ybis[:ind+1],n=2)>=0.)[0][-1] - 1)
        else:
            wn = 0
        width = x[wp]-x[wn]
        assert width>0.
        indl = np.arange(wn,wp+1)
        sig = np.ones((indl.size,))
        if (np.abs(np.mean(np.diff(ybis[ind:wp+1])))
            > np.abs(np.mean(np.diff(ybis[wn:ind+1])))):
            sig[indl<ind] = 1.5
            sig[indl>ind] = 0.5
        else:
            sig[indl<ind] = 0.5
            sig[indl>ind] = 1.5
        p0 = (ybis[ind],x00,width)#,0.)
        bounds = (np.r_[0.,x[wn],dx/2.],
                  np.r_[5.*ybis[ind],x[wp],5.*width])
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

def get_p0bounds(x, y, nmax=10):

    yflat, bck = remove_bck(x,y)
    A, x0, sigma = get_peaks(x, yflat, nmax=nmax)

    p0 = A.tolist() + x0.tolist() + sigma.tolist() + [bck]

    lx = [np.nanmin(x), np.nanmax(x)]
    Dx = np.diff(lx)
    dx = np.nanmin(np.diff(x))

    bA = (np.zeros(nmax,), np.full((nmax,),3.*np.nanmax(y)))
    bx0 = (np.full((nmax,),lx[0]-Dx/2.), np.full((nmax,),lx[1]+Dx/2.))
    bsigma = (np.full((nmax,),dx/2.), np.full((nmax,),Dx/2.))
    bbck0 = (0., np.nanmax(y))

    bounds = (np.r_[bA[0],bx0[0],bsigma[0], bbck0[0]],
              np.r_[bA[1],bx0[1],bsigma[1], bbck0[1]])
    if not np.all(bounds[0]<bounds[1]):
        msg = "Lower bounds must be < upper bounds !\n"
        msg += "    lower :  %s\n"+str(bounds[0])
        msg += "    upper :  %s\n"+str(bounds[1])
        raise Exception(msg)
    return p0, bounds


def get_func(n=5):
    def func_vect(x, A, x0, sigma, bck0):
        y = np.full((A.size+1, x.size), np.nan)
        for ii in range(A.size):
            y[ii,:] = A[ii]*np.exp(-(x-x0[ii])**2/sigma[ii]**2)
        y[-1,:] = bck0
        return y

    def func_sca(x, *args, n=n):
        A = np.r_[args[0:n]]
        x0 = np.r_[args[n:2*n]]
        sigma = np.r_[args[2*n:3*n]]
        bck0 = np.r_[args[3*n]]
        gaus = A[:,np.newaxis]*np.exp(-(x[np.newaxis,:]-x0[:,np.newaxis])**2/sigma[:,np.newaxis]**2)
        back = bck0
        return np.sum( gaus, axis=0) + back

    def func_sca_jac(x, *args, n=n):
        A = np.r_[args[0:n]][np.newaxis,:]
        x0 = np.r_[args[n:2*n]][np.newaxis,:]
        sigma = np.r_[args[2*n:3*n]][np.newaxis,:]
        bck0 = np.r_[args[3*n]]
        jac = np.full((x.size,3*n+1,), np.nan)
        jac[:,:n] = np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,n:2*n] = A*2*(x[:,np.newaxis]-x0)/(sigma**2) * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,2*n:3*n] = A*2*(x[:,np.newaxis]-x0)**2/sigma**3 * np.exp(-(x[:,np.newaxis]-x0)**2/sigma**2)
        jac[:,-1] = 1.
        return jac

    return func_vect, func_sca, func_sca_jac


def multiplegaussianfit(x, spectra, nmax=10, p0=None, bounds=None,
                        max_nfev=None, xtol=1.e-8, verbose=0,
                        percent=20, plot_debug=False):

    # Prepare
    if spectra.ndim==1:
        spectra = spectra.reshape((1,spectra.size))
    nt = spectra.shape[0]

    A = np.full((nt,nmax),np.nan)
    x0 = np.full((nt,nmax),np.nan)
    sigma = np.full((nt,nmax),np.nan)
    bck0 = np.full((nt,),np.nan)
    Astd = np.full((nt,nmax),np.nan)
    x0std = np.full((nt,nmax),np.nan)
    sigmastd = np.full((nt,nmax),np.nan)
    bck0std = np.full((nt,),np.nan)
    std = np.full((nt,),np.nan)

    # Prepare info
    if verbose is not None:
        print("----- Fitting spectra with {0} gaussians -----".format(nmax))
    nspect = spectra.shape[0]
    nstr = max(nspect//max(int(100/percent),1),1)

    # bounds and initial guess
    if p0 is None or bounds is None:
        p00, bounds0 = get_p0bounds(x, spectra[0,:], nmax=nmax)
    if p0 is None:
        p0 = p00
    if bounds is None:
        bounds = bounds0

    # Get fit
    func_vect, func_sca, func_sca_jac = get_func(nmax)
    lch = []
    for ii in range(0,nspect):

        if verbose is not None and ii%nstr==0:
            print("=> spectrum {0} / {1}".format(ii,nspect))

        try:
            popt, pcov = scpopt.curve_fit(func_sca, x, spectra[ii,:],
                                          jac=func_sca_jac,
                                          p0=p0, bounds=bounds,
                                          max_nfev=max_nfev, xtol=xtol,
                                          x_scale='jac',
                                          verbose=verbose)
        except Exception as err:
            msg = "    Convergence issue for {0} / {1}\n".format(ii,nspect)
            msg += "    => %s\n"%str(err)
            msg += "    => Resetting initial guess and bounds..."
            print(msg)
            try:
                p0, bounds = get_p0bounds(x, spectra[ii,:], nmax=nmax)
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
                continue


        A[ii,:] = popt[:nmax]
        x0[ii,:] = popt[nmax:2*nmax]
        sigma[ii,:] = popt[2*nmax:3*nmax]
        bck0[ii] = popt[3*nmax]

        stdi = np.sqrt(np.diag(pcov))
        Astd[ii,:] = stdi[:nmax]
        x0std[ii,:] = stdi[nmax:2*nmax]
        sigmastd[ii,:] = stdi[2*nmax:3*nmax]
        bck0std[ii] = stdi[3*nmax]
        std[ii] = np.sqrt(np.sum((spectra[ii,:]-func_sca(x,*popt))**2))

        p0[:] = popt[:]

        if plot_debug and ii in [0,1]:
            fit = func_vect(x, A[ii,:], x0[ii,:], sigma[ii,:], bck0[ii])

            plt.figure()
            ax0 = plt.subplot(2,1,1)
            ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)
            ax0.plot(x,spectra[ii,:], '.k',
                     x, np.sum(fit,axis=0), '-r')
            ax1.plot(x, fit.T)

            import ipdb         # DB
            ipdb.set_trace()    # DB

    return (A,Astd), (x0,x0std), (sigma,sigmastd), (bck0,bck0std), std, lch



def add_gaussianfits(dimp, nmax=10, verbose=0, percent=20,
                     path='./', save=False):
    assert type(dimp) in [str,dict]

    # Prepare
    if type(dimp) is str:
        imp = str(dimp)
        if '_' in imp:
            imp = imp.split('_')[0]
        dimp = dict(np.load(dimp+'_spectra.npz'))
        inds = np.argsort(dimp['angle'])
        for k in dimp.keys():
            if inds.size in dimp[k].shape:
                dimp[k] = dimp[k][inds] if dimp[k].ndim==1 else dimp[k][inds,:]
    else:
        imp = None
        save = False


    # Compute
    ind = np.arange(3,437)
    x = np.arange(0,dimp['spectra'].shape[1])
    spectra = dimp['spectra'][:,ind]
    A, x0, sigma, bck0, std, lch = multiplegaussianfit(x[ind], spectra,
                                                       nmax=nmax, percent=percent,
                                                       verbose=verbose)
    # Store
    dimp['nmax'] = nmax
    dimp['indch'] = np.r_[lch]
    dimp['x'] = x
    dimp['ind'] = ind
    dimp['A'] = A[0]
    dimp['A-std'] = A[1]
    dimp['x0'] = x0[0]
    dimp['x0-std'] = x0[1]
    dimp['sigma'] = sigma[0]
    dimp['sigma-std'] = sigma[1]
    dimp['bck0'] = bck0[0]
    dimp['bck0-std'] = bck0[1]
    dimp['std'] = std
    if imp is not None:
        dimp['imp'] = imp

    if save:
        name = '{0}_fitted{1}'.format(imp,nmax)
        path = os.path.abspath(path)
        pfe = os.path.join(path,name+'.npz')
        np.savez(pfe, **dimp)
        print("Saved in :", pfe)

    return dimp



def plot_gaussianfits(dimp, ind):

    # Prepare
    x = dimp['x']
    spectra = dimp['spectra'][ind,:]
    func_vect, func_sca, func_sca_jac = get_func(dimp['nmax'])
    fit = func_vect(x, dimp['A'][ind,:], dimp['x0'][ind,:],
                    dimp['sigma'][ind,:], dimp['bck0'][ind])

    # Plot
    plt.figure();
    ax0 = plt.subplot(2,1,1)
    ax1 = plt.subplot(2,1,2, sharex=ax0, sharey=ax0)

    ax0.plot(x, spectra, '.k', label='spectrum')
    ax0.plot(x, np.sum(fit,axis=0), '-r', label='fit')
    ax1.plot(x, fit.T)

    ax1.set_xlabel(r'x')
    ax0.set_ylabel(r'data')
    ax1.set_ylabel(r'data')


def plot_allraw(dimp):

    # Prepare
    x = dimp['x']
    spectra = dimp['spectra']
    nspect = spectra.shape[0]
    spectranorm = spectra / np.nanmax(spectra,axis=1)[:,np.newaxis]

    # Plot
    plt.figure();
    ax0 = plt.subplot(2,1,1)
    ax1 = plt.subplot(2,1,2, sharex=ax0)

    ax0.imshow(spectranorm, cmap=plt.cm.viridis,
               extent=(x.min(),x.max(),0,nspect),
               origin='lower',
               interpolation='bilinear')


def extract_lines(dimp):

    nspect, nx = dimp['spectra'].shape
    if dimp['imp'] == 'ArXVII':
        dlines = {'w': {'range':np.arange(280,nx)},
                  'z': {'range':np.arange(0,200)}}
        for k in dlines.keys():
            spect = np.copy(dimp['spectra'])
            spect[:,dlines[k]['range']] = 0.
            dlines[k].update({'x':np.full((nspect,),np.nan),
                              'x0':np.full((nspect,),np.nan),
                              'A':np.full((nspect,),np.nan),
                              'sigma':np.full((nspect,),np.nan)})
            for ii in range(0,dimp['spectra'].shape[0]):
                xl = dimp['x'][np.argmax(spect[ii,:])]
                ind = np.argmin(np.abs(dimp['x0'][ii,:]-xl))
                dlines[k]['x'][ii] = xl
                dlines[k]['x0'][ii] = dimp['x0'][ii,ind]
                dlines[k]['A'][ii] = dimp['A'][ii,ind]
                dlines[k]['sigma'][ii] = dimp['sigma'][ii,ind]

    dimp.update(**dlines)
    return dimp
