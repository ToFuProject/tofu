## core_optics.py

        # interpolation over (phi, lamb) grid: irregular to regular
        # length of z array must be either len(x)*len(y) for row/columns coords
        # or len(z) == len(x) == len(y) for each point coord
        # TBF/TBC : NaNs problems for interpolation inside gap_lamb[0, ...]
        z = gap_lamb[0,...].T.copy()
        if split:
            z[ np.isnan(z) ] = 2.0*1e-13
        else:
            z[ np.isnan(z) ] = 2.57*1e-13

        nb = 97
        lamb_min = np.min(lamb[0, ...])
        lamb_max = np.max(lamb[0, ...])
        phi_min = np.min(phi[0, ...])
        phi_max = np.max(phi[0, ...])
        lamb_interv = np.linspace(lamb_min, lamb_max, 487)
        phi_interv = np.linspace(phi_min, phi_max, 1467)

        ind_ok = ~np.isnan(gap_lamb[0,...])
        indsort = np.argsort(lamb[0, ind_ok][::nb])
        lamb_interp = lamb[0, ind_ok][::nb][indsort]
        phi_interp = phi[0, ind_ok][::nb][indsort]

        interp_plus = scpinterp.interp2d(
            lamb_interp,
            phi_interp,
            gap_lamb[0, ind_ok][::nb][indsort],
            kind='linear',
        )
        lamb_interp, phi_interp = np.mgrid[
            lamb[0,...].min():lamb[0,...].max():487,
            phi[0,...].min():phi[0,...].max():1467,
        ]
        interp_plus = scpinterp.bisplrep(
            lamb_interp,
            phi_interp,
            gap_lamb[0, ind_ok][::nb][indsort],
            s=0,
        )
        z_plus = scpinterp.bisplev(lamb_interv, phi_interv, interp_plus)

        interp_minus = scpinterp.interp2d(
            lamb_interp,
            phi_interp,
            gap_lamb[1, ind_ok1][::nb][indsort],
            kind='linear',
        )

        lamb_interp, phi_interp = np.mgrid[
            lamb[1,...].min():lamb[1,...].max():487,
            phi[1,...].min():phi[1,...].max():1467,
        ]
        ind_ok1 = ~np.isnan(gap_lamb[1,...])
        indsort = np.argsort(lamb[0, ind_ok1][::nb])
        lamb_interp = lamb[0, ind_ok1][::nb][indsort]
        phi_interp = phi[0, ind_ok1][::nb][indsort]

        interp_minus = scpinterp.bisplrep(
            lamb_interp,
            phi_interp,
            gap_lamb[1, ind_ok1][::nb][indsort],
            s=0,
        )
        z_minus = scpinterp.bisplev(lamb_interv, phi_interv ,interp_minus)

    # gap = f(xi, xj)
    #----------------
    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(8, 11, **dmargin)
        ax = fig.add_subplot(gs[:, :2])
        ax1 = fig.add_subplot(gs[:, 3:5])
        ax2 = fig.add_subplot(gs[:, 6:8])
        ax3 = fig.add_subplot(gs[:, 9:11])
        ax.set_ylabel('Xj [m]', fontsize=14)
        ax.set_xlabel('Xi [m]', fontsize=14)
        ax1.set_xlabel('Xi [m]', fontsize=14)
        ax2.set_xlabel('Xi [m]', fontsize=14)
        ax3.set_xlabel('Xi [m]', fontsize=14)

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=12, weight='bold')

    if det.get('outline') is not None:
        ax.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax1.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax2.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax3.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )

    ax.set_title(r'$\alpha$=0/$\beta$=0', fontsize=14)
    ax1.set_title(r'$\alpha$=0/$\beta=\pi/60$', fontsize=14)
    ax2.set_title(r'$\alpha$=0/$\beta=\pi/30$', fontsize=14)
    ax3.set_title(r'$\alpha$=0/$\beta=\pi/20$', fontsize=14)

    extent = (
        np.min(xi), np.max(xi), np.min(xj), np.max(xj),
    )
    errmap = ax.imshow(
        gap_xi[0, 0, ...].T,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    errmap1 = ax1.imshow(
        gap_xi[0, 1, ...].T,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    errmap2 = ax2.imshow(
        gap_xi[0, 2, ...].T,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    errmap3 = ax3.imshow(
        gap_xi[0, 3, ...].T,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    ax.contour(
        xi,
        xj,
        gap_xi[0, 0, ...].T,
        levels=10,
        colors='w',
        linestyles='-',
        linewidths=1.,
    )
    ax1.contour(
        xi,
        xj,
        gap_xi[0, 1, ...].T,
        levels=10,
        colors='w',
        linestyles='-',
        linewidths=1.,
    )
    ax2.contour(
        xi,
        xj,
        gap_xi[0, 2, ...].T,
        levels=10,
        colors='w',
        linestyles='-',
        linewidths=1.,
    )
    ax3.contour(
        xi,
        xj,
        gap_xi[0, 3, ...].T,
        levels=10,
        colors='w',
        linestyles='-',
        linewidths=1.,
    )
    cbar = plt.colorbar(
        errmap,
        orientation="vertical",
        ax=ax,
    )
    cbar1 = plt.colorbar(
        errmap1,
        orientation="vertical",
        ax=ax1,
    )
    cbar2 = plt.colorbar(
        errmap2,
        orientation="vertical",
        ax=ax2,
    )
    cbar3 = plt.colorbar(
        errmap3,
        label="Computed dshift [m]",
        orientation="vertical",
        ax=ax3,
    )

##plot_optics.py

    # interpolated gap = f(lamb, xj)
    #-------------------------------

    fig3 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(6, 6, **dmargin)
    ax = fig3.add_subplot(gs[:, :2])
    ax1 = fig3.add_subplot(gs[:, 4:])

    ax.set_ylabel('phi [rad]', fontsize=14)
    ax1.set_ylabel('phi [rad]', fontsize=14)
    ax.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax1.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax.set_xlim(3.92*1e-10, 4.03*1e-10)
    ax1.set_xlim(3.92*1e-10, 4.03*1e-10)

    errmap = ax.pcolormesh(lamb_interv, phi_interv, z_plus, shading='flat',)
    errmap1 = ax1.pcolormesh(lamb_interv, phi_interv, z_minus, shading='flat',)

    z_plus = interp_plus(lamb_interv, phi_interv)
    z_minus = interp_minus(lamb_interv, phi_interv)

    errmap = ax.imshow(
        z_plus,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='auto',
    )
    errmap = ax.scatter(
        lamb[0, ...].flatten()[::50],
        phi[0,...].flatten()[::50],
        s=6,
        c=z_plus.flatten(),
        cmap='viridis',
        marker='s',
        edgecolors="None",
    )
    cbar = plt.colorbar(
        errmap,
        label="Gap (0/3 arcsec) [m]",
        orientation="vertical",
        ax=ax,
    )


    ax0.set_title('Iso-lamb and iso-phi at crystal summit')
    ax1.set_title(f'Focalization error on $\lambda$ [{err_lamb_units}]')
    ax0.set_ylabel('Xj [m]', fontsize=14)
    ax0.set_xlabel('Xi [m]', fontsize=14)
    ax1.set_ylabel('Xj [m]', fontsize=14)
    ax1.set_xlabel('Xi [m]', fontsize=14)
    if plot_phi:
        ax2 = fig.add_subplot(
            gs[0, 2], aspect='equal', sharex=ax0, sharey=ax0,
        )
        ax2.set_title(f'Focalization error on $\phi$ [{err_phi_units}]')
        ax2.set_ylabel('Xj [m]', fontsize=14)
        ax2.set_xlabel('Xi [m]', fontsize=14)

    if split:
        ax0.contour(xi, xj, lamb[0, ...].T, 10, cmap=cmap)
        ax0.contour(xi, xj, phi[0, ...].T, 10, cmap=cmap, ls='--')
        ax0.contour(xi, xj, lamb[1, ...].T, 10, cmap=cmap)
        ax0.contour(xi, xj, phi[1, ...].T, 10, cmap=cmap, ls='--')
    else:
        ax0.contour(xi, xj, lamb.T, 10, cmap=cmap)
        ax0.contour(xi, xj, phi.T, 10, cmap=cmap, ls='--')

    imlamb = ax1.imshow(
        err_lamb.T,
        extent=extent, aspect='equal',
        origin='lower', interpolation='nearest',
        vmin=vmin, vmax=vmax,
    )

    ax1.contour(
        xi,
        xj,
        err_lamb.T,
        levels=11,
        colors='w',
        linesstyles='-',
        linewidths=1.,
    )

    plt.colorbar(imlamb, ax=ax1)

    if plot_phi:
        imphi = ax2.imshow(
            err_phi.T,
            extent=extent, aspect='equal',
            origin='lower', interpolation='nearest',
            vmin=vmin, vmax=vmax,
        )
        ax2.contour(
            xi,
            xj,
            err_phi.T,
            levels=11,
            colors='w',
            linewstyles='-',
            linewidths=1.,
        )

        plt.colorbar(imphi, ax=ax2)

    if wintit is not False:
        fig.canvas.manager.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')

    if not plot_phi:
        return ax0, ax1
    else:
        return ax0, ax1, ax2

##core_optics
    def gap_ray_tracing(
        self,
        lamb=None, n=None,
        nphi=None, npts=None,
        det=None,
        xi=None, xj=None,
        use_non_parallelism=None,
        lpsi=None, ldtheta=None,
        split=None, direction=None,
        relation=None,
        strict=None,
        rocking=None,
        ax=None, dleg=None,
        fs=None, dmargin=None,
        wintit=None, tit=None,
        plot=None,
    ):
        """
        With the plot_line_on_det_tracing() method, and using either the whole
        crystal or a splitted into two pieces, plotting gap between ray-tracing
        for a crystal with non-parallelism up to 3 arcsec and a perfect one.
        Parameters:
        ----------
        - lamb: float
            Provide np.array of min size 1, in 1e-10 [m]
        - det: dict
            Detector of reference on which ray tracing is done
        - nphi: float
            Number of points constituting ray tracing
        - npts: float
            Number of gap points computed on xi pixels
        - use_non_parallelism: boolean
            True or False if non parallelism have to be taken into account
        - split: boolean
            True or False to split the crystal
        - direction: 'e1' or 'e2'
            direction of splitting
        - relation: boolean
            True or False to plot xi's gap according to chosen wavelengths and
            for npts
        """
        # Check inputs
        if lamb is None:
            lamb = self._dbragg['lambref']
        lamb = np.atleast_1d(lamb).ravel()
        nlamb = lamb.size
        if nphi is None:
            nphi = 1467
        if npts is None:
            npts = 1467
        if plot is None:
            plot = True
        if use_non_parallelism is None:
            use_non_parallelism = True
        if split is None:
            split = False
        if direction is None:
            direction = 'e1'
        if relation is None:
            relation = False

        # Building arrays of alpha angle values & for results
        alphas = np.linspace(0, 0, 2)
        alphas_split = np.linspace((3/60)*np.pi/180, -(3/60)*np.pi/180, 2)
        lambdas1 = np.full((alphas.size, lamb.size), np.nan)
        lambdas2 = lambdas1.copy()
        xis1 = np.full((alphas.size, nlamb, xj.size), np.nan)
        xis2 = xis1.copy()
        xjs1 = np.full((alphas.size, nlamb, xj.size), np.nan)
        xjs2 = xjs1.copy()

        # Splitting crystal
        if split:
            cryst1, cryst2 = self.split(direction=direction, nb=2)

        # Calling plot_line_on_det_tracing()
        # 1st case: entire crystal, 2nd case: splitted crystal
        # In both cases are taking a fixed one & another taking
        # alpha angle values
        for ii in list(range(alphas.size)):
            if not split:
                self.update_non_parallelism(alpha=alphas[ii], beta=0)
                (
                    lambdas1[ii, :], xis1[ii, :, :], xjs1[ii, :, :],
                ) = self.plot_line_on_det_tracing(
                    nphi=nphi,
                    lamb=lamb,
                    det=det,
                    use_non_parallelism=use_non_parallelism,
                    lpsi=lpsi, ldtheta=ldtheta,
                    strict=strict,
                    rocking=rocking,
                    plot=False,
                )[:3]
                self.update_non_parallelism(alpha=alphas_split[ii], beta=0)
                (
                    lambdas2[ii, :], xis2[ii, :, :], xjs2[ii, :, :],
                ) = self.plot_line_on_det_tracing(
                    nphi=nphi,
                    lamb=lamb,
                    det=det,
                    use_non_parallelism=use_non_parallelism,
                    lpsi=lpsi, ldtheta=ldtheta,
                    strict=strict,
                    rocking=rocking,
                    plot=False,
                )[:3]
            else:
                cryst1.update_non_parallelism(alpha=alphas[ii], beta=0)
                (
                    lambdas1[ii, :], xis1[ii, :, :], xjs1[ii, :, :],
                ) = cryst1.plot_line_on_det_tracing(
                    nphi=nphi,
                    lamb=lamb,
                    det=det,
                    use_non_parallelism=use_non_parallelism,
                    lpsi=lpsi, ldtheta=ldtheta,
                    strict=strict,
                    rocking=rocking,
                    plot=False,
                )[:3]
                cryst2.update_non_parallelism(alpha=alphas_split[ii], beta=0)
                (
                    lambdas2[ii, :], xis2[ii, :, :], xjs2[ii, :, :],
                ) = cryst2.plot_line_on_det_tracing(
                    nphi=nphi,
                    lamb=lamb,
                    det=det,
                    use_non_parallelism=use_non_parallelism,
                    lpsi=lpsi, ldtheta=ldtheta,
                    strict=strict,
                    rocking=rocking,
                    plot=False,
                )[:3]

        ## Computing difference between:
        ## xi's coordinates of crystal of reference xis1 (alpha = 0") &
        ## xi's coordinates of crystal with non parallelism xis2 (alpha=+/-3")
        n = int(ceil(nphi/npts))
        gap_xi = np.full((2, nlamb, npts), np.nan)
        nalpha = alphas.size
        for ii in range(nalpha):
            for jj in range(nlamb):
                gap_xi[ii, jj, :] = (
                    xis1[ii, jj, ::n] - xis2[ii, jj, ::n]
                )

        # Reset cryst angles
        self.update_non_parallelism(alpha=0, beta=0)

        # Plot gap_xi function of detector's height
        if plot:
            ax = _plot_optics.CrystalBragg_gap_ray_tracing(
                lamb, gap_xi,
                xis1=xis1, xis2=xis2,
                xjs1=xjs1, xjs2=xjs2,
                npts=npts, nlamb=nlamb, n=n,
                det=det, ax=ax,
                split=split,
                relation=relation,
                dleg=dleg, fs=fs,
                dmargin=dmargin,
                wintit=wintit, tit=tit,
            )
        else:
            return lamb, gap_xi


##plot_optics
def CrystalBragg_gap_ray_tracing(
    lamb, gap_xi,
    xis1, xis2,
    xjs1, xjs2,
    npts, nlamb=None, n=None,
    det=None,
    split=None,
    relation=None,
    ax=None, dleg=None,
    fs=None, dmargin=None,
    wintit=None, tit=None,
):

    # Check inputs
    #-------------

    if dleg is None:
        dleg = {'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0)}
    if fs is None:
        fs = (15, 13)
    if dmargin is None:
        dmargin = {'left': 0.06, 'right': 0.99,
            'bottom': 0.06, 'top': 0.92,
            'wspace': None, 'hspace': 0.4}

    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        if split:
            tit = (
                u"Gap between each wavelength arc,"
                u" with & without non-parallelism, crystal splitted"
            )
        else:
            tit = (
                u"Gap between each wavelength arc,"
                u" with & without non-parallelism"
            )

    dcolor = ['red', 'orange', 'yellow', 'green',
             'blue', 'pink', 'purple', 'brown', 'black',
             ]
    dmarkers = ['o', 'v', 's', 'x', '*', 'P', '+', 'p']
    dls = [
        '--', ':', '-.',
        '--', ':', '-.',
        '--', ':', '-.',
        '--', ':', '-.',
    ]
    dlab = [r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = 3"',
        r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"',
    ]

    # Plot
    #-----

    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(4, 4, **dmargin)
        ax = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1])
        ax2 = fig.add_subplot(gs[:, 2:])

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=12, weight='bold')

    if det.get('outline') is not None:
        ax.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax1.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax2.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )

    ## Plotting ray-tracing on det (ax2) and
    ## xi's gap on each case of non-parallelism (ax & ax1)
    for l in range(nlamb):
        lab = r'$\lambda$'+' = {:6.3f} A\n'.format(lamb[l]*1.e10)
        ax.plot(
            gap_xi[1, l, :],  # plot diff between 0" and -3" [arcsec]
            xjs1[0, l, ::n],  # xj det coordinates
            ls=dls[1], lw=3.,
            c=dcolor[l],
        )
        ax1.plot(
            gap_xi[0, l, :],  # plot diff between 0" and +3" [arcsec]
            xjs1[0, l, ::n],
            ls=dls[0], lw=3.,
            c=dcolor[l],
        )
        ax2.plot(
            xis1[0, l, :], xjs1[0, l, :],  # plot ray-tracing without non-para
            ls='-', lw=3.,
            c=dcolor[l],
            label=lab
        )
        for ii in range(2):
            ax2.plot(
                xis2[ii, l, :], xjs2[ii, l, :],  # plot rays with non-para
                ls=dls[ii], lw=3.,
                c=dcolor[l],
                label=dlab[ii]
            )

        ax.set_xlim(
            np.nanmin(gap_xi[1, ...])-5e-5, np.nanmax(gap_xi[1, ...])+5e-5,
        )
        ax1.set_xlim(
            np.nanmin(gap_xi[0, ...])-5e-5, np.nanmax(gap_xi[0, ...])+5e-5,
        )
        ax.set_ylabel('Xj [m]')
        ax.set_xlabel('Gap [m]')
        ax1.set_xlabel('Gap [m]')
        ax2.set_xlabel('Xi [m]')
        ax.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"'],
                  loc='lower center')
        ax1.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = +3"'],
                  loc='lower center')

    if dleg is not False:
        ax2.legend(**dleg)

    # plot relation between gap and wavelength, according xj positions
    if relation is True:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1,
            left=0.1, right=0.95,
            bottom=0.1, top=0.92,
            wspace=None, hspace=0.2,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax.set_xlabel(r'$\lambda$ [m]')
        ax.set_ylabel('Xj [m]')
        ax2.set_xlabel(r'$\lambda$ [m]')
        ax2.set_ylabel('Xj [m]')

        for aa in range(npts):
            ax.plot(
                lamb,
                gap_xi[0, :, aa],
                ls='-',#dls[aa],
                lw=1.,
                c=dcolor[0],
                ms=4,
            )
            ax2.plot(
                lamb,
                gap_xi[1, :, aa],
                ls='-',#dls[aa],
                lw=1.,
                c=dcolor[1],
                ms=4,
            )
        ax.set_xlim(
            np.nanmin(lamb)-2*1e-12, np.nanmax(lamb)+2*1e-12,
        )
        ax2.set_xlim(
            np.nanmin(lamb)-2*1e-12, np.nanmax(lamb)+2*1e-12,
        )
        ax.set_ylim(
            np.nanmin(gap_xi[0, ...])-1e-4, np.nanmax(gap_xi[0, ...])+1e-4,
        )
        ax2.set_ylim(
            np.nanmin(gap_xi[1, ...])-1e-4, np.nanmax(gap_xi[1, ...])+1e-4,
        )
        ax.set_ylabel('Gap [m]')
        ax2.set_xlabel(r'$\lambda$ [m]')
        ax2.set_ylabel('Gap [m]')
        ax.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = 3"'])
        ax2.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"'])

    return ax, ax1, ax2

## _rockingcurve.py


    ## Create axis dictionary
    ## ----------------------
    dax = {
        'reflectivity_perp': {'handle': ax00},
        'reflectivity_para': {'handle': ax10},
        'reflectivity_perp_zoom': {'handle': ax01},
        'reflectivity_para_zoom': {'handle': ax11},
        'shift_perp_zoom': {'handle': ax02},
        'shift_para_zoom': {'handle': ax12},
        'shift_perp': {'handle': ax03},
        'shift_para': {'handle': ax13},
        'curve': {'handle': ax2},
    }

    ## Instanciate a DataStock object
    ## ------------------------------
    st = datastock.DataStock()

    st.add_ref(key='nT', size=theta.size)
    st.add_ref(key='nalpha', size=alpha.size)
    st.add_ref(key='nangle', size=th[0, 0, 0, :].size)

    st.add_data(key='TD', data=TD, ref=('nT',))
    st.add_data(key='alpha', data=alpha, ref=('nalpha',))
    st.add_data(key='angle', data=th[0, 0, 0, :], ref=('nangle',))
    st.add_data(
        key='power_ratio_perp',
        data=power_ratio[0, :, :, :],
        ref=('nT', 'nalpha', 'angle'),
    )
    st.add_data(
        key='power_ratio_para',
        data=power_ratio[1, :, :, :],
        ref=('nT', 'nalpha', 'angle'),
    )
    st.add_data(
        key='theta_perp',
        data=dth[0, :, :, :],
        ref=('nT', 'nalpha', 'angle'),
    )
    st.add_data(
        key='theta_para',
        data=dth[1, :, :, :],
        ref=('nT', 'nalpha', 'angle'),
    )
    st.add_data(
        key='reflectivity_integrated_perp',
        data=rhg_perp,
        ref=('nT', 'nalpha'),
    )
    st.add_data(
        key='reflectivity_integrated_para',
        data=rhg_para,
        ref=('nT', 'nalpha'),
    )
    st.add_data(
        key='shift_perp',
        data=shift_perp,
        ref=('nT', 'nalpha'),
    )
    st.add_data(
        key='shift_para',
        data=shift_para,
        ref=('nT', 'nalpha'),
    )


    ## Define dgroup
    ## -----------------

    dgroup = {
        'g0': {
            'ref': ['nT'],
            'data': ['index'],
            'nmax': 1,
        },
        'g1': {
            'ref': ['nalpha'],
            'data': ['index'],
            'nmax': 1,
        },
    }

    ## Plot mobile part
    ## -----------------
    nmax = 10

    kax = 'curve'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        # Power ratio vs glancing angle
        for ii in range(nmax):
            l0, = ax.plot(
                th[0, 0, 0, :],
                power_ratio[0, 0, 0, :],
                c='k',
                ls='-',
            )
            st.add_mobile(
                key=f'curve_perp{ii}',
                handle=l0,
                ax=ax,
                ref=('nT', 'nalpha'),
                data=('theta_perp', 'power_ratio_perp'),
                dtype=['xdata', 'ydata'],
                ind=ii,
            )
    kax = 'reflectivity_perp'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        # Integrated reflectivity, normal component
        for ii in range(nmax):
            l0, = ax.plot(
                alpha[0],
                TD[0],
                marker='s',
                markeredgecolor='k',
                markerfacecolor='None',
            )
            st.add_mobile(
                key='m0',
                handle=l0,
                ax=ax, #dax['reflectivity_perp_zoom']['handle'],
                ref=(
                    'nalpha', 'nT', 'reflectivity_integrated_perp',
                    'reflectivity_integrated_perp.T',
                    'ntransparency', 'ntext',
                ),
                ind=ii,
            )

    kax = 'reflectivity_para'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        # Integrated reflectivity, parallel component
        for ii in range(nmax):
            l0, = ax.plot(
                alpha[0],
                TD[0],
                marker='s',
                markeredgecolor='k',
                markerfacecolor='None',
            )
            st.add_mobile(
                key='m0',
                handle=l0,
                ax=ax,
                ref=('DT', 'alpha'),
                ind=ii,
            )

    kax = 'shift_perp'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        # Shift from the reference RC, normal component
        for ii in range(nmax):
            l0, = ax.plot(
                alpha[0],
                TD[0],
                marker='s',
                markeredgecolor='k',
                markerfacecolor='None',
            )
            st.add_mobile(
                key='m0',
                handle=l0,
                ax=ax,
                ref=('DT', 'alpha'),
                ind=ii,
            )

    kax = 'shift_para'
    if dax.get(kax) is not None:
        ax = dax[kax]
        # Shift from the reference RC, parallel component
        for ii in range(nmax):
            l0, = ax.plot(
                alpha[0],
                TD[0],
                marker='s',
                markeredgecolor='k',
                markerfacecolor='None',
            )
            st.add_mobile(
                key='m0',
                handle=l0,
                ax=ax,
                ref=('DT', 'alpha'),
                ind=ii,
            )

    st.add_axes(**dax)

    st.setup_interactivity(dgroup=dgroup)
    st.disconnect_old()
    st.connect()

    return st

# tofu/spectro/_rockingcurve_def.py

# #############################################################################
# #############################################################################
#                                   _DCRYST
# #############################################################################
# #############################################################################

_DCRYST = {
    'alpha-Quartz': {
        'name': 'alpha-Quartz',
        'symbol': 'aQz',
        'atoms': ['Si', 'O'],
        'atomic number': [14., 8.],
        'number of atoms': [3., 6.],
        'meshtype': 'hexagonal',
        'mesh positions': {
            'Si': {
                'u': 0.465,
                'x': None,
                'y': None,
                'z': None,
            },
            'O': {
                'u': [0.415, 0.272, 0.120],
                'x': None,
                'y': None,
                'z': None,
            },
        },
        'mesh positions sources': 'R.W.G. Wyckoff, Crystal Structures (1963)',
        'Interatomic distances': {
            'a0': 4.91304,
            'c0': 5.40463,
        },
        'Interatomic distances comments' : 'at 25°C, unit = Angstroms',
        'Interatomic distances sources': 'R.W.G. Wyckoff, Crystal Structures',
        'Thermal expansion coefs': {
            'a0': 13.37e-6,
            'c0': 7.97e-6,
        },
        'Thermal expansion coefs comments': 'unit = [°C⁻1]',
        'Thermal expansion coefs sources': 'R.W.G. Wyckoff, Crystal Structures',
        'sin(theta)/lambda': {
            'Si': [
                0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
            'O': [
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
            ],
        },
        'sin(theta)/lambda sources':
            'Intern. Tables for X-Ray Crystallography, Vol.I,II,III,IV (1985)',
        'atomic scattering factor': {
            'Si': [
                12., 11., 9.5, 8.8, 8.3, 7.7, 7.27, 6.25, 5.3,
                4.45, 3.75, 3.15, 2.7, 2.35, 2.07, 1.87, 1.71, 1.6,
            ],
            'O': [
                9., 7.836, 5.756, 4.068, 2.968, 2.313, 1.934, 1.710, 1.566,
                1.462, 1.373, 1.294,
            ],
        },
        'atomic scattering factor sources':
            'Intern. Tables for X-Ray Crystallography, Vol.I,II,III,IV (1985)',
    },
    'Germanium': {
    },
}


# #############################################################################
# #############################################################################
#                         Atoms positions in mesh
# #############################################################################

# Positions comments from literature
# ---------------------------------

# Si and O positions for alpha-Quartz crystal
# xsi = np.r_[-u, u, 0.]
# ysi = np.r_[-u, 0., u]
# zsi = np.r_[1./3., 0., 2./3.]
# xo = np.r_[x, y - x, -y, x - y, y, -x]
# yo = np.r_[y, -x, x - y, -y, x, y - x]
# zo = np.r_[z, z + 1./3., z + 2./3., -z, 2./3. - z, 1./3. - z]

# Atoms positions for Germanium crystal
None

# Attribution to alpha-Quartz
# ---------------------------

# Position of the 3 Si atoms in the unit cell
# -------------------------------------------
# Quartz_110

uSi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['u'][0]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['x'] = np.r_[
    -uSi,
    uSi,
    0.
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['y'] = np.r_[
    -uSi,
    0.,
    uSi
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['z'] = np.r_[
    1./3.,
    0.,
    2./3.
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['N'] = np.size(
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['x']
)

# Quartz_102
_DCRYST['Quartz_102']['mesh']['positions']['Si']['x'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['x']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['y'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['y']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['z'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['z']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['N'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['N']
)

# Position of the 6 O atoms in the unit cell
# ------------------------------------------

# Quartz_110
uOx = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][0]
uOy = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][1]
uOz = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][2]
_DCRYST['Quartz_110']['mesh']['positions']['O']['x'] = np.r_[
    uOx,
    uOy - uOx,
    -uOy,
    uOx - uOy,
    uOy,
    -uOx
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['y'] = np.r_[
    uOy,
    -uOx,
    uOx - uOy,
    -uOy,
    uOx,
    uOy - uOx
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['z'] = np.r_[
    uOz,
    uOz + 1./3.,
    uOz + 2./3.,
    -uOz,
    2./3. - uOz,
    1./3. - uOz
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['N'] = np.size(
    _DCRYST['Quartz_110']['mesh']['positions']['O']['x']
)

# Quartz_102
_DCRYST['Quartz_102']['mesh']['positions']['O']['x'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['x']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['y'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['y']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['z'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['z']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['N'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['N']
)

# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: Quartz_110 and Quartz_102
# ---------------------------------------------------------------


# Same values for 110- and Quartz_102
a = _DCRYST['Quartz_110']['inter_atomic']['distances']['a0']
c = _DCRYST['Quartz_110']['inter_atomic']['distances']['c0']

h110 = _DCRYST['Quartz_110']['miller'][0]
k110 = _DCRYST['Quartz_110']['miller'][1]
l110 = _DCRYST['Quartz_110']['miller'][2]
h102 = _DCRYST['Quartz_102']['miller'][0]
k102 = _DCRYST['Quartz_102']['miller'][1]
l102 = _DCRYST['Quartz_102']['miller'][2]

_DCRYST['Quartz_110']['volume'] = hexa_volume(a, c)
_DCRYST['Quartz_110']['d_hkl'] = hexa_spacing(h110, k110, l110, a, c) * 1.e-10
_DCRYST['Quartz_102']['volume'] = hexa_volume(a, c)
_DCRYST['Quartz_102']['d_hkl'] = hexa_spacing(h102, k102, l102, a, c) * 1e-10


# ---------------------------------
# Attribution to Germanium crystals
# ---------------------------------

# ################
# compute xi(dshift) for all configurations, transfrom into lamb-scale and plot
# ################

lambda_range = np.linspace(3.93e-10, 4.00e-10, 8)
det = cryst.get_detector_ideal()
xi = cryst.dshift_analytic_variation(crystal='Quartz_110', len_cryst=0.083592, split=False, rcurve=cryst.dgeom['rcurve'], lamb=lambda_range, miscut=False, alpha=np.r_[(3/60)*np.pi/180.])
print(xi); print(xi.shape)
bragg, _ , wavel = cryst.get_lambbraggphi_from_ptsxixj_dthetapsi(xi=xi, xj=0, det=det, miscut=False, return_lamb=True)
print(wavel); print(wavel.shape)  # don't do the computations correctly,
# obtain the same wavelengths than lambda_range
wavel = wavel.ravel()
print(wavel); print(wavel.shape)
dlamb = lambda_range - wavel
print(dlamb); print(dlamb.shape)
plt.plot(lambda_range, dlamb, 'ro--', label='single crystal, no miscut')
plt.xlabel(r'Wavelength [$\AA$]', fontsize=12)
plt.ylabel(r'dshift [$\AA$]', fontsize=12)
plt.title('Computed dshift parameter in wavelength scale', fontsize=12)
plt.legend()

# ################
# plot from the dratio/dshift parameters computations for 3 lambda-scales (wn3,
# xy, kjz) the errorbars with mean and std
# ################

lint = np.r_[3.93, 3.962, 3.98, 3.985, 4.009]*1e-10
lpts = np.r_[((lint[1]+lint[0])/2), ((lint[2]+lint[1])/2),((lint[4]+lint[3])/2)]
drm = np.r_[dratio_wn3_mean, dratio_xy_mean, dratio_kjz_mean]
dsm = np.r_[dshift_wn3_mean, dshift_xy_mean, dshift_kjz_mean]
drs = np.r_[dratio_wn3_std, dratio_xy_std, dratio_kjz_std]
dss = np.r_[dshift_wn3_std, dshift_xy_std, dshift_kjz_std]
lscale = np.r_[(lint[1]-lint[0])/2, (lint[2]-lint[1])/2,(lint[4]-lint[3])/2]
## dratio
plt.figure()
plt.errorbar(x=lpts, y=drm, xerr=lscale, yerr=drs, fmt='--ro', ms=1, ecolor='b', elinewidth=2, capsize=10, capthick=2)
plt.xlabel(r'Wavelength [$\AA$]', fontsize=15); plt.ylabel(r'dratio [$\%$]', fontsize=15)
# write for wn3
plt.text(lpts[0], 0.72, r'$\mu_{wn3}$='+str(np.round(dratio_wn3_mean, 4)), fontsize=15)
plt.text(lpts[0], 0.71, r'$\sigma_{wn3}$='+str(np.round(dratio_wn3_std, 4)), fontsize=15)
# for xy
plt.text(lpts[1], 0.75, r'$\mu_{xy}$='+str(np.round(dratio_xy_mean, 4)), fontsize=15)
plt.text(lpts[1], 0.74, r'$\sigma_{xy}$='+str(np.round(dratio_xy_std, 4)), fontsize=15)
# for k,j,z
plt.text(lpts[2], 0.68, r'$\mu_{kjz}$='+str(np.round(dratio_kjz_mean, 4)), fontsize=15)
plt.text(lpts[2], 0.67, r'$\sigma_{kjz}$='+str(np.round(dratio_kjz_std, 4)), fontsize=15)
## dshift
plt.figure()
plt.errorbar(x=lpts, y=dsm, xerr=lscale, yerr=dss, fmt='--ro', ms=1, ecolor='b', elinewidth=2, capsize=10, capthick=2)
plt.xlabel('Wavelength [$\AA$]', fontsize=15); plt.ylabel('dshift [$\AA$]', fontsize=15)
# write for wn3
plt.text(lpts[0]+0.001e-10, 0.0005, r'$\mu_{wn3}$='+str(np.round(dshift_wn3_mean, 6)), fontsize=15)
plt.text(lpts[0]+0.001e-10, 0.00045, r'$\sigma_{wn3}$='+str(np.round(dshift_wn3_std, 6)), fontsize=15)
# for xy
plt.text(lpts[1]+0.001e-10, 0.0005, r'$\mu_{xy}$='+str(np.round(dshift_xy_mean, 6)), fontsize=15)
plt.text(lpts[1]+0.001e-10, 0.00045, r'$\sigma_{xy}$='+str(np.round(dshift_xy_std, 6)), fontsize=15)
# for k,j,z
plt.text(lpts[2]+0.001e-10, 0.0005, r'$\mu_{kjz}$='+str(np.round(dshift_kjz_mean, 6)), fontsize=15)
plt.text(lpts[2]+0.001e-10, 0.00045, r'$\sigma_{kjz}$='+str(np.round(dshift_kjz_std, 6)), fontsize=15)
#Out[59]: Text(3.998e-10, 0.00045, '$\\sigma_{kjz}$=3e-06')

# _comp_optics line 547
# ###############################################
#           dshift paramter relations
# ###############################################


def dshift_analytic_variation(
    crystal=None,
    split=None,
    rcurve=None,
    len_cryst=None,
    lamb=None,
    bragg=None,
    braggref=None,
    miscut=None,
    alpha=None,
):

    # Computations of dshift per pixel xi
    # -----------------------------------

    # distance crystal summit S to camera center C
    k0 = rcurve*np.sin(braggref)

    # distances half-crystals summit S1 & S2 to camera center C
    # half-crystal C1 at left side of the crystal normal vector n_in
    # (pointing to the center of curvature), the farthest from the camera
    c0 = ['Quartz_110', 'Quartz_102']
    if crystal in c0 and split:
        phi = len_cryst/(8.*rcurve)
        K1 = np.sqrt( ((rcurve**2)/4)*(
            6 - 4.*np.cos(2.*braggref + 2.*phi) + 2.*np.cos(2.*braggref) - 4.*np.cos(2.*phi)
        ) )
        K2 = np.sqrt( ((rcurve**2)/4)*(
            6 - 4.*np.cos(2.*braggref - 2.*phi) + 2.*np.cos(2.*braggref) - 4.*np.cos(2.*phi)
        ) )

    # angular difference from the Bragg angle of reference of the crystal
    dbragg = np.full((bragg.size), np.nan)
    for i in range(bragg.size):
        dbragg[i] = bragg[i] - braggref

    # Single crystal, w/o miscut
    if not split and not miscut:
        xi = k0*(
            np.cos(braggref) - np.sin(braggref)/np.tan(braggref - dbragg)
        )

    # Single crystal, with miscut
    elif not split and miscut:
        xim = k0*(
            np.cos(braggref) - np.sin(braggref)/np.tan(braggref - dbragg - alpha)
        )

    # Splitted crystal, w/o miscut
    elif split and not miscut:
        A1 = (
            np.sin(braggref)*np.cos(2.*braggref + 2.*phi) - np.cos(braggref)*np.sin(2.*braggref + phi)
        )/(
            np.sin(bragg)*np.cos(2.*braggref + 2.*phi) - np.cos(bragg)*np.sin(2.*braggref + phi)
        )
        A2 = (
            np.sin(braggref)*np.cos(2.*braggref - 2.*phi) - np.cos(braggref)*np.sin(2.*braggref - phi)
        )/(
            np.sin(bragg)*np.cos(2.*braggref - 2.*phi) - np.cos(bragg)*np.sin(2.*braggref - phi)
        )
        xi1 = K1*(
            -A1*(
                np.sin(bragg)*np.sin(2.*braggref + 2.*phi) + np.cos(bragg)*np.cos(2.*braggref + phi)
            ) + np.sin(braggref)*np.sin(2.*braggref + 2.*phi) + np.cos(braggref)*np.cos(2.*braggref + phi)
        )
        xi2 = K2*(
            -A2*(
                np.sin(bragg)*np.sin(2.*braggref - 2.*phi) + np.cos(bragg)*np.cos(2.*braggref - phi)
            ) + np.sin(braggref)*np.sin(2.*braggref - 2.*phi) + np.cos(braggref)*np.cos(2.*braggref - phi)
        )

    # Splitted crystal, with miscut
    elif split and miscut:
        A1m = (
            np.sin(braggref)*np.cos(2.*braggref + 2.*phi) - np.cos(braggref)*np.sin(2.*braggref + phi)
        )/(
            np.sin(bragg)*np.cos(2.*braggref + 2.*phi - alpha) - np.cos(bragg)*np.sin(2.*braggref + phi - alpha)
        )
        A2m = (
            np.sin(braggref)*np.cos(2.*braggref - 2.*phi) - np.cos(braggref)*np.sin(2.*braggref - phi)
        )/(
            np.sin(bragg)*np.cos(2.*braggref - 2.*phi + alpha) - np.cos(bragg)*np.sin(2.*braggref - phi + alpha)
        )
        xi1m = K1*(
            -A1m*(
                np.sin(bragg)*np.sin(2.*braggref + 2.*phi - alpha) + np.cos(bragg)*np.cos(2.*braggref + phi - alpha)
            ) + np.sin(braggref)*np.sin(2.*braggref + 2.*phi) + np.cos(braggref)*np.cos(2.*braggref + phi)
        )
        xi2m = K2*(
            -A2m*(
                np.sin(bragg)*np.sin(2.*braggref - 2.*phi + alpha) + np.cos(bragg)*np.cos(2.*braggref - phi + alpha)
            ) + np.sin(braggref)*np.sin(2.*braggref - 2.*phi) + np.cos(braggref)*np.cos(2.*braggref - phi)
        )

    if not split and not miscut:
        return xi
    elif not split and miscut:
        return xim
    elif split and not miscut:
        return np.vstack((xi1, xi2))
    elif split and miscut:
        return np.vstack((xi1m, xi2m))

# _core_optics.py line 1977

    def dshift_analytic_variation(
        self,
        crystal=None,
        din=None,
        len_cryst=None,
        split=None,
        rcurve=None,
        lamb=None,
        miscut=None,
        alpha=None,
    ):

        """ Compute a theoretical dshift value for a specified value of
        wavelength (or angle or xi) on the detector from the XICS geometry
        Operationnal for single or splitted crystal, witth or w/o miscut
        Possibility to compute it for a spectral range, i.e. specific detector
        length.

        Parameters:
        -----------
        crystal:    str
            Crystal definition to use, among 'Quartz_110', 'Quartz_102'
            and soon 'Ge'
        din:    str
            Crystal definition dictionary to use, among 'Quartz_110',
            'Quartz_102' and soon 'Ge'
        len_cryst:    float
            Define the length of the crystal in the meridional plane [m]
        split:    bool
            Define if the crystal is splitted or not, for now cut parallel to
            the sagittal direction
        rcurve:    float
            Define the curvature radius of the crystal [m]
        lamb:    float
            array of min size 1, in 1e-10 [m]
        miscut:    bool
            Introduce miscut between dioptre and reflecting planes
        alpha:    float
            Miscut angle value, single or array accepted, in rad.
        """

        # Check inputs
        if crystal is None:
            msg = (
                "You must choose a type of crystal from "
                + "tofu/spectro/_rockingcurve_def.py to use among:\n"
                + "\t - Quartz_110:\n"
                + "\t\t - target: ArXVII"
                + "\t\t - Miller indices (h,k,l): (1,1,0)"
                + "\t\t - Material: Quartz\n"
                + "\t - Quartz_102:\n"
                + "\t\t - target: ArXVIII"
                + "\t\t - Miller indices (h,k,l): (1,0,2)"
                + "\t\t - Material: Quartz\n"
            )
            raise Exception(msg)
        elif crystal == 'Quartz_110':
            din = _rockingcurve_def._DCRYST['Quartz_110']
        elif crystal == 'Quartz_102':
            din = _rockingcurve_def._DCRYST['Quartz_102']

        if len_cryst is None and crystal == 'Quartz_110':
            len_cryst = 0.083592
        if split is None:
            split = False
        if rcurve is None:
            msg = "Please provide a curvature radius for your crystal geometry!"
            raise Exception(msg)

        lambref = din['target']['wavelength']
        if lamb is None:
            msg = (
                "Please choose 1 or more targetted wavelength(s).\n"
                "\t Provided:\n"
                "\t\t- wavelength = ({}) A\n".format(lamb)
                + "\t\t- wavelength of reference = ({}) A\n".format(lambref),
            )
            raise Exception(msg)
        bragg = np.full((lamb.size), np.nan)
        for i in range(lamb.size):
            bragg[i] = self.get_bragg_from_lamb(
                lamb=lamb[i],
            )
        braggref = self.get_bragg_from_lamb(
            lamb=lambref,
        )

        if miscut is None:
            miscut = False
        if alpha is None:
            alpha = np.r_[(1.5/60)*np.pi/180]

        # Call
        return _comp_optics.dshift_analytic_variation(
            crystal=crystal,
            split=split,
            rcurve=rcurve,
            len_cryst=len_cryst,
            lamb=lamb,
            bragg=bragg,
            braggref=braggref,
            miscut=miscut,
            alpha=alpha,
        )





############# _spectral_constraints.py


_DLINES = {
    'ArXVIII': [
        k0 for k0, v0 in _DLINES_TOT.items()
        if k0 in [
            'ArXVIII_W1_Kallne',  # 3.7300 e-10
            #'ArXVIII_Lya1_Rice',  # 3.7311 e-10
            'ArXVIII_W2_Kallne',  # 3.7352 e-10
            #'ArXVIII_Lya2_Rice',  # 3.7365 e-10
            'ArXVII_T_Kallne',    # 3.7544 e-10
            #'ArXVII_T_Rice',      # 3.75526 e-10
            'ArXVII_Q_Kallne',    # 3.7603 e-10
            'ArXVII_B_Kallne',    # 3.7626 e-10
            'ArXVII_R_Kallne',    # 3.7639 e-10
            'ArXVII_A_Kallne',    # 3.7657 e-10
            'ArXVII_J_Kallne',    # 3.7709 e-10
            #'ArXVII_J_Rice',      # 3.77179 e-10
        ]
    ],
    'FeXXV': [
        k0 for k0, v0 in _DLINES_TOT.items()
        if k0 in [
            'FeXXV_w_Bitter',   # 1.84980 e-10
            'FeXXV_x_Bitter',   # 1.85503 e-10
            'FeXXIV_t_Bitter',  # 1.85660 e-10
            'FeXXV_y_Bitter',   # 1.85900 e-10
            'FeXXIV_q_Bitter',  # 1.86050 e-10
            'FeXXIV_k_Bitter',  # 1.86260 e-10
            'FeXXIV_r_Bitter',  # 1.86310 e-10
            'FeXXIV_j_Bitter',  # 1.86540 e-10
            'FeXXV_z_Bitter',   # 1.86760 e-10
            'FeXXIII_beta',     # 1.87003 e-10
        ]
    ],
}

_DLINES_INDPHI = {
    'ArXVIII': [
        'ArXVIII_W1_Kallne',
        'ArXVIII_W2_Kallne',
        'ArXVII_T_Kallne',
        'ArXVII_J_Kallne',
    ],
    'FeXXV': [
        'FeXXV_w_Bitter',
        'FeXXV_x_Bitter',
        'FeXXIV_q_Bitter',
        'FeXXV_z_Bitter',
    ],
}

_AMPONBCK_THRESH = 1.
_DAMPONBCK_THRESH = {
    'ArXVIII': _AMPONBCK_THRESH,
    'FeXXV': _AMPONBCK_THRESH,
}

_DCONSTRAINTS = {
    'ArXVIII': {
        'double': True,
        'symmetry': False,
        'width': {
            'all': [
                'ArXVIII_W1_Kallne',
                'ArXVIII_W2_Kallne',
                'ArXVII_T_Kallne',
                'ArXVII_Q_Kallne',
                'ArXVII_B_Kallne',
                'ArXVII_R_Kallne',
                'ArXVII_A_Kallne',
                'ArXVII_J_Kallne',
            ],
        },
        'amp': None,
        'shift': {
            'w12': [
                'ArXVIII_W1_Kallne',
                'ArXVIII_W2_Kallne',
            ],
            'tqbraj': [
                'ArXVII_T_Kallne',
                'ArXVII_Q_Kallne',
                'ArXVII_B_Kallne',
                'ArXVII_R_Kallne',
                'ArXVII_A_Kallne',
                'ArXVII_J_Kallne',
            ],
        },
    },
    'FeXXV': {
        'double': True,
        'symmetry': False,
        'width': {
            'fe25': [
                'FeXXV_w_Bitter',
                'FeXXV_x_Bitter',
                'FeXXV_y_Bitter',
                'FeXXV_z_Bitter',
            ],
            'fe24': [
                'FeXXIV_t_Bitter',
                'FeXXIV_q_Bitter',
                'FeXXIV_k_Bitter',
                'FeXXIV_r_Bitter',
                'FeXXIV_j_Bitter',
            ],
            'fe24': [
                'FeXXIII_beta',
            ],
        },
        'amp': None,
        'shift': {
            'w': [
                'FeXXV_w_Bitter',
            ],
            'xt': [
                'FeXXV_x_Bitter',
                'FeXXIV_t_Bitter',
            ],
            'yqkr': [
                'FeXXV_y_Bitter',
                'FeXXIV_q_Bitter',
                'FeXXIV_k_Bitter',
                'FeXXIV_r_Bitter',
            ],
            'jzb': [
                'FeXXIV_j_Bitter',
                'FeXXV_z_Bitter',
                'FeXXIII_beta',
            ],
        },
    },
}

_DOMAIN = {
    'ArXVIII': {
        'phi': [-0.1, 0.1],
        'lamb': [
            # all spectrum
            [3.69e-10, 3.78e-10],  # included
        ],
    },
    'FeXXV': {
        'phi': [-0.1, 0.1],
        'lamb': [
            # all spectrum
            [1.84e-10, 1.89e-10],  # included
        ],
    },
}

_FOCUS = {
    'ArXVIII': [
      [3.7300e-10, 2.e-13],    # w1
      [3.7352e-10, 2.e-13],    # w2
      [3.7544e-10, 3.e-13],    # t
      [3.7709e-10, 3.e-13],    # j
    ],
    'FeXXV': [
      [1.8498e-10, 2.e-13],    # w
      [1.85503e-10, 2.e-13],    # x
      [1.8590e-10, 2.e-13],    # y
      [1.8654e-10, 2.e-13],    # j
    ],
}

# total mass of each ion species: (40,18)Ar16+, (,74)W44+, (56,26)Ir24+
mz = np.r_[9.70171983e-26, 4.31692694e-25] # , Xe-26
# Ti = width * conv * scpct.c**2 * mz
# from Doppler broadening formulae (k_B = 1)
wAr = 5e3 / (conv * scpct.c**2 * mz[0])
wW44 = 5e3 / (conv * scpct.c**2 * mz[1])
# wIr = 5e3 (?) / (conv * scpct.c**2 * mz[2])

_DSCALES = {
    'ArXVIII': {
        'width': {
            'all': wAr,
        },
        'dratio': 1.,
        'dshift': 0.0001,
    },
    'FeXXV': {
        'width': {
            'wIr': wIr,
        },
        'dratio': 1.,
        'dshift': 0.0001,
    },
}

_DX0 = {
    'ArXVII': {
        'dratio': 0.75,
        'dshift': 4,
    },
    'ArXVIII': {
        'dratio': 0.75,
        'dshift': 4,
    },
    'FeXXV': {
        'dratio': 0.75,
        'dshift': 4,
    },
}

_DBOUNDS = {
    'ArXVII': {
        'min': {'width': 0.01, 'dratio': 0.5, 'dshift': 3,},
        'max': {'width': 1., 'dratio': 0.9, 'dshift': 5,},
    },
    'ArXVIII': {
        'min': {'width': 0.01, 'dratio': 0.5, 'dshift': 3,},
        'max': {'width': 1., 'dratio': 0.9, 'dshift': 5,},
    },
    'FeXXV': {
        'min': {'width': 0.01, 'dratio': 0.5, 'dshift': 3,},
        'max': {'width': 1., 'dratio': 0.9, 'dshift': 5,},
    },
}

# routine for detector optimal position & orientation


from .Inputs._crystal_det import _DDET_SHOTS, _DET, _DCRYST_ANG, _DCRYST_SHOTS
import tofu as tf
from tf.spectro import _plot
from tf.geom import _core_optics

def alignment_det_on_data(
    shot=None,

):

    # ---------
    # load cryst and det
    # ---------
    """
    cryst = tf.load(
        '/Home/AD265925/ToFu_All/tofu_IRFM/tofu_west/SpectroX2D/Inputs/TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.5.0-235-ga951b6d4.npz'
    )
    cryst.move(angle_deg*np.pi/180.)
    det_th = dict(np.load(
        '/Home/AD265925/Tofu_All/tofu_west/SpectroX2D/Inputs/det37_CTVD_incC4_New.npz',
        allow_pickle=True,
    ))
    """

    angle_deg=dbonus.get('PositionTable')
    cryst, det_th, ... = _get_cryst_det_from_shot(
        shot,
        angle_deg=angle_deg,
    )

    # ---------
    # compute closer det test than real optimized one
    # ---------

    dout = _core_optics.get_coord_theo_det(
        # crystal specific
        cryst=cryst,
        bragg=cryst.dbragg['braggref'],
        det_th=det_th,
        # least-square methods specific
        ## center of the camera
        find_center=find_center,
        x0_cent=x0_cent,
        jac_cent='3-point',
        method_cent='trf',
        loss_cent='soft_l1',
        bounds_cent=bounds_cent,
        ftol_cent=None,
        xtol_cent=xtol_cent,
        x_scale_cent=x_scale_cent,
        f_scale_cent=f_scale_cent,
        tr_solver_cent='exact',
        ## vectors of the camera
        find_vector=find_vector,
        x0_vect=x0_vect,
        jac_vect='3-point',
        method_vect='trf',
        loss_vect='soft_l1',
        bounds_vect=bounds_vect,
        ftol_vect=None,
        xtol_vect=xtol_vect,
        x_scale_vect=x_scale_vect,
        f_scale_vect=f_scale_vect,
        tr_solver_vect='exact',
        # plot
        plot=False,
    )

    # ---------
    # Degrees of freedom for orientation unit vectors
    # ---------

    # Best values by lsq method
    #dtheta=(-0.45338807)*(np.pi/180.),
    dtheta_interv = np.linspace(-0.55, -0.35, 200)
    ndtheta = dtheta_interv.size
    #dpsi=(-0.90476715)*(np.pi/180.),
    dpsi_interv = np.linspace(-1., -0.8, 200)
    ndpsi = dpsi_interv.size
    #tilt=(0.20107897)*(np.pi/180.),
    tilt_interv = np.linspace(0.1, 0.3, 200)
    ntilt = tilt_interv.size

    for ii in range(ndtheta):
        for jj in range(ndpsi):
            for kk in range(ntilt):
                det_test = _core_optics.CrystalBragg.get_detector_ideal(
                    self=cryst,
                    dtheta=(dtheta_interv[ii])*(np.pi/180.),
                    dpsi=(dpsi_interv[ii])*(np.pi/180.),
                    tilt=(tilt_interv[ii])*(np.pi/180.),
                )
                (
                    det_test['nout'], det_test['ei'], det_test['ej'],
                ) = (
                    det_th['nout'], det_th['ei'], det_th['ej'],
                )

                # launch data treatment for the det set
                # TBD: do not forget to write a 'if det is None:' l.1124
                # of_treat.py in order to select the previously computed one

                out = _treat.treat_data(
                    # imas parameters
                    shot=shot,
                    # detector to use
                    det=det,
                    # treatment parameters
                    dconstants=None,
                    subarea=None,
                    valid_fraction=None,
                    valid_nsigma=None,
                    amp_on_bck_thresh=None,
                    tlim=None,
                    # return options
                    return_as_datastock=False,
                )

                # Extract (better redeability)
                # --------
                dprepare = out[1]['dinput']['dprepare']
                dinput = out[1]['dinput']
                d3 = out[2]['d3']
                # phi_lim = np.array(
                #     [[-0.0006  ,  0.0006  ],
                #      [-0.025475, -0.024275],
                #      [-0.05035 , -0.04915 ]]
                # )
                phi_lim = _plot._check_phi_lim(
                    phi=out[1]['dinput']['dprepare']['phi'],
                    phi_lim=phi_lim,
                )

                # Index of spectra to plot
                # --------
                if indspect is None:
                    indspect = dinput['valid']['indt'].nonzero()[0][0]
                indspect = np.atleast_1d(indspect).ravel()
                if indspect.dtype == 'bool':
                    indspect = np.nonzero(indspect)[0]

                # Extract data
                # --------
                lamb = dprepare['lamb']
                phi = dprepare['phi']
                data = dprepare['data'][indspect, ...]

                # Set to nan if not indok
                # --------
                for ii, ispect in enumerate(indspect):
                    data[ii, ~dprepare['indok_bool'][ispect, ...]] = np.nan

                # 1d slice
                # --------
                lspect_lamb = []
                lspect_data = []
                lspect_fit = []
                for jj in range(phi_lim.shape[0]):
                    indphi = (phi >= phi_lim[jj, 0]) & (phi < phi_lim[jj, 1])
                    spect_lamb = lamb[indphi]
                    indlamb = np.argsort(spect_lamb)
                    lspect_lamb.append(spect_lamb[indlamb])
                    lspect_data.append(data[:, indphi][:, indlamb])
                    lspect_fit.append(dextract['func_tot'](
                        lamb=lspect_lamb[jj],
                        phi=phi[indphi][indlamb],
                    )[indspect, ...])

                # spectral area indices [ind]
                w_area = np.full(
                    (ndtheta, ndpsi, ntilt, phi_lim.shape[0]), np.nan
                )
                (x_area, y_area, z_area) = (
                    w_area.copy(), w_area.copy(), w_area.copy()
                )
                # maximum lspect_fit indices [ind]
                (w_indmax, x_indmax, y_indmax, z_indmax) = (
                    w_area.copy(), w_area.copy(), w_area.copy(), w_area.copy()
                )
                # maximum lspect_fit values [counts/time]
                (w_fitmax, x_fitmax, y_fitmax, z_fitmax) = (
                    w_area.copy(), w_area.copy(), w_area.copy(), w_area.copy()
                )
                # wavelength at max lspect_fit values [lambda]
                (w_lambmax, x_lambmax, y_lambmax, z_lambmax) = (
                    w_area.copy(), w_area.copy(), w_area.copy(), w_area.copy()
                )

                for jj in range(phi_lim.shape[0]):
                    # ------
                    # w line
                    # ------
                    w_area[ii, jj, kk, :] = np.where(
                        lspect_lamb[jj] < 3.951e-10
                    )
                    w_indmax[ii, jj, kk, :] = np.where(
                        lspect_fit[jj][ii, ...] == np.max(
                            lspect_fit[jj][ii, :w_area[0][-1]]
                        )
                    )
                    w_fitmax[ii, jj, kk, :] = lspect_fit[jj][ii, w_indmax]
                    w_lambmax[ii, jj, kk, :] = lspect_lamb[jj][w_indmax]
                    # ------
                    # x line
                    # ------
                    x_area[ii, jj, kk, :] = np.where(
                        (lspect_lamb[jj] < 3.964e-10)
                        & (lspect_lamb[jj] < 3.968e-10)
                    )
                    x_indmax[ii, jj, kk, :] = np.where(
                        lspect_fit[jj][ii, ...] == np.max(
                            lspect_fit[jj][ii, x_area[0][0]]
                        )
                    )
                    x_fitmax[ii, jj, kk, :] = lspect_fit[jj][ii, x_indmax]
                    x_lambmax[ii, jj, kk, :] = lspect_lamb[jj][x_indmax]
                    # ------
                    # y line
                    # ------
                    y_area[ii, jj, kk, :] = np.where(
                        (lspect_lamb[jj] < 3.968e-10)
                        & (lspect_lamb[jj] < 3.974e-10)
                    )
                    y_indmax[ii, jj, kk, :] = np.where(
                        lspect_fit[jj][ii, ...] == np.max(
                            lspect_fit[jj][ii, y_area[0][0]]
                        )
                    )
                    y_fitmax[ii, jj, kk, :] = lspect_fit[jj][ii, y_indmax]
                    y_lambmax[ii, jj, kk, :] = lspect_lamb[jj][y_indmax]
                    # ------
                    # z line
                    # ------
                    z_area[ii, jj, kk, :] = np.where(
                        lspect_lamb[jj] > 3.992e-10
                    )
                    z_indmax[ii, jj, kk, :] = np.where(
                        lspect_fit[jj][ii, ...] == np.max(
                            lspect_fit[jj][ii, z_area[0][0]:]
                        )
                    )
                    z_fitmax[ii, jj, kk, :] = lspect_fit[jj][ii, z_indmax]
                    z_lambmax[ii, jj, kk, :] = lspect_lamb[jj][z_indmax]

                # Computing the shift between the y=0 slice and the 2 others
                w_sh_01 = np.full((ndtheta, ndpsi, ntilt, 1), np.nan)
                (
                    w_sh_02,
                    x_sh_01, x_sh_02,
                    y_sh_01, y_sh_02,
                    z_sh_01, z_sh_02
                ) = (
                    w_sh_01.copy(),
                    w_sh_01.copy(), w_sh_01.copy()
                    w_sh_01.copy(), w_sh_01.copy()
                    w_sh_01.copy(), w_sh_01.copy()
                )


