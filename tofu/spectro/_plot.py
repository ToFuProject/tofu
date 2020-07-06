

# Built-in
import os
import itertools as itt

# Common
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# tofu
from tofu.version import __version__

_GITHUB = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at %s'%(__version__, _GITHUB)

_QUIVERCOLOR = plt.cm.viridis(np.linspace(0, 1, 3))
_QUIVERCOLOR = np.array([[1., 0., 0., 1.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 1.]])
_QUIVERCOLOR = ListedColormap(_QUIVERCOLOR)


# #################################################################
# #################################################################
#                   data plot
# #################################################################
# #################################################################


def CrystalBragg_plot_data_vs_lambphi(xi, xj, bragg, lamb, phi, data,
                                      lambfit=None, phifit=None,
                                      spect1d=None, vertsum1d=None,
                                      lambax=None, phiax=None,
                                      phiminmax=None, dlines=None,
                                      lambmin=None, lambmax=None,
                                      xjcut=None, lambcut=None,
                                      phicut=None, spectcut=None,
                                      cmap=None, vmin=None, vmax=None,
                                      fs=None, dmargin=None,
                                      tit=None, wintit=None,
                                      angunits='deg'):

    # Check inputs
    # ------------

    if fs is None:
        fs = (14, 8)
    if tit is None:
        tit = False
    if wintit is None:
        wintit = _WINTIT
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.99,
                   'bottom':0.07, 'top':0.92,
                   'wspace':0.2, 'hspace':0.3}
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        phifit = phifit*180./np.pi
        if phiax is not None:
            phiax = 180*phiax/np.pi
        phiminmax = phiminmax*180./np.pi
        if phicut is not None:
            phicut = phicut*180./np.pi
    nspect = spect1d.shape[0]

    if dlines is not None:
        lines = [k0 for k0, v0 in dlines.items()
                 if (v0['lambda'] >= lambfit[0]
                     and v0['lambda'] <= lambfit[-1])]
        lions = sorted(set([dlines[k0]['ION'] for k0 in lines]))
        nions = len(lions)
        dions = {k0: [k1 for k1 in lines if dlines[k1]['ION'] == k0]
                 for k0 in lions}
        dions = {k0: {'lamb': np.array([dlines[k1]['lambda']
                                        for k1 in dions[k0]]),
                      'symbol': [dlines[k1]['symbol'] for k1 in dions[k0]]}
                 for k0 in lions}
        lcol = ['k', 'r', 'b', 'g', 'm', 'c']
        ncol = len(lcol)
        llamb = np.concatenate([dions[lions[ii]]['lamb']
                                for ii in range(nions)])
        indsort = np.argsort(llamb)
        lsymb = list(itt.chain.from_iterable([dions[lions[ii]]['symbol']
                                              for ii in range(nions)]))
        lsymb = [lsymb[ii] for ii in indsort]
        lcollamb = list(itt.chain.from_iterable(
            [[lcol[ii]]*dions[lions[ii]]['lamb'].size
             for ii in range(nions)]))
        lcollamb = [lcollamb[ii] for ii in indsort]

    lcolspect = ['k', 'r', 'b', 'g', 'y', 'm', 'c']
    ncolspect = len(lcolspect)

    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())
    if lambmin is None:
        lambmin = lambfit.min()
    if lambmax is None:
        lambmax = lambfit.max()

    # Plot
    # ------------

    fig = fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(4, 4, **dmargin)
    ax0 = fig.add_subplot(gs[:3, 0], aspect='equal')    #, adjustable='datalim'
    ax1 = fig.add_subplot(gs[:3, 1], aspect='equal',
                          sharex=ax0, sharey=ax0)       #, adjustable='datalim'
    axs1 = fig.add_subplot(gs[3, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:3, 2])
    axs2 = fig.add_subplot(gs[3, 2], sharex=ax2, sharey=axs1)
    ax3 = fig.add_subplot(gs[:3, 3], sharey=ax2)

    ax0.set_title('Coordinates transform')
    ax1.set_title('Camera image')
    axs1.set_title('Vertical average')
    ax2.set_title('Camera image transformed')

    ax0.set_ylabel(r'$x_j$ (m)')
    ax0.set_xlabel(r'$x_i$ (m)')
    axs1.set_ylabel(r'data')
    axs1.set_xlabel(r'$x_i$ (m)')
    ax2.set_ylabel(r'incidence angle ($deg$)')
    axs2.set_xlabel(r'$\lambda$ ($m$)')

    ax0.contour(xi, xj, bragg, 10, cmap=cmap)
    ax0.contour(xi, xj, phi, 10, cmap=cmap, ls='--')
    ax1.imshow(data, extent=extent, aspect='equal',
               origin='lower', vmin=vmin, vmax=vmax)
    axs1.plot(xi, np.nanmean(data, axis=0), c='k', ls='-')
    ax2.scatter(lamb.ravel(), phi.ravel(), c=data.ravel(), s=2,
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)

    if xjcut is None:
        ax3.plot(vertsum1d, phifit, c='k', ls='-')
    else:
        dphicut = 0.1*(phifit.max() - phifit.min())
        for ii in range(xjcut.size):
            ax1.axhline(xjcut[ii],
                        ls='-', lw=1., c='r')
            ax2.plot(lambcut[ii, :], phicut[ii, :],
                     ls='-', lw=1., c='r')
            ax3.plot(lambcut[ii, :],
                     (np.nanmean(phicut[ii, :])
                      + spectcut[ii, :]*dphicut/np.nanmax(spectcut[ii, :])),
                     ls='-', lw=1., c='r')

    if phiax is not None:
        ax2.plot(lambax, phiax, c='r', ls='-', lw=1.)

    # Plot known spectral lines
    if dlines is not None:
        for ii, k0 in enumerate(lions):
            for jj in range(dions[k0]['lamb'].size):
                x = dions[k0]['lamb'][jj]
                col = lcol[ii%ncol]
                axs2.axvline(x,
                             c=col, ls='--')
                axs2.annotate(dions[k0]['symbol'][jj],
                              xy=(x, 1.01), xytext=None,
                              xycoords=('data', 'axes fraction'),
                              color=col, arrowprops=None,
                              horizontalalignment='center',
                              verticalalignment='bottom')
                if xjcut is not None:
                    ax3.axvline(x,
                                c=col, ls='--')
        hand = [mlines.Line2D([], [], color=lcol[ii%ncol], ls='--')
                for ii in range(nions)]
        axs2.legend(hand, lions,
                    bbox_to_anchor=(1., 1.02), loc='upper left')

    # Plot 1d spectra and associated phi windows
    for ii in range(nspect):
        axs2.plot(lambfit, spect1d[ii], c=lcolspect[ii%ncolspect], ls='-')
        ax2.axhline(phiminmax[ii ,0], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax2.axhline(phiminmax[ii, 1], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax3.axhline(phiminmax[ii ,0], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax3.axhline(phiminmax[ii, 1], c=lcolspect[ii%ncolspect], ls='-', lw=1.)

    ax2.set_xlim(lambmin, lambmax)
    ax2.set_ylim(phifit.min(), phifit.max())
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')
    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    return [ax0, ax1]


# #################################################################
# #################################################################
#                   fit1d plot
# #################################################################
# #################################################################


def plot_fit1d(dfit1d=None, dout=None, showonly=None,
               indspect=None, fs=None, dmargin=None,
               tit=None, wintit=None):

    # Check inputs
    # ------------
    if fs is None:
        fs = (15, 8)
    if wintit is None:
        wintit = _WINTIT
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.85,
                   'bottom':0.07, 'top':0.85,
                   'wspace':0.2, 'hspace':0.3}

    # Index of spectra to plot
    if indspect is None:
        indspect = np.r_[0]
    indspect = np.atleast_1d(indspect).ravel()
    if indspect.dtype == 'bool':
        indspect = np.nonzero(indspect)[0]

    # Extract (better redeability)
    dprepare = dfit1d['dinput']['dprepare']
    dinput = dfit1d['dinput']

    # pre-compute
    # ------------
    if dinput['same_spectrum'] is True:
        nlines = int(dinput['nlines'] / dinput['same_spectrum_nspect'])
        dinput['lines'] = dinput['lines'][:nlines]
        dinput['ion'] = dinput['ion'][:nlines]
        dinput['symb'] = dinput['symb'][:nlines]
        nwidth = int(dinput['width']['keys'].size
                     / dinput['same_spectrum_nspect'])
        dinput['width']['keys'] = dinput['width']['keys'][:nwidth]
        dinput['width']['ind'] = dinput['width']['ind'][:, :nlines]
        dinput['width']['ind'] = dinput['width']['ind'][:nwidth, :]
        dinput['shift']['ind'] = dinput['shift']['ind'][:, :nlines]

    nlines = dinput['lines'].size
    ions_u = sorted(set(dinput['ion'].tolist()))
    nions = len(ions_u)

    indwidth = np.argmax(dinput['width']['ind'], axis=0)
    indshift = np.argmax(dinput['shift']['ind'], axis=0)

    x = dinput['lines'][None, :] + dout['shift']['values']

    lcol = ['k', 'r', 'b', 'g', 'm', 'c']
    ncol = len(lcol)
    if dout['Ti'] is not False:
        lfcol = ['y', 'g', 'c', 'm']
    else:
        lfcol = [None]
    nfcol = len(lfcol)
    if dout['vi'] is not False:
        lhatch = [None, '/', '\\', '|', '-', '+', 'x', '//']
    else:
        lhatch = [None]
    nhatch = len(lhatch)
    nspect = indspect.size

    # Plot
    # ------------

    for ii in range(nspect):
        ispect = indspect[ii]
        if tit is None:
            titi = ("spect {}\n".format(ispect+1)
                    + "({}/{})".format(ispect, dprepare['data'].shape[0]))
        else:
            titi = tit

        fig = fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(1, 1, **dmargin)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_ylabel(r'data (a.u.)')
        ax.set_xlabel(r'$\lambda$ (m)')

        ax.plot(dprepare['lamb'][dprepare['indok'][ispect, :]],
                dprepare['data'][ispect, dprepare['indok'][ispect, :]],
                marker='.', c='k', ls='None', ms=8)
        if showonly is not True:
            if dout['sol_detail'] is not False:
                ax.plot(dprepare['lamb'], dout['sol_detail'][ispect, :, 0],
                        ls='-', c='k')
            ax.set_prop_cycle(None)
            if dout['sol_detail'] is not False:
                if dout['Ti'] is not False or dout['vi'] is not False:
                    for jj in range(nlines):
                        col = lfcol[indwidth[jj]%nfcol]
                        hatch = lhatch[indshift[jj]%nhatch]
                        ax.fill_between(dprepare['lamb'],
                                        dout['sol_detail'][ispect, :, 1+jj],
                                        alpha=0.3, color=col, hatch=hatch)
                else:
                    ax.plot(dprepare['lamb'],
                            dout['sol_detail'][ispect, :, 1:].T)
            if dout['sol_tot'] is not False:
                ax.plot(dprepare['lamb'], dout['sol_tot'][ispect, :],
                        c='k', lw=2.)

        # Annotate lines
        for jj, k0 in enumerate(ions_u):
            col = lcol[jj%ncol]
            ind = (dinput['ion'] == k0).nonzero()[0]
            for nn in ind:
                ax.axvline(x[ispect, nn],
                           c=col, ls='--')
                lab = dinput['symb'][nn]
                if dout['coefs'] is not False:
                    val = dout['coefs']['values'][ispect, nn]
                    lab += '\n{:4.2e}'.format(val)
                if dout['shift'] is not False:
                    val = dout['shift']['values'][ispect, nn]*1.e10
                    lab += '\n({:+4.2e} A)'.format(val)
                ax.annotate(lab,
                            xy=(x[ispect, nn], 1.01), xytext=None,
                            xycoords=('data', 'axes fraction'),
                            color=col, arrowprops=None,
                            horizontalalignment='center',
                            verticalalignment='bottom')

        # Ion legend
        hand = [mlines.Line2D([], [], color=lcol[jj%ncol], ls='--',
                              label=ions_u[jj])
                for jj in range(nions)]
        legi = ax.legend(handles=hand,
                         title='ions',
                         bbox_to_anchor=(1.01, 1.), loc='upper left')
        ax.add_artist(legi)

        # Ti legend
        if dout['Ti'] is not False:
            hand = [mpatches.Patch(color=lfcol[jj%nfcol])
                    for jj in range(dinput['width']['ind'].shape[0])]
            lleg = [dinput['width']['keys'][jj]
                    + '  {:4.2f}'.format(dout['Ti']['values'][ispect, jj]*1.e-3)
                    for jj in range(dinput['width']['ind'].shape[0])]
            legT = ax.legend(handles=hand, labels=lleg,
                             title='Ti (keV)',
                             bbox_to_anchor=(1.01, 0.8), loc='upper left')
            ax.add_artist(legT)

        # vi legend
        if dout['vi'] is not False:
            hand = [mpatches.Patch(facecolor='w', edgecolor='k',
                                   hatch=lhatch[jj%nhatch])
                    for jj in range(dinput['shift']['ind'].shape[0])]
            lleg = [dinput['shift']['keys'][jj]
                    + '  {:4.2f}'.format(dout['vi']['values'][ispect, jj]*1.e-3)
                    for jj in range(dinput['shift']['ind'].shape[0])]
            legv = ax.legend(handles=hand, labels=lleg,
                             title='vi (km/s)',
                             bbox_to_anchor=(1.01, 0.5), loc='upper left')
            ax.add_artist(legv)

        # Ratios legend
        if dout['ratio'] is not False:
            nratio = dout['ratio']['values'].shape[1]
            hand = [mlines.Line2D([], [], c='k', ls='None')]*nratio
            lleg = ['{} =  {:4.2e}'.format(
                dout['ratio']['lab'][jj],
                dout['ratio']['values'][ispect, jj])
                    for jj in range(nratio)]
            legr = ax.legend(handles=hand,
                             labels=lleg,
                             title='line ratio',
                             bbox_to_anchor=(1.01, 0.11), loc='lower left')
            ax.add_artist(legr)

        # double legend
        if dinput['double'] is not False:
            hand = [mlines.Line2D([], [], c='k', ls='None')]*2
            lleg = ['ratio = {:4.2f}'.format(dout['dratio'][ispect]),
                    ('shift ' + r'$\approx$'
                     + ' {:4.2e}'.format(dout['dshift'][ispect]))]
            legr = ax.legend(handles=hand,
                             labels=lleg,
                             title='double',
                             bbox_to_anchor=(1.01, 0.), loc='lower left')

        ax.set_xlim(dinput['dprepare']['domain']['lamb']['minmax'])

        if titi is not False:
            fig.suptitle(titi, size=14, weight='bold')
        if wintit is not False:
            fig.canvas.set_window_title(wintit)
    return ax


def CrystalBragg_plot_data_fit2d(xi, xj, data, lamb, phi, indok=None,
                                 dfit2d=None,
                                 dax=None, indspect=None,
                                 spect1d=None, fit1d=None,
                                 lambfit=None, phiminmax=None,
                                 cmap=None, vmin=None, vmax=None,
                                 plotmode=None, fs=None, dmargin=None,
                                 angunits='deg', tit=None, wintit=None):

    # Check inputs
    # ------------

    if dax is None:
        if fs is None:
            fs = (18, 9)
        if cmap is None:
            cmap = plt.cm.viridis
        if dmargin is None:
            dmargin = {'left':0.05, 'right':0.99,
                       'bottom':0.06, 'top':0.95,
                       'wspace':None, 'hspace':0.8}
        if wintit is None:
            wintit = _WINTIT
        if tit is None:
            tit = '2D fitting of X-Ray Crystal Bragg spectrometer'
    if dfit2d['dinput']['symmetry'] is True:
        symaxis = dfit2d['dinput']['symmetry_axis'][indspect]
    if angunits is None:
        angunits = 'deg'
    assert angunits in ['deg', 'rad']




    phiflat = dfit2d['phi']
    pts_phi = dfit2d['pts_phi']
    ylim = np.r_[dfit2d['dinput']['phiminmax']]
    if angunits == 'deg':
        phiflat = phiflat*180./np.pi
        phi = phi*180./np.pi
        pts_phi = pts_phi*180./np.pi
        ylim = ylim*180./np.pi
        phiminmax = phiminmax*180./np.pi
        if dfit2d['dinput']['symmetry'] is True:
            symaxis = symaxis*180./np.pi
    if plotmode == 'transform':
        xlab = r'$\lambda$ (m)'
        ylab = r'$\phi$ ({})'.format(angunits)
    else:
        xlab = r'$x_i$' + r' (m)'
        ylab = r'$x_j$' + r' ({})'.format(angunits)
        spect1d = None

    # pre-compute
    # ------------

    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = max(np.nanmax(dfit2d['data'][indspect, :]),
                   np.nanmax(dfit2d['sol_tot'][indspect, :]))
    err = dfit2d['sol_tot'][indspect, :] - dfit2d['data'][indspect, :]
    errm = vmax/10.


    # Prepare figure if dax not provided
    # ------------
    if dax is None:
        fig = plt.figure(figsize=fs)
        naxh = (3*3
                + 2*(dfit2d['dinput']['Ti'] + dfit2d['dinput']['vi']
                     + (dfit2d['ratio'] not in [None, False])))
        naxv = 1
        gs = gridspec.GridSpec((3+naxv)*3, naxh, **dmargin)
        if plotmode == 'transform':
            ax0 = fig.add_subplot(gs[:9, :3])
            ax0.set_xlim(dfit2d['dinput']['lambminmax'])
            ax0.set_ylim(ylim)
        else:
            ax0 = fig.add_subplot(gs[:9, :3], aspect='equal')
        ax0c = fig.add_subplot(gs[10, :3])
        ax1 = fig.add_subplot(gs[:9, 3:6],
                          sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[:9, 6:9],
                          sharex=ax0, sharey=ax0)
        ax1c = fig.add_subplot(gs[10, 3:6])

        ax0.set_title('Residu')
        ax0.set_xlabel(xlab)
        ax0.set_ylabel(ylab)
        ax1.set_title('Original data')
        ax1.set_xlabel(xlab)
        ax2.set_title('2d fit')
        dax = {'err': ax0, 'err_colorbar': ax0c,
               'data': ax1, 'data_colorbar': ax1c,
               'fit': ax2}
        if tit is not False:
            fig.suptitle(tit)
        if wintit is not False:
            fig.canvas.set_window_title(wintit)

        if naxv == 1:
            dax['fit1d'] = fig.add_subplot(gs[9:11, 6:9], sharex=ax2)
            dax['fit1d'].set_title('1d sliced spectrum vs fit')
            dax['fit1d'].set_ylabel(r'data')
            dax['err1d'] = fig.add_subplot(gs[11, 6:9], sharex=ax2)
            dax['err1d'].set_ylabel(r'err')
            dax['err1d'].set_xlabel(xlab)

        nn = 1
        if dfit2d['dinput']['Ti'] is True:
            dax['Ti'] = fig.add_subplot(gs[:9, 9+2*(nn-1):9+2*nn], sharey=ax1)
            dax['Ti'].set_title(r'Width')
            dax['Ti'].set_xlabel(r'$\hat{T_i}$' + r' (keV)')
            nn += 1
        if dfit2d['dinput']['vi'] is True:
            dax['vi'] = fig.add_subplot(gs[:9, 9+2*(nn-1):9+2*nn], sharey=ax1)
            dax['vi'].set_title(r'Shift')
            dax['vi'].set_xlabel(r'$\hat{v_i}$' + r' (km/s)')
            dax['vi'].axvline(0, ls='-', lw=1., c='k')
            nn += 1
        if dfit2d['ratio'] not in [None, False]:
            dax['ratio'] = fig.add_subplot(gs[:9, 9+2*(nn-1):], sharey=ax1)
            dax['ratio'].set_title(r'Intensity Ratio')
            dax['ratio'].set_xlabel(r'ratio (a.u)')

    # Plot main images
    # ------------
    if plotmode == 'transform':
        if dax.get('err') is not None:
            errax = dax['err'].scatter(dfit2d['lamb'],
                                       phiflat,
                                       c=err,
                                       s=6, marker='s', edgecolors='None',
                                       vmin=-errm, vmax=errm,
                                       cmap=plt.cm.seismic)
        if dax.get('data') is not None:
            dataax = dax['data'].scatter(dfit2d['lamb'],
                                         phiflat,
                                         c=dfit2d['data'][indspect, :],
                                         s=6, marker='s', edgecolors='None',
                                         vmin=vmin, vmax=vmax, cmap=cmap)
        if dax.get('fit') is not None:
            dax['fit'].scatter(dfit2d['lamb'],
                               phiflat,
                               c=dfit2d['sol_tot'][indspect, :],
                               s=6, marker='s', edgecolors='None',
                               vmin=vmin, vmax=vmax, cmap=cmap)
            if dfit2d['dinput']['symmetry'] is True:
                dax['fit'].axhline(symaxis,
                                   c='k', ls='--', lw=2.)
        if dax.get('fit1d') is not None:
            # dax['fit1d'].plot()
            pass
    else:
        extent = (xi.min(), xi.max(), xj.min(), xj.max())
        if dax.get('err') is not None:
            dax['err'].imshow(err,
                              extent=extent, cmap=plt.cm.seismic,
                              origin='lower', vmin=errm, vmax=errm)
        if dax.get('data') is not None:
            dax['data'].imshow(data,
                               extent=extent, cmap=cmap,
                               origin='lower', vmin=vmin, vmax=vmax)
        if dax.get('fit') is not None:
            dax['fit'].imshow(fit,
                              extent=extent, cmap=cmap,
                              origin='lower', vmin=vmin, vmax=vmax)

    # Plot profiles
    # ------------
    if dax.get('Ti') is not None and dfit2d['dinput']['Ti'] is True:
        for ii in range(dfit2d['kTiev'].shape[1]):
            dax['Ti'].plot(dfit2d['kTiev'][indspect, ii, :]*1.e-3,
                           pts_phi,
                           ls='-', marker='.', ms=4,
                           label=dfit2d['dinput']['width']['keys'][ii])
        dax['Ti'].set_xlim(left=0.)
        dax['Ti'].legend(frameon=True,
                         loc='upper left', bbox_to_anchor=(0., -0.1))

    if dax.get('vi') is not None and dfit2d['dinput']['vi'] is True:
        for ii in range(dfit2d['vims'].shape[1]):
            dax['vi'].plot(dfit2d['vims'][indspect, ii, :]*1.e-3,
                           pts_phi,
                           ls='-', marker='.', ms=4,
                           label=dfit2d['dinput']['shift']['keys'][ii])
        dax['vi'].legend(frameon=True,
                         loc='upper left', bbox_to_anchor=(0., -0.1))

    if dax.get('ratio') not in [None, False]:
        for ii in range(dfit2d['ratio']['value'].shape[1]):
            dax['ratio'].plot(dfit2d['ratio']['value'][indspect, ii, :],
                              pts_phi,
                              ls='-', marker='.', ms=4,
                              label=dfit2d['ratio']['str'][ii])
        dax['ratio'].legend(frameon=True,
                            loc='upper left', bbox_to_anchor=(0., -0.1))

    # Add 1d prof
    # ------------
    if spect1d is not None:
        lcol = ['r', 'k', 'm']
        for ii in range(spect1d.shape[0]):
            l, = dax['fit1d'].plot(lambfit, fit1d[ii, :],
                                   ls='-', lw=1., c=lcol[ii%len(lcol)])
            col = l.get_color()
            dax['fit1d'].plot(lambfit, spect1d[ii, :],
                              ls='None', marker='.', c=col, ms=4)
            dax['err1d'].plot(lambfit, fit1d[ii, :] - spect1d[ii, :],
                              ls='-', lw=1., c=col)
            dax['fit'].axhline(phiminmax[ii][0], c=col, ls='-', lw=2.)
            dax['fit'].axhline(phiminmax[ii][1], c=col, ls='-', lw=2.)

    # double legend
    if dfit2d['dinput']['double'] is not False:
        hand = [mlines.Line2D([], [], c='k', ls='None')]*2
        c0 = (dfit2d['dinput']['double'] is True
              or dfit2d['dinput']['double'].get('dratio') is None)
        if c0:
            dratio = dfit2d['dratio'][indspect]
            dratiostr = ''
        else:
            dratio = dfit2d['dinput']['double']['dratio']
            dratiostr = '  (fixed)'
        c0 = (dfit2d['dinput']['double'] is True
              or dfit2d['dinput']['double'].get('dshift') is None)
        if c0:
            dshift = dfit2d['dshift'][indspect]
            dshiftstr = ''
        else:
            dshift = dfit2d['dinput']['double']['dshift']
            dshiftstr = '  (fixed)'
        lleg = ['dratio = {:4.2f}{}'.format(dratio, dratiostr),
                ('dshift = {:4.2e} * '.format(dshift)
                 + r'$\lambda${}'.format(dshiftstr))]
        legr = dax['err1d'].legend(handles=hand, labels=lleg, title='double',
                                   bbox_to_anchor=(1.01, 0.),
                                   loc='center left')

    # Polishing
    # ------------
    for kax in dax.keys():
        if kax != 'err':
            plt.setp(dax[kax].get_yticklabels(), visible=False)
    if dax.get('fit') is not None:
        plt.setp(dax['fit'].get_xticklabels(), visible=False)
    if dax.get('fit1d') is not None:
        plt.setp(dax['fit1d'].get_xticklabels(), visible=False)
    if dax.get('err_colorbar') is not None and dax['err'] is not None:
        plt.colorbar(errax, ax=dax['err'], cax=dax['err_colorbar'],
                     orientation='horizontal', extend='both')
    if (dax.get('data_colorbar') is not None
        and (dax['data'] is not None or dax['fit'] is not None)):
        plt.colorbar(dataax, ax=dax['data'], cax=dax['data_colorbar'],
                     orientation='horizontal')
    return dax


# #################################################################
# #################################################################
#                   noise plot
# #################################################################
# #################################################################

def plot_noise_analysis(dnoise=None,
                        margin=None, ms=None,
                        dax=None, fs=None, cmap=None, dmargin=None,
                        wintit=None, tit=None,
                        save=None, name=None, path=None, fmt=None):

    # Check inputs
    # ------------
    if save is None:
        save = False

    if margin is None:
        margin = 3.

    if ms is None:
        ms = 2.

    if dax is None:
        if fs is None:
            fs = (18, 9)
        if cmap is None:
            cmap = plt.cm.viridis
        if dmargin is None:
            dmargin = {'left':0.05, 'right':0.99,
                       'bottom':0.06, 'top':0.95,
                       'wspace':None, 'hspace':None}
        if wintit is None:
            wintit = _WINTIT
        if tit is None:
            tit = '2D fitting of X-Ray Crystal Bragg spectrometer'

    # Prepare data
    # ------------
    nspect = dnoise['datasort'].shape[0]
    nlamb = dnoise['chi2'].shape[1]


    chi2plot = dnoise['chi2'].ravel()
    lnbsplinesplot = np.tile(dnoise['lnbsplines'], (nspect, nlamb, 1)).ravel()

    indnan = dnoise['indnan'][:-1]
    phiplot = np.tile(np.insert(dnoise['phisort'], indnan, np.nan),
                      (nspect, 1)).ravel()
    dataplot = np.insert(dnoise['datasort'], indnan, np.nan, axis=1).ravel()
    fitplot = np.insert(dnoise['fitsort'], indnan, np.nan, axis=1).ravel()
    errplot = fitplot - dataplot
    var = dnoise['var']
    xdata = dnoise['var_xdata']

    # fit sqrt on sigma
    const = dnoise['var_const']
    indout = np.abs(errplot) > margin*const*np.sqrt(fitplot)
    indok = ~indout

    # Prepare figure if dax not provided
    # ------------
    if dax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(2, 4, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[0, 2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[0, 3])

        ax0.set_title('Profiles')
        ax0.set_ylabel(r'$\phi$ (rad)')
        ax0.set_xlabel(r'data (a.u.)')
        ax1.set_title('Err profiles')
        ax1.set_ylabel(r'$\phi$ (rad)')
        ax1.set_xlabel(r'data (a.u.)')
        ax2.set_title('nbspline scan')
        ax2.set_xlabel('nbsplines')
        ax2.set_ylabel(r'$\chi^2$')
        ax3.set_title('Variance')
        ax3.set_xlabel('data (a.u.)')
        ax3.set_ylabel('err (a.u.)')
        ax5.set_title('Variance')
        ax5.set_xlabel('data (a.u.)')
        ax5.set_ylabel(r'$\sigma$' + ' (a.u.)')
        ax6.set_title('Camera')
        ax6.set_xlabel(r'$x_i$ (m)')
        ax6.set_ylabel(r'$x_j$ (m)')

        dax = {'prof': ax0,
               'proferr': ax1,
               'scan': ax2,
               'err': ax3,
               'errbin': ax5,
               'cam': ax6,
              }

    # Plot main images
    # ------------

    dax['prof'].plot(dataplot[indok], phiplot[indok],
                     marker='.', ls='None', c='k', ms=ms)
    dax['prof'].plot(fitplot, phiplot,
                     marker='None', ls='-', c='b')
    dax['prof'].plot(dataplot[indout], phiplot[indout],
                     marker='.', ls='None', c='r', ms=ms)

    dax['proferr'].plot(errplot[indok], phiplot[indok],
                        marker='.', ls='None', c='k', ms=ms)
    dax['proferr'].plot(errplot[indout], phiplot[indout],
                        marker='.', ls='None', c='r', ms=ms)

    # dax['cam'].imshow()

    dax['scan'].plot(lnbsplinesplot, chi2plot,
                     marker='.', ls='None', c='k', ms=ms)
    # dax['scan'].plot(lnbsplinesplot[indout], chi2plot[indout],
    #                  marker='.', ls='None', c='r')

    dax['err'].fill_between(xdata,
                            -margin*const*np.sqrt(xdata),
                            margin*const*np.sqrt(xdata),
                            color=(0.7, 0.7, 0.7, 1.))
    dax['err'].plot(fitplot[indok], errplot[indok],
                    marker='.', ls='None', c='k', ms=ms)
    dax['err'].plot(fitplot[indout], errplot[indout],
                    marker='.', ls='None', c='r', ms=ms)
    dax['err'].fill_between(xdata,
                            -const*np.sqrt(xdata),
                            const*np.sqrt(xdata),
                            color=(0.4, 0.4, 0.4, 0.5))

    dax['errbin'].plot(xdata, np.sqrt(var),
                       marker='.', c='k', ls='None')
    lab = r'$\sigma=$' + r'{:5.3e}'.format(const) + r'$\sqrt{data}$'
    dax['errbin'].plot(xdata, const*np.sqrt(xdata),
                       marker='None', ls='--', c='k', label=lab)
    dax['errbin'].legend(loc='lower right', frameon=True)

    # save
    # ------------
    if save is True:
        if name is None:
            name = 'NoiseAnalysis'
        if fmt is None:
            fmt = 'png'
        if path is None:
            path = './'
        if name[-4:] != '.{}'.format(fmt):
            name = '{}.{}'.format(name, fmt)

        pfe = os.path.join(os.path.abspath(path), name)
        dax['prof'].figure.savefig(pfe, format=fmt)

        msg = "Saved in:\n\t{}".format(pfe)
        print(msg)

    return dax
