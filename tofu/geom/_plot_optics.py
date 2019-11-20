

import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# tofu
from tofu.version import __version__
from . import _def as _def

_GITHUB = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at %s'%(__version__, _GITHUB)

_QUIVERCOLOR = plt.cm.viridis(np.linspace(0, 1, 3))
_QUIVERCOLOR = np.array([[1., 0., 0., 1.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 1.]])
_QUIVERCOLOR = ListedColormap(_QUIVERCOLOR)


# Generic
def _check_projdax_mpl(dax=None, proj=None, fs=None, wintit=None):

    # ----------------------
    # Check proj
    if proj is None:
        proj = 'all'
    assert isinstance(proj, str)
    proj = proj.lower()
    lproj = ['cross', 'hor', '3d']
    assert proj in lproj + ['all']
    if proj == 'all':
        proj = ['cross', 'hor']
    else:
        proj = [proj]

    # ----------------------
    # Check dax
    lc = [dax is None, issubclass(dax.__class__, Axes),
          isinstance(dax, dict), isinstance(dax, list)]
    assert any(lc)
    if lc[0]:
        dax = dict.fromkeys(proj)
    elif lc[1]:
        assert len(proj) == 1
        dax = {proj[0]: dax}
    elif lc[2]:
        lcax = [(kk in proj
                 and (ax is None or issubclass(ax.__class__, Axes)))
                for kk, ax in dax.items()]
        if not all(lcax):
            msg = "Wrong key or axes in dax:\n"
            msg += "    - proj = %s"%str(proj)
            msg += "    - dax = %s"%str(dax)
            raise Exception(msg)
    else:
        assert len(dax) == 2
        assert all([ax is None or issubclass(ax.__class__, Axes)
                    for ax in dax])
        dax = {'cross': dax[0], 'hor': dax[1]}

    # Populate with default axes if necessary
    if 'cross' in proj and 'hor' in proj:
        if 'cross' in proj and 'hor' in proj:
            if dax['cross'] is None:
                assert dax['hor'] is None
                lax = _def.Plot_LOSProj_DefAxes('all', fs=fs,                                                                 wintit=wintit)
                dax['cross'], dax['hor'] = lax
        elif 'cross' in proj and dax['cross'] is None:
            dax['cross'] = _def.Plot_LOSProj_DefAxes('cross', fs=fs,
                                                     wintit=wintit)
        elif 'hor' in proj and dax['hor'] is None:
            dax['hor'] = _def.Plot_LOSProj_DefAxes('hor', fs=fs,
                                                     wintit=wintit)
        elif '3d' in proj  and dax['3d'] is None:
            dax['3d'] = _def.Plot_3D_plt_Tor_DefAxes(fs=fs,
                                                    wintit=wintit)
    for kk in lproj:
        dax[kk] = dax.get(kk, None)
    return dax



# #################################################################
# #################################################################
#                   Generic geometry plot
# #################################################################
# #################################################################

def CrystalBragg_plot(cryst, lax=None, proj=None, res=None, element=None,
                      color=None, dP=None,
                      dI=None, dBs=None, dBv=None,
                      dVect=None, dIHor=None, dBsHor=None, dBvHor=None,
                      dleg=None, indices=False,
                      draw=True, fs=None, wintit=None, tit=None, Test=True):

    # ---------------------
    # Check / format inputs

    if Test:
        msg = "Arg proj must be in ['cross','hor','all','3d'] !"
        assert type(draw) is bool, "Arg draw must be a bool !"
        assert cryst.__class__.__name__ == 'CrystalBragg'
    if wintit is None:
        wintit = _WINTIT
    if dleg is None:
         dleg = _def.TorLegd

    # ---------------------
    # call plotting functions

    kwa = dict(fs=fs, wintit=wintit, Test=Test)
    if proj == '3d':
        # Temporary matplotlib issue
        dleg = None
    else:
        dax = _CrystalBragg_plot_crosshor(cryst, proj=proj, res=res, dax=lax, element=element,
                                          color=color)

    # recompute the ax.dataLim
    ax0 = None
    for kk, vv in dax.items():
        if vv is None:
            continue
        dax[kk].relim()
        dax[kk].autoscale_view()
        if dleg is not None:
            dax[kk].legend(**dleg)
        ax0 = vv

    # set title
    if tit != False:
        ax0.figure.suptitle(tit)
    if draw:
        ax0.figure.canvas.draw()
    return dax

def _CrystalBragg_plot_crosshor(cryst, proj=None, dax=None, element=None, res=None,
                                Pdict=_def.TorPd, Idict=_def.TorId, Bsdict=_def.TorBsd,
                                Bvdict=_def.TorBvd, Vdict=_def.TorVind,
                                color=None, ms=None, quiver_cmap=None,
                                LegDict=_def.TorLegd, indices=False,
                                draw=True, fs=None, wintit=None, Test=True):
    if Test:
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'

    # ---------------------
    # Check / format inputs

    if element is None:
        element = 'oscvr'
    element = element.lower()
    if 'v' in element and quiver_cmap is None:
        quiver_cmap = _QUIVERCOLOR
    if color is None:
        if cryst._dmisc.get('color') is not None:
            color = cryst._dmisc['color']
        else:
            color = 'k'
    if ms is None:
        ms = 6

    # ---------------------
    # Prepare axe and data

    dax = _check_projdax_mpl(dax=dax, proj=proj, fs=fs, wintit=wintit)

    if 's' in element or 'v' in element:
        summ = cryst._dgeom['summit']
    if 'c' in element:
        cent = cryst._dgeom['center']
    if 'r' in element:
        ang = np.linspace(0, 2.*np.pi, 200)
        rr = 0.5*cryst._dgeom['rcurve']
        row = cryst._dgeom['center'] + rr*cryst._dgeom['nout']
        row = (row[:, None]
               + rr*(np.cos(ang)[None, :]*cryst._dgeom['nout'][:, None]
                     + np.sin(ang)[None, :]*cryst._dgeom['e1'][:, None]))

    # ---------------------
    # plot

    if 'o' in element:
        cont = cryst.sample_outline_plot(res=res)
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(cont[0,:], cont[1,:]), cont[2,:],
                              ls='-', c=color, marker='None',
                              label=cryst.Id.NameLTX+' contour')
        if dax['hor'] is not None:
            dax['hor'].plot(cont[0,:], cont[1,:],
                            ls='-', c=color, marker='None',
                            label=cryst.Id.NameLTX+' contour')
    if 's' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(summ[0], summ[1]), summ[2],
                              marker='^', ms=ms, c=color,
                              label=cryst.Id.NameLTX+" summit")
        if dax['hor'] is not None:
            dax['hor'].plot(summ[0], summ[1],
                            marker='^', ms=ms, c=color,
                            label=cryst.Id.NameLTX+" summit")
    if 'c' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(cent[0], cent[1]), cent[2],
                              marker='o', ms=ms, c=color,
                              label=cryst.Id.NameLTX+" center")
        if dax['hor'] is not None:
            dax['hor'].plot(cent[0], cent[1],
                            marker='o', ms=ms, c=color,
                            label=cryst.Id.NameLTX+" center")
    if 'r' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(row[0,:], row[1,:]), row[2,:],
                              ls='--', color=color, marker='None',
                              label=cryst.Id.NameLTX+' rowland')
        if dax['hor'] is not None:
            dax['hor'].plot(row[0,:], row[1,:],
                            ls='--', color=color, marker='None',
                            label=cryst.Id.NameLTX+' rowland')
    if 'v' in element:
        nin = cryst._dgeom['nin']
        e1, e2 = cryst._dgeom['e1'], cryst._dgeom['e2']
        p0 = np.repeat(summ[:,None], 3, axis=1)
        v = np.concatenate((nin[:, None], e1[:, None], e2[:, None]), axis=1)
        if dax['cross'] is not None:
            dax['cross'].quiver(np.hypot(p0[0,:], p0[1,:]), p0[2,:],
                                np.hypot(v[0,:], v[1,:]), v[2,:],
                                np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                                angles='xy', scale_units='xy',
                                label=cryst.Id.NameLTX+" unit vect", **Vdict)
        if dax['hor'] is not None:
            dax['hor'].quiver(p0[0,:], p0[1,:],
                              v[0,:], v[1,:],
                              np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                              angles='xy', scale_units='xy',
                              label=cryst.Id.NameLTX+" unit vect", **Vdict)

    return dax


# #################################################################
# #################################################################
#                   Bragg diffraction plot
# #################################################################
# #################################################################

# Deprecated ? re-use ?
def CrystalBragg_plot_approx_detector_params(Rrow, bragg, d, Z,
                                             frame_cent, nn):

    R = 2.*Rrow
    L = 2.*R
    ang = np.linspace(0., 2.*np.pi, 100)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8], aspect='equal')

    ax.axvline(0, ls='--', c='k')
    ax.plot(Rrow*np.cos(ang), Rrow + Rrow*np.sin(ang), c='r')
    ax.plot(R*np.cos(ang), R + R*np.sin(ang), c='b')
    ax.plot(L*np.cos(bragg)*np.r_[-1,0,1],
            L*np.sin(bragg)*np.r_[1,0,1], c='k')
    ax.plot([0, d*np.cos(bragg)], [Rrow, d*np.sin(bragg)], c='r')
    ax.plot([0, d*np.cos(bragg)], [Z, d*np.sin(bragg)], 'g')
    ax.plot([0, L/10*nn[1]], [Z, Z+L/10*nn[2]], c='g')
    ax.plot(frame_cent[1]*np.cos(2*bragg-np.pi),
            Z + frame_cent[1]*np.sin(2*bragg-np.pi), c='k', marker='o', ms=10)

    ax.set_xlabel(r'y')
    ax.set_ylabel(r'z')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), frameon=False)
    return ax


def CrystalBragg_plot_xixj_from_braggangle(bragg=None, xi=None, xj=None,
                                           data=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8], aspect='equal')

    for ii in range(len(bragg)):
        deg ='{0:07.3f}'.format(bragg[ii]*180/np.pi)
        ax.plot(xi[:,ii], xj[:,ii], '.', label='bragg %s'%deg)

    ax.set_xlabel(r'xi')
    ax.set_ylabel(r'yi')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), frameon=False)
    return ax




def CrystalBragg_plot_braggangle_from_xixj(xi=None, xj=None,
                                           bragg=None, angle=None,
                                           ax=None, plot=None,
                                           braggunits='rad', angunits='rad',
                                           **kwdargs):

    if isinstance(plot, bool):
        plot = 'contour'

    if ax is None:
        fig = plt.figure()
        ax0 = fig.add_axes([0.1, 0.1, 0.35, 0.8], aspect='equal')
        ax1 = fig.add_axes([0.55, 0.1, 0.35, 0.8], aspect='equal')
        ax = [ax0, ax1]
    if plot == 'contour':
        if 'levels' in kwdargs.keys():
            lvls = kwdargs['levels']
            del kwdargs['levels']
            obj0 = ax[0].contour(xi, xj, bragg, lvls, **kwdargs)
            obj1 = ax[1].contour(xi, xj, angle, lvls, **kwdargs)
        else:
            obj0 = ax[0].contour(xi, xj, bragg, **kwdargs)
            obj1 = ax[1].contour(xi, xj, angle, **kwdargs)
    elif plot == 'imshow':
        extent = (xi.min(), xi.max(), xj.min(), xj.max())
        obj0 = ax[0].imshow(bragg, extent=extent, aspect='equal',
                            adjustable='datalim', **kwdargs)
        obj1 = ax[1].imshow(angle, extent=extent, aspect='equal',
                            adjustable='datalim', **kwdargs)
    elif plot == 'pcolor':
        obj0 = ax[0].pcolor(xi, xj, bragg, **kwdargs)
        obj1 = ax[1].pcolor(xi, xj, angle, **kwdargs)
    ax[0].set_xlabel(r'xi')
    ax[1].set_xlabel(r'xi')
    ax[0].set_ylabel(r'yi')
    ax[1].set_ylabel(r'yi')
    cax0 = plt.colorbar(obj0, ax=ax[0])
    cax1 = plt.colorbar(obj1, ax=ax[1])
    cax0.ax.set_ylabel(r'$\theta_{bragg}$ (%s)'%braggunits)
    cax1.ax.set_ylabel(r'$ang$ (%s)'%angunits)
    return ax


def CrystalBragg_plot_data_vs_lambphi(xi, xj, bragg, lamb, phi, data,
                                      lambfit=None, phifit=None, spect1d=None,
                                      cmap=None, vmin=None, vmax=None,
                                      fs=None, dmargin=None,
                                      angunits='deg'):

    # Check inputs
    # ------------

    if fs is None:
        fs = (14,8)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.03, 'right':0.99,
                   'bottom':0.05, 'top':0.92,
                   'wspace':None, 'hspace':0.4}
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        phifit = phifit*180./np.pi


    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())
    extent2 = (lambfit.min(), lambfit.max(), phifit.min(), phifit.max())

    # Plot
    # ------------

    fig = fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(4, 3, **dmargin)
    ax0 = fig.add_subplot(gs[:3, 0], aspect='equal', adjustable='datalim')
    ax1 = fig.add_subplot(gs[:3, 1], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)
    axs1 = fig.add_subplot(gs[3, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:3, 2])
    axs2 = fig.add_subplot(gs[3, 2], sharex=ax2, sharey=axs1)

    ax0.set_title('Coordinates transform')
    ax1.set_title('Camera image')
    ax2.set_title('Camera image transformed')

    ax0.set_ylabel(r'incidence angle ($deg$)')

    ax0.contour(xi, xj, bragg, 10, cmap=cmap)
    ax0.contour(xi, xj, phi, 10, cmap=cmap, ls='--')
    ax1.imshow(data, extent=extent, aspect='equal',
               origin='lower', vmin=vmin, vmax=vmax)
    axs1.plot(xi, np.sum(data, axis=0), c='k', ls='-')
    ax2.scatter(lamb.ravel(), phi.ravel(), c=data.ravel(), s=2,
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)
    axs2.plot(lambfit, spect1d, c='k', ls='-')

    ax2.set_xlim(extent2[0], extent2[1])
    ax2.set_ylim(extent2[2], extent2[3])

    return [ax0, ax1]


def CrystalBragg_plot_data_vs_fit(xi, xj, bragg, lamb, phi, data, mask=None,
                                  lambfit=None, phifit=None, spect1d=None,
                                  dfit1d=None, dfit2d=None, lambfitbins=None,
                                  cmap=None, vmin=None, vmax=None,
                                  fs=None, dmargin=None,
                                  angunits='deg', dmoments=None):

    # Check inputs
    # ------------

    if fs is None:
        fs = (16, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.03, 'right':0.99,
                   'bottom':0.05, 'top':0.92,
                   'wspace':None, 'hspace':0.4}
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        phifit = phifit*180./np.pi


    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())
    extent2 = (lambfit.min(), lambfit.max(), phifit.min(), phifit.max())

    ind = np.digitize(lamb[mask].ravel(), lambfitbins)
    spect2dmean = np.zeros((lambfitbins.size+1,))
    for ii in range(lambfitbins.size+1):
        indi = ind==ii
        if np.any(indi):
            spect2dmean[ii] = np.nanmean(dfit2d['fit'][indi])

    # Plot
    # ------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(4, 6, **dmargin)
    ax0 = fig.add_subplot(gs[:3, 0], aspect='equal', adjustable='datalim')
    ax1 = fig.add_subplot(gs[:3, 1], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)
    axs1 = fig.add_subplot(gs[3, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:3, 2])
    axs2 = fig.add_subplot(gs[3, 2], sharex=ax2, sharey=axs1)
    ax3 = fig.add_subplot(gs[:3, 3], sharex=ax2, sharey=ax2)
    axs3 = fig.add_subplot(gs[3, 3], sharex=ax2)#, sharey=axs1)
    ax4 = fig.add_subplot(gs[:3, 4], sharex=ax2, sharey=ax2)
    axs4 = fig.add_subplot(gs[3, 3], sharex=ax2)#, sharey=axs1)
    ax5 = fig.add_subplot(gs[:3, 5], sharey=ax2)

    ax0.set_title('Coordinates transform')
    ax1.set_title('Camera image')
    ax2.set_title('Camera image transformed')
    ax3.set_title('2d spectral fit')
    ax4.set_title('2d error')
    ax5.set_title('Moments')

    ax4.set_xlabel('%s'%angunits)
    ax0.set_ylabel(r'incidence angle ($deg$)')

    ax0.contour(xi, xj, bragg, 10, cmap=cmap)
    ax0.contour(xi, xj, phi, 10, cmap=cmap, ls='--')
    ax1.imshow(data, extent=extent, aspect='equal',
               origin='lower', vmin=vmin, vmax=vmax)
    axs1.plot(xi, np.nanmean(data, axis=0), c='k', ls='-')
    ax2.scatter(lamb.ravel(), phi.ravel(), c=data.ravel(), s=1,
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)
    axs2.plot(lambfit, spect1d, c='k', ls='None', marker='.', ms=4)
    axs2.plot(lambfit, dfit1d['fit'].ravel(), c='r', ls='-', label='fit')
    for ll in dfit1d['lamb0']:
        axs2.axvline(ll, c='k', ls='--')

    # dfit2d
    ax3.scatter(lamb[mask].ravel(), phi[mask].ravel(), c=dfit2d['fit'], s=1,
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)
    axs3.plot(lambfit, spect1d, c='k', ls='None', marker='.')
    axs3.plot(lambfit, spect2dmean, c='b', ls='-')
    err = dfit2d['fit'] - data[mask].ravel()
    errmax = np.max(np.abs(err))
    ax4.scatter(lamb[mask].ravel(), phi[mask].ravel(), c=err, s=1,
                marker='s', edgecolors='None',
                cmap=plt.cm.seismic, vmin=-errmax, vmax=errmax)

    # Moments
    if dmoments is not None:
        if dmoments.get('ratio') is not None:
            ind = dmoments['ratio'].get('ind')
            if ind is None:
                ind = [np.argmin(np.abs(dfit2d['lamb0']-ll))
                        for ll in dmoments['ratio']['lamb']]
            for indi in ind:
                axs3.axvline(dfit2d['lamb0'][indi], c='k', ls='--')
            amp0 = BSpline(dfit2d['knots'], dfit2d['camp'][ind[0],:], dfit2d['deg'])(phifit)
            amp1 = BSpline(dfit2d['knots'], dfit2d['camp'][ind[1],:], dfit2d['deg'])(phifit)
            lab = dmoments['ratio']['name'] + '{} / {}'
            ratio = (amp0 / amp1) / np.nanmax(amp0 / amp1)
            ax5.plot(amp0 / amp1, phifit, ls='-', c='k', label=lab)
        if dmoments.get('sigma') is not None:
            ind = dmoments['sigma'].get('ind')
            if ind is None:
                ind = np.argmin(np.abs(dfit2d['lamb0']-dmoments['sigma']['lamb']))
            axs3.axvline(dfit2d['lamb0'][ind], c='b', ls='--')
            sigma = BSpline(dfit2d['knots'], dfit2d['csigma'][ind,:], dfit2d['deg'])(phifit)
            lab = r'$\sigma({} A)$'.format(np.round(dfit2d['lamb0'][ind]*1.e10),
                                        4)
            ax5.plot(sigma/np.nanmax(sigma), phifit, ls='-', c='b', label=lab)

    ax2.set_xlim(extent2[0], extent2[1])
    ax2.set_ylim(extent2[2], extent2[3])
    return [ax0, ax1]
