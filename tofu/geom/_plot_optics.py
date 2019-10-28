

import numpy as np
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
def _check_Lax(lax=None, n=2):
    assert n in [1,2]
    c0 = lax is None
    c1 = issubclass(lax.__class__,Axes)
    c2 = hasattr(lax, '__iter__')
    if c2:
        c2 = all([aa is None or issubclass(aa.__class__,Axes) for aa in lax])
        c2 = c2 and len(lax) in [1,2]
    if n==1:
        assert c0 or c1, "Arg ax must be None or a plt.Axes instance !"
    else:
        assert c0 or c1 or c2, "Arg lax must be an Axes or a list/tuple of such !"
        if c0:
            lax = [None,None]
        elif c1:
            lax = [lax,None]
        elif c2 and len(lax)==1:
            lax = [lax[0],None]
        else:
            lax = list(lax)
    return lax, c0, c1, c2


# #################################################################
# #################################################################
#                   Generic geometry plot
# #################################################################
# #################################################################

def CrystalBragg_plot(cryst, lax=None, proj=None, res=None, element=None, dP=None,
                      dI=None, dBs=None, dBv=None,
                      dVect=None, dIHor=None, dBsHor=None, dBvHor=None,
                      dleg=None, indices=False,
                      draw=True, fs=None, wintit=None, tit=None, Test=True):

    # ---------------------
    # Check / format inputs

    if proj is None:
        proj = 'cross'
    proj = proj.lower()
    if Test:
        msg = "Arg proj must be in ['cross','hor','all','3d'] !"
        assert proj in ['cross','hor','all','3d'], msg
        lax, c0, c1, c2 = _check_Lax(lax, n=2)
        assert type(draw) is bool, "Arg draw must be a bool !"
        assert cryst.__class__.__name__ == 'CrystalBragg'
    if wintit is None:
        wintit = _WINTIT

    # ---------------------
    # Check / format inputs

    kwa = dict(fs=fs, wintit=wintit, Test=Test)
    if proj=='3d':
        # Temporary matplotlib issue
        dleg = None

    elif proj == 'cross':
        lax = _CrystalBragg_plot_cross(cryst, res=res, ax=lax, element=element)

    elif proj == 'hor':
        pass

    elif proj == 'all':
        pass

    # recompute the ax.dataLim
    lax[0].relim()
    lax[0].autoscale_view()
    if proj == 'all':
        lax[1].relim()
        lax[1].autoscale_view()

    # set title
    if tit != False:
        lax[0].figure.suptitle(tit)

    # set leg
    if dleg != False:
        lax[0].legend(**dleg)
    if draw:
        lax[0].figure.canvas.draw()
    lax = lax if proj=='all' else lax[0]
    return lax

def _CrystalBragg_plot_cross(cryst, ax=None, element=None, res=None,
                             Pdict=_def.TorPd, Idict=_def.TorId, Bsdict=_def.TorBsd,
                             Bvdict=_def.TorBvd, Vdict=_def.TorVind,
                             quiver_cmap=None,
                             LegDict=_def.TorLegd, indices=False,
                             draw=True, fs=None, wintit=None, Test=True):
    if Test:
        ax, C0, C1, C2 = _check_Lax(ax, n=1)
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'

    # ---------------------
    # Check / format inputs

    if element is None:
        element = 'csv'
    element = element.lower()
    if 'v' in element and quiver_cmap is None:
        quiver_cmap = _QUIVERCOLOR


    # ---------------------
    # Prepare axe and data

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', fs=fs,
                                       wintit=wintit, Type=V.Id.Type)

    if 's' in element or 'v' in element:
        summ = cryst._dgeom['summit']

    # ---------------------
    # plot

    if 'c' in element:
        cont = cryst.sample_outline_plot(res=res)
        ax.plot(np.hypot(cont[0,:], cont[1,:]), cont[2,:],
                ls='-', c='k', marker='None', label='')
    if 's' in element:
        ax.plot(np.hypot(summ[0], summ[1]), summ[2],
                marker='^', ms=10)
    if 'v' in element:
        nin = cryst._dgeom['nIn']
        e1, e2 = cryst._dgeom['e1'], cryst._dgeom['e2']
        p0 = np.repeat(summ[:,None], 3, axis=1)
        v = np.concatenate((nin[:, None], e1[:, None], e2[:, None]), axis=1)
        ax.quiver(p0[0,:], p0[1,:], v[0,:], v[1,:],
                  np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                  angles='xy', scale_units='xy',
                  label=V.Id.NameLTX+" unit vect", **Vdict)

    # if indices:
        # for ii in range(0,V.dgeom['nP']):
            # ax.annotate(r"{0}".format(ii), size=10,
                        # xy = (midX[ii],midY[ii]),
                        # xytext = (midX[ii]-0.01*VInX[ii],
                                  # midY[ii]-0.01*VInY[ii]),
                        # horizontalalignment='center',
                        # verticalalignment='center')
    return ax


# #################################################################
# #################################################################
#                   Bragg diffraction plot
# #################################################################
# #################################################################


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


def CrystalBragg_plot_data_vs_braggangle(xi, xj, bragg, lamb, angle, data,
                                         lambrest=None, camp=None,
                                         cshift=None, cwidth=None,
                                         deg=None, knots=None,
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
        angle = angle*180./np.pi
        bragg = bragg*180./np.pi


    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())

    # rescaled spectrum
    Dx = lamb.max()-lamb.min()
    brlb = lamb.min() + Dx*np.linspace(0, 1, xi.size)
    brlbbins = 0.5*(brlb[1:] + brlb[:-1])
    ind = np.digitize(lamb, brlbbins)
    spectrum = np.array([np.sum(data[ind==ii]) for ii in np.unique(ind)])

    # Fitted spectrum
    if camp is not None:
        assert lambrest is not None
        assert cwidth is not None
        if cshift is None:
            cshift = 0.*camp

        import tofu.geom._comp_optics as _comp_optics
        xfit = brlb
        Dy = angle.max()-angle.min()
        yfit = angle.min() + Dy*np.linspace(0, 1, xj.size)
        extent2 = (xfit.min(), xfit.max(), yfit.min(), yfit.max())
        if deg is None or knots is None:
            deg = 2
            ymin, ymax = yfit.min(), yfit.max()
            yl, yu = ymin + (ymax-ymin)/3., ymin + 2.*(ymax-ymin)/3.
            knots = np.r_[ymin, ymin, ymin, yl, yu, ymax, ymax, ymax]
        func = _comp_optics.get_2dspectralfit_func(lambrest,
                                                   deg=deg, knots=knots)
        fitted = func(xfit, yfit,
                      camp=camp, cwidth=cwidth, cshift=cshift)
        error = (func(lamb, angle,
                     camp=camp, cwidth=cwidth, cshift=cshift, mesh=False)
                 - data)
        verrmax = max(np.abs(np.nanmin(error)), np.abs(np.nanmax(error)))
    else:
        fitted = None
        xfit = None
        yfit = None

    # Plot
    # ------------

    fig = fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(4, 5, **dmargin)
    ax0 = fig.add_subplot(gs[:3, 0], aspect='equal', adjustable='datalim')
    ax1 = fig.add_subplot(gs[:3, 1], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)
    axs1 = fig.add_subplot(gs[3, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:3, 2])
    axs2 = fig.add_subplot(gs[3, 2], sharex=ax2, sharey=axs1)
    ax3 = fig.add_subplot(gs[:3, 3], sharex=ax2, sharey=ax2)
    axs3 = fig.add_subplot(gs[3, 3], sharex=ax2)#, sharey=axs1)
    ax4 = fig.add_subplot(gs[:3, 4], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)

    ax0.set_title('Coordinates transform')
    ax1.set_title('Camera image')
    ax2.set_title('Camera image transformed')
    ax3.set_title('2d spectral fit')
    ax4.set_title('2d error')

    ax4.set_xlabel('%s'%angunits)
    ax0.set_ylabel(r'incidence angle ($deg$)')

    ax0.contour(xi, xj, bragg.T, 10, cmap=cmap)
    ax1.imshow(data.T, extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
    axs1.plot(xi, np.sum(data, axis=1), c='k', ls='-')
    ax2.scatter(lamb.ravel(), angle.ravel(), c=data.ravel(),
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)
    axs2.plot(brlb, spectrum, c='k', ls='-')
    if fitted is not None:
        ax3.imshow(fitted.T, extent=extent2, aspect='auto')
        axs3.plot(brlb, fitted.sum(axis=1), c='k', ls='-')
        ax4.imshow(error.T, extent=extent,
                   aspect='equal', cmap=plt.cm.seismic,
                   vmin=-verrmax, vmax=verrmax)

    if lambrest is not None:
        for ax in [axs2, axs3]:
            for ii in range(len(lambrest)):
                ax.axvline(lambrest[ii], c='k', ls='--')

    ax2.set_xlim(extent2[0], extent2[1])
    ax2.set_ylim(extent2[2], extent2[3])

    return [ax0, ax1]
