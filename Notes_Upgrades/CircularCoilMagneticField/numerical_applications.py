
import sys
import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = os.path.abspath(os.path.dirname(__file__))
_TOFU_MAG = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                         'tofu', 'mag')

# import local
sys.path.insert(1, _TOFU_MAG)
import _comp_circularcoil as _comp
_ = sys.path.pop(1)


###################################################
###################################################
#       Default values
###################################################


_RAD = 1.
_CENT_Z = 0.

_PTS_R = np.linspace(0., 2*_RAD, 200)
_PTS_Z = np.linspace(-_RAD, _RAD, 200)

_NSCAN = np.r_[1, 2, 5, 10, 20, 50, 100, 200, 500]


###################################################
###################################################
#       Ploting routines
###################################################


def plot_B_RZ_conv(rad=None, cent_Z=None,
                   ptsRZ=None, constraint=None, nscan=None,
                   fs=None, dmargin=None, wintit=None, tit=None, dleg=None):

    if rad is None:
        rad = _RAD
    if cent_Z is None:
        cent_Z = _CENT_Z

    if ptsRZ is None:
        ptsRZ = np.array([
            np.repeat(_PTS_R[None, :], _PTS_Z.size, axis=0),
            np.repeat(_PTS_Z[:, None], _PTS_R.size, axis=1)
        ])

    if nscan is None:
        nscan = _NSCAN
    nscan = np.unique(np.atleast_1d(nscan).ravel())
    ntot = nscan.size

    Btot = np.full(tuple(np.r_[ntot, ptsRZ.shape]), np.nan)
    for ii in range(ntot):
        Btot[ii, ...] = _comp.get_B_2d_RZ(rad=rad, cent_Z=cent_Z, ptsRZ=ptsRZ,
                                          nn=nscan[ii], constraint=constraint,
                                          returnas='sum')

    Bnorm = np.sqrt(np.sum(Btot**2, axis=1))
    errabs = np.sqrt(np.sum((Btot - Btot[ntot-1:ntot, ...])**2, axis=1))
    errrel = errabs / Bnorm[ntot-1:ntot, ...]
    errnorm = 100. * errrel
    vmax = 0.1

    dist = np.hypot(ptsRZ[0, ...] - rad, ptsRZ[1, ...] - cent_Z).ravel()
    distmax = np.nanmax(dist)

    dR = np.mean(np.diff(np.unique(ptsRZ[0, ...])))
    dZ = np.mean(np.diff(np.unique(ptsRZ[1, ...])))
    extent = (np.min(ptsRZ[0, ...])-dR/2., np.max(ptsRZ[0, ...])+dR/2.,
              np.min(ptsRZ[1, ...])-dZ/2., np.max(ptsRZ[1, ...])+dZ/2.)

    # ----------------
    #   plot

    # default args
    axCol = "w"
    if fs is None:
        fs = (14, 6)
    elif type(fs) is str:
        if fs.lower() == 'a4':
            fs = (8.27, 11.69)
        elif fs.lower() == 'a4.t':
            fs = (11.69, 8.27)
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.99,
                   'bottom':0.10, 'top':0.88,
                   'wspace':0.4, 'hspace':0.5}
    if dleg is None:
        dleg = {'loc': 'upper left',
                'bbox_to_anchor': (1, 1)}
    if tit is None:
        tit = False

    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit != False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, fontweight='bold')

    # Prepare
    gs = gridspec.GridSpec(3, ntot, **dmargin)

    dax = {'B': [None for ii in range(ntot)],
           'Bnorm': [None for ii in range(ntot)],
           'err': [None for ii in range(ntot)]}
    for ii in range(ntot):
        if ii == 0:
            dax['B'][ii] = fig.add_subplot(gs[0, ii])
        else:
            dax['B'][ii] = fig.add_subplot(gs[0, ii],
                                           sharex=dax['B'][0],
                                           sharey=dax['B'][0],
                                           aspect='equal')
        # dax['Bnorm'][ii] = fig.add_subplot(gs[1, ii],
                                           # sharex=dax['B'][0],
                                           # sharey=dax['B'][0])
        dax['err'][ii] = fig.add_subplot(gs[1, ii],
                                         sharex=dax['B'][0],
                                         sharey=dax['B'][0])

        dax['B'][ii].set_title('n = {}'.format(nscan[ii]))
        if ii == 0:
            dax['B'][ii].set_ylabel('Z (m)')
            # dax['Bnorm'][ii].set_ylabel('Z (m)')
            dax['err'][ii].set_ylabel('Z (m)')
        dax['err'][ii].set_xlabel('R (m)')

    dax['B'][0].set_aspect('equal')

    dax['conv'] = fig.add_subplot(gs[-1, :int(ntot/2)-1], xscale='log', yscale='log')
    dax['conv'].set_xlim(0, nscan[-2])
    dax['conv'].set_xlabel('n')
    dax['conv'].set_ylabel('err (%)')

    dax['convr'] = fig.add_subplot(gs[-1, int(ntot/2)-1:-2], xscale='linear', yscale='log')
    dax['convr'].set_xlim(right=distmax / rad)
    dax['convr'].set_xlabel('dist. to coil / radius')
    dax['convr'].set_ylabel('err (%)')

    dax['circ'] = fig.add_subplot(gs[-1, -1], xscale='linear', yscale='linear')
    dax['circ'].set_xlabel('radius / dist. to coil')
    dax['circ'].set_ylabel('B (T)')

    # plot
    for ii in range(ntot):
        dax['B'][ii].quiver(ptsRZ[0, ...].ravel(), ptsRZ[1, ...].ravel(),
                            Btot[ii, 0, ...].ravel(), Btot[ii, 1, ...].ravel(),
                            angles='uv', pivot='mid')
        # dax['Bnorm'][ii].scatter(ptsRZ[0, :], ptsRZ[1, :],
                                 # c=Bnorm[ii, :], s=8,
                                 # marker='s', edgecolors='None',
                                 # cmap=plt.cm.viridis)
        dax['err'][ii].imshow(errnorm[ii, ...],
                              extent=extent,
                              origin='lower',
                              interpolation='nearest',
                              aspect='equal',
                              cmap=plt.cm.seismic,
                              vmin=-vmax, vmax=vmax)
        dax['err'][ii].plot([rad], [cent_Z],
                            c='g', ls='None', marker='o', ms=2)

        if ii != ntot-1:
            dax['convr'].plot(dist/rad, errnorm[ii, ...].ravel(),
                              ls='None', marker='.',
                              label='n = {}'.format(nscan[ii]))
        # Add marker for coils

    dax['conv'].plot(nscan, np.sum(errnorm, axis=-1).sum(axis=-1),
                     c='k', ls='-', lw=1.)

    dax['convr'].axhline(vmax, ls='--', lw=1., c='k')
    dax['convr'].set_title('vmax = {} %'.format(vmax))

    dax['circ'].plot(rad/dist, Bnorm[-1, ...].ravel(),
                     ls='None', c='k', marker='.')

    if dleg is not False:
        dax['convr'].legend(**dleg)

    return dax
