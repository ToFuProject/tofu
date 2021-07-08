
import os
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as mplcolors

_HERE = os.path.dirname(__file__)
_TOFUPATH = os.path.abspath(os.path.join(_HERE, os.pardir))

sys.path.insert(1, _TOFUPATH)
import tofu as tf
from inputs_temp.dlines import dlines
_ = sys.path.pop(1)


_OUTPUTFILE = 'CrystalBragg_Benchmark_Fit2d_OK.npz'

_LBS = [5, 7, 9, 11]
_LSUB = np.r_[10000, 20000, 50000, 100000]
_LTOL = np.logspace(-3, -6, 4)

_CRYSTPATH = os.path.abspath(os.path.join(
    _HERE, '..',
    'inputs_temp',
    'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.4.1-174-g453d6a3.npz'
))
_DATAPATH = os.path.abspath(os.path.join(
    _HERE, '..',
    'inputs_temp',
    'SpectroX2D_WEST_Ar_55076_t84.358s.npz'
))
_MASKPATH = os.path.abspath(os.path.join(
    _HERE, '..',
    'inputs_temp',
    'XICS_mask.npz'
))


_DLINES = {
    k0: v0 for k0, v0 in dlines.items()
    if (
        (
            v0['source'] == 'Vainshtein 85'
            and v0['ION'] == 'ArXVII'
            and v0['symbol'] not in ['y2', 'z2']
        )
        or (
            v0['source'] == 'Goryaev 17'
            and v0['ION'] == 'ArXVI'
            and v0['symbol'] not in [
                'l', 'n3-h1', 'n3-h2', 'd',
                'n3-e1', 'n3-f4', 'n3-f2', 'n3-e2',
                'n3-f1', 'n3-g1', 'n3-g2', 'n3-g3',
                'n3-f3', 'n3-a1', 'n3-a2', 'n3-c1',
                'n3-c2', 'g', 'i', 'e', 'f', 'u',
                'v', 'h', 'c', 'b', 'n3-b1',
                'n3-b2', 'n3-b4', 'n3-d1', 'n3-d2',
            ]
        )
    )
}


_DLINESFE = {k0: v0 for k0, v0 in dlines.items()
             if v0['element'] == 'Fe'
}

_DCONST0 = {
    'double': True,
    'amp': {'ArXVI_j_Goryaev': {'key': 'kj', 'coef': 1.3576},
            'ArXVI_k_Goryaev': {'key': 'kj'}},
    'width': {'kj': ['ArXVI_j_Goryaev', 'ArXVI_k_Goryaev'],
              'qra': ['ArXVI_q_Goryaev', 'ArXVI_r_Goryaev', 'ArXVI_a_Goryaev'],
              'wxyz': ['ArXVII_w_Vainshtein', 'ArXVII_x_Vainshtein',
                       'ArXVII_y_Vainshtein', 'ArXVII_z_Vainshtein'],
              'nmst': ['ArXVI_n_Goryaev', 'ArXVI_m_Goryaev',
                       'ArXVI_s_Goryaev', 'ArXVI_t_Goryaev']},
    'shift': {'kj': ['ArXVI_j_Goryaev', 'ArXVI_k_Goryaev'],
              'qra': ['ArXVI_q_Goryaev', 'ArXVI_r_Goryaev', 'ArXVI_a_Goryaev'],
              'wxyz': ['ArXVII_w_Vainshtein', 'ArXVII_x_Vainshtein',
                       'ArXVII_y_Vainshtein', 'ArXVII_z_Vainshtein'],
              'nmst': ['ArXVI_n_Goryaev', 'ArXVI_m_Goryaev',
                       'ArXVI_s_Goryaev', 'ArXVI_t_Goryaev']}
}
_RATIO_XY = 0.70
_RATIO_ZXY = 1.35
_DCONST1 = copy.deepcopy(_DCONST0)
_DCONST1['amp'].update({
    'ArXVII_y_Vainshtein': {'key': 'xy'},
    'ArXVII_x_Vainshtein': {'key': 'xy', 'coef': _RATIO_XY}})
_DCONST2 = copy.deepcopy(_DCONST1)
_DCONST2['amp'].update({
    'ArXVII_z_Vainshtein': {'key': 'xy', 'coef': _RATIO_ZXY*(1.+_RATIO_XY)}})
_DCONST3 = copy.deepcopy(_DCONST0)
_DCONST3['width'] = {
    'kj': ['ArXVI_j_Goryaev', 'ArXVI_k_Goryaev'],
    'wxyz': ['ArXVII_w_Vainshtein', 'ArXVII_x_Vainshtein',
             'ArXVII_y_Vainshtein', 'ArXVII_z_Vainshtein'],
    'nmstqra': ['ArXVI_n_Goryaev', 'ArXVI_m_Goryaev',
                'ArXVI_s_Goryaev', 'ArXVI_t_Goryaev',
                'ArXVI_q_Goryaev', 'ArXVI_r_Goryaev', 'ArXVI_a_Goryaev']}
_LDCONST = [_DCONST0, _DCONST1, _DCONST2, _DCONST3]

_DX0 = {'amp': {'ArXVI_n_Goryaev': 0., 'ArXVI_m_Goryaev': 0.,
                'ArXVI_s_Goryaev': 0., 'ArXVI_t_Goryaev': 0.}}
_RATIO = {'up': ['ArXVII_w_Vainshtein'],
          'low': ['ArXVI_k_Goryaev']}
_KTIEV = 'wxyzikj'
_LAMBMIN = 3.94e-10
_LAMBMAX = 4.e-10
_DEG = 2
_NPTS = 40


def benchmark(output_file=_OUTPUTFILE,
              lbsplines=_LBS,
              lsub=_LSUB,
              ltol=_LTOL,
              crystpath=_CRYSTPATH,
              datapath=_DATAPATH,
              dlines=_DLINES,
              dconstraints=None,
              dx0=_DX0,
              maskpath=_MASKPATH,
              deg=_DEG,
              lambmin=_LAMBMIN,
              lambmax=_LAMBMAX,
              npts=_NPTS,
              ratio=_RATIO,
              kTiev_key=_KTIEV,
              verbose=2):

    # ----------
    # Prepare input
    if dconstraints is None:
        dconstraints = _DCONST0
    if isinstance(dconstraints, int):
        dconstraints = _LDCONST[dconstraints]
    assert isinstance(dconstraints, dict)

    cryst = tf.load(crystpath)
    det_cent, det_nout, det_ei, det_ej = cryst.get_detector_approx(
        ddist=0.04, di=-0.004, dj=0.,
        dtheta=0., dpsi=0., tilt=0.011,
        tangent_to_rowland=True)
    data = np.load(datapath)['data']
    xi = (np.arange(0, data.shape[1])-(data.shape[1]-1)/2.)*172e-6
    xj = (np.arange(0, data.shape[0])-(data.shape[0]-1)/2.)*172e-6
    xi_bounds, xj_bounds = (xi.min(), xi.max()), (xj.min(), xj.max())

    if maskpath is not None:
        mask = ~np.any(np.load(maskpath)['ind'], axis=0)
    else:
        mask = None

    ntol, nsub, nbs = len(ltol), len(lsub)+1, len(lbsplines)

    # ----------
    # Prepare output
    success = np.full((ntol, nsub, nbs), np.nan)
    time = np.full((ntol, nsub, nbs), np.nan)
    cost = np.full((ntol, nsub, nbs), np.nan)
    nfev = np.full((ntol, nsub, nbs), np.nan)
    ratiobis = np.full((ntol, nsub, nbs, npts), np.nan)
    kTiev = np.full((ntol, nsub, nbs, npts), np.nan)

    # ----------
    # Compute
    for jj in range(nsub):
        if jj == nsub - 1:
            subset = None
            nsub = np.r_[nsub, 450457]
        else:
            subset = tf.utils._get_subset_indices(lsub[jj], 450457)
        for ii in range(ntol):
            for ll in range(nbs):
                dfit2d = cryst.plot_data_fit2d_dlines(
                    xi=xi, xj=xj, data=data, mask=mask,
                    det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
                    lambmin=lambmin, lambmax=lambmax,
                    dlines=dlines,
                    dconstraints=dconstraints,
                    dx0=dx0, deg=deg, jac='call',
                    verbose=verbose, subset=subset,
                    ftol=ltol[ii], xtol=ltol[ii], gtol=ltol[ii],
                    nbsplines=lbsplines[ll],
                    ratio=ratio,
                    npts=npts, spect1d=None, plot=False,
                    returnas='dict')
                success[ii, jj, ll] = dfit2d['success'][0]
                time[ii, jj, ll] = dfit2d['time'][0]
                cost[ii, jj, ll] = dfit2d['cost'][0]
                nfev[ii, jj, ll] = dfit2d['nfev'][0]
                ind = dfit2d['kTiev_keys'].index(kTiev_key)
                kTiev[ii, jj, ll, :] = dfit2d['kTiev'][0, ind, :]
                ratiobis[ii, jj, ll, :] = dfit2d['ratio']['value'][0, 0, :]
                np.savez(output_file,
                         tol=ltol, nbs=lbsplines, nsub=lsub,
                         time=time, cost=cost, nfev=nfev,
                         kTiev=kTiev, ratio=ratiobis, phi=dfit2d['pts_phi'],
                         datapath=[datapath], dconstraints=dconstraints)


def plot_benchmark(input_file,
                   nsubmin=None, nsubmax=None,
                   tolmin=None, tolmax=None,
                   nbsmin=None, nbsmax=None,
                   plotnfev=False, tmax=None,
                   fs=None, dmargin=None, cmap=None):

    # Check input
    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.05, 'right': 0.95,
                   'bottom': 0.06, 'top': 0.93,
                   'wspace': None, 'hspace': 0.2}

    # Load data
    out = np.load(input_file)
    datapath = out.get('datapath', np.array([_DATAPATH])).tolist()[0]
    tol, nsub, nbs = [out[kk] for kk in ['tol', 'nsub', 'nbs']]
    cost, nfev, time = [out[kk] for kk in ['cost', 'nfev', 'time']]
    kTiev, ratio, phi = [out[kk] for kk in ['kTiev', 'ratio', 'phi']]
    if len(nsub) == cost.shape[1]-1:
        nsub = np.r_[nsub, 450457]

    # Add boundaries
    nsubmin = nsub.min()-1 if nsubmin is None else nsubmin
    nsubmax = nsub.max()+1 if nsubmax is None else nsubmax
    tolmin = tol.min()/2 if tolmin is None else tolmin
    tolmax = tol.max()*2 if tolmax is None else tolmax
    nbsmin = nbs.min()-0.5 if nbsmin is None else nbsmin
    nbsmax = nbs.max()+0.5 if nbsmax is None else nbsmax

    indsub = (nsub >= nsubmin) & (nsub <= nsubmax)
    indtol = (tol >= tolmin) & (tol <= tolmax)
    indnbs = (nbs >= nbsmin) & (nbs <= nbsmax)
    tol = tol[indtol]
    nsub = nsub[indsub]
    nbs = nbs[indnbs]
    cost = cost[indtol, ...][:, indsub, :][..., indnbs]
    time = time[indtol, ...][:, indsub, :][..., indnbs]
    nfev = nfev[indtol, ...][:, indsub, :][..., indnbs]
    kTiev = kTiev[indtol, ...][:, indsub, ...][:, :, indnbs, :]
    ratio = ratio[indtol, ...][:, indsub, ...][:, :, indnbs, :]

    # Adjust
    ntol, nnsub, nnbs = map(len, [tol, nsub, nbs])
    chin = np.sqrt(2.*cost) / nsub[None, :, None]
    time = time / 60.
    Ti = kTiev*1e-3

    vmcost = (np.nanmin(chin), np.nanmax(chin))
    vmtime = [np.nanmin(time), np.nanmax(time)]
    vmnfev = (np.nanmin(nfev), np.nanmax(nfev))
    if tmax is not None:
        vmtime[1] = tmax

    extent = (nbs.min()-0.5, nbs.max()+0.5, 0.5, ntol+0.5)
    xt = nbs
    xtl = ['{}'.format(ss) for ss in nbs]
    yt = np.arange(1, ntol+1)
    ytl = ['{:1.0e}'.format(tt) for tt in tol]

    alpha = 1. - mplcolors.Normalize(
        vmin=np.log(np.nanmin(chin)),
        vmax=np.log(np.nanmax(chin)),
    )(np.log(chin))

    # -------------
    # Plot 1
    nbaxv = 3 if plotnfev is True else 2

    fig = plt.figure(figsize=fs)
    fig.suptitle(datapath)
    gs = gridspec.GridSpec(nbaxv, (nnsub+1)*3, **dmargin)

    dax = {'cost': [None for ii in range(nnsub)],
           'time': [None for ii in range(nnsub)]}
    if plotnfev is True:
        dax['nfev'] = [None for ii in range(nnsub)]

    for ii in range(nnsub):
        if np.all(np.isnan(cost[:, ii, :])):
            continue
        sharex = None if ii == 0 else dax['cost'][0]
        sharey = None if ii == 0 else dax['cost'][0]
        dax['cost'][ii] = fig.add_subplot(gs[0, ii*3:(ii+1)*3],
                                          sharex=sharex, sharey=sharey,
                                          adjustable='datalim')
        dax['time'][ii] = fig.add_subplot(gs[1, ii*3:(ii+1)*3],
                                          sharex=sharex, sharey=sharey,
                                          adjustable='datalim')
        if plotnfev is True:
            dax['nfev'][ii] = fig.add_subplot(gs[2, ii*3:(ii+1)*3],
                                              sharex=sharex, sharey=sharey,
                                              adjustable='datalim')

        dax['cost'][ii].set_title('nsubset = {}'.format(nsub[ii]))
        if ii == 0:
            dax['cost'][ii].set_ylabel(r'$\chi_{norm}$'+'\ntol')
            dax['cost'][ii].set_yticks(yt)
            dax['cost'][ii].set_yticklabels(ytl)
            dax['time'][ii].set_ylabel('time\ntol')
            dax['time'][ii].set_yticks(yt)
            dax['time'][ii].set_yticklabels(ytl)
            if plotnfev is True:
                dax['nfev'][ii].set_ylabel('nfev\ntol')
                dax['nfev'][ii].set_yticks(yt)
                dax['nfev'][ii].set_yticklabels(ytl)
        else:
            plt.setp(dax['cost'][ii].get_yticklabels(), visible=False)
            plt.setp(dax['time'][ii].get_yticklabels(), visible=False)
            if plotnfev is True:
                plt.setp(dax['nfev'][ii].get_yticklabels(), visible=False)
        dax['cost'][ii].set_xticks(xt)
        dax['cost'][ii].set_xticklabels(xtl)
        dax['time'][ii].set_xticks(xt)
        dax['time'][ii].set_xticklabels(xtl)
        if plotnfev is True:
            dax['nfev'][ii].set_xticks(xt)
            dax['nfev'][ii].set_xticklabels(xtl)
            dax['nfev'][ii].set_xlabel('nbsplines')

        imcost = dax['cost'][ii].imshow(chin[:, ii, :],
                                        extent=extent, aspect='auto',
                                        interpolation='nearest',
                                        origin='lower', cmap=cmap,
                                        vmin=vmcost[0], vmax=vmcost[1])
        imtime = dax['time'][ii].imshow(time[:, ii, :],
                                        extent=extent, aspect='auto',
                                        interpolation='nearest',
                                        origin='lower', cmap=cmap,
                                        vmin=vmtime[0], vmax=vmtime[1])
        if plotnfev is True:
            imnfev = dax['nfev'][ii].imshow(nfev[:, ii, :],
                                            extent=extent, aspect='auto',
                                            interpolation='nearest',
                                            origin='lower', cmap=cmap,
                                            vmin=vmnfev[0], vmax=vmnfev[1])

    dax['cost_cb'] = fig.add_subplot(gs[0, -2])
    dax['time_cb'] = fig.add_subplot(gs[1, -2])
    dax['cost_cb'].set_title(r'$\chi_{norm}$')
    dax['time_cb'].set_title(r'time (min)')
    plt.colorbar(imcost, cax=dax['cost_cb'],
                 orientation='vertical')
    plt.colorbar(imtime, cax=dax['time_cb'],
                 orientation='vertical')
    if plotnfev is True:
        dax['nfev_cb'] = fig.add_subplot(gs[2, -2])
        dax['nfev_cb'].set_title(r'nfev')
        plt.colorbar(imnfev, cax=dax['nfev_cb'],
                     orientation='vertical')

    # -------------
    # Plot 2
    fig = plt.figure(figsize=fs)
    fig.suptitle(datapath)
    gs = gridspec.GridSpec(2, nnsub, **dmargin)

    lcol = ['y', 'c', 'g', 'r', 'b', 'k']
    ncol = len(lcol)
    lls = [':', '--', '-', '-.']
    nls = len(lls)

    dax2 = {'Ti': [None for ii in range(nnsub)],
            'ratio': [None for ii in range(nnsub)]}
    sharex, shareyTi, shareyR = None, None, None
    indax = -1
    for ii in range(nnsub):
        if np.all(np.isnan(Ti[:, ii, :, :])):
            continue
        indax += 1
        dax2['Ti'][ii] = fig.add_subplot(gs[0, ii],
                                         sharex=sharex, sharey=shareyTi)
        if ii == 0:
            sharex = dax2['Ti'][0]
            shareyTi = dax2['Ti'][0]
        dax2['ratio'][ii] = fig.add_subplot(gs[1, ii],
                                            sharex=sharex, sharey=shareyR)
        if ii == 0:
            shareyR = dax2['ratio'][ii]

        dax2['Ti'][ii].set_title('nsubset = {}'.format(nsub[ii]))
        dax2['ratio'][ii].set_xlabel(r'$\phi$')

        if ii == 0:
            dax2['Ti'][ii].set_ylabel(r'$kT_i$' + ' (keV)')
            dax2['ratio'][ii].set_ylabel(r'ratio' + ' (a.u.)')

        for jj in range(ntol):
            col0 = mplcolors.to_rgb(lcol[jj % ncol])
            for ll in range(nnbs):
                if np.all(np.isnan(Ti[jj, ii, ll, :])):
                    continue
                ls = lls[ll % nls]
                col = np.r_[col0, alpha[jj, ii, ll]]
                dax2['Ti'][ii].plot(phi, Ti[jj, ii, ll, :],
                                    c=col, ls=ls)
                dax2['ratio'][ii].plot(phi, ratio[jj, ii, ll, :],
                                       c=col, ls=ls)

    # Polish
    dax2['Ti'][0].set_ylim(bottom=0., top=3.)
    dax2['ratio'][0].set_ylim(bottom=0.)
    hand = [mlines.Line2D([], [], c='k', ls=lls[ll % nls])
            for ll in range(nnbs)]
    lab = ['{}'.format(nn) for nn in nbs]
    dax2['Ti'][indax].legend(hand, lab,
                             title='nbsplines',
                             loc='lower left',
                             bbox_to_anchor=(1.02, 0.))
    hand = [mlines.Line2D([], [], c=lcol[jj % ncol], ls='-')
            for jj in range(ntol)]
    lab = ['{:1.0e}'.format(tt) for tt in tol]
    dax2['ratio'][indax].legend(hand, lab,
                                title='tol',
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1.))
