
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
import scipy.stats as scpstats
import matplotlib.pyplot as plt


# ToFu-specific
import tofu.utils as utils
from . import _spectrafit2d_funccostjac as _funccostjac


_NPEAKMAX = 12
_DCONSTRAINTS = {'amp': False,
                 'width': False,
                 'shift': False,
                 'double': False,
                 'symmetry': False}
_SAME_SPECTRUM = False
_DEG = 2
_NBSPLINES = 6
_TOL1D = {'x': 1e-12, 'f': 1.e-12, 'g': 1.e-12}
_TOL2D = {'x': 1e-6, 'f': 1.e-6, 'g': 1.e-6}
_SYMMETRY_CENTRAL_FRACTION = 0.3
_BINNING = False
_SUBSET = False
_D3 = 'lines'


###########################################################
###########################################################
#
#           Preliminary
#       utility tools for 1d spectral fitting
#
###########################################################
###########################################################

# DEPRECATED
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


def get_symmetry_axis_1dprofile(phi, data, fraction=None):
    if fraction is None:
        fraction  = _SYMMETRY_CENTRAL_FRACTION

    # Find the phi in the central fraction
    phimin = np.nanmin(phi)
    phimax = np.nanmax(phi)
    phic = 0.5*(phimax + phimin)
    dphi = (phimax - phimin)*fraction
    indphi = np.abs(phi-phic) <= dphi/2.
    phiok = phi[indphi]

    # Compute new phi and associated costs
    phi2 = phi[:, None] - phiok[None, :]
    phi2min = np.min([np.nanmax(np.abs(phi2 * (phi2<0)), axis=0),
                      np.nanmax(np.abs(phi2 * (phi2>0)), axis=0)], axis=0)
    indout = np.abs(phi2) > phi2min[None, :]
    phi2p = np.abs(phi2)
    phi2n = np.abs(phi2)
    phi2p[(phi2<0) | indout] = np.nan
    phi2n[(phi2>0) | indout] = np.nan
    nok = np.min([np.sum((~np.isnan(phi2p)), axis=0),
                  np.sum((~np.isnan(phi2n)), axis=0)], axis=0)
    cost = np.full((data.shape[0], phiok.size), np.nan)
    for ii in range(phiok.size):
        indp = np.argsort(np.abs(phi2p[:, ii]))
        indn = np.argsort(np.abs(phi2n[:, ii]))
        cost[:, ii] = np.nansum(
            (data[:, indp] - data[:, indn])[:, :nok[ii]]**2,
            axis=1)
    return phiok[np.nanargmin(cost, axis=1)]


###########################################################
###########################################################
#
#           1d spectral fitting from dlines
#
###########################################################
###########################################################


def _dconstraints_double(dinput, dconstraints, defconst=_DCONSTRAINTS):
    dinput['double'] = dconstraints.get('double', defconst['double'])
    ltypes = [int, float, np.int_, np.float_]
    c0 = (isinstance(dinput['double'], bool)
          or (isinstance(dinput['double'], dict)
              and all([(kk in ['dratio', 'dshift']
                        and type(vv) in ltypes)
                       for kk, vv in dinput['double'].items()])))
    if c0 is False:
        msg = ("dconstraints['double'] must be either:\n"
               + "\t- False: no line doubling\n"
               + "\t- True:  line doublin with unknown ratio and shift\n"
               + "\t- {'dratio': float}: line doubling with:\n"
               + "\t  \t explicit ratio, unknown shift\n"
               + "\t- {'dshift': float}: line doubling with:\n"
               + "\t  \t unknown ratio, explicit shift\n"
               + "\t- {'dratio': floati, 'dshift': float}: line doubling with:\n"
               + "\t  \t explicit ratio, explicit shift")
        raise Exception(msg)


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
    print(0)        # DB
    _dconstraints_double(dinput, dconstraints, defconst=defconst)

    # ------------------------
    # Check / format width, shift, amp (groups with posssible ratio)
    # ------------------------
    print(1)        # DB
    for k0 in ['amp', 'width', 'shift']:
        dinput[k0] = _width_shift_amp(dconstraints.get(k0, defconst[k0]),
                                      keys=keys, nlines=nlines, dlines=dlines)

    # ------------------------
    # mz, symb, ion
    # ------------------------
    print(2)        # DB
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
    print(4)        # DB
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


def _dconstraints_symmetry(dinput, symmetry=None, dataphi1d=None, phi1d=None,
                           fraction=None, defconst=_DCONSTRAINTS):
    if symmetry is None:
        symmetry = defconst['symmetry']
    dinput['symmetry'] = symmetry
    if not isinstance(dinput['symmetry'], bool):
        msg = "dconstraints['symmetry'] must be a bool"
        raise Exception(msg)

    if dinput['symmetry'] is True:
        dinput['symmetry_axis'] = get_symmetry_axis_1dprofile(
            phi1d, dataphi1d, fraction=fraction)


def _checkformat_data_fit2d_dlines(data, lamb, phi,
                                   nxi=None, nxj=None, mask=None):
    msg = ("Args data, lamb, phi and mask must be:\n"
           + "\t- data: (nt, n1, n2) or (n1, n2) np.ndarray\n"
           + "\t- lamb, phi: both (n1, n2) np.ndarray\n"
           + "\t- mask: None or (n1, n2)")
    if not isinstance(data, np.ndarray):
        raise Exception(msg)
    c0 = (data.ndim in [2, 3]
          and lamb.ndim == phi.ndim == 2
          and lamb.shape == phi.shape == lamb.shape[-2:]
          and lamb.shape in [(nxi, nxj), (nxj, nxi)])
    if not c0:
        raise Exception(msg)
    if data.ndim == 2:
        data = data[None, :, :]
    if lamb.shape == (nxj, nxi):
        lamb = lamb.T
        phi = phi.T
        data = np.swapaxes(data, 1, 2)
    if mask is not None:
        if mask.shape != lamb.shape:
            if mask.T.shape == lamb.shape:
                mask = mask.T
            else:
                raise Exception(msg)
    return lamb, phi, data, mask


# ############################
#           Domain limitation
# ############################


def _checkformat_domain(domain=None):
    if domain is None:
        domain = {'lamb': {'spec': [np.inf*np.r_[-1., 1.]]},
                  'phi': {'spec': [np.inf*np.r_[-1., 1.]]}}
        return domain

    lk = ['lamb', 'phi']
    c0 = (isinstance(domain, dict)
          and all([k0 in lk for k0 in domain.keys()]))
    if not c0:
        msg = ("Arg domain must be a dict with keys {}\n".format(ls)
               + "\t- provided: {}".format(domain))
        raise Exception(msg)

    domain2 = {k0: v0 for k0, v0 in domain.items()}
    for k0 in lk:
        domain2[k0] = domain2.get(k0, [np.inf*np.r_[-1., 1.]])

    ltypesin = [list, np.ndarray]
    ltypesout = [tuple]
    for k0, v0 in domain2.items():
        c0 = (type(v0) in ltypesin + ltypesout
              and (all([(type(v1) in ltypesin + ltypesout
                         and len(v1) == 2
                         and v1[1] > v1[0]) for v1 in v0])
                   or (len(v0) == 2 and v0[1] > v0[0])))
        if not c0:
            msg = ("domain[{}] must be either a:\n".format(k0)
                   + "\t- np.ndarray or list of 2 increasing values: "
                    + "inclusive interval\n"
                   + "\t- tuple of 2 increasing values: exclusive interval\n"
                   + "\t- a list of combinations of the above\n"
                   + "  provided: {}".format(v0))
            raise Exception(msg)

        if type(v0) in ltypesout:
            v0 = [v0]
        else:
            c0 = all([(type(v1) in ltypesin + ltypesout
                       and len(v1) == 2
                       and v1[1] > v1[0]) for v1 in v0])
            if not c0:
                v0 = [v0]
        domain2[k0] = {'spec': v0,
                       'minmax': [np.nanmin(v0), np.nanmax(v0)]}
    return domain2


def apply_domain(lamb, phi, domain=None):

    domain = _checkformat_domain(domain=domain)
    ind = np.ones(lamb.shape, dtype=bool)
    for v1 in domain['lamb']['spec']:
        indi = (lamb >= v1[0]) & (lamb <= v1[1])
        if isinstance(v1, tuple):
            indi = ~indi
        ind &= indi
    for v1 in domain['phi']['spec']:
        indi = (phi >= v1[0]) & (phi <= v1[1])
        if isinstance(v1, tuple):
            indi = ~indi
        ind &= indi
    return ind, domain


# ############################
#           Domain limitation
# ############################


def _binning_check(binning, nlamb=None, nphi=None,
                   domain=None, nbsplines=None, deg=None):
    msg = ("binning must be dict of the form:\n"
           + "\t- provide number of bins:\n"
           + "\t  \t{'phi':  int,\n"
           + "\t  \t 'lamb': int}\n"
           + "\t- provide bin edges vectors:\n"
           + "\t  \t{'phi':  1d np.ndarray (increasing),\n"
           + "\t  \t 'lamb': 1d np.ndarray (increasing)}\n"
           + "  provided:\n{}".format(binning))

    # Check input
    if binning is None:
        binning = _BINNING
    if nbsplines is not None:
        c0 = isinstance(nbsplines, int) and nbsplines > 0
        if not c0:
            msg2 = ("Both nbsplines and deg must be positive int!\n"
                    + "\t- nbsplines: {}\n".format(nbsplines))
            raise Exception(msg2)

    # Check which format was passed and return None or dict
    ltypes0 = [int, float, np.int_, np.float_]
    ltypes1 = [tuple, list, np.ndarray]
    lc = [binning is False,
          (isinstance(binning, dict)
           and all([kk in ['phi', 'lamb'] for kk in binning.keys()])),
          type(binning) in ltypes0,
          type(binning) in ltypes1]
    if not any(lc):
        raise Exception(msg)
    if binning is False:
        return binning
    elif type(binning) in ltypes0:
        binning = {'phi': {'nbins': int(binning)},
                   'lamb': {'nbins': int(binning)}}
    elif type(binning) in ltypes1:
        binning = np.atleast_1d(binning).ravel()
        binning = {'phi': {'edges': binning},
                   'lamb': {'edges': binning}}
    for kk in binning.keys():
        if type(binning[kk]) in ltypes0:
            binning[kk] = {'nbins': int(binning[kk])}
        elif type(binning[kk]) in ltypes1:
            binning[kk] = {'edges': np.atleast_1d(binning[kk]).ravel()}

    c0 = all([all([k1 in ['edges', 'nbins'] for k1 in binning[k0].keys()])
              for k0 in binning.keys()])
    c0 = (c0 and
          all([((binning[k0].get('nbins') is None
                 or type(binning[k0].get('nbins')) in ltypes0)
                and (binning[k0].get('edges') is None
                 or type(binning[k0].get('edges')) in ltypes1))
              for k0 in binning.keys()]))
    if not c0:
        raise Exception(msg)

    # Check dict
    for k0 in binning.keys():
        c0 = all([k1 in ['nbins', 'edges'] for k1 in binning[k0].keys()])
        if not c0:
            raise Exception(msg)
        if binning[k0].get('nbins') is not None:
            binning[k0]['nbins'] = int(binning[k0]['nbins'])
            if binning[k0].get('edges') is None:
                binning[k0]['edges'] = np.linspace(
                    domain[k0]['minmax'][0], domain[k0]['minmax'][1],
                    binning[k0]['nbins'] + 1, endpoint=True)
            else:
                binning[k0]['edges'] = np.atleast_1d(
                    binning[k0]['edges']).ravel()
                if binning[k0]['nbins'] != binning[k0]['edges'].size - 1:
                    raise Exception(msg)
        elif binning[k0].get('bin_edges') is not None:
            binning[k0]['edges'] = np.atleast_1d(binning[k0]['edges']).ravel()
            binning[k0]['nbins'] = binning[k0]['edges'].size - 1
        else:
            raise Exception(msg)

        if not np.allclose(binning[k0]['edges'],
                           np.unique(binning[k0]['edges'])):
            raise Exception(msg)

    # Optional check vs nbsplines and deg
    if nbsplines is not None:
        if binning['phi']['nbins'] <= nbsplines:
            msg = ("The number of bins")
            raise Exception(msg)
    return binning


def binning_2d_data(lamb, phi, data, indok=None,
                    domain=None, binning=None, nbsplines=None):

    # ------------------
    # Checkformat input
    binning = _binning_check(binning, domain=domain, nbsplines=nbsplines)
    if binning is False:
        return lamb, phi, data, indok, binning

    nphi = binning['phi']['nbins']
    nlamb = binning['lamb']['nbins']
    bins = (binning['lamb']['edges'], binning['phi']['edges'])
    nspect = data.shape[0]
    npts = nlamb*nphi

    # ------------------
    # Compute

    databin = scpstats.binned_statistic_2d(
        lamb[indok], phi[indok], data[:, indok],
        statistic='mean', bins=bins,
        range=None, expand_binnumbers=True)[0]

    lambbin = 0.5*(binning['lamb']['edges'][1:]
                   + binning['lamb']['edges'][:-1])
    phibin = 0.5*(binning['phi']['edges'][1:]
                  + binning['phi']['edges'][:-1])
    lambbin = np.repeat(lambbin[:, None], nphi, axis=1)
    phibin = np.repeat(phibin[None, :], nlamb, axis=0)
    indok = np.any(~np.isnan(databin), axis=0)
    return lambbin, phibin, databin, indok, binning


# ############################
#           Prepare data
# ############################


def _get_subset_indices(subset, indlogical):
    if subset is None:
        subset = _SUBSET
    if subset is False:
        return indlogical

    msg = ("subset must be either:\n"
           + "\t- an array of bool of shape: {}\n".format(indlogical.shape)
           + "\t- a positive int (nb. of indices to be kept from indlogical)\n"
           + "You provided: {}".format(subset))
    c0 = ((isinstance(subset, np.ndarray)
           and subset.shape == indlogical.shape
           and 'bool' in subset.dtype.name)
          or (type(subset) in [int, float, np.int_, np.float_]
              and subset >= 0))
    if not c0:
        raise Exception(msg)

    if isinstance(subset, np.ndarray):
        indlogical = subset & indlogical
    else:
        subset = np.random.default_rng().choice(
            indlogical.sum(), size=int(indlogical.sum() - subset),
            replace=False, shuffle=False)
        ind = indlogical.nonzero()
        indlogical[ind[0][subset], ind[1][subset]] = False
    return indlogical


def multigausfit2d_from_dlines_prepare(data, lamb, phi,
                                       mask=None, domain=None,
                                       pos=None, binning=None,
                                       nbsplines=None, subset=None,
                                       noise_ind=None,
                                       nxi=None, nxj=None):

    # Check input
    if pos is None:
        pos = False
    if subset is None:
        if binning in [None, False]:
            subset = _SUBSET
        else:
            subset = False
    if noise_ind is None:
        noise_ind = False

    # Check shape of data (multiple time slices possible)
    lamb, phi, data, mask = _checkformat_data_fit2d_dlines(
        data, lamb, phi,
        nxi=nxi, nxj=nxj, mask=mask)

    if pos is True:
        data[data < 0.] = 0.

    # Use valid data only and optionally restrict lamb / phi
    indok, domain = apply_domain(lamb, phi, domain=domain)
    if mask is not None:
        indok &= mask
    indok &= np.any(~np.isnan(data), axis=0)
    domain['lamb']['minmax'] = [np.nanmin(lamb[indok]), np.nanmax(lamb[indok])]
    domain['phi']['minmax'] = [np.nanmin(phi[indok]), np.nanmax(phi[indok])]

    # Optionnal 2d binning
    lambbin, phibin, databin, indok, binning = binning_2d_data(
        lamb, phi, data, indok=indok,
        binning=binning, domain=domain, nbsplines=nbsplines)

    # Get vertical profile of mean data
    if binning is False:
        nphid = nxj
        phi1d_bins = np.linspace(domain['phi']['minmax'][0],
                                 domain['phi']['minmax'][1], nxj)
        phi1d = 0.5*(phi1d_bins[1:] + phi1d_bins[:-1])
        dataphi1d =  scpstats.binned_statistic(
            phi[indok], data[:, indok],
            bins=phi1d_bins, statistic='mean')[0]
    else:
        phi1d = (binning['phi']['edges'][1:] + binning['phi']['edges'][:-1])/2.
        dataphi1d = np.nanmean(databin, axis=1)

    # Optionally fit only on subset
    # randomly pick subset indices (replace=False => no duplicates)
    indok = _get_subset_indices(subset, indok)

    dprepare = {'data': databin, 'lamb': lambbin, 'phi': phibin,
                'domain': domain, 'binning': binning, 'indok': indok,
                'phi1d': phi1d, 'dataphi1d': dataphi1d,
                'pos': pos, 'subset': subset, 'nxi': nxi, 'nxj': nxj}
    return dprepare


def multigausfit2d_from_dlines_dbsplines(knots=None, deg=None, nbsplines=None,
                                         phimin=None, phimax=None,
                                         symmetryaxis=None):
    # Check / format input
    if deg is None:
        deg = _DEG
    if not (isinstance(deg, int) and deg <= 3):
        msg = "deg must be a int <= 3 (the degree of the bsplines to be used!)"
        raise Exception(msg)
    if symmetryaxis is None:
        symmetryaxis = False

    if nbsplines is None:
        nbsplines = _NBSPLINES
    if not isinstance(nbsplines, int):
        msg = "nbsplines must be a int (the degree of the bsplines to be used!)"
        raise Exception(msg)

    if knots is None:
        if phimin is None or phimax is None:
            msg = "Please provide phimin and phimax if knots is not provided!"
            raise Exception(msg)
        if symmetryaxis is False:
            knots = np.linspace(phimin, phimax, nbsplines + 1 - deg)
        else:
            symax = np.nanmean(symmetryaxis)
            phi2max = np.max(np.abs(np.r_[phimin, phimax] - symax))
            knots = np.linspace(0, phi2max, nbsplines + 1 - deg)

    if not np.allclose(knots, np.unique(knots)):
        msg = "knots must be a vector of unique values!"
        raise Exception(msg)

    # Get knots for scipy (i.e.: with multiplicity)
    if deg > 0:
        knots_mult = np.r_[[knots[0]]*deg, knots, [knots[-1]]*deg]
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


def valid_indices_phi(sig1d, phi1d, threshold=None):
    ind = sig1d < threshold
    return ind


def multigausfit2d_from_dlines_dinput(dlines=None,
                                      dconstraints=None,
                                      Ti=None, vi=None,
                                      deg=None, nbsplines=None, knots=None,
                                      domain=None,
                                      dataphi1d=None, phi1d=None,
                                      fraction=None, defconst=_DCONSTRAINTS):

    # ------------------------
    # Check / format basics
    # ------------------------

    # Select relevant lines (keys, lamb)
    keys = np.array([k0 for k0 in dlines.keys()])
    lamb = np.array([dlines[k0]['lambda'] for k0 in keys])
    if domain is not None:
        ind = ((lamb >= domain['lamb']['minmax'][0])
               & (lamb <= domain['lamb']['minmax'][1]))
        keys = keys[ind]
        lamb = lamb[ind]
    inds = np.argsort(lamb)
    keys, lamb = keys[inds], lamb[inds]
    nlines = lamb.size

    # Error message for constraints
    msg = "dconstraints must be a dict of constraints for spectrum fitting"

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
    ltypes = [int, float, np.int_, np.float_]
    dinput = {}

    # ------------------------
    # Check / format symmetry
    # ------------------------
    _dconstraints_symmetry(dinput, symmetry=dconstraints.get('symmetry'),
                           dataphi1d=dataphi1d, phi1d=phi1d,
                           fraction=fraction, defconst=defconst)

    # ------------------------
    # Check / format double
    # ------------------------
    _dconstraints_double(dinput, dconstraints, defconst=defconst)

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

    # Set Ti and vi flags
    if Ti is None:
        dinput['Ti'] = dinput['width']['ind'].shape[0] < nlines
    elif isinstance(Ti, bool):
        dinput['Ti'] = Ti
    else:
        msg = ("Arg Ti must be None, True or False!\n"
               + "\t- provided: {}".format(Ti))
        raise Exception(msg)
    if vi is None:
        dinput['vi'] = dinput['shift']['ind'].shape[0] < nlines
    elif isinstance(vi, bool):
        dinput['vi'] = vi
    else:
        msg = ("Arg vi must be None, True or False!\n"
               + "\t- provided: {}".format(vi))
        raise Exception(msg)

    # Get dict of bsplines
    dinput.update(multigausfit2d_from_dlines_dbsplines(
        knots=knots, deg=deg, nbsplines=nbsplines,
        phimin=domain['phi']['minmax'][0],
        phimax=domain['phi']['minmax'][1],
        symmetryaxis=dinput.get('symmetry_axis')))

    # S/N threshold indices
    # dinput['valid_indphi'] = _valid_indices(spectvect1d, phi1d,
                                            # threshold=threshold)
    # dinput['threshold'] = threshold
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
    elif isinstance(dinput['double'], dict):
        if dinput['double'].get('dshift') is None:
            dind['dshift'] = -1
            sizex += 1
        elif dinput['double'].get('dratio') is None:
            dind['dratio'] = -1
            sizex += 1

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
                                     domain=None, dinput=None,
                                     dind=None, nspect=None):
    if scales is None:
        scales = np.full((nspect, dind['sizex']), np.nan)
        Dphi = domain['phi']['minmax'][1] - domain['phi']['minmax'][0]
        Dlamb = domain['lamb']['minmax'][1] - domain['lamb']['minmax'][0]
        lambm = domain['lamb']['minmax'][0]
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
                indlamb = np.abs(lamb-dinput['lines'][indl]) < Dlamb/20.
                indj = ind & indlamb
                if not np.any(indj):
                    lamb0 = dinput['lines'][indl]
                    msg = ("All nan in region scanned for scale:\n"
                           + "\t- amp[{}]\n".format(jj)
                           + "\t- bspline[{}]\n".format(ii)
                           + "\t- phi approx {}\n".format(dinput['ptsx0'][ii])
                           + "\t- lamb approx {}".format(lamb0))
                    plt.figure()
                    plt.scatter(lamb, phi, s=6, c='k', marker='.')
                    plt.scatter(lamb[ind], phi[ind], s=6, c='r', marker='.')
                    plt.scatter(lamb[indlamb], phi[indlamb],
                                s=6, c='b', marker='.');
                    plt.gca().set_xlim(domain['lamb']['minmax'])
                    plt.gca().set_ylim(domain['phi']['minmax'])
                    raise Exception(msg)
                scales[:, iax[ii, jj]] = np.nanmean(data[:, indj], axis=1)
        scales[:, iwx] = (Dlamb/(20*lambm))**2
        scales[:, isx] = Dlamb/(50*lambm)
        if dinput['double'] is not False:
            if dinput['double'] is True:
                scales[:, dind['dratio']] = 1.
                scales[:, dind['dshift']] = Dlamb/(50*lambm)
            else:
                if dinput['double'].get('dratio') is None:
                    scales[:, dind['dratio']] = 1.
                if dinput['double'].get('dshift') is None:
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
    if double is not False:
        if double is True:
            x0_scale[:, dind['dratio']] = 0.7
            x0_scale[:, dind['dshift']] = 0.7
        else:
            if double.get('dratio') is None:
                x0_scale[:, dind['dratio']] = 0.7
            if double.get('dshift') is None:
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
    if double is not False:
        if double is True:
            xup[dind['dratio']] = 1.6
            xlo[dind['dratio']] = 0.4
            xup[dind['dshift']] = 10.
            xlo[dind['dshift']] = -10.
        else:
            if double.get('dratio') is None:
                xup[dind['dratio']] = 1.6
                xlo[dind['dratio']] = 0.4
            if double.get('dshift') is None:
                xup[dind['dshift']] = 10.
                xlo[dind['dshift']] = -10.
    bounds_scale = (xlo, xup)
    return bounds_scale


# ############################
#           Perform 2d fit
# ############################


def multigausfit2d_from_dlines(dprepare=None, dinput=None, dx0=None,
                               scales=None, x0_scale=None, bounds_scale=None,
                               method=None, tr_solver=None, tr_options=None,
                               xtol=None, ftol=None, gtol=None,
                               max_nfev=None, chain=None, verbose=None,
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
        jac = 'sparse'
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox', 'lm'], method
    if tr_solver is None:
        tr_solver = None
    if tr_options is None:
        tr_options = {}
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

    nspect = dprepare['data'].shape[0]

    # ---------------------------
    # Get indices dict
    dind = multigausfit2d_from_dlines_ind(dinput)

    # Get scaling
    if dinput['symmetry'] is True:
        phi2 = np.abs(dprepare['phi'] - np.nanmean(dinput['symmetry_axis']))
    else:
        phi2 = dprepare['phi']
    scales = multigausfit2d_from_dlines_scale(
        dprepare['data'], dprepare['lamb'], phi2,
        domain=dprepare['domain'], dinput=dinput,
        dind=dind, scales=scales, nspect=nspect)

    # Get initial guess
    x0_scale = multigausfit2d_from_dlines_x0(
        dind=dind, double=dinput['double'],
        nspect=nspect, nbs=dinput['nbs'],
        dx0=dx0, keys=dinput['keys'])

    # get bounds
    bounds_scale = multigausfit2d_from_dlines_bounds(dind['sizex'],
                                                     dind,
                                                     dinput['double'])

    # Get function, cost function and jacobian
    (func_detail,
     func_cost, jacob) = _funccostjac.multigausfit2d_from_dlines_funccostjac(
         dprepare['lamb'], phi2, indok=dprepare['indok'],
         binning=dprepare['binning'], dinput=dinput,
         dind=dind, jac=jac)

    # ---------------------------
    # Prepare output
    datacost = np.reshape(dprepare['data'][:, dprepare['indok']],
                          (nspect, dprepare['indok'].sum()))
    sol_x = np.full((nspect, dind['sizex']), np.nan)
    sol_tot = np.full(dprepare['data'].shape, np.nan)
    success = np.full((nspect,), np.nan)
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    validity = np.zeros((nspect,), dtype=int)
    message = ['' for ss in range(nspect)]
    errmsg = ['' for ss in range(nspect)]

    if dprepare.get('indok_var') is not None:
        msg = ('indok_var not implemented yet!')
        raise Exception(msg)
        if dprepare['indok_var'].ndim == 3:
            indok_var = dprepare['indok_var'].reshape((nspect,
                                                       dprepare['lamb'].size))
        else:
            indok_var = [dprepare['indok_var'].ravel()]*nspect
    else:
        indok_var = [False]*nspect
    dprepare['indok_var'] = indok_var

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
            res = scpopt.least_squares(
                func_cost, x0_scale[ii, :],
                jac=jacob, bounds=bounds_scale,
                method=method, ftol=ftol, xtol=xtol,
                gtol=gtol, x_scale=1.0, f_scale=1.0,
                loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options=tr_options,
                jac_sparsity=None, max_nfev=max_nfev,
                verbose=verbscp, args=(),
                kwargs={'data': datacost[ii, :],
                        'scales': scales[ii, :],
                        'indok_var': indok_var[ii]})

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
            sol_x[ii, :] = res.x

        except Exception as err:
            errmsg[ii] = str(err)
            validity[ii] = -1

    # Isolate dratio and dshift
    dratio, dshift = None, None
    if dinput['double'] is not False:
        if dinput['double'] is True:
            dratio = sol_x[:, dind['dratio']]*scales[:, dind['dratio']]
            dshift = sol_x[:, dind['dshift']]*scales[:, dind['dshift']]
        else:
            if dinput['double'].get('dratio') is None:
                dratio = sol_x[:, dind['dratio']]*scales[:, dind['dratio']]
            else:
                dratio = np.full((nspect,), dinput['double']['dratio'])
            if dinput['double'].get('dshift') is None:
                dshift = sol_x[:, dind['dshift']]*scales[:, dind['dshift']]
            else:
                dshift = np.full((nspect,), dinput['double']['dshift'])

    if verbose > 0:
        dt = (dtm.datetime.now()-t0).total_seconds()
        msg = ("Total computation time:"
               + "\t{} s for {} steps ({} s per step)".format(
                   round(dt, ndigits=3), nspect,
                   round(dt/nspect, ndigits=3)))
        print(msg)

    # ---------------------------
    # Format output as dict
    dout = {'dprepare': dprepare, 'dinput': dinput, 'dind': dind,
            'scales': scales, 'x0_scale': x0_scale,
            'bounds_scale': bounds_scale, 'phi2': phi2,
            'jac': jac, 'sol_x': sol_x,
            'dratio': dratio, 'dshift': dshift,
            'time': time, 'success': success,
            'validity': validity, 'errmsg': np.array(errmsg),
            'cost': cost, 'nfev': nfev, 'msg': np.array(message)}
    return dout


# ############################
#       Extract data from result         
# ############################


def fit2d_get_data_checkformat(dfit2d=None,
                               pts_phi=None, npts_phi=None,
                               amp=None, Ti=None, vi=None,
                               pts_lamb_phi_total=None,
                               pts_lamb_phi_detail=None,
                               dprepare=None, dinput=None):
    # dfit2d
    lk = ['dprepare', 'dinput', 'dind', 'sol_x', 'jac', 'phi2', 'scales']
    c0 = (isinstance(dfit2d, dict)
          and all([ss in dfit2d.keys() for ss in lk]))
    if not isinstance(dfit2d, dict):
        msg = ("dfit2d must be a dict with at least the following keys:\n"
               + "\t- {}\n".format(lk)
               + "\t- provided: {}".format(dfit2d))
        raise Exception(msg)

    d3 = {'amp': [amp, 'amp'],
          'Ti': [Ti, 'width'],
          'vi': [vi, 'shift']}
    # amp, Ti, vi
    for k0 in d3.keys():
        if d3[k0][0] is None:
            d3[k0][0] = True
        if d3[k0][0] is True:
            d3[k0][0] = _D3
        if d3[k0][0] is False:
            continue
        lc = [d3[k0][0] in ['lines', 'x'],
              isinstance(d3[k0][0], str),
              (isinstance(d3[k0][0], list)
               and all([isinstance(isinstance(ss, str) for ss in d3[k0][0])]))]
        if not any(lc):
            msg = ("Arg {} must be either:\n".format(k0)
                   + "\t- 'x': return all unique {}\n".format(k0)
                   + "\t- 'lines': return {} for all lines (inc. duplicates)\n"
                   + "\t- str: a key in:\n"
                   + "\t\t{}\n".format(dinput['keys'])
                   + "\t\t{}\n".format(dinput[d3[k0][1]]['keys'])
                   + "\t- list: a list of keys (see above)\n"
                   + "Provided: {}".format(d3[k0][0]))
            raise Exception(msg)
        if lc[0]:
            if d3[k0][0] == 'lines':
                d3[k0][0] = {'type': d3[k0][0],
                             'ind': np.arange(0, dinput['nlines'])}
            else:
                d3[k0][0] = {'type': d3[k0][0],
                             'ind': np.arange(0,
                                              dinput[d3[k0][1]]['keys'].size)}
        elif lc[1]:
            d3[k0][0] = [d3[k0][0]]

        if isinstance(d3[k0][0], list):
            lc = [all([ss in dinput['keys'] for ss in d3[k0][0]]),
                  all([ss in dinput[d3[k0][1]]['keys'] for ss in d3[k0][0]])]
            if not any(lc):
                msg = ("Arg must contain either keys from:\n"
                       + "\t- lines keys: {}\n".format(dinput['keys'])
                       + "\t- {} keys: {}".format(k0,
                                                  dinput[d3[k0][1]]['keys']))
                raise Exception(msg)
            if lc[0]:
                d3[k0][0] = {'type': 'lines',
                             'ind': np.array([
                                 (dinput['keys']==ss).nonzero()[0][0]
                                 for ss in d3[k0][0]], dtype=int)}
            else:
                d3[k0][0] = {'type': 'x',
                             'ind': np.array([
                                 (dinput[d3[k0][1]]['keys']==ss).nonzero()[0][0]
                                 for ss in d3[k0][0]], dtype=int)}
        d3[k0][0]['field'] = d3[k0][1]
        d3[k0] = d3[k0][0]

    # pts_phi, npts_phi
    c0 = any([v0 is not False for v0 in d3.values()])
    c1 = [pts_phi is not None, npts_phi is not None]
    if all(c1):
        msg = "Arg pts_phi and npts_phi cannot be both provided!"
        raise Exception(msg)
    if not any(c1):
        npts_phi = (2*dinput['deg']-1)*(dinput['knots'].size-1) + 1
    if npts_phi is not None:
        npts_phi = int(npts_phi)
        pts_phi = np.linspace(dprepare['domain']['phi']['minmax'][0],
                              dprepare['domain']['phi']['minmax'][1],
                              npts_phi)
    else:
        pts_phi = np.array(pts_phi).ravel()

    # pts_lamb_phi_total, pts_lamb_phi_detail
    if pts_lamb_phi_total is None:
        if dprepare is None:
            pts_lamb_phi_total = False
        else:
            pts_lamb_phi_total = np.array([dprepare['lamb'],
                                           dprepare['phi']])
    if pts_lamb_phi_detail is None:
        pts_lamb_phi_detail = False
    if pts_lamb_phi_total is not False:
        pts_lamb_phi_total = np.array(pts_lamb_phi_total)
    if pts_lamb_phi_detail is not False:
        pts_lamb_phi_detail = np.array(pts_lamb_phi_detail)

    return d3, pts_phi, pts_lamb_phi_total, pts_lamb_phi_detail


def _get_phi_profile(key,
                     nspect=None, dinput=None,
                     dind=None, sol_x=None, scales=None,
                     typ=None, ind=None, pts_phi=None):
    ncoefs = ind.size
    val = np.full((nspect, pts_phi.size, ncoefs), np.nan)
    BS = BSpline(dinput['knots_mult'],
                 np.ones((dinput['nbs'], ncoefs), dtype=float),
                 dinput['deg'],
                 extrapolate=False, axis=0)
    if typ == 'lines':
        keys = dinput['keys'][ind]
    else:
        keys = dinput[key]['keys'][ind]
    indbis = dind[key][typ][:, ind]
    for ii in range(nspect):
        BS.c = sol_x[ii, indbis] * scales[ii, indbis]
        val[ii, :, :] = BS(pts_phi)
    return keys, val


def fit2d_extract_data(dfit2d=None,
                       pts_phi=None, npts_phi=None,
                       amp=None, Ti=None, vi=None,
                       pts_lamb_phi_total=None, pts_lamb_phi_detail=None):

    # Check format input
    out = fit2d_get_data_checkformat(
        dfit2d=dfit2d,
        amp=amp, Ti=Ti, vi=vi,
        pts_phi=pts_phi, npts_phi=npts_phi,
        pts_lamb_phi_total=pts_lamb_phi_total,
        pts_lamb_phi_detail=pts_lamb_phi_detail,
        dprepare=dfit2d['dprepare'], dinput=dfit2d['dinput'])

    d3, pts_phi, pts_lamb_phi_total, pts_lamb_phi_detail = out
    nspect = dfit2d['dprepare']['data'].shape[0]

    # Prepare output
    shape = tuple(np.r_[nspect, pts_lamb_phi_total.shape])
    sol_tot = np.full(shape, np.nan)
    if any([v0 is not False for v0 in d3.values()]):
        nbs = dfit2d['dinput']['nbs']

    dout = {}
    # amp
    if d3['amp'] is not False:
        keys, val = _get_phi_profile(
            d3['amp']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['amp']['type'], ind=d3['amp']['ind'])
        dout['amp'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # Ti
    if d3['Ti'] is not False:
        keys, val = _get_phi_profile(
            d3['Ti']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['Ti']['type'], ind=d3['Ti']['ind'])
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        if d3['Ti']['type'] == 'lines':
            indTi = np.arange(0, dfit2d['dinput']['nlines'])
        else:
            indTi = np.array([iit[0]
                              for iit in dfit2d['dind']['width']['jac']])
        indTi = indTi[d3['Ti']['ind']]
        val = (conv * val
               * dfit2d['dinput']['mz'][indTi][None, None, :]
               * scpct.c**2)
        dout['Ti'] = {'keys': keys, 'values': val, 'units': 'eV'}

    # vi
    if d3['vi'] is not False:
        keys, val = _get_phi_profile(
            d3['vi']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['vi']['type'], ind=d3['vi']['ind'])
        val = val * scpct.c
        dout['vi'] = {'keys': keys, 'values': val, 'units': 'm.s^-1'}

    # sol_detail and sol_tot
    sold, solt = False, False
    if pts_lamb_phi_detail is not False or pts_lamb_phi_total is not False:

        func_detail = _funccostjac.multigausfit2d_from_dlines_funccostjac(
            dfit2d['dprepare']['lamb'], dfit2d['phi2'],
            indok=dfit2d['dprepare']['indok'],
            binning=dfit2d['dprepare']['binning'],
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], jac=dfit2d['jac'])[0]

        if pts_lamb_phi_detail is not False:
            shape = tuple(np.r_[nspect, pts_lamb_phi_detail.shape,
                                dfit2d['dinput']['nlines']+1,
                                dfit2d['dinput']['nbs']])
            sold = np.full(shape, np.nan)
        if pts_lamb_phi_total is not False:
            shape = tuple(np.r_[nspect, pts_lamb_phi_total.shape])
            solt = np.full(shape, np.nan)

        for ii in range(nspect):

            # Separate and reshape output
            fd = func_detail(dfit2d['sol_x'][ii, :],
                             scales=dfit2d['scales'][ii, :],
                             indok_var=dfit2d['dprepare']['indok_var'][ii])

            if pts_lamb_phi_detail is not False:
                sold[ii, ...] = fd
            if pts_lamb_phi_total is not False:
                solt[ii, ...] = np.nansum(np.nansum(fd, axis=-1), axis=-1)

    dout['sol_detail'] = sold
    dout['sol_tot'] = solt
    dout['units'] = 'a.u.'

    # Add input args
    dout['d3'] = d3
    dout['pts_phi'] = pts_phi
    dout['pts_lamb_phi_detail'] = pts_lamb_phi_detail
    dout['pts_lamb_phi_total'] = pts_lamb_phi_total
    return dout
