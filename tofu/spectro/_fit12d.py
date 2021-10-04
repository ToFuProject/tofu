
# Built-in
import os
import warnings
import itertools as itt
import copy
import datetime as dtm      # DB

# Common
import numpy as np
import scipy.optimize as scpopt
import scipy.interpolate as scpinterp
import scipy.constants as scpct
import scipy.sparse as sparse
from scipy.interpolate import BSpline
import scipy.stats as scpstats
import matplotlib.pyplot as plt


# ToFu-specific
import tofu.utils as utils
from . import _fit12d_funccostjac as _funccostjac
from . import _plot


__all__ = [
    'fit1d_dinput', 'fit2d_dinput',
    'fit12d_dvalid', 'fit12d_dscales',
    'fit1d', 'fit2d',
    'fit1d_extract', 'fit2d_extract',
]


_NPEAKMAX = 12
_DCONSTRAINTS = {
    'bck_amp': False,
    'bck_rate': False,
    'amp': False,
    'width': False,
    'shift': False,
    'double': False,
    'symmetry': False,
}
_DORDER = ['amp', 'width', 'shift']
_SAME_SPECTRUM = False
_DEG = 2
_NBSPLINES = 13
_TOL1D = {'x': 1e-10, 'f': 1.e-10, 'g': 1.e-10}
_TOL2D = {'x': 1e-6, 'f': 1.e-6, 'g': 1.e-6}
_SYMMETRY_CENTRAL_FRACTION = 0.3
_BINNING = False
_POS = False
_SUBSET = False
_CHAIN = True
_METHOD = 'trf'
_LOSS = 'linear'
_D3 = {
    'bck_amp': 'x',
    'bck_rate': 'x',
    'amp': 'x',
    'coefs': 'lines',
    'ratio': 'lines',
    'Ti': 'x',
    'width': 'x',
    'vi': 'x',
    'shift': 'lines',   # necessarily by line for de-normalization (*lamb0)
}
_VALID_NSIGMA = 6.
_VALID_FRACTION = 0.8
_SIGMA_MARGIN = 3.
_ALLOW_PICKLE = True
_LTYPES = [int, float, np.int_, np.float_]

_DBOUNDS = {
    'bck_amp': (0., 3.),
    'bck_rate': (-3., 3.),
    'amp': (0, 2),
    'width': (0.01, 2.),
    'shift': (-2, 2),
    'dratio': (0., 2.),
    'dshift': (-10., 10.),
    'bs': (-10., 10.),
}
_DX0 = {
    'bck_amp': 1.,
    'bck_rate': 0.,
    'amp': 1.,
    'width': 1.,
    'shift': 0.,
    'dratio': 0.5,
    'dshift': 0.,
    'bs': 1.,
}


###########################################################
###########################################################
#
#           Preliminary
#       utility tools for 1d spectral fitting
#
###########################################################
###########################################################


def get_symmetry_axis_1dprofile(phi, data, cent_fraction=None):
    """ On a series of 1d vertical profiles, find the best symmetry axis """

    if cent_fraction is None:
        cent_fraction = _SYMMETRY_CENTRAL_FRACTION

    # Find the phi in the central fraction
    phimin = np.nanmin(phi)
    phimax = np.nanmax(phi)
    phic = 0.5*(phimax + phimin)
    dphi = (phimax - phimin)*cent_fraction
    indphi = np.abs(phi-phic) <= dphi/2.
    phiok = phi[indphi]

    # Compute new phi and associated costs
    phi2 = phi[:, None] - phiok[None, :]
    phi2min = np.min([np.nanmax(np.abs(phi2 * (phi2 < 0)), axis=0),
                      np.nanmax(np.abs(phi2 * (phi2 > 0)), axis=0)], axis=0)
    indout = np.abs(phi2) > phi2min[None, :]
    phi2p = np.abs(phi2)
    phi2n = np.abs(phi2)
    phi2p[(phi2 < 0) | indout] = np.nan
    phi2n[(phi2 > 0) | indout] = np.nan
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


def _checkformat_dconstraints(dconstraints=None, defconst=None):
    # Check constraints
    if dconstraints is None:
        dconstraints = defconst

    # Check dconstraints keys
    lk = sorted(_DCONSTRAINTS.keys())
    c0 = (
        isinstance(dconstraints, dict)
        and all([k0 in lk for k0 in dconstraints.keys()])
    )
    if not c0:
        msg = (
            "\ndconstraints should contain constraints for spectrum fitting\n"
            + "It be a dict with the following keys:\n"
            + "\t- available keys: {}\n".format(lk)
            + "\t- provided keys: {}".format(dconstraints.keys())
        )
        raise Exception(msg)

    # copy to avoid modifying reference
    return copy.deepcopy(dconstraints)


def _checkformat_dconstants(dconstants=None, dconstraints=None):
    if dconstants is None:
        return

    lk = [kk for kk in sorted(dconstraints.keys()) if kk != 'symmetry']
    if not isinstance(dconstants, dict):
        msg = (
            "\ndconstants should be None or a dict with keys in:\n"
            + "\t- available keys: {}\n".format(lk)
            + "\t- provided : {}".format(type(dconstants))
        )
        raise Exception(msg)

    # Check dconstraints keys
    lc = [
        k0 for k0, v0 in dconstants.items()
        if not (
            k0 in lk
            and (
                (
                    k0 in _DORDER
                    and isinstance(v0, dict)
                    and all([
                        k1 in dconstraints[k0].keys()
                        and type(v1) in _LTYPES
                        for k1, v1 in v0.items()
                    ])
                )
                or (
                    k0 not in _DORDER
                    and type(v0) in _LTYPES
                )
            )
        )
    ]
    if len(lc) > 0:
        dc0 = [
            '\t\t{}: {}'.format(
                kk,
                sorted(dconstraints[kk].keys()) if kk in _DORDER else float
            )
            for kk in lk
        ]
        dc1 = [
            '\t\t{}: {}'.format(
                kk,
                sorted(dconstants[kk].keys())
                if kk in _DORDER else dconstants[kk]
            )
            for kk in sorted(dconstants.keys())
        ]
        msg = (
            "\ndconstants should be None or a dict with keys in:\n"
            + "\t- available keys:\n"
            + "\n".join(dc0)
            + "\n\t- provided keys:\n"
            + "\n".join(dc1)
        )
        raise Exception(msg)

    # copy to avoid modifying reference
    return copy.deepcopy(dconstants)


def _dconstraints_double(dinput, dconstraints, defconst=_DCONSTRAINTS):
    dinput['double'] = dconstraints.get('double', defconst['double'])
    c0 = (
        isinstance(dinput['double'], bool)
        or (
            isinstance(dinput['double'], dict)
            and all([
                kk in ['dratio', 'dshift'] and type(vv) in _LTYPES
                for kk, vv in dinput['double'].items()
            ])
        )
    )
    if c0 is False:
        msg = (
            "dconstraints['double'] must be either:\n"
            + "\t- False: no line doubling\n"
            + "\t- True:  line doubling with unknown ratio and shift\n"
            + "\t- {'dratio': float}: line doubling with:\n"
            + "\t  \t explicit ratio, unknown shift\n"
            + "\t- {'dshift': float}: line doubling with:\n"
            + "\t  \t unknown ratio, explicit shift\n"
            + "\t- {'dratio': float, 'dshift': float}: line doubling with:\n"
            + "\t  \t explicit ratio, explicit shift"
        )
        raise Exception(msg)


def _width_shift_amp(
    indict, dconstants=None,
    keys=None, dlines=None, nlines=None, k0=None,
):

    # ------------------------
    # Prepare error message
    msg = ''
    pavail = sorted(set(itt.chain.from_iterable([
        v0.keys() for v0 in dlines.values()
    ])))

    # ------------------------
    # Check case
    c0 = indict is False
    c1 = (
        isinstance(indict, str)
        and indict in pavail
    )
    c2 = (
        isinstance(indict, dict)
        and all([
            isinstance(k1, str)
            and (
                (isinstance(v1, str))   # and v0 in keys)
                or (
                    isinstance(v1, list)
                    and all([
                        isinstance(v2, str)
                        # and v1 in keys
                        for v2 in v1
                    ])
                )
            )
            for k1, v1 in indict.items()
        ])
    )
    c3 = (
        isinstance(indict, dict)
        and all([
            # ss in keys
            isinstance(vv, dict)
            and all([s1 in ['key', 'coef', 'offset'] for s1 in vv.keys()])
            and isinstance(vv['key'], str)
            for ss, vv in indict.items()
        ])
    )
    c4 = (
        isinstance(indict, dict)
        and isinstance(indict.get('keys'), list)
        and isinstance(indict.get('ind'), np.ndarray)
    )
    if not any([c0, c1, c2, c3, c4]):
        msg = (
            "dconstraints['{}'] shoud be either:\n".format(k0)
            + "\t- False ({}): no constraint\n".format(c0)
            + "\t- str ({}): key from dlines['<lines>'] ".format(c1)
            + "to be used as criterion\n"
            + "\t\t available crit: {}\n".format(pavail)
            + "\t- dict ({}): ".format(c2)
            + "{str: line_keyi or [line_keyi, ..., line_keyj}\n"
            + "\t- dict ({}): ".format(c3)
            + "{line_keyi: {'key': str, 'coef': , 'offset': }}\n"
            + "\t- dict ({}): ".format(c4)
            + "{'keys': [], 'ind': np.ndarray}\n"
            + "  Available line_keys:\n{}\n".format(sorted(keys))
            + "  You provided:\n{}".format(indict)
        )
        raise Exception(msg)

    # ------------------------
    # str key to be taken from dlines as criterion
    if c0:
        lk = keys
        ind = np.eye(nlines)
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': np.ones((nlines,)),
            'offset': np.zeros((nlines,)),
        }

    if c1:
        lk = sorted(set([dlines[k1].get(indict, k1) for k1 in keys]))
        ind = np.array([
            [dlines[k2].get(indict, k2) == k1 for k2 in keys]
            for k1 in lk
        ])
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': np.ones((nlines,)),
            'offset': np.zeros((nlines,)),
        }

    elif c2:
        lkl = []
        for k1, v1 in indict.items():
            if isinstance(v1, str):
                v1 = [v1]
            v1 = [k2 for k2 in v1 if k2 in keys]
            c0 = (
                len(set(v1)) == len(v1)
                and all([k2 not in lkl for k2 in v1])
            )
            if not c0:
                msg = (
                    "Inconsistency in indict[{}], either:\n".format(k1)
                    + "\t- v1 not unique: {}\n".format(v1)
                    + "\t- some v1 not in keys: {}\n".format(keys)
                    + "\t- some v1 in lkl:      {}".format(lkl)
                )
                raise Exception(msg)
            indict[k1] = v1
            lkl += v1
        for k1 in set(keys).difference(lkl):
            indict[k1] = [k1]
        lk = sorted(set(indict.keys()))
        ind = np.array([[k2 in indict[k1] for k2 in keys] for k1 in lk])
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': np.ones((nlines,)),
            'offset': np.zeros((nlines,)),
        }

    elif c3:
        lk = sorted(set([v0['key'] for v0 in indict.values()]))
        lk += sorted(set(keys).difference(indict.keys()))
        ind = np.array([
            [indict.get(k2, {'key': k2})['key'] == k1 for k2 in keys]
            for k1 in lk
        ])
        coefs = np.array([
            indict.get(k1, {'coef': 1.}).get('coef', 1.) for k1 in keys
        ])
        offset = np.array([
            indict.get(k1, {'offset': 0.}).get('offset', 0.) for k1 in keys
        ])
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': coefs,
            'offset': offset,
        }

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


###########################################################
###########################################################
#
#           2d spectral fitting from dlines
#
###########################################################
###########################################################


def _dconstraints_symmetry(
    dinput,
    dprepare=None,
    symmetry=None,
    cent_fraction=None,
    defconst=_DCONSTRAINTS,
):
    if symmetry is None:
        symmetry = defconst['symmetry']
    dinput['symmetry'] = symmetry
    if not isinstance(dinput['symmetry'], bool):
        msg = "dconstraints['symmetry'] must be a bool"
        raise Exception(msg)

    if dinput['symmetry'] is True:
        dinput['symmetry_axis'] = get_symmetry_axis_1dprofile(
            dprepare['phi1d'],
            dprepare['dataphi1d'],
            cent_fraction=cent_fraction,
        )


###########################################################
###########################################################
#
#           data, lamb, phi conformity checks
#
###########################################################
###########################################################


def _checkformat_data_fit12d_dlines_msg(data, lamb, phi=None, mask=None):
    datash = data.shape if isinstance(data, np.ndarray) else type(data)
    lambsh = lamb.shape if isinstance(lamb, np.ndarray) else type(lamb)
    phish = phi.shape if isinstance(phi, np.ndarray) else type(phi)
    masksh = mask.shape if isinstance(mask, np.ndarray) else type(mask)
    shaped = '(nt, n1)' if phi is None else '(nt, n1, n2)'
    shape = '(n1,)' if phi is None else '(n1, n2)'
    msg = ("Args data, lamb, phi and mask must be:\n"
           + "\t- data: {} or {} np.ndarray\n".format(shaped, shape)
           + "\t- lamb, phi: both {} np.ndarray\n".format(shape)
           + "\t- mask: None or {}\n".format(shape)
           + "  You provided:\n"
           + "\t - data: {}\n".format(datash)
           + "\t - lamb: {}\n".format(lambsh))
    if phi is not None:
        msg += "\t - phi: {}\n".format(phish)
    msg += "\t - mask: {}\n".format(masksh)
    return msg


def _checkformat_data_fit12d_dlines(
    data, lamb, phi=None,
    nxi=None, nxj=None, mask=None,
    is2d=False,
):

    # Check types
    c0 = isinstance(data, np.ndarray) and isinstance(lamb, np.ndarray)
    if is2d:
        c0 &= isinstance(phi, np.ndarray)

    if not c0:
        msg = _checkformat_data_fit12d_dlines_msg(
            data, lamb, phi=phi, mask=mask,
        )
        raise Exception(msg)

    # Check shapes 1
    mindim = 1 if phi is None else 2
    phi1d, lamb1d, dataphi1d, datalamb1d = None, None, None, None
    if is2d:

        # special case
        c1 = lamb.ndim == phi.ndim == 1
        if c1:
            if nxi is None:
                nxi = lamb.size
            if nxj is None:
                nxj = phi.size
            lamb1d = np.copy(lamb)
            phi1d = np.copy(phi)
            lamb = np.repeat(lamb[None, :], nxj, axis=0)
            phi = np.repeat(phi[:, None], nxi, axis=1)

        c0 = (
            data.ndim in mindim + np.r_[0, 1]
            and (
                lamb.ndim == mindim
                and lamb.shape == data.shape[-mindim:]
                and lamb.shape == phi.shape
                and lamb.shape in [(nxi, nxj), (nxj, nxi)]
            )
        )

    else:
        c0 = (
            data.ndim in mindim + np.r_[0, 1]
            and lamb.ndim == mindim
            and lamb.shape == data.shape[-mindim:]
        )

    if not c0:
        msg = _checkformat_data_fit12d_dlines_msg(
            data, lamb, phi=phi, mask=mask,
        )
        raise Exception(msg)

    # Check shapes 2
    if data.ndim == mindim:
        data = data[None, ...]
    if is2d and c1:
        dataphi1d = np.nanmean(data, axis=2)
        datalamb1d = np.nanmean(data, axis=1)
    if is2d and lamb.shape == (nxi, nxj):
        lamb = lamb.T
        phi = phi.T
        data = np.swapaxes(data, 1, 2)

    # mask
    if mask is not None:
        if mask.shape != lamb.shape:
            if phi is not None and mask.T.shape == lamb.shape:
                mask = mask.T
            else:
                msg = _checkformat_data_fit12d_dlines_msg(
                    data, lamb, phi=phi, mask=mask,
                )
                raise Exception(msg)

    if is2d:
        return lamb, phi, data, mask, phi1d, lamb1d, dataphi1d, datalamb1d
    else:
        return lamb, data, mask


###########################################################
###########################################################
#
#           Domain limitation
#
###########################################################
###########################################################


def _checkformat_domain(domain=None, keys=['lamb', 'phi']):

    if keys is None:
        keys = ['lamb', 'phi']
    if isinstance(keys, str):
        keys = [keys]

    if domain is None:
        domain = {k0: {'spec': [np.inf*np.r_[-1., 1.]]} for k0 in keys}
        return domain

    c0 = (
        isinstance(domain, dict)
        and all([k0 in keys for k0 in domain.keys()])
    )
    if not c0:
        msg = ("\nArg domain must be a dict with keys {}\n".format(keys)
               + "\t- provided: {}".format(domain))
        raise Exception(msg)

    domain2 = {k0: v0 for k0, v0 in domain.items()}
    for k0 in keys:
        domain2[k0] = domain2.get(k0, [np.inf*np.r_[-1., 1.]])

    ltypesin = [list, np.ndarray]
    ltypesout = [tuple]
    for k0, v0 in domain2.items():
        c0 = (
            type(v0) in ltypesin + ltypesout
            and (
                (
                    all([type(v1) in _LTYPES for v1 in v0])
                    and len(v0) == 2
                    and v0[1] > v0[0]
                )
                or (
                    all([
                        type(v1) in ltypesin + ltypesout
                        and all([type(v2) in _LTYPES for v2 in v1])
                        and len(v1) == 2
                        and v1[1] > v1[0]
                        for v1 in v0
                    ])
                )
            )
        )
        if not c0:
            msg = (
                "domain[{}] must be either a:\n".format(k0)
                + "\t- np.ndarray or list of 2 increasing values: "
                + "inclusive interval\n"
                + "\t- tuple of 2 increasing values: exclusive interval\n"
                + "\t- a list of combinations of the above\n"
                + "  provided: {}".format(v0)
            )
            raise Exception(msg)

        if type(v0) in ltypesout:
            v0 = [v0]
        else:
            c0 = all([
                type(v1) in ltypesin + ltypesout
                and len(v1) == 2
                and v1[1] > v1[0]
                for v1 in v0
            ])
            if not c0:
                v0 = [v0]
        domain2[k0] = {
            'spec': v0,
            'minmax': [np.nanmin(v0), np.nanmax(v0)],
        }
    return domain2


def apply_domain(lamb=None, phi=None, domain=None):

    lc = [lamb is not None, phi is not None]
    if not lc[0]:
        msg = "At least lamb must be provided!"
        raise Exception(msg)

    din = {'lamb': lamb}
    if lc[1]:
        din['phi'] = phi

    domain = _checkformat_domain(domain=domain, keys=din.keys())
    ind = np.ones(lamb.shape, dtype=bool)
    for k0, v0 in din.items():
        indin = np.zeros(v0.shape, dtype=bool)
        indout = np.zeros(v0.shape, dtype=bool)
        for v1 in domain[k0]['spec']:
            indi = (v0 >= v1[0]) & (v0 <= v1[1])
            if isinstance(v1, tuple):
                indout |= indi
            else:
                indin |= indi
        ind = ind & indin & (~indout)
    return ind, domain


###########################################################
###########################################################
#
#           binning (2d only)
#
###########################################################
###########################################################


def _binning_check(
    binning,
    domain=None, nbsplines=None,
):
    lk = ['phi', 'lamb']
    lkall = lk + ['nperbin']
    msg = (
        "binning must be dict of the form:\n"
        + "\t- provide number of bins:\n"
        + "\t  \t{'phi':  int,\n"
        + "\t  \t 'lamb': int}\n"
        + "\t- provide bin edges vectors:\n"
        + "\t  \t{'phi':  1d np.ndarray (increasing),\n"
        + "\t  \t 'lamb': 1d np.ndarray (increasing)}\n"
        + "  provided:\n{}".format(binning)
    )

    # Check input
    if binning is None:
        binning = _BINNING
    if nbsplines is None:
        nbsplines = False
    if nbsplines is not False:
        c0 = isinstance(nbsplines, int) and nbsplines > 0
        if not c0:
            msg2 = (
                "Both nbsplines and deg must be positive int!\n"
                + "\t- nbsplines: {}\n".format(nbsplines)
            )
            raise Exception(msg2)

    # Check which format was passed and return None or dict
    ltypes0 = _LTYPES
    ltypes1 = [tuple, list, np.ndarray]
    lc = [
        binning is False,
        (
            isinstance(binning, dict)
            and all([kk in lkall for kk in binning.keys()])
        ),
        type(binning) in ltypes0,
        type(binning) in ltypes1,
    ]
    if not any(lc):
        raise Exception(msg)
    if binning is False:
        return binning
    elif type(binning) in ltypes0:
        binning = {
            'phi': {'nbins': int(binning)},
            'lamb': {'nbins': int(binning)},
        }
    elif type(binning) in ltypes1:
        binning = np.atleast_1d(binning).ravel()
        binning = {
            'phi': {'edges': binning},
            'lamb': {'edges': binning},
        }
    for kk in lk:
        if type(binning[kk]) in ltypes0:
            binning[kk] = {'nbins': int(binning[kk])}
        elif type(binning[kk]) in ltypes1:
            binning[kk] = {'edges': np.atleast_1d(binning[kk]).ravel()}

    c0 = all([
        all([k1 in ['edges', 'nbins'] for k1 in binning[k0].keys()])
        for k0 in lk
    ])
    c0 = (
        c0
        and all([
            (
                (
                    binning[k0].get('nbins') is None
                    or type(binning[k0].get('nbins')) in ltypes0
                )
                and (
                    binning[k0].get('edges') is None
                    or type(binning[k0].get('edges')) in ltypes1
                )
            )
            for k0 in lk
        ])
    )
    if not c0:
        raise Exception(msg)

    # Check dict
    for k0 in lk:
        c0 = all([k1 in ['nbins', 'edges'] for k1 in binning[k0].keys()])
        if not c0:
            raise Exception(msg)
        if binning[k0].get('nbins') is not None:
            binning[k0]['nbins'] = int(binning[k0]['nbins'])
            if binning[k0].get('edges') is None:
                binning[k0]['edges'] = np.linspace(
                    domain[k0]['minmax'][0], domain[k0]['minmax'][1],
                    binning[k0]['nbins'] + 1,
                    endpoint=True,
                )
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
    if nbsplines is not False:
        if binning['phi']['nbins'] <= nbsplines:
            msg = (
                "The number of bins is too high:\n"
                + "\t- nbins =     {}\n".format(binning['phi']['nbins'])
                + "\t- nbsplines = {}".format(nbsplines)
            )
            raise Exception(msg)
    return binning


def binning_2d_data(
    lamb, phi, data, indok=None,
    domain=None, binning=None,
    nbsplines=None,
    phi1d=None, lamb1d=None,
    dataphi1d=None, datalamb1d=None,
):

    # ------------------
    # Checkformat input
    binning = _binning_check(
        binning,
        domain=domain, nbsplines=nbsplines,
    )

    nspect = data.shape[0]
    if binning is False:
        if phi1d is None:
            phi1d_bins = np.linspace(domain['phi'][0], domain['phi'][1], 100)
            lamb1d_bins = np.linspace(
                domain['lamb'][0], domain['lamb'][1], 100,
            )
            dataf = data.reshape((nspect, data.shape[1]*data.shape[2]))
            dataphi1d = scpstats.binned_statistics(
                phi.ravel(),
                dataf,
                statistic='sum',
            )
            datalamb1d = scpstats.binned_statistics(
                lamb.ravel(),
                dataf,
                statistic='sum',
            )
            phi1d = 0.5*(phi1d_bins[1:] + phi1d_bins[:-1])
            lamb1d = 0.5*(lamb1d_bins[1:] + lamb1d_bins[:-1])
            import pdb; pdb.set_trace()     # DB

        return (
            lamb, phi, data, indok, binning,
            phi1d, lamb1d, dataphi1d, datalamb1d,
        )

    else:
        nphi = binning['phi']['nbins']
        nlamb = binning['lamb']['nbins']
        bins = (binning['lamb']['edges'], binning['phi']['edges'])

        # ------------------
        # Compute
        databin = np.full((nspect, nphi, nlamb), np.nan)
        nperbin = np.full((nspect, nphi, nlamb), np.nan)
        for ii in range(nspect):
            databin[ii, ...] = scpstats.binned_statistic_2d(
                phi[indok[ii, ...]],
                lamb[indok[ii, ...]],
                data[indok[ii, ...]],
                statistic='sum', bins=bins,
                range=None, expand_binnumbers=True,
            )[0]
            nperbin[ii, ...] = scpstats.binned_statistic_2d(
                phi[indok[ii, ...]],
                lamb[indok[ii, ...]],
                np.ones((indok[ii, ...].sum(),), dtype=int),
                statistic='sum', bins=bins,
                range=None, expand_binnumbers=True,
            )[0]
        binning['nperbin'] = nperbin

        lambbin = 0.5*(
            binning['lamb']['edges'][1:] + binning['lamb']['edges'][:-1]
        )
        phibin = 0.5*(
            binning['phi']['edges'][1:] + binning['phi']['edges'][:-1]
        )
        lambbin = np.repeat(lambbin[None, :], nphi, axis=0)
        phibin = np.repeat(phibin[:, None], nlamb, axis=1)
        indok = ~np.isnan(databin)

        # dataphi1d
        phi1d = phibin
        lamb1d = lambbin
        dataphi1d = np.nanmean(databin, axis=2)
        datalamb1d = np.nanmean(databin, axis=1)

        return (
            lambbin, phibin, databin, indok, binning,
            phi1d, lamb1d, dataphi1d, datalamb1d,
        )


###########################################################
###########################################################
#
#           dprepare dict
#
###########################################################
###########################################################


def _get_subset_indices(subset, indlogical):
    if subset is None:
        subset = _SUBSET
    if subset is False:
        return indlogical

    c0 = ((isinstance(subset, np.ndarray)
           and subset.shape == indlogical.shape
           and 'bool' in subset.dtype.name)
          or (type(subset) in [int, float, np.int_, np.float_]
              and subset >= 0))
    if not c0:
        msg = ("subset must be either:\n"
               + "\t- an array of bool of shape: {}\n".format(indlogical.shape)
               + "\t- a positive int (nb. of ind. to keep from indlogical)\n"
               + "You provided:\n{}".format(subset))
        raise Exception(msg)

    if isinstance(subset, np.ndarray):
        indlogical = subset[None, ...] & indlogical
    else:
        subset = np.random.default_rng().choice(
            indlogical.sum(), size=int(indlogical.sum() - subset),
            replace=False, shuffle=False)
        for ii in range(indlogical.shape[0]):
            ind = indlogical[ii, ...].nonzero()
            indlogical[ii, ind[0][subset], ind[1][subset]] = False
    return indlogical


def _extract_lphi_spectra(
    data, phi, lamb,
    lphi=None, lphi_tol=None,
    databin=None, binning=None, nlamb=None,
):
    """ Extra several 1d spectra from 2d image at lphi """

    # --------------
    # Check input
    if lphi is None:
        lphi = False
    if lphi is False:
        lphi_tol = False
    if lphi is not False:
        lphi = np.atleast_1d(lphi).astype(float).ravel()
        lphi_tol = float(lphi_tol)

    if lphi is False:
        return False, False
    nphi = len(lphi)

    # --------------
    # Compute non-trivial cases

    if binning is False:
        if nlamb is None:
            nlamb = lamb.shape[1]
        lphi_lamb = np.linspace(lamb.min(), lamb.max(), nlamb+1)
        lphi_spectra = np.full((data.shape[0], lphi_lamb.size-1, nphi), np.nan)
        for ii in range(nphi):
            indphi = np.abs(phi - lphi[ii]) < lphi_tol
            lphi_spectra[:, ii, :] = scpstats.binned_statistic(
                lamb[indphi], data[:, indphi], bins=lphi_lamb,
                statistic='mean', range=None,
            )[0]

    else:
        lphi_lamb = 0.5*(
            binning['lamb']['edges'][1:] + binning['lamb']['edges'][:-1]
        )
        lphi_phi = 0.5*(
            binning['phi']['edges'][1:] + binning['phi']['edges'][:-1]
        )
        lphi_spectra = np.full((data.shape[0], nphi, lphi_lamb.size), np.nan)
        lphi_spectra1 = np.full((data.shape[0], nphi, lphi_lamb.size), np.nan)
        for ii in range(nphi):
            datai = databin[:, np.abs(lphi_phi - lphi[ii]) < lphi_tol, :]
            iok = np.any(~np.isnan(datai), axis=1)
            for jj in range(datai.shape[0]):
                if np.any(iok[jj, :]):
                    lphi_spectra[jj, ii, iok[jj, :]] = np.nanmean(
                        datai[jj, :, iok[jj, :]],
                        axis=1,
                    )

    return lphi_spectra, lphi_lamb


def _checkformat_possubset(pos=None, subset=None):
    if pos is None:
        pos = _POS
    c0 = isinstance(pos, bool) or type(pos) in _LTYPES
    if not c0:
        msg = ("Arg pos must be either:\n"
               + "\t- False: no positivity constraints\n"
               + "\t- True: all negative values are set to nan\n"
               + "\t- float: all negative values are set to pos")
        raise Exception(msg)
    if subset is None:
        subset = _SUBSET
    return pos, subset


def multigausfit1d_from_dlines_prepare(
    data=None, lamb=None,
    mask=None, domain=None,
    pos=None, subset=None,
):

    # --------------
    # Check input
    pos, subset = _checkformat_possubset(pos=pos, subset=subset)

    # Check shape of data (multiple time slices possible)
    lamb, data, mask = _checkformat_data_fit12d_dlines(
        data, lamb, mask=mask,
    )

    # --------------
    # Use valid data only and optionally restrict lamb
    indok, domain = apply_domain(lamb, domain=domain)
    if mask is not None:
        indok &= mask

    # Optional positivity constraint
    if pos is not False:
        if pos is True:
            data[data < 0.] = np.nan
        else:
            data[data < 0.] = pos

    # Introduce time-dependence (useful for valid)
    indok = indok[None, ...] & (~np.isnan(data))

    # Recompute domain
    domain['lamb']['minmax'] = [
        np.nanmin(lamb[np.any(indok, axis=0)]),
        np.nanmax(lamb[np.any(indok, axis=0)])
    ]

    # --------------
    # Optionally fit only on subset
    # randomly pick subset indices (replace=False => no duplicates)
    indok = _get_subset_indices(subset, indok)

    if np.any(np.isnan(data[indok])):
        msg = (
            "Some NaNs in data not caught by indok!"
        )
        raise Exception(msg)

    # --------------
    # Return
    dprepare = {
        'data': data,
        'lamb': lamb,
        'domain': domain,
        'indok': indok,
        'pos': pos,
        'subset': subset,
    }
    return dprepare


def multigausfit2d_from_dlines_prepare(
    data=None, lamb=None, phi=None,
    mask=None, domain=None,
    pos=None, binning=None,
    nbsplines=None, deg=None, subset=None,
    nxi=None, nxj=None,
    lphi=None, lphi_tol=None,
):

    # --------------
    # Check input
    pos, subset = _checkformat_possubset(pos=pos, subset=subset)

    # Check shape of data (multiple time slices possible)
    (
        lamb, phi, data, mask,
        phi1d, lamb1d, dataphi1d, datalamb1d,
    ) = _checkformat_data_fit12d_dlines(
        data, lamb, phi,
        nxi=nxi, nxj=nxj, mask=mask, is2d=True,
    )

    # --------------
    # Use valid data only and optionally restrict lamb / phi
    indok, domain = apply_domain(lamb, phi, domain=domain)
    if mask is not None:
        indok &= mask

    # Optional positivity constraint
    if pos is not False:
        if pos is True:
            data[data < 0.] = np.nan
        else:
            data[data < 0.] = pos

    # Introduce time-dependence (useful for valid)
    indok = indok[None, ...] & (~np.isnan(data))

    # Recompute domain
    domain['lamb']['minmax'] = [
        np.nanmin(lamb[np.any(indok, axis=0)]),
        np.nanmax(lamb[np.any(indok, axis=0)])
    ]
    domain['phi']['minmax'] = [
        np.nanmin(phi[np.any(indok, axis=0)]),
        np.nanmax(phi[np.any(indok, axis=0)])
    ]

    # --------------
    # Optionnal 2d binning
    (
        lambbin, phibin, databin, indok, binning,
        phi1d, lamb1d, dataphi1d, datalamb1d,
    ) = binning_2d_data(
        lamb, phi, data, indok=indok,
        binning=binning, domain=domain,
        nbsplines=nbsplines,
        phi1d=phi1d, lamb1d=lamb1d,
        dataphi1d=dataphi1d, datalamb1d=datalamb1d,
    )

    # --------------
    # Optionally fit only on subset
    # randomly pick subset indices (replace=False => no duplicates)
    indok = _get_subset_indices(subset, indok)

    # --------------
    # Optionally extract 1d spectra at lphi
    lphi_spectra, lphi_lamb = _extract_lphi_spectra(
        data, phi, lamb,
        lphi, lphi_tol,
        databin=databin,
        binning=binning,
    )

    # --------------
    # Return
    dprepare = {
        'data': databin, 'lamb': lambbin, 'phi': phibin,
        'domain': domain, 'binning': binning, 'indok': indok,
        'pos': pos, 'subset': subset, 'nxi': nxi, 'nxj': nxj,
        'lphi': lphi, 'lphi_tol': lphi_tol,
        'lphi_spectra': lphi_spectra, 'lphi_lamb': lphi_lamb,
        'phi1d': phi1d, 'dataphi1d': dataphi1d,
        'lamb1d': lamb1d, 'datalamb1d': datalamb1d,
    }
    return dprepare


def multigausfit2d_from_dlines_dbsplines(
    knots=None, deg=None, nbsplines=None,
    phimin=None, phimax=None,
    symmetryaxis=None,
):
    # Check / format input
    if nbsplines is None:
        nbsplines = _NBSPLINES
    c0 = [nbsplines is False, isinstance(nbsplines, int)]
    if not any(c0):
        msg = "nbsplines must be a int (degree of bsplines to be used!)"
        raise Exception(msg)

    if nbsplines is False:
        lk = ['knots', 'knots_mult', 'nknotsperbs', 'ptsx0', 'nbs', 'deg']
        return dict.fromkeys(lk, False)

    if deg is None:
        deg = _DEG
    if not (isinstance(deg, int) and deg <= 3):
        msg = "deg must be a int <= 3 (the degree of the bsplines to be used!)"
        raise Exception(msg)
    if symmetryaxis is None:
        symmetryaxis = False

    if knots is None:
        if phimin is None or phimax is None:
            msg = "Please provide phimin and phimax if knots is not provided!"
            raise Exception(msg)
        if symmetryaxis is False:
            knots = np.linspace(phimin, phimax, nbsplines + 1 - deg)
        else:
            phi2max = np.max(
                np.abs(np.r_[phimin, phimax][None, :] - symmetryaxis[:, None])
            )
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
        ptsx0 = np.r_[
            knots[0],
            np.mean(knots[:2]),
            knots[1:-1],
            np.mean(knots[-2:]),
            knots[-1],
        ]
        msg = ("degree 3 not fully implemented yet!"
               + "Approximate values for maxima positions")
        warnings.warn(msg)
    assert ptsx0.size == nbs
    dbsplines = {
        'knots': knots, 'knots_mult': knots_mult,
        'nknotsperbs': nknotsperbs, 'ptsx0': ptsx0,
        'nbs': nbs, 'deg': deg,
    }
    return dbsplines


###########################################################
###########################################################
#
#           dvalid dict (S/N ratio)
#
###########################################################
###########################################################


def _dvalid_checkfocus_errmsg(focus=None, focus_half_width=None,
                              lines_keys=None):
    msg = ("Please provide focus as:\n"
           + "\t- str: the key of an available spectral line:\n"
           + "\t\t{}\n".format(lines_keys)
           + "\t- float: a wavelength value\n"
           + "\t- a list / tuple / flat np.ndarray of such\n"
           + "  You provided:\n"
           + "{}\n\n".format(focus)
           + "Please provide focus_half_width as:\n"
           + "\t- float: a unique wavelength value for all focus\n"
           + "\t- a list / tuple / flat np.ndarray of such\n"
           + "  You provided:\n"
           + "{}".format(focus_half_width))
    return msg


def _dvalid_checkfocus(
    focus=None,
    focus_half_width=None,
    lines_keys=None,
    lines_lamb=None,
    lamb=None,
):
    """ Check the provided focus is properly formatted and convert it

    focus specifies the wavelength range of interest in which S/N is evaluated
    It can be provided as:
        - a spectral line key (or list of such)
        - a wavelength (or list of such)

    For each wavelength, a spectral range centered on it, is defined using
    the provided focus_half_width
    The focus_half_width can be a unique value applied to all or a list of
    values of the same length as focus.

    focus is then return as a (n, 2) array where:
        each line gives a central wavelength and halfwidth of interest

    """
    if focus in [None, False]:
        return False

    # Check focus and transform to array of floats
    lc0 = [
        type(focus) in [str] + _LTYPES,
        type(focus) in [list, tuple, np.ndarray]
    ]
    if not any(lc0):
        msg = _dvalid_checkfocus_errmsg(
            focus, focus_half_width, lines_keys,
        )
        raise Exception(msg)

    if lc0[0] is True:
        focus = [focus]
    for ii in range(len(focus)):
        if focus[ii] not in lines_keys and type(focus[ii]) not in _LTYPES:
            msg = _dvalid_checkfocus_errmsg(
                focus, focus_half_width, lines_keys,
            )
            raise Exception(msg)

    focus = np.array([
        lines_lamb[(lines_keys == ff).nonzero()[0][0]]
        if ff in lines_keys else ff for ff in focus
    ])

    # Check focus_half_width and transform to array of floats
    if focus_half_width is None:
        focus_half_width = (np.nanmax(lamb) - np.nanmin(lamb))/10.
    lc0 = [
        type(focus_half_width) in _LTYPES,
        (
            type(focus_half_width) in [list, tuple, np.ndarray]
            and len(focus_half_width) == focus.size
            and all([type(fhw) in _LTYPES for fhw in focus_half_width])
        )
    ]
    if not any(lc0):
        msg = _dvalid_checkfocus_errmsg(
            focus, focus_half_width, lines_keys,
        )
        raise Exception(msg)
    if lc0[0] is True:
        focus_half_width = np.full((focus.size,), focus_half_width)
    return np.array([focus, np.r_[focus_half_width]]).T


def fit12d_dvalid(
    data=None, lamb=None, phi=None,
    indok=None, binning=None,
    valid_nsigma=None, valid_fraction=None,
    focus=None, focus_half_width=None,
    lines_keys=None, lines_lamb=None, dphimin=None,
    nbs=None, deg=None, knots_mult=None, nknotsperbs=None,
    return_fract=None,
):
    """ Return a dict of valid time steps and phi indices

    data points are considered valid if there signal is sufficient:
        np.sqrt(data) >= valid_nsigma

    data is supposed to be provided in counts (or photons).. TBC!!!

    """

    # Check inputs
    if valid_nsigma is None:
        valid_nsigma = _VALID_NSIGMA
    if valid_fraction is None:
        valid_fraction = _VALID_FRACTION
    if binning is None:
        binning = False
    if dphimin is None:
        dphimin = 0.
    if return_fract is None:
        return_fract = False
    data2d = data.ndim == 3
    nspect = data.shape[0]

    focus = _dvalid_checkfocus(
        focus,
        focus_half_width=focus_half_width,
        lines_keys=lines_keys,
        lines_lamb=lines_lamb,
        lamb=lamb,
    )

    # Get indices of pts with enough signal
    ind = np.zeros(data.shape, dtype=bool)
    if indok is None:
        isafe = (~np.isnan(data))
        isafe[isafe] = data[isafe] >= 0.
        # Ok with and w/o binning if data provided as counts / photons
        # and binning was done by sum (and not mean)
        ind[isafe] = np.sqrt(data[isafe]) > valid_nsigma
    else:
        ind[indok] = np.sqrt(data[indok]) > valid_nsigma

    # Derive indt and optionally dphi and indknots
    indbs, dphi = False, False
    if focus is not False:
        # TBC
        lambok = np.rollaxis(
            np.array([np.abs(lamb - ff[0]) < ff[1] for ff in focus]),
            0,
            lamb.ndim+1,
        )
        indall = ind[..., None] & lambok[None, ...]

    if data2d is True:
        # Make sure there are at least deg + 2 different phi
        deltaphi = np.max(np.diff(knots_mult))
        # Code ok with and without binning :-)
        if focus is False:
            fract = np.full((nspect, nbs), np.nan)
            for ii in range(nbs):
                iphi = (
                    (phi >= knots_mult[ii])
                    & (phi < knots_mult[ii+nknotsperbs-1])
                )
                fract[:, ii] = (
                    np.sum(np.sum(ind & iphi[None, ...], axis=-1), axis=-1)
                    / np.sum(iphi)
                )
            indbs = fract > valid_fraction
        else:
            fract = np.full((nspect, nbs, len(focus)), np.nan)
            for ii in range(nbs):
                iphi = ((phi >= knots_mult[ii])
                        & (phi < knots_mult[ii+nknotsperbs-1]))
                fract[:, ii, :] = (
                    np.sum(np.sum(indall & iphi[None, ..., None],
                                  axis=1), axis=1)
                    / np.sum(np.sum(iphi[..., None] & lambok,
                                    axis=0), axis=0))
            indbs = np.all(fract > valid_fraction, axis=2)
        indt = np.any(indbs, axis=1)
        dphi = deltaphi*(deg + indbs[:, deg:-deg].sum(axis=1))

    else:
        # 1d spectra
        if focus is False:
            fract = ind.sum(axis=-1) / ind.shape[1]
            indt = fract > valid_fraction
        else:
            fract = np.sum(indall, axis=1) / lambok.sum(axis=0)[None, :]
            indt = np.all(fract > valid_fraction, axis=1)

    # Optional debug
    if focus is not False and False:
        indt_debug, ifocus = 40, 1
        if data2d is True:
            indall2 = indall.astype(int)
            indall2[:, lambok] = 1
            indall2[ind[..., None] & lambok[None, ...]] = 2
            plt.figure()
            plt.imshow(indall2[indt_debug, :, :, ifocus].T, origin='lower')
        else:
            plt.figure()
            plt.plot(lamb[~indall[indt_debug, :, ifocus]],
                     data[indt_debug, ~indall[indt_debug, :, ifocus]], '.k',
                     lamb[indall[indt_debug, :, ifocus]],
                     data[indt_debug, indall[indt_debug, :, ifocus]], '.r')
            plt.axvline(focus[ifocus, 0], ls='--', c='k')

    if not np.any(indt):
        msg = (
            "\nThere is no valid time step with the provided constraints:\n"
            + "\t- valid_nsigma = {}\n".format(valid_nsigma)
            + "\t- valid_fraction = {}\n".format(valid_fraction)
            + "\t- focus = {}\n".format(focus)
            + "\t- fract = {}\n".format(fract)
        )
        raise Exception(msg)

    # return
    dvalid = {
        'indt': indt, 'dphi': dphi, 'indbs': indbs, 'ind': ind,
        'focus': focus, 'valid_fraction': valid_fraction,
        'valid_nsigma': valid_nsigma,
    }
    if return_fract is True:
        dvalid['fract'] = fract
    return dvalid


###########################################################
###########################################################
#
#           dlines dict (lines vs domain)
#
###########################################################
###########################################################


def _checkformat_dlines(dlines=None, domain=None):
    if dlines is None:
        dlines = False

    if not isinstance(dlines, dict):
        msg = "Arg dlines must be a dict!"
        raise Exception(msg)

    lc = [
        (k0, type(v0)) for k0, v0 in dlines.items()
        if not (
            isinstance(k0, str)
            and isinstance(v0, dict)
            and 'lambda0' in v0.keys()
            and (
                type(v0['lambda0']) in _LTYPES
                or (
                    isinstance(v0['lambda0'], np.ndarray)
                    and v0['lambda0'].size == 1
                )
            )
        )
    ]
    if len(lc) > 0:
        lc = ["\t- {}: {}".format(*cc) for cc in lc]
        msg = (
            "Arg dlines must be a dict of the form:\n"
            + "\t{'line0': {'lambda0': float},\n"
            + "\t 'line1': {'lambda0': float},\n"
            + "\t  ...\n"
            + "\t 'lineN': {'lambda0': float}}\n"
            + "  You provided:\n{}".format('\n'.join(lc))
        )
        raise Exception(msg)

    # Select relevant lines (keys, lamb)
    lines_keys = np.array([k0 for k0 in dlines.keys()])
    lines_lamb = np.array([float(dlines[k0]['lambda0']) for k0 in lines_keys])
    if domain not in [None, False]:
        ind = (
            (lines_lamb >= domain['lamb']['minmax'][0])
            & (lines_lamb <= domain['lamb']['minmax'][1])
        )
        lines_keys = lines_keys[ind]
        lines_lamb = lines_lamb[ind]
    inds = np.argsort(lines_lamb)
    lines_keys, lines_lamb = lines_keys[inds], lines_lamb[inds]
    nlines = lines_lamb.size
    dlines = {k0: dict(dlines[k0]) for k0 in lines_keys}
    return dlines, lines_keys, lines_lamb


###########################################################
###########################################################
#
#           dinput dict (lines + spectral constraints)
#
###########################################################
###########################################################


def fit1d_dinput(
    dlines=None, dconstraints=None, dconstants=None, dprepare=None,
    data=None, lamb=None, mask=None,
    domain=None, pos=None, subset=None,
    same_spectrum=None, nspect=None, same_spectrum_dlamb=None,
    focus=None, valid_fraction=None, valid_nsigma=None, focus_half_width=None,
    valid_return_fract=None,
    dscales=None, dx0=None, dbounds=None,
    defconst=_DCONSTRAINTS,
):
    """ Check and format a dict of inputs to be fed to fit1d()

    This dict will contain all information relevant for solving the fit:
        - dlines: dict of lines (with 'lambda0': wavelength at rest)
        - lamb: vector of wavelength of the experimental spectrum
        - data: experimental spectrum, possibly 2d (time-varying)
        - dconstraints: dict of constraints on lines (amp, width, shift)
        - pos: bool, consider only positive data (False => replace <0 with nan)
        - domain:
        - mask:
        - subset:
        - same_spectrum:
        - focus:

    """

    # ------------------------
    # Check / format dprepare
    # ------------------------
    if dprepare is None:
        dprepare = multigausfit1d_from_dlines_prepare(
            data=data, lamb=lamb,
            mask=mask, domain=domain,
            pos=pos, subset=subset,
        )

    # ------------------------
    # Check / format dlines
    # ------------------------
    dlines, lines_keys, lines_lamb = _checkformat_dlines(
        dlines=dlines,
        domain=dprepare['domain'],
    )
    nlines = lines_lamb.size

    # Check same_spectrum
    if same_spectrum is None:
        same_spectrum = _SAME_SPECTRUM
    if same_spectrum is True:
        if type(nspect) not in [int, np.int]:
            msg = "Please provide nspect if same_spectrum = True"
            raise Exception(msg)
        if same_spectrum_dlamb is None:
            same_spectrum_dlamb = min(
                2*np.diff(dprepare['domain']['lamb']['minmax']),
                dprepare['domain']['lamb']['minmax'][0],
            )

    # ------------------------
    # Check / format dconstraints
    # ------------------------

    dconstraints = _checkformat_dconstraints(
        dconstraints=dconstraints, defconst=defconst,
    )
    dinput = {}

    # ------------------------
    # Check / format double
    # ------------------------
    _dconstraints_double(dinput, dconstraints, defconst=defconst)

    # ------------------------
    # Check / format width, shift, amp (groups with possible ratio)
    # ------------------------
    for k0 in ['amp', 'width', 'shift']:
        dinput[k0] = _width_shift_amp(
            dconstraints.get(k0, defconst[k0]),
            dconstants=dconstants,
            keys=lines_keys, nlines=nlines,
            dlines=dlines, k0=k0,
        )

    # ------------------------
    # add mz, symb, ION, keys, lamb
    # ------------------------
    mz = np.array([dlines[k0].get('m', np.nan) for k0 in lines_keys])
    symb = np.array([dlines[k0].get('symbol', k0) for k0 in lines_keys])
    ion = np.array([dlines[k0].get('ion', '?') for k0 in lines_keys])

    # ------------------------
    # same_spectrum
    # ------------------------
    if same_spectrum is True:
        keysadd = np.array([[kk+'_bis{:04.0f}'.format(ii) for kk in keys]
                            for ii in range(1, nspect)]).ravel()
        lines_lamb = (
            same_spectrum_dlamb*np.arange(0, nspect)[:, None]
            + lines_lamb[None, :]
        )
        keys = np.r_[keys, keysadd]

        for k0 in _DORDER:
            # Add other lines to original group
            keyk = dinput[k0]['keys']
            offset = np.tile(dinput[k0]['offset'], nspect)
            if k0 == 'shift':
                ind = np.tile(dinput[k0]['ind'], (1, nspect))
                coefs = (
                    dinput[k0]['coefs']
                    * lines_lamb[0, :] / lines_lamb
                ).ravel()
            else:
                coefs = np.tile(dinput[k0]['coefs'], nspect)
                keysadd = np.array([
                    [kk+'_bis{:04.0f}'.format(ii) for kk in keyk]
                    for ii in range(1, nspect)
                ]).ravel()
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
        lines_lamb = lines_lamb.ravel()

        # update mz, symb, ion
        mz = np.tile(mz, nspect)
        symb = np.tile(symb, nspect)
        ion = np.tile(ion, nspect)

    # ------------------------
    # add lines and properties
    # ------------------------
    dinput['keys'] = lines_keys
    dinput['lines'] = lines_lamb
    dinput['nlines'] = nlines

    dinput['mz'] = mz
    dinput['symb'] = symb
    dinput['ion'] = ion

    dinput['same_spectrum'] = same_spectrum
    if same_spectrum is True:
        dinput['same_spectrum_nspect'] = nspect
        dinput['same_spectrum_dlamb'] = same_spectrum_dlamb
    else:
        dinput['same_spectrum_nspect'] = False
        dinput['same_spectrum_dlamb'] = False

    # ------------------------
    # S/N threshold indices
    # ------------------------
    dinput['valid'] = fit12d_dvalid(
        data=dprepare['data'],
        lamb=dprepare['lamb'],
        indok=dprepare['indok'],
        valid_nsigma=valid_nsigma,
        valid_fraction=valid_fraction,
        focus=focus, focus_half_width=focus_half_width,
        lines_keys=lines_keys, lines_lamb=lines_lamb,
        return_fract=valid_return_fract,
    )

    # Update with dprepare
    dinput['dprepare'] = dict(dprepare)

    # Add dind
    dinput['dind'] = multigausfit1d_from_dlines_ind(dinput)

    # Add dscales, dx0 and dbounds
    dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    dinput['dconstants'] = fit12d_dconstants(
        dconstants=dconstants, dinput=dinput,
    )

    return dinput


def fit2d_dinput(
    dlines=None, dconstraints=None, dconstants=None, dprepare=None,
    deg=None, nbsplines=None, knots=None,
    data=None, lamb=None, phi=None, mask=None,
    domain=None, pos=None, subset=None, binning=None, cent_fraction=None,
    focus=None, valid_fraction=None, valid_nsigma=None, focus_half_width=None,
    valid_return_fract=None,
    dscales=None, dx0=None, dbounds=None,
    nxi=None, nxj=None,
    lphi=None, lphi_tol=None,
    defconst=_DCONSTRAINTS,
):
    """ Check and format a dict of inputs to be fed to fit2d()

    This dict will contain all information relevant for solving the fit:
        - dlines: dict of lines (with 'lambda0': wavelength at rest)
        - lamb: vector of wavelength of the experimental spectrum
        - data: experimental spectrum, possibly 2d (time-varying)
        - dconstraints: dict of constraints on lines (amp, width, shift)
        - pos: bool, consider only positive data (False => replace <0 with nan)
        - domain:
        - mask:
        - subset:
        - same_spectrum:
        - focus:

    """

    # ------------------------
    # Check / format dprepare
    # ------------------------
    if dprepare is None:
        dprepare = multigausfit2d_from_dlines_prepare(
            data=data, lamb=lamb, phi=phi,
            mask=mask, domain=domain,
            pos=pos, subset=subset, binning=binning,
            nbsplines=nbsplines, deg=deg,
            nxi=nxi, nxj=nxj,
            lphi=None, lphi_tol=None,
        )

    # ------------------------
    # Check / format dlines
    # ------------------------
    dlines, lines_keys, lines_lamb = _checkformat_dlines(
        dlines=dlines,
        domain=dprepare['domain'],
    )
    nlines = lines_lamb.size

    # ------------------------
    # Check / format dconstraints
    # ------------------------

    dconstraints = _checkformat_dconstraints(
        dconstraints=dconstraints, defconst=defconst,
    )
    dinput = {}

    # ------------------------
    # Check / format symmetry
    # ------------------------
    _dconstraints_symmetry(
        dinput, dprepare=dprepare, symmetry=dconstraints.get('symmetry'),
        cent_fraction=cent_fraction, defconst=defconst,
    )

    # ------------------------
    # Check / format double (spectral line doubling)
    # ------------------------
    _dconstraints_double(dinput, dconstraints, defconst=defconst)

    # ------------------------
    # Check / format width, shift, amp (groups with posssible ratio)
    # ------------------------
    for k0 in ['amp', 'width', 'shift']:
        dinput[k0] = _width_shift_amp(
            dconstraints.get(k0, defconst[k0]),
            dconstants=dconstants,
            keys=lines_keys, nlines=nlines,
            dlines=dlines, k0=k0,
        )

    # ------------------------
    # add mz, symb, ION, keys, lamb
    # ------------------------
    mz = np.array([dlines[k0].get('m', np.nan) for k0 in lines_keys])
    symb = np.array([dlines[k0].get('symbol', k0) for k0 in lines_keys])
    ion = np.array([dlines[k0].get('ION', '?') for k0 in lines_keys])

    # ------------------------
    # add lines and properties
    # ------------------------
    dinput['keys'] = lines_keys
    dinput['lines'] = lines_lamb
    dinput['nlines'] = nlines

    dinput['mz'] = mz
    dinput['symb'] = symb
    dinput['ion'] = ion

    # ------------------------
    # Get dict of bsplines
    # ------------------------
    dinput.update(multigausfit2d_from_dlines_dbsplines(
        knots=knots, deg=deg, nbsplines=nbsplines,
        phimin=dprepare['domain']['phi']['minmax'][0],
        phimax=dprepare['domain']['phi']['minmax'][1],
        symmetryaxis=dinput.get('symmetry_axis')
    ))

    # ------------------------
    # S/N threshold indices
    # ------------------------
    dinput['valid'] = fit12d_dvalid(
        data=dprepare['data'],
        lamb=dprepare['lamb'],
        phi=dprepare['phi'],
        binning=dprepare['binning'],
        indok=dprepare['indok'],
        valid_nsigma=valid_nsigma,
        valid_fraction=valid_fraction,
        focus=focus, focus_half_width=focus_half_width,
        lines_keys=lines_keys, lines_lamb=lines_lamb,
        nbs=dinput['nbs'],
        deg=dinput['deg'],
        knots_mult=dinput['knots_mult'],
        nknotsperbs=dinput['nknotsperbs'],
        return_fract=valid_return_fract,
    )

    # Update with dprepare
    dinput['dprepare'] = dict(dprepare)

    # Add dind
    dinput['dind'] = multigausfit2d_from_dlines_ind(dinput)

    # Add dscales, dx0 and dbounds
    dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    dinput['dconstants'] = fit12d_dconstants(
        dconstants=dconstants, dinput=dinput,
    )
    return dinput


###########################################################
###########################################################
#
#           dind dict (indices storing for fast access)
#
###########################################################
###########################################################


def multigausfit1d_from_dlines_ind(dinput=None):
    """ Return the indices of quantities in x to compute y """

    # indices
    # General shape: [bck, amp, widths, shifts]
    # If double [..., double_shift, double_ratio]
    # Except for bck, all indices should render nlines (2*nlines if double)
    dind = {
        'bck_amp': {'x': np.r_[0]},
        'bck_rate': {'x': np.r_[1]},
        'dshift': None,
        'dratio': None,
    }
    nn = dind['bck_amp']['x'].size + dind['bck_rate']['x'].size
    inddratio, inddshift = None, None
    for k0 in _DORDER:
        ind = dinput[k0]['ind']
        lnl = np.sum(ind, axis=1).astype(int)
        dind[k0] = {
            'x': nn + np.arange(0, ind.shape[0]),
            'lines': nn + np.argmax(ind, axis=0),
            'jac': [
                tuple(ind[ii, :].nonzero()[0]) for ii in range(ind.shape[0])
            ]
        }
        nn += dind[k0]['x'].size

    sizex = dind['shift']['x'][-1] + 1
    indx = np.r_[
        dind['bck_amp']['x'],
        dind['bck_rate']['x'],
        dind['amp']['x'],
        dind['width']['x'],
        dind['shift']['x'],
    ]
    assert np.all(np.arange(0, sizex) == indx)

    # check if double
    if dinput['double'] is True:
        dind['dshift'] = {'x': -2}
        dind['dratio'] = {'x': -1}
        sizex += 2
    elif isinstance(dinput['double'], dict):
        if dinput['double'].get('dshift') is None:
            dind['dshift'] = {'x': -1}
            sizex += 1
        elif dinput['double'].get('dratio') is None:
            dind['dratio'] = {'x': -1}
            sizex += 1

    dind['sizex'] = sizex
    dind['nbck'] = 2
    # dind['shapey1'] = dind['bck']['x'].size + dinput['nlines']

    # Ref line for amp (for dscales)
    amp_x0 = np.zeros((dinput['amp']['ind'].shape[0],), dtype=int)
    for ii in range(dinput['amp']['ind'].shape[0]):
        indi = dinput['amp']['ind'][ii, :].nonzero()[0]
        amp_x0[ii] = indi[np.argmin(np.abs(dinput['amp']['coefs'][indi]-1.))]
    dind['amp_x0'] = amp_x0
    return dind


def multigausfit2d_from_dlines_ind(dinput=None):
    """ Return the indices of quantities in x to compute y """

    # indices
    # General shape: [bck, amp, widths, shifts]
    # If double [..., double_shift, double_ratio]
    # Except for bck, all indices should render nlines (2*nlines if double)
    nbs = dinput['nbs']
    dind = {
        'bck_amp': {'x': np.r_[0]},
        'bck_rate': {'x': np.r_[1]},
        'dshift': None,
        'dratio': None,
    }
    nn = dind['bck_amp']['x'].size + dind['bck_rate']['x'].size
    inddratio, inddshift = None, None
    for k0 in _DORDER:
        # l0bs0, l0bs1, ..., l0bsN, l1bs0, ...., lnbsN
        ind = dinput[k0]['ind']
        lnl = np.sum(ind, axis=1).astype(int)
        dind[k0] = {
            'x': (
                nn
                + nbs*np.arange(0, ind.shape[0])[None, :]
                + np.arange(0, nbs)[:, None]
            ),
            'lines': (
                nn
                + nbs*np.argmax(ind, axis=0)[None, :]
                + np.arange(0, nbs)[:, None]
            ),
            # TBF / TBC !!!
            'jac': [ind[ii, :].nonzero()[0] for ii in range(ind.shape[0])],
        }
        nn += dind[k0]['x'].size

    sizex = dind['shift']['x'][-1, -1] + 1
    indx = np.r_[
        dind['bck_amp']['x'],
        dind['bck_rate']['x'],
        dind['amp']['x'].T.ravel(),
        dind['width']['x'].T.ravel(),
        dind['shift']['x'].T.ravel(),
    ]
    assert np.allclose(np.arange(0, sizex), indx)

    # check if double
    if dinput['double'] is True:
        dind['dshift'] = {'x': -2}
        dind['dratio'] = {'x': -1}
        sizex += 2
    elif isinstance(dinput['double'], dict):
        if dinput['double'].get('dshift') is None:
            dind['dshift'] = {'x': -1}
            sizex += 1
        elif dinput['double'].get('dratio') is None:
            dind['dratio'] = {'x': -1}
            sizex += 1

    dind['sizex'] = sizex
    dind['nbck'] = 2

    # Ref line for amp (for x0)
    # TBC !!!
    amp_x0 = np.zeros((dinput['amp']['ind'].shape[0],), dtype=int)
    for ii in range(dinput['amp']['ind'].shape[0]):
        indi = dinput['amp']['ind'][ii, :].nonzero()[0]
        amp_x0[ii] = indi[np.argmin(np.abs(dinput['amp']['coefs'][indi]-1.))]
    dind['amp_x0'] = amp_x0

    # Make bsplines selections easy
    # if dinput['valid']['dphi'] is not False:
    # dind['bs']['x'] =
    # import pdb; pdb.set_trace()     # DB
    # pass

    return dind


###########################################################
###########################################################
#
#    Common checks and format for scales, x0, bounds
#
###########################################################
###########################################################


def _fit12d_checkformat_dscalesx0(
    din=None, dinput=None,
    name=None, is2d=False,
):
    lkconst = ['dratio', 'dshift']
    lk = ['bck_amp', 'bck_rate']
    lkdict = _DORDER
    if din is None:
        din = {}
    lkfalse = [
        k0 for k0, v0 in din.items()
        if not (
            isinstance(din, dict)
            and (
                (k0 in lkconst and type(v0) in _LTYPES)
                or (k0 in lk and type(v0) in _LTYPES + [np.ndarray])
                or (
                    k0 in lkdict
                    and type(v0) in _LTYPES + [np.ndarray]
                    or (
                        isinstance(v0, dict)
                        and all([
                            k1 in dinput[k0]['keys']
                            and type(v1) in _LTYPES + [np.ndarray]
                            for k1, v1 in v0.items()
                        ])
                    )
                )
            )
        )
    ]

    if len(lkfalse) > 0:
        msg = (
            "Arg {} must be a dict of the form:\n".format(name)
            + "\t- {}\n".format({
                kk: 'float' if kk in lkconst+lk
                else {k1: 'float' for k1 in dinput[kk]['keys']}
                for kk in lkfalse
            })
            + "\t- provided: {}".format({
                kk: din[kk] for kk in lkfalse
            })
        )
        raise Exception(msg)

    return {
        k0: dict(v0) if isinstance(v0, dict) else v0
        for k0, v0 in din.items()
    }


def _fit12d_filldef_dscalesx0_dict(
    din=None, din_name=None,
    key=None, vref=None,
    nspect=None, dinput=None,
):

    # Check vref
    if vref is not None:
        if type(vref) not in _LTYPES and len(vref) not in [1, nspect]:
            msg = (
                "Non-conform vref for "
                + "{}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(vref)
            )
            raise Exception(msg)
        if type(vref) in _LTYPES:
            vref = np.full((nspect,), vref)
        elif len(vref) == 1:
            vref = np.full((nspect,), vref[0])

    # check din[key]
    if din.get(key) is None:
        assert vref is not None
        din[key] = {k0: vref for k0 in dinput[key]['keys']}

    elif not isinstance(din[key], dict):
        assert type(din[key]) in _LTYPES + [np.ndarray]
        if hasattr(din[key], '__len__') and len(din[key]) == 1:
            din[key] = din[key][0]
        if type(din[key]) in _LTYPES:
            din[key] = {
                k0: np.full((nspect,), din[key])
                for k0 in dinput[key]['keys']
            }
        elif din[key].shape == (nspect,):
            din[key] = {k0: din[key] for k0 in dinput[key]['keys']}
        else:
            msg = (
                "{}['{}'] not conform!".format(dd_name, key)
            )
            raise Exception(msg)

    else:
        for k0 in dinput[key]['keys']:
            if din[key].get(k0) is None:
                din[key][k0] = vref
            elif type(din[key][k0]) in _LTYPES:
                din[key][k0] = np.full((nspect,), din[key][k0])
            elif len(din[key][k0]) == 1:
                din[key][k0] = np.full((nspect,), din[key][k0][0])
            elif din[key][k0].shape != (nspect,):
                msg = (
                    "Non-conform value for "
                    + "{}['{}']['{}']\n".format(din_name, key, k0)
                    + "\t- expected: float or array (size {})\n".format(nspect)
                    + "\t- provided: {}".format(din[key][k0])
                )
                raise Exception(msg)
    return din


def _fit12d_filldef_dscalesx0_float(
    din=None, din_name=None,
    key=None, vref=None,
    nspect=None,
):
    if din.get(key) is None:
        if type(vref) in _LTYPES:
            din[key] = np.full((nspect,), vref)
        elif np.array(vref).shape == (1,):
            din[key] = np.full((nspect,), vref[0])
        elif np.array(vref).shape == (nspect,):
            din[key] = np.array(vref)
        else:
            msg = (
                "Non-conform vref for {}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(vref)
            )
            raise Exception(msg)
    else:
        if type(din[key]) in _LTYPES:
            din[key] = np.full((nspect,), din[key])
        elif din[key].shape == (1,):
            din[key] = np.full((nspect,), din[key][0])
        elif din[key].shape != (nspect,):
            msg = (
                "Non-conform vref for {}['{}']\n".format(din_name, key)
                + "\t- expected: float or array (size {})\n".format(nspect)
                + "\t- provided: {}".format(din[key])
            )
            raise Exception(msg)
    return din


###########################################################
###########################################################
#
#           scales (for variables scaling)
#
###########################################################
###########################################################


def fit12d_dscales(dscales=None, dinput=None):

    # --------------
    # Input checks
    dscales = _fit12d_checkformat_dscalesx0(
        din=dscales, dinput=dinput, name='dscales',
    )

    data = dinput['dprepare']['data']
    lamb = dinput['dprepare']['lamb']
    nspect = data.shape[0]

    # --------------
    # 2d spectrum = 1d spectrum + vert. profile
    is2d = data.ndim == 3
    if is2d is True:
        data = dinput['dprepare']['datalamb1d']
        datavert = dinput['dprepare']['dataphi1d']
        lamb = dinput['dprepare']['lamb1d']
        phi = dinput['dprepare']['phi1d']
        indok = np.any(dinput['dprepare']['indok'], axis=1)

        # bsplines modulation of bck and amp, if relevant
        # fit bsplines on datavert (vertical profile)
        # to modulate scales (bck and amp)

        dscales['bs'] = np.full((nspect, dinput['nbs']), np.nan)
        if dinput['symmetry'] is True:
            for ii in dinput['valid']['indt'].nonzero()[0]:
                indnonan = (
                    (~np.isnan(datavert[ii, :]))
                    & (
                        np.abs(phi-dinput['symmetry_axis'][ii])
                        < dinput['knots'][-1]
                    )
                ).nonzero()[0]
                indnonan = indnonan[
                    np.unique(
                        np.abs(phi[indnonan]-dinput['symmetry_axis'][ii]),
                        return_index=True,
                    )[1]
                ]
                bs = scpinterp.LSQUnivariateSpline(
                    np.abs(phi[indnonan]-dinput['symmetry_axis'][ii]),
                    datavert[ii, indnonan],
                    dinput['knots'][1:-1],
                    k=dinput['deg'],
                    bbox=dinput['knots'][np.r_[0, -1]],
                    ext=0,
                )
                dscales['bs'][ii, :] = bs.get_coeffs()
        else:
            for ii in dinput['valid']['indt'].nonzero()[0]:
                indnonan = (
                    (~np.isnan(datavert[ii, :]))
                    & (dinput['knots'][0] <= phi)
                    & (phi <= dinput['knots'][-1])
                )
                try:
                    bs = scpinterp.LSQUnivariateSpline(
                        phi[indnonan],
                        datavert[ii, indnonan],
                        dinput['knots'][1:-1],
                        k=dinput['deg'],
                        bbox=dinput['knots'][np.r_[0, -1]],
                        ext=0,
                    )
                except Exception as err:
                    import pdb; pdb.set_trace()     # DB
                    pass
                dscales['bs'][ii, :] = bs.get_coeffs()
        # Normalize to avoid double-amplification when amp*bs
        corr = np.max(dscales['bs'][dinput['valid']['indt'], :], axis=1)
        dscales['bs'][dinput['valid']['indt'], :] /= corr[:, None]
    else:
        indok = dinput['dprepare']['indok']

    # --------------
    # Default values for filling missing fields
    Dlamb = np.diff(dinput['dprepare']['domain']['lamb']['minmax'])
    lambm = dinput['dprepare']['domain']['lamb']['minmax'][0]
    if not (np.isfinite(Dlamb)[0] and Dlamb > 0):
        msg = (
            "lamb min, max seems to be non-finite or non-positive!\n"
            + "\t- dinput['dprepare']['domain']['lamb']['minmax'] = {}".format(
                dinput['dprepare']['domain']['lamb']['minmax']
            )
        )
        raise Exception(msg)
    if lambm == 0:
        lambm = Dlamb / 100.

    # bck_amp
    bck_amp = dscales.get('bck_amp')
    bck_rate = dscales.get('bck_rate')
    if bck_amp is None or bck_rate is None:
        indbck = (data > np.nanmean(data, axis=1)[:, None]) | (~indok)
        bcky = np.array(np.ma.masked_where(indbck, data).mean(axis=1))
        bckstd = np.array(np.ma.masked_where(indbck, data).std(axis=1))

        # bck_rate
        if bck_rate is None:
            bck_rate = (
                np.log((bcky+bckstd)/bcky) / (lamb.max()-lamb.min())
            )
        if bck_amp is None:
            # Assuming bck = A*exp(rate*(lamb-lamb.min()))
            bck_amp = bcky

    dscales = _fit12d_filldef_dscalesx0_float(
        din=dscales, din_name='dscales', key='bck_amp',
        vref=bck_amp, nspect=nspect,
    )
    dscales = _fit12d_filldef_dscalesx0_float(
        din=dscales, din_name='dscales', key='bck_rate',
        vref=bck_rate, nspect=nspect,
    )

    # amp
    dscales['amp'] = dscales.get('amp', dict.fromkeys(dinput['amp']['keys']))
    for ii, ij in enumerate(dinput['dind']['amp_x0']):
        key = dinput['amp']['keys'][ii]
        if dscales['amp'].get(key) is None:
            conv = np.exp(-(lamb-dinput['lines'][ij])**2/(2*(Dlamb/20.)**2))
            # indi = (
            # indok
            # & (np.abs(lamb-dinput['lines'][ij]) < Dlamb/20.)[None, :]
            # )
            # dscales['amp'][key] = np.array(np.ma.masked_where(
            # ~indbck, data,
            # ).mean(axis=1))
            dscales['amp'][key] = np.nansum(data*conv, axis=1) / np.sum(conv)
        else:
            if type(dscales['amp'][key]) in _LTYPES:
                dscales['amp'][key] = np.full((nspect,), dscales['amp'][key])
            else:
                assert dscales['amp'][key].shape == (nspect,)

    # width
    if dinput.get('same_spectrum') is True:
        lambm2 = (
            lambm
            + dinput['same_spectrum_dlamb']
            * np.arange(0, dinput['same_spectrum_nspect'])
        )
        nw0 = iwx.size / dinput['same_spectrum_nspect']
        lambmw = np.repeat(lambm2, nw0)
        widthref = (Dlamb/(20*lambmw))**2
    else:
        widthref = (Dlamb/(20*lambm))**2

    dscales = _fit12d_filldef_dscalesx0_dict(
        din=dscales, din_name='dscales', key='width', vref=widthref,
        nspect=nspect, dinput=dinput,
    )

    # shift
    shiftref = Dlamb/(25*lambm)
    dscales = _fit12d_filldef_dscalesx0_dict(
        din=dscales, din_name='dscales', key='shift', vref=shiftref,
        nspect=nspect, dinput=dinput,
    )

    # Double
    if dinput['double'] is not False:
        dratio = 1.
        dshift = float(Dlamb/(40*lambm))
        if dinput['double'] is True:
            pass
        else:
            if dinput['double'].get('dratio') is not None:
                dratio = dinput['double']['dratio']
            if dinput['double'].get('dshift') is not None:
                dratio = dinput['double']['dshift']
        din = {'dratio': dratio, 'dshift': dshift}
        for k0 in din.keys():
            dscales = _fit12d_filldef_dscalesx0_float(
                din=dscales, din_name='dscales', key=k0,
                vref=din[k0], nspect=nspect,
            )

    return dscales


###########################################################
###########################################################
#
#           x0 (initial guess)
#
###########################################################
###########################################################


def fit12d_dx0(dx0=None, dinput=None):

    # --------------
    # Input checks
    dx0 = _fit12d_checkformat_dscalesx0(
        din=dx0, dinput=dinput, name='dx0',
        is2d=dinput['dprepare']['data'].ndim == 3,
    )

    nspect = dinput['dprepare']['data'].shape[0]

    # --------------
    # 2d spectrum = 1d spectrum + vert. profile
    data2d = dinput['dprepare']['data'].ndim == 3
    if data2d is True:
        dx0 = _fit12d_filldef_dscalesx0_float(
            din=dx0, din_name='dx0', key='bs',
            vref=_DX0['bs'], nspect=nspect,
        )

    # --------------
    # Default values for filling missing fields

    # bck
    dx0 = _fit12d_filldef_dscalesx0_float(
        din=dx0, din_name='dx0', key='bck_amp',
        vref=_DX0['bck_amp'], nspect=nspect,
    )
    dx0 = _fit12d_filldef_dscalesx0_float(
        din=dx0, din_name='dx0', key='bck_rate',
        vref=_DX0['bck_rate'], nspect=nspect,
    )

    # amp, width, shift
    for k0 in _DORDER:
        dx0 = _fit12d_filldef_dscalesx0_dict(
            din=dx0, din_name='dx0', key=k0, vref=_DX0[k0],
            nspect=nspect, dinput=dinput,
        )

    # Double
    if dinput['double'] is not False:
        dratio = _DX0['dratio']
        dshift = _DX0['dshift']
        if dinput['double'] is True:
            pass
        else:
            if dinput['double'].get('dratio') is not None:
                dratio = dinput['double']['dratio']
            if dinput['double'].get('dshift') is not None:
                dratio = dinput['double']['dshift']

        din = {'dratio': dratio, 'dshift': dshift}
        for k0 in din.keys():
            dx0 = _fit12d_filldef_dscalesx0_float(
                din=dx0, din_name='dx0', key=k0,
                vref=din[k0], nspect=nspect,
            )

    # -------------
    # check
    lmsg = []
    for k0, v0 in dx0.items():
        if isinstance(dx0[k0], np.ndarray):
            c0 = (
                np.any(dx0[k0] < dinput['dbounds']['min'][k0])
                or np.any(dx0[k0] > dinput['dbounds']['max'][k0])
            )
            if c0:
                lmsg.append("dx0['{}'] = {}  (bounds = ({}, {}))".format(
                    k0, dx0[k0],
                    dinput['dbounds']['min'][k0],
                    dinput['dbounds']['max'][k0],
                ))
        elif isinstance(dx0[k0], dict):
            for k1, v1 in dx0[k0].items():
                c0 = (
                    np.any(dx0[k0][k1] < dinput['dbounds']['min'][k0][k1])
                    or np.any(dx0[k0][k1] > dinput['dbounds']['max'][k0][k1])
                )
                if c0:
                    lmsg.append(
                        "dx0['{}']['{}'] = {}  (bounds = ({}, {}))".format(
                            k0, k1, dx0[k0][k1],
                            dinput['dbounds']['min'][k0][k1],
                            dinput['dbounds']['max'][k0][k1],
                        )
                    )
    if len(lmsg) > 0:
        msg = (
            "The following values for dx0 are out of bounds:\n"
            + "\n".join(["\t- {}".format(mm) for mm in lmsg])
        )
        raise Exception(msg)

    return dx0


###########################################################
###########################################################
#
#           bounds
#
###########################################################
###########################################################


def fit12d_dbounds(dbounds=None, dinput=None):

    # --------------
    # Input checks
    if dbounds is None:
        dbounds = {'min': {}, 'max': {}}
    c0 = (
        isinstance(dbounds, dict)
        and all([
            kk in ['min', 'max'] and isinstance(vv, dict)
            for kk, vv in dbounds.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dbounds must be a dict of te form:\n"
            + "\t{'min': {...}, 'max': {}}"
        )
        raise Exception(msg)

    dbounds['min'] = _fit12d_checkformat_dscalesx0(
        din=dbounds['min'], dinput=dinput, name="dbounds['min']",
    )
    dbounds['max'] = _fit12d_checkformat_dscalesx0(
        din=dbounds['max'], dinput=dinput, name="dbounds['max']",
    )

    nspect = dinput['dprepare']['data'].shape[0]

    # --------------
    # 2d spectrum = 1d spectrum + vert. profile
    data2d = dinput['dprepare']['data'].ndim == 3
    if data2d is True:
        dbounds['min'] = _fit12d_filldef_dscalesx0_float(
            din=dbounds['min'], din_name="dbounds['min']",
            key='bs', vref=_DBOUNDS['bs'][0], nspect=nspect,
        )
        dbounds['max'] = _fit12d_filldef_dscalesx0_float(
            din=dbounds['max'], din_name="dbounds['max']",
            key='bs', vref=_DBOUNDS['bs'][1], nspect=nspect,
        )

    # --------------
    # Default values for filling missing fields

    # bck
    dbounds['min'] = _fit12d_filldef_dscalesx0_float(
        din=dbounds['min'], din_name="dbounds['min']",
        key='bck_amp', vref=_DBOUNDS['bck_amp'][0], nspect=nspect,
    )
    dbounds['max'] = _fit12d_filldef_dscalesx0_float(
        din=dbounds['max'], din_name="dbounds['max']",
        key='bck_amp', vref=_DBOUNDS['bck_amp'][1], nspect=nspect,
    )
    dbounds['min'] = _fit12d_filldef_dscalesx0_float(
        din=dbounds['min'], din_name="dbounds['min']",
        key='bck_rate', vref=_DBOUNDS['bck_rate'][0], nspect=nspect,
    )
    dbounds['max'] = _fit12d_filldef_dscalesx0_float(
        din=dbounds['max'], din_name="dbounds['max']",
        key='bck_rate', vref=_DBOUNDS['bck_rate'][1], nspect=nspect,
    )

    for k0 in _DORDER:
        dbounds['min'] = _fit12d_filldef_dscalesx0_dict(
            din=dbounds['min'], din_name="dbounds['min']",
            key=k0, vref=_DBOUNDS[k0][0], nspect=nspect,
            dinput=dinput,
        )
        dbounds['max'] = _fit12d_filldef_dscalesx0_dict(
            din=dbounds['max'], din_name="dbounds['max']",
            key=k0, vref=_DBOUNDS[k0][1], nspect=nspect,
            dinput=dinput,
        )

    # Double
    if dinput['double'] is not False:
        for k0 in ['dratio', 'dshift']:
            dbounds['min'] = _fit12d_filldef_dscalesx0_float(
                din=dbounds['min'], din_name="dbounds['min']",
                key=k0, vref=_DBOUNDS[k0][0], nspect=nspect,
            )
            dbounds['max'] = _fit12d_filldef_dscalesx0_float(
                din=dbounds['max'], din_name="dbounds['max']",
                key=k0, vref=_DBOUNDS[k0][1], nspect=nspect,
            )
    return dbounds


###########################################################
###########################################################
#
#           constants
#
###########################################################
###########################################################


def fit12d_dconstants(dconstants=None, dinput=None):

    # --------------
    # Input checks
    dconstants = _fit12d_checkformat_dscalesx0(
        din=dconstants, dinput=dinput, name="dconstants",
    )
    nspect = dinput['dprepare']['data'].shape[0]

    # --------------
    # 2d spectrum = 1d spectrum + vert. profile
    data2d = dinput['dprepare']['data'].ndim == 3

    # --------------
    # Default values for filling missing fields

    # bck
    dconstants = _fit12d_filldef_dscalesx0_float(
        din=dconstants, din_name="dconstants",
        key='bck_amp', vref=np.nan, nspect=nspect,
    )
    dconstants = _fit12d_filldef_dscalesx0_float(
        din=dconstants, din_name="dconstants",
        key='bck_rate', vref=np.nan, nspect=nspect,
    )

    for k0 in _DORDER:
        dconstants = _fit12d_filldef_dscalesx0_dict(
            din=dconstants, din_name="dconstants",
            key=k0, vref=np.nan, nspect=nspect,
            dinput=dinput,
        )

    # Double
    if dinput['double'] is not False:
        for k0 in ['dratio', 'dshift']:
            dconstants = _fit12d_filldef_dscalesx0_float(
                din=dconstants, din_name="dconstants",
                key=k0, vref=np.nan, nspect=nspect,
            )
    return dconstants


###########################################################
###########################################################
#
#           dict to vector (scales, x0, bounds)
#
###########################################################
###########################################################


def _dict2vector_dscalesx0bounds(
    dd=None, dd_name=None,
    dinput=None,
):
    nspect = dinput['dprepare']['data'].shape[0]
    x = np.full((nspect, dinput['dind']['sizex']), np.nan)

    x[:, dinput['dind']['bck_amp']['x'][0]] = dd['bck_amp']
    x[:, dinput['dind']['bck_rate']['x'][0]] = dd['bck_rate']
    for k0 in _DORDER:
        for ii, k1 in enumerate(dinput[k0]['keys']):
            x[:, dinput['dind'][k0]['x'][ii]] = dd[k0][k1]

    if dinput['double'] is not False:
        if dinput['double'] is True:
            x[:, dinput['dind']['dratio']['x']] = dd['dratio']
            x[:, dinput['dind']['dshift']['x']] = dd['dshift']
        else:
            if dinput['double'].get('dratio') is None:
                x[:, dinput['dind']['dratio']['x']] = dd['dratio']
            if dinput['double'].get('dshift') is None:
                x[:, dinput['dind']['dshift']['x']] = dd['dshift']
    return x


###########################################################
###########################################################
#
#           Load dinput
#
###########################################################
###########################################################


def _rebuild_dict(dd):
    for k0, v0 in dd.items():
        if isinstance(v0, np.ndarray) and v0.shape == ():
            dd[k0] = v0.tolist()
        if isinstance(dd[k0], dict):
            _rebuild_dict(dd[k0])


def _checkformat_dinput(dinput, allow_pickle=True):
    if isinstance(dinput, str):
        if not (os.path.isfile(dinput) and dinput[-4:] == '.npz'):
            msg = ("Arg dinput must be aither a dict or "
                   + "the absolute path to a .npz\n"
                   + "  You provided: {}".format(dinput))
            raise Exception(msg)
        dinput = dict(np.load(dinput, allow_pickle=allow_pickle))

    if not isinstance(dinput, dict):
        msg = (
            "dinput must be a dict!\n"
            + "  You provided: {}".format(type(dinput))
        )

    _rebuild_dict(dinput)
    return dinput


###########################################################
###########################################################
#
#           Main fitting sub-routines
#
###########################################################
###########################################################


def _checkformat_options(chain, method, tr_solver, tr_options,
                         xtol, ftol, gtol, loss, max_nfev, verbose):
    if chain is None:
        chain = _CHAIN
    if method is None:
        method = _METHOD
    assert method in ['trf', 'dogbox'], method
    if tr_solver is None:
        tr_solver = None
    if tr_options is None:
        tr_options = {}
    if xtol is None:
        xtol = _TOL1D['x']
    if ftol is None:
        ftol = _TOL1D['f']
    if gtol is None:
        gtol = _TOL1D['g']
    if loss is None:
        loss = _LOSS
    if max_nfev is None:
        max_nfev = None
    if verbose is None:
        verbose = 1
    if verbose == 3:
        verbscp = 2
    else:
        verbscp = 0

    return (chain, method, tr_solver, tr_options,
            xtol, ftol, gtol, loss, max_nfev, verbose, verbscp)


def multigausfit1d_from_dlines(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, chain=None, verbose=None,
    loss=None, jac=None,
):
    """ Solve multi_gaussian fit in 1d from dlines

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

    # ---------------------------
    # Check format options
    (
        chain, method, tr_solver, tr_options,
        xtol, ftol, gtol, loss, max_nfev,
        verbose, verbscp,
    ) = _checkformat_options(
         chain, method, tr_solver, tr_options,
         xtol, ftol, gtol, loss, max_nfev, verbose,
    )

    # ---------------------------
    # Load dinput if necessary
    dinput = _checkformat_dinput(dinput)
    dprepare, dind = dinput['dprepare'], dinput['dind']
    nspect = dprepare['data'].shape[0]

    # ---------------------------
    # If same spectrum => consider a single data set
    if dinput['same_spectrum'] is True:
        lamb = (
            dinput['same_spectrum_dlamb']*np.arange(0, nspect)[:, None]
            + dprepare['lamb'][None, :]
        ).ravel()
        datacost = dprepare['data'].ravel()[None, :]
        nspect = data.shape[0]
        chain = False
    else:
        lamb = dprepare['lamb']
        datacost = dprepare['data']

    # ---------------------------
    # Get scaling, x0, bounds from dict
    scales = _dict2vector_dscalesx0bounds(
        dd=dinput['dscales'], dd_name='dscales', dinput=dinput,
    )
    x0 = _dict2vector_dscalesx0bounds(
        dd=dinput['dx0'], dd_name='dx0', dinput=dinput,
    )
    boundmin = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['min'], dd_name="dbounds['min']", dinput=dinput,
    )
    boundmax = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['max'], dd_name="dbounds['max']", dinput=dinput,
    )
    bounds = np.array([boundmin[0, :], boundmax[0, :]])

    # ---------------------------
    # Separate free from constant parameters
    const = _dict2vector_dscalesx0bounds(
        dd=dinput['dconstants'], dd_name='dconstants', dinput=dinput,
    )
    indx = np.any(np.isnan(const), axis=0)
    const = const[:, ~indx]
    x0[:, ~indx] = const / scales[:, ~indx]

    # ---------------------------
    # Get function, cost function and jacobian
    (
        func_detail, func_cost, func_jac,
    ) = _funccostjac.multigausfit1d_from_dlines_funccostjac(
        lamb, dinput=dinput, dind=dind, jac=jac, indx=indx,
    )

    # ---------------------------
    # Prepare output
    sol_x = np.full((nspect, dind['sizex']), np.nan)
    success = np.full((nspect,), np.nan)
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    validity = np.zeros((nspect,), dtype=int)
    message = ['' for ss in range(nspect)]
    errmsg = ['' for ss in range(nspect)]

    # Prepare msg
    if verbose in [1, 2]:
        col = np.char.array([
            'spect', 'time (s)', 'cost', 'nfev', 'njev', 'msg',
        ])
        maxl = max(np.max(np.char.str_len(col)), 10)
        msg = '\n'.join([' '.join([cc.ljust(maxl) for cc in col]),
                         ' '.join(['-'*maxl]*6)])
        print(msg)

    # ---------------------------
    # Main loop
    end = '\r'
    t0 = dtm.datetime.now()     # DB
    for ii in range(nspect):

        if verbose == 3:
            msg = "\nspect {} / {}".format(ii+1, nspect)
            print(msg)

        try:
            dti = None
            t0i = dtm.datetime.now()     # DB
            if not dinput['valid']['indt'][ii]:
                continue

            # optimization
            res = scpopt.least_squares(
                func_cost, x0[ii, indx],
                jac=func_jac, bounds=bounds[:, indx],
                method=method, ftol=ftol, xtol=xtol,
                gtol=gtol, x_scale=1.0, f_scale=1.0,
                loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options=tr_options,
                jac_sparsity=None, max_nfev=max_nfev,
                verbose=verbscp, args=(),
                kwargs={
                    'data': datacost[ii, :],
                    'scales': scales[ii, :],
                    'const': const[ii, :],
                    'indok': dprepare['indok'][ii, :],
                },
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True and ii < nspect-1:
                x0[ii+1, indx] = res.x

            # cost, message, time
            success[ii] = res.success
            cost[ii] = res.cost
            nfev[ii] = res.nfev
            message[ii] = res.message
            time[ii] = round(
                (dtm.datetime.now()-t0i).total_seconds(),
                ndigits=3,
            )
            sol_x[ii, indx] = res.x
            sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

        except Exception as err:
            errmsg[ii] = str(err)
            validity[ii] = -1

        # Verbose
        if verbose in [1, 2]:
            if validity[ii] == 0:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    '{:5.3e}'.format(res.cost),
                    str(res.nfev),
                    str(res.njev),
                    res.message,
                ])
            else:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    ' - ', ' - ', ' - ',
                    errmsg[ii],
                ])
            msg = ' '.join([cc.ljust(maxl) for cc in col])
            if verbose == 1:
                if ii == nspect-1:
                    end = '\n'
                print(msg, end=end, flush=True)
            else:
                print(msg, end='\n')

    # ---------------------------
    # Reshape in case of same_spectrum
    if dinput['same_spectrum'] is True:
        nspect0 = dinput['same_spectrum_nspect']

        def reshape_custom(aa, nspect0=nspect0):
            return aa.reshape((nspect0, int(aa.size/nspect0)))

        nlamb = int(lamb.size / nspect0)
        nlines = int((sol_detail.shape[1]-1)/nspect0)
        lamb = lamb[:nlamb]

        nxbis = int(
            dind['bck_amp']['x'].size
            + dind['bck_rate']['x'].size
            + (dind['amp']['x'].size + dind['width']['x'].size)/nspect0
            + dind['shift']['x'].size
        )
        if dinput['double'] is not False:
            if dinput['double'] is True:
                nxbis += 2
            else:
                nxbis += (
                    dinput['double'].get('dratio') is not None
                    + dinput['double'].get('dshift') is not None
                )
        nba = dind['bck_amp']['x'].size
        nbr = dind['bck_rate']['x'].size
        nb = nba+nbr
        na = int(dind['amp']['x'].size/nspect0)
        nw = int(dind['width']['x'].size/nspect0)
        ns = dind['shift']['x'].size
        x2 = np.full((nspect0, nxbis), np.nan)
        x2[:, :nba] = sol_x[0, dind['bck_amp']['x']][None, :]
        x2[:, nba:nbr] = sol_x[0, dind['bck_rate']['x']][None, :]
        x2[:, nb:nb+na] = reshape_custom(sol_x[0, dind['amp']['x']])
        x2[:, nb+na:nb+na+nw] = reshape_custom(sol_x[0, dind['width']['x']])
        x2[:, nb+na+nw:nb+na+nw+ns] = sol_x[:, dind['shift']['x']]
        if dinput['double'] is True:
            x2[:, dind['dratio']['x']] = sol_x[:, dind['dratio']['x']]
            x2[:, dind['dshift']['x']] = sol_x[:, dind['dshift']['x']]
        import pdb; pdb.set_trace()     # DB
        sol_x = x2

    # Isolate dratio and dshift
    dratio, dshift = None, None
    if dinput['double'] is not False:
        if dinput['double'] is True:
            dratio = (
                sol_x[:, dind['dratio']['x']] * scales[:, dind['dratio']['x']]
            )
            dshift = (
                sol_x[:, dind['dshift']['x']] * scales[:, dind['dshift']['x']]
            )
        else:
            if dinput['double'].get('dratio') is None:
                dratio = (
                    sol_x[:, dind['dratio']['x']]
                    * scales[:, dind['dratio']['x']]
                )
            else:
                dratio = np.full((nspect,), dinput['double']['dratio'])

            if dinput['double'].get('dshift') is None:
                dshift = (
                    sol_x[:, dind['dshift']['x']]
                    * scales[:, dind['dshift']['x']]
                )
            else:
                dshift = np.full((nspect,), dinput['double']['dshift'])

    if verbose > 0:
        dt = (dtm.datetime.now()-t0).total_seconds()
        msg = (
            "Total computation time:"
            + "\t{} s for {} spectra ({} s per spectrum)".format(
                round(dt, ndigits=3),
                nspect,
                round(dt/nspect, ndigits=3),
            )
        )
        print(msg)

    # ---------------------------
    # Format output as dict
    dfit = {
        'dinput': dinput,
        'scales': scales, 'x0': x0, 'bounds': bounds,
        'jac': jac, 'sol_x': sol_x,
        'dratio': dratio, 'dshift': dshift,
        'indx': indx,
        'time': time, 'success': success,
        'validity': validity, 'errmsg': np.array(errmsg),
        'cost': cost, 'nfev': nfev, 'msg': np.array(message),
    }
    return dfit


def multigausfit2d_from_dlines(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, chain=None, verbose=None,
    loss=None, jac=None,
):
    """ Solve multi_gaussian fit in 1d from dlines

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

    # ---------------------------
    # Check format options
    (
        chain, method, tr_solver, tr_options,
        xtol, ftol, gtol, loss, max_nfev,
        verbose, verbscp,
    ) = _checkformat_options(
         chain, method, tr_solver, tr_options,
         xtol, ftol, gtol, loss, max_nfev, verbose,
    )

    # ---------------------------
    # Load dinput if necessary
    dinput = _checkformat_dinput(dinput)
    dprepare, dind = dinput['dprepare'], dinput['dind']
    nspect = dprepare['data'].shape[0]

    # ---------------------------
    # DEPRECATED?
    lamb = dprepare['lamb']
    if dinput['symmetry'] is True:
        phi = np.abs(phi - np.nanmean(dinput['symmetry_axis']))
    else:
        phi = dprepare['phi']

    # ---------------------------
    # Get scaling, x0, bounds from dict
    scales = _dict2vector_dscalesx0bounds(
        dd=dinput['dscales'], dd_name='dscales', dinput=dinput,
    )
    x0 = _dict2vector_dscalesx0bounds(
        dd=dinput['dx0'], dd_name='dx0', dinput=dinput,
    )
    boundmin = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['min'], dd_name="dbounds['min']", dinput=dinput,
    )
    boundmax = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['max'], dd_name="dbounds['max']", dinput=dinput,
    )
    bounds = np.array([boundmin[0, :], boundmax[0, :]])

    # ---------------------------
    # Separate free from constant parameters
    const = _dict2vector_dscalesx0bounds(
        dd=dinput['dconstants'], dd_name='dconstants', dinput=dinput,
    )
    indx = np.any(np.isnan(const), axis=0)
    const = const[:, ~indx]
    x0[:, ~indx] = const / scales[:, ~indx]

    # ---------------------------
    # Get function, cost function and jacobian
    (
        func_detail, func_cost, func_jac,
    ) = _funccostjac.multigausfit2d_from_dlines_funccostjac(
         lamb, phi2,
         dinput=dinput, dind=dind, jac=jac, indx=indx,
    )

    # TBF after multigausfit2d_from_dlines_funccostjac() is checked

    # ---------------------------
    # Prepare output
    datacost = np.reshape(
        dprepare['data'][:, dprepare['indok']],
        (nspect, dprepare['indok'].sum()))
    sol_x = np.full((nspect, dind['sizex']), np.nan)
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
            indok_var = dprepare['indok_var'].reshape(
                (nspect, dprepare['lamb'].size))
        else:
            indok_var = [dprepare['indok_var'].ravel()]*nspect
    else:
        indok_var = [False]*nspect
    dprepare['indok_var'] = indok_var

    # Prepare msg
    if verbose in [1, 2]:
        col = np.char.array(['Spect', 'time (s)', 'cost',
                             'nfev', 'njev', 'msg'])
        maxl = max(np.max(np.char.str_len(col)), 10)
        msg = '\n'.join([' '.join([cc.ljust(maxl) for cc in col]),
                         ' '.join(['-'*maxl]*6)])
        print(msg)

    # ---------------------------
    # Minimize
    end = '\r'
    t0 = dtm.datetime.now()     # DB
    for ii in range(nspect):
        if verbose == 3:
            msg = "\nSpect {} / {}".format(ii+1, nspect)
            print(msg)
        try:
            t0i = dtm.datetime.now()     # DB
            if not dinput['valid']['indt'][ii]:
                continue
            res = scpopt.least_squares(
                func_cost, x0_scale[ii, :],
                jac=func_jac, bounds=bounds_scale,
                method=method, ftol=ftol, xtol=xtol,
                gtol=gtol, x_scale=1.0, f_scale=1.0,
                loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options=tr_options,
                jac_sparsity=None, max_nfev=max_nfev,
                verbose=verbscp, args=(),
                kwargs={'data': datacost[ii, :],
                        'scales': scales[ii, :],
                        'indok_var': indok_var[ii],
                        'ind_bs': dinput['valid']['indbs'][ii, :]})
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True and ii < nspect-1:
                x0_scale[ii+1, :] = res.x

            # cost, message, time
            success[ii] = res.success
            cost[ii] = res.cost
            nfev[ii] = res.nfev
            message[ii] = res.message
            time[ii] = round((dtm.datetime.now()-t0i).total_seconds(),
                             ndigits=3)
            sol_x[ii, :] = res.x

        except Exception as err:
            errmsg[ii] = str(err)
            validity[ii] = -1

        if verbose in [1, 2]:
            if validity[ii] == 0:
                col = np.char.array(['{} / {}'.format(ii+1, nspect),
                                     '{}'.format(dti),
                                     '{:5.3e}'.format(res.cost),
                                     str(res.nfev), str(res.njev),
                                     res.message])
            else:
                col = np.char.array(['{} / {}'.format(ii+1, nspect),
                                     '{}'.format(dti),
                                     ' - ', ' - ', ' - ',
                                     errmsg[ii]])
            msg = ' '.join([cc.ljust(maxl) for cc in col])
            if verbose == 1:
                if ii == nspect-1:
                    end = '\n'
                print(msg, end=end, flush=True)
            else:
                print(msg, end='\n')

    # Isolate dratio and dshift
    dratio, dshift = None, None
    if dinput['double'] is not False:
        if dinput['double'] is True:
            dratio = (sol_x[:, dind['dratio']['x']]
                      * scales[:, dind['dratio']['x']])
            dshift = (sol_x[:, dind['dshift']['x']]
                      * scales[:, dind['dshift']['x']])
        else:
            if dinput['double'].get('dratio') is None:
                dratio = (sol_x[:, dind['dratio']['x']]
                          * scales[:, dind['dratio']['x']])
            else:
                dratio = np.full((nspect,), dinput['double']['dratio'])
            if dinput['double'].get('dshift') is None:
                dshift = (sol_x[:, dind['dshift']['x']]
                          * scales[:, dind['dshift']['x']])
            else:
                dshift = np.full((nspect,), dinput['double']['dshift'])

    if verbose > 0:
        dt = (dtm.datetime.now()-t0).total_seconds()
        msg = ("Total computation time:"
               + "\t{} s for {} spectra ({} s per spectrum)".format(
                   round(dt, ndigits=3), nspect,
                   round(dt/nspect, ndigits=3)))
        print(msg)

    # ---------------------------
    # Format output as dict
    dfit = {'dinput': dinput,
            'scales': scales, 'x0_scale': x0_scale,
            'bounds_scale': bounds_scale, 'phi2': phi2,
            'jac': jac, 'sol_x': sol_x,
            'dratio': dratio, 'dshift': dshift,
            'time': time, 'success': success,
            'validity': validity, 'errmsg': np.array(errmsg),
            'cost': cost, 'nfev': nfev, 'msg': np.array(message)}
    return dfit


###########################################################
###########################################################
#
#   Main fit functions
#
###########################################################
###########################################################


def fit1d(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, loss=None, chain=None,
    dx0=None, x0_scale=None, bounds_scale=None,
    jac=None, verbose=None, showonly=None,
    save=None, name=None, path=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_lamb_total=None, pts_lamb_detail=None,
    plot=None, fs=None, wintit=None, tit=None, dmargin=None,
    return_dax=None,
):

    # ----------------------
    # Check / format
    if showonly is None:
        showonly = False
    if save is None:
        save = False
    if plot is None:
        plot = False
    if return_dax is None:
        return_dax = False

    # ----------------------
    # Get dinput for 1d fitting from dlines, dconstraints, dprepare...
    if not isinstance(dinput, dict):
        msg = ("Please provide a properly formatted dict of inputs!\n"
               + "fit1d() needs the problem to be given as a dinput dict\n"
               + "  => Use dinput = fit1d_dinput()")
        raise Exception(msg)

    # ----------------------
    # Perform 2d fitting
    if showonly is True:
        msg = "TBF: lambfit and spect1d not defined"
        raise Exception(msg)

        dfit1d = {'shift': np.zeros((1, dinput['nlines'])),
                  'coefs': np.zeros((1, dinput['nlines'])),
                  'lamb': lambfit,
                  'data': spect1d,
                  'double': False,
                  'Ti': False,
                  'vi': False,
                  'ratio': None}
    else:
        dfit1d = multigausfit1d_from_dlines(
            dinput=dinput,
            method=method, max_nfev=max_nfev,
            tr_solver=tr_solver, tr_options=tr_options,
            xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
            chain=chain, verbose=verbose, jac=jac)

    # ----------------------
    # Optional saving
    if save is True:
        if name is None:
            name = 'custom'
        name = 'TFS_fit1d_doutput_{}_nbs{}_{}_tol{}_{}.npz'.format(
            name, dinput['nbs'], dinput['method'], dinput['xtol'])
        if name[-4:] != '.npz':
            name = name + '.npz'
        if path is None:
            path = './'
        pfe = os.path.join(os.path.abspath(path), name)
        np.savez(pfe, **dfit2d)
        msg = ("Saved in:\n"
               + "\t{}".format(pfe))
        print(msg)

    # ----------------------
    # Optional plotting
    if plot is True:
        dout = fit1d_extract(
            dfit1d,
            amp=amp, coefs=coefs, ratio=ratio,
            Ti=Ti, width=width, vi=vi, shift=shift,
            pts_lamb_total=pts_lamb_total,
            pts_lamb_detail=pts_lamb_detail,
        )
        # TBF
        dax = _plot.plot_fit1d(
            dfit1d=dfit1d, dout=dout, showonly=showonly,
            fs=fs, dmargin=dmargin,
            tit=tit, wintit=wintit)

    # ----------------------
    # return
    if return_dax is True:
        return dfit1d, dax
    else:
        return dfit1d


# TBF
def fit2d(
    dinput=None, dprepare=None, dlines=None, dconstraints=None,
    lamb=None, phi=None, data=None, mask=None,
    domain=None, pos=None, subset=None, binning=None,
    deg=None, knots=None, nbsplines=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, loss=None, chain=None,
    dx0=None, x0_scale=None, bounds_scale=None,
    jac=None, nxi=None, nxj=None, verbose=None, showonly=None,
    save=None, name=None, path=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_lamb_total=None, pts_lamb_detail=None,
    plot=None, fs=None, wintit=None, tit=None, dmargin=None,
    return_dax=None,
):

    # ----------------------
    # Check / format
    if showonly is None:
        showonly = False
    if save is None:
        save = False
    if plot is None:
        plot = False
    if return_dax is None:
        return_dax = False

    # ----------------------
    # Get dinput for 2d fitting from dlines, dconstraints, dprepare...
    if dinput is None:
        dinput = fit2d_dinput(
            dlines=dlines, dconstraints=dconstraints, dprepare=dprepare,
            data=data, lamb=lamb, phi=phi,
            mask=mask, domain=domain,
            pos=pos, subset=subset, binning=binning,
            nxi=nxi, nxj=nxj, lphi=None, lphi_tol=None,
            deg=deg, knots=knots, nbsplines=nbsplines)

    # ----------------------
    # Perform 2d fitting
    if showonly is True:
        # TBF
        pass
    else:
        dfit2d = multigausfit2d_from_dlines(
            dinput=dinput, dx0=dx0,
            x0_scale=x0_scale, bounds_scale=bounds_scale,
            method=method, max_nfev=max_nfev,
            tr_solver=tr_solver, tr_options=tr_options,
            xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
            chain=chain, verbose=verbose, jac=jac)

    # ----------------------
    # Optional saving
    if save is True:
        if name is None:
            name = 'custom'
        name = 'TFS_fit2d_doutput_{}_nbs{}_{}_tol{}_{}.npz'.format(
            name, dinput['nbs'], dinput['method'], dinput['xtol'])
        if name[-4:] != '.npz':
            name = name + '.npz'
        if path is None:
            path = './'
        pfe = os.path.join(os.path.abspath(path), name)
        np.savez(pfe, **dfit2d)
        msg = ("Saved in:\n"
               + "\t{}".format(pfe))
        print(msg)

    # ----------------------
    # Optional plotting
    if plot is True:
        dout = fit2d_extract(dfit2d)
        dax = None

    # ----------------------
    # return
    if return_dax is True:
        return dfit2d, dax
    else:
        return dfit2d


###########################################################
###########################################################
#
#   Extract data from pre-computed dict of fitted results
#
###########################################################
###########################################################


def fit12d_get_data_checkformat(
    dfit=None,
    pts_phi=None, npts_phi=None,
    bck=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_total=None, pts_detail=None,
    allow_pickle=None,
):

    # load file if str
    if isinstance(dfit, str):
        if not os.path.isfile(dfit) or not dfit[-4:] == '.npz':
            msg = ("Provided dfit must be either a dict or "
                   + "the absolute path to a saved .npz\n"
                   + "  You provided: {}".format(dfit))
            raise Exception(msg)
        if allow_pickle is None:
            allow_pickle = _ALLOW_PICKLE
        dfit = dict(np.load(dfit, allow_pickle=allow_pickle))
        _rebuild_dict(dfit)

    # check dfit basic structure
    lk = ['dprepare', 'dinput', 'dind', 'sol_x', 'jac', 'scales']
    c0 = isinstance(dfit, dict) and all([ss in dfit.keys() for ss in lk])
    if not isinstance(dfit, dict):
        msg = ("\ndfit must be a dict with at least the following keys:\n"
               + "\t- {}\n".format(lk)
               + "\t- provided: {}".format(dfit))
        raise Exception(msg)

    # Identify if fit1d or fit2d
    is2d = 'nbsplines' in dfit['dinput'].keys()
    if is2d is True and 'phi2' not in dfit.keys():
        msg = "dfit is a fit2d output but does not have key 'phi2'!"
        raise Exception(msg)

    # Extract dinput and dprepare (more readable)
    dinput = dfit['dinput']
    dprepare = dfit['dinput']['dprepare']

    # ratio
    if ratio is None:
        ratio = False
    if ratio is not False:
        coefs = True

    # Check / format amp, Ti, vi
    d3 = {
        'bck_amp': [bck, 'bck_amp'],
        'bck_rate': [bck, 'bck_rate'],
        'amp': [amp, 'amp'],
        'coefs': [coefs, 'amp'],
        'Ti': [Ti, 'width'],
        'width': [width, 'width'],
        'vi': [vi, 'shift'],
        'shift': [shift, 'shift'],
    }
    # amp, Ti, vi
    for k0 in d3.keys():
        if d3[k0][0] is None:
            d3[k0][0] = True
        if d3[k0][0] is True:
            d3[k0][0] = _D3[k0]
        if d3[k0][0] is False:
            d3[k0] = d3[k0][0]
            continue
        if 'bck' in k0:
            continue
        lc = [
            d3[k0][0] in ['lines', 'x'],
            isinstance(d3[k0][0], str),
            (
                isinstance(d3[k0][0], list)
                and all([isinstance(isinstance(ss, str) for ss in d3[k0][0])])
            )
        ]
        if not any(lc):
            msg = (
                "\nArg {} must be either:\n".format(k0)
                + "\t- 'x': return all unique {}\n".format(k0)
                + "\t- 'lines': return {} for all lines (inc. duplicates)\n"
                + "\t- str: a key in:\n"
                + "\t\t{}\n".format(dinput['keys'])
                + "\t\t{}\n".format(dinput[d3[k0][1]]['keys'])
                + "\t- list: a list of keys (see above)\n"
                + "Provided: {}".format(d3[k0][0])
            )
            raise Exception(msg)

        if lc[0]:
            if d3[k0][0] == 'lines':
                d3[k0][0] = {
                    'type': d3[k0][0],
                    'ind': np.arange(0, dinput['nlines']),
                }
            else:
                d3[k0][0] = {
                    'type': d3[k0][0],
                    'ind': np.arange(0, dinput[d3[k0][1]]['keys'].size),
                }
        elif lc[1]:
            d3[k0][0] = [d3[k0][0]]

        if isinstance(d3[k0][0], list):
            lc = [
                all([ss in dinput['keys'] for ss in d3[k0][0]]),
                all([ss in dinput[d3[k0][1]]['keys'] for ss in d3[k0][0]]),
            ]
            if not any(lc):
                msg = (
                    "\nArg must contain either keys from:\n"
                    + "\t- lines keys: {}\n".format(dinput['keys'])
                    + "\t- {} keys: {}".format(k0, dinput[d3[k0][1]]['keys']),
                )
                raise Exception(msg)
            if lc[0]:
                d3[k0][0] = {
                    'type': 'lines',
                    'ind': np.array(
                        [
                            (dinput['keys'] == ss).nonzero()[0][0]
                            for ss in d3[k0][0]
                        ],
                        dtype=int,
                    )
                }
            else:
                d3[k0][0] = {
                    'type': 'x',
                    'ind': np.array(
                        [
                            (dinput[d3[k0][1]]['keys'] == ss).nonzero()[0][0]
                            for ss in d3[k0][0]
                        ],
                        dtype=int),
                }
        d3[k0][0]['field'] = d3[k0][1]
        d3[k0] = d3[k0][0]

    # Ratio
    if ratio is not False:
        lkeys = dfit['dinput']['keys']
        lc = [
            isinstance(ratio, tuple),
            isinstance(ratio, list),
            isinstance(ratio, np.ndarray),
        ]
        msg = (
            "\nArg ratio (spectral lines magnitude ratio) must be either:\n"
            + "\t- False:  no line ration computed\n"
            + "\t- tuple of len=2: upper and lower keys of the lines\n"
            + "\t- list of tuple of len=2: upper and lower keys pairs\n"
            + "\t- np.ndarray of shape (2, N): upper keys and lower keys\n"
            + "  You provided: {}\n".format(ratio)
            + "  Available keys: {}".format(lkeys)
        )
        if not any(lc):
            raise Exception(msg)

        if lc[0]:
            c0 = (
                len(ratio) == 2
                and all([ss in lkeys for ss in ratio])
            )
            if not c0:
                raise Exception(msg)
            ratio = np.reshape(ratio, (2, 1))

        elif lc[1]:
            c0 = all([
                isinstance(tt, tuple)
                and len(tt) == 2
                and all([ss in lkeys for ss in tt])
                for tt in ratio
            ])
            if not c0:
                raise Exception(msg)
            ratio = np.array(ratio).T

        c0 = (
            isinstance(ratio, np.ndarray)
            and ratio.ndim == 2
            and ratio.shape[0] == 2
            and all([ss in lkeys for ss in ratio[0, :]])
            and all([ss in lkeys for ss in ratio[1, :]])
        )
        if not c0:
            raise Exception(msg)

    d3['ratio'] = ratio

    # pts_phi, npts_phi
    if is2d is True:
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

    # pts_total, pts_detail
    if pts_total is None:
        if dprepare is None:
            pts_total = False
        else:
            if is2d is True:
                pts_total = np.array([dprepare['lamb'], dprepare['phi']])
            else:
                pts_total = dprepare['lamb']
    if pts_detail is None:
        pts_detail = False
    if pts_detail is True and pts_total is not False:
        pts_detail = pts_total
    if pts_detail is not False:
        pts_detail = np.array(pts_detail)
    if pts_total is not False:
        pts_total = np.array(pts_total)

    return dfit, d3, pts_phi, pts_total, pts_detail


def fit1d_extract(
    dfit1d=None,
    bck=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_lamb_total=None, pts_lamb_detail=None,
):

    # -------------------
    # Check format input
    (
        dfit1d, d3, pts_phi,
        pts_lamb_total, pts_lamb_detail,
    ) = fit12d_get_data_checkformat(
        dfit=dfit1d,
        bck=bck,
        amp=amp, coefs=coefs, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        pts_total=pts_lamb_total,
        pts_detail=pts_lamb_detail,
    )

    # Extract dprepare and dind (more readable)
    dprepare = dfit1d['dinput']['dprepare']
    dind = dfit1d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare extract func
    def _get_values(key, pts_phi=None,
                    d3=d3, nspect=nspect, dinput=dfit1d['dinput'],
                    dind=dind, sol_x=dfit1d['sol_x'], scales=dfit1d['scales']):
        if d3[key]['type'] == 'lines':
            keys = dinput['keys'][d3[key]['ind']]
        else:
            keys = dinput[d3[key]['field']]['keys'][d3[key]['ind']]
        indbis = dind[d3[key]['field']][d3[key]['type']][d3[key]['ind']]
        val = sol_x[:, indbis] * scales[:, indbis]
        return keys, val

    # -------------------
    # Prepare output
    lk = [
        'bck_amp', 'bck_rate',
        'amp', 'coefs', 'ratio', 'Ti', 'width', 'vi', 'shift',
        'dratio', 'dshift',
    ]
    dout = dict.fromkeys(lk, False)

    # bck
    if d3['bck_amp'] is not False:
        dout['bck_amp'] = {
            'values': (
                dfit1d['sol_x'][:, dind['bck_amp']['x'][0]]
                * dfit1d['scales'][:, dind['bck_amp']['x'][0]]
            ),
            'units': 'a.u.',
        }
        dout['bck_rate'] = {
            'values': (
                dfit1d['sol_x'][:, dind['bck_rate']['x'][0]]
                * dfit1d['scales'][:, dind['bck_rate']['x'][0]]
            ),
            'units': 'a.u.',
        }

    # amp
    if d3['amp'] is not False:
        keys, val = _get_values('amp')
        dout['amp'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # coefs
    if d3['coefs'] is not False:
        keys, val = _get_values('coefs')
        dout['coefs'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # ratio
    if d3['ratio'] is not False:
        nratio = d3['ratio'].shape[1]
        indup = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][0, :]]]
        indlo = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][1, :]]]
        val = (dout['coefs']['values'][:, indup]
               / dout['coefs']['values'][:, indlo])
        lab = np.r_[['{} / {}'.format(dfit1d['dinput']['symb'][indup[ii]],
                                      dfit1d['dinput']['symb'][indlo[ii]])
                     for ii in range(nratio)]]
        dout['ratio'] = {'keys': dout['ratio'], 'values': val,
                         'lab': lab, 'units': 'a.u.'}

    # Ti
    if d3['Ti'] is not False:
        keys, val = _get_values('Ti')
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        indTi = np.array([iit[0] for iit in dind['width']['jac']])
        # if d3['Ti']['type'] == 'lines':
        # indTi = np.arange(0, dfit1d['dinput']['nlines'])
        indTi = indTi[d3['Ti']['ind']]
        val = (conv * val
               * dfit1d['dinput']['mz'][indTi][None, :]
               * scpct.c**2)
        dout['Ti'] = {'keys': keys, 'values': val, 'units': 'eV'}

    # width
    if d3['width'] is not False:
        keys, val = _get_values('width')
        dout['width'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # vi
    if d3['vi'] is not False:
        keys, val = _get_values('vi')
        val = val * scpct.c
        dout['vi'] = {'keys': keys, 'values': val, 'units': 'm.s^-1'}

    # shift
    if d3['shift'] is not False:
        keys, val = _get_values('shift')
        val = val * dfit1d['dinput']['lines'][None, :]
        dout['shift'] = {'keys': keys, 'values': val, 'units': 'm'}

    # double
    if dfit1d['dinput']['double'] is not False:
        double = dfit1d['dinput']['double']
        if double is True or double.get('dratio') is None:
            dout['dratio'] = dfit1d['sol_x'][:, dind['dratio']['x']]
        else:
            dout['dratio'] = np.full((nspect,), double['dratio'])
        if double is True or double.get('dratio') is None:
            dout['dshift'] = dfit1d['sol_x'][:, dind['dshift']['x']]
        else:
            dout['dshift'] = np.full((nspect,), double['dshift'])

    # -------------------
    # sol_detail and sol_tot
    sold, solt = False, False
    if pts_lamb_detail is not False or pts_lamb_total is not False:

        (func_detail,
         func_cost, _) = _funccostjac.multigausfit1d_from_dlines_funccostjac(
            dprepare['lamb'],
            dinput=dfit1d['dinput'],
            dind=dind, jac=dfit1d['jac'])

        if pts_lamb_detail is not False:
            shape = tuple(np.r_[nspect, pts_lamb_detail.shape,
                                dfit1d['dinput']['nlines']+1])
            sold = np.full(shape, np.nan)
            for ii in range(nspect):
                sold[ii, dprepare['indok'][ii, :], :] = func_detail(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=dprepare['indok'][ii, :],
                )
                # indok_var=dprepare['indok_var'][ii])

        if pts_lamb_total is not False:
            shape = tuple(np.r_[nspect, pts_lamb_total.shape])
            solt = np.full(shape, np.nan)
            for ii in range(nspect):
                solt[ii, dprepare['indok'][ii, :]] = func_cost(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=dprepare['indok'][ii, :],
                    data=0.)

            # Double-check consistency if possible
            c0 = (pts_lamb_detail is not False
                  and np.allclose(pts_lamb_total, pts_lamb_detail))
            if c0:
                if not np.allclose(solt, np.sum(sold, axis=-1),
                                   equal_nan=True):
                    msg = "Inconsistent computations detail vs total"
                    raise Exception(msg)

    dout['sol_detail'] = sold
    dout['sol_tot'] = solt
    dout['units'] = 'a.u.'

    # -------------------
    # Add input args
    dout['d3'] = d3
    dout['pts_lamb_detail'] = pts_lamb_detail
    dout['pts_lamb_total'] = pts_lamb_total
    return dout


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


def fit2d_extract(dfit2d=None,
                  amp=None, coefs=None, ratio=None,
                  Ti=None, width=None,
                  vi=None, shift=None,
                  pts_lamb_phi_total=None, pts_lamb_phi_detail=None):

    # -------------------
    # Check format input
    out = fit12d_get_data_checkformat(
        dfit=dfit2d,
        amp=amp, coefs=coefs, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        pts_total=pts_lamb_total,
        pts_detail=pts_lamb_detail)

    d3, pts_phi, pts_lamb_phi_total, pts_lamb_phi_detail = out

    # Extract dprepare and dind (more readable)
    dprepare = dfit1d['dinput']['dprepare']
    dind = dfit1d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare extract func
    # TBF
    def _get_values(key, pts_phi=None,
                    d3=d3, nspect=nspect, dinput=dfit1d['dinput'],
                    dind=dind, sol_x=dfit1d['sol_x'], scales=dfit1d['scales']):
        if d3[key]['type'] == 'lines':
            keys = dinput['keys'][d3[key]['ind']]
        else:
            keys = dinput[d3[key]['field']]['keys'][d3[key]['ind']]
        indbis = dind[d3[key]['field']][d3[key]['type']][d3[key]['ind']]

        # 1d vs 2d
        if pts_phi is None:
            val = sol_x[:, indbis] * scales[:, indbis]
        else:
            BS = BSpline(dinput['knots_mult'],
                         np.ones((dinput['nbs'], ncoefs), dtype=float),
                         dinput['deg'],
                         extrapolate=False, axis=0)
        for ii in range(nspect):
            BS.c = sol_x[ii, indbis] * scales[ii, indbis]
            val[ii, :, :] = BS(pts_phi)

        return keys, val

    # -------------------
    # Prepare output
    lk = ['amp', 'coefs', 'ratio', 'Ti', 'width', 'vi', 'shift',
          'dratio', 'dshift']
    dout = dict.fromkeys(lk, False)

    # amp
    if d3['amp'] is not False:
        keys, val = _get_values('amp')
        dout['amp'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # coefs
    if d3['coefs'] is not False:
        keys, val = _get_values('coefs')
        dout['coefs'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # ratio
    if d3['ratio'] is not False:
        nratio = d3['ratio'].shape[1]
        indup = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][0, :]]]
        indlo = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][1, :]]]
        val = (dout['coefs']['values'][:, indup]
               / dout['coefs']['values'][:, indlo])
        lab = np.r_[['{} / {}'.format(dfit1d['dinput']['symb'][indup[ii]],
                                      dfit1d['dinput']['symb'][indlo[ii]])
                     for ii in range(nratio)]]
        dout['ratio'] = {'keys': dout['ratio'], 'values': val,
                         'lab': lab, 'units': 'a.u.'}

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

    # -------------------
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

    # -------------------
    # Add input args
    dout['d3'] = d3
    dout['pts_phi'] = pts_phi
    dout['pts_lamb_phi_detail'] = pts_lamb_phi_detail
    dout['pts_lamb_phi_total'] = pts_lamb_phi_total
    return dout


###########################################################
###########################################################
#
#   Plot fitted data from pre-computed dict of fitted results
#
###########################################################
###########################################################

def fit2d_plot(dout=None):

    # ----------------------
    # Optional plotting
    if plot is True:
        if plotmode is None:
            plotmode = 'transform'
        if indspect is None:
            indspect = 0

        if spect1d is not None:
            # Compute lambfit / phifit and spectrum1d
            if nlambfit is None:
                nlambfit = 200
            ((spect1d, fit1d), lambfit,
             phifit, _, phiminmax) = self._calc_spect1d_from_data2d(
                 [dataflat[indspect, :], dfit2d['sol_tot'][indspect, :]],
                 lambflat, phiflat,
                 nlambfit=nlambfit, nphifit=10,
                 spect1d=spect1d, mask=None, vertsum1d=False)
        else:
            fit1d, lambfit, phiminmax = None, None, None

        dax = _plot_optics.CrystalBragg_plot_data_fit2d(
            xi=xi, xj=xj, data=dfit2d['data'],
            lamb=dfit2d['lamb'], phi=dfit2d['phi'], indspect=indspect,
            indok=indok, dfit2d=dfit2d,
            dax=dax, plotmode=plotmode, angunits=angunits,
            cmap=cmap, vmin=vmin, vmax=vmax,
            spect1d=spect1d, fit1d=fit1d,
            lambfit=lambfit, phiminmax=phiminmax,
            dmargin=dmargin, tit=tit, wintit=wintit, fs=fs)
    return dax


###########################################################
###########################################################
#
#           1d vertical fitting for noise analysis
#
###########################################################
###########################################################


def get_noise_costjac(deg=None, nbsplines=None, dbsplines=None, phi=None,
                      phiminmax=None, symmetryaxis=None, sparse=None):

    if sparse is None:
        sparse = False

    if dbsplines is None:
        dbsplines = multigausfit2d_from_dlines_dbsplines(
            knots=None, deg=deg, nbsplines=nbsplines,
            phimin=phiminmax[0], phimax=phiminmax[1],
            symmetryaxis=symmetryaxis)

    def cost(x,
             km=dbsplines['knots_mult'],
             deg=dbsplines['deg'],
             data=0., phi=phi):
        return scpinterp.BSpline(km, x, deg,
                                 extrapolate=False, axis=0)(phi) - data

    jac = np.zeros((phi.size, dbsplines['nbs']), dtype=float)
    km = dbsplines['knots_mult']
    kpb = dbsplines['nknotsperbs']
    lind = [(phi >= km[ii]) & (phi < km[ii+kpb-1])
            for ii in range(dbsplines['nbs'])]
    if sparse is True:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return scpsparse.csr_matrix(jac)
    else:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return jac
    return cost, jac_func


def _basic_loop(ilambu=None, ilamb=None, phi=None, data=None, mask=None,
                domain=None, nbs=None, dbsplines=None, nspect=None,
                method=None, tr_solver=None, tr_options=None, loss=None,
                xtol=None, ftol=None, gtol=None, max_nfev=None, verbose=None):

    # ---------------
    # Check inputs
    if method is None:
        method = _METHOD
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
        loss = _LOSS
    if max_nfev is None:
        max_nfev = None

    x0 = 1. - (2.*np.arange(nbs)/nbs - 1.)**2

    # ---------------
    # Prepare outputs
    dataint = np.full((nspect, ilambu.size), np.nan)
    fit = np.full(data.shape, np.nan)
    indsort = np.zeros((2, phi.size), dtype=int)
    indout_noeval = np.zeros(phi.shape, dtype=bool)
    chi2n = np.full((nspect, ilambu.size), np.nan)
    chi2_meandata = np.full((nspect, ilambu.size), np.nan)

    # ---------------
    # Main loop
    i0, indnan = 0, []
    for jj in range(ilambu.size):
        ind = ilamb == ilambu[jj]
        nind = ind.sum()
        isort = i0 + np.arange(0, nind)

        # skips cases with no points
        if not np.any(ind):
            continue

        inds = np.argsort(phi[ind])
        inds_rev = np.argsort(inds)
        indsort[0, isort] = ind.nonzero()[0][inds]
        indsort[1, isort] = ind.nonzero()[1][inds]

        phisort = phi[indsort[0, isort], indsort[1, isort]]
        datasort = data[:, indsort[0, isort], indsort[1, isort]]
        dataint[:, jj] = np.nanmean(datasort, axis=1)

        # skips cases with to few points
        indok = ~np.any(np.isnan(datasort), axis=0)
        if mask is not None:
            indok &= mask[indsort[0, isort], indsort[1, isort]]

        # Check there are enough phi vs bsplines
        indphimin = np.searchsorted(np.linspace(domain['phi']['minmax'][0],
                                                domain['phi']['minmax'][1],
                                                nbs + 1),
                                    phisort[indok])
        if np.unique(indphimin).size < nbs:
            indout_noeval[ind] = True
            continue
        indout_noeval[ind] = ~indok[inds_rev]

        # get bsplines func
        func_cost, func_jac = get_noise_costjac(phi=phisort[indok],
                                                dbsplines=dbsplines,
                                                sparse=False,
                                                symmetryaxis=False)
        for tt in range(nspect):
            if verbose > 0:
                msg = ("\tlambbin {} / {}".format(jj+1, ilambu.size)
                       + "    "
                       + "time step = {} / {}".format(tt+1, nspect))
                print(msg.ljust(50), end='\r', flush=True)

            if dataint[tt, jj] == 0.:
                continue

            datai = datasort[tt, indok] / dataint[tt, jj]
            res = scpopt.least_squares(
                func_cost, x0, jac=func_jac,
                method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                x_scale='jac', f_scale=1.0, loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options={}, jac_sparsity=None,
                max_nfev=max_nfev, verbose=0, args=(),
                kwargs={'data': datai})

            # Store in original shape
            fit[tt, ind] = (
                func_cost(res.x, phi=phisort, data=0.)
                * dataint[tt, jj]
            )[inds_rev]
            chi2_meandata[tt, jj] = np.nanmean(fit[tt, ind])
            chi2n[tt, jj] = np.nanmean(func_cost(x=res.x, data=datai)**2)

        i0 += nind
        indnan.append(i0)
    return (fit, dataint, indsort, np.array(indnan), indout_noeval,
            chi2n, chi2_meandata)


def noise_analysis_2d(
    data, lamb, phi, mask=None, margin=None, valid_fraction=None,
    deg=None, knots=None, nbsplines=None, nxerrbin=None,
    nlamb=None, loss=None, max_nfev=None,
    xtol=None, ftol=None, gtol=None,
    method=None, tr_solver=None, tr_options=None,
    verbose=None, plot=None,
    ms=None, dcolor=None,
    dax=None, fs=None, dmargin=None,
    wintit=None, tit=None, sublab=None,
    save_fig=None, name_fig=None, path_fig=None, fmt=None,
    return_dax=None,
):

    # -------------
    # Check inputs
    if not isinstance(nbsplines, int):
        msg = "Please provide a (>0) integer value for nbsplines"
        raise Exception(msg)

    if deg is None:
        deg = 2
    if plot is None:
        plot = True
    if verbose is None:
        verbose = 1
    if return_dax is None:
        return_dax = False

    c0 = lamb.shape == phi.shape == data.shape[1:]
    if c0 is not True:
        msg = (
            "input data, lamb, phi are non-conform!\n"
            + "\t- expected lamb.shape == phi.shape == data.shape[1:]\n"
            + "\t- provided:\n"
            + "\t\tlamb.shape = {}\n".format(lamb.shape)
            + "\t\tphi.shape = {}\n".format(phi.shape)
            + "\t\tdata.shape = {}\n".format(data.shape)
        )
        raise Exception(msg)

    nspect = data.shape[0]
    domain = {'lamb': {'minmax': [np.nanmin(lamb), np.nanmax(lamb)]},
              'phi': {'minmax': [np.nanmin(phi), np.nanmax(phi)]}}

    if nlamb is None:
        if lamb.ndim == 2:
            nlamb = lamb.shape[0]
        else:
            msg = ("Please provide a value for nlamb (nb of bins)!")
            raise Exception(msg)
    nlamb = int(nlamb)

    # -------------
    # lamb binning
    lambedges = np.linspace(domain['lamb']['minmax'][0],
                            domain['lamb']['minmax'][1], nlamb+1)
    ilamb = np.searchsorted(lambedges, lamb)
    ilambu = np.unique(ilamb)

    # -------------
    # bspline dict and plotting utilities
    dbsplines = multigausfit2d_from_dlines_dbsplines(
        knots=None, deg=deg, nbsplines=nbsplines,
        phimin=domain['phi']['minmax'][0],
        phimax=domain['phi']['minmax'][1],
        symmetryaxis=False)

    # plotting utils
    bs_phi = np.linspace(domain['phi']['minmax'][0],
                         domain['phi']['minmax'][1], 101)
    bs_val = np.array([
        scpinterp.BSpline.basis_element(
            dbsplines['knots_mult'][ii:ii+dbsplines['nknotsperbs']],
            extrapolate=False)(bs_phi)
        for ii in range(nbsplines)]).T

    # -------------
    # Perform fits
    (fit, dataint, indsort, indnan, indout_noeval,
     chi2n, chi2_meandata) = _basic_loop(
        ilambu=ilambu, ilamb=ilamb, phi=phi, data=data, mask=mask,
        domain=domain, nbs=nbsplines, dbsplines=dbsplines, nspect=nspect,
        method=method, tr_solver=tr_solver, tr_options=tr_options, loss=loss,
        xtol=xtol, ftol=ftol, gtol=gtol,
        max_nfev=max_nfev, verbose=verbose)

    # -------------
    # Identify outliers with respect to noise model
    (mean, var, xdata, const,
     indout_var, _, margin, valid_fraction) = get_noise_analysis_var_mask(
         fit=fit, data=data, mask=(mask & (~indout_noeval)),
         margin=margin, valid_fraction=valid_fraction)

    # Safety check
    if mask is None:
        indout_mask = np.zeros(lamb.shape, dtype=bool)
    else:
        indout_mask = ~mask
    indout_noeval[~mask] = False
    indout_tot = np.array([~mask,
                           indout_noeval,
                           np.any(indout_var, axis=0)])
    c0 = np.all(np.sum(indout_tot.astype(int), axis=0) <= 1)
    if not c0:
        msg = "Overlapping indout!"
        raise Exception(msg)

    indin = ~np.any(indout_tot, axis=0)

    # -------------
    # output dict
    dnoise = {
        'data': data, 'phi': phi, 'fit': fit,
        'chi2n': chi2n, 'chi2_meandata': chi2_meandata, 'dataint': dataint,
        'domain': domain, 'indin': indin, 'indout_mask': indout_mask,
        'indout_noeval': indout_noeval, 'indout_var': indout_var,
        'mask': mask, 'ind_noeval': None,
        'indsort': indsort, 'indnan': np.array(indnan),
        'nbsplines': nbsplines, 'bs_phi': bs_phi, 'bs_val': bs_val,
        'deg': deg, 'lambedges': lambedges, 'deg': deg,
        'ilamb': ilamb, 'ilambu': ilambu,
        'var_mean': mean, 'var': var, 'var_xdata': xdata,
        'var_const': const, 'var_margin': margin,
        'var_fraction': valid_fraction}

    # Plot
    if plot is True:
        try:
            dax = _plot.plot_noise_analysis(
                dnoise=dnoise,
                ms=ms, dcolor=dcolor,
                dax=dax, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, sublab=sublab,
                save=save_fig, name=name_fig, path=path_fig, fmt=fmt)
        except Exception as err:
            msg = ("Plotting failed: {}".format(str(err)))
            warnings.warn(msg)
    if return_dax is True:
        return dnoise, dax
    else:
        return dnoise


def noise_analysis_2d_scannbs(
    data, lamb, phi, mask=None, nxerrbin=None,
    deg=None, knots=None, nbsplines=None, lnbsplines=None,
    nlamb=None, loss=None, max_nfev=None,
    xtol=None, ftol=None, gtol=None,
    method=None, tr_solver=None, tr_options=None,
    verbose=None, plot=None,
    dax=None, fs=None, dmargin=None,
    wintit=None, tit=None, ms=None, sublab=None,
    save_fig=None, name_fig=None, path_fig=None,
    fmt=None, return_dax=None,
):

    # -------------
    # Check inputs
    if lnbsplines is None:
        lnbsplines = np.arange(5, 21)
    else:
        lnbsplines = np.atleast_1d(lnbsplines).ravel().astype(int)
    if nbsplines is None:
        nbsplines = int(lnbsplines.size/2)
    if nbsplines is not None:
        nbsplines = np.unique(np.atleast_1d(nbsplines)).astype(int)
    nlnbs = lnbsplines.size
    if nxerrbin is None:
        nxerrbin = 100

    if deg is None:
        deg = 2
    if plot is None:
        plot = True
    if verbose is None:
        verbose = 1
    if return_dax is None:
        return_dax = False

    c0 = lamb.shape == phi.shape == data.shape[1:]
    if c0 is not True:
        msg = ("input data, lamb, phi are non-conform!\n"
               + "\t- expected lamb.shape == phi.shape == data.shape[1:]\n"
               + "\t- provided: ")
        raise Exception(msg)

    nspect = data.shape[0]
    domain = {'lamb': {'minmax': [np.nanmin(lamb), np.nanmax(lamb)]},
              'phi': {'minmax': [np.nanmin(phi), np.nanmax(phi)]}}

    if nlamb is None:
        if lamb.ndim == 2:
            nlamb = lamb.shape[0]
        else:
            msg = ("Please provide a value for nlamb (nb of bins)!")
            raise Exception(msg)
    nlamb = int(nlamb)

    # -------------
    # lamb binning
    lambedges = np.linspace(domain['lamb']['minmax'][0],
                            domain['lamb']['minmax'][1], nlamb+1)
    ilamb = np.searchsorted(lambedges, lamb)
    ilambu = np.unique(ilamb)

    # -------------
    # Perform fits
    xdata_edge = np.linspace(0, np.nanmax(data[:, mask]), nxerrbin+1)
    xdata = 0.5*(xdata_edge[1:] + xdata_edge[:-1])
    dataint = np.full((nspect, ilambu.size), np.nan)
    # fit = np.full(data.shape, np.nan)
    indsort = np.zeros((2, phi.size), dtype=int)
    # indout_noeval = np.zeros(phi.shape, dtype=bool)
    chi2n = np.full((nlnbs, nspect, ilambu.size), np.nan)
    chi2_meandata = np.full((nlnbs, nspect, ilambu.size), np.nan)
    const = np.full((nlnbs,), np.nan)
    mean = np.full((nlnbs, nxerrbin), np.nan)
    var = np.full((nlnbs, nxerrbin), np.nan)
    bs_phidata, bs_data, bs_fit, bs_indin = [], [], [], []
    for ii in range(lnbsplines.size):
        nbs = int(lnbsplines[ii])
        # -------------
        # bspline dict and plotting utilities
        dbsplines = multigausfit2d_from_dlines_dbsplines(
            knots=None, deg=deg, nbsplines=nbs,
            phimin=domain['phi']['minmax'][0],
            phimax=domain['phi']['minmax'][1],
            symmetryaxis=False,
        )

        # -------------
        # Perform fits
        if verbose > 0:
            msg = "nbs = {} ({} / {})".format(nbs, ii+1, lnbsplines.size)
            print(msg)
        (fiti, dataint, indsort, indnan, indout_noeval,
         chi2n[ii, ...], chi2_meandata[ii, ...]) = _basic_loop(
             ilambu=ilambu, ilamb=ilamb, phi=phi, data=data, mask=mask,
             domain=domain, nbs=nbs, dbsplines=dbsplines, nspect=nspect,
             method=method, tr_solver=tr_solver, tr_options=tr_options,
             loss=loss, xtol=xtol, ftol=ftol, gtol=gtol,
             max_nfev=max_nfev, verbose=verbose)

        if ii == 0:
            ind_intmax = np.unravel_index(np.argmax(dataint, axis=None),
                                          dataint.shape)

        if nbs in nbsplines:
            isi = np.split(indsort, indnan, axis=1)[ind_intmax[1]]
            bs_phidata.append(phi[isi[0], isi[1]])
            bs_data.append(data[ind_intmax[0], isi[0], isi[1]])
            bs_fit.append(fiti[ind_intmax[0], isi[0], isi[1]])
            indini = ~np.any(np.array([~mask, indout_noeval]), axis=0)
            bs_indin.append(indini[isi[0], isi[1]])

        # -------------
        # Identify outliers with respect to noise model
        (meani, vari, xdatai, consti,
         _, inderrui, _, _) = get_noise_analysis_var_mask(
             fit=fiti, data=data, xdata_edge=xdata_edge,
             mask=(mask & (~indout_noeval)),
             margin=None, valid_fraction=False)

        const[ii] = consti
        mean[ii, inderrui] = meani
        var[ii, inderrui] = vari

    # -------------
    # output dict
    dnoise_scan = {
        'data': data,
        'chi2n': chi2n, 'chi2_meandata': chi2_meandata, 'dataint': dataint,
        'domain': domain, 'lnbsplines': lnbsplines, 'nbsplines': nbsplines,
        'deg': deg, 'lambedges': lambedges, 'deg': deg,
        'ilamb': ilamb, 'ilambu': ilambu,
        'bs_phidata': bs_phidata, 'bs_data': bs_data,
        'bs_fit': bs_fit, 'bs_indin': bs_indin,
        'var_mean': mean, 'var': var, 'var_xdata': xdata,
        'var_const': const,
    }

    # Plot
    if plot is True:
        try:
            dax = _plot.plot_noise_analysis_scannbs(
                dnoise=dnoise_scan, ms=ms,
                dax=dax, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, sublab=sublab,
                save=save_fig, name=name_fig, path=path_fig, fmt=fmt)
        except Exception as err:
            msg = ("Plotting failed: {}".format(str(err)))
            warnings.warn(msg)

    if return_dax is True:
        return dnoise_scan, dax
    else:
        return dnoise_scan


def get_noise_analysis_var_mask(fit=None, data=None,
                                xdata_edge=None, nxerrbin=None,
                                valid_fraction=None,
                                mask=None, margin=None):
    if margin is None:
        margin = _SIGMA_MARGIN
    if valid_fraction is None:
        valid_fraction = False
    if nxerrbin is None:
        nxerrbin = 100

    err = fit - data
    if mask is None:
        mask = np.ones(err.shape[1:], dtype=bool)
    if xdata_edge is None:
        xdata_edge = np.linspace(0, np.nanmax(fit[:, mask]), nxerrbin)
    inderr = np.searchsorted(xdata_edge[1:-1], fit[:, mask])
    inderru = np.unique(inderr[~np.isnan(err[:, mask])])
    xdata = 0.5*(xdata_edge[1:] + xdata_edge[:-1])[inderru]
    mean = np.full((inderru.size,), np.nan)
    var = np.full((inderru.size,), np.nan)
    nn = np.full((inderru.size,), np.nan)
    for ii in range(inderru.size):
        ind = inderr == inderru[ii]
        indok = ~np.isnan(err[:, mask][ind])
        nn[ii] = np.sum(indok)
        mean[ii] = np.nanmean(err[:, mask][ind])
        var[ii] = nn[ii] * np.nanmean(err[:, mask][ind]**2) / (nn[ii] - 1)

    # fit sqrt on sigma (weight by log10 to take into account diff. nb. of pts)
    const = np.nansum(
        (np.log10(nn)*np.sqrt(var / xdata)) / np.nansum(np.log10(nn))
    )

    # indout
    indok = (~np.isnan(err)) & mask[None, ...]
    indout = np.zeros(err.shape, dtype=bool)
    indout[indok] = (np.abs(err[indok])
                     > margin*const*np.sqrt(np.abs(fit[indok])))
    if valid_fraction is not False:
        indout = np.sum(indout, axis=0)/float(indout.shape[0]) > valid_fraction
    return mean, var, xdata, const, indout, inderru, margin, valid_fraction
