
# Built-in
import warnings
import itertools as itt
import copy
import datetime as dtm      # DB

# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
import matplotlib.pyplot as plt


__all__ = [
    'fit1d_dinput',
    'fit2d_dinput',
    'fit12d_dvalid',
    'fit12d_dscales',
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
_SYMMETRY_CENTRAL_FRACTION = 0.3
_BINNING = False
_POS = False
_SUBSET = False
_VALID_NSIGMA = 6.
_VALID_FRACTION = 0.8
_LTYPES = [int, float, np.int_, np.float_]
_DBOUNDS = {
    'bck_amp': (0., 3.),
    'bck_rate': (-3., 3.),
    'amp': (0, 10),
    'width': (0.01, 2.),
    'shift': (-1, 1),
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
_DINDOK = {
    0: 'ok',
    -1: 'mask',
    -2: 'out of domain',
    -3: 'neg or NaN',
    -4: 'binning=0',
    -5: 'S/N valid, excluded',
    -6: 'S/N non-valid, included',
    -7: 'S/N non-valid, excluded',
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

    # double the number of phiok
    phiok = np.linspace(phiok.min(), phiok.max(), phiok.size*2 + 1)

    # Compute new phi and associated costs
    phi2 = phi[:, None] - phiok[None, :]
    phi2min = np.min([
        np.nanmax(np.abs(phi2 * (phi2 < 0)), axis=0),
        np.nanmax(np.abs(phi2 * (phi2 > 0)), axis=0)],
        axis=0,
    )

    # prepare phi2 positive and negative
    indout = np.abs(phi2) > phi2min[None, :]
    phi2p = np.abs(phi2)
    phi2n = np.abs(phi2)
    phi2p[(phi2 < 0) | indout] = np.nan
    phi2n[(phi2 > 0) | indout] = np.nan
    nok = np.min(
        [
            np.sum((~np.isnan(phi2p)), axis=0),
            np.sum((~np.isnan(phi2n)), axis=0)
        ],
        axis=0,
    )

    # find phiok of minimum cost
    cost = np.full((data.shape[0], phiok.size), np.nan)
    for ii in range(phiok.size):
        indp = np.argsort(np.abs(phi2p[:, ii]))
        indn = np.argsort(np.abs(phi2n[:, ii]))
        cost[:, ii] = np.nansum(
            (data[:, indp] - data[:, indn])[:, :nok[ii]]**2,
            axis=1,
        )
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
            f"dconstraints['{k0}'] shoud be either:\n"
            f"\t- False ({c0}): no constraint\n"
            f"\t- str ({c1}): key from dlines['<lines>'] "
            "to be used as criterion\n"
            f"\t\t available crit: {pavail}\n"
            f"\t- dict ({c2}): "
            "{str: line_keyi or [line_keyi, ..., line_keyj}\n"
            f"\t- dict ({c3}): "
            "{line_keyi: {'key': str, 'coef': , 'offset': }}\n"
            f"\t- dict ({c4}): "
            "{'keys': [], 'ind': np.ndarray}\n"
            f"  Available line_keys:\n{sorted(keys)}\n"
            f"  You provided:\n{indict}"
        )
        raise Exception(msg)

    # ------------------------
    # str key to be taken from dlines as criterion
    if c0:
        lk = keys
        ind = np.eye(nlines, dtype=bool)
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': np.ones((nlines,)),
            'offset': np.zeros((nlines,)),
        }

    if c1:
        lk = sorted(set([dlines[k1].get(indict, k1) for k1 in keys]))
        ind = np.array(
            [
                [dlines[k2].get(indict, k2) == k1 for k2 in keys]
                for k1 in lk
            ],
            dtype=bool,
        )
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
        ind = np.array(
            [[k2 in indict[k1] for k2 in keys] for k1 in lk],
            dtype=bool,
        )
        outdict = {
            'keys': np.r_[lk],
            'ind': ind,
            'coefs': np.ones((nlines,)),
            'offset': np.zeros((nlines,)),
        }

    elif c3:
        lk = sorted(set([v0['key'] for v0 in indict.values()]))
        lk += sorted(set(keys).difference(indict.keys()))
        ind = np.array(
            [
                [indict.get(k2, {'key': k2})['key'] == k1 for k2 in keys]
                for k1 in lk
            ],
            dtype=bool,
        )
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
    # Remove group with no match

    indnomatch = np.sum(ind, axis=1) == 0
    if np.any(indnomatch):
        lknom = outdict['keys'][indnomatch]
        outdict['keys'] = outdict['keys'][~indnomatch]
        outdict['ind'] = outdict['ind'][~indnomatch, :]
        lstr = [f"\t- {k1}" for k1 in lknom]
        msg = (
            f"The following {k0} groups match no lines, they are removed:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # ------------------------
    # Ultimate conformity checks

    assert sorted(outdict.keys()) == ['coefs', 'ind', 'keys', 'offset']

    # check ind (root of all subsequent ind arrays)
    assert isinstance(outdict['ind'], np.ndarray)
    assert outdict['ind'].dtype == np.bool_
    assert outdict['ind'].shape == (outdict['keys'].size, nlines)
    # check each line is associated to a unique group
    assert np.all(np.sum(outdict['ind'], axis=0) == 1)
    # check each group is associated to at least one line
    assert np.all(np.sum(outdict['ind'], axis=1) >= 1)

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

        if nxi is None or nxj is None:
            msg = "Arg (nxi, nxj) must be provided for double-checking shapes"
            raise Exception(msg)

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
        domain = {
            k0: {
                'spec': [np.inf*np.r_[-1., 1.]],
                'minmax': np.inf*np.r_[-1., 1.],
            }
            for k0 in keys
        }
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
    dlamb_ref=None,
    dphi_ref=None,
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
            and all([kk in binning.keys() for kk in lk])
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

        # ------------
        # safet checks

        if np.any(~np.isfinite(binning[k0]['edges'])):
            msg = (
                f"Non-finite value in binning['{k0}']['edges']\n"
                + str(binning[k0]['edges'])
            )
            raise Exception(msg)

        if not np.allclose(
            binning[k0]['edges'],
            np.unique(binning[k0]['edges']),
        ):
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

    # --------------
    # Check binning

    for (dref, k0) in [(dlamb_ref, 'lamb'), (dphi_ref, 'phi')]:
        if dref is not None:
            di = np.mean(np.diff(binning[k0]['edges']))
            if di < dref:
                ni_rec = (
                    (domain[k0]['minmax'][1] - domain[k0]['minmax'][0]) / dref
                )
                msg = (
                    f"binning[{k0}] seems finer than the original!\n"
                    f"\t- estimated original step: {dref}\n"
                    f"\t- binning step: {di}\n"
                    f"  => nb. of recommended steps: {ni_rec:5.1f}"
                )
                warnings.warn(msg)

    return binning


def binning_2d_data(
    lamb, phi, data,
    indok=None,
    indok_bool=None,
    domain=None, binning=None,
    nbsplines=None,
    phi1d=None, lamb1d=None,
    dataphi1d=None, datalamb1d=None,
):

    # -------------------------
    # Preliminary check on bins

    dlamb_ref, dphi_ref = None, None
    if lamb.ndim == 2:
        indmid = int(lamb.shape[0]/2)
        dlamb_ref = (np.max(lamb[indmid, :]) - np.min(lamb[indmid, :]))
        dlamb_ref = dlamb_ref / lamb.shape[1]
        indmid = int(lamb.shape[1]/2)
        dphi_ref = (np.max(phi[:, indmid]) - np.min(phi[:, indmid]))
        dphi_ref = dphi_ref / lamb.shape[0]

    # ------------------
    # Checkformat input

    binning = _binning_check(
        binning,
        domain=domain,
        dlamb_ref=dlamb_ref,
        nbsplines=nbsplines,
    )

    nspect = data.shape[0]
    if binning is False:
        if phi1d is None:
            phi1d_edges = np.linspace(
                domain['phi']['minmax'][0], domain['phi']['minmax'][1], 100,
            )
            lamb1d_edges = np.linspace(
                domain['lamb']['minmax'][0], domain['lamb']['minmax'][1], 100,
            )
            dataf = data.reshape((nspect, data.shape[1]*data.shape[2]))
            dataphi1d = scpstats.binned_statistic(
                phi.ravel(),
                dataf,
                statistic='sum',
                bins=phi1d_edges,
            )[0]
            datalamb1d = scpstats.binned_statistic(
                lamb.ravel(),
                dataf,
                statistic='sum',
                bins=lamb1d_edges,
            )[0]
            phi1d = 0.5*(phi1d_edges[1:] + phi1d_edges[:-1])
            lamb1d = 0.5*(lamb1d_edges[1:] + lamb1d_edges[:-1])

        return (
            lamb, phi, data, indok, binning,
            phi1d, lamb1d, dataphi1d, datalamb1d,
        )

    else:
        nphi = binning['phi']['nbins']
        nlamb = binning['lamb']['nbins']
        bins = (binning['phi']['edges'], binning['lamb']['edges'])

        # ------------------
        # Compute

        databin = np.full((nspect, nphi, nlamb), np.nan)
        nperbin = np.full((nspect, nphi, nlamb), np.nan)
        indok_new = np.zeros((nspect, nphi, nlamb), dtype=np.int8)
        for ii in range(nspect):
            databin[ii, ...] = scpstats.binned_statistic_2d(
                phi[indok_bool[ii, ...]],
                lamb[indok_bool[ii, ...]],
                data[ii, indok_bool[ii, ...]],
                statistic='mean',       # Beware: for valid S/N use sum!
                bins=bins,
                range=None,
                expand_binnumbers=True,
            )[0]
            nperbin[ii, ...] = scpstats.binned_statistic_2d(
                phi[indok_bool[ii, ...]],
                lamb[indok_bool[ii, ...]],
                np.ones((indok_bool[ii, ...].sum(),), dtype=int),
                statistic='sum',
                bins=bins,
                range=None,
                expand_binnumbers=True,
            )[0]

        binning['nperbin'] = nperbin

        lamb1d = 0.5*(
            binning['lamb']['edges'][1:] + binning['lamb']['edges'][:-1]
        )
        phi1d = 0.5*(
            binning['phi']['edges'][1:] + binning['phi']['edges'][:-1]
        )
        lambbin = np.repeat(lamb1d[None, :], nphi, axis=0)
        phibin = np.repeat(phi1d[:, None], nlamb, axis=1)

        # reconstructing indok
        indok_new[np.isnan(databin)] = -1
        indok_new[nperbin == 0] = -4

        # dataphi1d
        dataphi1d = np.full(databin.shape[:2], np.nan)
        indok = ~np.all(np.isnan(databin), axis=2)
        dataphi1d[indok] = np.nanmean(databin[indok, :], axis=-1)
        datalamb1d = np.full(databin.shape[::2], np.nan)
        indok = ~np.all(np.isnan(databin), axis=1)
        datalamb1d[indok] = (
            np.nanmean(databin.swapaxes(1, 2)[indok, :], axis=-1)
            + np.nanstd(databin.swapaxes(1, 2)[indok, :], axis=-1)
        )

        return (
            lambbin, phibin, databin, indok_new, binning,
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

    c0 = (
        (
            isinstance(subset, np.ndarray)
            and subset.shape == indlogical.shape
            and 'bool' in subset.dtype.name
        )
        or (
            type(subset) in [int, float, np.int_, np.float_]
            and subset >= 0
        )
    )
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
            indlogical.sum(),
            size=int(indlogical.sum() - subset),
            replace=False,
            shuffle=False,
        )
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
    update_domain=None,
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
    indok = np.zeros(data.shape, dtype=np.int8)
    if mask is not None:
        indok[:, ~mask] = -1

    inddomain, domain = apply_domain(lamb, domain=domain)
    if mask is not None:
        indok[:, (~inddomain) & mask] = -2
    else:
        indok[:, ~inddomain] = -2

    # Optional positivity constraint
    if pos is not False:
        if pos is True:
            data[data < 0.] = np.nan
        else:
            data[data < 0.] = pos

    indok[(indok == 0) & np.isnan(data)] = -3

    # Recompute domain
    indok_bool = indok == 0

    if update_domain is None:
        update_domain = bool(np.any(np.isinf(domain['lamb']['minmax'])))

    if update_domain is True:
        domain['lamb']['minmax'] = [
            np.nanmin(lamb[np.any(indok_bool, axis=0)]),
            np.nanmax(lamb[np.any(indok_bool, axis=0)]),
        ]

    # --------------
    # Optionally fit only on subset
    # randomly pick subset indices (replace=False => no duplicates)
    # indok = _get_subset_indices(subset, indok)

    if np.any(np.isnan(data[indok_bool])):
        msg = (
            "Some NaNs in data not caught by indok!"
        )
        raise Exception(msg)

    if np.sum(indok_bool) == 0:
        msg = "There does not seem to be any usable data (no indok)"
        raise Exception(msg)

    # --------------
    # Return
    dprepare = {
        'data': data,
        'lamb': lamb,
        'domain': domain,
        'indok': indok,
        'indok_bool': indok_bool,
        'dindok': dict(_DINDOK),
        'pos': pos,
        'subset': subset,
    }
    return dprepare


def multigausfit2d_from_dlines_prepare(
    data=None, lamb=None, phi=None,
    mask=None, domain=None,
    update_domain=None,
    pos=None, binning=None,
    nbsplines=None, deg=None, subset=None,
    nxi=None, nxj=None,
    lphi=None, lphi_tol=None,
    return_raw_data=None,
):

    # --------------
    # Check input

    if return_raw_data is None:
        return_raw_data = False

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
    indok = np.zeros(data.shape, dtype=np.int8)
    if mask is not None:
        indok[:, ~mask] = -1

    inddomain, domain = apply_domain(lamb, phi, domain=domain)
    if mask is not None:
        indok[:, (~inddomain) & mask] = -2
    else:
        indok[:, ~inddomain] = -2

    # Optional positivity constraint
    if pos is not False:
        if pos is True:
            data[data < 0.] = np.nan
        else:
            data[data < 0.] = pos

    # Introduce time-dependence (useful for valid)
    indok[(indok == 0) & np.isnan(data)] = -3

    # Recompute domain
    indok_bool = indok == 0
    if not np.any(indok_bool):
        msg = "No valid point in data!"
        raise Exception(msg)

    if update_domain is None:
        update_domain = bool(
            np.any(np.isinf(domain['lamb']['minmax']))
            or np.any(np.isinf(domain['phi']['minmax']))
        )

    if update_domain is True:
        domain['lamb']['minmax'] = [
            np.nanmin(lamb[np.any(indok_bool, axis=0)]),
            np.nanmax(lamb[np.any(indok_bool, axis=0)]),
        ]
        domain['phi']['minmax'] = [
            np.nanmin(phi[np.any(indok_bool, axis=0)]),
            np.nanmax(phi[np.any(indok_bool, axis=0)]),
        ]

    # --------------
    # Optionnal 2d binning
    (
        lambbin, phibin, databin, indok, binning,
        phi1d, lamb1d, dataphi1d, datalamb1d,
    ) = binning_2d_data(
        lamb, phi, data,
        indok=indok,
        indok_bool=indok_bool,
        binning=binning,
        domain=domain,
        nbsplines=nbsplines,
        phi1d=phi1d, lamb1d=lamb1d,
        dataphi1d=dataphi1d, datalamb1d=datalamb1d,
    )
    indok_bool = indok == 0

    # --------------
    # Optionally fit only on subset
    # randomly pick subset indices (replace=False => no duplicates)
    # indok_bool = _get_subset_indices(subset, indok == 0)

    # --------------
    # Optionally extract 1d spectra at lphi
    lphi_spectra, lphi_lamb = _extract_lphi_spectra(
        data, phi, lamb,
        lphi, lphi_tol,
        databin=databin,
        binning=binning,
    )

    if np.sum(indok_bool) == 0:
        msg = "There does not seem to be any usable data (no indok)"
        raise Exception(msg)

    # --------------
    # Return
    dprepare = {
        'data': databin, 'lamb': lambbin, 'phi': phibin,
        'domain': domain, 'binning': binning,
        'indok': indok, 'indok_bool': indok_bool, 'dindok': dict(_DINDOK),
        'pos': pos, 'subset': subset, 'nxi': nxi, 'nxj': nxj,
        'lphi': lphi, 'lphi_tol': lphi_tol,
        'lphi_spectra': lphi_spectra, 'lphi_lamb': lphi_lamb,
        'phi1d': phi1d, 'dataphi1d': dataphi1d,
        'lamb1d': lamb1d, 'datalamb1d': datalamb1d,
    }
    if return_raw_data:
        dprepare['data_raw'] = data
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

        phimargin = (phimax - phimin)/1000.
        if symmetryaxis is False:
            knots = np.linspace(
                phimin - phimargin,
                phimax + phimargin,
                nbsplines + 1 - deg,
            )
        else:
            phi2max = np.max(
                np.abs(np.r_[phimin, phimax][None, :] - symmetryaxis[:, None])
            )
            knots = np.linspace(0, phi2max + phimargin, nbsplines + 1 - deg)

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
           + "\t- a np.array of shape (2, N) or (N, 2) (focus + halfwidth)"
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
    if isinstance(focus, tuple([str] + _LTYPES)):
        focus = [focus]

    lc = [
        isinstance(focus, (list, tuple, np.ndarray))
        and all([
            (isinstance(ff, tuple(_LTYPES)) and ff > 0.)
            or (isinstance(ff, str) and ff in lines_keys)
            for ff in focus
        ]),
        isinstance(focus, (list, tuple, np.ndarray))
        and all([
            isinstance(ff, (list, tuple, np.ndarray))
            for ff in focus
        ])
        and np.asarray(focus).ndim == 2
        and 2 in np.asarray(focus).shape
        and np.all(np.isfinite(focus))
        and np.all(np.asarray(focus) > 0)
    ]
    if not any(lc):
        msg = _dvalid_checkfocus_errmsg(
            focus, focus_half_width, lines_keys,
        )
        raise Exception(msg)

    # Centered on lines
    if lc[0]:

        focus = np.array([
            lines_lamb[(lines_keys == ff).nonzero()[0][0]]
            if isinstance(ff, str) else ff for ff in focus
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

        focus = np.array([focus, np.r_[focus_half_width]]).T

    elif lc[1]:
        focus = np.asarray(focus, dtype=float)
        if focus.shape[1] != 2:
            focus = focus.T

    return focus


def fit12d_dvalid(
    data=None, lamb=None, phi=None,
    indok_bool=None, binning=None,
    valid_nsigma=None, valid_fraction=None,
    focus=None, focus_half_width=None,
    lines_keys=None, lines_lamb=None, dphimin=None,
    nbs=None, deg=None,
    knots=None, knots_mult=None, nknotsperbs=None,
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
        focus=focus,
        focus_half_width=focus_half_width,
        lines_keys=lines_keys,
        lines_lamb=lines_lamb,
        lamb=lamb,
    )

    # Get indices of pts with enough signal
    ind = np.zeros(data.shape, dtype=bool)
    isafe = np.isfinite(data)
    isafe[isafe] = data[isafe] >= 0.
    if indok_bool is not None:
        isafe &= indok_bool

    # Ok with and w/o binning if data provided as counts
    if binning is False:
        ind[isafe] = np.sqrt(data[isafe]) > valid_nsigma
    else:
        # For S/N in binning, if counts => sum = mean * nbperbin
        ind[isafe] = (
            np.sqrt(data[isafe] * binning['nperbin'][isafe]) > valid_nsigma
        )

    # Derive indt and optionally dphi and indknots
    indbs, ldphi = False, False
    if focus is False:
        lambok = np.ones(tuple(np.r_[lamb.shape, 1]), dtype=bool)
        indall = ind[..., None]
    else:
        # TBC
        lambok = np.rollaxis(
            np.array([np.abs(lamb - ff[0]) < ff[1] for ff in focus]),
            0,
            lamb.ndim + 1,
        )
        indall = ind[..., None] & lambok[None, ...]
    nfocus = lambok.shape[-1]

    if data2d is True:
        # Code ok with and without binning :-)

        # Get knots intervals that are ok
        fract = np.full((nspect, knots.size-1, nfocus), np.nan)
        for ii in range(knots.size - 1):
            iphi = (phi >= knots[ii]) & (phi < knots[ii + 1])
            fract[:, ii, :] = (
                np.sum(np.sum(indall & iphi[None, ..., None],
                              axis=1), axis=1)
                / np.sum(np.sum(iphi[..., None] & lambok,
                                axis=0), axis=0)
            )
        indknots = np.all(fract > valid_fraction, axis=2)

        # Deduce ldphi
        ldphi = [[] for ii in range(nspect)]
        for ii in range(nspect):
            for jj in range(indknots.shape[1]):
                if indknots[ii, jj]:
                    if jj == 0 or not indknots[ii, jj-1]:
                        ldphi[ii].append([knots[jj]])
                    if jj == indknots.shape[1] - 1:
                        ldphi[ii][-1].append(knots[jj+1])
                else:
                    if jj > 0 and indknots[ii, jj-1]:
                        ldphi[ii][-1].append(knots[jj])

        # Safety check
        assert all([
            all([len(dd) == 2 and dd[0] < dd[1] for dd in ldphi[ii]])
            for ii in range(nspect)
        ])

        # Deduce indbs that are ok
        nintpbs = nknotsperbs - 1
        indbs = np.zeros((nspect, nbs), dtype=bool)
        for ii in range(nbs):
            ibk = np.arange(max(0, ii-(nintpbs-1)), min(knots.size-1, ii+1))
            indbs[:, ii] = np.any(indknots[:, ibk], axis=1)

        assert np.all(
            (np.sum(indbs, axis=1) == 0) | (np.sum(indbs, axis=1) >= deg + 1)
        )

        # Deduce indt
        indt = np.any(indbs, axis=1)

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
            + f"\t- fract max, mean = {np.max(fract), np.mean(fract)}\n"
            + "\t- fract = {}\n".format(fract)
        )
        raise Exception(msg)

    # return
    dvalid = {
        'indt': indt, 'ldphi': ldphi, 'indbs': indbs, 'ind': ind,
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
        ind = np.zeros((len(lines_keys),), dtype=bool)
        for ss in domain['lamb']['spec']:
            if isinstance(ss, (list, np.ndarray)):
                ind[(lines_lamb >= ss[0]) & (lines_lamb < ss[1])] = True
        for ss in domain['lamb']['spec']:
            if isinstance(ss, tuple):
                ind[(lines_lamb >= ss[0]) & (lines_lamb < ss[1])] = False
        lines_keys = lines_keys[ind]
        lines_lamb = lines_lamb[ind]
    inds = np.argsort(lines_lamb)
    lines_keys, lines_lamb = lines_keys[inds], lines_lamb[inds]
    nlines = lines_lamb.size
    dlines = {k0: dict(dlines[k0]) for k0 in lines_keys}

    # Warning if no lines left
    if len(lines_keys) == 0:
        msg = "There seems to be no lines left!"
        warnings.warn(msg)

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
    update_domain=None,
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
            update_domain=update_domain,
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
        indok_bool=dprepare['indok_bool'],
        valid_nsigma=valid_nsigma,
        valid_fraction=valid_fraction,
        focus=focus, focus_half_width=focus_half_width,
        lines_keys=lines_keys, lines_lamb=lines_lamb,
        return_fract=valid_return_fract,
    )

    # Update with dprepare
    dinput['dprepare'] = dict(dprepare)

    # Add dind
    dinput['dind'] = multigausfit12d_from_dlines_ind(dinput)

    # Add dscales, dx0 and dbounds
    dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    dinput['dconstants'] = fit12d_dconstants(
        dconstants=dconstants, dinput=dinput,
    )

    # add lambmin for bck
    dinput['lambmin_bck'] = np.min(dprepare['lamb'])
    return dinput


def fit2d_dinput(
    dlines=None, dconstraints=None, dconstants=None, dprepare=None,
    deg=None, nbsplines=None, knots=None,
    data=None, lamb=None, phi=None, mask=None,
    domain=None, pos=None, subset=None, binning=None, cent_fraction=None,
    update_domain=None,
    focus=None, valid_fraction=None, valid_nsigma=None, focus_half_width=None,
    valid_return_fract=None,
    dscales=None, dx0=None, dbounds=None,
    nxi=None, nxj=None,
    lphi=None, lphi_tol=None,
    defconst=_DCONSTRAINTS,
    return_raw_data=None,
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
            update_domain=update_domain,
            nbsplines=nbsplines, deg=deg,
            nxi=nxi, nxj=nxj,
            lphi=None, lphi_tol=None,
            return_raw_data=return_raw_data,
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
        indok_bool=dprepare['indok_bool'],
        valid_nsigma=valid_nsigma,
        valid_fraction=valid_fraction,
        focus=focus, focus_half_width=focus_half_width,
        lines_keys=lines_keys, lines_lamb=lines_lamb,
        nbs=dinput['nbs'],
        deg=dinput['deg'],
        knots=dinput['knots'],
        knots_mult=dinput['knots_mult'],
        nknotsperbs=dinput['nknotsperbs'],
        return_fract=valid_return_fract,
    )

    # Update with dprepare
    dinput['dprepare'] = dict(dprepare)

    # Add dind
    dinput['dind'] = multigausfit12d_from_dlines_ind(dinput)

    # Add dscales, dx0 and dbounds
    dinput['dscales'] = fit12d_dscales(dscales=dscales, dinput=dinput)
    dinput['dbounds'] = fit12d_dbounds(dbounds=dbounds, dinput=dinput)
    dinput['dx0'] = fit12d_dx0(dx0=dx0, dinput=dinput)
    dinput['dconstants'] = fit12d_dconstants(
        dconstants=dconstants, dinput=dinput,
    )

    # Update indok with non-valid phi
    # non-valid = ok but out of dphi
    for ii in range(dinput['dprepare']['indok'].shape[0]):
        iphino = dinput['dprepare']['indok'][ii, ...] == 0
        for jj in range(len(dinput['valid']['ldphi'][ii])):
            iphino &= (
                (
                    dinput['dprepare']['phi']
                    < dinput['valid']['ldphi'][ii][jj][0]
                )
                | (
                    dinput['dprepare']['phi']
                    >= dinput['valid']['ldphi'][ii][jj][1]
                )
            )

        # valid, but excluded (out of dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (dinput['valid']['ind'][ii, ...])
            & (iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -5

        # non-valid, included (in dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (~dinput['valid']['ind'][ii, ...])
            & (~iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -6

        # non-valid, excluded (out of dphi)
        iphi = (
            (dinput['dprepare']['indok'][ii, ...] == 0)
            & (~dinput['valid']['ind'][ii, ...])
            & (iphino)
        )
        dinput['dprepare']['indok'][ii, iphi] = -7

    # indok_bool True if indok == 0 or -5 (because ...)
    dinput['dprepare']['indok_bool'] = (
        (dinput['dprepare']['indok'] == 0)
        | (dinput['dprepare']['indok'] == -6)
    )

    # add lambmin for bck
    dinput['lambmin_bck'] = np.min(dinput['dprepare']['lamb'])
    return dinput


###########################################################
###########################################################
#
#           dind dict (indices storing for fast access)
#
###########################################################
###########################################################


def multigausfit12d_from_dlines_ind(dinput=None):
    """ Return the indices of quantities in x to compute y """

    # indices
    # General shape: [bck, amp, widths, shifts]
    # If double [..., double_shift, double_ratio]
    # Except for bck, all indices should render nlines (2*nlines if double)
    nbs = dinput.get('nbs', 1)
    dind = {
        'bck_amp': {'x': np.arange(0, nbs)[:, None]},
        'bck_rate': {'x': np.arange(nbs, 2*nbs)[:, None]},
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
    nvar_bs = 2 + np.sum([dinput[k0]['ind'].shape[0] for k0 in _DORDER])
    indx = np.r_[
        dind['bck_amp']['x'].ravel(order='F'),
        dind['bck_rate']['x'].ravel(order='F'),
        dind['amp']['x'].ravel(order='F'),
        dind['width']['x'].ravel(order='F'),
        dind['shift']['x'].ravel(order='F'),
    ]
    assert np.allclose(np.arange(0, sizex), indx)
    assert nvar_bs == sizex / nbs

    # check if double
    if dinput['double'] is True:
        dind['dshift'] = {'x': np.r_[-2][:, None]}
        dind['dratio'] = {'x': np.r_[-1][:, None]}
        sizex += 2
    elif isinstance(dinput['double'], dict):
        if dinput['double'].get('dshift') is None:
            dind['dshift'] = {'x': np.r_[-1][:, None]}
            sizex += 1
        elif dinput['double'].get('dratio') is None:
            dind['dratio'] = {'x': np.r_[-1][:, None]}
            sizex += 1

    dind['nvar_bs'] = nvar_bs      # nb of spectral variable with bs dependence
    dind['sizex'] = sizex
    dind['nbck'] = 2

    # Ref line for amp (for x0)
    # TBC !!!
    amp_x0 = np.zeros((dinput['amp']['ind'].shape[0],), dtype=int)
    for ii in range(dinput['amp']['ind'].shape[0]):
        indi = dinput['amp']['ind'][ii, :].nonzero()[0]
        if indi.size == 0:
            import pdb; pdb.set_trace()     # DB
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
    if not isinstance(din, dict):
        msg = f"Arg {name} must be a dict!"
        raise Exception(msg)

    lkfalse = [
        k0 for k0, v0 in din.items()
        if not (
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
    ]

    if len(lkfalse) > 0:
        msg = (
            f"Arg {name} must be a dict of the form:\n"
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


def _check_finit_dict(dd=None, dd_name=None, indtok=None, indbs=None):
    dfail = {}
    for k0, v0 in dd.items():
        if k0 in ['amp', 'width', 'shift']:
            for k1, v1 in v0.items():
                if np.any(~np.isfinite(v1[indtok, ...])):
                    dfail[f"'{k0}'['{k1}']"] = v1
        elif k0 == 'bs':
            if np.any(~np.isfinite(v0[indbs])):
                dfail[f"'{k0}'"] = v0
        else:
            if np.any(~np.isfinite(v0[indtok, ...])):
                dfail[f"'{k0}'"] = v0

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            f"The following {dd_name} values are non-finite:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)


# Double-check 1d vs 2d: TBF / TBC
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
        indok = np.any(dinput['dprepare']['indok_bool'], axis=1)

        # bsplines modulation of bck and amp, if relevant
        # fit bsplines on datavert (vertical profile)
        # to modulate scales (bck and amp)

        if dinput['symmetry'] is True:
            phitemp = np.abs(phi[None, :] - dinput['symmetry_axis'][:, None])
        else:
            phitemp = np.tile(phi, (nspect, 1))

        # Loop on time and bsplines
        dscales['bs'] = np.full((nspect, dinput['nbs']), np.nan)
        for ii in dinput['valid']['indt'].nonzero()[0]:
            for jj, jbs in enumerate(range(dinput['nbs'])):
                if dinput['valid']['indbs'][ii, jj]:
                    kn0 = dinput['knots_mult'][jj]
                    kn1 = dinput['knots_mult'][jj + dinput['nknotsperbs'] - 1]
                    indj = (
                        (~np.isnan(datavert[ii, :]))
                        & (kn0 <= phitemp[ii, :])
                        & (phitemp[ii, :] <= kn1)
                    )
                    if not np.any(indj):
                        msg = "Unconsistent indbs!"
                        raise Exception(msg)
                    dscales['bs'][ii, jj] = np.mean(datavert[ii, indj])

        # Normalize to avoid double-amplification when amp*bs
        corr = np.nanmax(dscales['bs'][dinput['valid']['indt'], :], axis=1)
        dscales['bs'][dinput['valid']['indt'], :] /= corr[:, None]
    else:
        indok = dinput['dprepare']['indok_bool']

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
            + "\n  => Please provide domain['lamb']"
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

        iok = (bcky > 0) & (bckstd > 0)
        if (bck_rate is None or nbck_amp is None) and not np.any(iok):
            bcky = 0.1*np.array(np.ma.masked_where(~indbck, data).mean(axis=1))
            bckstd = 0.1*bcky
        elif not np.all(iok):
            bcky[~iok] = np.mean(bcky[iok])
            bckstd[~iok] = np.mean(bckstd[iok])

        # bck_rate
        if bck_rate is None:
            bck_rate = (
                np.log((bcky + bckstd)/bcky) / (lamb.max()-lamb.min())
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
            # convoluate and estimate geometric mean
            conv = np.exp(
                    -(lamb - dinput['lines'][ij])**2 / (2*(Dlamb / 25.)**2)
                )[None, :]
            dscales['amp'][key] = np.nanmax(data*conv, axis=1)
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
        widthref = (Dlamb/(25*lambmw))**2
    else:
        widthref = (Dlamb/(25*lambm))**2

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
        dshift = float(Dlamb/(25*lambm))
        if dinput['double'] is True:
            pass
        else:
            if dinput['double'].get('dratio') is not None:
                dratio = dinput['double']['dratio']
            if dinput['double'].get('dshift') is not None:
                dshift = dinput['double']['dshift']
        din = {'dratio': dratio, 'dshift': dshift}
        for k0 in din.keys():
            dscales = _fit12d_filldef_dscalesx0_float(
                din=dscales, din_name='dscales', key=k0,
                vref=din[k0], nspect=nspect,
            )
    elif 'dratio' in dscales.keys():
        del dscales['dratio'], dscales['dshift']

    # check
    _check_finit_dict(
        dd=dscales,
        dd_name='dscales',
        indtok=dinput['valid']['indt'],
        indbs=dinput['valid']['indbs'],
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

    # no check: dconstant can be nan if indx not used
    # _check_finit_dict(dd=dconstants, dd_name='dconstants')

    return dconstants


###########################################################
###########################################################
#
#           dict to vector (scales, x0, bounds)
#
###########################################################
###########################################################


def _dict2vector_dscalesx0bounds(
    dd=None,
    dd_name=None,
    dinput=None,
):
    nspect = dinput['dprepare']['data'].shape[0]
    x = np.full((nspect, dinput['dind']['sizex']), np.nan)

    # 1d => (1, nvar)
    # 2d => (nbs, nvar)
    x[:, dinput['dind']['bck_amp']['x'][:, 0]] = dd['bck_amp'][:, None]
    x[:, dinput['dind']['bck_rate']['x'][:, 0]] = dd['bck_rate'][:, None]
    for k0 in _DORDER:
        for ii, k1 in enumerate(dinput[k0]['keys']):
            # 1d => 'x' (nlines,)
            # 2d => 'x' (nbs, nlines)
            x[:, dinput['dind'][k0]['x'][:, ii]] = dd[k0][k1][:, None]

    if dinput['double'] is not False:
        if dinput['double'] is True:
            x[:, dinput['dind']['dratio']['x'][:, 0]] = dd['dratio'][:, None]
            x[:, dinput['dind']['dshift']['x'][:, 0]] = dd['dshift'][:, None]
        else:
            for kk in ['dratio', 'dshift']:
                if dinput['double'].get(kk) is None:
                    x[:, dinput['dind'][kk]['x'][:, 0]] = dd[kk][:, None]

    if dd_name != 'dconstants' and not np.all(np.isfinite(x)):
        msg = (
            f"dict {dd_name} seems to have non-finite values!\n"
            f"\t- x: {x}"
        )
        raise Exception(msg)
    return x
