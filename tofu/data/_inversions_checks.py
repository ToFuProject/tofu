# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import scipy.sparse as scpsp


# specific
from . import _generic_check


# optional
try:
    from .. import tomotok2tofu
except Exception as err:
    tomotok2tofu = False


_DALGO = {
    'algo0': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_augTikho_sparse',
    },
    'algo1': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': False,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_dense',
    },
    'algo2': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': 'cholesky',
        'positive': False,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_chol_dense',
    },
    'algo3': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': 'cholesky',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_augTikho_chol_sparse',
    },
    'algo4': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': True,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_pos_dense',
    },
    'algo5': {
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'DisPrinc',
        'decomposition': '',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_DisPrinc_sparse',
    },
}
_LREGPARAM_ALGO = [
    'augTikho',
    'DisPrinc',
]


# #############################################################################
# #############################################################################
#                           info
# #############################################################################


def get_available_inversions_algo(
    # for filtering
    source=None,
    family=None,
    reg_operator=None,
    reg_param=None,
    decomposition=None,
    positive=None,
    sparse=None,
    isotropic=None,
    dalgo=None,
    # parameters
    returnas=None,
    verb=None,
):

    # --------------
    # check inputs

    returnas = _generic_check._check_var(
        returnas,
        'returnas',
        default=False,
        allowed=[False, dict, list, str]
    )

    verb = _generic_check._check_var(
        verb,
        'verb',
        default=returnas is False,
        types=bool,
    )

    # ------------------------------------
    # filter according to criteria, if any

    dalgo = match_algo(
        # filtering
        source=source,
        family=family,
        reg_operator=reg_operator,
        reg_param=reg_param,
        decomposition=decomposition,
        positive=positive,
        sparse=sparse,
        isotropic=isotropic,
    )

    # ------------
    # print or str

    if verb is True or returnas is str:

        head = ['key'] + [
            'source', 'family', 'reg_operator', 'reg_param', 'decomposition',
            'positive', 'sparse',
        ]
        sep = ['-'*len(kk) for kk in head]
        lstr = [head, sep] + [
            [k0] + [v0[k1] for k1 in head[1:]]
            for k0, v0 in dalgo.items()
        ]

        nmax = np.max(np.char.str_len(np.char.array(lstr)), axis=0)
        lstr = [
            '  '.join([str(ss).ljust(nmax[ii]) for ii, ss in enumerate(line)])
            for line in lstr
        ]
        msg = "\n".join(lstr)

        if verb:
            print(msg)

    # -------
    # return

    if returnas is dict:
        return dalgo
    elif returnas is list:
        return sorted(dalgo.keys())
    elif returnas is str:
        return msg


def match_algo(
    source=None,
    family=None,
    reg_operator=None,
    reg_param=None,
    decomposition=None,
    positive=None,
    sparse=None,
    isotropic=None,
):

    # ------------
    # check inputs

    dargs = {
        k0: v0 for k0, v0 in locals().items()
        if v0 is not None
    }

    # --------------
    # Get tofu algo

    dalgo = _DALGO

    # -----------------
    # Get tomotok algo

    if tomotok2tofu is not False:
        dalgo.update(tomotok2tofu.get_dalgo())

    # ------------
    # find matches

    if len(dargs) > 0:
        lmatch = [
            k0 for k0, v0 in dalgo.items()
            if all([v0[k1] == v1 for k1, v1 in dargs.items()])
        ]
        if len(lmatch) == 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in dargs.items()]
            msg = (
                "No / several algorithms matching the following criteria:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)
        dalgo = {k0: v0 for k0, v0 in dalgo.items() if k0 in lmatch}

    return copy.deepcopy(dalgo)


# #############################################################################
# #############################################################################
#                           main
# #############################################################################


def _compute_check(
    # input data
    coll=None,
    key_matrix=None,
    key_data=None,
    key_sigma=None,
    data=None,
    sigma=None,
    # regularity operator
    solver=None,
    operator=None,
    geometry=None,
    # choice of algo
    algo=None,
    # misc
    conv_crit=None,
    chain=None,
    verb=None,
    store=None,
    # algo and solver-specific options
    kwdargs=None,
    method=None,
    options=None,
):

    # ----
    # keys

    # key_matrix
    lk = list(coll.dobj.get('matrix', {}).keys())
    if key_matrix is None and len(lk):
        key_matrix = lk[0]
    key_matrix = _generic_check._check_var(
        key_matrix, 'key_matrix',
        types=str,
        allowed=lk,
    )
    keybs = coll.dobj['matrix'][key_matrix]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    matrix = coll.ddata[coll.dobj['matrix'][key_matrix]['data']]['data']
    shapemat = matrix.shape
    crop = coll.dobj['matrix'][key_matrix]['crop']

    if np.any(~np.isfinite(matrix)):
        msg = "Geometry matrix should not contain NaNs or infs!"
        raise Exception(msg)

    # key_data
    if key_data is not None or (key_data is None and data is None):
        lk = [
            kk for kk, vv in coll.ddata.items()
            if vv['data'].ndim in [1, 2]
            and vv['data'].shape[-1] == shapemat[0]
        ]
        if key_data is None and len(lk):
            key_data = lk[0]
        key_data = _generic_check._check_var(
            key_data, 'key_data',
            types=str,
            allowed=lk,
        )
        data = coll.ddata[key_data]['data']

    # ------------
    # data, sigma

    # data
    data = _generic_check._check_var(
        data, 'data',
        types=(np.ndarray, list, tuple),
    )
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.ndim not in [1, 2] or shapemat[0] not in data.shape:
        msg = (
            "Arg data must have dim in [1, 2]"
            f" and {shapemat[0]} must be in shape\n"
            f"\t- data.shape: {data.shape}"
        )
        raise Exception(msg)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != shapemat[0]:
        data = data.T
    if np.any(~np.isfinite(data)):
        msg = "Arg data should not contain NaNs or inf!"
        raise Exception(msg)

    # key_sigma
    if key_sigma is not None:
        lk = [
            kk for kk, vv in coll.ddata.items()
            if vv['data'].ndim in [1, 2]
            and vv['data'].shape[-1] == shapemat[0]
        ]
        key_sigma = _generic_check._check_var(
            key_sigma, 'key_sigma',
            types=str,
            allowed=lk,
        )
        sigma = coll.ddata[key_sigma]['data']

    # sigma
    if np.isscalar(sigma):
        sigma = np.full((shapemat[0],), sigma*np.nanmean(np.abs(data)))

    sigma = _generic_check._check_var(
        sigma, 'sigma',
        default=np.full((shapemat[0],), 0.05*np.nanmean(np.abs(data))),
        types=(np.ndarray, list, tuple),
    )

    if not isinstance(sigma, np.ndarray):
        sigma = np.asarray(sigma)
    if sigma.ndim not in [1, 2] or shapemat[0] not in sigma.shape:
        msg = (
            "Arg sigma must have dim in [1, 2]"
            f" and {shapemat[0]} must be in shape\n"
            f"\t- sigma.shape = {sigma.shape}"
        )
        raise Exception(msg)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    elif sigma.ndim == 2 and data.shape != sigma.shape:
        msg = (
            "Arg sigma must have the same shape as data!\n"
            f"\t- data.shape: {data.shape}\n"
            f"\t- sigma.shape: {sigma.shape}\n"
        )
        raise Exception(msg)
    if sigma.shape[1] != shapemat[0]:
        sigma = sigma.T

    if np.any(~np.isfinite(sigma)):
        msg = "Arg sigma should not contain NaNs or inf!"
        raise Exception(msg)

    # -------------------
    # regularity operator

    # get operator
    opmat, operator, geometry, dim, ref, crop = coll.add_bsplines_operator(
        key=keybs,
        operator=operator,
        geometry=geometry,
        returnas=True,
        store=False,
        crop=crop,
    )

    nchan, nbs = matrix.shape
    if isinstance(opmat, tuple):
        assert all([op.shape == (nbs, nbs) for op in opmat])
    elif opmat.ndim == 1:
        msg = "Inversion algorithm requires a quadratic operator!"
        raise Exception(msg)
    else:
        assert opmat.shape == (nbs,) or opmat.shape == (nbs, nbs)
        opmat = (opmat,)

    if not scpsp.issparse(opmat[0]):
        assert all([np.all(np.isfinite(op)) for op in opmat])

    assert data.shape[1] == nchan
    nt = data.shape[0]

    # --------------
    # choice of algo

    lok = list(_DALGO.keys())
    algo = _generic_check._check_var(
        algo, 'algo',
        default='algo1',
        types=str,
        allowed=lok,
    )

    # define dalgo
    dalgo = _DALGO[algo]
    dalgo['name'] = algo

    # check vs deg
    deg = coll.dobj['bsplines'][keybs]['deg']
    if dalgo['source'] == 'tomotok' and dalgo['reg_operator'] == 'MinFisher':
        if deg != 0:
            msg = (
                "MinFisher regularization from tomotok requires deg = 0\n"
                f"\t- deg: {deg}"
            )
            raise Exception(msg)

    # -------------------
    # consistent sparsity

    # sparse
    if dalgo['sparse'] is True:
        if not scpsp.issparse(matrix):
            matrix = scpsp.csc_matrix(matrix)
        if not scpsp.issparse(opmat[0]):
            opmat = [scpsp.csc_matrix(pp) for pp in opmat]
    elif dalgo['sparse'] is False:
        if scpsp.issparse(matrix):
            matrix = matrix.toarray()
        if scpsp.issparse(opmat[0]):
            opmat = [scpsp.csc_matrix(pp).toarray() for pp in opmat]

    # -----------------------
    # miscellaneous parameter

    # conv_crit
    conv_crit = _generic_check._check_var(
        conv_crit, 'conv_crit',
        default=1e-4,
        types=float,
    )

    # chain
    chain = _generic_check._check_var(
        chain, 'chain',
        default=True,
        types=bool,
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=(bool, int),
        allowed=[False, 0, True, 1, 2],
    )
    if verb is False:
        verb = 0
    if verb is True:
        verb = 1

    # store
    store = _generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )
    if key_data is None:
        store = False

    # solver
    solver = _generic_check._check_var(
        solver, 'solver',
        default='spsolve',
        types=str,
        allowed=['spsolve'],
    )

    # ----------------------------------------
    # algo-specific kwdargs and solver options

    # kwdargs, method, options
    kwdargs, method, options = _algo_check(
        dalgo,
        kwdargs=kwdargs,
        options=options,
        nchan=shapemat[0],
        nbs=shapemat[1],
        conv_crit=conv_crit,
    )

    return (
        key_matrix, key_data, key_sigma, keybs, keym,
        data, sigma, matrix, opmat, operator, geometry,
        dalgo,
        conv_crit, crop, chain, kwdargs, method, options,
        solver, verb, store,
    )


# #############################################################################
# #############################################################################
#                  ikwdargs / options for each algo
# #############################################################################


def _algo_check(
    dalgo,
    kwdargs=None,
    method=None,
    options=None,
    nchan=None,
    nbs=None,
    conv_crit=None,
):

    # ------------------------
    # generic kwdargs

    # kwdargs
    if kwdargs is None:
        kwdargs = {}

    # generic kwdargs
    if kwdargs.get('maxiter') is None:
        kwdargs['maxiter'] = 100

    if kwdargs.get('tol') is None:
        kwdargs['tol'] = 1.e-6

    # ------------------------
    # algo specific kwdargs

    # kwdargs specific to aug. tikhonov
    if dalgo['reg_param'] == 'augTikho':
        a0 = kwdargs.get('a0', 10)
        a1 = kwdargs.get('a1', 2)

        # to have [x]=1
        kwdargs['b0'] = np.math.factorial(a0)**(1 / (a0 + 1))
        kwdargs['b1'] = np.math.factorial(a1)**(1 / (a1 + 1))
        kwdargs['a0'] = a0
        kwdargs['a1'] = a1

        # Exponent for rescaling of a0bis
        # typically in [1/3 ; 1/2], but real limits are 0 < d < 1 (or 2 ?)
        if kwdargs.get('d') is None:
            kwdargs['d'] = 0.95

        if kwdargs.get('conv_reg') is None:
            kwdargs['conv_reg'] = True

        if kwdargs.get('nbs_fixed') is None:
            kwdargs['nbs_fixed'] = True

        if kwdargs['nbs_fixed']:
            kwdargs['a0bis'] = kwdargs['a0'] - 1. + 1200./2.
        else:
            kwdargs['a0bis'] = kwdargs['a0'] - 1. + nbs/2.
        kwdargs['a1bis'] = kwdargs['a1'] - 1. + nchan/2.

    # kwdargs specific to discrepancy principle
    elif dalgo['reg_param'] == 'DisPrinc':
        if kwdargs.get('chi2n_obj') is None:
            kwdargs['chi2n_obj'] = 1.
        if kwdargs.get('chi2n_tol') is None:
            kwdargs['chi2n_tol'] = 0.05

    # ------------------------
    # low-level solver options

    if dalgo['positive'] is True:

        if options is None:
            options = {}

        if method is None:
            method = 'L-BFGS-B'

        if method == 'L-BFGS-B':
            if options.get('ftol') is None:
                options['ftol'] = conv_crit/100.
            if options.get('disp') is None:
                options['disp'] = False
        else:
            raise NotImplementedError

    return kwdargs, method, options


# ##################################################################
# ##################################################################
#               _DALGO at import time
# ##################################################################


_DALGO = get_available_inversions_algo(returnas=dict, verb=False)
