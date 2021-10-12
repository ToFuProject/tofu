# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.sparse as scpsp


# specific
from . import _generic_check


_LALGO = [
    'inv_linear_augTikho_sparse',
    'inv_linear_augTikho_dense',
    'inv_linear_augTikho_chol_dense',
    'inv_linear_augTikho_chol_sparse',
    'inv_linear_augTikho_pos_dense',
    'inv_linear_DisPrinc_sparse',
]
_LREGPARAM_ALGO = [
    'augTikho',
    'DisPrinc',
]



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
    # choice of algo
    isotropic=None,
    sparse=None,
    positive=None,
    cholesky=None,
    regparam_algo=None,
    algo=None,
    # regularity operator
    solver=None,
    operator=None,
    geometry=None,
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

    # --------------
    # choice of algo

    lc = [
        algo is None,
        all([kk is None for kk in [isotropic, positive, sparse, cholesky]])
    ]

    if not any(lc):
        msg = (
            "Please provide either (xor):\n"
            "\t- algo: directly provide the algo name\n"
            "\t- flags for choosing the algo:\n"
            "\t\t- isotropic: whether to perform isotropic regularization\n"
            "\t\t- sparse: whether to use sparse matrices\n"
            "\t\t- positive: whether to enforce a positivity constraint\n"
            "\t\t- cholesky: whether to use cholesky factorization\n"
        )
        raise Exception(msg)

    if all(lc):
        algo = 'inv_linear_augTikho_sparse'
        lc[0] = False

    if not lc[0] and lc[1]:
        # extract keywrods from algo name
        isotropic = True
        positive = 'pos' in algo
        sparse = 'sparse' in algo
        cholesky = 'chol' in algo

        for aa in _LREGPARAM_ALGO:
            if f'_{aa}_' in algo:
                regparam_algo = aa
                break
        else:
            msg = 'Unreckognized algo for regularization parameter!'
            raise Exception(msg)

    elif lc[0] and not lc[1]:
        # get algo name from keywords

        # isotropic
        isotropic = _generic_check._check_var(
            isotropic, 'isotropic',
            default=True,
            types=bool,
        )
        if isotropic is False:
            msg = "Anisotropic regularization unavailable yet"
            raise NotImplementedError(msg)

        # sparse and matrix and operator
        sparse = _generic_check._check_var(
            sparse, 'sparse',
            default=True,
            types=bool,
        )

        # positive
        positive = _generic_check._check_var(
            positive, 'positive',
            default=False,
            types=bool,
        )

        # cholesky
        cholesky = _generic_check._check_var(
            cholesky, 'cholesky',
            default=False,
            types=bool,
        )
        if positive and cholesky is False:
            msg = "cholesky cannot be used for positive constraint!"
            raise Exception(msg)

        # regparam_algo
        regparam_algo = _generic_check._check_var(
            regparam_algo, 'regparam_algo',
            default='augTikho',
            types=str,
            allowed=_LREGPARAM_ALGO,
        )

        algo = f"inv_linear_{regparam_algo}"
        if cholesky:
            algo += '_chol'
        elif positive:
            algo += '_pos'
        algo += f"_{'sparse' if sparse else 'dense'}"

    # final algo check
    algo = _generic_check._check_var(
        algo, 'algo',
        default=None,
        types=str,
        allowed=_LALGO,
    )

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

    # -------------------
    # consistent sparsity

    # sparse
    if sparse is True:
        if not scpsp.issparse(matrix):
            matrix = scpsp.csc_matrix(matrix)
        if not scpsp.issparse(opmat[0]):
            opmat = [scpsp.csc_matrix(pp) for pp in opmat]
    elif sparse is False:
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
        algo,
        kwdargs=kwdargs,
        options=options,
        nchan=shapemat[0],
        nbs=shapemat[1],
        conv_crit=conv_crit,
    )

    return (
        key_matrix, key_data, key_sigma, keybs, keym,
        data, sigma, matrix, opmat, operator, geometry,
        isotropic, sparse, positive, cholesky, regparam_algo, algo,
        conv_crit, crop, chain, kwdargs, method, options,
        solver, verb, store,
    )


# #############################################################################
# #############################################################################
#                  ikwdargs / options for each algo
# #############################################################################


def _algo_check(
    algo,
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
    if 'augTikho' in algo:
        if kwdargs.get('a0') is None:
            kwdargs['a0'] = 10
        if kwdargs.get('a1') is None:
            kwdargs['a1'] = 2

        # to have [x]=1
        kwdargs['b0'] = np.math.factorial(kwdargs['a0'])**(1 / (kwdargs['a0'] + 1))
        kwdargs['b1'] = np.math.factorial(kwdargs['a1'])**(1 / (kwdargs['a1'] + 1))

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
    elif 'DisPrinc' in algo:
        if kwdargs.get('chi2n_obj') is None:
            kwdargs['chi2n_obj'] = 1.
        if kwdargs.get('chi2n_tol') is None:
            kwdargs['chi2n_tol'] = 0.05

    # ------------------------
    # low-level solver options

    if 'quad' in algo:

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
