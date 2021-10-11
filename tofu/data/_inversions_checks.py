# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.sparse as scpsp


# specific
from . import _generic_check


# #############################################################################
# #############################################################################
#                           main
# #############################################################################


def _compute_check(
    coll=None,
    key_matrix=None,
    key_data=None,
    key_sigma=None,
    data=None,
    sigma=None,
    conv_crit=None,
    isotropic=None,
    sparse=None,
    chain=None,
    positive=None,
    method=None,
    solver=None,
    operator=None,
    geometry=None,
    kwdargs=None,
    verb=None,
    store=None,
):

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

    # conv_crit
    conv_crit = _generic_check._check_var(
        conv_crit, 'conv_crit',
        default=1e-6,
        types=float,
    )

    # isotropic
    isotropic = _generic_check._check_var(
        isotropic, 'isotropic',
        default=True,
        types=bool,
    )
    if isotropic is False:
        raise NotImplementedError("Anisotropic regularization unavailable yet")

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

    # sparse and matrix and operator
    sparse = _generic_check._check_var(
        sparse, 'sparse',
        default=True,
        types=bool,
    )
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

    # positive
    positive = _generic_check._check_var(
        positive, 'positive',
        default=False,
        types=bool,
    )

    # method
    if positive is True:
        metdef = 'InvLinQuad_AugTikho_V1'
    else:
        if sparse:
            metdef = 'inv_linear_augTikho_chol_sparse'
        else:
            metdef = 'inv_linear_augTikho_v1'

    metok = [
        'inv_linear_augTikho_v1_sparse',
        'inv_linear_augTikho_v1',
        'inv_linear_augTikho_chol',
        'inv_linear_augTikho_chol_sparse',
        'InvLinQuad_AugTikho_V1',
        'InvQuad_AugTikho_V1',
        'InvLin_DisPrinc_V1',
    ]
    method = _generic_check._check_var(
        method, 'method',
        default=metdef,
        types=str,
        allowed=metok,
    )

    # solver
    solver = _generic_check._check_var(
        solver, 'solver',
        default='spsolve',
        types=str,
        allowed=['spsolve'],
    )

    # kwdargs
    if kwdargs is None:
        kwdargs = {}

    if len(kwdargs) == 0:
        c0 = (
            method in [
                'inv_linear_augTikho_v1_sparse',
                'inv_linear_augTikho_v1',
                'inv_linear_augTikho_chol',
                'inv_linear_augTikho_chol_sparse',
                'InvLinQuad_AugTikho_V1',
                'InvQuad_AugTikho_V1',
            ]
        )
        if c0:
            a0 = 10
            a1 = 2
            kwdargs = {
                'a0': a0,   # (Regul. parameter, larger a => larger variance)
                'b0': np.math.factorial(a0)**(1 / (a0 + 1)),    # To have [x]=1
                'a1': a1,   # (Noise), a as small as possible for small variance
                'b1': np.math.factorial(a1)**(1/(a1 + 1)),  # To have [x] = 1
                'd': 0.95,  # Exponent for rescaling of a0bis in V2, typically in [1/3 ; 1/2], but real limits are 0 < d < 1 (or 2 ?)
                'conv_reg':True,
                'nbs_fixed':True,
            }
        elif method == 'InvLin_DisPrinc_V1':
            kwdargs = {'chi2Tol': 0.05, 'chi2Obj': 1.}

    kwdargs['maxiter'] = 100

    return (
        key_matrix, key_data, key_sigma, keybs, keym, data, sigma, opmat,
        conv_crit, operator, isotropic, sparse, matrix, crop, chain,
        positive, method, solver, kwdargs, verb, store,
    )
