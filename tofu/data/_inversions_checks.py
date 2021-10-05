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
    key=None,
    data=None,
    sigma=None,
    isotropic=None,
    sparse=None,
    chain=None,
):

    # key
    lk = list(coll.dobj.get('matrix', {}).keys())
    if key is None and len(lk):
        key = lk[0]
    key = _generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lk,
    )
    keybs = coll.dobj['matrix'][key]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    matrix = coll.ddata[coll.dobj['matrix'][key]['data']]['data']
    shapemat = matrix.shape
    crop = coll.dobj['matrix'][key]['crop']

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
            f" and {shapemat[0]} must be in shape"
        )
        raise Exception(msg)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != shapemat[0]:
        data = data.T

    # sigma
    sigmadef = np.repeat(
        0.1*np.nanmean(data, axis=1)[:, None],
        shapemat[0],
        axis=1,
    )
    sigma = _generic_check._check_var(
        sigma, 'sigma',
        default=sigmadef,
        types=(np.ndarray, list, tuple),
    )
    if not isinstance(sigma, np.ndarray):
        sigma = np.asarray(sigma)
    if sigma.ndim not in [1, 2] or shapemat[0] not in sigma.shape:
        msg = (
            "Arg sigma must have dim in [1, 2]"
            f" and {shapemat[0]} must be in shape"
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

    # isotropic
    isotropic = _generic_check._check_var(
        isotropic, 'isotropic',
        default=True,
        types=bool,
    )
    if isotropic is False:
        raise NotImplementedError("Anisotropic regularization unavailable yet")

    # sparse and matrix
    sparse = _generic_check._check_var(
        sparse, 'sparse',
        default=True,
        types=bool,
    )
    if sparse is True and not scpsp.issparse(matrix):
        matrix = scpsp.csr_matrix(matrix)
    elif sparse is False and scpsp.issparse(matrix):
        matrix = matrix.toarray()

    # chain
    chain = _generic_check._check_var(
        chain, 'chain',
        default=True,
        types=bool,
    )

    return (
        key, keybs, keym, data, sigma, isotropic, sparse, matrix, crop, chain,
    )
