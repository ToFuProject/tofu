# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


def _mesh2DRect_X_check(
    x=None,
    res=None,
):
    """ Returns knots (x) and associated resolution

    res can be:
        - int: numbr of mesh elements desired between knots
        - float: desired average mesh element size
        - array of floats: (one for each x, desired approximate mesh size)

    """


    # ------------
    # Check inputs

    # x
    try:
        x = np.unique(np.ravel(x).astype(float))
    except Exception as err:
        msg = "x must be convertible to a sorted, flat array of floats!"
        raise Exception(msg)

    # res
    if res is None:
        res = 10

    lc = [
        isinstance(res, (int, np.int_)) and len(x) == 2,
        isinstance(res, (float, np.float_)) and len(x) == 2,
        isinstance(res, (list, tuple, np.ndarray)) and len(x) == len(res),
    ]
    if not any(lc):
        msg = (
            "Arg res must be:\n"
            "\t- int: nb of mesh elements along x\n"
            "\t       requires len(x) = 2\n"
            "\t- float: approximate desired mesh element size along x\n"
            "\t       requires len(x) = 2\n"
            "\t- iterable: approximate desired mesh element sizes along x\n"
            "\t       requires len(x) = len(res)\n"
        )
        raise Exception(msg)

    if lc[0]:
        x_new = np.linspace(x[0], x[1], int(res)+1)
        res_new = res
        indsep = None

    elif lc[1]:
        nb = int(np.ceil((x[1]-x[0]) / res))
        x_new = np.linspace(x[0], x[1], nb+1)
        res_new = np.mean(np.diff(x))
        indsep = None

    else:

        # check conformity
        res = np.ravel(res).astype(float)
        delta = np.diff(x)
        res_sum = res[:-1] + res[1:]
        ind = res_sum > delta
        if np.any(ind):
           msg = (
               "Desired resolution is not achievable for the following:\n"
               f"res_sum: {res_sum[ind]}\n"
               f"delta  : {delta[ind]}"
           )
           raise Exception(msg)

        # compute nn
        # nn = how many pairs can fit in the interval
        npairs = np.round(delta/res_sum).astype(int)
        res_sum_new = delta / npairs

        fract = res[:-1] / res_sum

        res_new = [None for ii in range(len(x)-1)]
        x_new = [None for ii in range(len(x)-1)]
        for ii in range(len(x)-1):
            res_new[ii] = (
                res_sum_new[ii]
                * np.linspace(fract[ii], 1.-fract[ii], 2*npairs[ii])
            )
            if ii == 0:
                res_add = np.concatenate(([0], np.cumsum(res_new[ii])))
            else:
                res_add = np.cumsum(res_new[ii])
            x_new[ii] = x[ii] + res_add

        indsep = np.cumsum(npairs[:-1]*2)
        res_new = np.concatenate(res_new)
        x_new = np.concatenate(x_new)

    return x_new, res_new, indsep



def _mesh2DRect_check(
    domain=None,
    res=None,
):

    # --------------
    # check inputs

    # domain
    c0 = (
        isinstance(domain, list)
        and len(domain) == 2
        and all([hasattr(dd, '__iter__') and len(dd) >= 2 for dd in domain])
    )
    if not c0:
        msg = (
            "Arg domain must be a list of 2 iterables of len() >= 2\n"
            f"Provided: {domain}"
        )
        raise Exception(msg)

    # res
    c0 = (
        res is None
        or np.isscalar(res)
        or isinstance(res, list) and len(res) == 2
    )
    if not c0:
        msg = (
            "Arg res must be a int, float or array or a list of 2 such\n"
            f"Provided: {res}"
        )
        raise Exception(msg)

    if np.isscalar(res) or res is None:
        res = [res, res]

    # -------------
    # check R and Z

    R, resR = _mesh2DRect_X_check(domain[0], res=res[0])
    Z, resZ = _mesh2DRect_X_check(domain[1], res=res[0])

    return R, Z, resR, resZ
