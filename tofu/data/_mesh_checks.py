# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# specific
from . import _generic_check


_ELEMENTS = 'knots'


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
        isinstance(res, (int, np.int64, np.int32)) and len(x) == 2,
        isinstance(res, (float, np.floating)) and len(x) == 2,
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
        ind = res_sum > delta + 1.e-14
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

    R, resR, indR = _mesh2DRect_X_check(domain[0], res=res[0])
    Z, resZ, indZ = _mesh2DRect_X_check(domain[1], res=res[1])

    return R, Z, resR, resZ, indR, indZ


def _mesh2DRect_to_dict(
    domain=None,
    res=None,
    key=None,
):

    # --------------------
    # check / format input

    if not isinstance(key, str):
        msg = "Arg key must be a str!"
        raise Exception(msg)

    kRknots, kZknots = f"{key}-R-knots", f"{key}-Z-knots"
    kRcent, kZcent = f"{key}-R-cents", f"{key}-Z-cents"

    R, Z, resR, resZ, indR, indZ = _mesh2DRect_check(domain=domain, res=res)
    Rcent = 0.5*(R[1:] + R[:-1])
    Zcent = 0.5*(Z[1:] + Z[:-1])

    variable = not (np.isscalar(resR) and np.isscalar(resZ))

    # --------------------
    # prepare dict

    dref = {
        kRknots: {
            'data': R,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'group': 'R',
        },
        kZknots: {
            'data': Z,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
        kRcent: {
            'data': Rcent,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'group': 'R',
        },
        kZcent: {
            'data': Zcent,
            'units': 'm',
            # 'source': None,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
    }

    # dobj
    dmesh = {
        key: {
            'type': 'rect',
            'knots': (kRknots, kZknots),
            'cents': (kRcent, kZcent),
            'ref': (kRcent, kZcent),
            'shape': (Rcent.size, Zcent.size),
            'variable': variable,
            'crop': False,
        },
    }
    return dref, dmesh


def _mesh2DRect_from_Config(config=None, key_struct=None):

    # ------------
    # check inputs

    if not config.__class__.__name__ == 'Config':
        msg = "Arg config must be a Config instance!"
        raise Exception(msg)

    # -------------
    # key_struct if None

    if key_struct is None:
        lk, ls = zip(*[
            (ss.Id.Name, ss.dgeom['Surf']) for ss in config.lStructIn
        ])
        key_struct = lk[np.argmin(ls)]

    # -------------
    # domain

    poly = config.dStruct['dObj']['Ves'][key_struct].Poly_closed
    domain = [
        [poly[0, :].min(), poly[0, :].max()],
        [poly[1, :].min(), poly[1, :].max()],
    ]
    return domain, poly


# #############################################################################
# #############################################################################
#                           Mesh2DRect - select
# #############################################################################


def _select_ind_check(
    ind=None,
    elements=None,
    returnas=None,
    crop=None,
):

    # ind
    lc = [
        ind is None,
        isinstance(ind, tuple)
        and len(ind) == 2
        and (
            all([np.isscalar(ss) for ss in ind])
            or all([
                hasattr(ss, '__iter__')
                and len(ss) == len(ind[0])
                for ss in ind
            ])
            or all([isinstance(ss, np.ndarray) for ss in ind])
        ),
        (
            np.isscalar(ind)
            or (
                hasattr(ind, '__iter__')
                and all([np.isscalar(ss) for ss in ind])
            )
            or isinstance(ind, np.ndarray)
        )
    ]

    if not any(lc):
        msg = (
            "Arg ind must be either:\n"
            "\t- None\n"
            "\t- int or array of int: int indices in mixed (R, Z) indexing\n"
            "\t- tuple of such: int indices in (R, Z) indexing respectively\n"
            f"Provided: {ind}"
        )
        raise Exception(msg)

    if lc[0]:
        pass
    elif lc[1]:
        if any([not isinstance(ss, np.ndarray) for ss in ind]):
            ind = (
                np.atleast_1d(ind[0]).astype(int),
                np.atleast_1d(ind[1]).astype(int),
            )
        lc0 = [
            [
                isinstance(ss, np.ndarray),
                np.issubdtype(ss.dtype, np.integer),
                ss.shape == ind[0].shape,
            ]
                for ss in ind
        ]
        if not all([all(cc) for cc in lc0]):
            ltype = [type(ss) for ss in ind]
            ltypes = [
                ss.dtype if isinstance(ss, np.ndarray) else False
                for ss in ind
            ]
            lshapes = [
                ss.shape if isinstance(ss, np.ndarray) else len(ss)
                for ss in ind
            ]
            msg = (
                "Arg ind must be a tuple of 2 arrays of int of same shape\n"
                f"\t- lc0: {lc0}\n"
                f"\t- types: {ltype}\n"
                f"\t- type each: {ltypes}\n"
                f"\t- shape: {lshapes}\n"
                f"\t- ind: {ind}"
            )
            raise Exception(msg)
    else:
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).astype(int)
        c0 = (
            np.issubdtype(ind.dtype, np.integer)
            or np.issubdtype(ind.dtype, np.bool_)
        )
        if not c0:
            msg = (
                "Arg ind must be an array of bool or int\n"
                f"Provided: {ind.dtype}"
            )
            raise Exception(msg)

    # elements
    elements = _generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default=tuple,
        allowed=[tuple, np.ndarray, 'tuple-flat', 'array-flat', bool],
    )

    # crop
    crop = _generic_check._check_var(
        crop, 'crop',
        types=bool,
        default=True,
    )

    return ind, elements, returnas, crop


def _select_check(
    elements=None,
    returnas=None,
    return_neighbours=None,
):

    # elements
    elements = _generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default='ind',
        allowed=['ind', 'data'],
    )

    # return_neighbours
    return_neighbours = _generic_check._check_var(
        return_neighbours, 'return_neighbours',
        types=bool,
        default=True,
    )

    return elements, returnas, return_neighbours,


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _mesh2DRect_bsplines(key=None, lkeys=None, deg=None):

    # key
    if key is None and len(lkeys) == 1:
        key = lkeys[0]
    if key not in lkeys:
        msg = (
            "Arg key must be a valid mesh identifier!\n"
            f"\t- available: {lkeys}\n"
            f"\t- provided: {key}"
        )
        raise Exception(msg)

    # deg
    if deg is None:
        deg = 2
    if not isinstance(deg, int) and deg in [0, 1, 2, 3]:
        msg = (
            "Arg deg must be a int in [0, 1, 2, 3]\n"
            f"Provided: {deg}"
        )
        raise Exception(msg)

    # keybs
    keybs = f'{key}-bs{deg}'

    return key, keybs, deg
