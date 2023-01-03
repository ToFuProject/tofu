# -*- coding: utf-8 -*-


# Built-in
import warnings


# Common
import numpy as np
import datastock as ds


from . import _class1_checks


_ELEMENTS = 'knots'


# #############################################################################
# #############################################################################
#                           mesh generic check
# #############################################################################


def _mesh1D_check(
    coll=None,
    E=None,
    key=None,
):

    # key
    key = ds._generic_check._obj_key(
        d0=coll._dobj.get(coll._which_msp, {}),
        short='msp',
        key=key,
    )

    # rect of tri ?
    dref, ddata, dmesh = _mesh1D_to_dict(
        E=E,
        key=key,
    )

    return dref, ddata, dmesh


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


def _mesh1D_to_dict(
    E=None,
    key=None,
):

    # --------------------
    # check / format input

    # keys
    kEk, kEc, kkE, kcE = _class1_checks._mesh_names(key=key, x_name='E')

    # E
    E, res, ind = _class1_checks._mesh1D_check(
        x=E,
        x_name='E',
        uniform=False,
    )

    Ecent = 0.5*(E[1:] + E[:-1])
    variable = not np.isscalar(res)

    # --------------------
    # prepare dict

    # dref
    dref = {
        kEk: {
            'size': E.size,
        },
        kEc: {
            'size': Ecent.size,
        },
    }

    # ddata
    ddata = {
        kkE: {
            'data': E,
            'units': 'eV',
            # 'source': None,
            'dim': 'energy',
            'quant': 'E',
            'name': 'E',
            'ref': kEk,
        },
        kcE: {
            'data': Ecent,
            'units': 'eV',
            # 'source': None,
            'dim': 'energy',
            'quant': 'E',
            'name': 'E',
            'ref': kEc,
        },
    }

    # dobj
    dmesh = {
        key: {
            'type': '1d',
            'knots': (kkE,),
            'cents': (kcE,),
            'shape-c': (Ecent.size,),
            'shape-k': (E.size,),
            'variable': variable,
        },
    }
    return dref, ddata, dmesh


# #############################################################################
# #############################################################################
#                           Mesh2DRect - select
# #############################################################################


def _select_ind_check(
    ind=None,
    elements=None,
    returnas=None,
    crop=None,
    meshtype=None,
    shape2d=None,
):

    # ----------------------
    # check basic conditions

    if shape2d:
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

    else:
        lc = [
            ind is None,
            np.isscalar(ind)
            or (
                hasattr(ind, '__iter__')
                and all([np.isscalar(ss) for ss in ind])
            )
            or isinstance(ind, np.ndarray)
        ]

    # check lc
    if not any(lc):
        if shape2d:
            msg = (
                "Arg ind must be either:\n"
                "\t- None\n"
                "\t- int or array of int: int indices in mixed (R, Z) index\n"
                "\t- tuple of such: int indices in (R, Z) index respectively\n"
                f"Provided: {ind}"
            )
        else:
            msg = (
                "Arg ind must be either:\n"
                "\t- None\n"
                "\t- int or array of int: int indices\n"
                "\t- array of bool: bool indices\n"
                f"Provided: {ind}"
            )
        raise Exception(msg)

    # ----------------------
    # adapt to each case

    if lc[0]:
        pass

    elif lc[1] and shape2d:
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

    elif lc[1] and not shape2d:
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
    elements = ds._generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    if shape2d:
        retdef = tuple
        retok = [tuple, np.ndarray, 'tuple-flat', 'array-flat', bool]
    else:
        retdef = bool
        retok = [int, bool]

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default=retdef,
        allowed=retok,
    )

    # crop
    crop = ds._generic_check._check_var(
        crop, 'crop',
        types=bool,
        default=True,
    )

    return ind, elements, returnas, crop


def _select_check(
    elements=None,
    returnas=None,
    return_ind_as=None,
    return_neighbours=None,
):

    # elements
    elements = ds._generic_check._check_var(
        elements, 'elements',
        types=str,
        default=_ELEMENTS,
        allowed=['knots', 'cents'],
    )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=None,
        default='ind',
        allowed=['ind', 'data'],
    )

    # return_ind_as
    return_ind_as = ds._generic_check._check_var(
        return_ind_as, 'return_ind_as',
        types=None,
        default=int,
        allowed=[int, bool],
    )

    # return_neighbours
    return_neighbours = ds._generic_check._check_var(
        return_neighbours, 'return_neighbours',
        types=bool,
        default=True,
    )

    return elements, returnas, return_ind_as, return_neighbours