# -*- coding: utf-8 -*-
"""
Polygons are assumed to be:
    - finite
    - simple (non self-intersecting)
    - non-explicitly closed
    - have no repeated points

"""


import numpy as np
import datastock as ds


from . import _class00_poly2d_check as _poly2d_check


# ###########################################################
# ###########################################################
#              Sample edges
# ###########################################################


def edges(
    x0=None,
    x1=None,
    key=None,
    # options
    res=None,
    factor=None,
):

    # ----------------
    # check inputs
    # ----------------

    # polygon formatting
    dout0 = _poly2d_check.check(
        x0=x0,
        x1=x1,
        key=key,
        # options
        closed=True,
        clockwise=True,
    )

    # extract x0, x1
    x0_closed = dout0['x0_closed']
    x1_closed = dout0['x1_closed']
    key = dout0['key']

    # options
    res, factor = _check(
        res=res,
        factor=factor,
    )

    # ----------------
    # prepare
    # ----------------

    # get all lengths
    dx0 = x0_closed[1:] - x0_closed[:-1]
    dx1 = x1_closed[1:] - x1_closed[:-1]
    dist = np.hypot(dx0, dx1)

    # ----------------
    # determine samplung res
    # ----------------

    if isinstance(res, str):

        # get res
        if res == 'min':
            res = np.min(dist)
        elif res == 'max':
            res = np.max(dist)
        else:
            raise NotImplementedError()

        # adjust
        res = res * factor

    # ----------------
    # sample res
    # ----------------

    # nb of points to be inserted in each segment
    npts_insert = np.round(dist / res, decimals=0).astype(int)
    npts_insert[npts_insert >= 1] -= 1

    # interpolation for x0
    out0 = np.ravel([
        np.r_[x0[ii]]
        if npts_insert[ii] == 0 else
        np.interp(
            np.linspace(0, 1, npts_insert[ii]+1, endpoint=False),
            [0, 1],
            x0[ii:ii+1],
        )
        for ii in range(x0.size)
    ])

    # interpolation for x1
    out1 = np.ravel([
        np.r_[x1[ii]]
        if npts_insert[ii] == 0 else
        np.interp(
            np.linspace(0, 1, npts_insert[ii]+1, endpoint=False),
            [0, 1],
            x1[ii:ii+1],
        )
        for ii in range(x1.size)
    ])

    # ----------------
    # return
    # ----------------

    dout = {
        'x0': out0,
        'x1': out1,
        'res': res,
        'factor': factor,
    }

    return dout


# ###########################################################
# ###########################################################
#              check edges
# ###########################################################


def _check(
    res=None,
    factor=None,
):

    # -------------
    # res
    # -------------

    res = ds._generic_check._check_var(
        res, 'res',
        types=(str, float),
    )

    if isinstance(res, str):
        res = ds._generic_check._check_var(
            res, 'res',
            types=str,
            default='min',
            allowed=['min', 'max'],
        )

    else:
        res = ds._generic_check._check_var(
            res, 'res',
            types=float,
            sign='>0',
        )

    # -------------
    # factor
    # -------------

    if isinstance(res, str):
        factor = ds._generic_check._check_var(
            factor, 'factor',
            types=float,
            sign='>0',
        )
    else:
        factor = None

    return res, factor