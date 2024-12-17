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


__all__ = ['edges', 'surface']


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
    x0_closed = dout0['x0']
    x1_closed = dout0['x1']
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
    npts = np.round(dist / res, decimals=0).astype(int)

    # interpolation for x0
    out0 = np.ravel([
        np.r_[x0_closed[ii]]
        if npts[ii] <= 1 else
        x0_closed[ii] + dx0[ii] * np.linspace(0, 1, npts[ii], endpoint=False)
        for ii in range(x0_closed.size-1)
    ])

    # interpolation for x1
    out1 = np.ravel([
        np.r_[x1_closed[ii]]
        if npts[ii] <= 1 else
        x1_closed[ii] + dx1[ii] * np.linspace(0, 1, npts[ii], endpoint=False)
        for ii in range(x1_closed.size-1)
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
        factor = float(ds._generic_check._check_var(
            factor, 'factor',
            types=(int, float),
            sign='>0',
        ))

    else:
        factor = None

    return res, factor


# ###########################################################
# ###########################################################
#              Sample surface
# ###########################################################


def surface(
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
    x0_closed = dout0['x0']
    x1_closed = dout0['x1']
    key = dout0['key']

    # options
    res, factor = _check(
        res=res,
        factor=factor,
    )

    # ----------------
    # prepare
    # ----------------


    # ----------------
    # return
    # ----------------

    dout = {
        'x0': x0,
        'x1': x1,
        'res': res,
        'factor': factor,
    }

    return dout