# -*- coding: utf-8 -*-
"""
Polygons are assumed to be:
    - finite
    - simple (non self-intersecting)
    - non-explicitly closed
    - have no repeated points

"""


import numpy as np


# ###########################################################
# ###########################################################
#              Sample edges
# ###########################################################


def edges(
    x0=None,
    x1=None,
    res=None,
    factor=None,
):

    # ----------------
    # check inputs
    # ----------------

    # get x0 and x1 closed
    x0, x1, res

    # ----------------
    # prepare
    # ----------------

    # get all lengths
    dx0 = np.r_[x0[1:], x0[0]] - x0
    dx1 = np.r_[x1[1:], x1[0]] - x1
    dist = np.hypot(dx0, dx1)

    # ----------------
    # determine samplung res
    # ----------------

    if isinstance(res, str):

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
    # store
    # ----------------

    dout = {
        'x0': out0,
        'x1': out1,
    }

    return dout


# ###########################################################
# ###########################################################
#              check edges
# ###########################################################


def _check_edges(
    x0=None,
    x1=None,
    res=None,
    factor=None,
):


    return x0, x1, res, factor