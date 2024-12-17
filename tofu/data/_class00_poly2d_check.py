# -*- coding: utf-8 -*-
""" Basic tools for formatting 2d polygons

"""


import numpy as np
import datastock as ds


# ###########################################################
# ###########################################################
#              check
# ###########################################################


def check(
    x0=None,
    x1=None,
    key=None,
    # options
    closed=None,
    clockwise=None,
):
    """ High-level routine to format a 2d polygon


    Parameters
    ----------
    x0 : sequence
        1st coordinate of the polygon
    x1 : sequence
        2nd coordinate of the polygon
    key : str / None
        To inform error messages
    closed : bool / None
        whether the polygon should be closed
    clockwise : bool / None
        whether the polygon should be clockwise

    Returns
    -------
    dout : dict
        {
            'x0': np.ndarray,
            'x0': np.ndarray,
            'closed': bool,
            'clockwise': bool,
            'key': str,
        }

    """

    # -------------
    # check inputs
    # -------------

    key, close, clockwise = _check(
        key=key,
        closed=closed,
        clockwise=clockwise,
    )

    # -------------
    # check x0, x1
    # -------------

    x0 = ds._generic_check._check_flat1darray(
        x0, 'x0',
        dtype=float,
        can_be_None=False,
        extra_msg=f"x0 of polygon '{key}'",
    )

    x1 = ds._generic_check._check_flat1darray(
        x1, 'x1',
        dtype=float,
        can_be_None=False,
        size=x0.size,
        extra_msg=f"x1 of polygon '{key}'",
    )

    # -------------
    # closed
    # -------------

    is_closed = np.allclose(np.r_[x0[0], x1[0]], np.r_[x0[-1], x1[-1]])
    if is_closed is True:
        x0_closed = x0
        x1_closed = x1

    else:
        ind_closed = np.r_[np.arange(0, x0.size), 0]
        x0_closed = x0[ind_closed]
        x1_closed = x1[ind_closed]

    # -------------
    # no duplicates
    # -------------

    uni = np.unique([x0_closed[:-1], x1_closed[:-1]], axis=1)
    if uni.shape[1] < (x0_closed.size - 1):
        ndup = x0.size - 1 - uni.shape[1]
        msg = (
            f"Polygon 2d '{key}' seems to have {ndup} duplicate points!\n"
            "\t- x0 = {x0_closed[:-1]}\n"
            "\t- x1 = {x1_closed[:-1]}\n"
        )
        raise Exception(msg)

    # -------------
    # clockwise
    # -------------

    # is already ?
    is_cw = is_clockwise(x0_closed, x1_closed)

    # adjust
    if is_cw != clockwise:
        x0_closed, x1_closed = x0_closed[::-1], x1_closed[::-1]

    # -------------
    # return
    # -------------

    dout = {
        'x0': x0_closed if closed else x0_closed[:-1],
        'x1': x1_closed if closed else x1_closed[:-1],
        'key': key,
        'closed': closed,
        'clockwise': clockwise,
    }

    return dout

# ###########################################################
# ###########################################################
#              check
# ###########################################################


def _check(
    key=None,
    closed=None,
    clockwise=None,
):

    # ---------------
    # key
    # ---------------

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        default='',
    )

    # ---------------
    # closed
    # ---------------

    closed = ds._generic_check._check_var(
        closed, 'closed',
        types=bool,
        default=False,
    )

    # ---------------
    # clockwise
    # ---------------

    clockwise = ds._generic_check._check_var(
        clockwise, 'clockwise',
        types=bool,
        default=True,
    )

    return key, closed, clockwise


# ###########################################################
# ###########################################################
#              is clockwise
# ###########################################################


def is_clockwise(x0, x1):
    area_signed = np.sum((x0[1:] - x0[:-1]) * (x1[1:] + x1[:-1]))
    return area_signed > 0