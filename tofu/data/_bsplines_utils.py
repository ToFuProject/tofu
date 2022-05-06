

import numpy as np


from . import _generic_check


# #################################################################
# #################################################################
#                   Get knots per bsplines 1d
# #################################################################


def _get_bs2d_func_knots(
    knots,
    deg=None,
    returnas=None,
    return_unique=None,
):

    # ----------
    # check input

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )
    return_unique = _generic_check._check_var(
        return_unique, 'return_unique',
        types=bool,
        default=False,
    )

    # ----------
    # compute

    nkpbs = 2 + deg
    size = knots.size
    nbs = size - 1 + deg

    if return_unique:
        if deg == 0:
            knots_per_bs = np.arange(0, size)
        elif deg == 1:
            knots_per_bs = np.r_[0, np.arange(0, size), size-1]
        elif deg == 2:
            knots_per_bs = np.r_[0, 0, np.arange(0, size), size-1, size-1]
        elif deg == 3:
            knots_per_bs = np.r_[
                0, 0, 0, np.arange(0, size), size-1, size-1, size-1,
            ]

    else:
        knots_per_bs = np.zeros((nkpbs, nbs), dtype=int)

        if deg == 0:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size-1),
                np.arange(1, size),
            ])

        elif deg == 1:
            knots_per_bs[:, 1:-1] = np.array([
                np.arange(0, size-2),
                np.arange(1, size-1),
                np.arange(2, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 1]
            knots_per_bs[:, -1] = [-2, -1, -1]

        elif deg == 2:
            knots_per_bs[:, 2:-2] = np.array([
                np.arange(0, size-3),
                np.arange(1, size-2),
                np.arange(2, size-1),
                np.arange(3, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 0, 1]
            knots_per_bs[:, 1] = [0, 0, 1, 2]
            knots_per_bs[:, -2] = [-3, -2, -1, -1]
            knots_per_bs[:, -1] = [-2, -1, -1, -1]

        elif deg == 3:
            knots_per_bs[:, 3:-3] = np.array([
                np.arange(0, size-4),
                np.arange(1, size-3),
                np.arange(2, size-2),
                np.arange(3, size-1),
                np.arange(4, size),
            ])
            knots_per_bs[:, 0] = [0, 0, 0, 0, 1]
            knots_per_bs[:, 1] = [0, 0, 0, 1, 2]
            knots_per_bs[:, 2] = [0, 0, 1, 2, 3]
            knots_per_bs[:, -3] = [-4, -3, -2, -1, -1]
            knots_per_bs[:, -2] = [-3, -2, -1, -1, -1]
            knots_per_bs[:, -1] = [-2, -1, -1, -1, -1]

    # ----------
    # return

    if returnas == 'data':
        knots_per_bs = knots[knots_per_bs]

    if return_unique:
        return knots_per_bs, nbs
    else:
        return knots_per_bs


def _get_bs_func_knots_poloidal(
    knots,
    deg=None,
    returnas=None,
    return_unique=None,
):

    # ----------
    # check input

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )
    return_unique = _generic_check._check_var(
        return_unique, 'return_unique',
        types=bool,
        default=False,
    )

    # --------------------------
    # compute number of bsplines

    nkpbs = 2 + deg
    size = knots.size

    if size < nkpbs:
        msg = (
            f"For the desired degree ({deg}), "
            f"a minimum of {nkpbs} poloidal knots is necessary\n"
            f"Provided: {knots}"
        )
        raise Exception(msg)
    nbs = size

    # --------------------------
    # compute knots per bsplines

    if return_unique:
        if deg == 0:
            knots_per_bs = np.arange(0, size)
        elif deg == 1:
            knots_per_bs = np.r_[0, np.arange(0, size), size-1]
        elif deg == 2:
            knots_per_bs = np.r_[0, 0, np.arange(0, size), size-1, size-1]
        elif deg == 3:
            knots_per_bs = np.r_[
                0, 0, 0, np.arange(0, size), size-1, size-1, size-1,
            ]

    else:
        knots_per_bs = np.zeros((nkpbs, nbs), dtype=int)

        if deg == 0:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size),
                np.r_[np.arange(1, size), 0],
            ])

        elif deg == 1:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size),
                np.r_[np.arange(1, size), 0],
                np.r_[np.arange(2, size), 0, 1],
            ])

        elif deg == 2:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size),
                np.r_[np.arange(1, size), 0],
                np.r_[np.arange(2, size), 0, 1],
                np.r_[np.arange(3, size), 0, 1, 2],
            ])

        elif deg == 3:
            knots_per_bs[:, :] = np.array([
                np.arange(0, size),
                np.r_[np.arange(1, size), 0],
                np.r_[np.arange(2, size), 0, 1],
                np.r_[np.arange(3, size), 0, 1, 2],
                np.r_[np.arange(4, size), 0, 1, 2, 3],
            ])

    # ----------
    # return

    if returnas == 'data':
        knots_per_bs = knots[knots_per_bs]

    if return_unique:
        return knots_per_bs, nbs
    else:
        return knots_per_bs



# #################################################################
# #################################################################
#                   Get cents per bsplines 1d
# #################################################################


def _get_bs2d_func_cents(cents, deg=None, returnas=None):

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )

    nkpbs = 1 + deg
    size = cents.size
    nbs = size + deg
    cents_per_bs = np.zeros((nkpbs, nbs), dtype=int)

    if deg == 0:
        cents_per_bs[0, :] = np.arange(0, size)

    elif deg == 1:
        cents_per_bs[:, 1:-1] = np.array([
            np.arange(0, size-1),
            np.arange(1, size),
        ])
        cents_per_bs[:, 0] = [0, 0]
        cents_per_bs[:, -1] = [-1, -1]

    elif deg == 2:
        cents_per_bs[:, 2:-2] = np.array([
            np.arange(0, size-2),
            np.arange(1, size-1),
            np.arange(2, size),
        ])
        cents_per_bs[:, 0] = [0, 0, 0]
        cents_per_bs[:, 1] = [0, 0, 1]
        cents_per_bs[:, -2] = [-2, -1, -1]
        cents_per_bs[:, -1] = [-1, -1, -1]

    elif deg == 3:
        cents_per_bs[:, 3:-3] = np.array([
            np.arange(0, size-3),
            np.arange(1, size-2),
            np.arange(2, size-1),
            np.arange(3, size),
        ])
        cents_per_bs[:, 0] = [0, 0, 0, 0]
        cents_per_bs[:, 1] = [0, 0, 0, 1]
        cents_per_bs[:, 2] = [0, 0, 1, 2]
        cents_per_bs[:, -3] = [-3, -2, -1, -1]
        cents_per_bs[:, -2] = [-2, -1, -1, -1]
        cents_per_bs[:, -1] = [-1, -1, -1, -1]

    if returnas == 'data':
        cents_per_bs = cents[cents_per_bs]

    return cents_per_bs
