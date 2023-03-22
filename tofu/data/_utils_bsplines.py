

import numpy as np
import datastock as ds


# #################################################################
# #################################################################
#                   Get knots per bsplines 1d
# #################################################################


def _get_knots_per_bs(
    knots,
    deg=None,
    returnas=None,
    return_unique=None,
    poloidal=None,
):

    # ----------
    # check input

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )
    return_unique = ds._generic_check._check_var(
        return_unique, 'return_unique',
        types=bool,
        default=False,
    )

    poloidal = ds._generic_check._check_var(
        poloidal, 'poloidal',
        default=False,
        types=bool,
    )

    # --------
    # prepare

    nkpbs = 2 + deg
    size = knots.size

    if poloidal is True:
        if size < nkpbs - 1:
            msg = (
                f"For the desired degree ({deg}), "
                f"a minimum of {nkpbs - 1} poloidal knots is necessary\n"
                f"Provided: {knots}"
            )
            raise Exception(msg)
        nbs = size
        if deg == 0 and size == nkpbs - 1:
            msg = "Using 2 pts for a deg = 0 bsplines leads to bspline!"
            raise Exception(msg)

    else:
        if size < 1 - deg:
            msg = (
                "For the desired degree ({deg}), "
                f"a minimum of {2 - deg} poloidal knots is necessary\n"
                f"Provided: {knots}"
            )
            raise Exception(msg)
        nbs = size - 1 + deg

    # ----------
    # compute

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
            if poloidal is True:
                knots_per_bs[:, :] = np.array([
                    np.arange(0, size),
                    np.r_[np.arange(1, size), 0],
                ])
            else:
                knots_per_bs[:, :] = np.array([
                    np.arange(0, size-1),
                    np.arange(1, size),
                ])

        elif deg == 1:
            if poloidal is True:
                if size == nkpbs - 1:
                    knots_per_bs[:, :] = np.array([
                        np.r_[0, 1],
                        np.r_[1, 0],
                        np.r_[0, 1],
                    ])
                else:
                    knots_per_bs[:, :] = np.array([
                        np.arange(0, size),
                        np.r_[np.arange(1, size), 0],
                        np.r_[np.arange(2, size), 0, 1],
                    ])
            else:
                knots_per_bs[:, 1:-1] = np.array([
                    np.arange(0, size-2),
                    np.arange(1, size-1),
                    np.arange(2, size),
                ])
                knots_per_bs[:, 0] = [0, 0, 1]
                knots_per_bs[:, -1] = [-2, -1, -1]

        elif deg == 2:
            if poloidal is True:
                if size == nkpbs - 1:
                    knots_per_bs[:, :] = np.array([
                        np.arange(0, size),
                        np.r_[np.arange(1, size), 0],
                        np.r_[np.arange(2, size), 0, 1],
                        np.arange(0, size),
                    ])
                else:
                    knots_per_bs[:, :] = np.array([
                        np.arange(0, size),
                        np.r_[np.arange(1, size), 0],
                        np.r_[np.arange(2, size), 0, 1],
                        np.r_[np.arange(3, size), 0, 1, 2],
                    ])
            else:
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
            if poloidal is True:
                if size == nkpbs - 1:
                    knots_per_bs[:, :] = np.array([
                        np.arange(0, size),
                        np.r_[np.arange(1, size), 0],
                        np.r_[np.arange(2, size), 0, 1],
                        np.r_[np.arange(3, size), 0, 1, 2],
                        np.arange(0, size),
                    ])
                else:
                    knots_per_bs[:, :] = np.array([
                        np.arange(0, size),
                        np.r_[np.arange(1, size), 0],
                        np.r_[np.arange(2, size), 0, 1],
                        np.r_[np.arange(3, size), 0, 1, 2],
                        np.r_[np.arange(4, size), 0, 1, 2, 3],
                    ])
            else:
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


# #################################################################
# #################################################################
#                   Get cents per bsplines 1d
# #################################################################


def _get_cents_per_bs(
    cents,
    deg=None,
    returnas=None,
    poloidal=None,
):

    # ------------
    # check inputs

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=str,
        default='data',
        allowed=['ind', 'data'],
    )

    poloidal = ds._generic_check._check_var(
        poloidal, 'poloidal',
        default=False,
        types=bool,
    )

    # -------
    # prepare

    nkpbs = 1 + deg
    size = cents.size
    nbs = size + deg
    cents_per_bs = np.zeros((nkpbs, nbs), dtype=int)

    # -------
    # compute

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

    # ------
    # return

    if returnas == 'data':
        cents_per_bs = cents[cents_per_bs]

    return cents_per_bs


# #################################################################
# #################################################################
#                   Get apex positions per bsplines 1d
# #################################################################


def _get_apex_per_bs(
    knots=None,
    knots_per_bs=None,
    deg=None,
    poloidal=None,
):

    # -------
    # prepare

    poloidal = ds._generic_check._check_var(
        poloidal, 'poloidal',
        default=False,
        types=bool,
    )

    nkpbs, nbs = knots_per_bs.shape

    # -------------
    # compute basis

    if nkpbs % 2 == 0:
        ii = int(nkpbs/2)
        apex = np.mean(knots_per_bs[ii-1:ii+1, :], axis=0)

    else:
        ii = int((nkpbs-1) / 2)
        apex = knots_per_bs[ii, :]

    # ------
    # adjust

    if poloidal is True:
        # make sure in [-pi; pi[
        apex = np.arctan2(np.sin(apex), np.cos(apex))
    else:
        # manage edges
        if deg == 1:
            apex[:deg] = knots[0]
            apex[-deg:] = knots[-1]
        elif deg == 2:
            apex[:deg] = [knots[0], 0.5*(knots[0] + knots[1])]
            apex[-deg:] = [0.5*(knots[-2] + knots[-1]), knots[-1]]
        elif deg == 3:
            apex[:deg] = [knots[0], 0.5*(knots[0]+knots[1]), knots[1]]
            apex[-deg:] = [knots[-2], 0.5*(knots[-2]+knots[-1]), knots[-1]]

    return apex


# #################################################################
# #################################################################
#                   Get apex positions per bsplines 1d
# #################################################################


def _coefs_axis(shapebs=None, coefs=None, axis=None, shapex=None):
    
    # -------------------------------------------------
    # initial safety check on coefs vs shapebs vs axis
    
    if np.isscalar(coefs):
        coefs = np.full(shapebs, coefs)
    
    c0 = (
        isinstance(coefs, np.ndarray)
        and coefs.ndim >= len(shapebs)
        and len(axis) == len(shapebs)
        and all([aa == axis[0] + ii for ii, aa in enumerate(axis)])
        and all([coefs.shape[aa] == shapebs[ii] for ii, aa in enumerate(axis)])
    )
    if not c0:
        msg = (
            f"Arg coefs must have a shape including {shapebs}\n"
            f"\t- shapebs: {shapebs}\n"
            f"\t- coefs.shape: {coefs.shape}\n"
            f"\t- axis: {axis}\n"
        )
        raise Exception(msg)
    
    # ----------------
    # shape for output 
    
    shape_pts, axis_x, ind_coefs, ind_x = [], [], [], []
    jj = 0
    for ii in range(coefs.ndim):
        if ii == axis[0]:
            for jj in range(len(shapex)):
                shape_pts.append(shapex[jj])
                axis_x.append(ii + jj)
                ind_x.append(None)
            ind_coefs.append(None)
            
        elif len(axis) > 1 and ii in axis[1:]:
            ind_coefs.append(None)
        else:
            shape_pts.append(coefs.shape[ii])
            ind_coefs.append(jj)
            ind_x.append(jj)
            jj += 1
    
    # ----------------
    # shape_other
    
    shape_other = tuple([
        ss for ii, ss in enumerate(coefs.shape)
        if ii not in axis
    ])
    
    return tuple(shape_pts), shape_other, axis_x, ind_coefs, ind_x


def _get_slices_iter(
    axis=None,
    axis_x=None,
    shape_coefs=None,
    shape_x=None,
    ind=None,
    ind_coefs=None,
    ind_x=None,
):
    
    sli_c = tuple([
        slice(None) if ii in axis else ind[ind_coefs[ii]]
        for ii in range(len(shape_coefs))
    ])

    sli_x =tuple([
        slice(None) if ii in axis_x else ind[ind_x[ii]]
        for ii in range(len(shape_x))
    ])
    
    return sli_c, sli_x


def _get_slices_iter_reverse_coefs(
    axis=None,
    shape_coefs=None,
    ind_coefs=None,
    return_as_list=None,
):
    
    sli_c = [
        ind_coefs if ii in axis else slice(None)
        for ii in range(len(shape_coefs))
    ]
    
    if return_as_list is True:
        return sli_c
    else:
        return tuple(sli_c)


def _get_slices_iter_reverse_x(
    axis_x=None,
    shape_x=None,
    ind_x=None,
):

    sli_x =tuple([
        ind_x if ii in axis_x else slice(None)
        for ii in range(len(shape_x))
    ])
    
    return sli_x


def _get_slice_out(indout=None, axis=None, shape_coefs=None):
    
    if len(axis) == 1:
        sli_out = tuple([
            indout if ii == axis[0] else slice(None)
            for ii in range(len(shape_coefs))
        ])
    else:
        sli_out = tuple([
            indout if ii == axis[0] else slice(None)
            for ii in range(len(shape_coefs))
            if ii not in axis[1:]
        ])
    
    return sli_out