# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################

def _select(
    ind=None,
    R=None,
    Z=None,
    returnas=None,
    return_neighbours=None,
):

    # ------------
    # non-trivial case

    if isinstance(ind, tuple):
        if returnas is tuple:
            out = ind
        elif returnas == 'flat':
            out = ind[0]*Z.size + ind[1]
        else:
            out = np.array([R[ind[0]], Z[ind[1]]])
    else:
        if returnas is tuple:
            out = (ind // Z.size, ind % Z.size)
        elif returnas == 'flat':
            out = ind
        else:
            out = np.array([np.tile(R)[ind], np.repeat(Z)[ind]])

    if return_neighbours is True:
        if returnas is tuple:
            neigh = (np.full((npts, 4), np.nan), np.full((4, npts), np.nan))
        elif returnas == 'flat':
            neigh = np.full((npts, 4), np.nan)
        else:
            neigh = np.full((2, npts, 4), np.nan)

        return out, neigh
    else:
        return out
