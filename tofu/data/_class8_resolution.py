

import numpy as np
import scipy.linalg as scplinalg


# ################################################
# ################################################
#              main
# ################################################


def main():

    return


# ################################################
# ################################################
#              compute
# ################################################


def _compute(
    # ind
    key_mesh=None,
    indr=None,
    indz=None,
    indphi=None,
    # sang
    sang=None,
):
    """ Assumes concatenated sang matrix and indices, in for (ndet, npts)


    """

    # -------------
    # store
    # -------------

    dout = {
        'sang': sang,
        'ndet': np.sum(sang > 0, axis=0),
        'rank': rank,
        'rank_sang': rank_sang,
    }

    # -------------
    # run svd
    # -------------

    dsvd = dict.fromkeys(['sang', 'ndet', 'rank', 'rank_sang'])
    for k0 in dsvd.keys():
        dsvd[k0]['U'], dsvd[k0]['s'], dsvd['V'] = scplinalg.svd(
            dout[k0],
            full_matrices=True,
            compute_uv=True,
            overwrite_a=False,
            check_finite=True,
        )

    return dout, dsvd


# ################################################################
# ################################################################
#                   Compute_rank
# ################################################################


def _compute_rank(
    sang=None,
):

    assert sang.ndim == 2

    bool_det = sang > 0.

    # ---------------------------
    # extract unique combinations
    # ---------------------------

    rank, ncounts = np.unique(
        idet,
        axis=1,
        return_counts=True,
    )

    n_per_rank = np.sum(rank, axis=0)
    sang_per_rank =

    rank_x = np.unique(n_per_rank)
    rank_y = np.zeros(rank_x.shape)
    rank_z = np.zeros(rank_x.shape)
    for ir, rr in enumerate(rank_x):
        ind = n_per_rank == rr
        rank_y[ir] = np.sum(ncounts[ind])
        rank_z[ir] = ind.sum()

    assert np.sum(rank_y) == np.sum(ncounts)

    # ------------
    # output
    # ------------

    drank = {
        'bool_det': bool_ndet,
        'rank': rank,
        'ncounts': ncounts,
        'n_per_rank': n_per_rank,
        'sang_per_rank': sang_per_rank,
        'rank_x': rank_x,
        'rank_y': rank_y,
        'rank_z': rank_z,
    }

    return drank
