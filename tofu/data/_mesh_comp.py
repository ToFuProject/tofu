# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
from . import _mesh_checks


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################

def _select(
    mesh=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    return_neighbours=None,
):

    # ------------
    # check inputs

    ind, elements, returnas, return_neighbours = _mesh_checks._select_check(
        ind=ind,
        elements=elements,
        returnas=returnas,
        return_neighbours=return_neighbours,
    )

    if key is None and len(mesh.dobj[mesh._groupmesh]) == 1:
        key = list(mesh.dobj[mesh._groupmesh].keys())[0]
    if key not in mesh.dobj[mesh._groupmesh].keys():
        msg = (
            "Arg key must be a valid mesh identifier!\n"
            f"\t available: {mesh.dobj[mesh._groupmesh].keys()}\n"
            f"\t- provided: {key}"
        )
        raise Exception(msg)

    # ------------
    # prepare

    kR = mesh.dobj[mesh._groupmesh][key][f'R-{elements}']
    kZ = mesh.dobj[mesh._groupmesh][key][f'Z-{elements}']
    R = mesh.ddata[kR]['data']
    Z = mesh.ddata[kZ]['data']
    nR = R.size
    nZ = Z.size

    # get ind as tuple
    if isinstance(ind, tuple):
        itup = ind
    else:
        # count lines first
        itup = (ind % nR, ind // nR)

    # ------------
    # non-trivial case

    if returnas is tuple:
        out = itup
    elif returnas == 'flat':
        # count lines first
        out = itup[0] + itup[1]*nR
    else:
        out = np.array([R[itup[0]], Z[itup[1]]])

    # ------------
    # neighbours

    if return_neighbours is True:

        elneigh = 'cent' if elements == 'knots' else 'knots'
        kRneigh = mesh.dobj[mesh._groupmesh][key][f'R-{elneigh}']
        kZneigh = mesh.dobj[mesh._groupmesh][key][f'Z-{elneigh}']
        Rneigh = mesh.ddata[kRneigh]['data']
        Zneigh = mesh.ddata[kZneigh]['data']
        nRneigh = Rneigh.size
        nZneigh = Zneigh.size

        # get tuple indices of neighbours
        npts = itup[0].size
        neigh = (
            np.zeros((npts, 4), dtype=int),
            np.zeros((npts, 4), dtype=int),
        )

        if elements == 'cent':
            neigh[0][...] = itup[0][:, None] + np.r_[0, 1, 1, 0][None, :]
            neigh[1][...] = itup[1][:, None] + np.r_[0, 0, 1, 1][None, :]
        elif elements == 'knots':
            neigh[0][...] = itup[0][:, None] + np.r_[-1, 0, 0, -1][None, :]
            neigh[1][...] = itup[1][:, None] + np.r_[-1, -1, 0, 0][None, :]
            neigh[0][(neigh[0] < 0) | (neigh[0] >= nRneigh)] = -1
            neigh[1][(neigh[1] < 0) | (neigh[1] >= nZneigh)] = -1

        # return neighbours in desired format
        if returnas is tuple:
            neigh_out = neigh
        elif returnas == 'flat':
            neigh_out = neigh[0] + neigh[1]*nRneigh
            neigh_out[neigh_out < 0] = -1
        else:
            neigh_out = np.array([
                Rneigh[neigh[0]], Zneigh[neigh[1]],
            ])
            neigh_out[:, (neigh[0] == -1) | (neigh[1] == -1)] = np.nan

        return out, neigh_out
    else:
        return out
