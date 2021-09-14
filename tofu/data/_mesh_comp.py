# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
from . import _mesh_checks
from . import _mesh_bsplines


# #############################################################################
# #############################################################################
#                           Mesh2DRect - select
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


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _mesh2DRect_bsplines(mesh=None, key=None, deg=None):

    # ----------------
    # prepare

    keybs = f'{key}-bs{deg}'

    # --------------
    # create bsplines

    kR = mesh.dobj[mesh._groupmesh][key]['R-knots']
    kZ = mesh.dobj[mesh._groupmesh][key]['Z-knots']
    Rknots = mesh.ddata[kR]['data']
    Zknots = mesh.ddata[kZ]['data']

    kRbsc = f'{keybs}-R'
    kZbsc = f'{keybs}-Z'

    (
        func_details, func_sum, shapebs, Rbs_cent, Zbs_cent,
    ) = _mesh_bsplines.get_bs2d_func(
        deg=deg,
        Rknots=Rknots,
        Zknots=Zknots,
    )

    # ----------------
    # format into dict

    dref = {
        kRbsc: {
            'data': Rbs_cent,
            'units': 'm',
            'dimension': 'distance',
            'quant': 'R',
            'name': 'R',
            'group': 'R',
        },
        kZbsc: {
            'data': Zbs_cent,
            'units': 'm',
            'dimension': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'group': 'Z',
        },
    }

    dobj = {
        'bsplines': {
            keybs: {
                'deg': deg,
                'mesh': key,
                'Rbs': kRbsc,
                'Zbs': kZbsc,
                'shapebs': shapebs,
                'func_details': func_details,
                'func_sum': func_sum,
            }
        },
    }

    return dref, dobj


# #############################################################################
# #############################################################################
#                           Mesh2DRect - sample
# #############################################################################


def sample(mesh, key=None, res=None, mode=None, grid=None):

    # -------------
    # check inputs

    # key
    lk = list(mesh.dobj[mesh._groupmesh].keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    if key not in lk:
        msg = (
            "Arg key must point to a valid mesh id!\n"
            f"\t- available: {lk}\n"
            f"\t- provided: {key}\n"
        )
        raise Exception(msg)

    # res
    if res is None:
        res = 0.1
    if np.isscalar(res):
        res = [res, res]
    c0 = (
        isinstance(res, list)
        and len(res) == 2
        and all([np.isscalar(rr) and rr > 0 for rr in res])
    )
    if not c0:
        msg = f"Arg res must be a list of 2 positive floats!\nProvided: {res}"
        raise Exception(msg)

    # mode
    if mode is None:
        mode = 'abs'
    if mode not in ['rel', 'abs']:
        msg = f"Arg mode must be in ['rel', 'abs']!\nProvided: {mode}"
        raise Exception(msg)

    # grid
    if grid is None:
        grid = False
    if not isinstance(grid, bool):
        msg = f"Arg grid must be a bool!\nProvided: {grid}"
        raise Exception(msg)

    # -------------
    # compute

    kR = mesh.dobj[mesh._groupmesh][key]['R-knots']
    kZ = mesh.dobj[mesh._groupmesh][key]['Z-knots']
    R = mesh.ddata[kR]['data']
    Z = mesh.ddata[kZ]['data']

    if mode == 'abs':
        nR = int(np.ceil((R[-1] - R[0]) / res[0]))
        nZ = int(np.ceil((Z[-1] - Z[0]) / res[1]))
        R = np.linspace(R[0], R[-1], nR)
        Z = np.linspace(Z[0], Z[-1], nZ)
    else:
        nR = int(np.ceil(1./res[0]))
        nZ = int(np.ceil(1./res[1]))
        kR = np.linspace(0, 1, nR, endpoint=False)[None, :]
        kZ = np.linspace(0, 1, nZ, endpoint=False)[None, :]
        R = np.concatenate(
            (R[:-1, None] + kR*np.diff(R)[:, None], R[-1:, :]),
            axis=0,
        ).ravel()
        Z = np.concatenate(
            (Z[:-1, None] + kZ*np.diff(Z)[:, None], Z[-1:, :]),
            axis=0,
        ).ravel()

    # ------------
    # grid

    if grid is True:
        nZ = Z.size
        nR = R.size
        R = np.tile(R, (nZ, 1))
        Z = np.repeat(Z[:, None], nR, axis=1)

    return R, Z


# #############################################################################
# #############################################################################
#                           Mesh2DRect - interp
# #############################################################################


def _interp_check(
    mesh=None,
    key=None,
    R=None,
    Z=None,
    grid=None,
    details=None,
):
    # key
    dk = {
        kk: [
            k1 for k1, v1 in mesh.dobj['bsplines'].items()
            if mesh.ddata[kk]['ref'][-2:] == (v1['Rbs'], v1['Zbs'])
        ][0]
        for kk in mesh.ddata.keys()
        if any([
            mesh.ddata[kk]['ref'][-2:] == (v1['Rbs'], v1['Zbs'])
            for v1 in mesh.dobj['bsplines'].values()
        ])
    }
    if key is None and len(dk) == 1:
        ky = list(dk.keys())[0]
    if key not in dk.keys():
        msg = (
            "Arg key must the key to a data referenced on a bsplines set\n"
            f"\t- available: {dk.keys()}\n"
            f"\t- provided: {key}\n"
        )
        raise Exception(msg)
    keybs = dk[key]

    # details
    if details is None:
        details = False
    if not isinstance(details, bool):
        msg = f"Arg details must be a bool!\nProvided: {details}"
        raise Exception(msg)

    # grid
    if grid is None:
        grid = True
    if not isinstance(grid, bool):
        msg = f"Arg grid must be a bool!\nProvided: {grid}"
        raise Exception(msg)

    # R, Z
    try:
        R = np.atleast_1d(R).astype(float)
        Z = np.atleast_1d(Z).astype(float)
    except Exception as err:
        msg = "R and Z must eb convertible to np.arrays of floats"
        raise Exception(msg)

    if grid is True and (R.ndim > 1 or Z.ndim > 1):
        msg = "If grid=True, R and Z must be 1d!"
        raise Exception(msg)
    elif grid is False and R.shape != Z.shape:
        msg = "If grid=False, R and Z must have the same shape!"
        raise Exception(msg)

    if grid is True:
        R = np.tile(R, Z.size)
        Z = np.repeat(Z, R.size)

    return key, keybs, R, Z, grid, details


def interp(mesh=None, key=None, R=None, Z=None, grid=None, details=None):

    # ---------------
    # check inputs

    key, keybs, R, Z, grid, details = _interp_check(
        mesh=mesh,
        key=key,
        R=R,
        Z=Z,
        grid=grid,
        details=details,
    )

    # ---------------
    # interp

    if details is True:
        val = mesh.dobj['bsplines'][keybs]['func_details'](
            R, Z, coefs=mesh.ddata[key]['data'],
        )
    else:
        val = mesh.dobj['bsplines'][keybs]['func_sum'](
            R, Z, coefs=mesh.ddata[key]['data'],
        )

    return val
