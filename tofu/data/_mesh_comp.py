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


def _select_ind(
    mesh=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
):
    """ ind can be:
            - None
            - tuple
            - 'flat'
    """

    # ------------
    # check inputs

    ind, elements, returnas = _mesh_checks._select_ind_check(
        ind=ind,
        elements=elements,
        returnas=returnas,
    )

    lk1 = list(mesh.dobj[mesh._groupmesh].keys())
    lk2 = list(mesh.dobj.get('bsplines', {}).keys())
    if key is None and len(lk1 + lk2) == 1:
        key = lk[0]
    if key not in lk1+lk2:
        msg = (
            "Arg key must be a valid mesh identifier!\n"
            f"\t available: {lk1+lk2}\n"
            f"\t- provided: {key}"
        )
        raise Exception(msg)

    if key in lk1:
        kR = mesh.dobj[mesh._groupmesh][key][f'R-{elements}']
        kZ = mesh.dobj[mesh._groupmesh][key][f'Z-{elements}']
    else:
        kR, kZ = mesh.dobj['bsplines'][key]['ref']

    nR = mesh.ddata[kR]['data'].size
    nZ = mesh.ddata[kZ]['data'].size

    # ------------
    # ind to tuple

    if ind is None:
        out = (
            np.tile(np.arange(0, nR), (nZ, 1)),
            np.repeat(np.arange(0, nZ)[:, None], nR, axis=1),
        )

    elif isinstance(ind, tuple):
        c0 = (
            np.all((ind[0] >= 0) & (ind[0] < nR))
            and np.all((ind[1] >= 0) & (ind[1] < nZ))
        )
        if not c0:
            msg = f"Arg ind has non-valid values (< 0 or >= size ({nR}, {nZ}))"
            raise Exception(msg)
        out = ind

    else:
        c0 = np.all((ind >= 0) & (ind < nR*nZ))
        if not c0:
            msg = f"Arg ind has non-valid values (< 0 or >= size ({nR*nZ}))"
            raise Exception(msg)
        out = (ind % nR, ind // nR)

    # ------------
    # tuple to return

    if returnas is tuple:
        pass
    else:
        out = out[0] + out[1]*nR

    return out


def _select_mesh(
    mesh=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    return_neighbours=None,
):
    """ ind is a tuple """

    # ------------
    # check inputs

    elements, returnas, return_neighbours = _mesh_checks._select_check(
        elements=elements,
        returnas=returnas,
        return_neighbours=return_neighbours,
    )

    lk = list(mesh.dobj[mesh._groupmesh])
    if key is None and len(lk) == 1:
        key = lk[0]
    if key not in lk:
        msg = (
            "Arg key must be a valid mesh identifier!\n"
            f"\t available: {lk}\n"
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

    # ------------
    # non-trivial case

    if returnas == 'ind':
        out = ind
    else:
        out = R[ind[0]], Z[ind[1]]

    # ------------
    # neighbours

    if return_neighbours is True:

        elneig = 'cent' if elements == 'knots' else 'knots'
        kRneig = mesh.dobj[mesh._groupmesh][key][f'R-{elneig}']
        kZneig = mesh.dobj[mesh._groupmesh][key][f'Z-{elneig}']
        Rneig = mesh.ddata[kRneig]['data']
        Zneig = mesh.ddata[kZneig]['data']
        nRneig = Rneig.size
        nZneig = Zneig.size

        # get tuple indices of neighbours
        shape = tuple(np.r_[ind[0].shape, 4])
        neig = (
            np.zeros(shape, dtype=int),
            np.zeros(shape, dtype=int),
        )
        rsh = tuple(
            [4 if ii == len(shape)-1 else 1 for ii in range(len(shape))]
        )

        if elements == 'cent':
            neig[0][...] = ind[0][..., None] + np.r_[0, 1, 1, 0].reshape(rsh)
            neig[1][...] = ind[1][..., None] + np.r_[0, 0, 1, 1].reshape(rsh)
        elif elements == 'knots':
            neig[0][...] = ind[0][..., None] + np.r_[-1, 0, 0, -1].reshape(rsh)
            neig[1][...] = ind[1][..., None] + np.r_[-1, -1, 0, 0].reshape(rsh)
            neig[0][(neig[0] < 0) | (neig[0] >= nRneig)] = -1
            neig[1][(neig[1] < 0) | (neig[1] >= nZneig)] = -1

        # return neighbours in desired format
        if returnas == 'ind':
            neig_out = neig
        else:
            neig_out = np.array([
                Rneig[neig[0]], Zneig[neig[1]],
            ])
            neig_out[:, (neig[0] == -1) | (neig[1] == -1)] = np.nan

        return out, neig_out
    else:
        return out


def _select_bsplines(
    mesh=None,
    key=None,
    ind=None,
    returnas=None,
    return_cents=None,
    return_knots=None,
):
    """ ind is a tuple """

    # ------------
    # check inputs

    _, returnas, _ = _mesh_checks._select_check(
        returnas=returnas,
    )

    lk = list(mesh.dobj['bsplines'])
    if key is None and len(lk) == 1:
        key = lk[0]
    if key not in lk:
        msg = (
            "Arg key must be a valid mesh identifier!\n"
            f"\t available: {lk}\n"
            f"\t- provided: {key}"
        )
        raise Exception(msg)

    # ------------
    # knots, cents

    out = _mesh2DRect_bsplines_knotscents(
        mesh=mesh,
        key=key,
        returnas=returnas,
        return_knots=return_knots,
        return_cents=return_cents,
        ind=ind,
    )

    # ------------
    # return

    if return_cents is True and return_knots is True:
        return ind, out[0], out[1]
    elif return_cents is True or return_knots is True:
        return ind, out
    else:
        return ind


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
                'ref': (kRbsc, kZbsc),
                # 'Rbs': kRbsc,
                # 'Zbs': kZbsc,
                'shapebs': shapebs,
                'func_details': func_details,
                'func_sum': func_sum,
            }
        },
    }

    return dref, dobj


def _mesh2DRect_bsplines_knotscents(
    mesh=None,
    key=None,
    returnas=None,
    return_knots=None,
    return_cents=None,
    ind=None,
):

    # -------------
    # check inputs

    return_knots = _mesh_checks._check_var(
        return_knots, 'return_knots',
        types=bool,
        default=True,
    )
    return_cents = _mesh_checks._check_var(
        return_cents, 'return_cents',
        types=bool,
        default=True,
    )
    if return_knots is False and return_cents is False:
        return

    # key
    lk = list(mesh.dobj['bsplines'].keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    if key not in lk:
        msg = (
        )
        raise Exception(msg)

    # -------------
    # prepare

    deg = mesh.dobj['bsplines'][key]['deg']
    mm = mesh.dobj['bsplines'][key]['mesh']

    # -------------
    # compute

    if return_knots is True:

        kR = mesh.dobj[mesh._groupmesh][mm]['R-knots']
        kZ = mesh.dobj[mesh._groupmesh][mm]['Z-knots']
        Rknots = mesh.ddata[kR]['data']
        Zknots = mesh.ddata[kZ]['data']

        knots_per_bs_R = _mesh_bsplines._get_bs2d_func_knots(
            Rknots, deg=deg, returnas=returnas,
        )
        knots_per_bs_Z = _mesh_bsplines._get_bs2d_func_knots(
            Zknots, deg=deg, returnas=returnas,
        )
        if ind is not None:
            knots_per_bs_R = knots_per_bs_R[:, ind[0]]
            knots_per_bs_Z = knots_per_bs_Z[:, ind[1]]

        nknots = knots_per_bs_R.shape[0]
        knots_per_bs_R = np.tile(knots_per_bs_R, (nknots, 1))
        knots_per_bs_Z = np.repeat(knots_per_bs_Z, nknots, axis=0)

    if return_cents is True:

        kR = mesh.dobj[mesh._groupmesh][mm]['R-cents']
        kZ = mesh.dobj[mesh._groupmesh][mm]['Z-cents']
        Rcents = mesh.ddata[kR]['data']
        Zcents = mesh.ddata[kZ]['data']

        cents_per_bs_R = _mesh_bsplines._get_bs2d_func_cents(
            Rcents, deg=deg, returnas=returnas,
        )
        cents_per_bs_Z = _mesh_bsplines._get_bs2d_func_cents(
            Zcents, deg=deg, returnas=returnas,
        )
        if ind is not None:
            cents_per_bs_R = cents_per_bs_R[:, ind[0]]
            cents_per_bs_Z = cents_per_bs_Z[:, ind[1]]

        ncents = cents_per_bs_R.shape[0]
        cents_per_bs_R = np.tile(cents_per_bs_R, (ncents, 1))
        cents_per_bs_Z = np.repeat(cents_per_bs_Z, ncents, axis=0)

    # -------------
    # return

    if return_knots is True and return_cents is True:
        out = (
            (knots_per_bs_R, knots_per_bs_Z), (cents_per_bs_R, cents_per_bs_Z)
        )
    elif return_knots is True:
        out = (knots_per_bs_R, knots_per_bs_Z)
    else:
        out = (cents_per_bs_R, cents_per_bs_Z)
    return out


# #############################################################################
# #############################################################################
#                           Mesh2DRect - sample
# #############################################################################


def sample_mesh(mesh, key=None, res=None, mode=None, grid=None, imshow=None):

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

    # imshow
    if imshow is None:
        imshow = False
    if not isinstance(imshow, bool):
        msg = f"Arg imshow must be a bool!\nProvided: {imshow}"
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
        R = np.concatenate((
            (R[:-1, None] + kR*np.diff(R)[:, None]).ravel(),
            R[-1:],
        ))
        Z = np.concatenate((
            (Z[:-1, None] + kZ*np.diff(Z)[:, None]).ravel(),
            Z[-1:],
        ))

    # ------------
    # grid

    if grid is True:
        nZ = Z.size
        nR = R.size
        if imshow is True:
            R = np.repeat(R[:, None], nZ, axis=1)
            Z = np.tile(Z, (nR, 1))
        else:
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
            if mesh.ddata[kk]['ref'][-2:] == v1['ref']
        ][0]
        for kk in mesh.ddata.keys()
        if any([
            mesh.ddata[kk]['ref'][-2:] == v1['ref']
            for v1 in mesh.dobj['bsplines'].values()
        ])
    }
    dk.update({kk: kk for kk in mesh.dobj['bsplines'].keys()})
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
    # prepare

    if keybs == key:
        coefs = None
    else:
        coefs = mesh.ddata[key]['data']

    if details is True:
        fname = 'func_details'
    else:
        fname = 'func_sum'

    # ---------------
    # interp

    val = mesh.dobj['bsplines'][keybs][fname](R, Z, coefs=coefs)

    return val
