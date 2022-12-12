# -*- coding: utf-8 -*-


# Built-in
import warnings

# Common
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from contourpy import contour_generator
import datastock as ds


# tofu
from . import _utils_bsplines
from . import _class1_checks as _checks
from . import _class1_bsplines_rect
from . import _class1_bsplines_tri
from . import _class1_bsplines_polar


# #############################################################################
# #############################################################################
#                           Mesh2D - select
# #############################################################################


def _select_ind(
    coll=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    crop=None,
):
    """ ind can be:
            - None
            - tuple: (R, Z), possibly 2d
            - 'tuple-flat': (R, Z) flattened
            - np.ndarray: array of unique indices
            - 'array-flat': flattened ordered array of unique
    """

    # ------------
    # check inputs

    # key = mesh or bspline ?
    lk1 = list(coll.dobj.get(coll._which_mesh, {}).keys())
    lk2 = list(coll.dobj.get('bsplines', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        allowed=lk1 + lk2,
        types=str,
    )

    shape2d = None
    cat = coll._which_mesh if key in lk1 else 'bsplines'
    if cat == coll._which_mesh:
        meshtype = coll.dobj[cat][key]['type']
        shape2d = len(coll.dobj[cat][key]['shape-c']) == 2
    else:
        km = coll.dobj[cat][key]['mesh']
        meshtype = coll.dobj[coll._which_mesh][km]['type']
        shape2d = len(coll.dobj[cat][key]['shape']) == 2

    # ind, elements, ...
    # elements = cents or knots
    ind, elements, returnas, crop = _checks._select_ind_check(
        ind=ind,
        elements=elements,
        returnas=returnas,
        crop=crop,
        meshtype=meshtype,
        shape2d=shape2d,
    )

    elem = f'{elements}' if cat == coll._which_mesh else 'ref'

    if shape2d:
        if cat == coll._which_mesh:
            ke = f'shape-{elem[0]}'
            nR, nZ = coll.dobj[cat][key][ke]
        else:
            nR, nZ = coll.dobj[cat][key]['shape']
    else:
        if cat == coll._which_mesh:
            ke = f'shape-{elem[0]}'
            nelem = coll.dobj[cat][key][ke][0]
        else:
            nelem = coll.dobj[cat][key]['shape'][0]

    # ------------
    # ind to tuple

    if shape2d:
        ind_bool = np.zeros((nR, nZ), dtype=bool)
        if ind is None:
            # make sure R is varying in dimension 0
            ind_tup = (
                np.repeat(np.arange(0, nR)[:, None], nZ, axis=1),
                np.tile(np.arange(0, nZ), (nR, 1)),
            )
            ind_bool[...] = True

        elif isinstance(ind, tuple):
            c0 = (
                np.all((ind[0] >= 0) & (ind[0] < nR))
                and np.all((ind[1] >= 0) & (ind[1] < nZ))
            )
            if not c0:
                msg = (
                    f"Non-valid values in ind (< 0 or >= size ({nR}, {nZ}))"
                )
                raise Exception(msg)
            ind_tup = ind
            ind_bool[ind_tup[0], ind_tup[1]] = True

        else:
            if np.issubdtype(ind.dtype, np.integer):
                c0 = np.all((ind >= 0) & (ind < nR*nZ))
                if not c0:
                    msg = (
                        f"Non-valid values in ind (< 0 or >= size ({nR*nZ}))"
                    )
                    raise Exception(msg)
                ind_tup = (ind % nR, ind // nR)
                ind_bool[ind_tup[0], ind_tup[1]] = True

            elif np.issubdtype(ind.dtype, np.bool_):
                if ind.shape != (nR, nZ):
                    msg = (
                        f"Arg ind, if bool, must have shape {(nR, nZ)}\n"
                        f"Provided: {ind.shape}"
                    )
                    raise Exception(msg)
                # make sure R varies first
                ind_tup = ind.T.nonzero()[::-1]
                ind_bool = ind

            else:
                msg = f"Unknown ind dtype!\n\t- ind.dtype: {ind.dtype}"
                raise Exception(msg)

        if ind_tup[0].shape != ind_tup[1].shape:
            msg = (
                "ind_tup components do not have the same shape!\n"
                f"\t- ind_tup[0].shape = {ind_tup[0].shape}\n"
                f"\t- ind_tup[1].shape = {ind_tup[1].shape}"
            )
            raise Exception(msg)

    # triangular + polar1d case
    else:
        ind_bool = np.zeros((nelem,), dtype=bool)
        if ind is None:
            ind_bool[...] = True
        elif np.issubdtype(ind.dtype, np.integer):
            c0 = np.all((ind >= 0) & (ind < nelem))
            if not c0:
                msg = (
                    f"Arg ind has non-valid values (< 0 or >= size ({nelem}))"
                )
                raise Exception(msg)
            ind_bool[ind] = True
        elif np.issubdtype(ind.dtype, np.bool_):
            if ind.shape != (nelem,):
                msg = (
                    f"Arg ind, when array of bool, must have shape {(nelem,)}"
                    f"\nProvided: {ind.shape}"
                )
                raise Exception(msg)
            ind_bool = ind
        else:
            msg = (
                "Non-valid ind format!"
            )
            raise Exception(msg)

    # ------------
    # optional crop

    crop = (
        crop is True
        and coll.dobj[cat][key].get('crop') not in [None, False]
        and bool(np.any(~coll.ddata[coll.dobj[cat][key]['crop']]['data']))
    )
    if crop is True:
        cropi = coll.ddata[coll.dobj[cat][key]['crop']]['data']
        if meshtype == 'rect':
            if cat == coll._which_mesh and elements == 'knots':
                cropiknots = np.zeros(ind_bool.shape, dtype=bool)
                cropiknots[:-1, :-1] = cropi
                cropiknots[1:, :-1] = cropiknots[1:, :-1] | cropi
                cropiknots[1:, 1:] = cropiknots[1:, 1:] | cropi
                cropiknots[:-1, 1:] = cropiknots[:-1, 1:] | cropi

                ind_bool = ind_bool & cropiknots

                # ind_tup is not 2d anymore
                ind_tup = ind_bool.T.nonzero()[::-1]  # R varies first
                warnings.warn("ind is not 2d anymore!")

            elif ind_tup[0].shape == cropi.shape:
                ind_bool = ind_bool & cropi
                # ind_tup is not 2d anymore
                ind_tup = ind_bool.T.nonzero()[::-1]  # R varies first
                warnings.warn("ind is not 2d anymore!")

            else:
                ind_bool = ind_bool & cropi
                ind_tup = ind_bool.T.nonzero()[::-1]
        else:
            ind_bool &= cropi

    # ------------
    # tuple to return

    if returnas is bool:
        out = ind_bool
    elif returnas is int:
        out = ind_bool.nonzero()[0]
    elif returnas is tuple:
        out = ind_tup
    elif returnas == 'tuple-flat':
        # make sure R is varying first
        out = (ind_tup[0].T.ravel(), ind_tup[1].T.ravel())
    elif returnas is np.ndarray:
        out = ind_tup[0] + ind_tup[1]*nR
    elif returnas == 'array-flat':
        # make sure R is varying first
        out = (ind_tup[0] + ind_tup[1]*nR).T.ravel()
    else:
        out = ind_bool

    return out


# #############################################################################
# #############################################################################
#                           Mesh2D - select mesh
# #############################################################################


def _select_mesh(
    coll=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    return_ind_as=None,
    return_neighbours=None,
):
    """ ind is a tuple for rect """

    # ------------
    # check inputs

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=list(coll.dobj['mesh'].keys())
    )
    meshtype = coll.dobj['mesh'][key]['type']

    (
        elements, returnas,
        return_ind_as, return_neighbours,
    ) = _checks._select_check(
        elements=elements,
        returnas=returnas,
        return_ind_as=return_ind_as,
        return_neighbours=return_neighbours,
    )

    # ------------
    # prepare

    if meshtype in ['rect', 'tri']:
        kR, kZ = coll.dobj[coll._which_mesh][key][elements]
        R = coll.ddata[kR]['data']
        Z = coll.ddata[kZ]['data']
        nR = R.size
        nZ = Z.size
    else:
        kr = coll.dobj[coll._which_mesh][key][elements][0]
        rad = coll.ddata[kr]['data']


    # ------------
    # non-trivial case

    if returnas == 'ind':
        out = ind
    else:
        if meshtype == 'rect':
            out = R[ind[0]], Z[ind[0]]
        elif meshtype == 'tri':
            out = R[ind], Z[ind]
        else:
            out = rad[ind]

    # ------------
    # neighbours

    if return_neighbours is True:
        if meshtype == 'rect':
            neigh = _select_mesh_neighbours_rect(
                coll=coll,
                key=key,
                ind=ind,
                elements=elements,
                returnas=returnas,
            )
        elif meshtype == 'tri':
            neigh = _select_mesh_neighbours_tri(
                coll=coll,
                key=key,
                ind=ind,
                elements=elements,
                returnas=returnas,
                return_ind_as=return_ind_as,
            )
        else:
            # TBF
            raise NotImplementedError()
            neigh = _select_mesh_neighbours_polar(
                coll=coll,
                key=key,
                ind=ind,
                elements=elements,
                returnas=returnas,
                return_ind_as=return_ind_as,
            )

        return out, neigh
    else:
        return out


def _select_mesh_neighbours_rect(
    coll=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
):
    """ ind is a tuple for rect """

    # ------------
    # neighbours

    elneig = 'cents' if elements == 'knots' else 'knots'
    kRneig, kZneig = coll.dobj[coll._which_mesh][key][f'{elneig}']
    Rneig = coll.ddata[kRneig]['data']
    Zneig = coll.ddata[kZneig]['data']
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

    if elements == 'cents':
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
        neig_out = np.array([Rneig[neig[0]], Zneig[neig[1]]])
        neig_out[:, (neig[0] == -1) | (neig[1] == -1)] = np.nan

    return neig_out


def _select_mesh_neighbours_tri(
    coll=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    return_ind_as=None,
):
    """ ind is a bool

    if returnas = 'ind', ind is returned as a bool array
    (because the nb. of neighbours is not constant on a triangular mesh)

    """
    # ------------
    # neighbours

    nind = ind.sum()
    kind = coll.dobj[coll._which_mesh][key]['ind']

    if returnas == 'data':
        elneig = 'cents' if elements == 'knots' else 'knots'
        kneig = coll.dobj[coll._which_mesh][key][elneig]
        Rneig = coll.ddata[kneig[0]]['data']
        Zneig = coll.ddata[kneig[1]]['data']

    if elements == 'cents':
        neig = coll.ddata[kind]['data'][ind, :]
        if returnas == 'ind':
            if return_ind_as is bool:
                kknots = coll.dobj[coll._which_mesh][key]['knots']
                nneig = coll.dref[f'{kknots}-ind']['size']
                neig_temp = np.zeros((nind, nneig), dtype=bool)
                for ii in range(nind):
                    neig_temp[ii, neig[ii, :]] = True
                neig = neig_temp
        else:
            neig = np.array([Rneig[neig], Zneig[neig]])
    else:
        ind_int = ind.nonzero()[0]
        neig = np.array([
            np.any(coll.ddata[kind]['data'] == ii, axis=1)
            for ii in ind_int
        ])
        c0 = returnas == 'ind' and return_ind_as is int
        if c0 or returnas == 'data':
            nmax = np.sum(neig, axis=1)
            if returnas == 'ind':
                neig_temp = -np.ones((nind, nmax.max()), dtype=int)
                for ii in range(nind):
                    neig_temp[ii, :nmax[ii]] = neig[ii, :].nonzero()[0]
            else:
                neig_temp = np.full((2, nind, nmax.max()), np.nan)
                for ii in range(nind):
                    neig_temp[0, ii, :nmax[ii]] = Rneig[neig[ii, :]]
                    neig_temp[1, ii, :nmax[ii]] = Zneig[neig[ii, :]]
            neig = neig_temp

    return neig


# TBF
def _select_mesh_neighbours_polar(
    coll=None,
    key=None,
    ind=None,
    elements=None,
    returnas=None,
    return_ind_as=None,
):
    """ ind is a bool

    if returnas = 'ind', ind is returned as a bool array
    (because the nb. of neighbours is not constant on a triangular mesh)

    """

    elneig = 'cents' if elements == 'knots' else 'knots'
    kneig = coll.dobj[coll._which_mesh][key][f'{elneig}']
    rneig = coll.ddata[kneig[0]]['data']
    nrneig = rneig.size


    # ----------------
    # radius + angle

    if len(kneig) == 2:
        aneig = coll.ddata[kneig[1]]['data']
        naneig = aneig.size

        # prepare indices
        shape = tuple(np.r_[ind[0].shape, 2])
        neig = (
            np.zeros((nrneig, 2), dtype=bool),
            np.zeros((naneig, 2), dtype=bool),
        )

        # get indices of neighbours
        if elements == 'cents':
            neig[0][...] = ind[0][..., None] + np.r_[0, 1, 1, 0].reshape(rsh)
            neig[1][...] = ind[1][..., None] + np.r_[0, 0, 1, 1].reshape(rsh)
        elif elements == 'knots':
            neig[0][...] = ind[0][..., None] + np.r_[-1, 0, 0, -1].reshape(rsh)
            neig[1][...] = ind[1][..., None] + np.r_[-1, -1, 0, 0].reshape(rsh)
            neig[0][(neig[0] < 0) | (neig[0] >= nRneig)] = -1
            neig[1][(neig[1] < 0) | (neig[1] >= nZneig)] = -1


    # ----------------
    # radius only

    else:
        # prepare indices
        neig = np.zeros((nrneig, 2), dtype=bool)

        # get indices of neighbours
        if elements == 'cents':
            neig[0][...] = ind[0][..., None] + np.r_[0, 1, 1, 0].reshape(rsh)
            neig[1][...] = ind[1][..., None] + np.r_[0, 0, 1, 1].reshape(rsh)

    # return neighbours in desired format
    if returnas == 'ind':
        neig_out = neig
    else:
        if len(kneig) == 2:
            neig_out = np.array([rneig[neig[0]], zneig[neig[1]]])
            neig_out[:, (neig[0] == -1) | (neig[1] == -1)] = np.nan
        else:
            neig_out = rneig[neig]

    return neig_out


# #############################################################################
# #############################################################################
#                           Mesh2D - select bsplines rect
# #############################################################################


def _select_bsplines(
    coll=None,
    key=None,
    ind=None,
    returnas=None,
    return_cents=None,
    return_knots=None,
    crop=None,
):
    """ ind is a tuple """

    # ------------
    # check inputs

    _, returnas, _, _ = _checks._select_check(
        returnas=returnas,
    )

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=list(coll.dobj.get('bsplines', {}).keys()),
    )

    keym = coll.dobj['bsplines'][key]['mesh']
    meshtype = coll.dobj['mesh'][keym]['type']

    # ----
    # ind

    if meshtype == 'rect':
        returnasind = tuple
    elif meshtype == 'polar' and len(coll.dobj['bsplines'][key]['shape']) == 2:
        returnasind = tuple
    else:
        returnasind = bool

    ind = _select_ind(
        coll=coll,
        key=key,
        ind=ind,
        elements=None,
        returnas=returnasind,
        crop=crop,
    )

    # ------------
    # knots, cents

    if meshtype == 'rect':
        kRk, kZk = coll.dobj['mesh'][keym]['knots']
        kRc, kZc = coll.dobj['mesh'][keym]['cents']

        out = _mesh2DRect_bsplines_knotscents(
            returnas=returnas,
            return_knots=return_knots,
            return_cents=return_cents,
            ind=ind,
            deg=coll.dobj['bsplines'][key]['deg'],
            Rknots=coll.ddata[kRk]['data'],
            Zknots=coll.ddata[kZk]['data'],
            Rcents=coll.ddata[kRc]['data'],
            Zcents=coll.ddata[kZc]['data'],
        )

    elif meshtype == 'tri':
        clas = coll.dobj['bsplines'][key]['class']
        out = clas._get_knotscents_per_bs(
            returnas=returnas,
            return_knots=return_knots,
            return_cents=return_cents,
            ind=ind,
        )

    else:
        clas = coll.dobj['bsplines'][key]['class']
        shape2d = len(coll.dobj['bsplines'][key]['shape']) == 2
        if return_cents is True and return_knots is True:
            if shape2d:
                out = (
                    (clas.knots_per_bs_r, clas.knots_per_bs_a),
                    (clas.cents_per_bs_r, clas.cents_per_bs_a),
                )
            else:
                out = ((clas.knots_per_bs_r,), (clas.cents_per_bs_r,))
        elif return_cents is True:
            if shape2d:
                out = (clas.cents_per_bs_r, clas.cents_per_bs_a)
            else:
                out = (clas.cents_per_bs_r,)
        elif return_knots is True:
            if shape2d:
                out = (clas.knots_per_bs_r, clas.knots_per_bs_a)
            else:
                out = (clas.knots_per_bs_r,)

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
#                           Mesh2D - select bsplines tri
# #############################################################################


# TODO


# #############################################################################
# #############################################################################
#                           Mesh2 - Tri - bsplines
# #############################################################################


def _mesh2DTri_bsplines(coll=None, keym=None, keybs=None, deg=None):

    # --------------
    # create bsplines

    kknots = coll.dobj[coll._which_mesh][keym]['knots']
    func_details, func_sum, clas = _class1_bsplines_tri.get_bs2d_func(
        deg=deg,
        knotsR=coll.ddata[kknots[0]]['data'],
        knotsZ=coll.ddata[kknots[1]]['data'],
        cents=coll.ddata[coll.dobj[coll._which_mesh][keym]['ind']]['data'],
        trifind=coll.dobj[coll._which_mesh][keym]['func_trifind'],
    )
    keybsr = f'{keybs}-nbs'
    kbscr = f'{keybs}-apR'
    kbscz = f'{keybs}-apZ'

    bs_cents = clas._get_bs_cents()

    # ----------------
    # format into dict

    dref = {
        # bs index
        keybsr: {
            'size': clas.nbs,
        },
    }

    ddata = {
        kbscr: {
            'data': bs_cents[0, :],
            'units': 'm',
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'ref': (keybsr,),
        },
        kbscz: {
            'data': bs_cents[1, :],
            'units': 'm',
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'ref': (keybsr,),
        },
    }

    dobj = {
        'bsplines': {
            keybs: {
                'deg': deg,
                'mesh': keym,
                'ref': (keybsr,),
                'ref-bs': (keybsr,),
                'apex': (kbscr, kbscz),
                'shape': (clas.nbs,),
                'crop': False,
                'func_details': func_details,
                'func_sum': func_sum,
                'class': clas,
            }
        },
    }

    return dref, ddata, dobj


# #############################################################################
# #############################################################################
#                           Mesh2DRect - bsplines
# #############################################################################


def _mesh2DRect_bsplines(coll=None, keym=None, keybs=None, deg=None):

    # --------------
    # create bsplines

    kR, kZ = coll.dobj[coll._which_mesh][keym]['knots']
    Rknots = coll.ddata[kR]['data']
    Zknots = coll.ddata[kZ]['data']

    keybsr = f'{keybs}-nbs'
    kRbsapn = f'{keybs}-nR'
    kZbsapn = f'{keybs}-nZ'
    kRbsap = f'{keybs}-apR'
    kZbsap = f'{keybs}-apZ'

    (
        shapebs, Rbs_apex, Zbs_apex,
        knots_per_bs_R, knots_per_bs_Z,
    ) = _class1_bsplines_rect.get_bs2d_RZ(
        deg=deg, Rknots=Rknots, Zknots=Zknots,
    )
    nbs = int(np.prod(shapebs))

    func_details, func_sum, clas = _class1_bsplines_rect.get_bs2d_func(
        deg=deg,
        Rknots=Rknots,
        Zknots=Zknots,
        shapebs=shapebs,
        # knots_per_bs_R=knots_per_bs_R,
        # knots_per_bs_Z=knots_per_bs_Z,
    )

    # ----------------
    # format into dict

    dref = {
        kRbsapn: {
            'size': Rbs_apex.size,
        },
        kZbsapn: {
            'size': Zbs_apex.size,
        },
        keybsr: {
            'size': nbs,
        },
    }

    ddata = {
        kRbsap: {
            'data': Rbs_apex,
            'units': 'm',
            'dim': 'distance',
            'quant': 'R',
            'name': 'R',
            'ref': kRbsapn,
        },
        kZbsap: {
            'data': Zbs_apex,
            'units': 'm',
            'dim': 'distance',
            'quant': 'Z',
            'name': 'Z',
            'ref': kZbsapn,
        },
    }

    dobj = {
        'bsplines': {
            keybs: {
                'deg': deg,
                'mesh': keym,
                'ref': (kRbsapn, kZbsapn),
                'ref-bs': (keybsr,),
                'apex': (kRbsap, kZbsap),
                'shape': shapebs,
                'crop': False,
                'func_details': func_details,
                'func_sum': func_sum,
                'class': clas,
            }
        },
    }

    return dref, ddata, dobj


def add_cropbs_from_crop(coll=None, keybs=None, keym=None):

    # ----------------
    # get

    kcropbs = False
    if coll.dobj[coll._which_mesh][keym]['crop'] is not False:
        kcropm = coll.dobj[coll._which_mesh][keym]['crop']
        cropbs = _get_cropbs_from_crop(
            coll=coll,
            crop=coll.ddata[kcropm]['data'],
            keybs=keybs,
        )
        kcropbs = f'{keybs}-crop'
        kcroppedbs = f'{keybs}-nbs-crop'

    # ----------------
    # optional crop

    if kcropbs is not False:

        # add cropped flat reference
        coll.add_ref(
            key=kcroppedbs,
            size=int(cropbs.sum()),
        )

        coll.add_data(
            key=kcropbs,
            data=cropbs,
            ref=coll._dobj['bsplines'][keybs]['ref'],
            dim='bool',
            quant='bool',
        )
        coll._dobj['bsplines'][keybs]['crop'] = kcropbs


def _mesh2DRect_bsplines_knotscents(
    returnas=None,
    return_knots=None,
    return_cents=None,
    ind=None,
    deg=None,
    Rknots=None,
    Zknots=None,
    Rcents=None,
    Zcents=None,
):

    # -------------
    # check inputs

    return_knots = ds._generic_check._check_var(
        return_knots, 'return_knots',
        types=bool,
        default=True,
    )
    return_cents = ds._generic_check._check_var(
        return_cents, 'return_cents',
        types=bool,
        default=True,
    )
    if return_knots is False and return_cents is False:
        return

    # -------------
    # compute

    if return_knots is True:

        knots_per_bs_R = _utils_bsplines._get_knots_per_bs(
            Rknots, deg=deg, returnas=returnas,
        )
        knots_per_bs_Z = _utils_bsplines._get_knots_per_bs(
            Zknots, deg=deg, returnas=returnas,
        )
        if ind is not None:
            knots_per_bs_R = knots_per_bs_R[:, ind[0]]
            knots_per_bs_Z = knots_per_bs_Z[:, ind[1]]

        nknots = knots_per_bs_R.shape[0]
        knots_per_bs_R = np.tile(knots_per_bs_R, (nknots, 1))
        knots_per_bs_Z = np.repeat(knots_per_bs_Z, nknots, axis=0)

    if return_cents is True:

        cents_per_bs_R = _utils_bsplines._get_cents_per_bs(
            Rcents, deg=deg, returnas=returnas,
        )
        cents_per_bs_Z = _utils_bsplines._get_cents_per_bs(
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
#                           Mesh2D - polar - bsplines
# #############################################################################


def _mesh2Dpolar_bsplines(
    coll=None,
    keym=None,
    keybs=None,
    angle=None,
    deg=None,
):

    # ---------------
    # create bsplines

    kknots = coll.dobj[coll._which_mesh][keym]['knots']
    knotsr = coll.ddata[kknots[0]]['data']
    if len(kknots) == 2:
        angle = coll.ddata[kknots[1]]['data']

    func_details, func_sum, clas = _class1_bsplines_polar.get_bs2d_func(
        deg=deg,
        knotsr=coll.ddata[kknots[0]]['data'],
        angle=angle,
        coll=coll,
    )

    keybsnr = f'{keybs}-nr'
    keybsn = f'{keybs}-nbs'
    keybsapr = f'{keybs}-apr'

    # ------------
    # refs

    if clas.knotsa is None:
        ref = (keybsnr,)
        apex = (keybsapr,)
    elif len(clas.shapebs) == 2:
        keybsna = f'{keybs}-na'
        keybsapa = f'{keybs}-apa'
        ref = (keybsnr, keybsna)
        apex = (keybsapr, keybsapa)
    else:
        ref = (keybsn,)
        apex = (keybsapr,)

        # check angle vs angle2d
        mesh = coll._which_mesh
        angle2d = coll.dobj[mesh][keym]['angle2d']
        if angle2d is None:
            msg = (
                "Poloidal bsplines require mesh with angle2d!\n"
                f"\t- self.dobj['{mesh}']['{keym}']['angle2d'] = {angle2d}"
            )
            raise Exception(msg)

    # bs_cents = clas._get_bs_cents()

    # ----------------
    # format into dict

    # dref
    dref = {
        # bs index
        keybsnr: {'size': clas.nbs_r},
        keybsn: {'size': clas.nbs},
    }
    if len(clas.shapebs) == 2:
        dref[keybsna] = {'size': clas.nbs_a_per_r[0]}

    # ddata
    ddata = {
        keybsapr: {
            'data': clas.apex_per_bs_r,
            'units': '',
            'dim': '',
            'quant': '',
            'name': '',
            'ref': (keybsnr,),
        },
    }
    if len(clas.shapebs) == 2:
        ddata[keybsapa] = {
            'data': clas.apex_per_bs_a[0],
            'units': 'rad',
            'dim': 'angle',
            'quant': '',
            'name': '',
            'ref': (keybsna,),
        }

    # dobj
    dobj = {
        'bsplines': {
            keybs: {
                'deg': deg,
                'mesh': keym,
                'ref': ref,
                'ref-bs': (keybsn,),
                'apex': apex,
                'shape': clas.shapebs,
                'func_details': func_details,
                'func_sum': func_sum,
                'class': clas,
                'crop': coll.dobj[coll._which_mesh][keym]['crop'],
            }
        },
    }

    return dref, ddata, dobj


def _mesh2DPolar_bsplines_knotscents(
    returnas=None,
    return_knots=None,
    return_cents=None,
    ind=None,
    deg=None,
    # resources
    clas=None,
    rknots=None,
    aknots=None,
    rcents=None,
    acents=None,
):

    # -------------
    # check inputs

    return_knots = ds._generic_check._check_var(
        return_knots, 'return_knots',
        types=bool,
        default=True,
    )
    return_cents = ds._generic_check._check_var(
        return_cents, 'return_cents',
        types=bool,
        default=True,
    )
    if return_knots is False and return_cents is False:
        return

    # -------------
    # compute

    if return_knots is True:

        knots_per_bs_r = _utils_bsplines._get_knots_per_bs(
            rknots, deg=deg, returnas=returnas,
        )
        knots_per_bs_Z = _utils_bsplines._get_knots_per_bs(
            Zknots, deg=deg, returnas=returnas,
        )
        if ind is not None:
            knots_per_bs_R = knots_per_bs_R[:, ind[0]]
            knots_per_bs_Z = knots_per_bs_Z[:, ind[1]]

        nknots = knots_per_bs_R.shape[0]
        knots_per_bs_R = np.tile(knots_per_bs_R, (nknots, 1))
        knots_per_bs_Z = np.repeat(knots_per_bs_Z, nknots, axis=0)

    if return_cents is True:

        cents_per_bs_R = _utils_bsplines._get_cents_per_bs(
            Rcents, deg=deg, returnas=returnas,
        )
        cents_per_bs_Z = _utils_bsplines._get_cents_per_bs(
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


def _get_sample_mesh_res(
    coll=None,
    keym=None,
    mtype=None,
):
    if mtype == 'rect':
        kR, kZ = coll.dobj[coll._which_mesh][keym]['knots']
        res = min(
            np.min(np.diff(coll.ddata[kR]['data'])),
            np.min(np.diff(coll.ddata[kZ]['data'])),
        )
    elif mtype == 'tri':
        res = 0.02
    else:
        keyr2d = coll.dobj[coll._which_mesh][keym]['radius2d']
        keybs0 = coll.ddata[keyr2d]['bsplines']
        keym0 = coll.dobj['bsplines'][keybs0]['mesh']
        mtype0 = coll.dobj[coll._which_mesh][keym0]['type']
        res = _get_sample_mesh_res(coll=coll, keym=keym0, mtype=mtype0)
    return res


def _sample_mesh_check(
    coll=None,
    key=None,
    res=None,
    mode=None,
    grid=None,
    imshow=None,
    R=None,
    Z=None,
    DR=None,
    DZ=None,
):

    # -----------
    # Parameters

    # key
    key = ds._generic_check._check_var(
        key, 'key',
        allowed=list(coll.dobj.get('mesh', {}).keys()),
        types=str,
    )
    meshtype = coll.dobj['mesh'][key]['type']

    # for polar mesh => sample underlying mesh
    if meshtype == 'polar':
        key = coll.dobj[coll._which_mesh][key]['submesh']
        meshtype = coll.dobj['mesh'][key]['type']

    # res
    if res is None:
        res = _get_sample_mesh_res(coll=coll, keym=key, mtype=meshtype)

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
    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='abs',
    )

    # grid
    grid = ds._generic_check._check_var(
        grid, 'grid',
        types=bool,
        default=False,
    )

    # imshow
    imshow = ds._generic_check._check_var(
        imshow, 'imshow',
        types=bool,
        default=False,
    )

    # R, Z
    if R is None and Z is None:
        pass
    elif R is None and np.isscalar(Z):
        pass
    elif Z is None and np.isscalar(R):
        pass
    else:
        msg = (
            "For mesh discretisation, (R, Z) can be either:\n"
            "\t- (None, None): will be created\n"
            "\t- (scalar, None): A vertical line will be created\n"
            "\t- (None, scalar): A horizontal line will be created\n"
        )
        raise Exception(msg)

    # -------------
    # R, Z

    if meshtype == 'rect':
        kR, kZ = coll.dobj['mesh'][key]['knots']
        Rk = coll.ddata[kR]['data']
        Zk = coll.ddata[kZ]['data']

        # custom R xor Z for vertical / horizontal lines only
        if R is None and Z is not None:
            R = Rk
        if Z is None and R is not None:
            Z = Zk
    else:
        kknots = coll.dobj['mesh'][key]['knots']
        Rk = coll.ddata[kknots[0]]['data']
        Zk = coll.ddata[kknots[1]]['data']

    # custom DR or DZ for mode='abs' only
    if DR is not None or DZ is not None:
        if mode != 'abs':
            msg = "Custom DR or DZ can only be provided with mode = 'abs'!"
            raise Exception(msg)

        for DD, DN in [(DR, 'DR'), (DZ, 'DZ')]:
            if DD is not None:
                c0 = (
                    hasattr(DD, '__iter__')
                    and len(DD) == 2
                    and all([
                        rr is None or (np.isscalar(rr) and np.isfinite(rr))
                        for rr in DD
                    ])
                )
                if not c0:
                    msg = f'Arg {DN} must be an iterable of 2 scalars!'
                    raise Exception(msg)

    if DR is None:
        DR = [Rk.min(), Rk.max()]
    if DZ is None:
        DZ = [Zk.min(), Zk.max()]

    return key, res, mode, grid, imshow, R, Z, DR, DZ, Rk, Zk


def sample_mesh(
    coll=None,
    key=None,
    res=None,
    mode=None,
    R=None,
    Z=None,
    DR=None,
    DZ=None,
    grid=None,
    imshow=None,
):

    # -------------
    # check inputs

    key, res, mode, grid, imshow, R, Z, DR, DZ, Rk, Zk = _sample_mesh_check(
        coll=coll,
        key=key,
        res=res,
        mode=mode,
        grid=grid,
        imshow=imshow,
        R=R,
        Z=Z,
        DR=DR,
        DZ=DZ,
    )

    # -------------
    # compute

    if mode == 'abs':
        if R is None:
            nR = int(np.ceil((DR[1] - DR[0]) / res[0]))
            R = np.linspace(DR[0], DR[1], nR)
        if Z is None:
            nZ = int(np.ceil((DZ[1] - DZ[0]) / res[1]))
            Z = np.linspace(DZ[0], DZ[1], nZ)
    else:
        if R is None:
            nR = int(np.ceil(1./res[0]))
            kR = np.linspace(0, 1, nR, endpoint=False)[None, :]
            R = np.concatenate((
                (Rk[:-1, None] + kR*np.diff(Rk)[:, None]).ravel(),
                Rk[-1:],
            ))
        if Z is None:
            nZ = int(np.ceil(1./res[1]))
            kZ = np.linspace(0, 1, nZ, endpoint=False)[None, :]
            Z = np.concatenate((
                (Zk[:-1, None] + kZ*np.diff(Zk)[:, None]).ravel(),
                Zk[-1:],
            ))

    if np.isscalar(R):
        R = np.full(Z.shape, R)
    if np.isscalar(Z):
        Z = np.full(R.shape, Z)

    # ------------
    # grid

    if grid is True:
        nZ = Z.size
        nR = R.size
        if imshow is True:
            R = np.tile(R, (nZ, 1))
            Z = np.repeat(Z[:, None], nR, axis=1)
        else:
            R = np.repeat(R[:, None], nZ, axis=1)
            Z = np.tile(Z, (nR, 1))

    return R, Z


# #############################################################################
# #############################################################################
#                           Mesh2DRect - crop
# #############################################################################


def _crop_check(
    coll=None,
    key=None,
    crop=None,
    thresh_in=None,
    remove_isolated=None,
):

    # key
    lkm = list(coll.dobj[coll._which_mesh].keys())
    key = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lkm,
    )
    meshtype = coll.dobj[coll._which_mesh][key]['type']

    if meshtype != 'rect':
        raise NotImplementedError()

    # shape
    shape = coll.dobj[coll._which_mesh][key]['shape-c']

    # crop
    c0 = (
        isinstance(crop, np.ndarray)
        and crop.ndim == 2
        and np.all(np.isfinite(crop))
        and (
            (
                crop.shape[0] == 2
                and np.allclose(crop[:, 0], crop[:, -1])
                and (
                    np.issubdtype(crop.dtype, np.integer)
                    or np.issubdtype(crop.dtype, np.floating)
                )
            )
            or (
                crop.shape == shape
                and crop.dtype == np.bool_
            )
        )
    )
    if not c0:
        msg = (
            "Arg crop must be either:\n"
            f"\t- array of bool: mask of shape {shape}\n"
            f"\t- array of floats: (2, npts) closed (R, Z) polygon\n"
            f"Provided:\n{crop}"
        )
        raise Exception(msg)

    cropbool = crop.dtype == np.bool_

    # thresh_in and maxth
    if thresh_in is None:
        thresh_in = 3
    maxth = 5 if coll.dobj[coll._which_mesh][key]['type'] == 'rect' else 4

    c0 = isinstance(thresh_in, (int, np.integer)) and (1 <= thresh_in <= maxth)
    if not c0:
        msg = (
            f"Arg thresh_in must be a int in {1} <= thresh_in <= {maxth}\n"
            f"Provided: {thresh_in}"
        )
        raise Exception(msg)

    # remove_isolated
    remove_isolated = ds._generic_check._check_var(
        remove_isolated, 'remove_isolated',
        default=True,
        types=bool,
    )

    return key, cropbool, thresh_in, remove_isolated


def crop(
    coll=None,
    key=None,
    crop=None,
    thresh_in=None,
    remove_isolated=None,
):
    """ Crop a rect mesh

    Parameters
    ----------
    key:        str
        key of the rect mesh to be cropped
    crop:      np.ndarray
        Can be either:
            - bool: a boolean mask array
            - float: a closed 2d polygon used for cropping
    threshin:   int
        minimum nb. of corners for a mesh element to be included
    remove_isolated: bool
        flag indicating whether to remove isolated mesh elements

    Return
    ------
    crop:       np.ndarray
        bool mask
    key:        str
        key of the rect mesh to be cropped
    thresh_in:  int
        minimum nb. of corners for a mesh element to be included

    """

    # ------------
    # check inputs

    key, cropbool, thresh_in, remove_isolated = _crop_check(
        coll=coll, key=key, crop=crop, thresh_in=thresh_in,
        remove_isolated=remove_isolated,
    )

    # -----------
    # if crop is a poly => compute as bool

    if not cropbool:

        (Rc, Zc), (Rk, Zk) = coll.select_mesh_elements(
            key=key, elements='cents',
            return_neighbours=True, returnas='data',
        )
        nR, nZ = Rc.shape
        npts = Rk.shape[-1] + 1

        pts = np.concatenate(
            (
                np.concatenate((Rc[:, :, None], Rk), axis=-1)[..., None],
                np.concatenate((Zc[:, :, None], Zk), axis=-1)[..., None],
            ),
            axis=-1,
        ).reshape((npts*nR*nZ, 2))

        isin = Path(crop.T).contains_points(pts).reshape((nR, nZ, npts))
        crop = np.sum(isin, axis=-1) >= thresh_in

        # Remove isolated pixelsi
        if remove_isolated is True:
            # All pixels should have at least one neighbour in R and one in Z
            # This constraint is useful for discrete gradient evaluation (D1N2)
            neighR = np.copy(crop)
            neighR[0, :] &= neighR[1, :]
            neighR[-1, :] &= neighR[-2, :]
            neighR[1:-1, :] &= (neighR[:-2, :] | neighR[2:, :])
            neighZ = np.copy(crop)
            neighZ[:, 0] &= neighZ[:, 1]
            neighZ[:, -1] &= neighZ[:, -2]
            neighZ[:, 1:-1] &= (neighZ[:, :-2] | neighZ[:, 2:])
            crop = neighR & neighZ

    return crop, key, thresh_in


def _get_cropbs_from_crop(coll=None, crop=None, keybs=None):

    if isinstance(crop, str) and crop in coll.ddata.keys():
        crop = coll.ddata[crop]['data']

    shref = coll.dobj[coll._which_mesh][coll.dobj['bsplines'][keybs]['mesh']]['shape-c']
    if crop.shape != shref:
        msg = "Arg crop seems to have the wrong shape!"
        raise Exception(msg)

    keym = coll.dobj['bsplines'][keybs][coll._which_mesh]
    kRk, kZk = coll.dobj['mesh'][keym]['knots']
    kRc, kZc = coll.dobj['mesh'][keym]['cents']

    cents_per_bs_R, cents_per_bs_Z = _mesh2DRect_bsplines_knotscents(
        returnas='ind',
        return_knots=False,
        return_cents=True,
        ind=None,
        deg=coll.dobj['bsplines'][keybs]['deg'],
        Rknots=coll.ddata[kRk]['data'],
        Zknots=coll.ddata[kZk]['data'],
        Rcents=coll.ddata[kRc]['data'],
        Zcents=coll.ddata[kZc]['data'],
    )

    shapebs = coll.dobj['bsplines'][keybs]['shape']
    cropbs = np.array([
        [
            np.all(crop[cents_per_bs_R[:, ii], cents_per_bs_Z[:, jj]])
            for jj in range(shapebs[1])
        ]
        for ii in range(shapebs[0])
    ], dtype=bool)

    return cropbs


# #############################################################################
# #############################################################################
#                           Mesh2DRect - interp utility
# #############################################################################


def _get_keyingroup_ddata(
    dd=None, dd_name='data',
    key=None, monot=None,
    msgstr=None, raise_=False,
):
    """ Return the unique data key matching key

    Here, key can be interpreted as name / source / units / quant...
    All are tested using select() and a unique match is returned
    If not unique match an error message is either returned or raised

    """

    # ------------------------
    # Trivial case: key is actually a ddata key

    if key in dd.keys():
        return key, None

    # ------------------------
    # Non-trivial: check for a unique match on other params

    dind = _select(
        dd=dd, dd_name=dd_name,
        dim=key, quant=key, name=key, units=key, source=key,
        monot=monot,
        log='raw',
        returnas=bool,
    )
    ind = np.array([ind for kk, ind in dind.items()])

    # Any perfect match ?
    nind = np.sum(ind, axis=1)
    sol = (nind == 1).nonzero()[0]
    key_out, msg = None, None
    if sol.size > 0:
        if np.unique(sol).size == 1:
            indkey = ind[sol[0], :].nonzero()[0]
            key_out = list(dd.keys())[indkey]
        else:
            lstr = "[dim, quant, name, units, source]"
            msg = "Several possible matches in {} for {}".format(lstr, key)
    else:
        lstr = "[dim, quant, name, units, source]"
        msg = "No match in {} for {}".format(lstr, key)

    # Complement error msg and optionally raise
    if msg is not None:
        lk = ['dim', 'quant', 'name', 'units', 'source']
        dk = {
            kk: (
                dind[kk].sum(),
                sorted(set([vv[kk] for vv in dd.values()]))
            ) for kk in lk
        }
        msg += (
            "\n\nRequested {} could not be identified!\n".format(msgstr)
            + "Please provide a valid (unique) key/name/dim/quant/units:\n\n"
            + '\n'.join([
                '\t- {} ({} matches): {}'.format(kk, dk[kk][0], dk[kk][1])
                for kk in lk
            ])
            + "\nProvided:\n\t'{}'".format(key)
        )
        if raise_:
            raise Exception(msg)
    return key_out, msg


def _get_possible_ref12d(
    dd=None,
    key=None, ref1d=None, ref2d=None,
    group1d='radius',
    group2d='mesh2d',
):

    # Get relevant lists
    kq, msg = _get_keyingroup_ddata(
        dd=dd,
        key=key, group=group2d, msgstr='quant', raise_=False,
    )

    if kq is not None:
        # The desired quantity is already 2d
        k1d, k2d = None, None

    else:
        # Check if the desired quantity is 1d
        kq, msg = _get_keyingroup_ddata(
            dd=dd,
            key=key, group=group1d,
            msgstr='quant', raise_=True,
        )

        # Get dict of possible {ref1d: lref2d}
        ref = [rr for rr in dd[kq]['ref'] if dd[rr]['group'] == (group1d,)][0]
        lref1d = [
            k0 for k0, v0 in dd.items()
            if ref in v0['ref'] and v0['monot'][v0['ref'].index(ref)] is True
        ]

        # Get matching ref2d with same quant and good group
        lquant = list(set([dd[kk]['quant'] for kk in lref1d]))
        dref2d = {
            k0: [
                kk for kk in _select(
                    dd=dd, quant=dd[k0]['quant'],
                    log='all', returnas=str,
                )
                if group2d in dd[kk]['group']
                and not isinstance(dd[kk]['data'], dict)
            ]
            for k0 in lref1d
        }
        dref2d = {k0: v0 for k0, v0 in dref2d.items() if len(v0) > 0}

        if len(dref2d) == 0:
            msg = (
                "No match for (ref1d, ref2d) for ddata['{}']".format(kq)
            )
            raise Exception(msg)

        # check ref1d
        if ref1d is None:
            if ref2d is not None:
                lk = [k0 for k0, v0 in dref2d.items() if ref2d in v0]
                if len(lk) == 0:
                    msg = (
                        "\nNon-valid interpolation intermediate\n"
                        + "\t- provided:\n"
                        + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                        + "\t- valid:\n{}".format(
                            '\n'.join([
                                '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                                    k0, v0
                                )
                                for k0, v0 in dref2d.items()
                            ])
                        )
                    )
                    raise Exception(msg)
                if kq in lk:
                    ref1d = kq
                else:
                    ref1d = lk[0]
            else:
                if kq in dref2d.keys():
                    ref1d = kq
                else:
                    ref1d = list(dref2d.keys())[0]
        else:
            ref1d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref1d, group=group1d,
                msgstr='ref1d', raise_=False,
            )
        if ref1d not in dref2d.keys():
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

        # check ref2d
        if ref2d is None:
            ref2d = dref2d[ref1d][0]
        else:
            ref2d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref2d, group=group2d,
                msgstr='ref2d', raise_=False,
            )
        if ref2d not in dref2d[ref1d]:
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

    return kq, ref1d, ref2d


# #############################################################################
# #############################################################################
#                           Mesh2DRect - interp
# #############################################################################


def _interp2d_check_RZ(
    R=None,
    Z=None,
    grid=None,
):
    # R and Z provided
    if not isinstance(R, np.ndarray):
        try:
            R = np.atleast_1d(R).astype(float)
        except Exception as err:
            msg = "R must be convertible to np.arrays of floats"
            raise Exception(msg)
    if not isinstance(Z, np.ndarray):
        try:
            Z = np.atleast_1d(Z).astype(float)
        except Exception as err:
            msg = "Z must be convertible to np.arrays of floats"
            raise Exception(msg)

    # grid
    grid = ds._generic_check._check_var(
        grid, 'grid',
        default=R.shape != Z.shape,
        types=bool,
    )

    if grid is True and (R.ndim > 1 or Z.ndim > 1):
        msg = "If grid=True, R and Z must be 1d!"
        raise Exception(msg)
    elif grid is False and R.shape != Z.shape:
        msg = "If grid=False, R and Z must have the same shape!"
        raise Exception(msg)

    if grid is True:
        R = np.tile(R, Z.size)
        Z = np.repeat(Z, R.size)
    return R, Z, grid


def _interp2d_check(
    # ressources
    coll=None,
    # interpolation base, 1d or 2d
    key=None,
    # external coefs (optional)
    coefs=None,
    # interpolation points
    R=None,
    Z=None,
    radius=None,
    angle=None,
    grid=None,
    radius_vs_time=None,
    azone=None,
    # time: t or indt
    t=None,
    indt=None,
    indt_strict=None,
    # bsplines
    indbs=None,
    # parameters
    details=None,
    reshape=None,
    res=None,
    crop=None,
    nan0=None,
    val_out=None,
    imshow=None,
    return_params=None,
    store=None,
    inplace=None,
):

    # -------------
    # keys

    dk = {
        k0: v0['bsplines']
        for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') not in ['', None]
        and 'crop' not in k0
    }
    dk.update({kk: kk for kk in coll.dobj['bsplines'].keys()})
    if key is None and len(dk) == 1:
        key = list(dk.keys())[0]
    if key not in dk.keys():
        msg = (
            "Arg key must the key to a data referenced on a bsplines set\n"
            f"\t- available: {list(dk.keys())}\n"
            f"\t- provided: {key}\n"
        )
        raise Exception(msg)

    keybs = dk[key]
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    # -------------
    # details

    details = ds._generic_check._check_var(
        details, 'details',
        types=bool,
        default=False,
    )

    # -------------
    # crop

    crop = ds._generic_check._check_var(
        crop, 'crop',
        types=bool,
        default=mtype == 'rect',
        allowed=[False, True] if mtype == 'rect' else [False],
    )

    # -------------
    # nan0

    nan0 = ds._generic_check._check_var(
        nan0, 'nan0',
        types=bool,
        default=True,
    )

    # -------------
    # val_out

    val_out = ds._generic_check._check_var(
        val_out, 'val_out',
        default=np.nan,
        allowed=[False, np.nan, 0.]
    )

    # -----------
    # time

    # refbs and hastime
    refbs = coll.dobj['bsplines'][keybs]['ref']

    if mtype == 'polar':
        radius2d = coll.dobj[coll._which_mesh][keym]['radius2d']
        # take out key if bsplines
        lk = [kk for kk in [key, radius2d] if kk != keybs]

        # 
        hastime, reft, keyt, t, dind = coll.get_time_common(
            keys=lk,
            t=t,
            indt=indt,
            ind_strict=indt_strict,
        )

        rad2d_hastime = radius2d in dind.keys()
        if key == keybs:
            kind = radius2d
        else:
            kind = key
            hastime = key in dind.keys()

        if hastime:

            indt = dind[kind].get('ind')

            # Special case: all times match
            if indt is not None:
                rtk = coll.get_time(key)[2]
                if indt.size == coll.dref[rtk]['size']:
                    if np.allclose(indt, np.arange(0, coll.dref[rtk]['size'])):
                        indt = None

            if indt is not None:
                indtu = np.unique(indt)
                indtr = np.array([indt == iu for iu in indtu])
            else:
                indtu, indtr = None, None
        else:
            indt, indtu, indtr = None, None, None

    elif key != keybs:
        # hastime, t, indit
        hastime, hasvect, reft, keyt, t, dind = coll.get_time(
            key=key,
            t=t,
            indt=indt,
            ind_strict=indt_strict,
        )
        if dind is None:
            indt, indtu, indtr = None, None, None
        else:
            indt, indtu, indtr = dind['ind'], dind['indu'], dind['indr']
    else:
        hastime, hasvect = False, False
        reft, keyt, indt, indtu, indtr = None, None, None, None, None

    # -----------
    # indbs

    if details is True:
        if mtype == 'rect':
            returnas = 'tuple-flat'
        elif mtype == 'tri':
            returnas = int
        elif mtype == 'polar':
            if len(coll.dobj['bsplines'][keybs]['shape']) == 2:
                returnas = 'array-flat'
            else:
                returnas = int

        # compute validated indbs array with appropriate form
        indbs_tf = coll.select_ind(
            key=keybs,
            returnas=returnas,
            ind=indbs,
        )
    else:
        indbs_tf = None

    # -----
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
        allowed=[False] if details else [True, False],
    )

    # -----------
    # coordinates

    # (R, Z) vs (radius, angle)
    lc = [
        R is None and Z is None and radius is None,
        R is not None and Z is not None and store is False,
        (R is None and Z is None)
        and (radius is not None and mtype == 'polar') and store is False
    ]

    if not any(lc):
        msg = (
            "Please provide either:\n"
            "\t- R and Z: for any mesh type\n"
            "\t- radius (and angle): for polar mesh\n"
            "\t- None: for rect / tri mesh\n"
        )
        raise Exception(msg)

    # R, Z
    if lc[0]:

        if store is True:
            if imshow:
                msg = "Storing only works if imshow = False"
                raise Exception(msg)

        # no spec => sample mesh
        R, Z = coll.get_sample_mesh(
            key=keym,
            res=res,
            mode='abs',
            grid=True,
            R=R,
            Z=Z,
            imshow=imshow,
        )
        lc[1] = True

    if lc[1]:

        # check R, Z
        R, Z, grid = _interp2d_check_RZ(R=R, Z=Z, grid=grid)

        # special case if polar mesh => (radius, angle) from (R, Z)
        if mtype == 'polar':

            rad2d_indt = dind[radius2d].get('ind') if rad2d_hastime else None

            # compute radius2d at relevant times
            radius, _, _ = coll.interpolate_profile2d(
                # coordinates
                R=R,
                Z=Z,
                grid=False,
                # quantities
                key=radius2d,
                details=False,
                # time
                indt=rad2d_indt,
            )

            # compute angle2d at relevant times
            angle2d = coll.dobj[coll._which_mesh][keym]['angle2d']
            if angle2d is not None:
                angle, _, _ = coll.interpolate_profile2d(
                    # coordinates
                    R=R,
                    Z=Z,
                    grid=False,
                    # quantities
                    key=angle2d,
                    details=False,
                    # time
                    indt=rad2d_indt,
                )

            # simplify if not time-dependent
            if radius2d not in dind:
                assert radius.ndim == R.ndim
                if angle2d is not None:
                    assert angle.ndim == R.ndim

            # check consistency
            if rad2d_hastime != (radius.ndim == R.ndim + 1):
                msg = f"Inconsistency! {radius.shape}"
                raise Exception(msg)

            radius_vs_time = rad2d_hastime

        else:
            radius_vs_time = False

    else:
        radius_vs_time = ds._generic_check._check_var(
            radius_vs_time, 'radius_vs_time',
            types=bool,
            default=False,
        )

    # -------------
    # radius, angle

    if mtype == 'polar':

        # check same shape
        if not isinstance(radius, np.ndarray):
            radius = np.atleast_1d(radius)

        # angle vs angle2d
        angle2d = coll.dobj[coll._which_mesh][keym]['angle2d']
        if angle2d is not None and angle is None:
            msg = (
                f"Arg angle must be provided for bsplines {keybs}"
            )
            raise Exception(msg)

        # angle vs radius
        if angle is not None:
            if not isinstance(angle, np.ndarray):
                angle = np.atleast_1d(angle)

            if radius.shape != angle.shape:
                msg = (
                    "Args radius and angle must be np.ndarrays of same shape!\n"
                    f"\t- radius.shape: {radius.shape}\n"
                    f"\t- angle.shape: {angle.shape}\n"
                )
                raise Exception(msg)

    # -------------
    # coefs

    shapebs = coll.dobj['bsplines'][keybs]['shape']
    # 3 possible coefs shapes:
    #   - None (if details = True or key provided)
    #   - scalar or shapebs if details = False and key = keybs

    if details is True:
        coefs = None
        assert key == keybs, (key, keybs)

    else:
        if coefs is None:
            if key == keybs:
                if mtype == 'polar' and rad2d_hastime:
                    if rad2d_indt is None:
                        r2dnt = coll.dref[dind[radius2d]['ref']]['size']
                    else:
                        r2dnt = rad2d_indt.size
                    coefs = np.ones(tuple(np.r_[r2dnt, shapebs]), dtype=float)
                else:
                    coefs = np.ones(shapebs, dtype=float)
            else:
                coefs = coll.ddata[key]['data']

        elif key != keybs:
            msg = f"Arg coefs can only be provided if key = keybs!\n\t- key: {key}"
            raise Exception(msg)

        elif np.isscalar(coefs):
            coefs = np.full(shapebs, coefs)

        # consistency
        nshbs = len(shapebs)
        c0 = (
            coefs.shape[-nshbs:] == shapebs
            and (
                (hastime and coefs.ndim == nshbs + 1)           # pre-shaped
                or (not hastime and coefs.ndim == nshbs + 1)    # pre-shaped
                or (not hastime and coefs.ndim == nshbs)        # [None, ...]
            )
        )
        if not c0:
            msg = (
                f"Inconsistency of '{key}' shape:\n"
                f"\t- shape: {coefs.shape}\n"
                f"\t- shapebs: {shapebs}\n"
                f"\t- hastime: {hastime}\n"
                f"\t- radius_vs_time: {radius_vs_time}\n"
            )
            raise Exception(msg)

        # Make sure coefs is time dependent
        if hastime:
            if indt is not None and (mtype == 'polar' or indtu is None):
                if coefs.shape[0] != indt.size:
                    # in case coefs is already provided with indt
                    coefs = coefs[indt, ...]

            if radius_vs_time and coefs.shape[0] != radius.shape[0]:
                msg = (
                    "Inconstistent coefs vs radius!\n"
                    f"\t- coefs.shape = {coefs.shape}\n"
                    f"\t- radius.shape = {radius.shape}\n"
                )
                raise Exception(msg)
        else:
            if radius_vs_time is True:
                sh = tuple([radius.shape[0]] + [1]*len(shapebs))
                coefs = np.tile(coefs, sh)
            elif coefs.ndim == nshbs:
                coefs = coefs[None, ...]

    # -------------
    # azone

    azone = ds._generic_check._check_var(
        azone, 'azone',
        types=bool,
        default=True,
    )

    # -------------
    # return_params

    return_params = ds._generic_check._check_var(
        return_params, 'return_params',
        types=bool,
        default=False,
    )

    # -------
    # inplace

    inplace = ds._generic_check._check_var(
        inplace, 'inplace',
        types=bool,
        default=store,
    )

    return (
        key, keybs,
        R, Z,
        radius, angle,
        coefs,
        hastime,
        reft, keyt,
        shapebs,
        radius_vs_time,
        azone,
        indbs, indbs_tf,
        t, indt, indtu, indtr,
        details, crop,
        nan0, val_out,
        return_params,
        store, inplace,
    )


def interp2d(
    # ressources
    coll=None,
    # interpolation base, 1d or 2d
    key=None,
    # external coefs (instead of key, optional)
    coefs=None,
    # interpolation points
    R=None,
    Z=None,
    radius=None,
    angle=None,
    grid=None,
    radius_vs_time=None,        # if radius is provided, in case radius vs time 
    azone=None,                 # if angle2d is interpolated, exclusion zone
    # time: t or indt
    t=None,
    indt=None,
    indt_strict=None,
    # bsplines
    indbs=None,
    # parameters
    details=None,
    reshape=None,
    res=None,
    crop=None,
    nan0=None,
    val_out=None,
    imshow=None,
    return_params=None,
    store=None,
    inplace=None,
):

    # ---------------
    # check inputs

    (
        key, keybs,
        R, Z,
        radius, angle,
        coefs,
        hastime,
        reft, keyt,
        shapebs,
        radius_vs_time,
        azone,
        indbs, indbs_tf,
        t, indt, indtu, indtr,
        details, crop,
        nan0, val_out,
        return_params,
        store, inplace,
    ) = _interp2d_check(
        # ressources
        coll=coll,
        # interpolation base, 1d or 2d
        key=key,
        # external coefs (optional)
        coefs=coefs,
        # interpolation points
        R=R,
        Z=Z,
        radius=radius,
        angle=angle,
        grid=grid,
        radius_vs_time=radius_vs_time,
        azone=azone,
        # time: t or indt
        t=t,
        indt=indt,
        indt_strict=indt_strict,
        # bsplines
        indbs=indbs,
        # parameters
        details=details,
        res=res,
        crop=crop,
        nan0=nan0,
        val_out=val_out,
        imshow=imshow,
        return_params=return_params,
        store=store,
        inplace=inplace,
    )
    keym = coll.dobj['bsplines'][keybs]['mesh']
    meshtype = coll.dobj['mesh'][keym]['type']

    # ----------------------
    # which function to call

    if details is True:
        fname = 'func_details'
    elif details is False:
        fname = 'func_sum'
    else:
        raise Exception("Unknown details!")

    # ---------------
    # cropping ?

    cropbs = coll.dobj['bsplines'][keybs]['crop']
    if cropbs not in [None, False]:
        cropbs = coll.ddata[cropbs]['data']
        nbs = cropbs.sum()
    else:
        nbs = np.prod(coll.dobj['bsplines'][keybs]['shape'])

    if indbs_tf is not None:
        if isinstance(indbs_tf, tuple):
            nbs = indbs_tf[0].size
        else:
            nbs = indbs_tf.size

    # -----------
    # Interpolate

    if meshtype in ['rect', 'tri']:

        # manage time
        if indtu is not None:
            val0 = np.full(tuple(np.r_[indt.size, R.shape]), np.nan)
            coefs = coefs[indtu, ...]

        val = coll.dobj['bsplines'][keybs][fname](
            R=R,
            Z=Z,
            coefs=coefs,
            crop=crop,
            cropbs=cropbs,
            indbs_tf=indbs_tf,
            val_out=val_out,
        )

        # manage time
        if indtu is not None:
            for ii, iu in enumerate(indtu):
                val0[indtr[ii], ...] = val[ii, ...]
            val = val0

        shape_pts = R.shape

    elif meshtype == 'polar':

        val = coll.dobj['bsplines'][keybs][fname](
            radius=radius,
            angle=angle,
            coefs=coefs,
            indbs_tf=indbs_tf,
            radius_vs_time=radius_vs_time,
            val_out=val_out,
        )

        shape_pts = radius.shape

    # ---------------
    # post-treatment for angle2d only (discontinuity)

    if azone is True:
        lkm0 = [
            k0 for k0, v0 in coll.dobj[coll._which_mesh].items()
            if key == v0.get('angle2d')
        ]
        if len(lkm0) > 0:
            ind = angle2d_inzone(
                coll=coll,
                keym0=lkm0[0],
                keya2d=key,
                R=R,
                Z=Z,
                t=t,
                indt=indt,
            )
            assert val.shape == ind.shape
            val[ind] = np.pi

    # ---------------
    # post-treatment

    if nan0 is True:
        val[val == 0] = np.nan

    if not hastime and not radius_vs_time:
        c0 = (
            (
                details is False
                and val.shape == tuple(np.r_[1, shape_pts])
            )
            or (
                details is True
                and val.shape == tuple(np.r_[shape_pts, nbs])
            )
        )
        if not c0:
            import pdb; pdb.set_trace()     # DB
            pass
        if details is False:
            val = val[0, ...]
            reft = None

    # ------
    # store

    # ref
    ct = (
        (hastime or radius_vs_time)
        and (
            reft in [None, False]
            or (
                reft not in [None, False]
                and indt is not None
                and not (
                    indt.size == coll.dref[reft]['size']
                    or np.allclose(indt, np.arange(0, coll.dref[reft]['size']))
                )
            )
        )
    )
    if ct:
        reft = f'{key}-nt'

    # store
    if store is True:
        Ru = np.unique(R)
        Zu = np.unique(Z)
        nR, nZ = Ru.size, Zu.size

        knR, knZ = f'{key}-nR', f'{key}-nZ'
        kR, kZ = f'{key}-R', f'{key}-Z'

        if inplace is True:
            coll2 = coll
        else:
            coll2 = ds.DataStock()

        # add ref nR, nZ
        coll2.add_ref(key=knR, size=nR)
        coll2.add_ref(key=knZ, size=nZ)

        # add data Ru, Zu
        coll2.add_data(
            key=kR,
            data=Ru,
            ref=knR,
            dim='distance',
            name='R',
            units='m',
        )
        coll2.add_data(
            key=kZ,
            data=Zu,
            ref=knZ,
            dim='distance',
            name='Z',
            units='m',
        )

        # ref
        if hastime or radius_vs_time:
            if ct:
                coll2.add_ref(key=reft, size=t.size)
                coll2.add_data(
                    key=f'{key}-t',
                    data=t,
                    ref=reft,
                    dim='time',
                    units='s',
                )
            else:
                if reft not in coll2.dref.keys():
                    coll2.add_ref(key=reft, size=coll.dref[reft]['size'])
                if keyt is not None and keyt not in coll2.ddata.keys():
                    coll2.add_data(
                        key=keyt,
                        data=coll.ddata[keyt]['data'],
                        dim='time',
                    )

            ref = (reft, knR, knZ)
        else:
            ref = (knR, knZ)

        coll2.add_data(
            data=val,
            key=f'{key}-map',
            ref=ref,
            dim=coll.ddata[key]['dim'],
            quant=coll.ddata[key]['quant'],
            name=coll.ddata[key]['name'],
            units=coll.ddata[key]['units'],
        )

    else:

        ref = []
        c0 = (
            reft not in [None, False]
            and (hastime or radius_vs_time)
            and not (
                meshtype == 'polar'
                and R is None
                and coefs is None
                and radius_vs_time is False
            )
        )
        if c0:
            ref.append(reft)

        if meshtype in ['rect', 'tri']:
            for ii in range(R.ndim):
                ref.append(None)
            if grid is True:
                for ii in range(Z.ndim):
                    ref.append(None)
        else:
            for ii in range(radius.ndim):
                if radius_vs_time and ii == 0:
                    continue
                ref.append(None)
            if grid is True and angle is not None:
                for ii in range(angle.ndim):
                    ref.append(None)

        if details is True:
            refbs = coll.dobj['bsplines'][keybs]['ref-bs'][0]
            if crop is True:
                refbs = f"{refbs}-crop"
            ref.append(refbs)
        ref = tuple(ref)

        if ref[0] == 'emiss-nt':
            import pdb; pdb.set_trace()     # DB

        if len(ref) != val.ndim:
            msg = (
                "Mismatching ref vs val.shape:\n"
                f"\t- key = {key}\n"
                f"\t- keybs = {keybs}\n"
                f"\t- val.shape = {val.shape}\n"
                f"\t- ref = {ref}\n"
                f"\t- reft = {reft}\n"
                f"\t- hastime = {hastime}\n"
                f"\t- radius_vs_time = {radius_vs_time}\n"
                f"\t- details = {details}\n"
                f"\t- indbs_tf = {indbs_tf}\n"
                f"\t- key = {key}\n"
                f"\t- meshtype = {meshtype}\n"
                f"\t- grid = {grid}\n"
            )
            if coefs is not None:
                msg += f"\t- coefs.shape = {coefs.shape}\n"
            if R is not None:
                msg += (
                    f"\t- R.shape = {R.shape}\n"
                    f"\t- Z.shape = {Z.shape}\n"
                )
            if meshtype == 'polar':
                msg += f"\t- radius.shape = {radius.shape}\n"
                if angle is not None:
                    msg += f"\t- angle.shape = {angle.shape}\n"
            raise Exception(msg)

    # ------
    # return

    if store and inplace is False:
        return coll2
    else:
        if return_params is True:
            return val, t, ref, dparams
        else:
            return val, t, ref


# #############################################################################
# #############################################################################
#                   2d points to 1d quantity interpolation
# #############################################################################


def interp2dto1d(
    coll=None,
    key=None,
    R=None,
    Z=None,
    indbs=None,
    indt=None,
    grid=None,
    details=None,
    reshape=None,
    res=None,
    crop=None,
    nan0=None,
    imshow=None,
    return_params=None,
):

    # ---------------
    # check inputs

    # TBD
    pass

    # ---------------
    # post-treatment

    if nan0 is True:
        val[val == 0] = np.nan

    # ------
    # return

    if return_params is True:
        return val, dparams
    else:
        return val


# #############################################################################
# #############################################################################
#                           Mesh2DRect - operators
# #############################################################################


def get_bsplines_operator(
    coll,
    key=None,
    operator=None,
    geometry=None,
    crop=None,
    store=None,
    returnas=None,
    # specific to deg = 0
    centered=None,
    # to return gradR, gradZ, for D1N2 deg 0, for tomotok
    returnas_element=None,
):

    # check inputs
    lk = list(coll.dobj.get('bsplines', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lk,
    )

    store = ds._generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=store is False,
        types=bool,
    )

    crop = ds._generic_check._check_var(
        crop, 'crop',
        default=True,
        types=bool,
    )

    # cropbs
    cropbs = coll.dobj['bsplines'][key]['crop']
    keycropped = coll.dobj['bsplines'][key]['ref-bs'][0]
    if cropbs not in [None, False] and crop is True:
        cropbs_flat = coll.ddata[cropbs]['data'].ravel(order='F')
        if coll.dobj['bsplines'][key]['deg'] == 0:
            cropbs = coll.ddata[cropbs]['data']
        keycropped = f"{keycropped}-crop"
    else:
        cropbs = False
        cropbs_flat = False

    # compute and return
    (
        opmat, operator, geometry, dim,
    ) = coll.dobj['bsplines'][key]['class'].get_operator(
        operator=operator,
        geometry=geometry,
        cropbs_flat=cropbs_flat,
        # specific to deg=0
        cropbs=cropbs,
        centered=centered,
        # to return gradR, gradZ, for D1N2 deg 0, for tomotok
        returnas_element=returnas_element,
    )

    # cropping
    if operator == 'D1':
        ref = (keycropped, keycropped)
    elif operator == 'D0N1':
        ref = (keycropped,)
    elif 'N2' in operator:
        ref = (keycropped, keycropped)

    return opmat, operator, geometry, dim, ref, crop, store, returnas, key


# #################################################################
# #################################################################
#               Contour computation
# #################################################################


def _get_contours(
    RR=None,
    ZZ=None,
    val=None,
    levels=None,
    largest=None,
    uniform=None,
):
    """ Return R, Z coordinates of contours (time-dependent)

    For contourpy algorithm, the dimensions shoud be (ny, nx), from meshgrid

    RR = (nz, nr)
    ZZ = (nz, nr)
    val = (nt, nz, nr)
    levels = (nlevels,)

    cR = (nt, nlevels, nmax) array of R coordinates
    cZ = (nt, nlevels, nmax) array of Z coordinates

    The contour coordinates are uniformzied to always have the same nb of pts

    """

    # -------------
    # check inputs

    if largest is None:
        largest = False

    if uniform is None:
        uniform = True

    # val.shape = (nt, nR, nZ)
    lc = [
        val.shape == RR.shape,
        val.ndim == RR.ndim + 1 and val.shape[1:] == RR.shape,
    ]
    if lc[0]:
        val = val[None, ...]
    elif lc[1]:
        pass
    else:
        msg = "Incompatible val.shape!"
        raise Exception(msg)

    nt, nR, nZ = val.shape

    # ------------------------
    # Compute list of contours

    # compute contours at rknots
    # see https://github.com/matplotlib/matplotlib/blob/main/src/_contour.h

    contR = [[] for ii in range(nt)]
    contZ = [[] for ii in range(nt)]
    for ii in range(nt):

        # define map
        contgen = contour_generator(
            x=RR,
            y=ZZ,
            z=val[ii, ...],
            name='serial',
            corner_mask=None,
            line_type='Separate',
            fill_type=None,
            chunk_size=None,
            chunk_count=None,
            total_chunk_count=None,
            quad_as_tri=True,       # for sub-mesh precision
            # z_interp=<ZInterp.Linear: 1>,
            thread_count=0,
        )

        for jj in range(len(levels)):

            # compute concatenated contour
            no_cont = False
            cj = contgen.lines(levels[jj])

            c0 = (
                isinstance(cj, list)
                and all([
                    isinstance(cjj, np.ndarray)
                    and cjj.ndim == 2
                    and cjj.shape[1] == 2
                    for cjj in cj
                ])
            )
            if not c0:
                msg = f"Wrong output from contourpy!\n{cj}"
                raise Exception(msg)

            if len(cj) > 0:
                cj = [
                    cc[np.all(np.isfinite(cc), axis=1), :]
                    for cc in cj
                    if np.sum(np.all(np.isfinite(cc), axis=1)) >= 3
                ]

                if len(cj) == 0:
                    no_cont = True
                elif len(cj) == 1:
                    cj = cj[0]
                elif len(cj) > 1:
                    if largest:
                        nj = [
                            0.5*np.abs(np.sum(
                                (cc[1:, 0] + cc[:-1, 0])
                                *(cc[1:, 1] - cc[:-1, 1])
                            ))
                            for cc in cj
                        ]
                        cj = cj[np.argmax(nj)]
                    else:
                        ij = np.cumsum([cc.shape[0] for cc in cj])
                        cj = np.concatenate(cj, axis=0)
                        cj = np.insert(cj, ij, np.nan, axis=0)

                elif np.sum(np.all(~np.isnan(cc), axis=1)) < 3:
                    no_cont = True
            else:
                no_cont = True

            if no_cont is True:
                cj = np.full((3, 2), np.nan)

            contR[ii].append(cj[:, 0])
            contZ[ii].append(cj[:, 1])

    # ------------------------------------------------
    # Interpolate / concatenate to uniformize as array

    if uniform:
        ln = [[pp.size for pp in cc] for cc in contR]
        nmax = np.max(ln)
        cR = np.full((nt, len(levels), nmax), np.nan)
        cZ = np.full((nt, len(levels), nmax), np.nan)

        for ii in range(nt):
            for jj in range(len(levels)):
                cR[ii, jj, :] = np.interp(
                    np.linspace(0, ln[ii][jj], nmax),
                    np.arange(0, ln[ii][jj]),
                    contR[ii][jj],
                )
                cZ[ii, jj, :] = np.interp(
                    np.linspace(0, ln[ii][jj], nmax),
                    np.arange(0, ln[ii][jj]),
                    contZ[ii][jj],
                )

        return cR, cZ
    else:
        return contR, contZ


# #############################################################################
# #############################################################################
#                   Polygon simplification
# #############################################################################


def _simplify_polygon(pR=None, pZ=None, res=None):
    """ Use convex hull with a constraint on the maximum discrepancy """

    # ----------
    # preliminary 1: check there is non redundant point

    dp = np.sqrt((pR[1:] - pR[:-1])**2 + (pZ[1:] - pZ[:-1])**2)
    ind = (dp > 1.e-6).nonzero()[0]
    pR = pR[ind]
    pZ = pZ[ind]

    # check new poly is closed
    if (pR[0] != pR[-1]) or (pZ[0] != pZ[-1]):
        pR = np.append(pR, pR[0])
        pZ = np.append(pZ, pZ[0])

    # check it is counter-clockwise
    clock = np.nansum((pR[1:] - pR[:-1]) * (pZ[1:] + pZ[:-1]))
    if clock > 0:
        pR = pR[::-1]
        pZ = pZ[::-1]

    # threshold = diagonal of resolution + 10%
    thresh = res * np.sqrt(2) * 1.1

    # ----------
    # preliminary 2: get convex hull and copy

    poly = np.array([pR, pZ]).T
    iconv = ConvexHull(poly, incremental=False).vertices

    # close convex hull to iterate on edges
    pR_conv = np.append(pR[iconv], pR[iconv[0]])
    pZ_conv = np.append(pZ[iconv], pZ[iconv[0]])

    # copy to create new polygon that will serve as buffer
    pR_bis, pZ_bis = np.copy(pR), np.copy(pZ)

    # -------------------------
    # loop on convex hull edges

    for ii in range(pR_conv.size - 1):

        pR1, pR2 = pR_conv[ii], pR_conv[ii+1]
        pZ1, pZ2 = pZ_conv[ii], pZ_conv[ii+1]
        i0 = np.argmin(np.hypot(pR_bis - pR1, pZ_bis - pZ1))

        # make sure it starts from p1
        pR_bis = np.append(pR_bis[i0:], pR_bis[:i0])
        pZ_bis = np.append(pZ_bis[i0:], pZ_bis[:i0])

        # get indices of closest points to p1, p2
        i1 = np.argmin(np.hypot(pR_bis - pR1, pZ_bis - pZ1))
        i2 = np.argmin(np.hypot(pR_bis - pR2, pZ_bis - pZ2))

        # get corresponding indices of poly points to be included
        if i2 == i1 + 1:
            itemp = [i1, i2]

        else:
            # several points in-between
            # => check they are all within distance before exclusing them

            # get unit vector of segment
            norm12 = np.hypot(pR2 - pR1, pZ2 - pZ1)
            u12R = (pR2 - pR1) / norm12
            u12Z = (pZ2 - pZ1) / norm12

            # get points standing between p1 nd p2
            lpR = pR_bis[i1 + 1:i2]
            lpZ = pZ_bis[i1 + 1:i2]

            # indices of points standing too far from edge (use cross-product)
            iout = np.abs(u12R*(lpZ - pZ1) - u12Z*(lpR - pR1)) > thresh

            # if any pts too far => include all pts
            if np.any(iout):
                itemp = np.arange(i1, i2 + 1)
            else:
                itemp = [i1, i2]

        # build pts_in
        pR_in = pR_bis[itemp]
        pZ_in = pZ_bis[itemp]

        # concatenate to add to new polygon
        pR_bis = np.append(pR_in, pR_bis[i2 + 1:])
        pZ_bis = np.append(pZ_in, pZ_bis[i2 + 1:])

    # check new poly is closed
    if (pR_bis[0] != pR_bis[-1]) or (pZ_bis[0] != pZ_bis[-1]):
        pR_bis = np.append(pR_bis, pR_bis[0])
        pZ_bis = np.append(pZ_bis, pZ_bis[0])

    return pR_bis, pZ_bis


# #############################################################################
# #############################################################################
#                   radius2d special points handling
# #############################################################################


def radius2d_special_points(
    coll=None,
    key=None,
    keym0=None,
    res=None,
):

    keybs = coll.ddata[key]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    assert mtype in ['rect', 'tri']

    # get map sampling
    RR, ZZ = coll.get_sample_mesh(
        key=keym,
        res=res,
        grid=True,
    )

    # get map
    val, t, _ = coll.interpolate_profile2d(
        key=key,
        R=RR,
        Z=ZZ,
        grid=False,
        imshow=True,        # for contour
    )

    # get min max values
    rmin = np.nanmin(val)
    rmax = np.nanmax(val)

    # get contour of 0
    cR, cZ = _get_contours(
        RR=RR,
        ZZ=ZZ,
        val=val,
        levels=[rmin + 0.05*(rmax-rmin)],
    )

    # dref
    ref_O = f'{keym0}-pts-O-n'
    dref = {
        ref_O: {'size': 1},
    }

    # get barycenter 
    if val.ndim == 3:
        assert cR.shape[1] == 1
        ax_R = np.nanmean(cR[:, 0, :], axis=-1)[:, None]
        ax_Z = np.nanmean(cZ[:, 0, :], axis=-1)[:, None]
        reft = coll.ddata[key]['ref'][0]
        ref = (reft, ref_O)
    else:
        ax_R = np.r_[np.nanmean(cR)]
        ax_Z = np.r_[np.nanmean(cZ)]
        ref = (ref_O,)

    kR = f'{keym0}-pts-O-R'
    kZ = f'{keym0}-pts-O-Z'
    ddata = {
        kR: {
            'ref': ref,
            'data': ax_R,
            'dim': 'distance',
            'quant': 'R',
            'name': 'O-points_R',
            'units': 'm',
        },
        kZ: {
            'ref': ref,
            'data': ax_Z,
            'dim': 'distance',
            'quant': 'Z',
            'name': 'O-points_Z',
            'units': 'm',
        },
    }

    return dref, ddata, kR, kZ


# #############################################################################
# #############################################################################
#                   angle2d discontinuity handling
# #############################################################################


def angle2d_zone(
    coll=None,
    key=None,
    keyrad2d=None,
    key_ptsO=None,
    res=None,
    keym0=None,
):

    keybs = coll.ddata[key]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    assert mtype in ['rect', 'tri']

    # --------------
    # prepare

    hastime, hasvect, reft, keyt = coll.get_time(key=key)[:4]
    if hastime:
        nt = coll.dref[reft]['size']
    else:
        msg = (
            "Non time-dependent angle2d not implemented yet\n"
            "=> ping @Didou09 on Github to open an issue"
        )
        raise NotImplementedError(msg)

    if res is None:
        res = _get_sample_mesh_res(
            coll=coll,
            keym=keym,
            mtype=mtype,
        )

    # get map sampling
    RR, ZZ = coll.get_sample_mesh(
        key=keym,
        res=res/2.,
        grid=True,
        imshow=True,    # for contour
    )

    # get map
    val, t, _ = coll.interpolate_profile2d(
        key=key,
        R=RR,
        Z=ZZ,
        grid=False,
        azone=False,
    )
    val[np.isnan(val)] = 0.
    amin = np.nanmin(val)
    amax = np.nanmax(val)

    # get contours of absolute value
    cRmin, cZmin = _get_contours(
        RR=RR,
        ZZ=ZZ,
        val=val,
        levels=[amin + 0.10*(amax - amin)],
        largest=True,
        uniform=True,
    )
    cRmax, cZmax = _get_contours(
        RR=RR,
        ZZ=ZZ,
        val=val,
        levels=[amax - 0.10*(amax - amin)],
        largest=True,
        uniform=True,
    )

    cRmin, cZmin = cRmin[:, 0, :], cZmin[:, 0, :]
    cRmax, cZmax = cRmax[:, 0, :], cZmax[:, 0, :]

    rmin = np.full(cRmin.shape, np.nan)
    rmax = np.full(cRmax.shape, np.nan)

    # get points inside contour 
    for ii in range(nt):
        rmin[ii, :], _, _ = coll.interpolate_profile2d(
            key=keyrad2d,
            R=cRmin[ii, :],
            Z=cZmin[ii, :],
            grid=False,
            indt=ii,
        )
        rmax[ii, :], _, _ = coll.interpolate_profile2d(
            key=keyrad2d,
            R=cRmax[ii, :],
            Z=cZmax[ii, :],
            grid=False,
            indt=ii,
        )

    # get magnetic axis
    kR, kZ = key_ptsO
    axR = coll.ddata[kR]['data']
    axZ = coll.ddata[kZ]['data']
    assert coll.ddata[kR]['ref'][0] == coll.ddata[key]['ref'][0]

    start_min = np.nanargmin(rmin, axis=-1)
    start_max = np.nanargmin(rmax, axis=-1)

    # re-order from start_min, start_max
    lpR, lpZ = [], []
    for ii in range(rmin.shape[0]):
        imin = np.r_[
            np.arange(start_min[ii], rmin.shape[1]),
            np.arange(0, start_min[ii]),
        ]

        cRmin[ii] = cRmin[ii, imin]
        cZmin[ii] = cZmin[ii, imin]
        rmin[ii] = rmin[ii, imin]
        # check it is counter-clockwise
        clock = np.nansum(
            (cRmin[ii, 1:] - cRmin[ii, :-1])
            *(cZmin[ii, 1:] + cZmin[ii, :-1])
        )
        if clock > 0:
            cRmin[ii, :] = cRmin[ii, ::-1]
            cZmin[ii, :] = cZmin[ii, ::-1]
            rmin[ii, :] = rmin[ii, ::-1]

        imax = np.r_[
            np.arange(start_max[ii], rmax.shape[1]),
            np.arange(0, start_max[ii])
        ]
        cRmax[ii] = cRmax[ii, imax]
        cZmax[ii] = cZmax[ii, imax]
        rmax[ii] = rmax[ii, imax]
        # check it is clockwise
        clock = np.nansum(
            (cRmax[ii, 1:] - cRmax[ii, :-1])
            *(cZmax[ii, 1:] + cZmax[ii, :-1])
        )
        if clock < 0:
            cRmax[ii, :] = cRmax[ii, ::-1]
            cZmax[ii, :] = cZmax[ii, ::-1]
            rmax[ii, :] = rmax[ii, ::-1]

        # i0
        dr = np.diff(rmin[ii, :])
        i0 = (np.isnan(dr) | (dr < 0)).nonzero()[0][0]
        # rmin[ii, i0-1:] = np.nan
        dr = np.diff(rmax[ii, :])
        i1 = (np.isnan(dr) | (dr < 0)).nonzero()[0][0]
        # rmax[ii, i1-1:] = np.nan

        # polygon
        pR = np.r_[axR[ii], cRmin[ii, :i0-1], cRmax[ii, :i1-1][::-1]]
        pZ = np.r_[axZ[ii], cZmin[ii, :i0-1], cZmax[ii, :i1-1][::-1]]

        pR, pZ = _simplify_polygon(pR=pR, pZ=pZ, res=res)

        lpR.append(pR)
        lpZ.append(pZ)

    # Ajust sizes
    nb = np.array([pR.size for pR in lpR])

    # 
    nmax = np.max(nb)
    pR = np.full((nt, nmax), np.nan)
    pZ = np.full((nt, nmax), np.nan)

    for ii in range(nt):
        pR[ii, :] = np.interp(
            np.linspace(0, nb[ii], nmax),
            np.arange(0, nb[ii]),
            lpR[ii],
        )
        pZ[ii, :] = np.interp(
            np.linspace(0, nb[ii], nmax),
            np.arange(0, nb[ii]),
            lpZ[ii],
        )

    # ----------------
    # prepare output dict

    # ref
    kref = f'{keym0}-azone-npt'
    dref = {
        kref: {'size': nmax}
    }

    # data
    kR = f'{keym0}-azone-R'
    kZ = f'{keym0}-azone-Z'
    ddata = {
        kR: {
            'data': pR,
            'ref': (reft, kref),
            'units': 'm',
            'dim': 'distance',
            'quant': 'R',
            'name': None,
        },
        kZ: {
            'data': pZ,
            'ref': (reft, kref),
            'units': 'm',
            'dim': 'distance',
            'quant': 'R',
            'name': None,
        },
    }

    return dref, ddata, kR, kZ


def angle2d_inzone(
    coll=None,
    keym0=None,
    keya2d=None,
    R=None,
    Z=None,
    t=None,
    indt=None,
):


    # ------------
    # prepare points

    if R.ndim == 1:
        shape0 = None
        pts = np.array([R, Z]).T
    else:
        shape0 = R.shape
        pts = np.array([R.ravel(), Z.ravel()]).T

    # ------------
    # prepare path

    kazR, kazZ = coll.dobj[coll._which_mesh][keym0]['azone']
    pR = coll.ddata[kazR]['data']
    pZ = coll.ddata[kazZ]['data']

    hastime, hasvect, reft, keyt, tnew, dind = coll.get_time(
        key=kazR,
        t=t,
        indt=indt,
    )

    # ------------
    # test points

    if hastime:
        if dind is None:
            nt = coll.dref[reft]['size']
            ind = np.zeros((nt, R.size), dtype=bool)
            for ii in range(nt):
                path = Path(np.array([pR[ii, :], pZ[ii, :]]).T)
                ind[ii, :] = path.contains_points(pts)
        else:
            import pdb; pdb.set_trace()     # DB
            raise NotImplementedError()
            # TBC / TBF
            nt = None
            ind = np.zeros((nt, R.size), dtype=bool)
            for ii in range(nt):
                path = Path(np.array([pR[ii, :], pZ[ii, :]]).T)
                ind[ii, :] = path.contains_points(pts)

    else:
        path = Path(np.array([pR, pZ]).T)
        ind = path.contains_points(pts)

    # -------------------------
    # fromat output and return

    if shape0 is not None:
        if hastime:
            ind = ind.reshape(tuple(np.r_[nt, shape0]))
        else:
            ind = ind.reshape(shape0)

    return ind
