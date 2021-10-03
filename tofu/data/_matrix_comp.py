# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np


# tofu
from . import _generic_check
from . import _mesh_bsplines


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def _compute_check(
    mesh=None,
    key=None,
    method=None,
    resMode=None,
    crop=None,
    name=None,
    verb=None,
):

    # key
    lk = list(mesh.dobj.get('bsplines', {}).keys())
    if key is None and len(lk) == 1:
        key = lk[0]
    if key not in lk:
        msg = (
            "Arg key must be a valid bspline identifier!\n"
            f"\t- available: {lk}\n"
            f"\t- provided:  {key}"
        )
        raise Exception(msg)

    # method
    method = _generic_check._check_var(
        method, 'method',
        default='los',
        types=str,
        allowed=['los'],
    )

    # resMode
    resMode = _generic_check._check_var(
        resMode, 'resMode',
        default='abs',
        types=str,
        allowed=['abs', 'rel'],
    )

    # crop
    crop = _generic_check._check_var(
        crop, 'crop',
        default=True,
        types=bool,
    )
    crop = crop and mesh.dobj['bsplines'][key]['crop'] is not False

    # name
    if name is None:
        lmat = [
            kk for kk in mesh.dobj.get('matrix', {}).keys()
            if kk.startswith('matrix')
        ]
        name = f'matrix{len(lmat)}'
    c0 = (
        isinstance(name, str)
        and name not in mesh.dobj.get('matrix', {}).keys()
    )
    if not c0:
        msg = (
            "Arg name must be a str not already taken!\n"
            f"\t- already taken: {mesh.dobj.get('matrix', {}).keys()}\n"
            f"\t- provided: {name}"
        )
        raise Exception(msg)

    # verb
    if verb is None:
        verb = True
    if not isinstance(verb, bool):
        msg = (
            f"Arg verb must be a bool!\n"
            f"\t- provided: {verb}"
        )
        raise Exception(msg)

    return key, method, resMode, crop, name, verb


def compute(
    mesh=None,
    key=None,
    cam=None,
    res=None,
    resMode=None,
    method=None,
    crop=None,
    name=None,
    verb=None,
):
    """ Compute the geometry matrix using:
            - a mesh2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """

    # -----------
    # check input

    key, method, resMode, crop, name, verb = _compute_check(
        mesh=mesh, key=key, method=method, resMode=resMode,
        crop=crop, name=name, verb=verb,
    )

    # -----------
    # prepare

    nlos = cam.nRays
    shapebs = mesh.dobj['bsplines'][key]['shape']

    # -----------
    # compute

    if method == 'los':
        # discretize lines once, then evaluated at points
        pts, reseff, ind = cam.get_sample(
            res=res,
            resMode=resMode,
            DL=None,
            method='sum',
            ind=None,
            pts=True,
            compact=True,
            num_threads=10,
            Test=True,
        )
        lr = np.split(np.hypot(pts[0, :], pts[1, :]), ind)
        lz = np.split(pts[2, :], ind)

        if verb:
            nmax = len(f"Geometry matrix, channel {nlos} / {nlos}")
            nn = 10**(np.log10(nlos)-1)

        # prepare indices
        indbs = mesh.select_ind(
            key=key,
            returnas=tuple,
            crop=crop,
        )

        # prepare matrix
        shapemat = tuple(np.r_[nlos, indbs[0].size])
        mat = np.zeros(shapemat, dtype=float)

        for ii in range(nlos):

            # verb
            if verb:
                msg = f"Geom. matrix, chan {ii+1} / {nlos}".ljust(nmax)
                end = '\n' if ii == nlos-1 else '\r'
                print(msg, end=end, flush=True)

            # compute
            mat[ii, :] = np.nansum(
                mesh.interp2d(
                    key=key,
                    R=lr[ii],
                    Z=lz[ii],
                    grid=False,
                    indbs=indbs,
                    details=True,
                    reshape=False,
                ),
                axis=0,
            )

        mat = mat * reseff[:, None]
        # scpintg.simps(val, x=None, axis=-1, dx=loc_eff_res[0])

    # -----------
    # return

    # extract existing parts relevant to the geometry matrix (mesh + bsplines)

    # dobj
    km = mesh.dobj['bsplines'][key]['mesh']
    dobj = copy.deepcopy({
        'mesh': {
            km: mesh.dobj['mesh'][km],
        },
        'bsplines': {
            key: mesh.dobj['bsplines'][key],
        },
    })

    # dref
    keycropped = f'{key}-cropped' if crop is True else key
    lref = (
        list(mesh.dobj['mesh'][km]['cents'])
        + list(mesh.dobj['mesh'][km]['knots'])
        + list(mesh.dobj['bsplines'][key]['ref'])
        + [key]
    )
    if crop is True:
        lref.append(keycropped)

    dref = {k0: mesh.dref[k0] for k0 in lref}
    for k0 in lref:
        dref[k0].update({
            k1: v1 for k1, v1 in mesh.ddata[k0].items()
            if k1 not in ['ref', 'group']
        })
    dref = copy.deepcopy(dref)

    # ddata
    lcrop = [mesh.dobj['mesh'][km]['crop'], mesh.dobj['bsplines'][key]['crop']]
    ddata = copy.deepcopy({
        k0: v0 for k0, v0 in mesh.ddata.items()
        if k0 in lcrop
    })

    # add new parts relevant to the geometry matrix (matrix + ref)
    dref.update({
        'channels': {
            'data': np.arange(0, nlos),
            'group': 'chan',
        },
    })

    ddata.update({
        name: {
            'data': mat,
            'ref': ('channels', keycropped)
        },
    })

    dobj.update({
        'matrix': {
            name: {
                'bsplines': key,
                'cam': cam.Id.Name,
                'data': name,
                'crop': crop,
                'shape': mat.shape,
            },
        },
    })

    return dref, ddata, dobj
