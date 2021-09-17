# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np


# tofu
from . import _mesh_checks
from . import _mesh_bsplines


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def _compute_check(mesh=None, key=None, method=None, resMode=None, name=None):

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
    if method is None:
        method = 'los'
    lok = ['los']
    if method not in lok:
        msg = (
            f"Arg method must be in {lok}!\n"
            f"\t- provided: {method}"
        )
        raise Exception(msg)

    # resMode
    if resMode is None:
        resMode = 'abs'
    lok = ['abs', 'rel']
    if resMode not in lok:
        msg = (
            f"Arg resMode must be in {lok}!\n"
            f"\t- provided: {resMode}"
        )
        raise Exception(msg)

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

    return key, method, resMode, name


def compute(
    mesh=None,
    key=None,
    cam=None,
    res=None,
    resMode=None,
    method=None,
    name=None,
):
    """ Compute the geometry matrix using:
            - a mesh2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """


    # -----------
    # check input

    key, method, resMode, name = _compute_check(
        mesh=mesh, key=key, method=method, resMode=resMode, name=name,
    )

    # -----------
    # prepare

    nlos = cam.nRays
    shapebs = mesh.dobj['bsplines'][key]['shape']
    shapemat = tuple(np.r_[nlos, shapebs])
    mat = np.zeros(shapemat, dtype=float)

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

        for ii in range(nlos):
            mat[ii, ...] = np.nansum(
                mesh.interp(
                    key=key, R=lr[ii], Z=lz[ii],
                    grid=False, details=True,
                )[0, ...],
                axis=0,
            )

        if mat.ndim > 2:
            indflat = mesh.select_ind(key=key, returnas='tuple-flat')
            mat = mat[:, indflat[0], indflat[1]]

        mat = mat * reseff[:, None]
        # scpintg.simps(val, x=None, axis=-1, dx=loc_eff_res[0])

    # -----------
    # return

    # extract existing parts relevant to the geometry matrix (mesh + bsplines)
    km = mesh.dobj['bsplines'][key]['mesh']
    dobj = copy.deepcopy({
        'mesh': {
            km: mesh.dobj['mesh'][km],
        },
        'bsplines': {
            key: mesh.dobj['bsplines'][key],
        },
    })

    lref = (
        [
            mesh.dobj['mesh'][km][ss]
            for ss in ['R-knots', 'Z-knots', 'R-cents', 'Z-cents']
        ]
        + list(mesh.dobj['bsplines'][key]['ref'])
    )
    dref = {k0: mesh.dref[k0] for k0 in lref}
    for k0 in lref:
        dref[k0].update({
            k1: v1 for k1, v1 in mesh.ddata[k0].items()
            if k1 not in ['ref', 'group']
        })
    dref = copy.deepcopy(dref)

    # add new parts relevant to the geometry matrix (matrix + ref)
    dref.update({
        'channels': {
            'data': np.arange(0, nlos),
            'group': 'chan',
        },
        key: {
            'data': np.arange(0, mat.shape[1]),
            'group': 'bsplines',
        },
    })

    ddata = {
        name: {
            'data': mat,
            'ref': ('channels', key)
        },
    }

    dobj.update({
        'matrix': {
            name: {
                'bsplines': key,
                'cam': cam.Id.Name,
                'data': name,
            },
        },
    })

    return dref, ddata, dobj
