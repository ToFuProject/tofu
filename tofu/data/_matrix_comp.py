# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
from . import _mesh_checks
from . import _mesh_bsplines


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def _compute_check(mesh=None, key=None, method=None):

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

    return key, method


def compute(
    mesh=None,
    key=None,
    cam=None,
    res=None,
    resMode=None,
    method=None,
):
    """ Compute the geometry matrix using:
            - a mesh2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """


    # -----------
    # check input

    key, method = _compute_check(mesh=mesh, key=key, method=method)

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
        r, z = np.hypoth(pts[0, :], pts[1, :]), pts[2, :]

        import pdb; pdb.set_trace() # DB
        val = mesh.interp(key=key, R=r, Z=z, details=True)

        import pdb; pdb.set_trace() # DB
        sig = np.nansum(val)


        # scpintg.simps(val, x=None, axis=-1, dx=loc_eff_res[0])

    return dout
