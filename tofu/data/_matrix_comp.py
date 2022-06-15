# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np


# tofu
from . import _generic_check


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def _compute_check(
    coll=None,
    key=None,
    method=None,
    resMode=None,
    crop=None,
    name=None,
    store=None,
    verb=None,
):

    # key
    lk = list(coll.dobj.get('bsplines', {}).keys())
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
    crop = crop and coll.dobj['bsplines'][key]['crop'] not in [None, False]

    # name
    if name is None:
        lmat = [
            kk for kk in coll.dobj.get('matrix', {}).keys()
            if kk.startswith('matrix')
        ]
        name = f'matrix{len(lmat)}'
    c0 = (
        isinstance(name, str)
        and name not in coll.dobj.get('matrix', {}).keys()
    )
    if not c0:
        msg = (
            "Arg name must be a str not already taken!\n"
            f"\t- already taken: {coll.dobj.get('matrix', {}).keys()}\n"
            f"\t- provided: {name}"
        )
        raise Exception(msg)

    # store
    store = _generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )

    # verb
    if verb is None:
        verb = True
    if not isinstance(verb, bool):
        msg = (
            f"Arg verb must be a bool!\n"
            f"\t- provided: {verb}"
        )
        raise Exception(msg)

    return key, method, resMode, crop, name, store, verb


def compute(
    coll=None,
    key=None,
    key_chan=None,
    cam=None,
    res=None,
    resMode=None,
    method=None,
    crop=None,
    name=None,
    store=None,
    verb=None,
):
    """ Compute the geometry matrix using:
            - a Plasma2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """

    # -----------
    # check input

    key, method, resMode, crop, name, store, verb = _compute_check(
        coll=coll, key=key, method=method, resMode=resMode,
        crop=crop, name=name, store=store, verb=verb,
    )

    # -----------
    # prepare

    nlos = cam.nRays
    shapebs = coll.dobj['bsplines'][key]['shape']
    km = coll.dobj['bsplines'][key]['mesh']
    mtype = coll.dobj[coll._which_mesh][km]['type']

    # prepare indices
    indbs = coll.select_ind(
        key=key,
        returnas=bool,
        crop=crop,
    )

    # prepare matrix
    is3d = False
    if mtype == 'polar':
        radius2d = coll.dobj[coll._which_mesh][km]['radius2d']
        r2d_reft = coll.get_time(key=radius2d)[2]
        if r2d_reft is not None:
            r2d_nt = coll.dref[r2d_reft]['size']
            if r2d_nt > 1:
                shapemat = tuple(np.r_[r2d_nt, nlos, indbs.sum()])
                is3d = True

    if not is3d:
        shapemat = tuple(np.r_[nlos, indbs.sum()])

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

        if verb:
            nmax = len(f"Geometry matrix for {key}, channel {nlos} / {nlos}")
            nn = 10**(np.log10(nlos)-1)

        for ii in range(nlos):

            # verb
            if verb:
                msg = f"Geom. matrix for {key}, chan {ii+1} / {nlos}"
                end = '\n' if ii == nlos-1 else '\r'
                print(msg.ljust(nmax), end=end, flush=True)

            # compute
            mati = coll.interpolate_profile2d(
                key=key,
                R=lr[ii],
                Z=lz[ii],
                grid=False,
                indbs=indbs,
                details=True,
                reshape=False,
                return_params=False,
            )[0]
            assert mati.ndim in [2, 3], mati.shape

            # integrate
            if is3d:
                mat[:, ii, :] = np.nansum(mati, axis=1) * reseff[ii]
            elif mati.ndim == 3 and mati.shape[0] == 1:
                mat[ii, :] = np.nansum(mati[0, ...], axis=0) * reseff[ii]
            else:
                mat[ii, :] = np.nansum(mati, axis=0) * reseff[ii]

        # scpintg.simps(val, x=None, axis=-1, dx=loc_eff_res[0])

    # -----------
    # return

    if store:

        # add key chan if necessary
        dref = None
        if key_chan is None:
            lrchan = [
                k0 for k0, v0 in coll.dref.items()
                if k0.startswith('chan') and k0[4:].isdecimal()
            ]
            if len(lrchan) == 0:
                chann = 0
            else:
                chann = max([int(k0.replace('chan', '')) for k0 in lrchan]) + 1
            key_chan = f'chan{chann}'

            dref = {
                key_chan: {
                    'data': np.arange(0, nlos),
                },
            }

        # add matrix data
        keycropped = coll.dobj['bsplines'][key]['ref-bs'][0]
        if crop is True:
            keycropped = f'{keycropped}-crop'

        # ref
        if is3d:
            ref = (r2d_reft, key_chan, keycropped)
        else:
            ref = (key_chan, keycropped)

        # add data
        ddata = {
            name: {
                'data': mat,
                'ref': ref,
            },
        }

        # add matrix obj
        dobj = {
            'matrix': {
                name: {
                    'bsplines': key,
                    'cam': cam.Id.Name,
                    'data': name,
                    'crop': crop,
                    'shape': mat.shape,
                },
            },
        }

        coll.update(dref=dref, ddata=ddata, dobj=dobj)

    else:
        return mat
