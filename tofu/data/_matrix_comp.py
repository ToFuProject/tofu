# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds

# tofu
from . import _generic_check


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def _compute_check(
    coll=None,
    key=None,
    key_chan=None,
    nlos=None,
    method=None,
    resMode=None,
    crop=None,
    name=None,
    store=None,
    verb=None,
):

    # key
    lk = list(coll.dobj.get('bsplines', {}).keys())
    key = _generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lk,
    )

    # key_chan
    lk = (
        [k0 for k0, v0 in coll.dref.items() if v0['size'] == nlos]
        + [None]
    )
    key_chan = _generic_check._check_var(
        key_chan, 'key_chan',
        allowed=lk,
    )

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

    return key, key_chan, method, resMode, crop, name, store, verb


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

    nlos = cam.nRays
    key, key_chan, method, resMode, crop, name, store, verb = _compute_check(
        coll=coll,
        key=key,
        key_chan=key_chan,
        nlos=nlos,
        method=method,
        resMode=resMode,
        crop=crop,
        name=name,
        store=store,
        verb=verb,
    )

    # -----------
    # prepare

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
                nan0=False,
                nan_out=False,
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


# #############################################################################
# #############################################################################
#               retrofit                   
# #############################################################################


def _compute_retrofit_data_check(
    # resources
    coll=None,
    # inputs
    key=None,
    key_matrix=None,
    key_profile2d=None,
    t=None,
    # parameters
    store=None,
):

    #----------
    # keys

    # key
    lout = coll.ddata.keys()
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        excluded=lout,
    )

    # key_matrix
    lok = coll.dobj.get('matrix', {}).keys()
    key_matrix = ds._generic_check._check_var(
        key_matrix, 'key_mtrix',
        types=str,
        allowed=lok,
    )
    keybs = coll.dobj['matrix'][key_matrix]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    nchan, nbs = coll.ddata[key_matrix]['data'].shape[-2:]
    refchan, refbs = coll.ddata[key_matrix]['ref'][-2:]

    # key_pofile2d
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['bsplines'] == keybs
    ]
    key_profile2d = ds._generic_check._check_var(
        key_profile2d, 'key_profile2d',
        types=str,
        allowed=lok,
    )

    # time management
    hastime, reft, keyt, t_out, dind = coll.get_time_common(
        keys=[key_matrix, key_profile2d],
        t=t,
        ind_strict=False,
    )
    if hastime and t_out is not None and reft is None:
        reft = f'{key}-nt'
        keyt = f'{key}-t'

    ist_mat = coll.get_time(key=key_matrix)[0]
    ist_prof = coll.get_time(key=key_profile2d)[0]

    # reft, keyt and refs
    if hastime and t_out is not None:
        nt = t_out.size
        refs = (reft, refchan)
    else:
        nt = 0
        reft = None
        keyt = None
        refs = (refchan,)

    return (
        key, keybs, keym, mtype,
        key_matrix, key_profile2d,
        hastime, t_out, keyt, reft, refs,
        nt, nchan, nbs,
        ist_mat, ist_prof, dind,
    )


def compute_retrofit_data(
    # resources
    coll=None,
    # inputs
    key=None,
    key_matrix=None,
    key_profile2d=None,
    t=None,
    # parameters
    store=None,
):

    # ------------
    # check inputs

    (
        key, keybs, keym, mtype,
        key_matrix, key_profile2d,
        hastime, t, keyt, reft, refs,
        nt, nchan, nbs,
        ist_mat, ist_prof, dind,
    ) = _compute_retrofit_data_check(
        # resources
        coll=coll,
        # inputs
        key=key,
        key_matrix=key_matrix,
        key_profile2d=key_profile2d,
        t=t,
        # parameters
        store=store,
    )

    # --------
    # compute

    matrix = coll.ddata[key_matrix]['data']
    coefs = coll.ddata[key_profile2d]['data']
    if mtype == 'rect':
        indbs_tf = coll.select_bsplines(
            key=keybs,
            returnas='ind',
        )
        if hastime and ist_prof:
            coefs = coefs[:, indbs_tf[0], indbs_tf[1]]
        else:
            coefs = coefs[indbs_tf[0], indbs_tf[1]]

    if hastime:

        retro = np.full((nt, nchan, nbs), np.nan)

        # get time indices
        if ist_mat:
            if dind.get(key_matrix, {}).get('ind') is not None:
                imat = dind[key_matrix]['ind']
            else:
                imat = np.arange(nt)

        if ist_prof:
            if dind.get(key_profile2d, {}).get('ind') is not None:
                iprof = dind[key_profile2d]['ind']
            else:
                iprof = np.arange(nt)

        # compute matrix product
        if ist_mat and ist_prof:
            retro = np.array([
                matrix[imat[ii], :, :].dot(coefs[iprof[ii], :])
                for ii in range(nt)
            ])
        elif ist_mat:
            retro = np.array([
                matrix[imar[ii], :, :].dot(coefs)
                for ii in range(nt)
            ])
        elif ist_prof:
            retro = np.array([
                matrix.dot(coefs[iprof[ii], :])
                for ii in range(nt)
            ])
    else:
        retro = matrix.dot(coefs)

    # --------
    # store

    if store:

        # add data
        ddata = {
            key: {
                'data': retro,
                'ref': refs,
                'dim': None,
                'quant': None,
                'name': None,
            },
        }

        # add reft + t if new
        if hastime and keyt not in coll.ddata.keys():
            ddata[keyt] = {'data': t, 'ref': reft, 'dim': 'time'}
        if hastime and reft not in coll.dref.keys():
            dref = {reft: {'size': t.size}}
        else:
            dref = None

        # update
        coll.update(dref=dref, ddata=ddata)

    else:
        return retro, t, keyt, reft
