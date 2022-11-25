# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def compute(
    coll=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    res=None,
    resMode=None,
    method=None,
    crop=None,
    store=None,
    verb=None,
):
    """ Compute the geometry matrix using:
            - a Plasma2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """

    # -----------
    # check input
    # -----------

    # nlos = cam.nRays
    key, key_bsplines, key_diag, key_cam, method, resMode, crop, store, verb = _compute_check(
        coll=coll,
        key_bsplines=key_bsplines,
        key_diag=key_diag,
        key_cam=key_cam,
        # sampling
        nlos=nlos,
        method=method,
        resMode=resMode,
        crop=crop,
        store=store,
        verb=verb,
    )

    # -----------
    # prepare
    # -----------

    shapebs = coll.dobj['bsplines'][key_bsplines]['shape']
    km = coll.dobj['bsplines'][key_bsplines]['mesh']
    mtype = coll.dobj[coll._which_mesh][km]['type']

    # prepare indices
    indbs = coll.select_ind(
        key=key_bsplines,
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
    # -----------

    if method == 'los':
        dout, units = _compute_los(
            coll=coll,
            is2d=is2d,
            key_bsplines=key_bsplines,
            key_diag=key_diag,
            key_cam=key_cam,
            # sampling
            indbs=indbs,
            res=res,
            mode=mode,
            key_integrand=key_integrand,
            radius_max=radius_max,
            groupby=groupby,
            val_init=val_init,
            brightness=brightness,
        )

    else:
        pass


    if method == 'los':

    # ---------------
    # store / return
    # ---------------

    if store:
        _store(
            coll=coll,
        )

    else:
        return mat


# ###################
#   checking
# ###################


def _compute_check(
    coll=None,
    key=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    nlos=None,
    method=None,
    resMode=None,
    crop=None,
    store=None,
    verb=None,
):

    # key
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('geom matrix', {}),
        short='gmat',
        key=key,
    )

    # key_bsplines
    lk = list(coll.dobj.get('bsplines', {}).keys())
    key_bsplines = ds._generic_check._check_var(
        key_bsplines, 'key_bsplines',
        types=str,
        allowed=lk,
    )

    # key_diag, key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        default='los',
        types=str,
        allowed=['los'],
    )

    # resMode
    resMode = ds._generic_check._check_var(
        resMode, 'resMode',
        default='abs',
        types=str,
        allowed=['abs', 'rel'],
    )

    # crop
    crop = ds._generic_check._check_var(
        crop, 'crop',
        default=True,
        types=bool,
    )
    crop = crop and coll.dobj['bsplines'][key]['crop'] not in [None, False]

    # store
    store = ds._generic_check._check_var(
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

    return key, key_bsplines, key_diag, key_cam, method, resMode, crop, store, verb


# ###################
#   compute_los                   
# ###################


def _compute_los(
    coll=None,
    is2d=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    val_init=None,
    brightness=None,
):

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']

        # -----------------------
        # loop on group of pixels (to limit memory footprint)

        for ii in range(npix):

            R, Z, length = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=ii,
                radius_max=radius_max,
                concatenate=False,
                return_coords=['R', 'z', 'ltot'],
            )

            # -------------
            # interpolate

            datai, units, refi = coll.interpolate_profile2d(
                key=key_bsplines,
                R=R,
                Z=Z,
                grid=False,
                azone=None,
                indbs=indbs,
                details=True,
                reshape=None,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )





            axis = refi.index(None)
            if ii == 0:
                shape = list(datai.shape)
                shape[axis] = npix
                data = np.full(shape, val_init)
                ref = list(refi)

            # ------------
            # integrate

            iok2 = np.isfinite(datai)
            sli0 = [slice(None) for aa in range(len(refi))]
            for jj in range(nnan):

                # slice datai
                indi = np.arange(inannb[jj]+1, inannb[jj+1])
                sli0[axis] = indi
                slii = tuple(sli0)
                if not np.any(iok2[slii]):
                    continue

                # set nan to 0 for integration
                dataii = datai[slii]
                dataii[~iok2[slii]] = 0.

                # slice data
                ind = i0 + jj
                sli0[axis] = ind
                sli = tuple(sli0)

                # if jj in [50, 51]:
                    # plt.figure();
                    # plt.subplot(1,2,1)
                    # plt.plot(dataii)
                    # plt.subplot(1,2,2)
                    # plt.plot(dataii.T)
                    # plt.gcf().suptitle(f"jj = {jj}", size=12)

                # integrate
                data[sli] = scpinteg.simpson(
                    dataii,
                    x=length[indi],
                    axis=axis,
                )

        # --------------
        # post-treatment

        # brightness
        if brightness is False:
            ketend = doptics[k0]['etendue']
            etend = coll.ddata[ketend]['data']
            sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
            data *= etend.reshape(sh_etend)

        # reshape if 2d
        if is2d:
            sh_data = list(data.shape)
            sh_data[axis] = coll.dobj['camera'][k0]['dgeom']['shape']
            sh_data = tuple(np.r_[
                sh_data[:axis], sh_data[axis], sh_data[axis+1:]
            ].astype(int))
            data = data.reshape(sh_data)

        # set ref
        ref[axis] = coll.dobj['camera'][k0]['dgeom']['ref']
        ref = tuple(np.r_[ref[:axis], ref[axis], ref[axis+1:]])

        # fill dout
        dout[k0] = {
            'data': data,
            'ref': ref,
        }

    # -----
    # units

    units0 = coll.ddata[key_integrand]['units']
    units = units0 * asunits.m
    if brightness is False:
        units = units * coll.ddata[ketend]['units']










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
            val_out=False,
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

    return dout, units


# ###################
#   compute_vos                   
# ###################


def _compute_vos(
    coll=None,
    is2d=None,
    key_diag=None,
    key_cam=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    groupby=None,
    val_init=None,
    brightness=None,
):



    return None, None


# ###################
#   storing                   
# ###################


def _store(
    coll=None,
):

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


# #############################################################################
# #############################################################################
#               retrofit                   
# #############################################################################


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


# ###################
#   checking
# ###################


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
