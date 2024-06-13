# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:03:27 2024

@author: dvezinet
"""

# Built-in


# Common
import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits


# #############################################################################
# #############################################################################
#                           LOS
# #############################################################################


def _compute_los(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    is3d=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
    # slicing
    shape_mat=None,
    sli_mat=None,
    axis_pix=None,
    # parameters
    brightness=None,
    verb=None,
):

    # -----
    # units

    units = asunits.m
    units_coefs = asunits.Unit()

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']
        key_mat = f'{key}_{k0}'

        sh = tuple([npix if ss is None else ss for ss in shape_mat])
        mat = np.zeros(sh, dtype=float)

        # -----------------------
        # loop on group of pixels (to limit memory footprint)

        anyok = False
        for ii in range(npix):

            # verb
            if verb is True:
                msg = (
                    f"\t- '{key_mat}' for cam '{k0}': pixel {ii + 1} / {npix}"
                    f"\t{(mat > 0).sum()} / {mat.size}\t\t"
                )
                end = '\n' if ii == npix - 1 else '\r'
                print(msg, flush=True, end=end)

            # sample los
            out_sample = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_ch=ii,
                radius_max=radius_max,
                concatenate=False,
                return_coords=['R', 'z', 'ltot'],
            )

            if out_sample is None or out_sample[0] is None:
                continue

            R, Z, length = out_sample

            # -------------
            # interpolate

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=None,
                ref_key=key_bs,
                # interpolation pts
                x0=R[:, 0],
                x1=Z[:, 0],
                submesh=True,
                grid=False,
                # common ref
                ref_com=ref_com,
                ref_vector_strategy=ref_vector_strategy,
                # bsplines-specific
                # azone=None,
                indbs_tf=indbs,
                details=True,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[f'{key_bs}_details']

            datai, refi = douti['data'], douti['ref']
            axis = refi.index(None)
            iok = np.isfinite(datai)

            if not np.any(iok):
                continue

            datai[~iok] = 0.

            # ------------
            # integrate

            # check and update slice
            assert datai.ndim in [2, 3], datai.shape
            sli_mat[axis_pix] = ii

            # integrate
            mat[tuple(sli_mat)] = scpinteg.simpson(
                datai,
                x=length[:, 0],
                axis=axis,
            )

            anyok = True

        # --------------
        # post-treatment

        if anyok:
            # brightness
            if brightness is False:
                ketend = doptics[k0]['etendue']
                units_coefs = coll.ddata[ketend]['units']
                etend = coll.ddata[ketend]['data']
                sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
                mat *= etend.reshape(sh_etend)

            # set ref
            refi = list(refi)
            refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref_flat']
            refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        else:
            refi = None
            axis = None

        # fill dout
        dout[key_mat] = {
            'data': mat,
            'ref': refi,
            'units': units * units_coefs,
        }

    return dout, axis


# #############################################################################
# #############################################################################
#                           VOS
# #############################################################################


def _compute_vos(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
    # dvos
    dvos=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    is3d=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
    # slicing
    shape_mat=None,
    sli_mat=None,
    axis_pix=None,
    # parameters
    brightness=None,
    verb=None,
):

    # -----
    # units

    units = asunits.Unit(dvos[key_cam[0]]['sang_cross']['units'])
    units_coefs = asunits.Unit()

    # -------------
    # mesh sampling

    lkm = list(set([v0['keym'] for v0 in dvos.values()]))
    lres = list(set([tuple(v0['res_RZ']) for v0 in dvos.values()]))

    if len(lkm) != 1:
        lstr = [f"\t- '{k0}': '{v0['keym']}'" for k0, v0 in dvos.items()]
        msg = (
            "All cameras vos were not sampled using the same mesh!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    if len(lres) != 1:
        lstr = [f"\t- '{k0}': '{v0['res_RZ']}'" for k0, v0 in dvos.items()]
        msg = (
            "All cameras vos were not sampled using the same resolution!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # extract
    keym = lkm[0]
    res_RZ = list(lres[0])

    # mesh sampling
    dsamp = coll.get_sample_mesh(
        key=keym,
        res=res_RZ,
        mode='abs',
        grid=False,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=None,
        Dx1=None,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    x0u = dsamp['x0']['data']
    x1u = dsamp['x1']['data']

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_mat = f'{key}_{k0}'

        # -------------
        # slicing

        is2d = coll.dobj['camera'][k0]['dgeom']['nd'] == '2d'
        if is2d:
            n0, n1 = coll.dobj['camera'][k0]['dgeom']['shape']
            sli = lambda ii: (ii // n1, ii % n1, slice(None))
        else:
            sli = lambda ii: (ii, slice(None))

        # shape, key
        sh = tuple([npix if ss is None else ss for ss in shape_mat])
        mat = np.zeros(sh, dtype=float)

        # ---------------------------------------------------
        # loop on group of pixels (to limit memory footprint)

        anyok = False
        for ii in range(npix):

            # verb
            if verb is True:
                msg = (
                    f"\t- '{key_mat}' for cam '{k0}': pixel {ii + 1} / {npix}"
                    f"\t{(mat > 0).sum()} / {mat.size}\t\t"
                )
                end = '\n' if ii == npix - 1 else '\r'
                print(msg, flush=True, end=end)

            # sample los
            indok = np.isfinite(dvos[k0]['sang_cross']['data'][sli(ii)])
            if not np.any(indok):
                continue

            # indices + dv
            indr = dvos[k0]['indr_cross']['data'][sli(ii)][indok]
            indz = dvos[k0]['indz_cross']['data'][sli(ii)][indok]

            # -------------
            # interpolate

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=None,
                ref_key=key_bs,
                x0=x0u[indr],
                x1=x1u[indz],
                submesh=True,
                grid=False,
                # common ref
                ref_com=ref_com,
                ref_vector_strategy=ref_vector_strategy,
                # bsplines-specific
                # azone=None,
                indbs_tf=indbs,
                details=True,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[f'{key_bs}_details']

            datai, refi = douti['data'], douti['ref']
            axis = refi.index(None)
            iok = np.isfinite(datai)

            if not np.any(iok):
                continue

            datai[~iok] = 0.

            # ------------
            # integrate

            # check and update slice
            assert datai.ndim in [2, 3], datai.shape
            sli_mat[axis_pix] = ii

            # integrate
            mat[tuple(sli_mat)] = np.sum(
                datai * dvos[k0]['sang_cross']['data'][sli(ii)][indok][:, None],
                axis=axis,
            )

            anyok = True

        # --------------
        # post-treatment

        if anyok:
            # brightness
            if brightness is True:
                ketend = doptics[k0]['etendue']
                units_coefs = coll.ddata[ketend]['units']
                etend = coll.ddata[ketend]['data']
                sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
                mat /= etend.reshape(sh_etend)

            # set ref
            refi = list(refi)
            refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref_flat']
            refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        else:
            refi = None
            axis = None

        # fill dout
        dout[key_mat] = {
            'data': mat,
            'ref': refi,
            'units': units / units_coefs,
        }

    return dout, axis