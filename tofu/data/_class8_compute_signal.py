

import numpy as np


import datastock as ds


# ##################################################################
# ##################################################################
#               Main routine
# ##################################################################


def compute_signal(
    coll=None,
    key_diag=None,
    key_cam=None,
    # to be integrated
    key_emiss=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    groupby=None,
    # signal
    brightness=None,
    # store
    store=None,
    key_signal=None,
    # return
    returnas=None,
):

    # -------------
    # check inputs 
    # --------------

    (
        key_diag, key_cam, spectro, is2d,
        method, groupby, brightness,
        key_emiss, key_mesh0,
        store, key_signal,
        returnas,
    ) = _compute_signal_check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        # sampling
        method=method,
        res=res,
        mode=mode,
        # signal
        brightness=brightness,
        # to be integrated
        key_emiss=key_emiss,
        # store
        store=store,
        key_signal=key_signal,
        # return
        returnas=returnas,
    )

    # -------------
    # prepare 
    # --------------

    shape_emiss = coll.ddata[key_emiss]['shape']

    key_kR = coll.dobj['mesh'][key_mesh0]['knots'][0]
    radius_max = np.max(coll.ddata[key_kR]['data'])

    # -------------
    # compute 
    # --------------

    if method == 'los':
        dout = _compute_los(
            coll=coll,
            is2d=is2d,
            key_diag=key_diag,
            key_cam=key_cam,
            res=res,
            mode=mode,
            key_emiss=key_emiss,
            radius_max=radius_max,
            groupby=groupby,
        )

    else:
        pass


    return


# ##################################################################
# ##################################################################
#               CHECK
# ##################################################################


def _compute_signal_check(
    coll=None,
    key_diag=None,
    key_cam=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    groupby=None,
    # signal
    brightness=None,
    # to be integrated
    key_emiss=None,
    # store
    store=None,
    key_signal=None,
    # return
    returnas=None,
):

    # key, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(key=key_diag, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        types=str,
        default='los',
        allowed=['los', 'vos'],
    )

    # groupby
    groupby = ds._generic_check._check_var(
        groupby, 'groupby',
        types=int,
        default=200,
    )

    # brightness
    brightness = ds._generic_check._check_var(
        brightness, 'brightness',
        types=bool,
        default=False,
    )

    # key_emiss
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
    ]
    key_emiss = ds._generic_check._check_var(
        key_emiss, 'key_emiss',
        types=str,
        allowed=lok,
    )

    # key_mesh0
    key_bs = coll.ddata[key_emiss]['bsplines']
    key_mesh = coll.dobj['bsplines'][key_bs]['mesh']
    mtype = coll.dobj['mesh'][key_mesh]['type']
    if mtype == 'polar':
        key_mesh0 = coll.dobj['mesh'][key_mesh]['submesh']
    else:
        key_mesh0 = key_mesh

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # key_signal
    key_signal = ds._generic_check._check_var(
        key_signal, 'key_signal',
        types=str,
        default=f'{key_diag}_synth',
        excluded=list(coll.ddata.keys()),
    )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=False if store is True else dict,
        allowed=[dict, False],
    )

    return (
        key_diag, key_cam, spectro, is2d,
        method, groupby, brightness,
        key_emiss, key_mesh0,
        store, key_signal,
        returnas,
    )


# ##################################################################
# ##################################################################
#               LOS
# ##################################################################


def _compute_los(
    coll=None,
    is2d=None,
    key_diag=None,
    key_cam=None,
    res=None,
    mode=None,
    key_emiss=None,
    radius_max=None,
    groupby=None,
):

    # loop on cameras
    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']

        ngroups = npix // groupby
        if npix % groupby > 0:
            groupby += 1

        for ii in range(ngroup):

            i0 = ii*groupby
            i1 = min((ii + 1)*groupby, npix)
            ni = i1 - i0

            R, Z, ll = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=np.arange(i0, i1),
                radius_max=radius_max,
                concatenate=True,
                return_coords=['R', 'z', 'l'],
            )

            inan = np.r_[0, np.isnan(R).nonzero()[0]]
            nnan = inan.size - 1
            assert nnan == ni, f"{nnan} vs {ni}"
            iok = ~inan

            datai, units, refi = coll.interpolate_profile2d(
                key=key_emiss,
                coefs=None,
                R=R[iok],
                Z=Z[iok],
                grid=False,
                radius_vs_time=None,
                azone=None,
                t=None,
                indt=None,
                indt_strict=None,
                indbs=None,
                details=False,
                reshape=None,
                res=None,
                crop=None,
                nan0=None,
                nan_out=None,
                imshow=None,
                return_params=None,
                store=False,
                inplace=None,
            )

            for jj in range(nnan):
                ind = i0 + jj
                sli = ()
                slii = ()
                data[sli] = scpinteg.simpson(
                    ll[inan[jj]:inan[jj+1]],
                    datai[slii],
                    axis=axis,
                )

            import pdb; pdb.set_trace()     # DB

        dout[k0] = {
            'data': data,
            'ref': ref,
            'units': units,
        }

    return dout


# ##################################################################
# ##################################################################
#               VOS
# ##################################################################


def _compute_vos(
    coll=None,
    key_cam=None,
    res=None,
    mode=None,
    key_emiss=None,
):

    dout = None

    return dout
