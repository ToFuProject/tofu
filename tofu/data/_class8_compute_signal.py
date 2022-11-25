

import itertools as itt


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt     # DB



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
    key_integrand=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    groupby=None,
    val_init=None,
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
        method, groupby, val_init, brightness,
        key_integrand, key_mesh0,
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
        groupby=groupby,
        val_init=val_init,
        # signal
        brightness=brightness,
        # to be integrated
        key_integrand=key_integrand,
        # store
        store=store,
        key_signal=key_signal,
        # return
        returnas=returnas,
    )

    # -------------
    # prepare 
    # --------------

    shape_emiss = coll.ddata[key_integrand]['shape']

    key_kR = coll.dobj['mesh'][key_mesh0]['knots'][0]
    radius_max = np.max(coll.ddata[key_kR]['data'])

    # -------------
    # compute 
    # --------------

    if method == 'los':
        dout, units = _compute_los(
            coll=coll,
            is2d=is2d,
            key_diag=key_diag,
            key_cam=key_cam,
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

    # -------------
    # post-format 
    # --------------

    if store is True:

        doptics = coll._dobj['diagnostic'][key_diag]['doptics']
        dsig = coll._dobj['diagnostic'][key_diag].get('dsignal')
        if dsig is None:
            dsig = {}

        dsig.update({
            key_signal: {
                'integrand': key_integrand,
                'camera': [],
                'signal': [],
                'method': method,
                'res': res,
            },
        })

        for k0, v0 in dout.items():

            # add data
            ksig = f'{key_signal}_{k0}'
            coll.add_data(
                key=ksig,
                data=v0['data'],
                ref=v0['ref'],
                units=units,
            )

            # add as synthetic signal in dobj
            dsig[key_signal]['camera'].append(k0)
            dsig[key_signal]['signal'].append(ksig)

        coll._dobj['diagnostic'][key_diag]['dsignal'] = dsig

    # -------------
    # return 
    # --------------

    if returnas is dict:
        return dout


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
    val_init=None,
    # signal
    brightness=None,
    # to be integrated
    key_integrand=None,
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

    # key_integrand
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
    ]
    key_integrand = ds._generic_check._check_var(
        key_integrand, 'key_integrand',
        types=str,
        allowed=lok,
    )

    # key_mesh0
    key_bs = coll.ddata[key_integrand]['bsplines']
    key_mesh = coll.dobj['bsplines'][key_bs]['mesh']
    mtype = coll.dobj['mesh'][key_mesh]['type']
    if mtype == 'polar':
        key_mesh0 = coll.dobj['mesh'][key_mesh]['submesh']
    else:
        key_mesh0 = key_mesh

    # val_init
    val_init = ds._generic_check._check_var(
        val_init, 'val_init',
        default=np.nan,
        allowed=[np.nan, 0.]
    )

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
        method, groupby, val_init, brightness,
        key_integrand, key_mesh0,
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
    key_integrand=None,
    radius_max=None,
    groupby=None,
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

        ngroup = npix // groupby
        if groupby * ngroup < npix:
            ngroup += 1

        # -----------------------
        # loop on group of pixels (to limit memory footprint)

        for ii in range(ngroup):

            i0 = ii*groupby
            i1 = min((ii + 1)*groupby, npix)
            ni = i1 - i0

            R, Z, length = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=np.arange(i0, i1),
                radius_max=radius_max,
                concatenate=True,
                return_coords=['R', 'z', 'ltot'],
            )

            inan = np.isnan(R)
            inannb = np.r_[-1, inan.nonzero()[0]]
            nnan = inan.sum()
            assert nnan == ni, f"{nnan} vs {ni}"
            iok = ~inan

            # -------------
            # interpolate

            datai, units, refi = coll.interpolate_profile2d(
                key=key_integrand,
                R=R,
                Z=Z,
                grid=False,
                radius_vs_time=None,
                azone=None,
                t=None,
                indt=None,
                indt_strict=None,
                indbs=None,
                details=False,
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

    return dout, units


# ##################################################################
# ##################################################################
#               VOS
# ##################################################################


def _compute_vos(
    coll=None,
    key_cam=None,
    res=None,
    mode=None,
    key_integrand=None,
):

    dout = None

    return dout
