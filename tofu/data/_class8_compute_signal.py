

import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits


import datastock as ds


# ################################################################
# ################################################################
#               Main routine
# ################################################################


def compute_signal(
    coll=None,
    key=None,
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
    ref_com=None,
    # signal
    brightness=None,
    # verb
    verb=None,
    # store
    store=None,
    # return
    returnas=None,
):

    # -------------
    # check inputs
    # --------------

    (
        key_diag, key_cam, spectro, PHA, is2d,
        method, mode, groupby, val_init, brightness,
        key_integrand, key_mesh0, key_bs,
        key_ref_spectro, key_bs_spectro,
        verb, store, key,
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
        # verb
        verb=verb,
        # store
        store=store,
        key=key,
        # return
        returnas=returnas,
    )

    # -------------
    # prepare
    # --------------

    # shape_emiss = coll.ddata[key_integrand]['shape']

    if mode == 'abs':
        key_kR = coll.dobj['mesh'][key_mesh0]['knots'][0]
        radius_max = np.max(coll.ddata[key_kR]['data'])
    else:
        radius_max = None

    # -------------
    # verb
    # --------------

    if verb is True:
        msg = (
            "\nComputing synthetic signal for:\n"
            f"\t- diag: {key_diag}\n"
            f"\t- cam: {key_cam}\n"
            f"\t- integrand: {key_integrand}\n"
            f"\t- method: {method}\n"
            f"\t- res: {res}, {mode}\n"
        )
        print(msg)

    # -------------
    # compute
    # --------------

    if method == 'los':
        dout = _compute_los(
            coll=coll,
            is2d=is2d,
            spectro=spectro,
            PHA=PHA,
            key=key,
            key_diag=key_diag,
            key_cam=key_cam,
            key_bs=key_bs,
            res=res,
            mode=mode,
            key_integrand=key_integrand,
            key_ref_spectro=key_ref_spectro,
            key_bs_spectro=key_bs_spectro,
            radius_max=radius_max,
            groupby=groupby,
            val_init=val_init,
            ref_com=ref_com,
            brightness=brightness,
        )

    else:
        pass

    # -------------
    # store
    # --------------

    if store is True:
        _store(
            coll=coll,
            key=key,
            key_diag=key_diag,
            dout=dout,
            key_integrand=key_integrand,
            method=method,
            res=res,
        )

    # -------------
    # return
    # --------------

    if returnas is dict:
        return dout


# ##################################################################
# ##################################################################
#               STORE
# ##################################################################


def _store(
    coll=None,
    key=None,
    key_diag=None,
    dout=None,
    # synthetic signal
    key_integrand=None,
    method=None,
    res=None,
    # retrofit
    key_matrix=None,
):

    # ---------
    # check

    lc = [key_integrand is not None, key_matrix is not None]
    if np.sum(lc) != 1:
        msg = "Please provide key_integrand xor key_matrix"
        raise Exception(msg)

    typ = 'retrofit' if key_integrand is None else 'synthetic'

    # ---------
    # prepare

    dobj = {
        'synth sig':{
            key: {
                'diag': key_diag,
                'camera': list(dout.keys()),
                'data': [v0['key'] for v0 in dout.values()],
                # synthetic
                'integrand': key_integrand,
                'method': method,
                'res': res,
                # retrofit
                'geom matrix': key_matrix,
            },
        },
    }

    # ----------
    # add data

    for k0, v0 in dout.items():
        coll.add_data(**v0)

    # add obj
    lsynth = coll._dobj['diagnostic'][key_diag].get('signal')
    if lsynth is None:
        lsynth = [key]
    else:
        lsynth.append(key)
    coll._dobj['diagnostic'][key_diag]['signal'] = lsynth
    coll.update(dobj=dobj)


# ##################################################################
#               CHECK
# ##################################################################


def _compute_signal_check(
    coll=None,
    key=None,
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
    # verb
    verb=None,
    # store
    store=None,
    # return
    returnas=None,
):

    wm = coll._which_mesh
    wbs = coll._which_bsplines

    # key_diag, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(key=key_diag, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    PHA = coll.dobj['diagnostic'][key_diag]['PHA']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        types=str,
        default='los',
        allowed=['los', 'vos'],
    )

    # mode
    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='abs',
        allowed=['abs', 'rel'],
    )

    # groupby
    groupby = ds._generic_check._check_var(
        groupby, 'groupby',
        types=int,
        default=1 if (PHA or spectro) else 200,
        allowed=[1] if (PHA or spectro) else None,
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
        if v0.get(wbs) is not None
        and any([
            coll.dobj[wm][coll.dobj[wbs][k1][wm]]['nd'] == '2d'
            or coll.dobj[wm][coll.dobj[wbs][k1][wm]]['submesh'] is not None
            for k1 in v0[wbs]
        ])
    ]

    key_integrand = ds._generic_check._check_var(
        key_integrand, 'key_integrand',
        types=str,
        allowed=lok,
    )

    # key_mesh0
    key_bs = [
        kk for kk in coll.ddata[key_integrand][wbs]
        if coll.dobj[wm][coll.dobj[wbs][kk][wm]]['nd'] == '2d'
        or coll.dobj[wm][coll.dobj[wbs][kk][wm]]['submesh'] is not None
    ]
    if len(key_bs) == 1:
        key_bs = key_bs[0]
    else:
        msg = f"Multiple possible 2d bsplines for integrand '{key_integrand}'"
        raise Exception(msg)

    key_mesh = coll.dobj[wbs][key_bs][wm]
    mtype = coll.dobj[wm][key_mesh]['type']
    submesh = coll.dobj[wm][key_mesh]['submesh']
    if submesh is not None:
        key_mesh0 = submesh
    else:
        key_mesh0 = key_mesh

    # key_ref_spectro
    if spectro:
        key_ref_spectro, key_bs_spectro = _get_ref_bs_spectro(
            coll=coll,
            key_integrand=key_integrand,
            key_bs=key_bs,
        )
    else:
        key_ref_spectro = None
        key_bs_spectro = None

    # val_init
    val_init = ds._generic_check._check_var(
        val_init, 'val_init',
        default=np.nan,
        allowed=[np.nan, 0.]
    )

    # verb
    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # key
    if store is True:
        key = ds._generic_check._obj_key(
            coll.dobj.get('synth sig', {}),
            short='synth',
            key=key,
            ndigits=2,
        )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=False if store is True else dict,
        allowed=[dict, False],
    )

    return (
        key_diag, key_cam, spectro, PHA, is2d,
        method, mode, groupby, val_init, brightness,
        key_integrand, key_mesh0, key_bs,
        key_ref_spectro, key_bs_spectro,
        verb, store, key,
        returnas,
    )


def _get_ref_bs_spectro(coll=None, key_integrand=None, key_bs=None):

    # get ref of integrand
    kref = coll.ddata[key_integrand]['ref']
    wbs = coll._which_bsplines

    key_ref_spectro = None
    key_bs_spectro = None

    # get list of non-spatial ref
    lkspectro = [
        k0 for k0 in kref
        if k0 not in coll.dobj[wbs][key_bs]['ref']
    ]

    # If none => error
    if len(lkspectro) == 0:
        msg = (
            "Integrand '{key_integrand}' does not seem to "
            "have a spectral dimension"
        )
        raise Exception(msg)

    # unique => ok
    if len(lkspectro) == 1:
        key_ref_spectro = lkspectro[0]
    else:
        pass

    # check if bs
    lbs_spectro = [
        k0 for k0 in coll.ddata[key_integrand][wbs]
        if k0 != key_bs
    ]
    if len(lbs_spectro) == 0:
        pass
    elif len(lbs_spectro) == 1:
        key_bs_spectro = lbs_spectro[0]
        if key_ref_spectro is None:
            key_ref_spectro = coll.dobj[wbs][key_bs_spectro]['ref'][0]
        assert key_ref_spectro == coll.dobj[wbs][key_bs_spectro]['ref'][0]
    else:
        pass

    # --------
    # safety check

    if key_ref_spectro is None:
        msg = "Spectral dimension of '{key_integrand} could not be identified'"
        raise Exception(msg)

    return key_ref_spectro, key_bs_spectro


# ##################################################################
# ##################################################################
#               LOS
# ##################################################################


def _compute_los(
    coll=None,
    is2d=None,
    spectro=None,
    PHA=None,
    key=None,
    key_diag=None,
    key_cam=None,
    key_bs=None,
    res=None,
    mode=None,
    key_integrand=None,
    key_ref_spectro=None,
    key_bs_spectro=None,
    radius_max=None,
    groupby=None,
    val_init=None,
    ref_com=None,
    brightness=None,
):

    # -----------------
    # prepare

    if spectro:

        

        kspect_ref_vect = coll.get_ref_vector(ref=key_ref_spectro)[3]
        spect_ref_vect = coll.ddata[kspect_ref_vect]['data']
        
        ref = coll.ddata[key_integrand]['ref']
        axis_spectro = ref.index(key_ref_spectro)

        wbs = coll._which_bsplines
        rbs = coll.dobj[wbs][key_bs]['ref'][0]
        if axis_spectro > ref.index(rbs):
            axis_spectro -= len(coll.dobj[wbs][key_bs]['ref']) - 1

        units_spectro = coll.ddata[kspect_ref_vect]['units']

        E, _ = coll.get_diagnostic_lamb(
            key_diag,
            lamb='lamb',
            units=units_spectro,
        )
        dE, _ = coll.get_diagnostic_lamb(
            key_diag,
            lamb='dlamb',
            units=units_spectro,
        )
        E_flat = E.ravel()
        dE_flat = dE.ravel()

    else:
        dict_E = None
        dict_dE = None

    # units
    units0, units_bs = _units_integration(
        coll=coll,
        key_integrand=key_integrand,
        key_bs=key_bs,
    )
    units = units0 * units_bs

    domain = None

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']
        key_pts0 = coll.dobj['rays'][key_los]['pts'][0]
        ilosok = np.isfinite(coll.ddata[key_pts0]['data'][0, ...].ravel())

        ngroup = npix // groupby
        if groupby * ngroup < npix:
            ngroup += 1

        # ---------------------------------------------------
        # loop on group of pixels (to limit memory footprint)

        shape = None
        for ii in range(ngroup):

            # indices
            i0 = ii*groupby
            i1 = min((ii + 1)*groupby, npix)
            ni = i1 - i0

            # get rid of undefined LOS
            ind_flat = [jj for jj in range(i0, i1) if ilosok[jj]]
            ni = len(ind_flat)

            # no valid los in group
            if len(ind_flat) == 0:
                continue

            # LOS sampling
            R, Z, length = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=ind_flat,
                radius_max=radius_max,
                concatenate=True,
                return_coords=['R', 'z', 'ltot'],
            )
            
            if R is None:
                continue

            # safety checks
            inan = np.isnan(R)
            inannb = np.r_[-1, inan.nonzero()[0]]
            nnan = inan.sum()

            # some lines can be nan if non-existant
            assert nnan == ni, f"{nnan} vs {ni}"

            # -------------------
            # domain for spectro

            if spectro:
                ind = np.argmin(np.abs(spect_ref_vect - E_flat[ind_flat[0]]))
                domain = {key_ref_spectro: {'ind': np.r_[ind]}}

            # ---------------------
            # interpolate spacially

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=key_integrand,
                ref_key=key_bs,
                x0=R,
                x1=Z,
                grid=False,
                submesh=True,
                ref_com=ref_com,
                domain=domain,
                # azone=None,
                details=False,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[key_integrand]

            # ----------------------
            # interpolate spectrally

            if spectro:             
                douti['data'] = np.take(douti['data'], 0, axis_spectro)
                douti['ref'] = tuple([
                    rr for jj, rr in enumerate(douti['ref'])
                    if jj != axis_spectro
                ])

            # ------------
            # extract data

            datai, refi = douti['data'], douti['ref']
            axis = refi.index(None)
            if shape is None:
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
                sli0[axis] = ind_flat[jj]
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
            unitsi = units * coll.ddata[ketend]['units']
        else:
            unitsi = units

        # spectral bins if spectro
        if spectro:
            sh_dE = [-1 if aa == axis else 1 for aa in range(len(refi))]
            data *= dE_flat.reshape(sh_dE)
            unitsi = unitsi * units_spectro

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
            'key': f'{key}_{k0}',
            'data': data,
            'ref': ref,
            'units': unitsi,
        }

    # -----
    # units

    return dout


def _units_integration(
    coll=None,
    key_integrand=None,
    key_bs=None,
):

    units0 = coll.ddata[key_integrand]['units']

    wbs = coll._which_bsplines
    kap = coll.dobj[wbs][key_bs]['apex']
    lunits = list({coll.ddata[k0]['units'] for k0 in kap})
    if len(lunits) == 1:
        units_bs = lunits[0]
    else:
        msg = "Don't know how to interpret line-integration units from bspline"
        raise Exception(msg)

    wm = coll._which_mesh
    keym = coll.dobj[wbs][key_bs][wm]
    subbs = coll.dobj[wm][keym]['subbs']
    if subbs is not None:
        kap = coll.dobj[wbs][subbs]['apex']
        lunits = list({coll.ddata[k0]['units'] for k0 in kap})
        if len(lunits) == 1:
            units_bs = lunits[0]
        else:
            msg = "Don't know how to interpret line-integration units from bspline"
            raise Exception(msg)

    return units0, units_bs


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
