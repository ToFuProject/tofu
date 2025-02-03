

import datetime as dtm
import itertools as itt


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
    # spectral ref
    key_ref_spectro=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    groupby=None,
    val_init=None,
    ref_com=None,
    # vos
    dvos=None,
    # signal
    brightness=None,
    spectral_binning=None,
    # verb
    verb=None,
    # timing
    timing=None,
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
        method, mode, groupby, val_init,
        brightness, spectral_binning,
        key_integrand, key_mesh0, key_bs,
        key_ref_spectro, key_bs_spectro,
        dvos,
        verb, timing, store, key,
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
        # vos
        dvos=dvos,
        # signal
        brightness=brightness,
        spectral_binning=spectral_binning,
        # to be integrated
        key_integrand=key_integrand,
        # spectral ref
        key_ref_spectro=key_ref_spectro,
        # verb
        verb=verb,
        # timing
        timing=timing,
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

    # pick routine
    if method == 'los':
        func = _compute_los
    else:
        if spectro is True:
            func = _compute_vos_spectro
        else:
            func = _compute_vos_broadband

    # call routine
    dout, dt = func(
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
        spectral_binning=spectral_binning,
        # vos
        dvos=dvos,
        # verb
        verb=verb,
        # timing
        timing=timing,
    )

    # ----------
    # timing
    # ----------

    if timing is True:
        lstr = [f"{k0}:\t {v0} s" for k0, v0 in dt.items()]
        msg = (
            "Timing for "
            f"compute_diagnostic_signal('{key_diag}', method='{method}')\n"
            + "\n".join(lstr)
        )
        print(msg)

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

    # ---------
    # prepare

    lkcam = list(dout.keys())
    dobj = {
        'synth sig':{
            key: {
                'diag': key_diag,
                'camera': lkcam,
                'data': [dout[k0]['key'] for k0 in lkcam],
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
    # vos
    dvos=None,
    # signal
    brightness=None,
    spectral_binning=None,
    # to be integrated
    key_integrand=None,
    # spectral ref
    key_ref_spectro=None,
    # verb
    verb=None,
    # timing
    timing=None,
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
            key_ref_spectro=key_ref_spectro,
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

    # dvos
    if method == 'vos':
        # single camera + get dvos
        key_diag, dvos, isstore = coll.check_diagnostic_dvos(
            key_diag,
            key_cam=key_cam,
            dvos=dvos,
        )

        if spectro is True and (not isstore):
            msg = "spectro synthetic signal with 'vos' required stored vos!"
            raise Exception(msg)

    # verb
    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # timing
    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
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
        method, mode, groupby, val_init,
        brightness, spectral_binning,
        key_integrand, key_mesh0, key_bs,
        key_ref_spectro, key_bs_spectro,
        dvos,
        verb, timing, store, key,
        returnas,
    )


def _get_ref_bs_spectro(
    coll=None,
    key_integrand=None,
    key_bs=None,
    key_ref_spectro=None,
):

    # -----------
    # prepare

    # get ref of integrand
    kref = coll.ddata[key_integrand]['ref']
    wbs = coll._which_bsplines

    # get list of non-spatial ref
    lkspectro = [
        k0 for k0 in kref
        if k0 not in coll.dobj[wbs][key_bs]['ref']
    ]

    # If none => error
    if len(lkspectro) == 0:
        msg = (
            f"Integrand '{key_integrand}' does not seem to "
            "have a spectral dimension"
        )
        raise Exception(msg)

    # lbs_spectro
    lbs_spectro = [
        k0 for k0 in coll.ddata[key_integrand][wbs]
        if k0 != key_bs
    ]

    # --------------------------
    # Determine key_ref_spectro

    key_bs_spectro = None
    if key_ref_spectro is None:

        # unique => ok
        if len(lkspectro) == 1:
            key_ref_spectro = lkspectro[0]
        else:
            pass

    # check if bs
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
        msg = f"Spectral dimension of '{key_integrand} could not be identified'"
        raise Exception(msg)

    return key_ref_spectro, key_bs_spectro


# ################################################################
# ################################################################
#               LOS
# ################################################################


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
    spectral_binning=None,
    # verb
    verb=None,
    # timing
    timing=None,
    # unused
    **kwdargs,
):

    # --------------
    # prepare timing

    dt = None
    if timing is True:
        lk = [
            '\tpreparation',
            '\tsample rays',
            '\tspectral binning',
            '\tinterpolate on los',
            '\textract data',
            '\tintegrate on los',
            '\tformat output'
        ]
        dt = dict.fromkeys(lk, 0)
        t0 = dtm.datetime.now()     # Timing

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

        # --------------------------
        # optional spectral binning

        # spectral_binning
        spectral_binning = ds._generic_check._check_var(
            spectral_binning, 'spectral_binning',
            types=bool,
            default=True,
        )

        # if spectral binning => add bins of len 2 for temporary storing
        if spectral_binning is True:
            ktemp_bin = f'{key_ref_spectro}_temp_bin'
            coll.add_bins(
                key=ktemp_bin,
                edges=[0, 1],
                units=units_spectro,
            )
            ktemp_binc = coll.dobj['bins'][ktemp_bin]['cents'][0]

    else:
        pass

    # units
    units0, units_bs = _units_integration(
        coll=coll,
        key_integrand=key_integrand,
        key_bs=key_bs,
    )
    units = units0 * units_bs
    domain = None

    # timing
    if timing is True:
        t1 = dtm.datetime.now()
        dt['\tpreparation'] = (t1 - t0).total_seconds()

    # ----------------
    # loop on cameras

    key_integrand_interp = str(key_integrand)
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

        # ----------------
        # spectro

        if spectro:
            E, _ = coll.get_diagnostic_lamb(
                key_diag,
                lamb='lamb',
                key_cam=k0,
                units=units_spectro,
            )
            dE, _ = coll.get_diagnostic_lamb(
                key_diag,
                lamb='dlamb',
                key_cam=k0,
                units=units_spectro,
            )
            E_flat = E.ravel()
            dE_flat = dE.ravel()

        # ---------------------------------------------------
        # loop on group of pixels (to limit memory footprint)

        shape_cam = coll.dobj['camera'][k0]['dgeom']['shape']
        shape = None
        for ii in range(ngroup):

            # verb
            if verb is True:
                msg = f"\tpix group {ii+1} / {ngroup}"
                end = "\n" if ii == ngroup - 1 else "\r"
                print(msg, end=end, flush=True)

            # timing
            if timing is True:
                t01 = dtm.datetime.now()

            # indices
            i0 = ii*groupby
            i1 = min((ii + 1)*groupby, npix)
            ni = i1 - i0

            # get rid of undefined LOS
            ind_ch_flat = np.array([jj for jj in range(i0, i1) if ilosok[jj]], dtype=int)
            ind_ch = np.unravel_index(ind_ch_flat, shape_cam)
            ni = ind_ch_flat.size

            # no valid los in group
            if ni == 0:
                continue

            # LOS sampling
            R, Z, length = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_ch=ind_ch,
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
            # one nan for each separation between concatenated LOS
            if nnan != ni:
                msg = (
                    "Mismatch between expected and observed nb of NaNs!\n"
                    f"\t- key_diag = {key_diag}\n"
                    f"\t- kcam = {k0}\n"
                    f"\t- ii = {ii}\n"
                    f"\t- i1 - i0 = {i1} - {i0} = {ni}\n"
                    f"\t- groupby, npix = {groupby}    {npix}\n"
                    f"\t- ni = ind_ch_flat.size = {ni}\n"
                    f"\t- shape_cam = {shape_cam}\n"
                    f"\t- ind_ch = {ind_ch}\n"
                    f"\t- res, mode = {res}   {mode}\n"
                    f"\t- R.shape = {R.shape}\n"
                    f"\t- nnan = {nnan}\n"
                )
                raise Exception(msg)

            # timing
            if timing is True:
                t02 = dtm.datetime.now()
                dt['\tsample rays'] += (t02-t01).total_seconds()

            # -------------------
            # domain for spectro

            if spectro and spectral_binning:

                # add bins for storing
                coll._ddata[ktemp_binc]['data'] = E_flat[ii]
                edges = E_flat[ii] + dE_flat[ii] * 0.5 * np.r_[-1, 1]
                coll._dobj['bins'][ktemp_bin]['edges'] = edges

                # bin spectrally before spatial interpolation
                kbinned = f"{key_integrand}_bin_{k0}_{ii}"
                #try:
                coll.binning(
                    data=key_integrand,
                    bin_data0=key_ref_spectro if key_bs_spectro is None else key_bs_spectro,
                    bins0=ktemp_bin,
                    integrate=True,
                    verb=verb,
                    store=True,
                    returnas=False,
                    store_keys=kbinned,
                )
                # except Exception as err:
                #     msg = (
                #         err.args[0]
                #         + "\n\n"
                #         f"\t- k0 = {k0}\n"
                #         f"\t- ii = {ii}\n"
                #         f"\t- key_integrand = {key_integrand}\n"
                #         f"\t- key_bs_spectro = {key_bs_spectro}\n"
                #         f"\t- ktemp_bin = {ktemp_bin}\n"
                #         f"\t- edges = {edges}\n"
                #         f"\t- E_flat[ii] = {E_flat[ii]}\n"
                #     )
                #     err.args = (msg,)
                #     raise err

                domain = None
                key_integrand_interp = kbinned

            elif spectro:
                ind = np.argmin(np.abs(spect_ref_vect - E_flat[ind_ch_flat[0]]))
                domain = {key_ref_spectro: {'ind': np.r_[ind]}}

            # timing
            if timing is True:
                t03 = dtm.datetime.now()
                dt['\tspectral binning'] += (t03-t02).total_seconds()

            # ---------------------
            # interpolate spacially

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=key_integrand_interp,
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
            )[key_integrand_interp]

            # timing
            if timing is True:
                t04 = dtm.datetime.now()
                dt['\tinterpolate on los'] += (t04-t03).total_seconds()

            # ----------------------
            # interpolate spectrally

            if spectro:
                douti['data'] = np.take(douti['data'], 0, axis_spectro)

                if spectral_binning is True:
                    coll.remove_data(kbinned)

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

            # timing
            if timing is True:
                t05 = dtm.datetime.now()
                dt['\textract data'] += (t05-t04).total_seconds()

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
                sli0[axis] = ind_ch_flat[jj]
                sli = tuple(sli0)

                # if jj in [50, 51]:
                    # plt.figure();
                    # plt.subplot(1,2,1)
                    # plt.plot(dataii)
                    # plt.subplot(1,2,2)
                    # plt.plot(dataii.T)
                    # plt.gcf().suptitle(f"jj = {jj}", size=12)

                # integrate
                data[sli] = scpinteg.trapezoid(
                    dataii,
                    x=length[indi],
                    axis=axis,
                )

            # timing
            if timing is True:
                t06 = dtm.datetime.now()
                dt['\tintegrate on los'] += (t06-t05).total_seconds()

        # --------------
        # safety check

        if shape is None:
            msg = (
                "Looks like no single LOS can see non-zero emissivity!\n"
                f"\t- key_los = {key_los}\n"
                f"\t- ni = {ni}\n"
                f"\t- R = {R}\n"
                "\ncoll.show('rays'):\n"
                + coll.show('rays', returnas=str, verb=False)
            )
            raise Exception(msg)

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
            unitsi = unitsi * units_spectro
            if spectral_binning is True:
                pass
            else:
                sh_dE = [-1 if aa == axis else 1 for aa in range(len(refi))]
                data *= dE_flat.reshape(sh_dE)

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

        # timing
        if timing is True:
            t07 = dtm.datetime.now()
            dt['\tformat output'] += (t07-t06).total_seconds()

        # fill dout
        dout[k0] = {
            'key': f'{key}_{k0}',
            'data': data,
            'ref': ref,
            'units': unitsi,
        }

    # ----------
    # clean up

    if spectro is True and spectral_binning is True:
        coll.remove_bins(ktemp_bin)

    return dout, dt


def _units_integration(
    coll=None,
    key_integrand=None,
    key_bs=None,
):

    # -----------------
    # integrand

    # units0
    units0 = coll.ddata[key_integrand]['units']

    # --------------
    # los dimension

    # units_bs
    wbs = coll._which_bsplines
    kap = coll.dobj[wbs][key_bs]['apex']
    lunits = list({coll.ddata[k0]['units'] for k0 in kap})
    if len(lunits) == 1:
        units_bs = lunits[0]
    else:
        msg = "Don't know how to interpret line-integration units from bspline"
        raise Exception(msg)

    # units_bs from subbs if relevant
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
#               VOS - Broadband
# ##################################################################


def _compute_vos_broadband(
    coll=None,
    is2d=None,
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
    key_ref_cos=None,
    groupby=None,
    val_init=None,
    ref_com=None,
    brightness=None,
    spectral_binning=None,
    # dvos
    dvos=None,
    # verb
    verb=None,
    # unused
    **kwdargs,
):

    # ------------------------
    # check uniformity of dvos

    dt = None
    # check all keym and res_RZ are similar
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

    # -----------------
    # get mesh sampling

    # get dsamp
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

    # --------------
    # prepare

    # units
    units0 = coll.ddata[key_integrand]['units']

    # -----------------------------
    # loop on cameras

    dout = {k0: {} for k0 in dvos.keys()}
    for k0, v0 in dvos.items():

        # group
        units_vos = v0['sang_cross']['units']

        # --------------
        # loop on pixels

        shape = None
        shape_cam = v0['indr_cross']['data'].shape[:-1]
        assert shape_cam == coll.dobj['camera'][k0]['dgeom']['shape']
        ref_cam = coll.dobj['camera'][k0]['dgeom']['ref']
        for ind in np.ndindex(shape_cam):

            # no valid los in group
            iok = v0['indr_cross']['data'][ind] >= 0
            if not np.any(iok):
                continue

            # vos re-creation
            ind_RZ = tuple(list(ind) + [iok])
            R = x0u[v0['indr_cross']['data'][ind_RZ]]
            Z = x1u[v0['indz_cross']['data'][ind_RZ]]

            # -----------------------------
            # interpolate on matching wavelength ?

            douti = coll.interpolate(
                keys=key_integrand,
                ref_key=key_bs,
                x0=R,
                x1=Z,
                grid=False,
                submesh=True,
                ref_com=ref_com,
                domain=None,
                # azone=None,
                details=False,
                crop=None,
                nan0=False,
                val_out=0.,
                return_params=False,
                store=False,
            )[key_integrand]

            # extracti shape and ref from integrand
            datai, refi = douti['data'], douti['ref']
            if shape is None:
                axis = refi.index(None)
                shape = list(datai.shape)
                shape = tuple(
                    np.r_[shape[:axis], shape_cam, shape[axis+1:]].astype(int)
                )
                ref = tuple(np.r_[refi[:axis], ref_cam, refi[axis+1:]])
                data = np.full(shape, val_init)
                ind_data = [[slice(None)] for ii in range(datai.ndim)]
                ind_sa = [[None] for ii in range(datai.ndim)]

            # ------------
            # integrate

            ind_data[axis] = ind
            ind_sa[axis] = ind_RZ
            data[tuple(itt.chain.from_iterable(ind_data))] = np.nansum(
                datai
                * v0['sang_cross']['data'][tuple(itt.chain.from_iterable(ind_sa))],
                axis=axis,
            )

        # --------------
        # post-treatment

        unitsi = units0 * units_vos

        # fill dout
        dout[k0] = {
            'key': f'{key}_{k0}',
            'data': data,
            'ref': ref,
            'units': unitsi,
        }

    # ----------
    # clean up

    return dout, dt


# ##################################################################
# ##################################################################
#               VOS - spectro
# ##################################################################


def _compute_vos_spectro(
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
    key_ref_cos=None,
    radius_max=None,
    groupby=None,
    val_init=None,
    ref_com=None,
    brightness=None,
    spectral_binning=None,
    # dvos
    dvos=None,
    # verb
    verb=None,
    # unused
    **kwdargs,
):
    """

    spectral_binning = bool
        if True, bins spectrum
        if False, interpolates spectrum

    """

    # --------------
    # prepare timing

    dt = None

    # ------------------------
    # check uniformity of dvos

    # check all keym and res_RZ are similar
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

    # -----------------
    # get mesh sampling

    # get dsamp
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

    # --------------
    # prepare

    kspect_ref_vect = coll.get_ref_vector(ref=key_ref_spectro)[3]
    units_spectro = coll.ddata[kspect_ref_vect]['units']

    wbs = coll._which_bsplines

    # -----------------------------
    # interpolate per wavelength ?

    dout = {k0: {} for k0 in dvos.keys()}
    for k0, v0 in dvos.items():

        R = x0u[v0['indr_cross']['data']]
        Z = x1u[v0['indz_cross']['data']]

        kapex = coll.dobj[wbs][key_bs_spectro]['apex'][0]
        dlamb_ref = np.mean(np.diff(coll.ddata[kapex]['data']))
        lamb = v0['lamb']['data']
        dlamb = np.mean(np.diff(lamb))
        if spectral_binning is None:
            spectral_binningi = dlamb > dlamb_ref
        else:
            spectral_binningi = spectral_binning

        # ------------------------------------------------------
        # interpolate on matching wavelength or bin spectrally?

        # pre-interpolate at lamb
        # (nt, nbs0, nbs1, nlamb_ref, ncos) => (nt, nbs0, nbs1, nlamb, ncos)
        # or
        # (nt, nbs0, nbs1, nlamb_ref) => (nt, nbs0, nbs1, nlamb)

        key_integrand_interp_lamb = f"{key_integrand}_interp_lamb"
        if spectral_binningi is True:

            ktemp_bin = f'{key_bs_spectro}_{k0}_temp_bin'
            edge_spectro = np.r_[
                lamb[0] - 0.5*(lamb[1] - lamb[0]),
                0.5 * (lamb[1:] + lamb[:-1]),
                lamb[-1] + 0.5*(lamb[-1] - lamb[-2]),
            ]

            coll.add_bins(
                key=ktemp_bin,
                edges=edge_spectro,
                units=units_spectro,
            )

            coll.binning(
                data=key_integrand,
                bin_data0=key_bs_spectro,
                bins0=ktemp_bin,
                integrate=True,
                verb=verb,
                store=True,
                returnas=False,
                store_keys=key_integrand_interp_lamb,
            )

            # adjust ref
            refb = coll.dobj['bins'][ktemp_bin]['ref'][0]
            iref = coll.ddata[key_integrand_interp_lamb]['ref'].index(refb)
            # safety check
            shapei = coll.ddata[key_integrand_interp_lamb]['data'].shape
            assert shapei[iref] == lamb.size
            refb = list(coll.ddata[key_integrand_interp_lamb]['ref'])
            refb[iref] = v0['lamb']['ref'][0]
            coll.ddata[key_integrand_interp_lamb]['ref'] = tuple(refb)

            # remove bins
            coll.remove_bins(ktemp_bin)

        else:

            _ = coll.interpolate(
                keys=key_integrand,
                ref_key=key_bs_spectro,
                x0=v0['lamb']['key'],
                x1=None,
                grid=False,
                submesh=None,
                ref_com=None,
                domain=None,
                # azone=None,
                details=False,
                crop=None,
                nan0=False,
                val_out=0.,
                return_params=False,
                store=True,
                inplace=True,
                store_keys=key_integrand_interp_lamb,
            )

            # adjust data and units for integration
            dlamb = np.mean(np.diff(v0['lamb']['data']))
            coll.ddata[key_integrand_interp_lamb]['data'] *= dlamb
            coll.ddata[key_integrand_interp_lamb]['units'] = (
                asunits.Unit(coll.ddata[key_integrand_interp_lamb]['units'])
                * asunits.Unit(v0['lamb']['units'])
            )

        # -----------------------------
        # interpolate vs local cos

        if key_ref_cos is not None:

            # (nt, nbs0, nbs1, nlamb, ncos) => (nt, nbs0, nbs1, nlamb)
            douti = coll.interpolate(
                keys=key_integrand_interp_lamb,
                ref_key=key_bs,
                x0=R,
                x1=Z,
                grid=False,
                submesh=True,
                ref_com=ref_com,
                domain=None,
                # azone=None,
                details=False,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[key_integrand_interp_lamb]

        # -----------------------------
        # interpolate per spatial pts

        # (nt, nbs0, nbs1, nlamb) => (nt, npts, nlamb)

        douti = coll.interpolate(
            keys=key_integrand_interp_lamb,
            ref_key=key_bs,
            x0=R,
            x1=Z,
            grid=False,
            submesh=True,
            ref_com=ref_com,
            domain=None,
            # azone=None,
            details=False,
            crop=None,
            nan0=True,
            val_out=np.nan,
            return_params=False,
            store=False,
        )[key_integrand_interp_lamb]

        # -------------------
        # sum to get signal

        # ref
        ref_vos = list(coll.ddata[v0['ph']['key']]['ref'])
        ref_data = list(douti['ref'])
        ref_data[ref_data.index(None)] = ref_vos[-2]
        refc = [rr for rr in ref_data if rr in ref_vos]
        ref = (
            ref_data[:ref_data.index(refc[0])]
            + ref_vos[:ref_vos.index(refc[0])]
            + refc
        )

        # (nt, npts, nlamb)
        shape_data = douti['data'].shape

        # (n0, n1, npts, nlamb)
        shape_vos = v0['ph']['data'].shape

        # (nt, n0, n1)
        shape_sig = [
            shape_data[ref_data.index(rr)]
            if rr in ref_data else shape_vos[ref_vos.index(rr)]
            for ii, rr in enumerate(ref)
            if rr not in refc
        ]

        # slicing
        sli_vos = [
            slice(None) if rr in ref_vos[-2:]
            else (0 if rr in ref_vos else None)
            for ii, rr in enumerate(ref)
        ]

        # slicing sig
        sli_sig = [
            slice(None) if rr not in ref_vos else 0
            for ii, rr in enumerate(ref)
            if rr not in ref_vos[-2:]
        ]

        # axis for summation
        ax_lamb_data = ref_data.index(ref_vos[-1])
        ax_pts_data = ref_data.index(ref_vos[-2])
        ax_n0_vos = ref.index(ref_vos[0])
        ax_n1_vos = ref.index(ref_vos[1])
        ax_n0_sig = ref[:-2].index(ref_vos[0])
        ax_n1_sig = ref[:-2].index(ref_vos[1])

        # sum on pts and lamb
        # loop to save memory
        sig = np.full(shape_sig, np.nan)
        for (i0, i1) in itt.product(range(shape_vos[0]), range(shape_vos[1])):
            sli_sig[ax_n0_sig] = i0
            sli_sig[ax_n1_sig] = i1
            sli_vos[ax_n0_vos] = i0
            sli_vos[ax_n1_vos] = i1
            sig[tuple(sli_sig)] = np.nansum(
                douti['data']
                * v0['ph']['data'][tuple(sli_vos)],
                axis=(ax_lamb_data, ax_pts_data),
            )

        # new ref
        ref = [rr for rr in ref if rr not in refc]

        # units
        units = asunits.Unit(v0['ph']['units']) * asunits.Unit(douti['units'])

        # --------------
        # post-treatment

        # brightness
        if brightness is True:
            ketend = coll.dobj['diagnostic'][key_diag]['doptics'][k0]['etendue']
            etend = coll.ddata[ketend]['data']
            ref_e = coll.ddata[ketend]['ref']
            sh_etend = (
                [1]*ref.index(ref_e[0])
                + ref_e
                + [1]*(len(ref) - ref.index(ref_e[-1]) - 1)
            )
            sig /= etend.reshape(sh_etend)
            units = units / coll.ddata[ketend]['units']
        else:
            units = units

        # --------------
        # storing

        dout[k0] = {
            'key': f'{key}_{k0}',
            'data': sig,
            'ref': ref,
            'units': units,
        }

        # ----------
        # clean up

        if spectral_binning is True:
            coll.remove_bins(ktemp_bin)
        coll.remove_data(key_integrand_interp_lamb)

    return dout, dt