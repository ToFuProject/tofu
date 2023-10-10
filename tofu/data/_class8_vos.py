# -*- coding: utf-8 -*-


import copy


import numpy as np
import datastock as ds


import datetime as dtm      # DB


from . import _class8_vos_broadband as _vos_broadband
from . import _class8_vos_spectro as _vos_spectro
from . import _class8_los_angles


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def compute_vos(
    # resources
    coll=None,
    key_diag=None,
    key_cam=None,
    key_mesh=None,
    config=None,
    # parameters
    res_RZ=None,
    res_phi=None,
    res_lamb=None,
    res_rock_curve=None,
    n0=None,
    n1=None,
    convexHull=None,
    # margins
    margin_poly=None,
    # user-defined limits
    user_limits=None,
    # options
    add_points=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    # bool
    visibility=None,
    convex=None,
    check=None,
    verb=None,
    debug=None,
    # storing
    store=None,
    overwrite=None,
    replace_poly=None,
    timing=None,
):

    if timing:
        t0 = dtm.datetime.now()     # DB

    # ------------
    # check inputs

    (
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res_RZ,
        res_phi,
        res_lamb,
        convexHull,
        visibility,
        verb,
        debug,
        store,
        overwrite,
        timing,
    ) = _check(
        # resources
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        key_mesh=key_mesh,
        # parameters
        res_RZ=res_RZ,
        res_phi=res_phi,
        res_lamb=res_lamb,
        convexHull=convexHull,
        # bool
        visibility=visibility,
        verb=verb,
        debug=debug,
        store=store,
        overwrite=overwrite,
        timing=timing,
    )

    # ----------
    # verb

    if verb is True:
        msg = f"\nComputing vos for diag '{key_diag}':"
        print(msg)

    # -----------
    # prepare

    if spectro:
        func = _vos_spectro._vos
    else:
        func = _vos_broadband._vos

    # ------------
    # sample mesh

    dsamp = coll.get_sample_mesh(
        key=key_mesh,
        res=res_RZ,
        mode='abs',
        grid=True,
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

    sh = dsamp['x0']['data'].shape
    x0f = dsamp['x0']['data'].ravel()
    x1f = dsamp['x1']['data'].ravel()

    sh1 = tuple([ss + 2 for ss in sh])
    bool_cross = np.zeros(sh1, dtype=bool)

    x0u = dsamp['x0']['data'][:, 0]
    x1u = dsamp['x1']['data'][0, :]
    x0l = np.r_[x0u[0] - (x0u[1] - x0u[0]), x0u, x0u[-1] + (x0u[-1] - x0u[-2])]
    x1l = np.r_[x1u[0] - (x1u[1] - x1u[0]), x1u, x1u[-1] + (x1u[-1] - x1u[-2])]
    x0l = np.repeat(x0l[:, None], x1l.size, axis=1)
    x1l = np.repeat(x1l[None, :], x0l.shape[0], axis=0)

    dx0 = x0u[1] - x0u[0]
    dx1 = x1u[1] - x1u[0]

    # --------------
    # prepare optics

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    # -------------
    # user-defined

    user_limits = _get_user_limits(
        user_limits=user_limits,
        doptics=doptics,
        key_cam=list(dcompute.keys()),
        # lims
        x0u=x0u,
        x1u=x1u,
    )


    # timing
    if timing:
        t1 = dtm.datetime.now()     # DB
        dt1 = (t1 - t0).total_seconds()
    dt11, dt22 = 0, 0
    dt111, dt222, dt333 = 0, 0, 0
    dt1111, dt2222, dt3333, dt4444 = 0, 0, 0, 0

    # --------------
    # prepare optics

    dvos, dref = {}, {}
    for k0 in dcompute.keys():

            # ------------------
            # call relevant func

            (
                dvos[k0], dref[k0],
                dt11, dt22,
                dt111, dt222, dt333,
                dt1111, dt2222, dt3333, dt4444,
            ) = func(
                # ressources
                coll=coll,
                doptics=doptics,
                key_diag=key_diag,
                key_cam=k0,
                dsamp=dsamp,
                # inputs sample points
                x0u=x0u,
                x1u=x1u,
                x0f=x0f,
                x1f=x1f,
                x0l=x0l,
                x1l=x1l,
                dx0=dx0,
                dx1=dx1,
                # options
                sh=sh,
                res_RZ=res_RZ,
                res_phi=res_phi,
                res_lamb=res_lamb,
                res_rock_curve=res_rock_curve,
                n0=n0,
                n1=n1,
                convexHull=convexHull,
                bool_cross=bool_cross,
                # user-defined limits
                user_limits=user_limits,
                # parameters
                margin_poly=margin_poly,
                config=config,
                visibility=visibility,
                verb=verb,
                # debug
                debug=debug,
                # timing
                timing=timing,
                dt11=dt11,
                dt111=dt111,
                dt1111=dt1111,
                dt2222=dt2222,
                dt3333=dt3333,
                dt4444=dt4444,
                dt222=dt222,
                dt333=dt333,
                dt22=dt22,
            )

            dvos[k0]['keym'] = key_mesh
            dvos[k0]['res_RZ'] = res_RZ
            dvos[k0]['res_phi'] = res_phi
            if spectro is True:
                dvos[k0]['res_lamb'] = res_lamb
                dvos[k0]['res_rock_curve'] = res_rock_curve

    # timing
    if timing:
        t2 = dtm.datetime.now()     # DB
        print("\nTIMING\n--------")
        print(f'Prepare: {dt1} s')
        print(f'\tdt11 (pepare cam): {dt11} s')
        print(f'\t\tdt111 (prepare pix): {dt111} s')
        print(f"\t\t\tdt1111 (prepare): {dt1111} s")
        print(f"\t\t\tdt2222 (compute): {dt2222} s")
        print(f"\t\t\tdt3333 (format):  {dt3333} s")
        print(f"\t\t\tdt4444 (ind_bool):  {dt4444} s")
        print(f'\t\tdt222 (compute): {dt222} s')
        print(f'\t\tdt333 (get poly): {dt333} s')
        print(f'\tdt22 (interp poly): {dt22} s')
        print(f'loop total: {(t2-t1).total_seconds()} s')

    # -------------
    # replace

    if store is True:

        _store(
            coll=coll,
            key_diag=key_diag,
            dvos=dvos,
            dref=dref,
            spectro=spectro,
            overwrite=overwrite,
            replace_poly=replace_poly,
        )

    return dvos, dref


# ###########################################################
# ###########################################################
#               check
# ###########################################################


def _check(
    coll=None,
    key_diag=None,
    key_cam=None,
    key_mesh=None,
    res_RZ=None,
    res_phi=None,
    res_lamb=None,
    convexHull=None,
    # bool
    visibility=None,
    check=None,
    verb=None,
    debug=None,
    store=None,
    overwrite=None,
    timing=None,
):

    # --------
    # key_diag

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([len(v1['optics']) > 0 for v1 in v0['doptics'].values()])
    ]
    key_diag = ds._generic_check._check_var(
        key_diag, 'key_diag',
        types=str,
        allowed=lok,
    )

    # spectro, is2d
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # doptics
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']

    # ------------
    # key_cam

    lok = [k0 for k0, v0 in doptics.items() if len(v0['optics']) > 0]
    if isinstance(key_cam, str):
        key_cam = [key_cam]
    key_cam = ds._generic_check._check_var_iter(
        key_cam, 'key_cam',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    # -----------------
    # doptics, dcompute

    doptics = {k0: v0 for k0, v0 in doptics.items() if k0 in key_cam}

    dcompute = {
        k0: {'compute': len(v0['optics']) > 0}
        for k0, v0 in doptics.items()
    }

    # --------
    # key_mesh

    wm = coll._which_mesh
    lok = list(coll.dobj.get(wm, {}).keys())
    key_mesh = ds._generic_check._check_var(
        key_mesh, 'key_mesh',
        types=str,
        allowed=lok,
    )

    # -------------------------------------------------
    # ldeti: list of individual camera dict (per pixel)

    for k0, v0 in doptics.items():

        dgeom = coll.dobj['camera'][k0]['dgeom']
        cx, cy, cz = coll.get_camera_cents_xyz(key=k0)
        dvect = coll.get_camera_unit_vectors(key=k0)
        outline = dgeom['outline']
        out0 = coll.ddata[outline[0]]['data']
        out1 = coll.ddata[outline[1]]['data']
        is2d = dgeom['nd'] == '2d'
        par = dgeom['parallel']
        dcompute[k0]['shape0'] = cx.shape

        if is2d:
            cx = cx.ravel()
            cy = cy.ravel()
            cz = cz.ravel()

        nd = cx.size

        dcompute[k0]['ldet'] = [
            {
                'cents_x': cx[ii],
                'cents_y': cy[ii],
                'cents_z': cz[ii],
                'outline_x0': out0,
                'outline_x1': out1,
                'nin_x': dvect['nin_x'] if par else dvect['nin_x'][ii],
                'nin_y': dvect['nin_y'] if par else dvect['nin_y'][ii],
                'nin_z': dvect['nin_z'] if par else dvect['nin_z'][ii],
                'e0_x': dvect['e0_x'] if par else dvect['e0_x'][ii],
                'e0_y': dvect['e0_y'] if par else dvect['e0_y'][ii],
                'e0_z': dvect['e0_z'] if par else dvect['e0_z'][ii],
                'e1_x': dvect['e1_x'] if par else dvect['e1_x'][ii],
                'e1_y': dvect['e1_y'] if par else dvect['e1_y'][ii],
                'e1_z': dvect['e1_z'] if par else dvect['e1_z'][ii],
            }
            for ii in range(nd)
        ]

    # -----------
    # res_RZ

    if res_RZ is None:
        res_RZ = 0.01
    if np.isscalar(res_RZ):
        res_RZ = np.r_[res_RZ, res_RZ]
    res_RZ = np.atleast_1d(res_RZ).ravel().astype(float)
    assert res_RZ.size == 2
    res_RZ = res_RZ.tolist()

    # -----------
    # res_phi

    if res_phi is None:
        res_phi = 0.01

    # -----------
    # res_lamb

    if res_lamb is None:
        res_lamb = 0.01e-10

    # -----------
    # convexHull - to get overall pcross and phor, faster if many pixels

    convexHull = ds._generic_check._check_var(
        convexHull, 'convexHull',
        types=bool,
        default=False,
    )

    # -----------
    # visibility

    visibility = ds._generic_check._check_var(
        visibility, 'visibility',
        types=bool,
        default=True,
    )

    # -----------
    # verb

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # -----------
    # debug

    debug = ds._generic_check._check_var(
        debug, 'debug',
        types=bool,
        default=False,
    )

    # -----------
    # store

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
    )

    # -----------
    # overwrite

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # -----------
    # timing

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    return (
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res_RZ,
        res_phi,
        res_lamb,
        convexHull,
        visibility,
        verb,
        debug,
        store,
        overwrite,
        timing,
    )


def _get_user_limits(
    user_limits=None,
    doptics=None,
    key_cam=None,
    # lims
    x0u=None,
    x1u=None,
):

    # ---------------
    # check

    # default
    c0 = all([v0['los'] is None for v0 in doptics.values()])
    if user_limits is None and c0:
        user_limits = True

    # dict
    if user_limits not in [None, True]:

        c0 = (
            isinstance(user_limits, dict)
            and any([
                user_limits.get(ss) is not None
                and user_limits[ss]
                for ss in ['DR', 'DZ', 'Dphi']
            ])
        )

        if not c0:
            msg = (
                "user_limits must be either:\n"
                "\t- None: not used\n"
                "\t- True: remove all limits\n"
                "\t- dict: specify limits with at least one of the keys:\n"
                "\t\t- 'DR': list of 2 scalars or None\n"
                "\t\t- 'DZ': list of 2 scalars or None\n"
                "\t\t- 'Dphi': list of 2 scalars or None\n"
            )
            raise Exception(msg)

    # --------------
    # trivial

    if user_limits is None:
        return

    elif user_limits is True:
        user_limits = {}

    # ---------------
    # user-defined limits

    # DR
    if user_limits.get('DR') is None:
        user_limits['DR'] = [x0u[0]-1e-9, x0u[-1]+1e-9]
    user_limits['DR'] = ds._generic_check._check_flat1darray(
        user_limits['DR'], "user_limits.get('DR')",
        dtype=float,
        size=2,
        unique=True,
        sign='>=0',
    )

    # DZ
    if user_limits.get('DZ') is None:
        user_limits['DZ'] = [x1u[0]-1e-9, x1u[-1]+1e-9]
    user_limits['DZ'] = ds._generic_check._check_flat1darray(
        user_limits['DZ'], "user_limits['DZ']",
        dtype=float,
        size=2,
        unique=True,
    )

    # pcross_user
    DR = user_limits['DR']
    DZ = user_limits['DZ']
    user_limits['pcross_user'] = np.array([
        np.r_[DR[0], DR[1], DR[1], DR[0]],
        np.r_[DZ[0], DZ[0], DZ[1], DZ[1]],
    ])

    # Dphi
    # phor_user
    if user_limits.get('Dphi') is None:
        user_limits['phor_user'] = np.array([
            DR[1] * np.r_[-1, 1, 1, -1],
            DR[1] * np.r_[-1, -1, 1, 1],
        ])

    else:
        phi = np.linspace(user_limits['Dphi'][0], user_limits['Dphi'][1], 50)
        user_limits['phor_user'] = np.array([
            np.r_[
                DR[0]*np.cos(phi),
                DR[1]*np.cos(phi[::-1]),
            ],
            np.r_[
                DR[0]*np.sin(phi),
                DR[1]*np.sin(phi[::-1]),
            ],
        ])

    return user_limits


# ###########################################################
# ###########################################################
#               store
# ###########################################################


def _store(
    coll=None,
    key_diag=None,
    dvos=None,
    dref=None,
    spectro=None,
    overwrite=None,
    replace_poly=None,
):

    # ------------
    # check inputs

    replace_poly = ds._generic_check._check_var(
        replace_poly, 'replace_poly',
        types=bool,
        default=True,
    )

    # ----------------------
    # prepare what to store

    lk_com = ['indr', 'indz']
    if spectro is True:
        lk = [
            'lamb',
            'ph', 'cos', 'ncounts',
            'phi_min', 'phi_max',
            # optional
            'lamb0', 'dlamb',
            'phi_mean',
            'dV', 'etendlen',
        ]
    else:
        lk = ['sang']


    # ------------
    # store

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    for k0, v0 in dvos.items():

        # ----------------
        # pcross replacement

        if replace_poly and v0.get('pcross0') is not None:

            if v0.get('phor0') is None:
                phor0, phor1 = None, None
            else:
                phor0, phor1 = v0['phor0']['data'], v0['phor1']['data']

            # re-use previous keys
            if doptics[k0].get('dvos') is None:

                _class8_los_angles._vos_from_los_store(
                    coll=coll,
                    key=key_diag,
                    key_cam=k0,
                    pcross0=v0['pcross0']['data'],
                    pcross1=v0['pcross1']['data'],
                    phor0=phor0,
                    phor1=phor1,
                    dphi=None,
                )

            else:
                kpc0, kpc1 = doptics[k0]['dvos']['pcross']
                kr = coll.ddata[kpc0]['ref'][0]

                # safety check
                shape_pcross = v0['pcross0']['data'].shape
                if coll.ddata[kpc0]['data'].shape[1:] != shape_pcross[1:]:
                    msg = "Something is wrong"
                    raise Exception(msg)

                coll._dref[kr]['size'] = shape_pcross[0]
                coll._ddata[kpc0]['data'] = v0['pcross0']['data']
                coll._ddata[kpc1]['data'] = v0['pcross1']['data']
                if phor0 is not None:
                    kph0, kph1 = doptics[k0]['dvos']['phor']
                    coll._ddata[kph0]['data'] = v0['phor0']['data']
                    coll._ddata[kph1]['data'] = v0['phor1']['data']

        # ----------------
        # add ref of sang

        for k1, v1 in dref[k0].items():
            if v1['key'] in coll.dref.keys():
                if overwrite is True:
                    coll.remove_ref(v1['key'], propagate=True)
                    coll.add_ref(**v1)
                elif v1['size'] != coll.dref[v1['key']]['size']:
                    msg = (
                        f"Mismatch between new vs existing size ref {k1} '{v1['key']}'"
                        f"\t- existing size = {coll.dref[k1]['size']}\n"
                        f"\t- new size      = {v1['size']}\n"
                    )
                    raise Exception(msg)
                else:
                    pass
            else:
                coll.add_ref(**v1)

        # ----------------
        # add data

        for k1 in lk_com + lk:

            if k1 not in v0.keys():
                continue

            if v0[k1]['key'] in coll.ddata.keys():
                if overwrite is True:
                    coll.remove_data(key=v0[k1]['key'])
                else:
                    msg = (
                        f"Not overwriting existing data '{k1}'\n"
                        "To force update use overwrite = True"
                    )
                    raise Exception(msg)

            coll.add_data(**v0[k1])

        # ---------------
        # add in doptics

        doptics[k0]['dvos']['keym'] = v0['keym']
        doptics[k0]['dvos']['res_RZ'] = v0['res_RZ']
        doptics[k0]['dvos']['res_phi'] = v0['res_phi']
        doptics[k0]['dvos']['ind'] = (v0['indr']['key'], v0['indz']['key'])
        if spectro:
            doptics[k0]['dvos']['res_lamb'] = v0['res_lamb']
            doptics[k0]['dvos']['res_rock_curve'] = v0['res_rock_curve']

        # -----------------
        # add data keys to doptics

        for k1 in lk:
            if k1 in v0.keys():
                doptics[k0]['dvos'][k1] = v0[k1]['key']


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def _check_get_dvos(
    coll=None,
    key=None,
    key_cam=None,
    dvos=None,
):

    # ------------
    # keys

    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
    )
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']

    # -------------------
    # prepare keys

    lk_sca = ['res_RZ', 'res_phi']
    if spectro is True:
        lk_sca += ['res_lamb', 'res_rock_curve']
        lk = [
            'lamb',
            'phi_min', 'phi_max', 'phi_mean',
            'ph', 'ncounts', 'cos',
            'lamb0', 'dlamb',
            'dV', 'etendlen',
        ]
    else:
        lk = ['sang']

    lk_all = lk_sca + lk + ['keym', 'indr', 'indz']

    # ------
    # dvos

    if dvos is None:

        dvos = {}
        doptics = coll.dobj['diagnostic'][key_diag]['doptics']
        for k0 in key_cam:

            # safety check 1
            if doptics[k0].get('dvos') is None:
                msg = (
                    "Please provide dvos if coll.dobj['diagnostic']"
                    f"['{key_diag}']['{k0}']['doptics']['dvos'] is None!"
                )
                raise Exception(msg)

            dop = doptics[k0]['dvos']

            # safety check 2
            if dop.get('keym') is None:
                msg = (
                    "dvos was neither pre-computed nor provided for:\n"
                    f"\t- diag: '{key_diag}'\n"
                    f"\t- cam:  '{k0}'"
                )
                raise Exception(msg)

            # fill in dict with mesh and indices
            dvos[k0] = {
                'keym': dop['keym'],
                'indr': coll.ddata[dop['ind'][0]],
                'indz': coll.ddata[dop['ind'][1]],
            }

            # fill in with res
            for k1 in lk_sca:
                dvos[k0][k1] = dop[k1]

            # fill in with the rest
            for k1 in lk:
                if k1 in dop.keys():
                    dvos[k0][k1] = {
                        'key': dop[k1],
                        **coll.ddata[dop[k1]],
                    }
    else:
        pass

    # copy to avoid changing the original
    dvos = copy.deepcopy(dvos)

    # ------------------
    # check keys of dvos

    # check
    c0 = (
        isinstance(dvos, dict)
        and all([
            k0 in dvos.keys()
            and all([k1 in dvos[k0].keys() for k1 in lk_all])
            for k0 in key_cam
        ])
    )

    # raise exception
    if not c0:
        msg = (
            "Arg dvos must be a dict with, for each camera, the keys:\n"
            + str(lk_all)
        )
        raise Exception(msg)

    # only keep desired cams
    lkout = [k0 for k0 in dvos.keys() if k0 not in key_cam]
    for k0 in lkout:
        del dvos[k0]

    return dvos