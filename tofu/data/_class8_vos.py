# -*- coding: utf-8 -*-


import copy


import numpy as np
import datastock as ds


import datetime as dtm      # DB


from . import _class8_vos_broadband as _vos_broadband
from . import _class8_vos_spectro as _vos_spectro
from . import _class8_vos_utilities as _vos_utilities


# ########################################################
# ########################################################
#                Main
# ########################################################


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
    lamb=None,
    res_lamb=None,
    n0=None,
    n1=None,
    convexHull=None,
    # margins
    margin_poly=None,
    # user-defined limits
    user_limits=None,
    # keep
    keep_cross=None,
    keep_hor=None,
    keep_3d=None,
    return_vector=None,
    # options
    add_points=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    # cleanup
    cleanup_pts=None,
    cleanup_lamb=None,
    compact_lamb=None,
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
        dkeep,
        cleanup_pts,
        cleanup_lamb,
        compact_lamb,
        return_vector,
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
        lamb=lamb,
        res_lamb=res_lamb,
        convexHull=convexHull,
        # cleanup
        cleanup_pts=cleanup_pts,
        cleanup_lamb=cleanup_lamb,
        compact_lamb=compact_lamb,
        # bool
        keep_cross=keep_cross,
        keep_hor=keep_hor,
        keep_3d=keep_3d,
        return_vector=return_vector,
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

    (
        func_RZphi_from_ind,
        func_ind_from_domain,
    ) = coll.get_sample_mesh_3d_func(
        key=key_mesh,
        res_RZ=res_RZ,
        mode='abs',
        res_phi=res_phi,
    )
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
        coll=coll,
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
            # new version
            func_RZphi_from_ind=func_RZphi_from_ind,
            func_ind_from_domain=func_ind_from_domain,
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
            lamb=lamb,
            res_lamb=res_lamb,
            n0=n0,
            n1=n1,
            convexHull=convexHull,
            bool_cross=bool_cross,
            # user-defined limits
            user_limits=user_limits,
            # keep
            dkeep=dkeep,
            return_vector=return_vector,
            # cleanup
            cleanup_pts=cleanup_pts,
            cleanup_lamb=cleanup_lamb,
            compact_lamb=compact_lamb,
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

        _vos_utilities._store_dvos(
            coll=coll,
            key_diag=key_diag,
            dvos=dvos,
            dref=dref,
            overwrite=overwrite,
            replace_poly=replace_poly,
            # mesh / res
            keym=key_mesh,
            res_RZ=res_RZ,
            res_phi=res_phi,
            res_lamb=res_lamb if spectro else None,
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
    lamb=None,
    res_lamb=None,
    convexHull=None,
    # keep3d
    keep_cross=None,
    keep_hor=None,
    keep_3d=None,
    return_vector=None,
    # cleanup
    cleanup_pts=None,
    cleanup_lamb=None,
    compact_lamb=None,
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

    if res_lamb is None and lamb is None:
        res_lamb = 0.01-10

    # -------------
    # what to keep
    # -------------

    # -----------
    # keep_cross

    keep_cross = ds._generic_check._check_var(
        keep_cross, 'keep_cross',
        types=bool,
        default=True,
    )

    # -----------
    # keep_hor

    keep_hor = ds._generic_check._check_var(
        keep_hor, 'keep_hor',
        types=bool,
        default=False,
    )

    # -----------
    # keep_3d

    keep_3d = ds._generic_check._check_var(
        keep_3d, 'keep_3d',
        types=bool,
        default=False,
    )

    # -------------
    # at least one

    dkeep = {
        '3d': keep_3d,
        'cross': keep_cross,
        'hor': keep_hor,
    }

    if not any([v0 for v0 in dkeep.values()]):
        msg = (
            "When computing VOS, you must keep at least one of:\n"
            "\t- keep_cross: cross-section projection of VOS\n"
            "\t\ti.e.: integrated toroidally\n"
            "\t- keep_hor  : horizontal projection of VOS\n"
            "\t\ti.e.: integrated vertically\n"
            "\t- keep_3d: Full 3d VOS (heavier)\n"
        )
        raise Exception(msg)

    # -----------
    # cleanup_pts

    cleanup_pts = ds._generic_check._check_var(
        cleanup_pts, 'cleanup_pts',
        types=bool,
        default=True,
    )

    # -----------
    # cleanup_lamb

    cleanup_lamb = ds._generic_check._check_var(
        cleanup_lamb, 'cleanup_lamb',
        types=bool,
        default=False,
    )

    # -----------
    # compact_lamb

    compact_lamb = ds._generic_check._check_var(
        compact_lamb, 'compact_lamb',
        types=bool,
        default=True,
    )

    # -----------
    # return_vector

    return_vector = ds._generic_check._check_var(
        return_vector, 'return_vector',
        types=bool,
        default=False,
    )

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

    if debug is None:
        debug = False

    if callable(debug):
        pass
    else:
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
        dkeep,
        cleanup_pts,
        cleanup_lamb,
        compact_lamb,
        return_vector,
        convexHull,
        visibility,
        verb,
        debug,
        store,
        overwrite,
        timing,
    )


def _user_limits_err(user_limits, lc=None):
    msg = (
        "user_limits must be either:\n"
        "\t- None: not used\n"
        "\t- True: remove all limits\n"
        "\t- dict: specify limits with at least one of the keys:\n"
        "\t\t- 'DR': list of 2 scalars or None\n"
        "\t\t- 'DZ': list of 2 scalars or None\n"
        "\t\t- 'Dphi': list of 2 scalars or None\n"
        + ("" if lc is None else f"lc = {lc}\n")
        + f"Provided:\n{user_limits}\n"
    )
    raise Exception(msg)


def _get_user_limits(
    coll=None,
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

        if not isinstance(user_limits, dict):
            _user_limits_err(user_limits)

        lc = [
            user_limits.get(ss) is not None
            and len(user_limits[ss]) == 2
            for ss in ['DR', 'DZ', 'Dphi']
        ]

        if not any(lc):
            _user_limits_err(user_limits, lc)

    # --------------
    # trivial

    if user_limits is None:
        return

    elif user_limits is True:
        user_limits = {}

    # --------------------
    # Preliminary check on Dphi
    # --------------------

    if user_limits.get('Dphi') is not None:
        user_limits['Dphi'] = ds._generic_check._check_flat1darray(
            user_limits['Dphi'], "user_limits.get('Dphi')",
            dtype=float,
            size=2,
            unique=True,
        )
        phi = np.linspace(user_limits['Dphi'][0], user_limits['Dphi'][1], 50)

    # --------------------
    # user-defined pcross
    # --------------------

    if user_limits.get('DR') is not None or user_limits.get('DZ') is not None:

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

        # --------------------
        # user-defined phor
        # --------------------

        # Dphi
        # phor_user
        if user_limits.get('Dphi') is not None:
            user_limits['phor_user'] = np.array([
                DR[1] * np.r_[-1, 1, 1, -1],
                DR[1] * np.r_[-1, -1, 1, 1],
            ])
        else:

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

    # -------------------
    # case with only Dphi => use each pixel's pcross
    # -------------------

    elif user_limits.get('Dphi') is not None:

        # Give phor_user the proper shape like phor TBF
        user_limits['phor0'] = {}
        user_limits['phor1'] = {}
        user_limits['dphi'] = {}

        for kcam in key_cam:
            kpc0, kpc1 = doptics[kcam]['dvos']['pcross']

            pcross0 = coll.ddata[kpc0]['data']
            pcross1 = coll.ddata[kpc1]['data']

            R = np.hypot(pcross0, pcross1)
            Rmin = np.min(R, axis=0)
            Rmax = np.max(R, axis=0)

            phi = phi.reshape(tuple([phi.size] + [1]*Rmin.ndim))
            phor0 = np.concatenate(
                (
                    Rmin[None, ...] * np.cos(phi),
                    Rmax[None, ...] * np.cos(phi[::-1]),
                ),
                axis=0,
            )

            phor1 = np.concatenate(
                (
                    Rmin[None, ...] * np.sin(phi),
                    Rmax[None, ...] * np.sin(phi[::-1]),
                ),
                axis=0,
            )

            user_limits['phor0'][kcam] = phor0
            user_limits['phor1'][kcam] = phor1

            shape_cam = coll.dobj['camera'][kcam]['dgeom']['shape']
            user_limits['dphi'][kcam] = np.array([
                np.full(shape_cam, user_limits['Dphi'][0]),
                np.full(shape_cam, user_limits['Dphi'][1]),
            ])

    # -----------
    # clean-up
    # ----------

    if len(user_limits) == 0:
        user_limits = None

    return user_limits


# ####################################################
# ####################################################
#               check vos proj
# ####################################################


def _check_vos_proj(
    coll=None,
    key=None,
    key_cam=None,
    logic=None,
    reduced=None,
):

    # ------------
    # inputs
    # ------------

    # key_diag, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
    )

    wdiag = coll._which_diagnostic
    doptics = coll.dobj[wdiag][key_diag]['doptics']

    # logic
    logic = ds._generic_check._check_var(
        logic, 'logic',
        allowed=[list, 'all', 'any'],
        default=list,
    )

    # reduced
    reduced = ds._generic_check._check_var(
        reduced, 'reduced',
        types=bool,
        default=False,
    )

    # ------------
    # dvosproj
    # ------------

    lproj = ['cross', 'hor', '3d']
    dvosproj = {kproj: [] for kproj in lproj}
    for kproj in lproj:
        istr = f"ind_{kproj}"
        for kcam in key_cam:
            dvos = doptics[kcam].get('dvos')
            c0 = (
                dvos is not None
                and dvos.get(istr) is not None
                and all([kk is not None for kk in dvos[istr]])
            )
            if c0 is True:
                dvosproj[kproj].append(kcam)

    # ------------
    # logic
    # ------------

    if logic == 'any':
        for kproj, lcam in dvosproj.items():
            dvosproj[kproj] = len(lcam) > 0

    elif logic == 'all':
        ncam = len(key_cam)
        for kproj, lcam in dvosproj.items():
            dvosproj[kproj] = len(lcam) == ncam

    # -----------
    # reduced
    # -----------

    if reduced is True:
        dvosproj = {
            k0: v0 for k0, v0 in dvosproj.items()
            if v0 is True
            or (logic is list and len(v0) > 0)
        }

    return dvosproj


# ####################################################
# ####################################################
#               get / check
# ####################################################


def _check_get_dvos(
    coll=None,
    key=None,
    key_cam=None,
    dvos=None,
):

    # ------------
    # keys
    # ------------

    # key_diag, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
    )

    # spectro?
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']

    # cross vs hor vs 3d
    dvosproj = coll.check_diagnostic_vos_proj(
        key=key_diag,
        key_cam=key_cam,
        logic=list,
    )

    # -------------------
    # prepare keys
    # -------------------

    # common as-is
    lk_asis = ['keym', 'res_RZ', 'res_phi']

    # common data
    lk_data = []

    # spectro-specific
    if spectro is True:
        lk_asis += ['res_lamb']
        lk_data = [
            'lamb', 'etendlen',
            # 'phi_min', 'phi_max', 'phi_mean',
        ]

    # tuple and projection-specific
    dk_tuple = {}
    for kproj, lcam in dvosproj.items():
        lk_data += [f'sang_{kproj}', f'dV_{kproj}', f'ndV_{kproj}']
        if spectro is True:
            lk_data += [f'ph_{kproj}', f'ncounts_{kproj}', f"indlamb_{kproj}"]

        dk_tuple[f'ind_{kproj}'] = [f"indr_{kproj}"]
        dk_tuple[f'vect_{kproj}'] = [
            f'vectx_{kproj}', f'vecty_{kproj}', f'vectz_{kproj}',
        ]
        if kproj != 'hor':
            dk_tuple[f'ind_{kproj}'].append(f"indz_{kproj}")
        if kproj != 'cross':
            dk_tuple[f'ind_{kproj}'].append(f"indphi_{kproj}")

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
            dvos[k0] = {}

            # fill in with res
            for k1 in lk_asis:
                dvos[k0][k1] = dop.get(k1)

            # fill in with the rest
            for k1 in lk_data:
                if k1 in dop.keys():
                    dvos[k0][k1] = {
                        'key': dop[k1],
                        **coll.ddata[dop[k1]],
                    }

            # tuples
            for ktup, vtup in dk_tuple.items():
                c0 = (
                    dop.get(ktup) is not None
                    and all([k1 is not None for k1 in dop[ktup]])
                )
                if c0:
                    for k1, k2 in zip(vtup, dop[ktup]):
                        dvos[k0][k1] = {
                            'key': k2,
                            **coll.ddata[k2],
                        }

        isstore = True

    else:
        isstore = False
        raise NotImplementedError("Check user-provided dvos not implemented")

    return key_diag, copy.deepcopy(dvos), isstore


# #########################################################
# #########################################################
#                 get vos to 3d
# #########################################################


# DEPRECATED ?
def get_dvos_xyz(coll=None, key_diag=None, key_cam=None, dvos=None):

    # ---------
    # get dvos

    key_diag, dvos, isstore = coll.check_diagnostic_dvos(
        key=key_diag,
        key_cam=key_cam,
        dvos=dvos,
    )

    # check
    k3d = 'indr_3d'
    if not all([v0.get(k3d) is not None for v0 in dvos.values()]):
        lstr = [
            f"\t\t- dvos['{k0}']: '{k3d}' {'not' if v0.get(k3d) is None else ''} available"
            for k0, v0 in dvos.items()
        ]
        msg = (
            "dvos can only provide (x, y, z) if it contains 3d information!\n"
            f"\t- key_diag: {key_diag}\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ---------
    # to xyz

    for k0, v0 in dvos.items():

        # sample
        dsamp = coll.get_sample_mesh(
            key=v0['keym'],
            res=v0['res_RZ'],
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

        # R, Z
        x0u = dsamp['x0']['data']
        x1u = dsamp['x1']['data']

        cos = np.cos(v0['phi_3d']['data'])
        sin = np.sin(v0['phi_3d']['data'])

        # store
        ref = v0['indr_3d']['ref']
        dvos[k0].update({
            'ptsx_3d': {
                'data': x0u[v0['indr_3d']['data']] * cos,
                'ref': ref,
                'units': 'm',
            },
            'ptsy_3d': {
                'data': x0u[v0['indr_3d']['data']] * sin,
                'ref': ref,
                'units': 'm',
            },
            'ptsz_3d': {
                'data': x1u[v0['indz_3d']['data']],
                'ref': ref,
                'units': 'm',
            },
        })

    return dvos
