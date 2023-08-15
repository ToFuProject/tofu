# -*- coding: utf-8 -*-


import copy


import numpy as np
import datastock as ds


import datetime as dtm      # DB


from . import _class8_vos_broadband as _vos_broadband
from . import _class8_vos_spectro as _vos_spectro


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def compute_vos(
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
    # margins
    margin_poly=None,
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
    store=None,
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
        visibility,
        verb,
        debug,
        store,
        timing,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        key_mesh=key_mesh,
        res_RZ=res_RZ,
        res_phi=res_phi,
        res_lamb=res_lamb,
        visibility=visibility,
        verb=verb,
        debug=debug,
        store=store,
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

    # ------------
    # prepare output

    # dvos = _prepare_output(
        # coll=coll,
        # key_diag=key_diag,
        # shape_samp=sh,
        # spectro=spectro,
    # )

    # --------------
    # prepare optics

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    # timing
    if timing:
        t1 = dtm.datetime.now()     # DB
        dt1 = (t1 - t0).total_seconds()
    dt11, dt22 = 0, 0
    dt111, dt222, dt333 = 0, 0, 0
    dt1111, dt2222, dt3333, dt4444 = 0, 0, 0, 0

    # --------------
    # prepare optics

    dvos = {}
    for k0 in dcompute.keys():

            # ------------------
            # call relevant func

            (
                dvos[k0],
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
                bool_cross=bool_cross,
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
            spectro=spectro,
            replace_poly=replace_poly,
        )

    return dvos


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
    visibility=None,
    check=None,
    verb=None,
    debug=None,
    store=None,
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

    if store is True and spectro is True:
        msg = "storing vos is not available yet for spectrometers!"
        raise NotImplementedError(msg)

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
        visibility,
        verb,
        debug,
        store,
        timing,
    )


# ###########################################################
# ###########################################################
#               Prepare ouput
# ###########################################################


def _prepare_output(
    coll=None,
    key_diag=None,
    shape_samp=None,
    spectro=None,
):

    # -------
    # spectro

    if spectro is True:
        pass

    # -----------
    # non-spectro

    else:

        dvos = {'solid_angle_int': {}}
        for k0 in coll.dobj['diagnostic'][key_diag]['doptics'].keys():
            dgeom = coll.dobj['camera'][k0]['dgeom']
            sh = dgeom['shape']
            shape = tuple(np.r_[shape_samp, sh])
            ref = tuple([None, None] + list(dgeom['ref']))

            # --------
            # slice

            if is2d:
                def sli(ir, iz, ii):
                    pass

            else:
                def sli(ir, iz, ii):
                    pass



            # --------
            # dvos

            dvos['solid_angle_int'][k0] = {
                'data': None,
                'units': 'sr.m',
                'dim': '',
                'quant': '',
                'name': '',
                'ref': ref,
            }

    return dvos


# ###########################################################
# ###########################################################
#               store
# ###########################################################


def _store(
    coll=None,
    key_diag=None,
    dvos=None,
    spectro=None,
    replace_poly=None,
):

    # ------------
    # check inputs

    replace_poly = ds._generic_check._check_var(
        replace_poly, 'replace_poly',
        types=bool,
        default=True,
    )

    # ------------
    # store

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    for k0, v0 in dvos.items():

        # ----------------
        # pcross

        if replace_poly and v0.get('pcross0') is not None:

            # re-use previous keys
            kpc0, kpc1 = doptics[k0]['dvos']['pcross']
            kr = coll.ddata[kpc0]['ref'][0]

            # safety check
            if coll.ddata[kpc0]['data'].shape[1:] != v0['pcross0'].shape[1:]:
                msg = "Something is wrong"
                raise Exception(msg)

            coll._dref[kr]['size'] = v0['pcross0'].shape[0]
            coll._ddata[kpc0]['data'] = v0['pcross0']
            coll._ddata[kpc1]['data'] = v0['pcross1']

        # ----------------
        # 2d mesh sampling

        knpts = f'{k0}_vos_npts'
        kir = f'{k0}_vos_ir'
        kiz = f'{k0}_vos_iz'

        ref = tuple(list(coll.dobj['camera'][k0]['dgeom']['ref']) + [knpts])

        if knpts not in coll.dref.keys():
            coll.add_ref(knpts, size=v0['indr'].shape[1])

        # indr
        if kir not in coll.ddata.keys():
            coll.add_data(
                key=kir,
                data=v0['indr'],
                ref=ref,
                units='',
                dim='index',
            )

        # indz
        if kiz not in coll.ddata.keys():
            coll.add_data(
                key=kiz,
                data=v0['indz'],
                ref=ref,
                units='',
                dim='index',
            )

        # add in doptics
        doptics[k0]['dvos']['keym'] = v0['keym']
        doptics[k0]['dvos']['res_RZ'] = v0['res_RZ']
        doptics[k0]['dvos']['res_phi'] = v0['res_phi']
        doptics[k0]['dvos']['ind'] = (kir, kiz)

        # ------------
        # spectro

        if spectro:

            # keys
            kcos = f"{k0}_vos_cos"
            kph = f"{k0}_vos_ph"
            # klambmin =
            # klambmax =

            # add data
            coll.add_data(
                key=kcos,
                data=v0['cos'],
                ref=(knpts, kchan),
                units='',
            )

            coll.add_data(
                key=kph,
                data=v0['ph_counts'],
                ref=(knpts, kchan),
                units='sr.m3.m',
            )

            # add in doptics
            doptics[k0]['dvos']['cos'] = v0['cos']
            doptics[k0]['dvos']['ph'] = v0['ph']

        else:

            # keys
            ksa = f'{k0}_vos_sa'

            # add data
            coll.add_data(
                key=ksa,
                data=v0['sang']['data'],
                ref=ref,
                units=v0['sang']['units'],
            )

            # add in doptics
            doptics[k0]['dvos']['sang'] = ksa


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

    # ------
    # dvos

    if dvos is None:
        dvos = {}
        for k0 in key_cam:
            dop = coll.dobj['diagnostic'][key_diag]['doptics'][k0]['dvos']

            if dop.get('keym') is None:
                msg = (
                    "dvos was neither pre-computed nor provided for:\n"
                    f"\t- diag: '{key_diag}'\n"
                    f"\t- cam:  '{k0}'"
                )
                raise Exception(msg)

            dvos[k0] = {
                'keym': dop['keym'],
                'res_RZ': dop['res_RZ'],
                'indr': coll.ddata[dop['ind'][0]]['data'],
                'indz': coll.ddata[dop['ind'][1]]['data'],
                'sang': {
                    'data': coll.ddata[dop['sang']]['data'],
                    'units': coll.ddata[dop['sang']]['units'],
                },
            }

    else:
        dvos = copy.deepcopy(dvos)

    # ------------------
    # check keys of dvos

    # default
    if spectro is True:
        lk = [
            'keym', 'res_RZ', 'res_phi', 'indr', 'indz',
            'phi_min', 'phi_max', 'phi_mean',
            'ph_count', 'ncounts', 'cos',
            'lamb', 'lambmin', 'lambmax',
            'dV', 'etendlen', 'res_rock_curve',
        ]
    else:
        lk = ['keym', 'res_RZ', 'indr', 'indz', 'sang']

    # check
    c0 = (
        isinstance(dvos, dict)
        and all([
            k0 in dvos.keys()
            and all([k1 in dvos[k0].keys() for k1 in lk])
            for k0 in key_cam
        ])
    )

    # raise exception
    if not c0:
        msg = (
            "Arg dvos must be a dict with, for each camera, the keys:\n"
            + str(lk)
        )
        raise Exception(msg)

    # only keep desired cams
    lkout = [k0 for k0 in dvos.keys() if k0 not in key_cam]
    for k0 in lkout:
        del dvos[k0]

    return dvos