# -*- coding: utf-8 -*-


import warnings
import itertools as itt


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


# ################################################################
# ################################################################
#                       Diagnostics
# ################################################################


def _diagnostics_check(
    coll=None,
    key=None,
    doptics=None,
    stack=None,
):

    # ------
    # key
    # ------

    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('diagnostic', {}),
        short='d',
        key=key,
    )

    # -----------
    # doptics
    # -----------

    doptics, lap, lfilt, lcryst, lgrat = _check_doptics_basics(
        coll=coll,
        doptics=doptics,
        key=key,
    )

    # -----------------
    # types of camera
    # -----------------

    lcam = list(doptics.keys())
    types = [coll.dobj['camera'][k0]['dgeom']['nd'] for k0 in lcam]

    if len(set(types)) > 1:
        msg = (
            f"diag '{key}': all cameras must be of the same type (1d or 2d)!\n"
            f"\t- cameras: {lcam}\n"
            f"\t- types: {types}"
        )
        raise Exception(msg)

    is2d = types[0] == '2d'

    # -------------------------------------------------
    # check all optics are on good side of each camera
    # -------------------------------------------------

    for cam in lcam:

        dgeom_cam = coll.dobj['camera'][cam]['dgeom']
        last_ref = cam
        last_ref_cls = 'camera'

        is2d = (dgeom_cam['nd'] == '2d')
        parallel = dgeom_cam['parallel']

        # -----------------------------------
        # all optics are common to all pixels

        if doptics[cam]['pinhole'] is True:
            for ii, oo in enumerate(doptics[cam]['optics']):
                last_ref_cls, last_ref = _check_optic(
                    coll=coll,
                    key=key,
                    # optic
                    oo=oo,
                    ocls=doptics[cam]['cls'][ii],
                    cam=cam,
                    is2d=is2d,
                    parallel=parallel,
                    ind_pix=None,
                    dgeom_cam=dgeom_cam,
                    # iterated over
                    last_ref_cls=last_ref_cls,
                    last_ref=last_ref,
                )

        # ------------------------------
        # each pixel has specific optics

        else:

            optics = np.array(doptics[cam]['optics'])
            lcls = np.array(doptics[cam]['cls'])
            paths = doptics[cam]['paths']

            lind = [range(ss) for ss in dgeom_cam['shape']]
            for ind in itt.product(*lind):

                sli = tuple(list(ind) + [slice(None)])
                iop = paths[sli]

                lopi = optics[iop]
                lclsi = lcls[iop]

                for ii, oo in enumerate(lopi):
                    last_ref_cls, last_ref = _check_optic(
                        coll=coll,
                        key=key,
                        # optic
                        oo=oo,
                        ocls=lclsi[ii],
                        cam=cam,
                        is2d=is2d,
                        parallel=parallel,
                        ind_pix=ind,
                        dgeom_cam=dgeom_cam,
                        # iterated over
                        last_ref_cls=last_ref_cls,
                        last_ref=last_ref,
                    )

            # no spectro
            assert last_ref_cls == 'camera', (last_ref_cls, last_ref)

    # -----------------
    # is spectro, PHA
    # -----------------

    spectro, PHA = _get_spectro_PHA(
        coll=coll,
        doptics=doptics,
        key=key,
        lcam=lcam,
    )

    # -----------
    # ispectro
    # -----------

    if spectro:
        for k0, v0 in doptics.items():
            doptics[k0]['ispectro'] = [
                ii for ii, cc in enumerate(v0['cls'])
                if cc in ['grating', 'crystal']
            ]

    # -----------------
    # stack
    # -----------------

    stack = ds._generic_check._check_var(
        stack, 'stack',
        types=str,
        default='horizontal',
        allowed=['horizontal', 'vertical'],
    )

    return key, lcam, doptics, is2d, spectro, stack, PHA


# ##################################################################
# ##################################################################
#                 check doptics basics
# ##################################################################


def _check_doptics_basics(
    coll=None,
    doptics=None,
    key=None,
):

    # --------------------
    # level 0: class
    # --------------------

    if isinstance(doptics, str):
        doptics = {doptics: []}

    if isinstance(doptics, (list, tuple)):
        doptics = {doptics[0]: type(doptics)(doptics[1:])}

    if not isinstance(doptics, dict):
        _err_doptics(
            key=key,
            doptics=doptics,
            extra_msg="\nShould be a dict!\n",
        )

    # ----------
    # check keys

    lcam = list(coll.dobj.get('camera', {}).keys())

    c0 = all([
        (isinstance(k0, str) and k0 in lcam)
        and isinstance(v0, (str, list, tuple, dict))
        for k0, v0 in doptics.items()
    ])

    if not c0:
        _err_doptics(
            key=key,
            doptics=doptics,
            extra_msg="\nSome keys (cam) are not valid!\n",
        )

    # --------------------------
    # start re-arranging as dict

    for k0, v0 in doptics.items():

        if isinstance(v0, str):
            doptics[k0] = {'optics': [v0]}
        elif isinstance(v0, (list, tuple)):
            doptics[k0] = {'optics': v0}
        elif isinstance(doptics.get('optics'), str):
            doptics[k0] = {'optics': [doptics[k0]]}

        c0 = (
            isinstance(doptics[k0].get('optics'), (list, tuple))
            and all([isinstance(k1, str) for k1 in doptics[k0]['optics']])
        )
        if not c0:
            _err_doptics(
                key=key,
                doptics=doptics,
                extra_msg="\n'optics' must be a list or tuple of known optics!\n",
            )

    # --------------------
    # level 1: checking each optics class
    # --------------------

    # -------
    # classes

    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())

    lop = lap + lfilt + lcryst + lgrat

    # -----------
    # check names

    dkout = {
        k0: [k1 for k1 in v0['optics'] if k1 not in lop]
        for k0, v0 in doptics.items()
        if any([k1 not in lop for k1 in v0['optics']])
    }

    if len(dkout) > 0:
        _err_doptics(key=key, dkout=dkout, doptics=doptics)

    # -----------------
    # check path matrix
    # -----------------

    doptics2 = {}
    for k0, v0 in doptics.items():

        # -------------------
        # get relevant shapes

        shape_cam = coll.dobj['camera'][k0]['dgeom']['shape']
        noptics = len(v0['optics'])
        shape = tuple(np.r_[shape_cam, noptics])

        # ---------------------------------------
        # no matrix => pinhole or pure collimator

        if v0.get('paths') is None:

            # pinhole camera
            if isinstance(v0['optics'], list):
                pinhole = True
                doptics[k0]['paths'] = None

            # pure collimator camera
            elif isinstance(v0['optics'], tuple) and len(shape_cam) == 1:

                mod = noptics % shape_cam[0]
                if mod != 0:
                    msg = (
                        "Directly providing optics as tuple for collimators:\n"
                        f"\t- diag = '{key}'\n"
                        f"\t- cam = '{k0}'\n"
                        f"\t- shape_cam = {shape_cam}\n"
                        f"\t- noptics = {noptics}\n"
                        f"\t- mod = {mod}\n"
                        "noptics should be a multiple of the nb of sensors!\n"
                    )
                    raise Exception(msg)

                nmult = noptics // shape_cam[0]
                paths = np.zeros((shape_cam[0], noptics), dtype=bool)
                for ii in range(shape_cam[0]):
                    paths[ii, (ii*nmult):(ii+1)*nmult] = True
                doptics[k0]['paths'] = paths

                doptics[k0]['optics'] = list(v0['optics'])

            else:
                emsg = "\nNeither list nor tuple (shape_cam = {shape_cam})!\n"
                _err_doptics(
                    key=key,
                    doptics=doptics,
                    extra_msg=emsg,
                )

        # ------------------------
        # matrix user-provided

        else:

            # check paths is a boolean array of the right shape
            try:
                doptics[k0]['paths'] = np.asarray(v0['paths'], dtype=bool)
                assert doptics[k0]['paths'].shape == shape

            except Exception as err:
                emsg = (
                    f"\n doptics['{k0}']['paths'].shape = "
                    f"{doptics[k0]['paths'].shape} vs {shape}\n"
                )
                err0 = _err_doptics(
                    key=key,
                    doptics=doptics,
                    returnas=True,
                    shape_cam=emsg,
                )
                raise err0 from err

        # -------------------------------------------
        # check if pinhole (all apertures are common)

        if doptics[k0]['paths'] is not None:

            pinhole = np.all(doptics[k0]['paths'])
            if pinhole is True:
                doptics[k0]['paths'] = None

        # ---------
        # get lcls

        lcls = _get_optics_cls(
            coll=coll,
            optics=doptics[k0]['optics'],
        )[1]

        # ---------------------
        # populate doptics2

        doptics2[k0] = {
            'camera': k0,
            'optics': doptics[k0]['optics'],
            'cls': lcls,
            'pinhole': bool(pinhole),
            'paths': doptics[k0]['paths'],
            'los': None,
            'etendue': None,
            'etend_type': None,
            'amin': None,
            'amax': None,
        }

    return doptics2, lap, lfilt, lcryst, lgrat


# -----------
# raise err
# -----------


def _err_doptics(
    key=None,
    dkout=None,
    doptics=None,
    returnas=None,
    extra_msg=None,
):

    # ---------------
    # msg

    msg = (
        f"diag '{key}': arg doptics must be a dict with:\n"
        "\t- keys: key to existing camera\n"
        "\t- values: one of the following:\n\n"
        "\t\t- list of keys to apertures\n"
        "\t\t\tIn this case tofu will assume pinhole camera\n"
        "\t\t\tMeaning all apertures are common to all pixels\n"
        "\t\t- tuple of keys to aperures\n"
        "\t\t\tIn this case tofu will assume collimator camera\n"
        "\t\t\tMeaning each pixel is associated to N apertures\n"
        "\t\t\tWhere N is an integer napertures = N x ncam\n"
        "\t\t\t(the first N apertures go to the first pixel...)\n"
        "\t\t- (most general) a dict of the form:\n"
        "\t\t\t'optics': list of apertures\n"
        "\t\t\t'paths': (shape_cam, noptics) bool array\n"
    )
    if dkout is not None and len(dkout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dkout.items()]
        msg += "Wrong key / value pairs:\n" + "\n".join(lstr)

    if extra_msg is not None:
        msg += extra_msg

    msg += f"\nProvided:\n{doptics}"

    # ---------------
    # raise vs return

    if returnas is True:
        return Exception(msg)
    else:
        raise Exception(msg)


# ##################################################################
# ##################################################################
#                 check optics 3d vs camera
# ##################################################################


def _check_optic(
    coll=None,
    key=None,
    # optic
    oo=None,
    ocls=None,
    cam=None,
    is2d=None,
    parallel=None,
    ind_pix=None,
    dgeom_cam=None,
    # iterated over
    last_ref_cls=None,
    last_ref=None,
):

    # --------------
    # prepare
    # --------------

    dgeom = coll.dobj[ocls][oo]['dgeom']
    px, py, pz = coll.get_optics_poly(key=oo)
    dgeom_lastref = coll.dobj[last_ref_cls][last_ref]['dgeom']

    # -------------------
    # single cent and nin
    # -------------------

    if (last_ref == cam and is2d) or last_ref != cam:

        cent = dgeom_lastref['cent']
        nin = dgeom_lastref['nin']

        iout = (
            (px - cent[0])*nin[0]
            + (py - cent[1])*nin[1]
            + (pz - cent[2])*nin[2]
        ) <= 0

        if np.any(iout):
            msg = (
                f"diag '{key}':\n"
                f"The following points of {ocls} '{oo}' are on the wrong"
                f"side of lastref {last_ref_cls} '{last_ref}':\n"
                f"{iout.nonzero()[0]}\n\n"
                f"last ref {last_ref_cls} '{last_ref}':\n{dgeom_lastref}\n\n"
                f"optics {ocls} '{oo}':\n{dgeom}\n\n"
                "Tip:\n"
                "\tMake sure to provide optics ordered from camera to plasma"
            )
            raise Exception(msg)

    # --------------
    # prepare
    # --------------

    else:

        # --------------------
        # get pixel center(s)

        cx, cy, cz = dgeom_cam['cents']
        if ind_pix is None:
            cx = coll.ddata[cx]['data'][None, ...]
            cy = coll.ddata[cy]['data'][None, ...]
            cz = coll.ddata[cz]['data'][None, ...]

        else:
            cx = coll.ddata[cx]['data'][ind_pix]
            cy = coll.ddata[cy]['data'][ind_pix]
            cz = coll.ddata[cz]['data'][ind_pix]

        # ------------------------
        # get pixel unit vector(s)

        if dgeom_cam['parallel']:
            ninx, niny, ninz = dgeom_cam['nin']
        else:
            ninx, niny, ninz = dgeom_cam['nin']
            if ind_pix is None:
                ninx = coll.ddata[ninx]['data'][None, ...]
                niny = coll.ddata[niny]['data'][None, ...]
                ninz = coll.ddata[ninz]['data'][None, ...]
            else:
                ninx = coll.ddata[ninx]['data'][ind_pix]
                niny = coll.ddata[niny]['data'][ind_pix]
                ninz = coll.ddata[ninz]['data'][ind_pix]

        # --------------------------
        # spot points out of domain

        iout = (
            (px[:, None] - cx)*ninx
            + (py[:, None] - cy)*niny
            + (pz[:, None] - cz)*ninz
        ) <= 0

        # ---------------------
        # raise warning if any

        if np.any(iout):
            msg = (
                f"diag '{key}':\n"
                f"The following points of {ocls} '{oo}' are on the wrong"
                f"side of lastref {last_ref_cls} '{last_ref}':\n"
                f"{iout.nonzero()[0]}\n\n"
                f"last ref {last_ref_cls} '{last_ref}':\n{dgeom_lastref}\n\n"
                f"optics {ocls} '{oo}':\n{dgeom}\n\n"
                "Tip:\n"
                "\tMake sure to provide optics ordered from camera to plasma\n"
            )
            warnings.warn(msg)

    # --------------
    # update last_ref ?
    # --------------

    if ocls in ['crystal', 'grating']:
        last_ref = oo
        last_ref_cls = ocls

    return last_ref_cls, last_ref


# ##################################################################
# ##################################################################
#                Get spectro PHA
# ##################################################################


def _get_spectro_PHA(
    coll=None,
    doptics=None,
    key=None,
    lcam=None,
):

    # --------------
    # is spectro
    # --------------

    dspectro = {
        k0: (
            any([
                k1 in coll.dobj.get('crystal', {}).keys()
                or k1 in coll.dobj.get('grating', {}).keys()
                for k1 in v0['optics']
            ])
        )
        for k0, v0 in doptics.items()
    }

    lc = [
        all([v0 for v0 in dspectro.values()]),
        all([not v0 for v0 in dspectro.values()]),
    ]
    if np.sum(lc) != 1:
        msg = (
            f"diag '{key}' must be either all spectro or all non-spectro!\n"
            + "\n".join([f"\t- {k0}: {v0}" for k0, v0 in dspectro.items()])
        )
        raise Exception(msg)

    spectro = lc[0] is True

    # --------
    # is PHA
    # --------

    dPHA = {
        k0 : coll.dobj['camera'][k0]['dmat']['mode'] == 'PHA'
        for k0 in lcam
        if coll.dobj['camera'][k0].get('dmat') is not None
    }

    if len(dPHA) > 0:
        lc = [
            all([v0 for v0 in dPHA.values()]),
            all([not v0 for v0 in dPHA.values()]),
        ]

        if np.sum(lc) != 1:
            msg = (
                f"diag '{key}' must be either all PHA or all non-PHA!\n"
                + "\n".join([f"\t- {k0}: {v0}" for k0, v0 in dPHA.items()])
            )
            raise Exception(msg)

        PHA = lc[0]
    else:
        PHA = False

    return spectro, PHA


# ##################################################################
# ##################################################################
#                           check
# ##################################################################


def _diagnostics(
    coll=None,
    key=None,
    doptics=None,
    stack=None,
    **kwdargs,
):

    # ------------
    # check inputs

    (
        key, lcam, doptics, is2d, spectro, stack, PHA,
    ) = _diagnostics_check(
        coll=coll,
        key=key,
        doptics=doptics,
        stack=stack,
    )

    # --------
    # dobj

    dobj = {
        'diagnostic': {
            key: {
                'camera': lcam,
                'doptics': doptics,
                'ncam': len(doptics),
                # 'npix tot.': np.sum(),
                'is2d': is2d,
                'spectro': spectro,
                'PHA': PHA,
                'stack': stack,
                'signal': None,
            },
        },
    }

    # -----------
    # kwdargs

    if len(kwdargs) > 0:
        for k0, v0 in kwdargs.items():
            if not isinstance(k0, str):
                continue
            elif k0 in dobj['diagnostic'][key].keys():
                continue
            else:
                dobj['diagnostic'][key][k0] = v0

    return None, None, dobj


# ##################################################################
# ##################################################################
#                           get ref
# ##################################################################


def _get_default_cam(coll=None, key=None, key_cam=None, default=None):

    # ----------
    # key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    is2d = coll.dobj['diagnostic'][key]['is2d']
    # spectro = coll.dobj['diagnostic'][key]['spectro']

    # ---------------
    # default

    default = ds._generic_check._check_var(
        default, 'default',
        types=str,
        default='first' if is2d else 'all',
        allowed=['all', 'first'],
    )

    # -----------------------
    # key_cam (only 1 if is2d)

    lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
    if default == 'first':
        # 2d: can only select one camera at a time
        key_cam_def = [coll.dobj['diagnostic'][key]['camera'][0]]
    else:
        key_cam_def = None

    if isinstance(key_cam, str):
        key_cam = [key_cam]

    key_cam = ds._generic_check._check_var_iter(
        key_cam, 'key_cam',
        types=list,
        types_iter=str,
        allowed=lok,
        default=key_cam_def,
    )

    return key, key_cam


def get_ref(coll=None, key=None):

    # ---------
    # check key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # -----------
    # return

    cam = coll.dobj['diagnostic'][key]['optics'][0]
    return coll.dobj['camera'][cam]['dgeom']['ref']


# ################################################################
# ################################################################
#                      get optics
# ################################################################


def _get_optics_cls(coll=None, optics=None):

    # ---------
    # check key

    lcls = ['camera', 'aperture', 'filter', 'crystal', 'grating']

    if isinstance(optics, str):
        optics = [optics]

    lok = list(itt.chain.from_iterable([
        list(coll.dobj.get(cc, {}).keys())
        for cc in lcls
    ]))

    # check
    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # -----------
    # optics_cls

    derr = {}
    optics_cls = []
    for ii, oo in enumerate(optics):
        lc = [cc for cc in lcls if oo in coll.dobj.get(cc, {}).keys()]
        if len(lc) == 1:
            optics_cls.append(lc[0])
        else:
            derr[oo] = lc

    # --------
    # error

    if len(derr) > 0:
        msg = (
            "The following have no / several classes:\n"
            + "\n".join([f"\t- {k0}: {v0}" for k0, v0 in derr.items()])
        )
        raise Exception(msg)

    return optics, optics_cls


# def _get_diagnostic_doptics(coll=None, key=None):

#     # ---------
#     # check key

#     lok = list(coll.dobj.get('diagnostic', {}).keys())
#     key = ds._generic_check._check_var(
#         key, 'key',
#         types=str,
#         allowed=lok,
#     )

#     # ------------------
#     # optics and classes

#     doptics = {}
#     for k0, v0 in coll.dobj['diagnostic'][key]['doptics']:
#         doptics[k0]['camera'] = k0
#         doptics[k0]['optics'], doptics[k0]['cls'] = _get_optics_cls(
#             coll=coll,
#             optics=v0,
#         )

#     # -----------
#     # ispectro

#     if coll.dobj['diagnostic'][key]['spectro']:
#         for k0, v0 in doptics.items():
#             doptics[k0]['ispectro'] = [
#                 ii for ii, cc in enumerate(v0['cls'])
#                 if cc in ['grating', 'crystal']
#             ]

#     return doptics


# ##################################################################
# ##################################################################
#                           set color
# ##################################################################


def _set_optics_color(
    coll=None,
    key=None,
    color=None,
):

    # ------------
    # check inputs

    # key
    lk = ['aperture', 'filter', 'crystal', 'grating', 'camera', 'diagnostic']
    dk = {
        k0: list(coll.dobj.get(k0, {}).keys())
        for k0 in lk
    }
    lok = itt.chain.from_iterable([vv for vv in dk.values()])

    if isinstance(key, str):
        key = [key]

    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # color
    if color is None:
        color = 'k'

    if not mcolors.is_color_like(color):
        msg = (
            f"Arg color for '{key}' must be a matplotlib color!\n"
            f"Provided: {color}\n"
        )
        raise Exception(msg)

    color = mcolors.to_rgba(color)

    # ------------
    # set color

    for k0 in key:
        cls = [k1 for k1, v1 in dk.items() if k0 in v1][0]
        if cls == 'diagnostic':
            for k2 in coll._dobj[cls][k0]['optics']:
                cls2 = [k1 for k1, v1 in dk.items() if k2 in v1][0]
                coll._dobj[cls2][k2]['dmisc']['color'] = color
        else:
            coll._dobj[cls][k0]['dmisc']['color'] = color

    return


# ##################################################################
# ##################################################################
#                           remove
# ##################################################################


def _remove(
    coll=None,
    key=None,
    key_cam=None,
):

    # ------------
    # check inputs

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    doptics = coll.dobj['diagnostic'][key]['doptics']

    if key_cam is None:
        key_cam = coll.dobj['diagnostic'][key]['camera']

    else:
        if isinstance(key_cam, str):
            key_cam = [key_cam]

        key_cam = ds._generic_check._check_var_iter(
            key_cam, 'key_cam',
            types=list,
            types_iter=str,
            allowed=coll.dobj['diagnostic'][key]['camera'],
        )

    # ------------
    # list data

    lkd = ['etendue']
    ld = []
    for k0 in lkd:
        for k1 in key_cam:
            if doptics[k1][k0] is not None:
                ld.append(doptics[k1][k0])

    # -----------
    # remove data

    if len(ld) > 0:
        coll.remove_data(ld, propagate=True)

    # ----------
    # remove los

    for k1 in key_cam:
        if doptics[k1]['los'] is not None:
            coll.remove_rays(key=doptics[k1]['los'])

    # -----------
    # remove diag

    if len(key_cam) == len(doptics):
        del coll._dobj['diagnostic'][key]