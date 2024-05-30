# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:38:16 2024

@author: dvezinet
"""

import itertools as itt


import numpy as np
import datastock as ds


from . import _class08_get_data_def


# ##################################################################
# ##################################################################
#                   get data
# ##################################################################


def _get_data(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
    rocking_curve=None,
    units=None,
    default=None,
    print_full_doc=None,
    **kwdargs,
):

    # ----------------
    # check inputs
    # ----------------

    (
     key, key_cam,
     spectro, is_vos, is_3d,
     data,
     lok, lcam, lquant, llamb, lcomp, lsynth, lraw, lvos,
     davail,
    ) = _get_data_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        data=data,
        rocking_curve=rocking_curve,
        units=units,
        default=default,
        print_full_doc=print_full_doc,
        **kwdargs,
    )

    # print_full_doc
    if key is None:
        return

    # print diag-specific
    if data is None and len(kwdargs) == 0:
        return

    # ----------------
    # build ddata
    # ----------------

    # -----------
    # initialize

    ddata = {}
    static = True
    daxis = None

    # comp = False
    if data is None or data in lquant:

        # --------------------------
        # data is None => kwdargs

        if data is None:

            # reorder
            ddata = {
                cc: lok[lcam.index(cc)]
                for cc in key_cam if cc in lcam
            }

        # -----------------
        # data in lquant

        elif data in lquant:
            for cc in key_cam:
                # if data == 'los':
                #     kr = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                #     dd = coll.dobj['rays'][kr]['pts']
                # else:
                dd = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                lc = [
                    isinstance(dd, str) and dd in coll.ddata.keys(),
                    # isinstance(dd, tuple)
                    # and all([isinstance(di, str) for di in dd])
                    # and all([di in coll.ddata.keys() for di in dd])
                ]
                if lc[0]:
                    ddata[cc] = dd
                # elif lc[1]:
                #     ddata[cc] = list(dd)
                elif dd is None:
                    pass
                else:
                    msg = f"Unknown data: '{data}'"
                    raise Exception(msg)

        # dref
        dref = {
            k0: coll.ddata[v0]['ref']
            for k0, v0 in ddata.items()
        }

        # units
        if len(ddata) > 0:
            units = coll.ddata[ddata[key_cam[0]]]['units']
        else:
            units = None

        # get actual data
        ddata = {
            k0 : coll.ddata[v0]['data']
            for k0, v0 in ddata.items()
        }

    # --------------------
    # data to be computed

    elif data in lcomp:

        # comp = True
        ddata = {}
        dref = {}

        if data in llamb:
            for cc in key_cam:
               ddata[cc], dref[cc] = coll.get_diagnostic_lamb(
                   key=key,
                   key_cam=cc,
                   rocking_curve=rocking_curve,
                   lamb=data,
                   units=units,
               )
            if data in ['lamb', 'lambmin', 'lambmax', 'dlamb']:
                units = 'm'
            else:
                units = ''

        elif data in ['length', 'tangency_radius', 'alpha']:
            for cc in key_cam:
                ddata[cc], _, dref[cc] = coll.get_rays_quantity(
                    key=key,
                    key_cam=cc,
                    quantity=data,
                    segment=-1,
                    lim_to_segments=False,
                )
            if data in ['length', 'tangency_radius']:
                units = 'm'
            else:
                units = 'rad'

        elif data == 'alpha_pixel':
            for cc in key_cam:

                klos = coll.dobj['diagnostic'][key]['doptics'][cc]['los']
                vectx, vecty, vectz = coll.get_rays_vect(klos)
                dvect = coll.get_camera_unit_vectors(cc)
                sca = (
                    dvect['nin_x'] * vectx
                    + dvect['nin_y'] * vecty
                    + dvect['nin_z'] * vectz
                )

                ddata[cc] = np.arccos(sca)
                dref[cc] = coll.dobj['camera'][cc]['dgeom']['ref']
                units = 'rad'

    # ------------------------------
    # data from synthetic diagnostic

    elif data in lsynth:

        dref = {}
        daxis = {}
        dsynth = coll.dobj['synth sig'][data]
        for cc in key_cam:
            kdat = dsynth['data'][dsynth['camera'].index(cc)]
            refcam = coll.dobj['camera'][cc]['dgeom']['ref']
            ref = coll.ddata[kdat]['ref']

            c0 = (
                tuple([rr for rr in ref if rr in refcam]) == refcam
                and len(ref) in [len(refcam), len(refcam) + 1]
            )
            if not c0:
                msg = (
                    "Can only plot data that is either:\n"
                    "\t- static: same refs as the camera\n"
                    "\t- has a unique extra dimension\n"
                    "Provided:\n"
                    "\t- refcam: {refcam}\n"
                    "\t- ['{kdat}']['ref']: {ref}"
                )
                raise Exception(msg)

            if len(ref) == len(refcam) + 1:
                static = False
                daxis[cc] = [
                    ii for ii, rr in enumerate(ref) if rr not in refcam
                ][0]

            ddata[cc] = coll.ddata[kdat]['data']
            dref[cc] = ref

            units = coll.ddata[kdat]['units']

    # -----------------
    # raw data

    elif data in lraw:
        ddata = {key_cam[0]: coll.ddata[data]['data']}
        dref = {key_cam[0]: coll.dobj['camera'][key_cam[0]]['dgeom']['ref']}
        units = coll.ddata[data]['units']
        static = True

    # -----------------
    # vos data

    elif data in lvos:

        static = True
        ddata, dref = {}, {}
        doptics = coll.dobj['diagnostic'][key]['doptics']


        # sample cross if needed
        if data.startswith('vos_cross_'):

            # check
            pass









        for cc in key_cam:

            # safety check
            ref = coll.dobj['camera'][cc]['dgeom']['ref']
            dvos = doptics[cc].get('dvos')
            if dvos is None:
                msg = (
                    f"Data '{data}' cannot be retrived for diag '{key}' "
                    "cam '{cc}' because no dvos computed"
                )
                raise Exception(msg)

            # cases
            if data == 'vos_sang_integ':
                kdata = dvos['sang_cross']
                ddata[cc] = np.nansum(coll.ddata[kdata]['data'], axis=-1)
                dref[cc] = ref
                units = coll.ddata[kdata]['units']

            elif data in ['vos_lamb', 'vos_dlamb', 'vos_ph_integ']:
                kph = dvos['ph']
                ph = coll.ddata[kph]['data']
                ph_tot = np.sum(ph, axis=(-1, -2))

                if data == 'vos_ph_integ':
                    out = ph_tot
                    kout = kph
                else:
                    kout = dvos['lamb']
                    re_lamb = [1 for rr in ref] + [1, -1]
                    lamb = coll.ddata[kout]['data'].reshape(re_lamb)

                    i0 = ph == 0
                    if data == 'vos_lamb':
                        out = np.sum(ph * lamb, axis=(-1, -2)) / ph_tot
                    else:
                        for ii, i1 in enumerate(re_lamb[:-1]):
                            lamb = np.repeat(lamb, ph.shape[ii], axis=ii)

                        lamb[i0] = -np.inf
                        lambmax = np.max(lamb, axis=(-1, -2))
                        lamb[i0] = np.inf
                        lambmin = np.min(lamb, axis=(-1, -2))
                        out = lambmax - lambmin
                    out[np.all(i0, axis=(-1, -2))] = np.nan

                ddata[cc] = out
                dref[cc] = ref
                units = coll.ddata[kout]['units']

            elif data == 'vos_cross_sang':

                # -----------------
                # get mesh sampling

                dsamp = coll.get_sample_mesh(
                    key=dvos['keym'],
                    res=dvos['res_RZ'],
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

                # -----------------
                # prepare image

                n0, n1 = dsamp['x0']['data'].size, dsamp['x1']['data'].size
                shape = (n0, n1)
                sang = np.full(shape, np.nan)
                sang_tot = np.full(shape, 0.)



            elif data == 'vos_cross_rz':

                # get indices
                # kindr, kindz = v0['dvos']['ind_cross']
                pass

            elif data == 'vos_vect_cross_ang':

                # solid angle map
                ksang = 'sang_cross'
                ref = coll.ddata[kph]['ref']
                iok = np.isfinite(coll.ddata[kph]['data'])

                # pixel centers
                # cr =
                # cz =

                # vect_cross
                ang = np.full(iok.shape, np.nan)
                sli = [None for ss in iok.shape]
                sli[-1] = slice(None)
                sli = tuple(sli)

                # ang[iok] = np.arctan2(
                #    R[sli] - cz[..., None],
                #    Z[sli] - cr[..., None],
                # )

                ddata[cc] = ang
                dref[cc] = ref
                units = coll.ddata[kdata]['units']

    return ddata, dref, units, static, daxis


# ##################################################################
# ##################################################################
#                   check input
# ##################################################################


def _get_data_check(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
    rocking_curve=None,
    units=None,
    default=None,
    print_full_doc=None,
    **kwdargs,
):

    # --------------
    # print full doc
    # --------------

    pdef = (
        len(coll.dobj.get('diagnostic')) == 0
        or ((len(coll.dobj.get('diagnostic')) > 1) and key is None)
    )
    print_full_doc = ds._generic_check._check_var(
        print_full_doc, 'print_full_doc',
        types=bool,
        default=pdef,
        allowed=[True] if pdef else [True, False]
    )

    # execute
    if print_full_doc is True:
        davail = _class08_get_data_def.get_davail()
        _get_data_print(davail)

    # stop here if relevant
    if pdef is True:
        return [None]*14

    # --------------
    # key, key_cam
    # --------------

    # -------------
    # key, key_cam

    key, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
        default=default,
    )

    # -----------------------
    # spectro, is_vos, is_3d
    # -----------------------

    # -------------
    # spectro

    spectro = coll.dobj['diagnostic'][key]['spectro']

    # -------------
    # is_vos, is_3d

    doptics = coll.dobj['diagnostic'][key]['doptics']
    is_vos = doptics[key_cam[0]].get('dvos') is not None
    is_3d = is_vos and doptics[key_cam[0]]['dvos']['indr_3d'] is not None

    # ---------------------------
    # get dict of available data
    # ---------------------------

    davail = _class08_get_data_def.get_davail(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # conditions
        spectro=spectro,
        is_vos=is_vos,
        is_3d=is_3d,
    )

    # -----------------------
    # initialize output
    # -----------------------

    lok, lcam, lquant = None, None, None
    llamb, lcomp, lsynth = None, None, None
    lraw, lvos = None, None

    # -----------------------
    # preliminary
    # -----------------------

    lc = [data is None, len(kwdargs) == 0]
    if np.sum(lc) < 1:
        msg = (
            "Please provide data xor kwdargs, not both!\n"
            "providing none of them will print the doc\n"
        )
        raise Exception(msg)

    # -----------------------
    # basic checks on data
    # -----------------------

    if all(lc) and print_full_doc is False:

        _get_data_print(
            davail,
            key=key,
            spectro=spectro,
            is_vos=is_vos,
            is_3d=is_3d,
        )

    elif data is None:

        # -------------
        # check kwdargs

        dparam = coll.get_param(which='data', returnas=dict)
        lkout = [k0 for k0 in kwdargs.keys() if k0 not in dparam.keys()]

        if len(lkout) > 0:
            msg = (
                "The following args correspond to no data parameter:\n"
                + "\n".join([f"\t- {k0}" for k0 in lkout])
            )
            raise Exception(msg)

        # -----------------------
        # list all available data

        lok = [
            k0 for k0, v0 in coll.ddata.items()
            if v0.get('camera') in key_cam
        ]

        # -------------------
        # Adjust with kwdargs

        if len(kwdargs) > 0:
            lok2 = coll.select(
                which='data', log='all', returnas=str, **kwdargs,
            )
            lok = [k0 for k0 in lok2 if k0 in lok]

        # -----------------------------
        # check there is 1 data per cam

        lcam = [
            coll.ddata[k0]['camera'] for k0 in lok
            if coll.ddata[k0]['camera'] in key_cam
        ]

        if len(set(lcam)) > len(key_cam):
            msg = (
                "There are more / less data identified than cameras:\n"
                f"\t- key_cam:  {key_cam}\n"
                f"\t- data cam: {lcam}\n"
                f"\t- data: {data}"
            )
            raise Exception(msg)

        elif len(set(lcam)) < len(key_cam):
            pass

    else:

        # ----------------
        # allowable values

        lquant = ['etendue', 'amin', 'amax']  # 'los'
        lcomp = ['length', 'tangency radius', 'alpha', 'alpha_pixel']

        # --------------------------
        # spectro and vos - specific

        if spectro:
            llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
            lvos = ['vos_lamb', 'vos_dlamb', 'vos_ph_integ']
        else:
            llamb = []
            lvos = ['vos_cross_rz', 'vos_sang_integ', 'vos_vect_cross']

        # -----------------
        # synthetic signals

        lsynth = coll.dobj['diagnostic'][key]['signal']

        #
        if len(key_cam) == 1:
            lraw = [
                k0 for k0, v0 in coll.ddata.items()
                if v0['ref'] == coll.dobj['camera'][key_cam[0]]['dgeom']['ref']
            ]
        else:
            lraw = []

        if lsynth is None:
            lsynth = []
        lcomp += llamb

        # -------------
        # overall check

        data = ds._generic_check._check_var(
            data, 'data',
            types=str,
            allowed=lquant + lcomp + lsynth + lraw + lvos,
        )

        # -------------------
        # post-check - refine

        if data in lvos and is_vos is False:
            msg = (
                "The following data is not available (dvos not computed):\n"
                f"\t- key: {key}\n"
                f"\t- data: {data}\n"
            )
            raise Exception(msg)

    return (
     key, key_cam,
     spectro, is_vos, is_3d,
     data,
     lok, lcam, lquant, llamb, lcomp, lsynth, lraw, lvos,
     davail,
    )


# ##################################################################
# ##################################################################
#                   get data doc printing
# ##################################################################


def _get_data_print(davail, key=None, spectro=None, is_vos=None, is_3d=None):

    # -----------------
    # initialize
    # -----------------

    msg = "\n\n##############################################\n"
    if key is None:
        msg += "The following data is available in general:\n\n"
    else:
        msg += (
            f"The following data is available for '{key}':\n"
            f"- spectro: {spectro}\n"
            f"- is_vos: {is_vos}\n"
            f"- is_3d: {is_3d}\n\n"
        )

    # -----------------
    # loop on davail
    # -----------------

    for k0, v0 in davail.items():

        # header
        msg += f"\n{k0}\n------------------\n"

        # list of data
        for k1, v1 in v0['fields'].items():
            msg += f"\t- {k1.ljust(20)}: \t{v1['doc']}\n"

    # -----------------
    # print
    # -----------------

    print(msg)
    return