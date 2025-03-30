# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:38:16 2024

@author: dvezinet
"""

import itertools as itt


import numpy as np
import datastock as ds


from . import _class08_get_data_def
from . import _class08_get_data_vos_broadband
from . import _class08_get_data_vos_spectro


# ##################################################################
# ##################################################################
#                   get data
# ##################################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
    # relevant for LOS data
    segment=None,
    # relevant for spectro data
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
        lok, lcam, lquant,
        davail,
    ) = _check(
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
    # extract lists
    # ----------------

    dav = {k0: list(v0['fields'].keys()) for k0, v0 in davail.items()}

    # ----------------
    # build ddata
    # ----------------

    # -----------
    # initialize

    ddata = {}
    dref = {}

    static = True
    daxis = None

    # -------------
    # data quantity

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
                dd = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                lc = [
                    isinstance(dd, str) and dd in coll.ddata.keys(),
                ]
                if lc[0]:
                    ddata[cc] = dd
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

    # ------------------------------------
    # standard LOS-related

    elif data in dav.get('standard - LOS', []):

        if data in ['length', 'tangency_radius', 'alpha']:
            for cc in key_cam:
                ddata[cc], _, dref[cc] = coll.get_rays_quantity(
                    key=key,
                    key_cam=cc,
                    quantity=data,
                    segment=segment,
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
                    dvect['nin_x'] * vectx[0, ...]
                    + dvect['nin_y'] * vecty[0, ...]
                    + dvect['nin_z'] * vectz[0, ...]
                )

                ddata[cc] = np.arccos(sca)
                dref[cc] = coll.dobj['camera'][cc]['dgeom']['ref']
                units = 'rad'

    # ------------------------------------
    # wavelength-related for spectro diags

    elif data in dav.get('spectro - lamb', []):

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

    # ------------------------------
    # data from synthetic diagnostic

    elif data in dav.get('synth', []):

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

    elif data in dav.get('raw', []):

        ddata = {key_cam[0]: coll.ddata[data]['data']}
        dref = {key_cam[0]: coll.dobj['camera'][key_cam[0]]['dgeom']['ref']}
        units = coll.ddata[data]['units']
        static = True

    # -----------------
    # vos broadband data

    elif data in dav.get('broadband - vos', []):

        ddata, dref, units, static = _class08_get_data_vos_broadband.main(
            coll=coll,
            key=key,
            key_cam=key_cam,
            data=data,
        )

    # -----------------
    # vos spectro data

    elif data in dav.get('spectro - vos', []):

        ddata, dref, units, static = _class08_get_data_vos_spectro.main(
            coll=coll,
            key=key,
            key_cam=key_cam,
            data=data,
        )

    return ddata, dref, units, static, daxis


# ##################################################################
# ##################################################################
#                   check input
# ##################################################################


def _check(
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
        _print(davail)

    # stop here if relevant
    if pdef is True:
        return [None]*10

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
    is_3d = is_vos and doptics[key_cam[0]]['dvos'].get('indr_3d') is not None

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

        _print(
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

        # -------------
        # overall check

        lok = list(itt.chain.from_iterable([
            list(v0['fields'].keys()) for v0 in davail.values()
        ]))

        try:
            data = ds._generic_check._check_var(
                data, 'data',
                types=str,
                allowed=lok,
            )
        except Exception as err:
            msg = (
                f"{err}\n"
                + _print(
                    davail,
                    key=key,
                    spectro=spectro,
                    is_vos=is_vos,
                    is_3d=is_3d,
                    returnas_str=True,
                )
            )
            raise Exception(msg)

    return (
        key, key_cam,
        spectro, is_vos, is_3d,
        data,
        lok, lcam, lquant,
        davail,
    )


# ##################################################################
# ##################################################################
#                   get data doc printing
# ##################################################################


def _print(
    davail,
    key=None,
    spectro=None,
    is_vos=None,
    is_3d=None,
    returnas_str=None,
):

    # -----------------
    # check inputs
    # -----------------

    returnas_str = ds._generic_check._check_var(
        returnas_str, 'returnas_str',
        types=bool,
        default=False,
    )

    # -----------------
    # initialize
    # -----------------

    msg = "\n\n##############################################\n"
    if key is None:
        msg += "The following built-in data is available in general:\n\n"
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

        # get max length
        ll = [len(k1) for k1 in v0['fields'].keys()]
        nmax = np.max(ll) + 1

        # list of data
        for k1, v1 in v0['fields'].items():
            msg += f"\t- {k1.ljust(nmax)}:  {v1['doc']}\n"

    # -----------------
    # print
    # -----------------

    if returnas_str is True:
        return msg
    else:
        print(msg)
        return
