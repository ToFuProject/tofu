# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:32:58 2024

@author: dvezinet
"""


import os
import itertools as itt
import json
import warnings


import numpy as np
import datastock as ds


# #################################################################
# #################################################################
#          Default values
# #################################################################


_NAME = 'test'


# #################################################################
# #################################################################
#          Main
# #################################################################


def main(
    # ---------------
    # input from tofu
    coll=None,
    key=None,
    key_cam=None,
    # -------------
    # options
    factor=None,
    color=None,
    # -------------
    # saving
    pfe_save=None,
    overwrite=None,
):


    # ----------------
    # check inputs
    # --------------

    fname = os.path.split(pfe_save)[-1][:-4]

    # ----------------
    # get file content
    # ----------------

    # ---------------------
    # Header (not used yet)

    _ = _get_header(
        fname=fname,
    )

    # -------------
    # Diagnostic

    # initialze
    dout = {}

    # fill
    dout, dcls = _extract_diagnostic(
        coll=coll,
        key=key,
        key_cam=key_cam,
    )

    # ---------------
    # list of classes

    for kcls, lkeys in dcls.items():
        dout[kcls] = _DFUNC[kcls](
            coll=coll,
            keys=lkeys,
        )

    # ---------------------
    # json-ify numpy arrays
    # ---------------------

    _jsonify(dout)

    # -------------
    # save to stp
    # -------------

    _save(
        dout=dout,
        pfe_save=pfe_save,
        overwrite=overwrite,
    )

    return


# #################################################################
# #################################################################
#          HEADER
# #################################################################


def _get_header(fname=None):

    return


# #################################################################
# #################################################################
#          extract - diagnostic
# #################################################################


def _extract_diagnostic(
    coll=None,
    key=None,
    key_cam=None,
    excluded=None,
):

    # ----------------------
    # check inputs
    # ----------------------

    exdef = [
        'doptics',
        'signal',
        'nb geom matrix'
    ]
    if isinstance(excluded, str):
        excluded = [excluded]
    excluded = ds._generic_check._check_var_iter(
        excluded, 'excluded',
        default=exdef,
        types=list,
        types_iter=str,
    )

    # ----------------------
    # initialize
    # ----------------------

    # extract
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # dcls
    lcls = list(itt.chain.from_iterable([v1['cls'] for v1 in doptics.values()]))
    loptics = list(itt.chain.from_iterable([v1['optics'] for v1 in doptics.values()]))

    # dcls
    dcls = {
        k0: [k1 for ii, k1 in enumerate(loptics) if lcls[ii] == k0]
        for k0 in set(lcls)
    }
    dcls['camera'] = lcam

    # dout
    dout = {k0: {} for k0 in set(lcls)}
    dout['diagnostic'] = {'key': key}

    # ----------------------
    # Fill diagnostic with simple values
    # ----------------------

    # initialze with simple values
    dout['diagnostic'].update({
        k0: v0
        for k0, v0 in coll.dobj['diagnostic'][key].items()
        if not (
                isinstance(v0, dict)
                or k0 in excluded
            )
    })

    # ----------------------
    # prepare doptics extraction
    # ----------------------

    # prepare doptics extraction
    dout['diagnostic']['doptics'] = {}
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']

    ldirect = [
        'vos', 'dvos',
        'etendue',
        'los', 'amin', 'amax',
    ]

    # ----------------------
    # extract points
    # ----------------------

    for i0, kcam in enumerate(lcam):

        # -----------------------------
        # initialize with simple values


        dout['diagnostic']['doptics'][kcam] = {
            k1: v1 for k1, v1 in doptics[kcam].items()
            if k1 not in ldirect
        }

        # --------
        # data

        for k0 in ['etendue', 'amin', 'amax']:

            kd = doptics[kcam].get(k0)
            if kd is None:
                continue
            dout['diagnostic']['doptics'][kcam][k0] = {
                'data': coll.ddata[kd]['data'],
                'units': coll.ddata[kd]['units'],
            }

        # ------------
        # etendue key

        dout['diagnostic']['doptics'][kcam]['etendue_key'] = doptics[kcam]['etendue']

        # ---------
        # los

        klos = doptics[kcam].get('los')
        if klos is not None:
            ptsx, ptsy, ptsz = coll.get_rays_pts(klos)

            klos = doptics[kcam]['los']

            dout['diagnostic']['doptics'][kcam].update({
                'los_x_start': {
                    'data': ptsx[0, ...],
                    'units': 'm',
                },
                'los_y_start': {
                    'data': ptsy[0, ...],
                    'units': 'm',
                },
                'los_z_start': {
                    'data': ptsz[0, ...],
                    'units': 'm',
                },
                'los_x_end': {
                    'data': ptsx[1:, ...],
                    'units': 'm',
                },
                'los_y_end': {
                    'data': ptsy[1:, ...],
                    'units': 'm',
                },
                'los_z_end': {
                    'data': ptsz[1:, ...],
                    'units': 'm',
                },
                'los_key': klos,
                'los_alpha': {
                    'data': coll.ddata[coll.dobj['rays'][klos]['alpha']]['data'],
                    'units': 'rad'
                },
                'los_dalpha': {
                    'data': coll.ddata[coll.dobj['rays'][klos]['reflect_dalpha']]['data'],
                    'units': 'rad'
                },
                'los_dbeta': {
                    'data': coll.ddata[coll.dobj['rays'][klos]['reflect_dbeta']]['data'],
                    'units': 'rad'
                },
            })

        # ------------
        # vos - pcross

        if doptics[kcam].get('dvos', {}).get('pcross') is not None:
            pc0, pc1 = doptics[kcam]['dvos']['pcross']
            dout['diagnostic']['doptics'][kcam].update({
                'pcross_x0': {
                    'key': pc0,
                    'data': coll.ddata[pc0]['data'],
                    'units': coll.ddata[pc0]['units'],
                    'ref': coll.ddata[pc0]['ref'],
                },
                'pcross_x1': {
                    'key': pc1,
                    'data': coll.ddata[pc1]['data'],
                    'units': coll.ddata[pc1]['units'],
                    'ref': coll.ddata[pc1]['ref'],
                },
            })

        # ------------
        # vos - phor

        if doptics[kcam].get('dvos', {}).get('phor') is not None:
            ph0, ph1 = doptics[kcam]['dvos']['phor']
            dout['diagnostic']['doptics'][kcam].update({
                'phor_x0': {
                    'key': ph0,
                    'data': coll.ddata[ph0]['data'],
                    'units': coll.ddata[ph0]['units'],
                    'ref': coll.ddata[ph0]['ref'],
                },
                'phor_x1': {
                    'key': ph1,
                    'data': coll.ddata[ph1]['data'],
                    'units': coll.ddata[ph1]['units'],
                    'ref': coll.ddata[ph1]['ref'],
                },
            })

        # ------------
        # vos - dphi

        if doptics[kcam].get('dvos', {}).get('dphi') is not None:
            dout['diagnostic']['doptics'][kcam].update({
                'dphi': {
                    'data': doptics[kcam]['dvos']['dphi'],
                    'units': 'rad',
                },
            })

    return dout, dcls


# #################################################################
# #################################################################
#          extract - dgeom - dmisc
# #################################################################


def _extract_dgeom_dmisc(
    coll=None,
    which=None,
    keys=None,
    lok_geom=None,
    lok_misc=None,
):

    # ----------------------
    # check inputs
    # ----------------------

    if lok_geom is None:
        lok_geom=[
            'type', 'nd', 'parallel', 'shape',
            'poly', 'cents', 'outline',
            'curve_r',
            'cent', 'nin', 'e0', 'e1',
        ]

    if lok_misc is None:
        lok_misc = ['color']

    # ----------------------
    # extract points
    # ----------------------

    dout = {}
    for i0, key in enumerate(keys):

        # -----------
        # initialize

        dgeom = coll.dobj[which][key]['dgeom']

        dout[key] = {
            'key': key,
            'dgeom': {},
            'dmisc': {},
        }

        for i1, (k1, v1) in enumerate(dgeom.items()):

            if v1 is None:
                continue

            # -----------------------------
            # deal with simple cases

            if k1 in lok_geom:

                if isinstance(v1, (str, bool)):
                    dout[key]['dgeom'][k1] = v1

                elif isinstance(v1, np.ndarray):
                    dout[key]['dgeom'][k1] = {
                        'data': v1,
                        'units': 'm' if k1 == 'cent' else None,
                    }

                elif k1 == 'curve_r':
                    dout[key]['dgeom'][k1] = {
                        'data': v1,
                        'units': 'm'
                    }

                else:
                    c0 = (
                        isinstance(v1, tuple)
                        and all([isinstance(v2, str) for v2 in v1])

                    )
                    if c0 and len(v1) == 3:
                        kx, ky, kz = v1
                        dout[key]['dgeom'].update(
                            {
                                f'{k1}_x': {
                                    'data': coll.ddata[kx]['data'],
                                    'units': coll.ddata[kx]['units'],
                                },
                                f'{k1}_y': {
                                    'data': coll.ddata[ky]['data'],
                                    'units': coll.ddata[ky]['units'],
                                },
                                f'{k1}_z': {
                                    'data': coll.ddata[kz]['data'],
                                    'units': coll.ddata[kz]['units'],
                                },
                            }
                        )

                    elif c0 and len(v1) == 2:
                        kx0, kx1 = v1
                        dout[key]['dgeom'].update(
                            {
                                f'{k1}_x0': {
                                    'data': coll.ddata[kx0]['data'],
                                    'units': coll.ddata[kx0]['units'],
                                },
                                f'{k1}_x1': {
                                    'data': coll.ddata[kx1]['data'],
                                    'units': coll.ddata[kx1]['units'],
                                },
                            }
                        )


            # -------------------
            # more complex cases

            if k1 == 'extenthalf':
                if dgeom.get('curve_r') is not None:
                    units = [
                        'm' if np.isinf(dgeom['curve_r'][ii]) else 'rad'
                        for ii in range(len(v1))
                    ]
                else:
                    units = 'm'

                dout[key]['dgeom'][k1] = {
                    'data': v1,
                    'units': tuple(units),
                }

        # -------------
        # dmisc

        dmisc = coll.dobj[which][key]['dmisc']

        for i1, (k1, v1) in enumerate(dmisc.items()):

            if v1 is None:
                continue

            if k1 in lok_misc:
                dout[key]['dmisc'][k1] = v1


    return dout



# #################################################################
# #################################################################
#          extract - camera
# #################################################################


def _extract_camera(
    coll=None,
    keys=None,
):

    # ----------------------
    # dgeom, dmisc
    # ----------------------

    dout = _extract_dgeom_dmisc(
        coll=coll,
        which='camera',
        keys=keys,
    )

    return dout


# #################################################################
# #################################################################
#          extract - aperture
# #################################################################


def _extract_aperture(
    coll=None,
    keys=None,
):

    # ----------------------
    # dgeom, dmisc
    # ----------------------

    dout = _extract_dgeom_dmisc(
        coll=coll,
        which='aperture',
        keys=keys,
    )

    return dout

# #################################################################
# #################################################################
#          extract - filter
# #################################################################


def _extract_filter(
    coll=None,
    keys=None,
):

    # ----------------------
    # dgeom, dmisc
    # ----------------------

    dout = _extract_dgeom_dmisc(
        coll=coll,
        which='filter',
        keys=keys,
    )

    return dout


# #################################################################
# #################################################################
#          extract - crystal
# #################################################################


def _extract_crystal(
    coll=None,
    keys=None,
):

    # ----------------------
    # dgeom, dmisc
    # ----------------------

    which = 'crystal'
    dout = _extract_dgeom_dmisc(
        coll=coll,
        which=which,
        keys=keys,
    )

    # --------------
    # extract dmat
    # --------------

    lok = [
        'material', 'name', 'symbol', 'miller', 'd_hkl', 'target',
        'mesh',
    ]

    for key in keys:

        # extract dmat
        dmat = coll.dobj[which][key]['dmat']

        # populate
        dout[key]['dmat'] = {
            k0: dmat[k0] for k0 in lok
        }

    return dout


# #################################################################
# #################################################################
#          json-ify
# #################################################################


def _jsonify(dout):

    for k0, v0 in dout.items():

        if isinstance(v0, np.ndarray):
            dout[k0] = v0.tolist()

        elif k0 == 'units':
            dout[k0] = str(v0)

        elif isinstance(v0, dict):
            _jsonify(dout[k0])

    return


# #################################################################
# #################################################################
#          save to json
# #################################################################


def _save(
    dout=None,
    pfe_save=None,
    overwrite=None,
):

    # -------------
    # check before overwriting

    if os.path.isfile(pfe_save):
        err = "File already exists!"
        if overwrite is True:
            err = f"{err} => overwriting"
            warnings.warn(err)
        else:
            err = f"{err}\nFile:\n\t{pfe_save}"
            raise Exception(err)

    # ----------
    # save

    with open(pfe_save, 'w') as fn:
        json.dump(dout, fn, indent=4)

    # --------------
    # verb

    msg = f"Saved to:\n\t{pfe_save}"
    print(msg)

    return


# #################################################################
# #################################################################
#          DICT of FUNCTIONS
# #################################################################


_DFUNC = {
    'camera': _extract_camera,
    'aperture': _extract_aperture,
    'filter': _extract_filter,
    'crystal': _extract_crystal,
    # 'grating': _extract_grating,
}