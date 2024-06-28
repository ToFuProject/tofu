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


from . import _class02_save2stp


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
    # ---------------
    # options
    factor=None,
    color=None,
    empty_name=None,
    # ---------------
    # saving
    pfe_save=None,
    overwrite=None,
):


    # ----------------
    # check inputs
    # --------------

    (
        key, key_cam,
        lcls,
        factor, color,
        pfe_save, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # options
        factor=factor,
        color=color,
        # saving
        pfe_save=pfe_save,
        overwrite=overwrite,
        ext='json',
    )

    fname = os.path.split(pfe_save)[-1][:-4]

    # ----------------
    # get file content
    # ----------------

    # -----------
    # Header

    msg_header = _get_header(
        fname=fname,
    )

    # -------------
    # Diagnostic

    # initialze
    dout = {}

    # fill
    dout['diagnostic'] = _extract_diagnostic(
        coll=coll,
        key=key,
        key_cam=key_cam,
    )

    # ------------
    # list of classes

    for kcls, lkeys in dcls.items():
        dout[kcls] = _DFUNC[kcls](
            coll=coll,
            keys=lkeys,
        )

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
#          check
# #################################################################


def _check(
    coll=None,
    key=None,
    key_cam=None,
    # options
    factor=None,
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
    ext='stp',
):


    # ---------------
    # key
    # ---------------

    key, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
        default='all',
    )

    # ---------------
    # factor
    # ---------------

    factor = float(ds._generic_check._check_var(
        factor, 'factor',
        types=(float, int),
        default=1.,
    ))

    # ---------------
    # pfe_save
    # ---------------

    # Default
    if pfe_save is None:
        path = os.path.abspath('.')
        name = key if key is not None else _NAME
        pfe_save = os.path.join(path, f"{name}.{ext}")

    # check
    c0 = (
        isinstance(pfe_save, str)
        and (
            os.path.split(pfe_save)[0] == ''
            or os.path.isdir(os.path.split(pfe_save)[0])
        )
    )
    if not c0:
        msg = (
            f"Arg pfe_save must be a saving file str ending in '.{ext}'!\n"
            f"Provided: {pfe_save}"
        )
        raise Exception(msg)

    # makesure extension is included
    if not pfe_save.endswith(f'.{ext}'):
        pfe_save = f"{pfe_save}.{ext}"

    # ----------------
    # overwrite
    # ----------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return (
        key, key_cam,
        factor, color,
        pfe_save, overwrite,
    )


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

    # initialze with simple values
    dout = {
        k0: v0
        for k0, v0 in coll.dobj['diagnostic'][key].items()
        if not (
                isinstance(v0, dict)
                or k0 in excluded
            )
    }

    # prepare doptics extraction
    dout['doptics'] = {}
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ----------------------
    # extract points
    # ----------------------

    for i0, kcam in enumerate(lcam):

        # -----------------------------
        # initialize with simple values

        dout['doptics'][kcam] = {

        }



    return dout


# #################################################################
# #################################################################
#          extract - camera
# #################################################################


def _extract_camera(
    coll=None,
    keys=None,
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

    # initialze with simple values
    dout = {
        k0: v0
        for k0, v0 in coll.dobj['camera'][key].items()
        if not (
                isinstance(v0, dict)
                or k0 in excluded
            )
    }

    # prepare doptics extraction
    dout['doptics'] = {}
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ----------------------
    # extract points
    # ----------------------

    for i0, kcam in enumerate(lcam):

        # -----------------------------
        # initialize with simple values

        dout['doptics'][kcam] = {

        }



    return dout


# #################################################################
# #################################################################
#          extract - aperture
# #################################################################


def _extract_aperture(
    coll=None,
    keys=None,
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

    # initialze with simple values
    dout = {
        k0: v0
        for k0, v0 in coll.dobj['camera'][key].items()
        if not (
                isinstance(v0, dict)
                or k0 in excluded
            )
    }

    # prepare doptics extraction
    dout['doptics'] = {}
    lcam = coll.dobj['diagnostic'][key]['camera']
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ----------------------
    # extract points
    # ----------------------

    for i0, kcam in enumerate(lcam):

        # -----------------------------
        # initialize with simple values

        dout['doptics'][kcam] = {

        }



    return dout


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
        json.dump(dout, fn)

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
    'grating': _extract_grating,
}