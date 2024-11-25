# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:55:54 2024

@author: dvezinet
"""


import os
import itertools as itt


import numpy as np
import datastock as ds


from . import _class08_save2json
from . import _class08_loadfromjson
from . import _class08_save2stp


# #################################################################
# #################################################################
#          Default values
# #################################################################


_NAME = 'test'


# #################################################################
# #################################################################
#          save
# #################################################################


def save(
    # ---------------
    # input from tofu
    coll=None,
    key=None,
    key_cam=None,
    key_optics=None,
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
    #   check
    # ----------------

    (
        key, key_cam,
        factor, color,
        pfe_save, overwrite,
    ) = _check_save(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # options
        factor=factor,
        color=color,
        # saving
        pfe_save=pfe_save,
        overwrite=overwrite,
        ext=None,
    )

    # ----------------
    #   call
    # ----------------

    if pfe_save.endswith('.json'):
        _class08_save2json.main(
            coll=coll,
            key=key,
            key_cam=key_cam,
            # options
            factor=factor,
            color=color,
            # saving
            pfe_save=pfe_save,
            overwrite=overwrite,
        )

    elif pfe_save.endswith('.stp'):
        _class08_save2stp.main(
            # input from tofu
            coll=coll,
            key=key,
            key_cam=key_cam,
            key_optics=key_optics,
            # options
            factor=factor,
            color=color,
            empty_name=empty_name,
            # saving
            pfe_save=pfe_save,
            overwrite=overwrite,
        )

    return


# #################################################################
# #################################################################
#          load
# #################################################################


def load(
    pfe=None,
    coll=None,
    returnas=None,
):

    # ----------------
    #   check
    # ----------------

    coll, pfe, fname = _check_load(
        coll=coll,
        pfe=pfe,
    )

    # ----------------
    #   call
    # ----------------

    if pfe.endswith('.json'):
        return _class08_loadfromjson.main(
            coll=coll,
            pfe=pfe,
            fname=fname,
            returnas=returnas,
        )


# #################################################################
# #################################################################
#          check - save
# #################################################################


def _check_save(
    coll=None,
    key=None,
    key_cam=None,
    key_optics=None,
    # options
    factor=None,
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
    ext='json',
):

    # ---------------
    # ext
    # ---------------

    lok = ['json', 'stp']
    ext = ds._generic_check._check_var(
        ext, 'ext',
        types=str,
        default='json',
        allowed=lok,
    )

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
    ext_file = pfe_save.split('.')[-1]
    if ext_file not in lok:
        pfe_save = f"{pfe_save}.{ext}"

    # ---------------
    # factor
    # ---------------

    factor = float(ds._generic_check._check_var(
        factor, 'factor',
        types=(float, int),
        default=1.,
    ))

    # ----------------
    # overwrite
    # ----------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # ---------------
    # key
    # ---------------

    if ext == 'json':
        key, key_cam = coll.get_diagnostic_cam(
            key=key,
            key_cam=key_cam,
            default='all',
        )

    return (
        key, key_cam,
        factor, color,
        pfe_save, overwrite,
    )


# #################################################################
# #################################################################
#          check
# #################################################################


def _check_load(
    coll=None,
    pfe=None,
):

    # ---------------
    # coll
    # ---------------

    from ._class10_Inversion import Inversion as Collection

    if coll is None:
        coll = Collection()

    else:
        c0 = isinstance(coll, Collection)
        if not c0:
            msg = (
                "Arg coll must be a tf.data.Collection instance!\n"
                f"\t Provided:\n{coll}\n"
            )
            raise Exception(msg)

    # ---------------
    # pfe
    # ---------------

    c0 = (
        isinstance(pfe, str)
        and os.path.isfile(pfe)
        and pfe.endswith('.json')
    )
    if not c0:
        msg = (
            "Arg 'pfe' must be a 'path/file.ext' to an existing json file!\n"
            f"\t- Provided: {pfe}\n"
        )
        raise Exception(msg)

    # -------------
    # fname
    # -------------

    fname = os.path.split(pfe)[-1][:-4]

    return coll, pfe, fname