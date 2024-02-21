# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:32:58 2024

@author: dvezinet
"""


import os
import getpass
import warnings
import datetime as dtm


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


from . import _class02_save2stp


# #################################################################
# #################################################################
#          Default values
# #################################################################


_COLOR = 'k'
_NAME = 'rays'


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
    key_optics=None,
    # ---------------
    # options
    factor=None,
    color=None,
    # ---------------
    # saving
    pfe_save=None,
    overwrite=None,
):


    # ----------------
    # check inputs
    # --------------

    (
        key, key_optics,
        outline_only, factor, color,
        iso,
        pfe_save, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        key_optics=key_optics,
        # options
        factor=factor,
        color=color,
        # saving
        pfe_save=pfe_save,
        overwrite=overwrite,
    )

    fname = os.path.split(pfe_save)[-1][:-4]

    # -------------
    # extract and pre-format data
    # -------------

    dptsx, dptsy, dptsz = _extract(
        coll=coll,
        key=key,
        key_cam=key_cam,
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        outline_only=outline_only,
        pfe_in=pfe_in,
        fname=fname,
    )

    # scaling factor
    for k0 in dptsx.keys():
        dptsx[k0] = factor * dptsx[k0]
        dptsy[k0] = factor * dptsy[k0]
        dptsz[k0] = factor * dptsz[k0]

    # ---------------
    # get color dict
    # ---------------

    dcolor = _get_dcolor(dptsx=dptsx, color=color)

    # ----------------
    # get file content
    # ----------------

    # HEADER
    msg_header = _class02_save2stp._get_header(
        fname=fname,
        iso=iso,
    )

    # DATA
    msg_data = _get_data(
        dptsx=dptsx,
        dptsy=dptsy,
        dptsz=dptsz,
        fname=fname,
        # options
        dcolor=dcolor,
        # norm
        iso=iso,
    )

    # -------------
    # save to stp
    # -------------

    _save(
        msg=msg_header + "\n" + msg_data,
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
    ptsx=None,
    ptsy=None,
    ptsz=None,
    pfe_in=None,
    # options
    outline_only=None,
    factor=None,
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
):

    # --------------
    # coll vs pfe_in
    # -------------

    lc = [coll is not None, ptsx is not None, pfe_in is not None]
    if np.sum(lc) != 1:
        msg = (
            "Please provide eiter a (Collection, key) pair xor array xor pfe_in!\n"
            f"\t- coll is None: {coll is None}\n"
            f"\t- (ptsx, ptsy, ptsz) is None: {ptsx is None}\n"
            f"\t- pfe_in is None: {pfe_in is None}\n"
        )
        raise Exception(msg)

    # ---------------
    # coll
    # ---------------

    if lc[0]:


        # ------------
        # coll

        if not issubclass(coll.__class__, ds.DataStock):
            msg = (
                "Arg coll must be a subclass of datastock.Datastock!\n"
                f"\t- type(coll) = {type(coll)}"
            )
            raise Exception(msg)

        # --------------
        # key

        lok_rays = list(coll.dobj.get('rays', {}).keys())
        lok_diag = list(coll.dobj.get('diagnostic', {}).keys())
        key = ds._generic_check._check_var(
            key, 'key',
            types=str,
            allowed=lok_rays + lok_diag,
        )

        if key in lok_diag:
            if isinstance(key_cam, str):
                key_cam = [key_cam]

            lok = coll.dobj['diagnostic'][key]['camera']
            key_cam = ds._generic_check._check_var_iter(
                key_cam, 'key_cam',
                types=(list, tuple),
                types_iter=str,
                allowed=lok,
                default=lok,
            )

        else:
            key_cam = None

    # ---------------
    # array
    # ---------------

    elif lc[1]:

        c0 = all([
            isinstance(pp, np.ndarray)
            and pp.ndim >= 2
            and pp.shape[0] >= 2
            for pp in [ptsx, ptsy, ptsz]
        ])
        if not c0:
            msg = (
                "Args (ptsx, ptsy, ptsz) must be np.ndarrays of shape (npts>=2, nlos)\n"
                f"\t- ptsx: {ptsx}\n"
                f"\t- ptsy: {ptsy}\n"
                f"\t- ptsz: {ptsz}\n"
            )
            raise Exception(msg)

        if not (ptsx.shape == ptsy.shape == ptsz.shape):
            msg = (
                "Args (ptsx, ptsy, ptsz) must have the same shape!\n"
                f"\t- ptsx.shape: {ptsx.shape}\n"
                f"\t- ptsy.shape: {ptsy.shape}\n"
                f"\t- ptsz.shape: {ptsz.shape}\n"
            )
            raise Exception(msg)

    # ---------------
    # pfe_in
    # ---------------

    else:

        c0 = (
            isinstance(pfe_in, str)
            and os.path.isfile(pfe_in)
            and (pfe_in.endswith('.csv') or pfe_in.endswith('.dat'))
        )

        if not c0:
            msg = (
                "Arg pfe_in must be a path to a .csv file!\n"
                f"Provided: {pfe_in}"
            )
            raise Exception(msg)
        key = None

    # ---------------
    # outline_only
    # ---------------

    outline_only = ds._generic_check._check_var(
        outline_only, 'outline_only',
        types=bool,
        default=True,
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
    # iso
    # ---------------

    iso = 'ISO-10303-21'

    # ---------------
    # pfe_save
    # ---------------

    # Default
    if pfe_save is None:
        path = os.path.abspath('.')
        name = key if key is not None else _NAME
        pfe_save = os.path.join(path, f"{name}.stp")

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
            "Arg pfe_save must be a saving file str ending in '.stp'!\n"
            f"Provided: {pfe_save}"
        )
        raise Exception(msg)

    # makesure extension is included
    if not pfe_save.endswith('.stp'):
        pfe_save = f"{pfe_save}.stp"

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
        ptsx, ptsy, ptsz,
        pfe_in,
        outline_only, factor, color,
        iso,
        pfe_save, overwrite,
    )


# #################################################################
# #################################################################
#          extract
# #################################################################