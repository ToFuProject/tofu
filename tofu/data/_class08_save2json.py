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

    # HEADER
    msg_header = _get_header(
        fname=fname,
    )

    # DATA
    dout = _extract(
        coll=coll,
        key=key,
        key_cam=key_cam,
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
#          extract
# #################################################################


def _extract(
    coll=None,
    key=None,
    key_cam=None,
):

    # ----------------------
    # initialize
    # ----------------------

    dout = {}

    # ----------------------
    # extract points
    # ----------------------

    for i0, k0 in enumerate(key_optics):

        # points of polygons
        ptsx, ptsy, ptsz = coll.get_optics_poly(
            key=k0,
            add_points=False,
            # min_threshold=4e-3,
            min_threshold=None,
            mode=None,
            closed=True,
            ravel=None,
            total=True,
            return_outline=False,
        )

        # store
        if ptsx.ndim == 1:
            dptsx[k0], dptsy[k0], dptsz[k0] = ptsx, ptsy, ptsz

        elif ptsx.ndim == 2:
            for ii in range(ptsx.shape[1]):
                key = f"{k0}_{ii}"
                dptsx[key] = ptsx[:, ii]
                dptsy[key] = ptsy[:, ii]
                dptsz[key] = ptsz[:, ii]

        else:
            raise NotImplementedError(str(ptsx.shape))

        # unit vectors
        k0, cls = coll.get_optics_cls(optics=k0)
        k0, cls = k0[0], cls[0]
        if cls == 'camera':

            if coll.dobj[cls][k0]['dgeom']['parallel'] is True:
                duvect[k0] = {
                    'nin': coll.dobj[cls][k0]['dgeom']['nin'],
                    'e0': coll.dobj[cls][k0]['dgeom']['e0'],
                    'e1': coll.dobj[cls][k0]['dgeom']['e1'],
                }

            else:
                dv = coll.get_camera_unit_vectors(k0)
                lv = [
                    'nin_x', 'nin_y', 'nin_z',
                    'e0_x', 'e0_y', 'e0_z',
                    'e1_x', 'e1_y', 'e1_z',
                ]
                nin_x, nin_y, nin_z, e0x, e0y, e0z, e1x, e1y, e1z = [
                    dv[k1] for k1 in lv
                ]

                if e0x.ndim == 1:
                    for ii in range(ptsx.shape[1]):
                        key = f"{k0}_{ii}"
                        duvect[key] = {
                            'nin': np.r_[nin_x[ii], nin_y[ii], nin_z[ii]],
                            'e0': np.r_[e0x[ii], e0y[ii], e0z[ii]],
                            'e1': np.r_[e1x[ii], e1y[ii], e1z[ii]],
                        }

                else:
                    raise NotImplementedError(str(e0x.shape))

        else:
            duvect[k0] = {
                'nin': coll.dobj[cls][k0]['dgeom']['nin'],
                'e0': coll.dobj[cls][k0]['dgeom']['e0'],
                'e1': coll.dobj[cls][k0]['dgeom']['e1'],
            }

    # ------------------------
    # correspondence pts vect

    dcor_ptsvect = {}
    for k0 in dptsx.keys():
        if k0 in duvect.keys():
            dcor_ptsvect[k0] = k0
        else:
            key = '_'.join(k0.split('_')[:-1])
            assert key in duvect.keys(), (key, duvect.keys())
            dcor_ptsvect[k0] = key

    return dptsx, dptsy, dptsz, duvect, dcor_ptsvect


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