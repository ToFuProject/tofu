# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:36:19 2023

@author: dvezinet
"""


import os


import numpy as np
import datastock as ds


# ###############################################################
# ###############################################################
#                           Main
# ###############################################################


def add_camera(
    coll=None,
    cam=None,
    key=None,
):

    # -------------------
    # check inputs
    # -------------------

    coll, cam, key = _check(
        coll=coll,
        cam=cam,
        key=key,
    )

    # --------------------
    # extract geometry
    # --------------------

    dgeom, typ = _extract_dgeom(cam)

    # --------------------
    # extract material
    # --------------------

    dmat = _extract_dmat(cam)

    # --------------------
    # add
    # --------------------

    if typ == '1d':
        coll.add_camera_1d(
            key=key,
            dgeom=dgeom,
            dmat=dmat,
            color=None,
        )

    elif typ == '2d':
        coll.add_camera_2d(
            key=key,
            dgeom=dgeom,
            dmat=dmat,
            color=None,
        )

    else:
        raise NotImplementedError()

    return


# ###############################################################
# ###############################################################
#                           Check
# ###############################################################

def _check(
    coll=None,
    cam=None,
    key=None,
):


    # -------------------
    # check coll
    # -------------------

    import tofu as tf
    if coll is None:
        coll = tf.data.Collection()

    if not isinstance(coll, tf.data.Collection):
        msg = (
            "Arg coll must be a tf.data.Collection instance!\n"
            "Provided:\n"
            "\t- class: {type(coll)}\n"
            "\t- value: {coll}\n"
        )
        raise Exception(msg)

    # -------------------
    # check camera
    # -------------------

    # load from file
    if isinstance(cam, str):
        error = False
        if not (os.path.isfile(cam) and cam.endswith('.npz')):
            error = 'not a valid file name'
        else:
            try:
                cam = tf.load(cam)
            except Exception as err:
                try:
                    cam = dict(np.load(cam, allow_pickle=True))
                except Exception as err:
                    error = f'tf.load() failed to load {cam}\n{str(err)}'

        # raise error
        if error is not False:
            msg = (
                f"Arg cam must be a valid .npz file, loadable!\n{error}"
            )
            raise Exception(msg)

    # check class
    if isinstance(cam, dict):
        pass
    elif 'Camera' not in cam.__class__.__name__:
        msg = (
            "Arg cam must be a legacy tf.geom.CrystalBragg instance!\n"
            "Provided:\n"
            f"\t- class: {type(cam)}\n"
            f"\t- value: {cam}\n"
        )
        raise Exception(msg)

    # -------------------
    # check key
    # -------------------

    # default value
    if isinstance(cam, dict):
        kdef = "cam"
    else:
        kdef = cam.Id.Name

    # check
    lout = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        default=kdef,
        excluded=lout,
    )

    return coll, cam, key


# ###############################################################
# ###############################################################
#                       Extract dgeom
# ###############################################################


def _extract_dgeom(cam):


    if isinstance(cam, dict):

        typ = cam.get('type', '2d')

        x0 = cam['xi']
        x1 = cam['xj']

        dgeom = {
            'cent': np.copy(cam['cent']),
            'cents_x0': np.copy(x0),
            'cents_x1': np.copy(x1),
            'nin': np.copy(cam['nout']),
            'e0': np.copy(cam['ei']),
            'e1': np.copy(cam['ej']),
        }

        # outline of individual pixels
        if cam.get('outline_x0') is None:
            dx0 = 0.5 * np.mean(np.diff(x0))
            out0 = dx0 * np.r_[-1, 1, 1, -1]
            dx1 = 0.5 * np.mean(np.diff(x1))
            out1 = dx1 * np.r_[-1, -1, 1, 1]
            dgeom['outline_x0'] = out0
            dgeom['outline_x1'] = out1
        else:
            dgeom['outline_x0'] = np.copy(cam['outline_x0'])
            dgeom['outline_x1'] = np.copy(cam['outline_x1'])

    else:
        raise NotImplementedError()

    return dgeom, typ


# ###############################################################
# ###############################################################
#                       Extract dmat
# ###############################################################


def _extract_dmat(cam):

    # ------------
    # dmat

    dmat = {}

    return dmat