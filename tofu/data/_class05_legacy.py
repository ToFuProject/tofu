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


def add_crystal(
    coll=None,
    cryst=None,
    key=None,
):

    # -------------------
    # check inputs
    # -------------------

    coll, cryst, key = _check(
        coll=coll,
        cryst=cryst,
        key=key,
    )

    # --------------------
    # extract geometry
    # --------------------

    dgeom = _extract_dgeom(cryst)

    # --------------------
    # extract material
    # --------------------

    dmat = _extract_dmat(cryst)

    # --------------------
    # add
    # --------------------

    coll.add_crystal(
        key=key,
        dgeom=dgeom,
        dmat=dmat,
        color=cryst._dmisc['color'],
    )

    return


# ###############################################################
# ###############################################################
#                           Check
# ###############################################################

def _check(
    coll=None,
    cryst=None,
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
    # check crystal
    # -------------------

    # load from file
    if isinstance(cryst, str):
        error = False
        if not (os.path.isfile(cryst) and cryst.endswith('.npz')):
            error = 'not a valid file name'
        else:
            try:
                cryst = tf.load(cryst)
            except Exception as err:
                error = f'tf.load() failed to load {cryst}\n{str(err)}'

        # raise error
        if error is not False:
            msg = (
                f"Arg cryst must be a valid .npz file, loadable!\n{error}"
            )
            raise Exception(msg)

    # check class
    if not cryst.__class__.__name__ == 'CrystalBragg':
        msg = (
            "Arg cryst must be a legacy tf.geom.CrystalBragg instance!\n"
            "Provided:\n"
            f"\t- class: {type(cryst)}\n"
            f"\t- value: {cryst}\n"
        )
        raise Exception(msg)

    # -------------------
    # check key
    # -------------------

    # default value
    kdef = cryst.Id.Name

    # check
    lout = list(coll.dobj.get('crystal', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        default=kdef,
        excluded=lout,
    )

    return coll, cryst, key


# ###############################################################
# ###############################################################
#                       Extract dgeom
# ###############################################################


def _extract_dgeom(cryst):

    # ----------------------------
    # intialize with unit vectors

    dgeom = {
        'nin': np.copy(cryst._dgeom['nin']),
        'e0': np.copy(-cryst._dgeom['e1']),
        'e1': np.copy(cryst._dgeom['e2']),
    }

    # ----------
    # get center

    dgeom['cent'] = np.copy(cryst._dgeom['summit'])

    # ---------------
    # get extenthalf

    dgeom['extenthalf'] = np.copy(cryst._dgeom['extenthalf'])

    # -----------------------
    # get radius of curvature

    typ = cryst.dgeom['Type']
    rc = float(cryst._dgeom['rcurve'])
    if typ == 'sph':
        dgeom['curve_r'] = np.r_[rc, rc]
    else:
        msg = (
            "Suspicious, legacy tofu only handles spherical CrystalBragg!\n"
            f"Provided:\n\t- 'Type': {typ}"
        )
        raise NotImplementedError(msg)

    return dgeom


# ###############################################################
# ###############################################################
#                       Extract dmat
# ###############################################################


def _extract_dmat(cryst):

    # ------------
    # dmat

    dmat = {
        'material': str(cryst._dmat['formula']),
        'name': None,
        'symbol': None,
        'mesh': {
            'type': cryst._dmat['symmetry'],
        },
        'target': {
            'lamb': float(cryst._dbragg['lambref']),
            'bragg': float(cryst._dbragg['braggref']),
        },
        'd_hkl': float(cryst._dmat['d']),
        'alpha': float(cryst._dmat['alpha']),
        'beta': float(cryst._dmat['beta']),
        'density': float(cryst._dmat['density']),
        'miller': np.r_[cryst._dmat['cut'][:2], cryst._dmat['cut'][-1]],
    }

    # -------------
    # rocking curve

    if cryst._dbragg.get('rockingcurve') is not None:
        dmat['drock'] = {
            'angle_rel': cryst._dbragg['rockingcurve']['dangle'],
            'power_ratio': cryst._dbragg['rockingcurve']['value'],
        }

    return dmat