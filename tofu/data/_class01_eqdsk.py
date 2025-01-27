# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:41:08 2023

@author: dvezinet
"""


# Common
import numpy as np


# ########################################################
# ########################################################
#               Units
# ########################################################


_DUNITS = {
    # ------------
    # str / scalar
    'comment': {
        'units': None,
        'ref': 'neq',
    },
    'shot': {
        'units': 'm',
        'ref': 'neq',
    },
    'current': {
        'key': 'Ip',
        'units': 'A',
        'ref': 'neq',
    },
    # redundant with current
    # 'cpasma': {
    #     'units': 'A',
    #     'ref': 'neq',
    # },
    # --------------
    # Magnetic axis
    'rmagx': {
        'key': 'magaxR',
        'units': 'm',
        'ref': 'neq',
    },
    'zmagx': {
        'key': 'magaxZ',
        'units': 'm',
        'ref': 'neq',
    },
    # Redundant with rmagx and zmagx
    # 'rmaxis': {
    #     'units': 'm',
    #     'ref': 'neq',
    # },
    # 'zmaxis': {
    #     'units': 'm',
    #     'ref': 'neq',
    # },
    'psi_axis': {
        'key': 'psi_magax',
        'units': None,
        'ref': 'neq',
    },
    'psi_boundary': {
        'key': 'psi_sep',
        'units': None,
        'ref': 'neq',
    },
    # ---------
    # mRZ
    'psi': {
        'units': None,
        'ref': ('neq', 'mRZ'),
    },
    # -------------------
    # mRZ: grad-shafranov
    # 'f': {
    #     'units': '',
    #     'ref': 'mRZ',
    # },
    # 'pprime': {
    #     'units': '',
    # },
    # -------------------
    # lim => first wall
    # 'rlim': {
    #     'units': 'm',
    #     'ref': ('neq', 'nlim'),
    # },
    # 'zlim': {
    #     'units': 'm',
    #     'ref': ('neq', 'nlim'),
    # },
    # 'rcentr': {
    #     'units': 'm',
    #     'ref': 'neq',
    # },
    # -------------------
    # bdry => separatrix
    'rbdry': {
        'key': 'sepR',
        'units': 'm',
        'ref': ('neq', 'nsep'),
    },
    'zbdry': {
        'key': 'sepZ',
        'units': 'm',
        'ref': ('neq', 'nsep'),
    },
    # redundant with rbdry and zbdry
    # 'rbbbs': {
    #     'units': 'm',
    #     'ref': ('neq', 'nsep'),
    # },
    # 'zbbbs': {
    #     'units': 'm',
    #     'ref': ('neq', 'nsep'),
    # },
}


for k0, v0 in _DUNITS.items():
    if isinstance(v0['ref'], str):
        _DUNITS[k0]['ref'] = (v0['ref'],)


# ########################################################
# ########################################################
#               load pfe
# ########################################################


def get_load_pfe():

    # -----------------
    # check dependency
    # -----------------

    try:
        from freeqdsk import geqdsk
    except Exception as err:
        msg = (
            "loading an eqdsk file requires an optional dependency:\n"
            "\t- file trying to load: {pfe}\n"
            "\t- required dependency: freeqdsk"
        )
        err.args = (msg,)
        raise err

    # -----------------
    # define load_pfe
    # -----------------

    def func(pfe):

        # ----------
        # load

        data = geqdsk.read(pfe)
        dout = {
            {
                katt: getattr(data, katt) for katt in dir(data)
                if not k0.startswith('__')
            }
        }

        return dout

    return func


# ########################################################
# ########################################################
#               extract grid
# ########################################################


def _extract_grid(dout):

    # -------------------
    # preliminary checks
    # -------------------

    c0 = (
        (dout['nx'] == dout['nr'])
        and (dout['ny'] == dout['nz'])
    )
    if not c0:
        msg = (
            "Something strange with nx, ny, nr, nz:\n"
            f"\t- nx, ny = {dout['nx']}, {dout['ny']}\n"
            f"\t- nr, nz = {dout['nr']}, {dout['nz']}\n"
        )
        raise Exception(msg)

    # -------------------
    # build
    # -------------------

    # extract nb of knots
    nR = dout['nx']
    nZ = dout['ny']

    # extract R
    R = dout['rleft'] + np.linspace(0, dout['rdim'], nR)

    # extract Z
    Z = dout['zmid'] + 0.5 * dout['zdim'] * np.linspace(-1, 1, nZ)

    # -------------------
    # final checks
    # -------------------

    c0 = (
        np.allclose(dout['r_grid'], np.repeat(R[:, None], Z.size, axis=1))
        and np.allclose(dout['z_grid'], np.repeat(Z[None, :], R.size, axis=0))
    )
    if not c0:
        msg = (
            "Something strange with r_grid, z_grid:\n"
            f"\t- r_grid.shape = {dout['r_grid'].shape}\n"
            f"\t- R.size, Z.size = {R.size}, {Z.size}\n"
            f"\t- r_grid = {dout['r_grid']}\n"
            f"\t- R = {R}\n"
            f"\t- Z = {Z}\n"
        )
        raise Exception(msg)

    return R, Z


# ########################################################
# ########################################################
#               Derived
# ########################################################


def _add_rhopn(ddata=None):

    psi0 = ddata['psi_axis']['data']
    psi = ddata['psi']['data']

    rhopn = (psi0[:, None, None] - psi) / psi0[:, None, None]

    return {
        'key': 'rhopn',
        'data': rhopn,
        'units': None,
        'ref': ddata['psi']['ref'],
    }


# def _add_BRZ(ddata=None):

#     psi = psi0 = ddata['psi']['data']

#     # ---------------
#     # BR
#     # ----------------

#     dR = np.diff(dmesh['mRZ']['knots0'])
#     assert np.allclose(dR, dR[0])
#     dR = None

#     psiRp =
#     psiRm =
#     BR = (psiRp - psiRm) / dR

#     dBR = {
#         'key': 'BR',
#         'data': BR,
#         'units': 'T',
#         'ref': ddata['psi']['ref'],
#     }

#     # ---------------
#     # BZ
#     # ----------------

#     dR = np.diff(dmesh['mRZ']['knots0'])
#     assert np.allclose(dR, dR[0])
#     dR =

#     psiRp =
#     psiRm =
#     BR = (psiRp - psiRm) / dR

#     dBR = {
#         'key': 'BR',
#         'data': BR,
#         'units': 'T',
#         'ref': ddata['psi']['ref'],
#     }

#     return dBR, dBZ