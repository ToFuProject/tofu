# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:41:08 2023

@author: dvezinet
"""

# Built-in
import os


# Common
import numpy as np
import datastock as ds
import bsplines2d as bs2


__all_ = ['load_eqdsk']


# ########################################################
# ########################################################
#               EQDSK
# ########################################################


def load_eqdsk(
    dpfe=None,
    returnas=None,
    strict=None,
    # keys
    kmesh=None,
    # optional time
    t=None,
    ktime=None,
    knt=None,
    t_units=None,
):
    """ load multiple eqdsk equilibria files and concatenate them

    If provided, vector t is used as the time vector
    Otherwise t is just a range

    Parameters
    ----------
    dpfe : str of dict of {path: [patterns]}
        DESCRIPTION. The default is None.
    returnas : dict, True or Collection
        DESCRIPTION. The default is None.
    kmesh : str
        DESCRIPTION. The default is None.
    t : sequence, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    error
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    out : dict or Collection
        DESCRIPTION.

    """

    # --------------------
    # check inputs
    # --------------------

    lpfe, returnas, coll, kmesh, t, ktime, knt, t_units, geqdsk = check_inputs(
        dpfe=dpfe,
        returnas=returnas,
        kmesh=kmesh,
        strict=strict,
        # optional time
        t=t,
        ktime=ktime,
        knt=knt,
        t_units=t_units,
    )

    # ----------------
    # load and extract
    # ----------------

    # loop on all files
    dfail = {}
    npfe = len(lpfe)
    for ii, pfe in enumerate(lpfe):

        # --------------
        # open and load

        with open(pfe, "r") as ff:
            data = geqdsk.read(ff)

        # ----------
        # initialize

        if ii == 0:
            # extract nb of knots
            nR = data['nx']
            nZ = data['ny']

            # extract R
            R = data['rleft'] + np.linspace(0, data['rdim'], nR)

            # extract Z
            Z = data['zmid'] + 0.5 * data['zdim'] * np.linspace(-1, 1, nZ)

            # initialize psi
            psi = np.full((npfe, nR, nZ), np.nan)

        # ------------
        # safety check

        c0 = (
            data['nx'] == nR
            and data['ny'] == nZ
            and data['rleft'] == R[0]
            and data['zmid'] == 0.5*(Z[0] + Z[-1])
        )
        if not c0:
            dfail[pfe] = f"({data['nx']}, {data['ny']})"

        # ---------------
        # extract psi map

        psi[ii, :, :] = data['psi']

    # -------------
    # sort vs time

    if t is None:
        psi = psi[0, :, :]

    # ----------------------
    # raise error if needed

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following files have unmatching (R, Z) grids:\n"
            f"\t- reference from {lpfe[0]}: ({nR}, {nZ})\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # build output
    # -------------------

    if returnas is dict:
        ddata, dref = _to_dict(
            R=R,
            Z=Z,
            psi=psi,
            t=t,
            knt=knt,
            ktime=ktime,
            t_units=t_units,
        )

    else:
        _to_Collection(
            coll=coll,
            kmesh=kmesh,
            R=R,
            Z=Z,
            psi=psi,
            # optional time
            t=t,
            ktime=ktime,
            knt=knt,
        )

    # -------------------
    # return
    # -------------------

    if returnas is dict:
        out = ddata, dref
    elif returnas is True:
        out = coll
    else:
        out = None

    return out


# ########################################################
# ########################################################
#               check
# ########################################################


def check_inputs(
    dpfe=None,
    returnas=None,
    strict=None,
    # keys
    kmesh=None,
    t=None,
    knt=None,
    ktime=None,
    t_units=None,
):

    # --------------------
    # check dependency
    # --------------------

    # import dependency
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

    # --------------------
    # check pfe
    # --------------------

    lpfe = ds.get_files(
        dpfe=dpfe,
        returnas=list,
        strict=strict,
    )

    # --------------------
    # check returnas
    # --------------------

    if returnas is None:
        returnas = True

    coll = None
    if returnas is True:
        from ._class10_Inversion import Inversion as Collection
        coll = Collection()

    elif issubclass(returnas.__class__, bs2.BSplines2D):
        coll = returnas

    elif returnas is not dict:
        msg = (
            "returnas must be either:\n"
            "\t- dict: return mesh in ddata and dref dict\n"
            "\t- True: return mesh in new Collection instance\n"
            "\t- Collection instance: add mesh 2d rect to it, named kmesh"
        )
        raise Exception(msg)

    # -------------------
    # kmesh
    # -------------------

    if coll is not None:
        wm = coll._which_mesh
        kmesh = ds._generic_check._obj_key(
            d0=coll.dobj.get(wm, {}),
            short='m',
            key=kmesh,
            ndigits=2,
        )

    # -------------------
    # time
    # -------------------

    if len(lpfe) > 1:

        # time vector
        if t is None:
            t = np.arange(0, len(lpfe))
        t = ds._generic_check._check_flat1darray(
            t, 't',
        )

        # key ref time
        knt = ds._generic_check._obj_key(
            d0=coll.dref,
            short='nt',
            key=knt,
            ndigits=2,
        )

        # key time
        ktime = ds._generic_check._obj_key(
            d0=coll.ddata,
            short='t',
            key=ktime,
            ndigits=2,
        )

        # t_units
        t_units = ds._generic_check._check_var(
            t_units, 't_units',
            types=str,
            default='s',
        )

    else:
        t, ktime, knt, t_units = None, None, None, None

    return lpfe, returnas, coll, kmesh, t, ktime, knt, t_units, geqdsk


# ########################################################
# ########################################################
#               to dict and Collection
# ########################################################


def _to_dict(
    R=None,
    Z=None,
    psi=None,
    t=None,
    knt=None,
    ktime=None,
    t_units=None,
):

    nR = R.size
    nZ = Z.size

    # ref keys
    knR = 'nR'
    knZ = 'nZ'
    kpsi = 'psi'

    # ref
    dref = {
        'nR': {'size': nR},
        'nZ': {'size': nZ},
    }
    if t is not None:
        dref[knt] = {'size': t.size}
        ref = (knt, knR, knZ)
    else:
        ref = (knR, knZ)

    # data keys
    kR = 'R'
    kZ = 'Z'
    kpsi = 'psi2d'

    ddata = {
        kR: {
            'data': R,
            'ref': (knR,),
            'units': 'm',
        },
        kZ: {
            'data': Z,
            'ref': (knZ,),
            'units': 'm',
        },
        kpsi: {
            'data': psi,
            'ref': ref,
            'units': '',
        },
    }
    if t is not None:
        ddata[ktime] = {
            'data': t,
            'ref': knt,
            'units': t_units,
        }

    return ddata, dref


def _to_Collection(
    coll=None,
    kmesh=None,
    R=None,
    Z=None,
    psi=None,
    # optional time
    t=None,
    ktime=None,
    knt=None,
    t_units=None,
):

    # add mesh
    coll.add_mesh_2d_rect(
        key=kmesh,
        knots0=R,
        knots1=Z,
        deg=1,
    )

    # add time
    if t is not None:
        coll.add_ref(key=knt, size=t.size)
        coll.add_data(
            key=ktime,
            data=t,
            ref=knt,
            units=t_units,
        )

    # add psi2d
    kbs = f"{kmesh}_bs1"
    ref = kbs if t is None else (knt, kbs)

    coll.add_data(
        key='psi2d',
        data=psi,
        ref=ref,
        units='',
    )

    # # add rhopn2d
    psi0 = np.nanmin(psi, axis=(-2, -1))
    if t is not None:
        rhopn2d = (psi0[:, None, None] - psi) / psi0[:, None, None]
    else:
        rhopn2d = (psi0 - psi) / psi0

    coll.add_data(
        key='rhopn2d',
        data=rhopn2d,
        ref=ref,
        units='',
    )