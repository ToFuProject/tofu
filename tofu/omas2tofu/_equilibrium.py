

import itertools as itt
import warnings


import numpy as np


from . import _ddef
from . import _utils


# ########################################################
# ########################################################
#                X-points
# ########################################################


def main(
    din=None,
    coll=None,
    ids='equilibrium',
    prefix=None,
    dshort=None,
    strict=None,
    warn=None,
):

    short = 'Xpt'
    dfail = {}

    # ---------------------
    # get reft
    # ---------------------
    ref0 = dshort[ids]['t']['ref0']
    ids_short = _ddef._DIDS[ids]
    reft = [kk for kk in coll.dref.keys() if f"{ids_short}_{ref0}" in kk]
    if len(reft) != 1:
        msg = "No / several timeref found for Xpt!"
        if strict is True:
            raise Exception(msg)
        else:
            dfail[short] = msg
            _warnings(ids, warn, dfail)

    reft = reft[0]
    nt = coll.dref[reft]['size']

    # ---------------------
    # check presence of Xpt
    # ---------------------

    c0 = (
        len(din[ids]['time_slice']) > 0
        and any([
            ts['boundary'].get('x_point') is not None
            and len(ts['boundary']['x_point']) > 0
            for ts in din[ids]['time_slice']
        ])
    )
    dfail = {}
    if not c0:
        msg = "no X point found in {ids} from file!\n"
        if strict is True:
            raise Exception(msg)
        else:
            dfail[short] = msg
            _warnings(ids, warn, dfail)

    # -------------
    # prepare
    # -------------

    npts = [
        0 if ts['boundary'].get('x_point') is None
        else len(ts['boundary']['x_point'])
        for ts in din[ids]['time_slice']
    ]
    npts_max = np.max(npts)

    XptsR = np.full((nt, npts_max), np.nan)
    XptsZ = np.full((nt, npts_max), np.nan)
    for ii, ts in enumerate(din[ids]['time_slice']):
        for jj in range(npts[ii]):
            XptsR[ii, jj] = ts['boundary']['x_point'][jj]['r']
            XptsZ[ii, jj] = ts['boundary']['x_point'][jj]['z']

    # -------------
    # keys
    # -------------

    # ref nXpts`
    krnpts = _utils._make_key(
        prefix=prefix,
        ids=ids,
        short='nXpts',
    )

    # XptsR
    kXptsR = _utils._make_key(
        prefix=prefix,
        ids=ids,
        short='XptsR',
    )

    # XptsZ
    kXptsZ = _utils._make_key(
        prefix=prefix,
        ids=ids,
        short='XptsZ',
    )

    # -------------
    # store
    # -------------

    # reference
    coll.add_ref(
        krnpts,
        size=npts_max,
    )

    # XptsR
    coll.add_data(
        kXptsR,
        data=XptsR,
        ref=(reft, krnpts),
        units='m',
        dim='distance',
    )

    # XptsZ
    coll.add_data(
        kXptsZ,
        data=XptsZ,
        ref=(reft, krnpts),
        units='m',
        dim='distance',
    )

    return


# #######################################################
# #######################################################
#               WARNINGS
# #######################################################


def _warnings(ids, warn, dfail):
    if warn is True and len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following data could not be loaded:\n"
            f"From ids = {ids}\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)


# ########################################################
# ########################################################
#           Subkey for 1d-2d mesh
# ########################################################


def _get_subkey(
    coll=None,
    ids=None,
    shape=None,
    axis=None,
    ddata=None,
    ldata=None,
    lk2d=None,
    k1d=None,
    q1d=None,
    # unused
    **kwdargs,
):
    # ------------------
    # back-up
    # ------------------

    # get k2
    lk1 = [kk for kk in ldata if kk.endswith(k1d[:-1])]
    lk2 = [
        kk for kk in lk2d
        if kk.endswith(k1d.replace('1d', '2d')[:-1])
    ]
    if len(lk1) != 1:
        msg = f"Unidentified 1d base for 2d subkey: {lk1}"
        raise Exception(msg)
    if len(lk2) != 1:
        msg = f"Unidentified 2d subkey: {lk2}"
        raise Exception(msg)
    k2d = lk2[0]

    # slices
    sli0 = tuple([
        0 if ii == axis else slice(None) for ii in range(len(shape))
    ])
    sli1 = tuple([
        -1 if ii == axis else slice(None) for ii in range(len(shape))
    ])
    sli2 = tuple(itt.chain.from_iterable([
        [None, None] if ii == axis
        else [slice(None)] for ii in range(len(shape))
    ]))

    # compute normalization
    q1d0 = ddata[lk1[0]]['data'][sli0][sli2]
    q1d1 = ddata[lk1[0]]['data'][sli1][sli2]
    q2dn = (coll.ddata[k2d]['data'] - q1d0) / (q1d1 - q1d0)

    # add data2d
    k2dn = f"{k2d}n"
    coll.add_data(
        key=k2dn,
        data=q2dn,
        ref=coll.ddata[k2d]['ref'],
    )

    return k1d, q1d, k2dn
