# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:27:55 2025

@author: dvezinet
"""


import numpy as np
import datastock as ds


from ..geom import Config, CamLOS1D


# ###########################################################
# ###########################################################
#                   Main
# ###########################################################


def main(
    coll=None,
    key=None,
    config=None,
    allowed=None,
    excluded=None,
):

    # -----------------
    # check inputs
    # -----------------

    key, lstruct = _check(
        coll=coll,
        key=key,
        config=config,
        allowed=allowed,
        excluded=excluded,
    )

    # -----------------
    # prepare
    # -----------------

    startx, starty, startz = coll.get_rays_start(key)
    vx, vy, vz = coll.get_rays_start(key)

    mask = np.isfinite(startx)
    maskn = mask.nonzero()

    # -----------------
    # ray-tracing
    # -----------------

    # call legacy code  -- SOMETHING WRONG HERE
    cam = CamLOS1D(
        dgeom=(
            np.array([startx[mask], starty[mask], startz[mask]]),
            np.array([vx[mask], vy[mask], vz[mask]]),
        ),
        config=config,
        Name='',
        Diag='',
        Exp='',
        strict=False,
    )

    # -----------------
    # ectract
    # -----------------

    dtouch = cam.get_touch_dict()

    # intitialize
    itouch = -np.ones(startx.shape, dtype=int)

    for ii, ks in enumerate(lstruct):

        if ks not in dtouch.keys():
            continue

        sli = tuple([ind[dtouch[ks]['indok']] for ind in maskn])

        itouch[sli] = ii

    # -----------------
    # output
    # -----------------

    dout = {
        'lstruct': lstruct,
        'ind': itouch,
    }

    return dout


# ###########################################################
# ###########################################################
#                   check inputs
# ###########################################################


def _check(
    coll=None,
    key=None,
    config=None,
    allowed=None,
    excluded=None,
):

    # -----------------
    # key
    # -----------------

    wrays = coll._which_rays
    lok = list(coll.dobj.get(wrays, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # -----------------
    # config
    # -----------------

    if not isinstance(config, Config):
        msg = "Arg config must be a tofu Config instance!"
        raise Exception(msg)

    lok = [
        f"{ss.__class__.__name__}_{ss.Id.Name}" for ss in config.lStruct
    ]

    # -------------------
    # allowed vs excluded
    # -------------------

    lc = [allowed is not None, excluded is not None]
    if np.sum(lc) > 1:
        msg = "Please provide either excluded xor allowed, not both!"
        raise Exception(msg)

    # -----------------
    # lstruct
    # -----------------

    if allowed is None and excluded is None:
        lstruct = [
            f"{ss.__class__.__name__}_{ss.Id.Name}" for ss in config.lStruct
            if ss.get_visible()
        ]

    elif allowed is not None:

        if isinstance(allowed, str):
            allowed = [allowed]
        allowed = ds._generic_check._check_var_iter(
            allowed, 'allowed',
            allowed=lok,
            types=(list, tuple),
            types_iter=str,
        )

        lstruct = allowed

    else:
        if isinstance(excluded, str):
            excluded = [excluded]
        excluded = ds._generic_check._check_var_iter(
            excluded, 'excluded',
            allowed=lok,
            types=(list, tuple),
            types_iter=str,
        )

        lstruct = [ss for ss in lok if ss not in excluded]

    return key, lstruct