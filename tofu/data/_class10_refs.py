# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:00:59 2023

@author: dvezinet
"""

import warnings
import itertools as itt


import numpy as np


# ########################################################
# ########################################################
#               ref_vector_common
# ########################################################


def _get_ref_vector_common(
    coll=None,
    ddata=None,
    key_matrix=None,
    key_profile2d=None,
    dconstraints=None,
    strategy=None,
    strategy_bounds=None,
    dref_vector=None,
):

    # ------------
    # check inputs

    lc = [
        ddata is not None,
        key_profile2d is not None
    ]
    if key_matrix is None or np.sum(lc) != 1:
        msg = "Please provide key_matrix and (ddata or key_profile2d)!"
        raise Exception(msg)

    if dref_vector is None:
        dref_vector = {}

    # ------------
    # geom matrix

    refc0 = None
    wbs = coll._which_bsplines
    key_bs = coll.dobj['geom matrix'][key_matrix]['bsplines']
    key_cam = coll.dobj['geom matrix'][key_matrix]['camera']
    refbs = coll.dobj[wbs][key_bs]['ref-bs']
    lgeom = coll.dobj['geom matrix'][key_matrix]['data']
    for ii, k0 in enumerate(lgeom):
        camdgeom = coll.dobj['camera'][key_cam[ii]]['dgeom']
        camref = camdgeom['ref'] + camdgeom['ref_flat']
        refi = [
            rr for rr in coll.ddata[k0]['ref']
            if rr not in camref
            and rr not in refbs
        ]

        # safety check
        if len(refi) > 1:
            msg = (
                "Geometry matrix with > 1 extra ref not handled yet\n"
                f"\t- gome. mat: '{key_matrix}'\n"
                f"\t- ref: {coll.ddata[k0]['ref']}"
            )
            raise NotImplementedError(msg)

        # sort cases
        if len(refi) == 0:
            assert refc0 is None

        elif len(refi) == 1:
            if refc0 is None:
                assert ii == 0
                refc0 = refi[0]

            assert refc0 == refi[0]

    lk = list(coll.dobj['geom matrix'][key_matrix]['data'])

    # ------------------
    # data or profile2d

    refc1 = None
    if ddata is not None:
        key_cam = ddata['keys_cam']
        for ii, k0 in enumerate(ddata['keys']):
            refi = [
                rr for rr in coll.ddata[k0]['ref']
                if rr not in coll.dobj['camera'][key_cam[ii]]['dgeom']['ref']
            ]
            assert len(refi) <= 1
            if len(refi) == 0:
                assert refc1 is None
            elif len(refi) == 1:
                if refc1 is None:
                    assert ii == 0
                    refc1 = refi[0]
                assert refc1 == refi[0]

        lk += ddata['keys']

        # ------------
        # dconstraints

        if dconstraints is not None:
            if isinstance(dconstraints.get('rmax', {}).get('val'), str):
                lk.append(dconstraints['rmax']['val'])
            if isinstance(dconstraints.get('rmin', {}).get('val'), str):
                lk.append(dconstraints['rmin']['val'])

    # ----------
    # profile2d

    else:

        lrefbs = list(itt.chain.from_iterable([
            v0['ref'] for v0 in coll.dobj[wbs].values()
        ]))

        key_bs = coll.get_profiles2d()[key_profile2d]
        refbs = coll.dobj[wbs][key_bs]['ref-bs']
        refi = [
            rr for rr in coll.ddata[key_profile2d]['ref']
            if rr not in refbs
            # and (refc0 is None or rr == refc0)
            and rr not in lrefbs
        ]
        if len(refi) > 1:
            msg = (key_profile2d, coll.ddata[key_profile2d]['ref'], refi, lrefbs)
            raise Exception(msg)

        if len(refi) == 1:
            refc1 = refi[0]

        lk += [key_profile2d]

    # -----
    # refc

    if refc0 is None and refc1 is None:
        return False, None, None, None, None

    else:

        if refc0 is not None and refc1 is not None:

            if refc0 != refc1:
                msg = (
                    "Non-consistent references with:\n"
                    f"\t- dref_vector: {dref_vector}\n"
                    "For:\n"
                    f"\t- geom matrix: {refc0}\n"
                )
                if ddata is None:
                    msg += f"\t- {key_profile2d}: {refc1}\n"
                else:
                    msg += f"\t- {ddata['keys']}: {refc1}\n"
                warnings.warn(msg)
                refc = None

            else:
                refc = refc0

        elif refc0 is None:
            refc = refc1

        else:
            refc = refc0

        return coll.get_ref_vector_common(
            keys=lk,
            ref=refc,
            strategy=strategy,
            strategy_bounds=strategy_bounds,
            **dref_vector,
        )