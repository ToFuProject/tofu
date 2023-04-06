# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds


# ##################################################################
# ##################################################################
#                       Diagnostics
# ##################################################################


def _isconvex(coll=None, keys=None):

    # ------------
    # check inputs

    if isinstance(keys, str):
        keys = [keys]

    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())

    lop = lap + lfilt + lcryst + lgrat

    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=lop,
    )

    # ------
    # loop

    isconvex = []
    for k0 in keys:

        # get class
        if k0 in lap:
            cls = 'aperture'
        elif k0 in lfilt:
            cls = 'filter'
        elif k0 in lcryst:
            cls = 'crystal'
        else:
            cls = 'grating'

        # dgeom
        curve_r = coll.dobj[cls][k0]['dgeom'].get('curve_r')
        if curve_r is None:
            convex = False
        else:
            if any([np.isfinite(rr) and rr > 0. for rr in curve_r]):
                convex = True
            else:
                convex = False
        isconvex.append(convex)

    return isconvex
