# -*- coding: utf-8 -*-
"""
Polygons are assumed to be:
    - finite
    - simple (non self-intersecting)
    - non-explicitly closed
    - have no repeated points

"""


import numpy as np
from matplotlib.path import Path
import datastock as ds


from . import _class00_poly2d_check as _poly2d_check


__all__ = ['main']


# ###########################################################
# ###########################################################
#              main
# ###########################################################


def main(
    x0=None,
    x1=None,
    key=None,
    # options for edges
    dedge=None,
    dsurface=None,
):

    # ----------------
    # check inputs
    # ----------------

    # polygon formatting
    din = _poly2d_check.check(
        x0=x0,
        x1=x1,
        key=key,
        # options
        closed=True,
        clockwise=True,
    )

    # dedge, dsurface
    dedge, dsurface = _check(
        dedge=dedge,
        dsurface=dsurface,
    )

    # ----------------
    # sample edges
    # ----------------

    if dedge is not None:
        dout_edge = _edges(
            x0_closed=din['x0'],
            x1_closed=din['x1'],
            key=din['key'],
            # options
            **{k0: dedge.get(k0) for k0 in ['res', 'factor']},
        )
    else:
        dout_edge = {
            'x0': [],
            'x1': [],
        }

    # ----------------
    # sample surface
    # ----------------

    if dsurface is not None:
        dout_surf = _surface(
            x0_closed=din['x0'],
            x1_closed=din['x1'],
            key=din['key'],
            # options
            **{k0: dsurface.get(k0) for k0 in ['res', 'nb']},
        )
    else:
        dout_surf = {
            'x0': [],
            'x1': [],
        }

    # ----------------
    # combine
    # ----------------

    dout = {
        'x0': np.r_[dout_edge['x0'], dout_surf['x0']],
        'x1': np.r_[dout_edge['x1'], dout_surf['x1']],
        'edge_res': dout_edge.get('res'),
        'edge_factor': dout_edge.get('factor'),
        'surface_nb': dout_surf.get('nb'),
        'key': din['key'],
    }

    return dout


# ###########################################################
# ###########################################################
#              check main
# ###########################################################


def _check(
    dedge=None,
    dsurface=None,
):

    # --------------
    # dedge
    # --------------

    if dedge is not None:
        lk = ['res', 'factor']
        c0 = (
            isinstance(dedge, dict)
            and all([kk in lk for kk in dedge.keys()])
        )

        if not c0:
            lstr = [f"\t- {kk}" for kk in lk]
            msg = (
                "Arg dedge must be None or a dict with keys:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # --------------
    # dsurface
    # --------------

    if dsurface is not None:
        lk = ['res', 'nb']
        c0 = (
            isinstance(dsurface, dict)
            and all([kk in lk for kk in dsurface.keys()])
        )

        if not c0:
            lstr = [f"\t- {kk}" for kk in lk]
            msg = (
                "Arg dsurface must be None or a dict with keys:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    return dedge, dsurface


# ###########################################################
# ###########################################################
#              Sample edges
# ###########################################################


def _edges(
    x0_closed=None,
    x1_closed=None,
    key=None,
    # options
    res=None,
    factor=None,
):

    # ----------------
    # check inputs
    # ----------------

    # options
    res, factor = _check_edges(
        res=res,
        factor=factor,
    )

    # ----------------
    # prepare
    # ----------------

    # get all lengths
    dx0 = x0_closed[1:] - x0_closed[:-1]
    dx1 = x1_closed[1:] - x1_closed[:-1]
    dist = np.hypot(dx0, dx1)

    # ----------------
    # determine samplung res
    # ----------------

    if isinstance(res, str):

        # get res
        if res == 'min':
            res = np.min(dist)
        elif res == 'max':
            res = np.max(dist)
        else:
            raise NotImplementedError()

        # adjust
        res = res * factor

    # ----------------
    # sample res
    # ----------------

    # nb of points to be inserted in each segment
    npts = np.round(dist / res, decimals=0).astype(int)

    # interpolation for x0
    out0 = np.concatenate([
        np.r_[x0_closed[ii]]
        if npts[ii] <= 1 else
        x0_closed[ii] + dx0[ii] * np.linspace(0, 1, npts[ii], endpoint=False)
        for ii in range(x0_closed.size-1)
    ])

    # interpolation for x1
    out1 = np.concatenate([
        np.r_[x1_closed[ii]]
        if npts[ii] <= 1 else
        x1_closed[ii] + dx1[ii] * np.linspace(0, 1, npts[ii], endpoint=False)
        for ii in range(x1_closed.size-1)
    ])

    # ----------------
    # return
    # ----------------

    dout = {
        'x0': out0,
        'x1': out1,
        'res': res,
        'factor': factor,
    }

    return dout


# ###########################################################
# ###########################################################
#              check edges
# ###########################################################


def _check_edges(
    res=None,
    factor=None,
):

    # -------------
    # res
    # -------------

    res = ds._generic_check._check_var(
        res, 'res',
        types=(str, float),
    )

    if isinstance(res, str):
        res = ds._generic_check._check_var(
            res, 'res',
            types=str,
            default='min',
            allowed=['min', 'max'],
        )

    else:
        res = ds._generic_check._check_var(
            res, 'res',
            types=float,
            sign='>0',
        )

    # -------------
    # factor
    # -------------

    if isinstance(res, str):
        factor = float(ds._generic_check._check_var(
            factor, 'factor',
            types=(int, float),
            default=1,
            sign='>0',
        ))

    else:
        factor = None

    return res, factor


# ###########################################################
# ###########################################################
#              Sample surface
# ###########################################################


def _surface(
    x0_closed=None,
    x1_closed=None,
    key=None,
    # options
    res=None,
    nb=None,
):

    # ----------------
    # check inputs
    # ----------------

    # Dx0
    x0_min = x0_closed.min()
    x0_max = x0_closed.max()
    Dx0 = (x0_max - x0_min)

    # Dx1
    x1_min = x1_closed.min()
    x1_max = x1_closed.max()
    Dx1 = (x1_max - x1_min)

    # options
    nb = _check_surfaces(
        res=res,
        nb=nb,
        key=key,
        Dx0=Dx0,
        Dx1=Dx1,
    )

    # ----------------
    # prepare
    # ----------------

    # out0
    dx0 = Dx0 / nb[0]
    out0 = np.linspace(x0_min + dx0/2, x0_max - dx0/2, nb[0])

    # out1
    dx1 = Dx1 / nb[1]
    out1 = np.linspace(x1_min + dx1/2, x1_max - dx1/2, nb[1])

    # ----------------
    # Check in polygon
    # ----------------

    pts0 = np.repeat(out0[:, None], out1.size, axis=1).ravel()
    pts1 = np.repeat(out1[None, :], out1.size, axis=0).ravel()

    pp = Path(np.array([x0_closed, x1_closed]).T)
    ind = pp.contains_points(np.array([pts0, pts1]).T)

    # ----------------
    # return
    # ----------------

    dout = {
        'x0': pts0[ind],
        'x1': pts1[ind],
        'nb': nb,
    }

    return dout


# ###########################################################
# ###########################################################
#              check surfaces
# ###########################################################


def _check_surfaces(
    res=None,
    nb=None,
    key=None,
    Dx0=None,
    Dx1=None,
):

    # -------------
    # res vs nb
    # -------------

    lc = [
        nb is not None and res is not None,
        nb is None and res is None,
    ]
    if any(lc):
        msg = (
            "Polygon '{key}' surface sampling, please provide res xor nb!\n"
            f"\t- res = {res} \n"
            f"\t- nb = {nb}\n"
        )
        raise Exception(msg)

    # -------------
    # res
    # -------------

    if res is not None:

        if np.isscalar(res):
            res = [res, res]

        res = ds._generic_check._check_var_iter(
            res, 'res',
            types=(list, tuple),
            types_iter=(int, float),
            size=2,
        )

        res = tuple([float(rr) for rr in res])

        nb = [np.ceil(Dx0/res[0]), np.ceil(Dx1/res[1])]

    # -------------
    # nb
    # -------------

    if np.isscalar(nb):
        nb = [nb, nb]

    nb = ds._generic_check._check_var_iter(
        nb, 'nb',
        types=(list, tuple),
        types_iter=(int, float),
        size=2,
    )

    nb = tuple([int(nn) for nn in nb])

    return nb