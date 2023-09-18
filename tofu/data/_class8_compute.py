# -*- coding: utf-8 -*-


import copy
import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import datastock as ds


from . import _utils_surface3d
from . import _spectralunits


# ##################################################################
# ##################################################################
#                   optics outline
# ##################################################################


def get_optics_outline(
    coll=None,
    key=None,
    add_points=None,
    min_threshold=None,
    mode=None,
    closed=None,
    ravel=None,
    total=None,
):

    # ------------
    # check inputs

    # key, cls
    key, cls = coll.get_optics_cls(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    # total
    total = ds._generic_check._check_var(
        total, 'total',
        types=bool,
        default=(cls == 'camera' and dgeom['nd'] == '2d'),
    )
    if cls == 'camera' and dgeom['nd'] != '2d':
        total = False

    # --------
    # compute

    if dgeom['type'] == '3d':
        return None, None

    if cls == 'camera' and total:
        # get centers
        cx0, cx1 = dgeom['cents']
        cx0 = coll.ddata[cx0]['data']
        cx1 = coll.ddata[cx1]['data']

        k0, k1 = dgeom['outline']

        # derive half-spacing
        if cx0.size == 1:
            dx0 = coll.ddata[k0]['data'].max() - coll.ddata[k0]['data'].min()
        else:
            dx0 = np.mean(np.diff(cx0))

        if cx1.size == 1:
            dx1 = coll.ddata[k1]['data'].max() - coll.ddata[k1]['data'].min()
        else:
            dx1 = np.mean(np.diff(cx1))

        # half
        dx0, dx1 = 0.5*dx0, 0.5*dx1

        # derive global outline (not pixel outline)
        p0 = np.r_[
            cx0[0] - dx0, cx0[-1] + dx0,
            cx0[-1] + dx0, cx0[0] - dx0,
        ]
        p1 = np.r_[
            cx1[0] - dx1, cx1[0] - dx1,
            cx1[-1] + dx1, cx1[-1] + dx1,
        ]

    else:
        out = dgeom['outline']
        p0 = coll.ddata[out[0]]['data']
        p1 = coll.ddata[out[1]]['data']

    # -----------
    # add_points

    if add_points is None:
        if cls == 'camera':
            add_points = 0
        else:
            if dgeom['type'] == 'planar':
                add_points = 0
            else:
                add_points = 3

    return _interp_poly(
        lp=[p0, p1],
        add_points=add_points,
        mode=mode,
        isclosed=False,
        closed=closed,
        ravel=ravel,
        min_threshold=min_threshold,
        debug=None,
    )


# ################################################################
# ################################################################
#                   optics poly
# ################################################################


def get_optics_poly(
    coll=None,
    key=None,
    add_points=None,
    min_threshold=None,
    mode=None,
    closed=None,
    ravel=None,
    total=None,
    return_outline=None,
):

    # ------------
    # check inputs

    key, cls = coll.get_optics_cls(optics=key)
    key, cls = key[0], cls[0]

    return_outline = ds._generic_check._check_var(
        return_outline, 'return_outline',
        types=bool,
        default=False,
    )

    ravel = ds._generic_check._check_var(
        ravel, 'ravel',
        default=False,
        types=bool,
    )

    # --------
    # compute

    dgeom = coll.dobj[cls][key]['dgeom']
    if cls in ['aperture', 'filter', 'crystal', 'grating']:

        if dgeom['type'] != '3d':
            p0, p1 = coll.get_optics_outline(
                key=key,
                add_points=add_points,
                min_threshold=min_threshold,
                mode=mode,
                closed=closed,
                ravel=ravel,
                total=total,
            )

            px, py, pz = _utils_surface3d._get_curved_poly(
                gtype=dgeom['type'],
                outline_x0=p0,
                outline_x1=p1,
                curve_r=dgeom['curve_r'],
                cent=dgeom['cent'],
                nin=dgeom['nin'],
                e0=dgeom['e0'],
                e1=dgeom['e1'],
            )

        else:
            px, py, pz = dgeom['poly']
            px = coll.ddata[px]['data']
            py = coll.ddata[py]['data']
            pz = coll.ddata[pz]['data']

    elif cls == 'camera':

        p0, p1 = coll.get_optics_outline(
            key=key,
            add_points=add_points,
            min_threshold=min_threshold,
            mode=mode,
            closed=closed,
            ravel=ravel,
            total=total,
        )

        # vectors
        dv = coll.get_camera_unit_vectors(key)
        lv = ['e0_x', 'e0_y', 'e0_z', 'e1_x', 'e1_y', 'e1_z']
        e0x, e0y, e0z, e1x, e1y, e1z = [dv[k0] for k0 in lv]
        if not np.isscalar(e0x):
            e0x = e0x[:, None]
            e0y = e0y[:, None]
            e0z = e0z[:, None]
            e1x = e1x[:, None]
            e1y = e1y[:, None]
            e1z = e1z[:, None]

        if dgeom['nd'] == '2d' and total:
            cx, cy, cz = dgeom['cent']
            p02, p12 = p0, p1
        else:
            cx, cy, cz = coll.get_camera_cents_xyz(key)
            shape = [1 for ii in range(cx.ndim)] + [p0.size]
            cx, cy, cz = cx[..., None], cy[..., None], cz[..., None]
            p02 = p0.reshape(shape)
            p12 = p1.reshape(shape)

        # make 3d
        px = cx + p02 * e0x + p12 * e1x
        py = cy + p02 * e0y + p12 * e1y
        pz = cz + p02 * e0z + p12 * e1z

    # ----------
    # ravel

    if ravel is True and px.ndim > 1:
        nan = np.full(tuple(np.r_[px.shape[:-1], 1]), np.nan)
        px = np.concatenate((px, nan), axis=-1).ravel()
        py = np.concatenate((py, nan), axis=-1).ravel()
        pz = np.concatenate((pz, nan), axis=-1).ravel()

    # return
    if return_outline is True:
        return p0, p1, px, py, pz
    else:
        return px, py, pz


# ##################################################################
# ##################################################################
#           optics as input dict for solid angle computation
# ##################################################################


def get_optics_as_input_solid_angle(
    coll=None,
    keys=None,
):

    # ------------
    # check inputs

    if isinstance(keys, str):
        keys = [keys]

    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())

    lok = lap + lfilt + lcryst + lgrat

    keys = ds._generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # -----------
    # prepare

    # get classes
    lcls = coll.get_optics_cls(keys)[1]

    # ------------------
    # build list of dict

    dap = {}
    for ii, k0 in enumerate(keys):

        poly_x, poly_y, poly_z = coll.get_optics_poly(k0)

        dap[k0] = {
            'poly_x': poly_x,
            'poly_y': poly_y,
            'poly_z': poly_z,
            'nin': coll.dobj[lcls[ii]][k0]['dgeom']['nin'],
        }

    return dap


# ##################################################################
# ##################################################################
#                   Poly interpolation utilities
# ##################################################################


def _interp_poly_check(
    add_points=None,
    mode=None,
    closed=None,
    ravel=None,
    min_threshold=None,
):

    # -------
    # mode

    mode = ds._generic_check._check_var(
        mode, 'mode',
        default=None,
        allowed=[None, 'mean', 'min', 'thr'],
    )

    # ----------
    # add_points

    defadd = 1 if mode == 'min' else 0
    add_points = ds._generic_check._check_var(
        add_points, 'add_points',
        types=int,
        default=defadd,
        sign='>= 0',
    )

    # -------
    # closed

    closed = ds._generic_check._check_var(
        closed, 'closed',
        default=False,
        types=bool,
    )

    # -------
    # ravel

    ravel = ds._generic_check._check_var(
        ravel, 'ravel',
        default=False,
        types=bool,
    )

    # -------
    # min_threshold

    min_threshold = ds._generic_check._check_var(
        min_threshold, 'min_threshold',
        default=100e-6,
        types=float,
        sign='>=0',
    )

    return add_points, mode, closed, ravel, min_threshold


def _interp_poly(
    lp=None,
    add_points=None,
    mode=None,
    isclosed=None,
    closed=None,
    ravel=None,
    min_threshold=None,
    debug=None,
):
    """ Interpolate list of polygons by adding points ons segments

    lp = list of coordinates arrays describing polygons
        - [px, py]
        - [px, py, pz]
    Each coordinate array can be 1d or 2d

    modes: determine a multiplicative factor to add_points
        - None: 1
        - min: determined by minimum segment length
        - mean: determined by mean segment length

    """

    # ------------
    # check inputs

    add_points, mode, closed, ravel, min_threshold = _interp_poly_check(
        add_points=add_points,
        mode=mode,
        closed=closed,
        ravel=ravel,
        min_threshold=min_threshold,
    )

    # ------------
    # trivial case

    if add_points == 0:

        if isclosed is False and closed is True:
            for ii, pp in enumerate(lp):
                if pp is None:
                    continue

                if pp.ndim == 2:
                    lp[ii] = np.concatenate((pp, pp[:, 0:1]), axis=1)
                else:
                    lp[ii] = np.r_[pp, pp[0]]

        elif isclosed is True and closed is False:
            for ii, pp in enumerate(lp):
                if pp is None:
                    continue

                if pp.ndim == 2:
                    lp[ii] = pp[:, :-1]
                else:
                    lp[ii] = pp[:-1]
        return lp

    # ------------
    # compute

    # close for interpolation
    if isclosed is not True:
        for ii, pp in enumerate(lp):

            if pp is None:
                continue

            if pp.ndim == 2:
                lp[ii] = np.concatenate((pp, pp[:, 0:1]), axis=1)
            else:
                lp[ii] = np.append(pp, pp[0])

    # -----------
    # mode

    if mode is not None:
        if len(lp) == 3:
            dist = np.sqrt(
                np.diff(lp[0], axis=-1)**2
                + np.diff(lp[1], axis=-1)**2
                + np.diff(lp[2], axis=-1)**2
            )
        elif len(lp) == 2:
            dist = np.sqrt(
                np.diff(lp[0], axis=-1)**2
                + np.diff(lp[1], axis=-1)**2
            )

        if dist.ndim == 2:
            import pdb; pdb.set_trace()     # DB

        if mode == 'thr':
            mindist = min_threshold
            add_points = np.ceil(dist / mindist).astype(int) - 1
        else:
            min_threshold = min(min_threshold, np.max(dist)/3.)
            if mode == 'min':
                mindist = np.min(dist[dist > min_threshold])
            elif mode == 'mean':
                mindist = np.mean(dist[dist > min_threshold])

            add_points = add_points * np.ceil(dist / mindist).astype(int) - 1

    # -----------
    # add_points

    shape = [pp for pp in lp if pp is not None][0].shape
    nb = shape[-1]
    if np.isscalar(add_points):
        add_points = np.full((nb-1,), add_points, dtype=int)

    # -----------
    # indices

    ind0 = np.arange(0, nb)
    ind = np.concatenate(tuple([
        np.linspace(
            ind0[ii],
            ind0[ii+1],
            2 + add_points[ii],
            endpoint=True,
        )[:-1]
        for ii in range(nb-1)
    ] + [[ind0[-1]]]))

    # -----------
    # interpolate

    for ii, pp in enumerate(lp):

        if pp is None:
            continue

        lp[ii] = scpinterp.interp1d(
            ind0, pp, kind='linear', axis=-1,
        )(ind)

    # ------------
    # closed

    if closed is False:

        for ii, pp in enumerate(lp):
            if pp is None:
                continue

            if pp.ndim == 2:
                lp[ii] = pp[:, :-1]
            else:
                lp[ii] = pp[:-1]

    # ------------
    # ravel

    if ravel and len(shape) == 2:
        nan = np.full((pp.shape[0], 1), np.nan)
        for ii, pp in enumerate(lp[2:]):
            lp[ii+2] = np.concatenate((pp, nan), axis=1).ravel()

    return lp


def _harmonize_polygon_sizes(
    lp0=None,
    lp1=None,
    nmin=0,
):
    """ From a list of polygons return an array

    """

    # prepare
    npoly = len(lp0)
    ln = [p0.size if p0 is not None else 0 for p0 in lp0]
    nmax = max(np.max(ln), nmin)
    nan = np.full((nmax,), np.nan)

    # prepare output
    x0 = np.full((npoly, nmax), np.nan)
    x1 = np.full((npoly, nmax), np.nan)

    for ii, p0 in enumerate(lp0):

        if p0 is None:
            continue

        elif ln[ii] < nmax:

            ndif = nmax - ln[ii]
            ind0 = np.arange(0, ln[ii] + 1)

            # create imax
            iseg = np.arange(0, ndif) % ln[ii]
            npts = np.unique(iseg, return_counts=True)[1]
            if npts.size < ln[ii]:
                npts = np.r_[
                    npts, np.zeros((ln[ii] - npts.size,))
                ].astype(int)

            imax = np.concatenate(tuple([
                np.linspace(
                    ind0[ii],
                    ind0[ii+1],
                    2 + npts[ii],
                    endpoint=True,
                )[:-1]
                for ii in range(ln[ii])
            ]))

            # interpolate
            x0[ii, :] = scpinterp.interp1d(
                ind0,
                np.r_[p0, p0[0]],
                kind='linear',
            )(imax)
            x1[ii, :] = scpinterp.interp1d(
                ind0,
                np.r_[lp1[ii], lp1[ii][0]],
                kind='linear',
            )(imax)

        else:
            x0[ii, :] = p0
            x1[ii, :] = lp1[ii]

    return x0, x1



# ###############################################################
# ###############################################################
#                       dplot
# ###############################################################


def _dplot_check(
    coll=None,
    key=None,
    key_cam=None,
    optics=None,
    elements=None,
    vect_length=None,
    axis_length=None,
    dx0=None,
    dx1=None,
    default=None,
):
    # -----
    # key

    key, key_cam = coll.get_diagnostic_cam(
        key,
        key_cam,
        default=default,
    )

    # ------
    # optics

    if isinstance(optics, str):
        optics = [optics]

    lok = list(itt.chain.from_iterable([
        [k0] + v0['optics']
        for k0, v0 in coll.dobj['diagnostic'][key]['doptics'].items()
        if k0 in key_cam
    ]))
    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        default=lok,
        allowed=lok,
    )

    # -------
    # elements

    lok = ['o', 'c', 'v', 'r']
    elements = ds._generic_check._check_var_iter(
        elements, 'elements',
        types=str,
        types_iter=str,
        default=''.join(lok),
        allowed=lok,
    )

    # -----------
    # vect_length

    vect_length = ds._generic_check._check_var(
        vect_length, 'vect_length',
        default=0.2,
        types=(float, int),
        sign='>= 0.'
    )

    # -----------
    # axis_length

    axis_length = ds._generic_check._check_var(
        axis_length, 'axis_length',
        default=1.,
        types=(float, int),
        sign='>= 0.'
    )

    # ---------------
    # dx0, dx1

    # dx0
    dx0 = float(ds._generic_check._check_var(
        dx0, 'dx0',
        types=(int, float),
        default=0.,
    ))

    # dx1
    dx1 = float(ds._generic_check._check_var(
        dx1, 'dx1',
        types=(int, float),
        default=0.,
    ))


    return (
        key, key_cam, optics, elements,
        vect_length, axis_length,
        dx0, dx1,
    )


def _dplot(
    coll=None,
    key=None,
    key_cam=None,
    optics=None,
    elements=None,
    vect_length=None,
    axis_length=None,
    dx0=None,
    dx1=None,
    default=None,
):

    # ------------
    # check inputs

    (
        key, key_cam, optics, elements,
        vect_length, axis_length,
        dx0, dx1,
    ) = _dplot_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        optics=optics,
        elements=elements,
        vect_length=vect_length,
        axis_length=axis_length,
        dx0=dx0,
        dx1=dx1,
        default=default,
    )

    # ------------
    # build dict

    dlw = {
        'camera': 2,
        'aperture': 1.,
        'filter': 1.,
        'crystal': 1.,
        'grating': 1.,
    }
    dplot = {k0: {} for k0 in optics}
    for k0 in optics:

        if k0 in coll.dobj.get('camera', []):
            cls = 'camera'
        elif k0 in coll.dobj.get('aperture', []):
            cls = 'aperture'
        elif k0 in coll.dobj.get('filter', []):
            cls = 'filter'
        elif k0 in coll.dobj.get('crystal', []):
            cls = 'crystal'
        elif k0 in coll.dobj.get('grating', []):
            cls = 'grating'
        else:
            msg = f"Unknown optics '{k0}'"
            raise Exception(msg)

        v0 = coll.dobj[cls][k0]['dgeom']
        color = coll.dobj[cls][k0]['dmisc']['color']

        # -----------
        # prepare data

        # cent
        if 'c' in elements or 'v' in elements or 'r' in elements:
            if v0.get('cent') is not None:
                cx, cy, cz = v0['cent'][:, None]
            elif 'cents' in v0.keys():
                cx, cy, cz = v0['cents']
                cx = coll.ddata[cx]['data']
                cy = coll.ddata[cy]['data']
                cz = coll.ddata[cz]['data']
            cr = np.hypot(cx, cy)

        # vectors
        if 'v' in elements or 'r' in elements:
            ninx, niny, ninz = v0['nin']
            e0x, e0y, e0z = v0['e0']
            e1x, e1y, e1z = v0['e1']
            if isinstance(ninx, str):
                vinx = coll.ddata[ninx]['data'] * vect_length
                viny = coll.ddata[niny]['data'] * vect_length
                vinz = coll.ddata[ninz]['data'] * vect_length
                v0x = coll.ddata[e0x]['data'] * vect_length
                v0y = coll.ddata[e0y]['data'] * vect_length
                v0z = coll.ddata[e0z]['data'] * vect_length
                v1x = coll.ddata[e1x]['data'] * vect_length
                v1y = coll.ddata[e1y]['data'] * vect_length
                v1z = coll.ddata[e1z]['data'] * vect_length
            else:
                vinx, viny, vinz = np.r_[ninx, niny, ninz] * vect_length
                v0x, v0y, v0z = np.r_[e0x, e0y, e0z] * vect_length
                v1x, v1y, v1z = np.r_[e1x, e1y, e1z] * vect_length

        # radius
        if 'r' in elements and v0['type'] not in ['planar', '', '3d']:
            if v0['type'] == 'cylindrical':
                icurv = (np.isfinite(v0['curve_r'])).nonzero()[0][0]
                rc = v0['curve_r'][icurv]
                eax = [(e0x, e0y, e0z), (e1x, e1y, e1z)][1 - icurv]
            elif v0['type'] == 'spherical':
                rc = v0['curve_r'][0]
            elif v0['type'] == 'toroidal':
                imax = np.argmax(v0['curve_r'])
                imin = 1 - imax
                rmax = v0['curve_r'][imax]
                rmin = v0['curve_r'][imin]
                emax = [(e0x, e0y, e0z), (e1x, e1y, e1z)][imax]
            # extenthalf = v0['extenthalf']

        # -----------------
        # get plotting data

        # outline
        if 'o' in elements:

            p0, p1, px, py, pz = coll.get_optics_poly(
                key=k0,
                add_points=3,
                closed=True,
                ravel=True,
                return_outline=True,
                total=True,
            )

            dplot[k0]['o'] = {
                'x0': p0 + dx0,
                'x1': p1 + dx1,
                'x': px,
                'y': py,
                'z': pz,
                'r': np.hypot(px, py),
                'props': {
                    'label': f'{k0}-o',
                    'lw': dlw[cls],
                    'c': color,
                },
            }

        # center
        if 'c' in elements:

            dplot[k0]['c'] = {
                'x': cx,
                'y': cy,
                'z': cz,
                'r': cr,
                'props': {
                    'label': f'{k0}-o',
                    'ls': 'None',
                    'marker': 'o',
                    'ms': 4,
                    'c': color,
                },
            }

        # unit vectors
        if 'v' in elements:

            vinr = np.hypot(cx + vinx, cy + viny) - cr
            v0r = np.hypot(cx + v0x, cy + v0y) - cr
            v1r = np.hypot(cx + v1x, cy + v1y) - cr

            # dict

            dplot[k0]['v-nin'] = {
                'x': cx,
                'y': cy,
                'z': cz,
                'r': cr,
                'ux': vinx,
                'uy': viny,
                'uz': vinz,
                'ur': vinr,
                'props': {
                    'label': f'{k0}-nin',
                    'fc': 'r',
                    'color': 'r',
                },
            }

            dplot[k0]['v-e0'] = {
                'x': cx,
                'y': cy,
                'z': cz,
                'r': cr,
                'ux': v0x,
                'uy': v0y,
                'uz': v0z,
                'ur': v0r,
                'props': {
                    'label': f'{k0}-e0',
                    'fc': 'g',
                    'color': 'g',
                },
            }

            dplot[k0]['v-e1'] = {
                'x': cx,
                'y': cy,
                'z': cz,
                'r': cr,
                'ux': v1x,
                'uy': v1y,
                'uz': v1z,
                'ur': v1r,
                'props': {
                    'label': f'{k0}-e1',
                    'fc': 'b',
                    'color': 'b',
                },
            }

        # rowland / axis for curved optics
        if 'r' in elements and cls in ['crystal', 'grating']:

            if v0['type'] not in ['cylindrical', 'spherical', 'toroidal']:
                continue

            theta = np.linspace(-1, 1, 50) * np.pi
            if v0['type'] == 'cylindrical':
                c2x = cx + ninx * rc
                c2y = cy + niny * rc
                c2z = cz + ninz * rc
                px = c2x + np.r_[-1, 1] * axis_length * eax[0]
                py = c2y + np.r_[-1, 1] * axis_length * eax[1]
                pz = c2z + np.r_[-1, 1] * axis_length * eax[2]

                lab = f'{k0}-axis',

            elif v0['type'] == 'spherical':
                c2x = cx + ninx * 0.5 * rc
                c2y = cy + niny * 0.5 * rc
                c2z = cz + ninz * 0.5 * rc
                px = (
                    c2x
                    + 0.5 * rc * np.cos(theta) * (-ninx)
                    + 0.5 * rc * np.sin(theta) * e0x
                )
                py = (
                    c2y
                    + 0.5 * rc * np.cos(theta) * (-niny)
                    + 0.5 * rc * np.sin(theta) * e0y
                )
                pz = (
                    c2z
                    + 0.5 * rc * np.cos(theta) * (-ninz)
                    + 0.5 * rc * np.sin(theta) * e0z
                )

                lab = f'{k0}-rowland',

            elif v0['type'] == 'toroidal':
                c2x = cx + ninx * (rmin + rmax)
                c2y = cy + niny * (rmin + rmax)
                c2z = cz + ninz * (rmin + rmax)
                px = (
                    c2x
                    + rmax * np.cos(theta) * (-ninx)
                    + rmax * np.sin(theta) * emax[0]
                )
                py = (
                    c2y
                    + rmax * np.cos(theta) * (-niny)
                    + rmax * np.sin(theta) * emax[1]
                )
                pz = (
                    c2z
                    + rmax * np.cos(theta) * (-ninz)
                    + rmax * np.sin(theta) * emax[2]
                )

                lab = f'{k0}-majorR'

            dplot[k0]['r'] = {
                'x': px,
                'y': py,
                'z': pz,
                'r': np.hypot(px, py),
                'props': {
                    'label': lab,
                    'ls': '--',
                    'lw': 1.,
                    'color': color,
                },
            }

    return dplot


# ##################################################################
# ##################################################################
#                   Wavelength from angle
# ##################################################################


def get_lamb_from_angle(
    coll=None,
    key=None,
    key_cam=None,
    lamb=None,
    rocking_curve=None,
    units=None,
    returnas=None,
):
    """"""

    # ----------
    # check

    # key
    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # key_cam
    lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
    key_cam = ds._generic_check._check_var(
        key_cam, 'key_cam',
        types=str,
        allowed=lok,
    )

    # doptics
    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam]
    if 'crystal' not in doptics['cls']:
        raise Exception(f"Diag '{key}' is not a spectro!")

    kcryst = doptics['optics'][doptics['cls'].index('crystal')]

    dok = {
        'lamb': 'alpha',
        'lambmin': 'amin',
        'lambmax': 'amax',
        'dlamb': 'dlamb',
        'res': 'res',
    }
    lok = list(dok.keys())
    lamb = ds._generic_check._check_var(
        lamb, 'lamb',
        types=str,
        allowed=lok,
    )

    # ----------
    # compute

    lv = []
    data = None
    lk = ['lamb', 'lambmin', 'lambmax']
    for kk in lk:
        if lamb in [kk, 'dlamb', 'res']:

            if kk == 'lamb':
                klos = coll.dobj['diagnostic'][key]['doptics'][key_cam]['los']
                ka = coll.dobj['rays'][klos][dok[kk]]
                ang = coll.ddata[ka]['data'][0, ...]
                ref = coll.ddata[ka]['ref'][1:]
            else:
                ka = coll.dobj['diagnostic'][key]['doptics'][key_cam][dok[kk]]
                ang = coll.ddata[ka]['data']
                ref = coll.ddata[ka]['ref']

            dd = coll.get_crystal_bragglamb(
                key=kcryst,
                bragg=ang,
                rocking_curve=rocking_curve,
            )[1]
            if lamb == kk:
                data = dd
                break
            else:
                lv.append(dd)

    # ----------------
    # units conversion

    if units not in [None, 'm']:
        if data is None:

            for ii in range(3):
                lv[ii] = _spectralunits.convert_spectral(
                    data_in=lv[ii],
                    units_in='m',
                    units_out=units,
                )[0]

        else:
            data = _spectralunits.convert_spectral(
                data_in=data,
                units_in='m',
                units_out=units,
            )[0]

    # -----------
    # return

    if lamb == 'dlamb':
        data = np.abs(lv[2] - lv[1])
    elif lamb == 'res':
        data = lv[0] / np.abs(lv[2] - lv[1])

    return data, ref


# ##################################################################
# ##################################################################
#                   get data
# ##################################################################


def _get_data(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
    rocking_curve=None,
    units=None,
    default=None,
    **kwdargs,
):

    # key, key_cam
    key, key_cam = coll.get_diagnostic_cam(
        key=key,
        key_cam=key_cam,
        default=default,
    )
    spectro = coll.dobj['diagnostic'][key]['spectro']
    # is2d = coll.dobj['diagnostic'][key]['is2d']

    # basic check on data
    if data is not None:
        lquant = ['etendue', 'amin', 'amax']  # 'los'
        lcomp = ['length', 'tangency radius', 'alpha', 'alpha_pixel']
        if spectro:
            llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
            lvos = ['vos_lamb', 'vos_dlamb', 'vos_ph_integ']
        else:
            llamb = []
            lvos = ['vos_sang_integ']
        lsynth = coll.dobj['diagnostic'][key]['signal']

        if len(key_cam) == 1:
            lraw = [
                k0 for k0, v0 in coll.ddata.items()
                if v0['ref'] == coll.dobj['camera'][key_cam[0]]['dgeom']['ref']
            ]
        else:
            lraw = []

        if lsynth is None:
            lsynth = []
        lcomp += llamb

        data = ds._generic_check._check_var(
            data, 'data',
            types=str,
            allowed=lquant + lcomp + lsynth + lraw + lvos,
        )

    # build ddata
    ddata = {}
    static = True
    daxis = None

    # comp = False
    if data is None or data in lquant:

        # --------------------------
        # data is None => kwdargs

        if data is None:
            # check kwdargs
            dparam = coll.get_param(which='data', returnas=dict)
            lkout = [k0 for k0 in kwdargs.keys() if k0 not in dparam.keys()]

            if len(lkout) > 0:
                msg = (
                    "The following args correspond to no data parameter:\n"
                    + "\n".join([f"\t- {k0}" for k0 in lkout])
                )
                raise Exception(msg)

            # list all available data
            lok = [
                k0 for k0, v0 in coll.ddata.items()
                if v0.get('camera') in key_cam
            ]

            # Adjust with kwdargs
            if len(kwdargs) > 0:
                lok2 = coll.select(
                    which='data', log='all', returnas=str, **kwdargs,
                )
                lok = [k0 for k0 in lok2 if k0 in lok]

            # check there is 1 data per cam
            lcam = [
                coll.ddata[k0]['camera'] for k0 in lok
                if coll.ddata[k0]['camera'] in key_cam
            ]

            if len(set(lcam)) > len(key_cam):
                msg = (
                    "There are more / less data identified than cameras:\n"
                    f"\t- key_cam:  {key_cam}\n"
                    f"\t- data cam: {lcam}\n"
                    f"\t- data: {data}"
                )
                raise Exception(msg)

            elif len(set(lcam)) < len(key_cam):
                pass

            # reorder
            ddata = {
                cc: lok[lcam.index(cc)]
                for cc in key_cam if cc in lcam
            }

        # -----------------
        # data in lquant

        elif data in lquant:
            for cc in key_cam:
                # if data == 'los':
                #     kr = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                #     dd = coll.dobj['rays'][kr]['pts']
                # else:
                dd = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                lc = [
                    isinstance(dd, str) and dd in coll.ddata.keys(),
                    # isinstance(dd, tuple)
                    # and all([isinstance(di, str) for di in dd])
                    # and all([di in coll.ddata.keys() for di in dd])
                ]
                if lc[0]:
                    ddata[cc] = dd
                # elif lc[1]:
                #     ddata[cc] = list(dd)
                elif dd is None:
                    pass
                else:
                    msg = f"Unknown data: '{data}'"
                    raise Exception(msg)

        # dref
        dref = {
            k0: coll.ddata[v0]['ref']
            for k0, v0 in ddata.items()
        }

        # units
        if len(ddata) > 0:
            units = coll.ddata[ddata[key_cam[0]]]['units']
        else:
            units = None

        # get actual data
        ddata = {
            k0 : coll.ddata[v0]['data']
            for k0, v0 in ddata.items()
        }

    # --------------------
    # data to be computed

    elif data in lcomp:

        # comp = True
        ddata = {}
        dref = {}

        if data in llamb:
            for cc in key_cam:
               ddata[cc], dref[cc] = coll.get_diagnostic_lamb(
                   key=key,
                   key_cam=cc,
                   rocking_curve=rocking_curve,
                   lamb=data,
                   units=units,
               )
            if data in ['lamb', 'lambmin', 'lambmax', 'dlamb']:
                units = 'm'
            else:
                units = ''

        elif data in ['length', 'tangency radius', 'alpha']:
            for cc in key_cam:
                ddata[cc], _, dref[cc] = coll.get_rays_quantity(
                    key=key,
                    key_cam=cc,
                    quantity=data,
                    segment=-1,
                    lim_to_segments=False,
                )
            if data in ['length', 'tangency radius']:
                units = 'm'
            else:
                units = 'rad'

        elif data == 'alpha_pixel':
            for cc in key_cam:

                klos = coll.dobj['diagnostic'][key]['doptics'][cc]['los']
                vectx, vecty, vectz = coll.get_rays_vect(klos)
                dvect = coll.get_camera_unit_vectors(cc)
                sca = (
                    dvect['nin_x'] * vectx
                    + dvect['nin_y'] * vecty
                    + dvect['nin_z'] * vectz
                )

                ddata[cc] = np.arccos(sca)
                dref[cc] = coll.dobj['camera'][cc]['dgeom']['ref']
                units = 'rad'

    elif data in lsynth:

        dref = {}
        daxis = {}
        dsynth = coll.dobj['synth sig'][data]
        for cc in key_cam:
            kdat = dsynth['data'][dsynth['camera'].index(cc)]
            refcam = coll.dobj['camera'][cc]['dgeom']['ref']
            ref = coll.ddata[kdat]['ref']

            c0 = (
                tuple([rr for rr in ref if rr in refcam]) == refcam
                and len(ref) in [len(refcam), len(refcam) + 1]
            )
            if not c0:
                msg = (
                    "Can only plot data that is either:\n"
                    "\t- static: same refs as the camera\n"
                    "\t- has a unique extra dimension\n"
                    "Provided:\n"
                    "\t- refcam: {refcam}\n"
                    "\t- ['{kdat}']['ref']: {ref}"
                )
                raise Exception(msg)

            if len(ref) == len(refcam) + 1:
                static = False
                daxis[cc] = [
                    ii for ii, rr in enumerate(ref) if rr not in refcam
                ][0]

            ddata[cc] = coll.ddata[kdat]['data']
            dref[cc] = ref

            units = coll.ddata[kdat]['units']

    elif data in lraw:
        ddata = {key_cam[0]: coll.ddata[data]['data']}
        dref = {key_cam[0]: coll.dobj['camera'][key_cam[0]]['dgeom']['ref']}
        units = coll.ddata[data]['units']
        static = True

    elif data in lvos:

        static = True
        ddata, dref = {}, {}
        doptics = coll.dobj['diagnostic'][key]['doptics']
        for cc in key_cam:

            # safety check
            ref = coll.dobj['camera'][cc]['dgeom']['ref']
            dvos = doptics[cc].get('dvos')
            if dvos is None:
                msg = (
                    f"Data '{data}' cannot be retrived for diag '{key}' "
                    "cam '{cc}' because no dvos computed"
                )
                raise Exception(msg)

            # cases
            if data == 'vos_sang_integ':
                kdata = dvos['sang']
                ddata[cc] = np.nansum(coll.ddata[kdata]['data'], axis=-1)
                dref[cc] = ref
                units = coll.ddata[kdata]['units']

            elif data in ['vos_lamb', 'vos_dlamb', 'vos_ph_integ']:
                kph = dvos['ph']
                ph = coll.ddata[kph]['data']
                ph_tot = np.sum(ph, axis=(-1, -2))

                if data == 'vos_ph_integ':
                    out = ph_tot
                    kout = kph
                else:
                    kout = dvos['lamb']
                    re_lamb = [1 for rr in ref] + [1, -1]
                    lamb = coll.ddata[kout]['data'].reshape(re_lamb)

                    i0 = ph == 0
                    if data == 'vos_lamb':
                        out = np.sum(ph * lamb, axis=(-1, -2)) / ph_tot
                    else:
                        for ii, i1 in enumerate(re_lamb[:-1]):
                            lamb = np.repeat(lamb, ph.shape[ii], axis=ii)

                        lamb[i0] = -np.inf
                        lambmax = np.max(lamb, axis=(-1, -2))
                        lamb[i0] = np.inf
                        lambmin = np.min(lamb, axis=(-1, -2))
                        out = lambmax - lambmin
                    out[np.all(i0, axis=(-1, -2))] = np.nan

                ddata[cc] = out
                dref[cc] = ref
                units = coll.ddata[kout]['units']

    return ddata, dref, units, static, daxis


# ##################################################################
# ##################################################################
#                   concatenate data
# ##################################################################


# def _concatenate_check(
    # coll=None,
    # key=None,
    # key_cam=None,
    # data=None,
    # rocking_curve=None,
    # returnas=None,
    # # naming
    # key_data=None,
    # key_ref=None,
    # **kwdargs,
    # ):

    # # -------------
    # # key, key_cam
    # # -------------

    # key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    # spectro = coll.dobj['diagnostic'][key]['spectro']
    # is2d = coll.dobj['diagnostic'][key]['is2d']
    # # stack = coll.dobj['diagnostic'][key]['stack']

    # if is2d and len(key_cam) > 1:
        # msg = (
            # "Cannot yet concatenate several 2d cameras\n"
            # "\t- key: '{key}'\n"
            # "\t- is2d: {is2d}\n"
            # "\t- key_cam: {key_cam}\n"
        # )
        # raise NotImplementedError(msg)

    # # ---------------
    # # build ddata
    # # -------------

    # # basic check on data
    # if data is not None:
        # lquant = ['los', 'etendue', 'amin', 'amax']
        # lcomp = ['tangency radius']
        # if spectro:
            # lcomp += ['lamb', 'lambmin', 'lambmax', 'res']

        # data = ds._generic_check._check_var(
            # data, 'data',
            # types=str,
            # allowed=lquant + lcomp,
        # )

    # # build ddata
    # ddata = {}
    # comp = False
    # if data is None or data in lquant:

        # # --------------------------
        # # data is None => kwdargs

        # if data is None:
            # # check kwdargs
            # dparam = coll.get_param(which='data', returnas=dict)
            # lkout = [k0 for k0 in kwdargs.keys() if k0 not in dparam.keys()]

            # if len(lkout) > 0:
                # msg= (
                    # "The following args correspond to no data parameter:\n"
                    # + "\n".join([f"\t- {k0}" for k0 in lkout])
                # )
                # raise Exception(msg)

            # # list all available data
            # lok = [
                # k0 for k0, v0 in coll.ddata.items()
                # if v0.get('camera') in key_cam
            # ]

            # # Adjust with kwdargs
            # if len(kwdargs) > 0:
                # lok2 = coll.select(
                    # which='data', log='all', returnas=str, **kwdargs,
                # )
                # lok = [k0 for k0 in lok2 if k0 in lok]

            # # check there is 1 data per cam
            # lcam = [
                # coll.ddata[k0]['camera'] for k0 in lok
                # if coll.ddata[k0]['camera'] in key_cam
            # ]

            # if len(set(lcam)) > len(key_cam):
                # msg = (
                    # "There are more / less data identified than cameras:\n"
                    # f"\t- key_cam:  {key_cam}\n"
                    # f"\t- data cam: {lcam}\n"
                    # f"\t- data: {data}"
                # )
                # raise Exception(msg)
            # elif len(set(lcam)) < len(key_cam):
                # pass

            # # reorder
            # ddata = {
                # cc: [lok[lcam.index(cc)]]
                # for cc in key_cam if cc in lcam
            # }

        # # -----------------
        # # data in lquant

        # elif data in lquant:
            # for cc in key_cam:
                # if data == 'los':
                    # kr = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                    # dd = coll.dobj['rays'][kr]['pts']
                # else:
                    # dd = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                # lc = [
                    # isinstance(dd, str) and dd in coll.ddata.keys(),
                    # isinstance(dd, tuple)
                    # and all([isinstance(di, str) for di in dd])
                    # and all([di in coll.ddata.keys() for di in dd])
                # ]
                # if lc[0]:
                    # ddata[cc] = [dd]
                # elif lc[1]:
                    # ddata[cc] = list(dd)
                # elif dd is None:
                    # pass
                # else:
                    # msg = f"Unknown data: '{data}'"
                    # raise Exception(msg)

        # # dref
        # dref = {
            # k0: [coll.ddata[k1]['ref'] for k1 in v0]
            # for k0, v0 in ddata.items()
        # }

    # # --------------------
    # # data to be computed

    # # TBF
    # elif data in lcomp:

        # comp = True
        # ddata = {[None] for cc in key_cam}
        # dref = {[None] for cc in key_cam}

        # if data in ['lamb', 'lambmin', 'lambmax', 'res']:
            # for cc in key_cam:
               # ddata[cc][0], dref[cc][0] = coll.get_diagnostic_lamb(
                   # key=key,
                   # key_cam=cc,
                   # rocking_curve=rocking_curve,
                   # lamb=data,
               # )

        # elif data == 'tangency radius':
            # ddata[cc][0], _, dref[cc][0] = coll.get_rays_tangency_radius(
                # key=key,
                # key_cam=key_cam,
                # segment=-1,
                # lim_to_segments=False,
            # )

    # # -----------------------------------
    # # Final safety checks and adjustments
    # # -----------------------------------

    # # adjust key_cam
    # key_cam = [cc for cc in key_cam if cc in ddata.keys()]

    # # ddata vs dref vs key_cam
    # lcd = sorted(list(ddata.keys()))
    # lcr = sorted(list(dref.keys()))
    # if not (sorted(key_cam) == lcd == lcr):
        # msg = (
            # "Wrong keys!\n"
            # f"\t- key_cam: {key_cam}\n"
            # f"\t- ddata.keys(): {lcd}\n"
            # f"\t- dref.keys(): {lcr}\n"
        # )
        # raise Exception(msg)

    # # nb of data per cam
    # ln = [len(v0) for v0 in ddata.values()]
    # if len(set(ln)) != 1:
        # msg = (
            # "Not the same number of data per cameras!\n"
            # + str(ddata)
        # )
        # raise Exception(msg)

    # # check shapes and ndim
    # dshapes = {
        # k0: [tuple([coll.dref[k2]['size'] for k2 in k1]) for k1 in v0]
        # for k0, v0 in dref.items()
    # }

    # # all same ndim
    # ndimref = None
    # for k0, v0 in dshapes.items():
        # lndim = [len(v1) for v1 in v0]
        # if len(set(lndim)) > 1:
            # msg = "All data must have same number of dimensions!\n{dshapes}"
            # raise Exception(msg)
        # if ndimref is None:
            # ndimref = lndim[0]
        # elif lndim[0] != ndimref:
            # msg = "All data must have same number of dimensions!\n{dshapes}"
            # raise Exception(msg)

    # # check indices of camera ref in data ref
    # indref = None
    # for k0, v0 in dref.items():
        # for v1 in v0:
            # ind = [v1.index(rr) for rr in coll.dobj['camera'][k0]['dgeom']['ref']]
            # if indref is None:
                # indref = ind
            # elif ind != indref:
                # msg = "All data must have same index of cam ref!\n{drf}"
                # raise Exception(msg)

    # if len(indref) > 1:
        # msg = "Cannot conatenate 2d cameras so far"
        # raise Exception(msg)

    # # check all shapes other than camera shapes are identical
    # if ndimref > len(indref):
        # ind = np.delete(np.arange(0, ndimref), indref)
        # shape0 = tuple(np.r_[dshapes[key_cam[0]][0]][ind])
        # lcout = [
            # cc for cc in key_cam
            # if any([tuple(np.r_[vv][ind]) != shape0 for vv in dshapes[cc]])
        # ]
        # if len(lcout) > 0:
            # msg = (
                # "The cameras data shall all have same shape (except pixels)\n"
                # + str(dshapes)
            # )
            # raise Exception(msg)

    # # check indices of camera ref in data ref
    # ref = None
    # for k0, v0 in dref.items():
        # for v1 in v0:
            # if ref is None:
                # ref = [
                    # None if ii == indref[0] else rr
                    # for ii, rr in enumerate(v1)
                # ]
            # else:
                # lc = [
                    # v1[ii] == ref[ii] for ii in range(ndimref)
                    # if ii not in indref
                # ]
                # if not all(lc):
                    # msg = (
                        # "All ref axcept the camera ref must be the same!\n"
                        # f"\t- ref: {ref}\n"
                        # f"\t- indref: {indref}\n"
                        # f"\t- ndimref: {ndimref}\n"
                        # f"\t- v1: {v1}\n"
                        # f"\t- lc: {lc}\n"
                        # + str(dref)
                    # )
                    # raise Exception(msg)

    # # -----------------------------------
    # # keys for new data and ref
    # # -----------------------------------

    # if key_data is None:
        # if data in lquant + lcomp:
            # if data == 'los':
                # key_data = [
                    # f'{key}_los_ptsx',
                    # f'{key}_los_ptsy',
                    # f'{key}_los_ptsz',
                # ]
            # else:
                # key_data = [f'{key}_{data}']
        # else:
            # key_data = [f'{key}_data']
    # elif isinstance(key_data, str):
        # key_data = [key_data]

    # if key_ref is None:
        # key_ref = f'{key}_npix'

    # ref = tuple([key_ref if rr is None else rr for rr in ref])

    # # -----------------------------------
    # # Other variables
    # # -----------------------------------

    # # returnas
    # returnas = ds._generic_check._check_var(
        # returnas, 'returnas',
        # default='Datastock',
        # allowed=[dict, 'Datastock'],
    # )

    # return (
        # key, key_cam, is2d,
        # ddata, ref, comp,
        # dshapes, ndimref, indref,
        # key_data, key_ref,
        # returnas,
    # )


def _concatenate_data_check(
    coll=None,
    key=None,
    key_data=None,
    flat=None,
):

    # ------------
    # key

    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=None)
    spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']
    stack = coll.dobj['diagnostic'][key]['stack']

    # ------------
    # key_data

    # key_data
    if isinstance(key_data, str):

        if key_data in coll.ddata.keys():
            key_data = [key_data]

        else:
            lok = coll.dobj['diagnostic'][key]['signal']
            if lok is None:
                lok = []
            key_data = ds._generic_check._check_var(
                key_data, 'key_data',
                types=str,
                allowed=lok,
            )

            key_data = coll.dobj['synth sig'][key_data]['data']

    # basic check
    c0 = (
        isinstance(key_data, list)
        and all([
            isinstance(kk, str) and kk in coll.ddata.keys()
            for kk in key_data
        ])
    )
    if not c0:
        msg = (
            "Arg key_data must be a list of valid data keys!\n"
            f"Provided: {key_data}"
        )
        raise Exception(msg)

    # check cameras
    dout = {
        k0: coll.ddata[k0].get('camera')
        for k0 in key_data
        if coll.ddata[k0].get('camera') not in key_cam
    }
    if len(dout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            f"The following data refers to no known camera in diag '{key}:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # check unicity of cameras
    lcam = [coll.ddata[k0]['camera'] for k0 in key_data]
    if len(set(lcam)) > len(lcam):
        msg = (
            f"Non-unique camera references for diag '{key}'\n:"
            f"\t- key_data = {key_data}\n"
            f"\t- lcam     = {lcam}\n"
        )
        raise Exception(msg)

    # re-order
    lk = []
    for k0 in coll.dobj['diagnostic'][key]['camera']:
        if k0 in lcam:
            lk.append(key_data[lcam.index(k0)])

    key_data = lk
    lcam = [coll.ddata[k0]['camera'] for k0 in key_data]

    # ref uniformity
    lref = [coll.ddata[k0]['ref'] for k0 in key_data]
    if any([len(ref) != len(lref[0]) for ref in lref[1:]]):
        msg = (
            f"Non uniform refs for data in diag '{key}':\n"
            f"\t- key_data = {key_data}\n"
            f"\t- lref = {lref}\n"
        )
        raise Exception(msg)

    # ref axis
    laxcam = [
        [
            ref.index(rr)
            for rr in coll.dobj['camera'][lcam[ii]]['dgeom']['ref']
        ]
        for ii, ref in enumerate(lref)
    ]
    if any([len(ax) != len(laxcam[0]) for ax in laxcam[1:]]):
        msg = (
            f"Non-uniform camera concatenation axis for diag '{key}':\n"
            f"\t- key_data: {key_data}\n"
            f"\t- laxcam:   {laxcam}"
        )
        import pdb; pdb.set_trace()     # DB
        raise Exception(msg)

    laxcam = np.array(laxcam)
    if not np.allclose(laxcam, laxcam[0:1, :]):
        msg = (
            f"Non-uniform axis for concatenation for diag '{key}':\n"
            f"\t- laxcam: {laxcam}"
        )
        raise Exception(msg)
    axcam = laxcam[0]

    if len(axcam) != 1 + is2d:
        msg = (
            f"ref not consistent with is2d for diag '{key}':\n"
            f"\t- is2d: {is2d}\n"
            f"\t- axcam: {axcam}\n"
        )
        raise Exception(msg)

    ref = list(lref[0])
    for ii in axcam:
        ref[ii] = None

    # ------
    # flat

    flat = ds._generic_check._check_var(
        flat, 'flat',
        types=bool,
        default=is2d,
    )

    return key, key_data, key_cam, is2d, stack, ref, flat


def _concatenate_data(
    coll=None,
    key=None,
    key_data=None,
    flat=None,
):

    # ------------
    # check inputs

    key, key_data, key_cam, is2d, stack, ref, flat = _concatenate_data_check(
        coll=coll,
        key=key,
        key_data=key_data,
        flat=flat,
    )

    # ------------
    # prepare

    if is2d:
        ax0 = ref.index(None)
        ax1 = len(ref) - 1 - ref[::-1].index(None)
        if flat:
            ldata = []
            for k0 in key_data:
                sh = list(coll.ddata[k0]['data'].shape)
                size = sh[ax0] * sh[ax1]
                sh = tuple(np.r_[sh[:ax0], size, sh[ax1+1:]].astype(int))
                ldata.append(coll.ddata[k0]['data'].reshape(sh))
            axis = ax0
        else:
            axis = ax0 if stack == 'horizontal' else ax1
            ldata = [coll.ddata[k0]['data'] for k0 in key_data]
            lsize = [dd.shape[axis] for dd in ldata]
            if len(set(lsize)) != 1:
                msg = (
                    "Data for diag '{key}' cannot be stacked {stack}:\n"
                    f"\t- shapes: {[dd.shape for dd in ldata]}\n"
                    f"\t- axis: {axis}"
                )
                raise Exception(msg)

    else:
        ldata = [coll.ddata[k0]['data'] for k0 in key_data]
        axis = ref.index(None)

    # ------------
    # concatenate

    data = np.concatenate(tuple(ldata), axis=axis)
    units = coll.ddata[key_data[0]]['units']

    # dind
    i0 = 0
    dind = {}
    for ii, k0 in enumerate(key_data):
        npix = ldata[ii].shape[axis]
        ind = i0 + np.arange(0, npix)
        dind[k0] = ind
        i0 += npix

    return {
        'data': data,
        'keys': key_data,
        'keys_cam': key_cam,
        'units': units,
        'ref': ref,
        'axis': axis,
        'flat': flat,
        'dind': dind,
    }
