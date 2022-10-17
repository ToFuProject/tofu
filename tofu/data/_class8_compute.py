# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp


import datastock as ds


from. import _utils_surface3d


# ##################################################################
# ##################################################################
#                   optics outline
# ##################################################################


def get_optics_outline(
    coll=None,
    key=None,
    add_points=None,
    mode=None,
    closed=None,
    ravel=None,
    total=None,
):

    # ------------
    # check inputs

    # key, cls
    key, cls = coll.get_diagnostic_optics(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    # total
    total = ds._generic_check._check_var(
        total, 'total',
        types=bool,
        default=(cls == 'camera' and dgeom['type'] == '2d'),
    )
    if cls == 'camera' and dgeom['type'] != '2d':
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

        # derive half-spacing
        dx0 = np.mean(np.diff(cx0)) / 2.
        dx1 = np.mean(np.diff(cx1)) / 2.

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

    return _interp_poly(
        lp=[p0, p1],
        add_points=add_points,
        mode=mode,
        isclosed=False,
        closed=closed,
        ravel=ravel,
    )


# ##################################################################
# ##################################################################
#                   optics poly
# ##################################################################


def get_optics_poly(
    coll=None,
    key=None,
    add_points=None,
    mode=None,
    closed=None,
    ravel=None,
    total=None,
    return_outline=None,
):

    # ------------
    # check inputs

    key, cls = coll.get_diagnostic_optics(optics=key)
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

        if dgeom['type'] == '2d' and total:
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
#                   Poly interpolation utilities
# ##################################################################


def _interp_poly_check(
    add_points=None,
    mode=None,
    closed=None,
    ravel=None,
):

    # -------
    # mode

    mode = ds._generic_check._check_var(
        mode, 'mode',
        default=None,
        allowed=[None, 'min'],
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
    return add_points, mode, closed, ravel


def _interp_poly(
    lp=None,
    add_points=None,
    mode=None,
    isclosed=None,
    closed=None,
    ravel=None,
    min_threshold=1.e-6,
):

    # ------------
    # check inputs

    add_points, mode, closed, ravel = _interp_poly_check(
        add_points=add_points,
        mode=mode,
        closed=closed,
        ravel=ravel,
    )

    # ------------
    # trivial case

    if add_points == 0:
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

    if mode == 'min':
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

        min_threshold = min(min_threshold, np.max(dist)/3.)
        mindist = np.min(dist[dist > min_threshold])
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
        nan = np.full((px.shape[0], 1), np.nan)
        for ii, pp in enumerate(lp[2:]):
            lp[ii+2] = np.concatenate((pp, nan), axis=1).ravel()
    return lp


# ##################################################################
# ##################################################################
#                       dplot
# ##################################################################


def _dplot_check(
    coll=None,
    key=None,
    optics=None,
    elements=None,
    vect_length=None,
    axis_length=None,
):
    # -----
    # key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        allowed=lok,
    )

    # ------
    # optics

    if isinstance(optics, str):
        optics = [optics]

    lok = coll.dobj['diagnostic'][key]['optics']
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

    return key, optics, elements, vect_length, axis_length


def _dplot(
    coll=None,
    key=None,
    optics=None,
    elements=None,
    vect_length=None,
    axis_length=None,
):

    # ------------
    # check inputs

    key, optics, elements, vect_length, axis_length = _dplot_check(
        coll=coll,
        key=key,
        optics=optics,
        elements=elements,
        vect_length=vect_length,
        axis_length=axis_length,
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
        if 'c' in elements or 'v' in elementsi or 'r' in elements:
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
        if 'r' in elements and v0['type'] not in ['planar', '1d', '2d', '3d']:
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
            extenthalf = v0['extenthalf']

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
                'x0': p0,
                'x1': p1,
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
    lamb=None,
    rocking_curve=None,
):
    """"""

    # ----------
    # check

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    optics, optics_cls = coll.get_diagnostic_optics(key)
    if 'crystal' not in optics_cls:
        raise Exception(f"Diag '{key}' is not a spectro!")

    kcryst = optics[optics_cls.index('crystal')]

    dok = {
        'lamb': 'alpha',
        'lambmin': 'amin',
        'lambmax': 'amax',
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
    lk = ['lamb', 'lambmin', 'lambmax']
    for kk in lk:
        if lamb in [kk, 'res']:
            if kk == 'lamb':
                klos = coll.dobj['diagnostic'][key]['los']
                ka = coll.dobj['rays'][klos][dok[kk]]
                ang = coll.ddata[ka]['data'][0, ...]
                ref = coll.ddata[ka]['ref'][1:]
            else:
                ka = coll.dobj['diagnostic'][key][dok[kk]]
                ang = coll.ddata[ka]['data']
                ref = coll.ddata[ka]['ref']
            dd = coll.get_crystal_bragglamb(
                key=kcryst,
                bragg=ang,
                rocking_curve=rocking_curve,
            )[1]
            if lamb == kk:
                data = dd
            else:
                lv.append(dd)

    if lamb == 'res':
        data = lv[0] / (lv[2] - lv[1])

    return data, ref
