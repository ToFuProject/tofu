# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp


import datastock as ds


from ..geom import _etendue


# ##################################################################
# ##################################################################
#                   optics outline
# ##################################################################


def _get_optics_outline_check(
    coll=None,
    key=None,
    add_points=None,
    closed=None,
    ravel=None,
):

    # -------
    # key

    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    lcam = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lap + lfilt + lcryst + lgrat + lcam,
    )

    if key in lap:
        cls = 'aperture'
    elif key in lfilt:
        cls = 'filter'
    elif key in lcryst:
        cls = 'crystal'
    elif key in lgrat:
        cls = 'grating'
    elif key in lcam:
        cls = 'camera'

    # ----------
    # add_points

    if add_points is None:
        add_points = False
    if add_points is False:
        add_points = 0

    add_points = ds._generic_check._check_var(
        add_points, 'add_points',
        types=int,
    )

    if add_points < 0:
        msg = f"Arg add_points must be positive!\nProvided: {add_points}"
        raise Exception(msg)

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

    return key, cls, add_points, closed, ravel


def get_optics_outline(
    coll=None,
    key=None,
    add_points=None,
    closed=None,
    ravel=None,
):

    # ------------
    # check inputs

    key, cls, add_points, closed, ravel = _get_optics_outline_check(
        coll=coll,
        key=key,
        add_points=add_points,
        closed=closed,
        ravel=ravel,
    )

    # --------
    # compute

    dgeom = coll.dobj[cls][key]['dgeom']
    if cls in ['aperture', 'filter', 'crystal', 'grating']:
        px, py, pz = dgeom['poly']
        px = coll.ddata[px]['data']
        py = coll.ddata[py]['data']
        pz = coll.ddata[pz]['data']

        if dgeom['type'] == 'planar':
            p0, p1 = dgeom['outline']
            p0 = coll.ddata[p0]['data']
            p1 = coll.ddata[p1]['data']
        else:
            p0, p1 = None, None

    elif cls == 'camera':

        if dgeom['parallel'] is True:
            e0 = dgeom['e0']
            e1 = dgeom['e1']

            if dgeom['type'] == '2d':
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

                # convert to 3d
                cx, cy, cz = dgeom['cent']
                px = cx + p0 * e0[0] + p1 * e1[0]
                py = cy + p0 * e0[1] + p1 * e1[1]
                pz = cz + p0 * e0[2] + p1 * e1[2]

            else:
                # get centers
                cx, cy, cz = dgeom['cents']
                cx = coll.ddata[cx]['data']
                cy = coll.ddata[cy]['data']
                cz = coll.ddata[cz]['data']

                # get outline 2d
                p0, p1 = dgeom['outline']
                p0 = coll.ddata[p0]['data']
                p1 = coll.ddata[p1]['data']

                # make 3d
                px = cx[:, None] + p0[None, :] * e0[0] + p1[None, :] * e1[0]
                py = cy[:, None] + p0[None, :] * e0[1] + p1[None, :] * e1[1]
                pz = cz[:, None] + p0[None, :] * e0[2] + p1[None, :] * e1[2]

        else:
            # unit vectors
            e0x, e0y, e0z = dgeom['e0']
            e1x, e1y, e1z = dgeom['e1']
            e0x = coll.ddata[e0x]['data'][:, None]
            e0y = coll.ddata[e0y]['data'][:, None]
            e0z = coll.ddata[e0z]['data'][:, None]
            e1x = coll.ddata[e1x]['data'][:, None]
            e1y = coll.ddata[e1y]['data'][:, None]
            e1z = coll.ddata[e1z]['data'][:, None]

            # get centers
            cx, cy, cz = dgeom['cents']
            cx = coll.ddata[cx]['data']
            cy = coll.ddata[cy]['data']
            cz = coll.ddata[cz]['data']

            # get outline 2d
            out0, out1 = dgeom['outline']
            p0 = coll.ddata[out0]['data']
            p1 = coll.ddata[out1]['data']

            # make 3d
            px = cx[:, None] + p0[None, :] * e0x + p1[None, :] * e1x
            py = cy[:, None] + p0[None, :] * e0y + p1[None, :] * e1y
            pz = cz[:, None] + p0[None, :] * e0z + p1[None, :] * e1z

    # ------------
    # closed

    if closed is True:
        if p0 is not None:
            p0 = np.append(p0, p0[0])
            p1 = np.append(p1, p1[0])

        if px.ndim == 2:
            px = np.concatenate((px, px[:, 0:1]), axis=1)
            py = np.concatenate((py, py[:, 0:1]), axis=1)
            pz = np.concatenate((pz, pz[:, 0:1]), axis=1)
        else:
            px = np.append(px, px[0])
            py = np.append(py, py[0])
            pz = np.append(pz, pz[0])

    # -----------
    # add_points

    if add_points is not False:

        nb = px.shape[-1]
        ind0 = np.arange(0, nb)
        ind = np.linspace(0, nb-1, (nb - 1)*(1 + add_points) + 1)

        if p0 is not None:
            p0 = scpinterp.interp1d(ind0, p0, kind='linear')(ind)
            p1 = scpinterp.interp1d(ind0, p1, kind='linear')(ind)

        px = scpinterp.interp1d(ind0, px, kind='linear', axis=-1)(ind)
        py = scpinterp.interp1d(ind0, py, kind='linear', axis=-1)(ind)
        pz = scpinterp.interp1d(ind0, pz, kind='linear', axis=-1)(ind)

    # ------------------
    # ravel

    if ravel and px.ndim == 2:
        nan = np.full((px.shape[0], 1), np.nan)
        px = np.concatenate((px, nan), axis=1).ravel()
        py = np.concatenate((py, nan), axis=1).ravel()
        pz = np.concatenate((pz, nan), axis=1).ravel()

    return {
        'x0': p0,
        'x1': p1,
        'x': px,
        'y': py,
        'z': pz,
    }


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

            dplot[k0]['o'] = coll.get_optics_outline(
                key=k0,
                add_points=3,
                closed=True,
                ravel=True,
            )

            dplot[k0]['o'].update({
                'r': np.hypot(dplot[k0]['o']['x'], dplot[k0]['o']['y']),
                'props': {
                    'label': f'{k0}-o',
                    'lw': dlw[cls],
                    'c': color,
                },
            })

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
#                       Etendue
# ##################################################################


def _diag_compute_etendue_check(
    coll=None,
    key=None,
    analytical=None,
    numerical=None,
    store=None,
):

    # --------
    # key

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if len(v0['optics']) > 1
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    optics = coll.dobj['diagnostics'][key]['optics']
    lspectro = [
        oo for oo in optics
        if oo in coll.dobj.get('cryst', {})
        or oo in in coll.dobj.get('grating', {})
    ]

    # -----------
    # analytical

    analytical = ds._generic_check._check_var(
        analytical, 'analytical',
        types=bool,
        default=True,
    )

    # -----------
    # numerical

    numerical = ds._generic_check._check_var(
        numerical, 'numerical',
        types=bool,
        default=False,
    )

    # -----------
    # store

    lok = [False]
    if analytical is True:
        lok.append('analytical')
    if numerical is True:
        lok.append('numerical')
    store = ds._generic_check._check_var(
        store, 'store',
        default=lok[-1],
        allowed=lok,
    )

    return (
        key,
        optics,
        lspectro,
        analytical,
        numerical,
        store,
    )


def _diag_compute_etendue_los(
    coll=None,
    key=None,
    # parameters
    analytical=None,
    numerical=None,
    res=None,
    check=None,
    # for storing los
    config=None,
    length=None,
    reflections_nb=None,
    reflections_type=None,
    # bool
    verb=None,
    plot=None,
    store=None,
):

    # ------------
    # check inputs

    (
        key,
        optics,
        lspectro,
        analytical,
        numerical,
        store,
    ) = _diag_compute_etendue_check(
        coll=coll,
        key=key,
        analytical=analytical,
        numerical=numerical,
        store=store,
    )

    # prepare optics
    key_cam = optics[0]

    # ------------------------------------
    # compute equivalent optics if spectro

    if len(lspectro) == 1:
        _diag_spectro_equivalent_apertures(
        )

    elif len(lspectro) > 1:
        raise NotImplementedError()

    # --------
    # etendues

    detend = _etendue.compute_etendue(
        det=coll.get_as_dict(key=key_cam)[key_cam],
        aperture=coll.get_as_dict(key=optics[1:]),
        analytical=analytical,
        numerical=numerical,
        res=res,
        margin_par=None,
        margin_perp=None,
        check=check,
        verb=verb,
        plot=plot,
    )

    # ----------
    # store

    if store is not False:

        # ref
        ref = coll.dobj['camera'][key_cam]['dgeom']['ref']

        # data
        etendue = detend[store][-1, :]

        if store == 'analytical':
            etend_type = store
        else:
            etend_type = res[-1]

        # keys
        ketendue = f'{key}-etend'
        klos = f'{key}-los'

        ddata = {
            ketendue: {
                'data': etendue,
                'ref': ref,
                'dim': 'etendue',
                'quant': 'etendue',
                'name': 'etendue',
                'units': 'm2.sr'
            },
        }
        coll.update(ddata=ddata)

        coll.set_param(
            which='diagnostic',
            key=key,
            param='etendue',
            value=ketendue,
        )
        coll.set_param(
            which='diagnostic',
            key=key,
            param='etend_type',
            value=etend_type,
        )
        coll.set_param(
            which='diagnostic',
            key=key,
            param='los',
            value=klos,
        )

        # add los
        cx, cy, cz = coll.get_camera_cents_xyz(key=key_cam)

        coll.add_rays(
            key=klos,
            start_x=cx,
            start_y=cy,
            start_z=cz,
            vect_x=detend['los_x'],
            vect_y=detend['los_y'],
            vect_z=detend['los_z'],
            ref=ref,
            config=config,
            length=length,
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
        )

    return detend
