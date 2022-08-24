# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp


import datastock as ds


from ..geom import _etendue


# #############################################################################
# #############################################################################
#                   optics outline
# #############################################################################


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
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    lcam = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lap + lcryst + lgrat + lcam,
    )

    if key in lap:
        cls = 'aperture'
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

    if cls in ['aperture', 'crystal', 'grating']:
        px, py, pz = coll.dobj[cls][key]['dgeom']['poly']
        px = coll.ddata[px]['data']
        py = coll.ddata[py]['data']
        pz = coll.ddata[pz]['data']

        if coll.dobj[cls][key]['dgeom']['type'] == 'planar':
            p0, p1 = coll.dobj[cls][key]['dgeom']['outline']
            p0 = coll.ddata[p0]['data']
            p1 = coll.ddata[p1]['data']
        else:
            p0, p1 = None, None

    elif cls == 'camera':
        is2d = coll.dobj['camera'][key]['type'] == '2d'
        parallel = coll.dobj['camera'][key]['parallel'] is True

        if parallel:
            e0 = coll.dobj['camera'][key]['e0']
            e1 = coll.dobj['camera'][key]['e1']

            if is2d:
                # get centers
                cx0, cx1 = coll.dobj['camera'][key]['cents']
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
                cx, cy, cz = coll.dobj['camera'][key]['cent']
                px = cx + p0 * e0[0] + p1 * e1[0]
                py = cy + p0 * e0[1] + p1 * e1[1]
                pz = cz + p0 * e0[2] + p1 * e1[2]

            else:
                # get centers
                cx, cy, cz = coll.dobj['camera'][key]['cents']
                cx = coll.ddata[cx]['data']
                cy = coll.ddata[cy]['data']
                cz = coll.ddata[cz]['data']

                # get outline 2d
                p0, p1 = coll.dobj['camera'][key]['outline']
                p0 = coll.ddata[p0]['data']
                p1 = coll.ddata[p1]['data']

                # make 3d
                px = cx[:, None] + p0[None, :] * e0[0] + p1[None, :] * e1[0]
                py = cy[:, None] + p0[None, :] * e0[1] + p1[None, :] * e1[1]
                pz = cz[:, None] + p0[None, :] * e0[2] + p1[None, :] * e1[2]

        else:
            # unit vectors
            e0x, e0y, e0z = coll.dobj['camera'][key]['e0']
            e1x, e1y, e1z = coll.dobj['camera'][key]['e1']
            e0x = coll.ddata[e0x]['data'][:, None]
            e0y = coll.ddata[e0y]['data'][:, None]
            e0z = coll.ddata[e0z]['data'][:, None]
            e1x = coll.ddata[e1x]['data'][:, None]
            e1y = coll.ddata[e1y]['data'][:, None]
            e1z = coll.ddata[e1z]['data'][:, None]

            # get centers
            cx, cy, cz = coll.dobj['camera'][key]['cents']
            cx = coll.ddata[cx]['data']
            cy = coll.ddata[cy]['data']
            cz = coll.ddata[cz]['data']

            # get outline 2d
            out0, out1 = coll.dobj['camera'][key]['outline']
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


# #############################################################################
# #############################################################################
#                       dplot
# #############################################################################


def _dplot_check(
    coll=None,
    key=None,
    optics=None,
    elements=None,
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

    if isinstance(elements, str):
        elements = [elements]

    lok = ''.join(['o', 'v', 'c', 'r'])
    elements = ds._generic_check._check_var_iter(
        elements, 'elements',
        types=str,
        default=lok,
        allowed=lok,
    )

    return key, optics, elements


def _dplot(
    coll=None,
    key=None,
    optics=None,
    elements=None,
    vect_length=None,
):

    # ------------
    # check inputs

    key, optics, elements = _dplot_check(
        coll=coll,
        key=key,
        optics=optics,
        elements=elements,
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
        elif k0 in coll.dobj.get('crystal', []):
            cls = 'crystal'
        else:
            msg = f"Unknown optics '{k0}'"
            raise Exception(msg)

        v0 = coll.dobj[cls][k0]

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
                    'c': 'k',
                },
            })

        # center
        if 'c' in elements:

            if v0.get('cent') is not None:
                cx, cy, cz = v0['cent'][:, None]
            elif 'cents' in v0.keys():
                cx, cy, cz = v0['cents']
                cx = coll.ddata[cx]['data']
                cy = coll.ddata[cy]['data']
                cz = coll.ddata[cz]['data']

            dplot[k0]['c'] = {
                'x': cx,
                'y': cy,
                'z': cz,
                'r': np.hypot(cx, cy),
                'props': {
                    'label': f'{k0}-o',
                    'ls': 'None',
                    'marker': 'o',
                    'ms': 4,
                    'c': 'k',
                },
            }

        # unit vectors
        if 'v' in elements:

            pass

        # rowland / axis for curved optics
        if 'r' in elements and cls in ['crystal', 'grating']:
            pass

    return dplot


# #############################################################################
# #############################################################################
#                       Etendue
# #############################################################################


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

    return key, analytical, numerical, store


def _diag_compute_etendue(
    coll=None,
    key=None,
    analytical=None,
    numerical=None,
    res=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
):

    # ------------
    # check inputs

    key, analytical, numerical, store = _diag_compute_etendue_check(
        coll=coll,
        key=key,
        analytical=analytical,
        numerical=numerical,
        store=store,
    )

    # prepare optics
    optics = coll.dobj['diagnostic'][key]['optics']
    key_cam = optics[0]

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

        # data
        etendue = detend[store][-1, :]

        if store == 'analytical':
            etend_type = store
        else:
            etend_type = res[-1]

        # dict for etendue
        ketendue = f'{key}-etend'

        ddata = {
            ketendue: {
                'data': etendue,
                'ref': coll.dobj['camera'][key_cam]['ref'],
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

    return detend
