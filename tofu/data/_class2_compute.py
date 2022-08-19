

import numpy as np
import scipy.interpolate as scpinterp


import datastock as ds


from ..geom import _etendue


# #############################################################################
# #############################################################################
#                       outlines
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
    lcam = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lap + lcam,
    )

    if key in lap:
        cls = 'aperture'
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
        px, py, pz = coll.dobj[cls][key]['poly']
        px = coll.ddata[px]['data']
        py = coll.ddata[py]['data']
        pz = coll.ddata[pz]['data']

        if coll.dobj[cls][key]['type'] == 'planar':
            p0, p1 = coll.dobj['aperture'][key]['outline']
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
