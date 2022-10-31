# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
    key, cls = coll.get_optics_cls(optics=key)
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
        nan = np.full((pp.shape[0], 1), np.nan)
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
    key_cam=None,
    lamb=None,
    rocking_curve=None,
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
            else:
                lv.append(dd)

    if lamb == 'res':
        data = lv[0] / (lv[2] - lv[1])

    return data, ref


# ##################################################################
# ##################################################################
#                   concatenate data
# ##################################################################

# TBF
def _concatenate_cam(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
    rocking_curve=None,
    returnas=None,
    **kwdargs,
    ):
    
    # ------------
    # key, key_cam
    
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']
    stack = coll.dobj['diagnostic'][key]['stack']
    
    lquant = ['etendue', 'amin', 'amax']
    
    # ---------------
    # data vs kwdargs
    
    if data is None:
    
        dparam = coll.get_param(which='data', returnas=dict)
        lkout = [k0 for k0 in kwdargs.keys() if k0 not in dparam.keys()]
        
        if len(lkout) > 0:
            msg= (
                "The following args correspond to no data parameter:\n"
                + "\n".join([f"\t- {k0}" for k0 in lkout])
            )
            raise Exception(msg)
            
    else:
        
        lok = lquant + ['tangency radius']
        if spectro:
            lok += ['lamb', 'lambmin', 'lambmax', 'res']
    
        data = ds._generic_check._check_var(
            data, 'data',
            types=str,
            allowed=lok,
        )
    
    # -----------
    # ldata, lref
    
    if data is None or data in lquant:

        if data is None:
            # list all available data
            lok = [
                k0 for k0, v0 in coll.ddata.items()
                if v0.get('camera') in key_cam
            ]
            
            if len(kwdargs) > 0:
                lok2 = coll.select(
                    which='data', log='all', returnas=str, **kwdargs,
                )
                lok = [k0 for k0 in lok2 if k0 in lok]
                
            # check there is one data per cam
            lcam = [coll.ddata[k0]['camera'] for k0 in lok]
            if len(set(lcam)) != len(key_cam):
                msg = (
                    "There are more / less data identified than cameras:\n"
                    f"\t- key_cam:  {key_cam}\n"
                    f"\t- data cam: {lcam}\n"
                    f"\t- data: {data}"
                )
                raise Exception(msg)
            
            # reorder
            lok = [lok[lcam.index(cc)] for cc in key_cam]
        
        else:
            lok = [
                coll.dobj['diagnostic'][key]['doptics'][cc][data]
                for cc in key_cam
            ]
            
        # safety check
        lout = [ii for ii in range(len(key_cam)) if lok[ii] is None]
        if len(lout) == len(key_cam):
            msg = (
                "data not available for the desired cameras\n"
                f"\t- camera: {key_cam}\n"
                f"\t- data: {lok}"
                )
            raise Exception(msg)
        elif len(lout) > 0:
            key_cam = [k0 for ii, k0 in enumerate(key_cam) if ii not in lout]
            lok = [k0 for ii, k0 in enumerate(lok) if ii not in lout]
        
        # lref, ldata
        lref = [coll.ddata[dd]['ref'] for dd in lok]
        ldata = [coll.ddata[dd]['data'] for dd in lok]
        
    else:
        if data in ['lamb', 'lambmin', 'lambmax', 'res']:
            ldata, lref = coll.get_diagnostic_lamb(
                key=key,
                key_cam=key_cam,
                rocking_curve=rocking_curve,
                lamb=data,
            )
        elif data == 'tangency radius':
            ldata, _, lref = coll.get_rays_tangency_radius(
                key=key,
                key_cam=key_cam,
                segment=-1,
                lim_to_segments=False,
                )

    # ------------
    # check shapes
    
    lshapes = [dd.shape for dd in ldata]
    if not all([len(ss) == len(lshapes[0]) for ss in lshapes[1:]]):
        msg = f"All data shall have the same ndim!\n\t- {lshapes}"
        raise Exception(msg)
    
    # indices in ref
    if is2d:
        if stack == 'horizontal':
            lref0 = [coll.dobj['camera'][cc]['dgeom']['ref'][0] for cc in key_cam]
        else:
            lref0 = [coll.dobj['camera'][cc]['dgeom']['ref'][1] for cc in key_cam]
    else:
        lref0 = [coll.dobj['camera'][cc]['dgeom']['ref'][0] for cc in key_cam]
    
    lind = [rr.index(lref0[ii]) for ii, rr in enumerate(lref)]
    if not all([li == lind[0] for li in lind[1:]]):
        msg = f"All shapes must have a common ref index!\n\t- {lind}"
        raise Exception(msg)
        
    axis = lind[0]
    
    # other dimensions should be equal
    lshapes = [
        [si for ii, si in enumerate(ss) if ii != axis]
        for ss in lshapes
    ]
    if not all([ss == lshapes[0] for ss in lshapes[1:]]):
        msg = f"All shapes must be identical except along axis!\n\t- {lshapes}"
        raise Exception(msg)
    
    # ------------
    # concatenate
    
    if len(key_cam) == 1:
        data_out = ldata[0]
    
    else:
        data_out = np.concatenate(tuple(ldata), axis=axis)
    
    # ------------
    # return
    
    ref = None
    if returnas is dict:
        
        # ref
        kref = None
        dref = {kref: {'size': data_out.shape[axis]}}
        
        # ddata
        kdata = None
        ddata = {
            kdata: {
                'ref': ref,
                'data': data,
            },
        }
        
    else:
        return key, key_cam, data_out, ref
    

# ##################################################################
# ##################################################################
#             interpolated along los
# ##################################################################


def _interpolated_along_los(
    coll=None,
    key=None,
    key_cam=None,
    key_data_x=None,
    key_data_y=None,
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
    plot=None,
    dcolor=None,
    dax=None,
    ):
    
    # -------------
    # check inputs
    
    # key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    
    # key_data
    lok_coords = ['x', 'y', 'z', 'R', 'phi', 'k', 'kabs']
    lok_2d = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
    ]
    
    key_data_x = ds._generic_check._check_var(
        key_data_x, 'key_data_x',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )
    
    key_data_y = ds._generic_check._check_var(
        key_data_y, 'key_data_y',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )
    
    # segment
    segment = ds._generic_check._check_var(
        segment, 'segment',
        types=int,
        default=-1,
    )
    
    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )
    
    # dcolor
    if not isinstance(dcolor, dict):
        if dcolor is None:
            lc = ['k', 'r', 'g', 'b', 'm', 'c']
        elif mcolors.is_color_like(dcolor):
            lc = [dcolor]
            
        dcolor = {
            kk: lc[ii%len(lc)]
            for ii, kk in enumerate(key_cam)
            }
    
    # --------------
    # prepare output
    
    ncam = len(key_cam)
    
    xx = [None for ii in range(ncam)]
    yy = [None for ii in range(ncam)]
    
    # ---------------
    # loop on cameras
    
    if key_data_x in lok_coords and key_data_y in lok_coords:
        
        for ii, kk in enumerate(key_cam):
            
            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue
            
            xx[ii], yy[ii] = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=[key_data_x, key_data_y],
                )    
    
    elif key_data_x in lok_coords or key_data_y in lok_coords:
    
        if key_data_x in lok_coords:
            cll = key_data_x
            c2d = key_data_y
        else:
            cll = key_data_y
            c2d = key_data_x

        for ii, kk in enumerate(key_cam): 
            
            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue
            
            pts_x, pts_y, pts_z, pts_ll = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=['x', 'y', 'z', cll],
                )      
            
            Ri = np.hypot(pts_x, pts_y)
            
            q2d, _ = coll.interpolate_profile2d(
                key=c2d,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                nan_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )  
                
            isok = ~(np.isnan(q2d) & (~np.isnan(Ri)))
            if key_data_x in lok_coords:
                xx[ii] = pts_ll[isok]
                yy[ii] = q2d[isok]
            else:
                xx[ii] = q2d[isok]
                yy[ii] = pts_ll[isok]
    
    else:
        for ii, kk in enumerate(key_cam):   
            
            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue
            
            pts_x, pts_y, pts_z = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=['x', 'y', 'z'],
                )      
    
            Ri = np.hypot(pts_x, pts_y)
            
            q2dx, _ = coll.interpolate_profile2d(
                key=key_data_x,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                nan_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )  
    
            q2dy, _ = coll.interpolate_profile2d(
                key=key_data_y,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                nan_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )  
    
            isok = ~((np.isnan(q2dx) | np.isnan(q2dy)) & (~np.isnan(Ri)))
            xx[ii] = q2dx[isok]
            yy[ii] = q2dy[isok]
   
    # ------------
    # plot
    
    if plot is True:
        if dax is None:
            
            fig = plt.figure()
            
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            
            tit = f"{key} LOS\nminor radius vs major radius"
            ax.set_title(tit, size=12, fontweight='bold')
            ax.set_xlabel('R (m)')
            ax.set_ylabel(r'$\rho_{p,norm}$')
            
            dax = {'main': ax}
            
        # main
        kax = 'main'
        if dax.get(kax) is not None:
            ax = dax[kax]
            
            for ii, kk in enumerate(key_cam):
                ax.plot(
                    xx[ii],
                    yy[ii],
                    c=dcolor[kk],
                    marker='.',
                    ls='-',
                    ms=8,
                    label=kk,
                )
     
            ax.legend()
    
    return xx, yy, dax