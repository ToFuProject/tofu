# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp


import datastock as ds


from ..geom import _etendue


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
        det=coll.get_as_dict(which='camera', key=optics[0]),
        aperture=coll.get_as_dict(which='aperture', key=optics[1:]),
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
