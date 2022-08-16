

import datastock as ds


from ..geom import _etendue


# #############################################################################
# #############################################################################
#                       outlines
# #############################################################################


def get_optics_outline(
    coll=None,
    key=None,
    add_points=None,
    closed=None,
):

    # ------------
    # check inputs

    key, cls, add_points, closed = _get_optics_outline_check(
        coll=coll,
        key=key,
        add-points=add_points,
        closed=closed,
    )

    # --------
    # compute

    if cls == 'aperture':
        px = coll.dobj['aperture'][key]['poly_x']
        py = coll.dobj['aperture'][key]['poly_y']
        pz = coll.dobj['aperture'][key]['poly_z']

    elif cls == 'camera':
        if coll.dobj['']



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
    # element

    if isinstance(element, str):
        element = [element]

    lok = ['o', 'v', 'c', 'r']
    optics = ds._generic_check._check_var_iter(
        element, 'element',
        default=lok,
        allowed=lok,
    )

    return key, optics, element


def _dplot(
    coll=None,
    key=None,
    optics=None,
    elements=None,
    vect_length=None,
):

    # ------------
    # check inputs

    key, optics, element = _dplot_check(
        coll=coll,
        key=key,
        optics=optics,
        element=element,
    )

    # ------------
    # build dict

    dplot = dict.fromkeys(k0, {})
    for k0 in optics:

        if k0 in coll.dobj.get('camera', []):
            v0 = coll.dobj['camera'][k0]
        elif k0 in coll.dobj.get('aperture', []):
            v0 = coll.dobj['aperture'][k0]
        elif k0 in coll.dobj.get('crystal', []):
            v0 = coll.dobj['crystal'][k0]
        else:
            msg = f"Unknown optics '{k0}'"
            raise Exception(msg)

        # outline
        if 'o' in element:

            px, py, pz = coll.get_optics_outline(
                key=k0,
                add_points=3,
                closed=True,
            )

            dplot[k0]['o'] = {
                'x': px,
                'y': py,
                'z': pz,
                'r': np.hypot(px, py),
                'label': f'{k0}-o',
            }

        # unit vectors
        if 'v' in element:

            pass

        # summit
        if 'c' in element:

            if 'cent' in v0.keys():
                cx, cy, cz = v0['cent']
            elif 'cents' in v0.keys():
                cx, cy, cz = v0['cents']
                cx = coll.ddata[cx]['data']
                cy = coll.ddata[cy]['data']
                cz = coll.ddata[cz]['data']

            dplot[k0]['c'] = {
                'x': cx,
                'y': yy,
                'z': cz,
                'r': np.hypot(cx, cy),
                'label': f'{k0}-o',
            }

        # rowland / axis for curved optics

        # label

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
        default=True,
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

    return detend
