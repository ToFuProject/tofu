

import numpy as np


from ..geom._comp_solidangles import _check_polygon_2d, _check_polygon_3d


# #############################################################################
# #############################################################################
#                           Aperture
# #############################################################################


def _check_unitvector(uv=None, uv_name=None):
    try:
        uv = np.atleast_1d(uv).ravel().astype(float)
        assert uv.shape == (3,)
    except Exception as err:
        msg = str(err) + (
            f"\nArg {uv_name} not convertible to (3,) float np.ndarray!"
            "Provided: {uv}"
        )
        raise Exception(msg)

    # enforce normalization
    return uv / np.linalg.norm(uv)


def _check_nine0e1(nin=None, e0=None, e1=None):

    # e0 or e0 provided => compute missing one
    if e0 is None and e1 is not None:
        e0 = np.cross(e1, nin)
    elif e0 is not None and e1 is None:
        e1 = np.cross(nin, e0)

    # either e0 and e1 provided or none
    if e0 is not None:
        dv = {
            'nin.e0': np.abs(np.sum(nin*e0)),
            'nin.e1': np.abs(np.sum(nin*e1)),
            'e0.e1': np.abs(np.sum(e0*e1)),
            '|nin.(e0 x e1)|': np.linalg.norm(np.cross(nin, np.cross(e0, e1))),
        }
        dv = {k0: v0 for k0, v0 in dv.items() if v0 > 1.e-15}
        if len(dv) > 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in dv.items()]
            msg = (
                "Args (e0, e1, nin) muist form a direct orthonormal basis!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    return nin, e0, e1


def _aperture_check(
    coll=None,
    key=None,
    # 2d outline
    outline_x0=None,
    outline_x1=None,
    cent=None,
    # 3d outline
    poly_x=None,
    poly_y=None,
    poly_z=None,
    # normal vector
    nin=None,
    e0=None,
    e1=None,
):

    # ----
    # key

    lout = list(coll.dobj.get('aperture', {}).keys())
    if key is None:
        if len(lout) == 0:
            nb = 0
        else:
            lnb = [
                int(k0[2:]) for k0 in lout if k0.startswith('ap')
                and k0[2:].isnumeric()
            ]
            nb = min([ii for ii in range(max(lnb)+2) if ii not in lnb])
        key = f'ap{nb}'

    key = ds_generic_check._check_var(
        key, 'key',
        types=str,
        excluded=lout,
    )

    # -----------
    # cent

    if cent is not None:
        cent = np.atleast_1d(cent).ravel().astype(float)
        assert cent.shape == (3,)

    # -----------
    # unit vectors

    nin = _check_unitvector(uv=nin, uv_name='nin')

    if e0 is None and e1 is None:
        if np.abs(nin[2]) < 0.99:
            e0 = np.r_[-nin[1], nin[0], 0.]
        else:
            e0 = np.r_[np.sign(nin[2]), 0., 0.]

    if e0 is not None:
        e0 = _check_unitvector(uv=e0, uv_name='e0')
    if e1 is not None:
        e1 = _check_unitvector(uv=e1, uv_name='e1')

    if e0 is not None or e1 is not None:
        nin, e0, e1 _check_nine0e1(nin=nin, e0=e0, e1=e1)

    # ---------------
    # outline vs poly

    lc = [
        all([pp is not None for pp in [outline_x0, outline_x1]])
        and e0 is not None and cent is not None,
        all([pp is not None for pp in [poly_x, poly_y, poly_z]])
    ]
    if np.sum(lc) != 1:
        msg = (
            "Please provide either (not both):\n"
            "\t- outline_x0, outline_x1 and e0, e1\n"
            "xor\n"
            "\t- poly_x, poly_y, poly_z"
        )
        raise Exception(msg)

    # --------------
    # outline

    planar = None
    if outline_x0 is not None:

        # planar
        planar = True

        # check outline
        outline_x0, outline_x1, area = _check_polygon_2d(
            poly_x=outline_x0,
            poly_y=outline_x1,
            poly_name=f'{key}-outline',
            can_be_None=False,
            closed=False,
            counter_clockwise=True,
            return_area=True,
        )

        # derive poly 3d
        poly_x = cent[0] + outline_x0 * e0[0] + outline_x1 * e1[0]
        poly_y = cent[1] + outline_x0 * e0[1] + outline_x1 * e1[1]
        poly_z = cent[2] + outline_x0 * e0[2] + outline_x1 * e1[2]

    # -----------
    # poly3d

    poly_x, poly_y, poly_z = _check_polygon_3d(
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        poly_name=f'{key}-polygon',
        can_be_None=False,
        closed=False,
        counter_clockwise=True,
        normal=nin,
    )


    if outline_x0 is None:

        # ----------
        # cent

        if cent is None:
            cent = np.r_[np.mean(poly_x), np.mean(poly_y), np.mean(poly_z)]

        # ----------
        # planar 

        diff_x = poly_x[1:] - poly_x[0]
        diff_y = poly_y[1:] - poly_y[0]
        diff_z = poly_z[1:] - poly_z[0]
        norm = np.sqrt(diff_x**2 + diff_y**2 + diff_x**2)
        diff_x = diff_x / norm
        diff_y = diff_y / norm
        diff_z = diff_z / norm

        sca = np.abs(nin[0]*diff_x + nin[1]*diff_y + nin[2]*diff_z)

        if np.all(sca < 2.e-12):
            # all deviation smaller than 1.e-10 degree
            planar = True

            # derive outline
            outline_x0 = (
                (poly_x - cent[0]) * e0[0]
                + (poly_y - cent[1]) * e0[1]
                + (poly_z - cent[2]) * e0[2]
            )
            outline_x1 = (
                (poly_x - cent[0]) * e1[0]
                + (poly_y - cent[1]) * e1[1]
                + (poly_z - cent[2]) * e1[2]
            )

            # check outline
            outline_x0, outline_x1, area = _check_polygon_2d(
                poly_x=outline_x0,
                poly_y=outline_x1,
                poly_name=f'{key}-outline',
                can_be_None=False,
                closed=False,
                counter_clockwise=True,
                return_area=True,
            )

        else:
            planar = False
            area = np.nan

    assert planar == outline_x0 is not None

    return (
        key, cent,
        outline_x0, outline_x1,
        poly_x, poly_y, poly_z,
        nin, e0, e1,
        area, planar,
    )


def _aperture(
    coll=None,
    key=None,
    # 2d outline
    outline_x0=None,
    outline_x1=None,
    # 3d outline
    poly_x=None,
    poly_y=None,
    poly_z=None,
    # normal vector
    nin=None,
    e0=None,
    e1=None,
):

    # ------------
    # check inputs

    (
        key,
        cent,
        outline_x0, outline_x1,
        poly_x, poly_y, poly_z,
        nin, e0, e1,
        area, planar,
    ) = _aperture_check(
        coll=coll,
        key=key,
        # 3d outline
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        # normal vector
        nin=nin,
    )

    # ----------
    # create dict

    # keys
    knpts = f'{key}-npts'
    kpx = f'{key}-x'
    kpy = f'{key}-y'
    kpz = f'{key}-z'
    if planar:
        kp0 = f'{key}-x0'
        kp0 = f'{key}-x1'
        outline = (kp0, kp1)
    else:
        outline = None

    # refs
    npts = poly_x.size

    dref = {
        knpts: {'size': npts},
    }

    # data
    ddata = {
        kpx: {
            'data': poly_x,
            'ref': knpts,
            'dim': 'distance',
            'name': 'x',
            'quant': 'x',
            'units': 'm',
        },
        kpy: {
            'data': poly_y,
            'ref': knpts,
            'dim': 'distance',
            'name': 'y',
            'quant': 'y',
            'units': 'm',
        },
        kpz: {
            'data': poly_z,
            'ref': knpts,
            'dim': 'distance',
            'name': 'z',
            'quant': 'z',
            'units': 'm',
        },
    }
    if planar:
        ddata.update({
            kp0: {
                'data': outline_x0,
                'ref': knpts,
                'dim': 'distance',
                'name': 'x0',
                'quant': 'x0',
                'units': 'm',
            },
            kp1: {
                'data': outline_x1,
                'ref': knpts,
                'dim': 'distance',
                'name': 'x1',
                'quant': 'x1',
                'units': 'm',
            },
        })

    # dobj
    dobj = {
        'aperture': {
            key: {
                'poly': (kpx, kpy, kpz),
                'outline': outline,
                'planar': planar,
                'area': area,
                'cent': cent,
                'nin': nin,
                'e0': e0,
                'e1': e1,
            },
        },
    }

    return dref, ddata, dobj


# #############################################################################
# #############################################################################
#                           Camera
# #############################################################################


def _camera(
    key=None,
    # common 2d outline
    outline_x0=None,
    outline_x1=None,
    # centers of all pixels
    cents_x=None,
    cents_y=None,
    cents_z=None,
    # inwards normal vectors
    nin_x=None,
    nin_y=None,
    nin_z=None,
    # orthonormal direct base
    e0_x=None,
    e0_y=None,
    e0_z=None,
    e1_x=None,
    e1_y=None,
    e1_z=None,
    # quantum efficiency
    lamb=None,
    energy=None,
    qeff=None,
):

    pass
