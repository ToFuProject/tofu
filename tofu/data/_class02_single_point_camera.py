

import numpy as np
import datastock as ds


# ########################################################
# ########################################################
#              DEFAULTS
# ########################################################


_DANGLES = {
    'angle0': {
        'nb': 101,
        'lim': np.r_[-1, 1] * 90,
        'units': 'deg',
    },
    'angle1': {
        'nb': 101,
        'lim': np.r_[-1, 1] * 90,
        'units': 'deg',
    },
}


# ########################################################
# ########################################################
#              Main
# ########################################################


def main(
    coll=None,
    key=None,
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    angle0=None,
    angle1=None,
    config=None,
    strict=None,
    # optional naming
    key_angle0=None,
    key_angle1=None,
    ref_angle0=None,
    ref_angle1=None,
    units_angles=None,
):

    # -------------
    # check inputs
    # -------------

    key, cent, nin, e0, e1, dangles, strict = _check(
        coll=coll,
        key=key,
        cent=cent,
        nin=nin,
        e0=e0,
        e1=e1,
        angle0=angle0,
        angle1=angle1,
        strict=strict,
        # optional naming
        key_angle0=key_angle0,
        key_angle1=key_angle1,
        ref_angle0=ref_angle0,
        ref_angle1=ref_angle1,
        units_angles=units_angles,
    )

    # -------------
    # prepare
    # -------------

    # -----------
    # angles ref

    for k0, v0 in dangles.items():
        if v0['ref'] not in coll.dref.keys():
            coll.add_ref(v0['ref'], size=v0['data'].size)

    # ------------
    # angles data

    for k0, v0 in dangles.items():
        if v0['key'] not in coll.dref.keys():
            coll.add_data(**v0)

    # -------------
    # unit vectors
    # -------------

    # angles, full shape
    angle0f = dangles['angle0']['data'][:, None]
    if dangles['angle0']['units'] == 'deg':
        angle0f = angle0f * np.pi/180.

    angle1f = dangles['angle1']['data'][None, :]
    if dangles['angle1']['units'] == 'deg':
        angle1f = angle1f * np.pi/180.

    # unit vectors
    vx = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[0]) + np.sin(angle0f) * e0[0])
        + np.sin(angle1f) * e1[0]
    )
    vy = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[1]) + np.sin(angle0f) * e0[1])
        + np.sin(angle1f) * e1[1]
    )
    vz = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[2]) + np.sin(angle0f) * e0[2])
        + np.sin(angle1f) * e1[2]
    )

    # -------------
    # compute
    # -------------

    coll.add_rays(
        key,
        start_x=np.full(vx.shape, cent[0]),
        start_y=np.full(vx.shape, cent[1]),
        start_z=np.full(vx.shape, cent[2]),
        # vectors
        vect_x=vx,
        vect_y=vy,
        vect_z=vz,
        # ref
        ref=(dangles['angle0']['ref'], dangles['angle1']['ref']),
        # config
        config=config,
        strict=strict,
    )

    return


# ########################################################
# ########################################################
#              Check
# ########################################################


def _check(
    coll=None,
    key=None,
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    angle0=None,
    angle1=None,
    strict=None,
    # optional naming
    key_angle0=None,
    key_angle1=None,
    ref_angle0=None,
    ref_angle1=None,
    units_angles=None,
):

    # --------------
    # key
    # --------------

    wrays = coll._which_rays
    lout = list(coll.dobj.get(wrays, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        default='ptcam',
        excluded=lout,
    )

    # --------------
    # cent
    # --------------

    cent = ds._generic_check._check_flat1darray(
        cent, 'cent',
        dtype=float,
        size=3,
    )

    # --------------
    # unit vectors
    # --------------

    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
        direct=False,
    )

    # --------------
    # angle0, angle1 angles
    # --------------

    # angle0
    dangles = {
        'angle0': _check_angles(
            coll=coll,
            key=key,
            ang=angle0,
            ang_name='angle0',
            ang_key=key_angle0,
            ang_ref=ref_angle0,
        ),
        'angle1': _check_angles(
            coll=coll,
            key=key,
            ang=angle1,
            ang_name='angle1',
            ang_key=key_angle1,
            ang_ref=ref_angle1,
        ),
    }

    # --------------
    # options
    # --------------

    # strict
    strict = ds._generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    return key, cent, nin, e0, e1, dangles, strict


def _check_angles(
    coll=None,
    key=None,
    ang=None,
    ang_name=None,
    ang_key=None,
    ang_ref=None,
    ang_units=None,
):

    # --------------
    # existing key
    # --------------

    if isinstance(ang, str):
        lok = [
            k0 for k0, v0 in coll.ddata.items()
            if v0['monot'] == (True,)
        ]
        ang = ds._generic_check._check_var(
            ang, ang_name,
            types=str,
            allowed=lok,
        )

        ang_key = ang
        ang_data = coll.ddata[ang_key]['data']
        ang_ref = coll.ddata[ang_key]['ref'][0]
        ang_units = coll.ddata[ang_key]['units']

    # --------------
    # int, float, array
    # --------------

    else:

        if ang is None:
            ang = _DANGLES[ang_name]['nb']

        if isinstance(ang, (float, int)):
            ang = np.linspace(
                _DANGLES[ang_name]['lim'][0],
                _DANGLES[ang_name]['lim'][1],
                int(ang),
            )
            ang_units = _DANGLES[ang_name]['units']

        # --------------
        # flat unique array

        ang_data = ds._generic_check._check_flat1darray(
            ang, ang_name,
            dtype=float,
            unique=True,
        )

        # -------
        # keys

        lout = list(coll.ddata.keys())
        ang_key = ds._generic_check._check_var(
            ang_key, ang_name,
            types=str,
            default=f"{key}_{ang_name}",
            excluded=lout,
        )

        # -------
        # ref

        lout = list(coll.dref.keys())
        ang_ref = ds._generic_check._check_var(
            ang_ref, f"{ang_name}",
            types=str,
            default=f"{key}_n{ang_name}",
            excluded=lout,
        )

        # -------
        # units

        ang_units = ds._generic_check._check_var(
            ang_units, "units_angles",
            types=str,
            default="deg",
            allowed=['deg', 'rad'],
        )

    return {
        'key': ang_key,
        'data': ang_data,
        'ref': ang_ref,
        'units': ang_units,
    }
