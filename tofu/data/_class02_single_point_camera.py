

import numpy as np
import datastock as ds


# ########################################################
# ########################################################
#              DEFAULTS
# ########################################################


_DANGLES = {
    'angle0': {
        'nb': 51,
        'lim': np.r_[-1, 1] * 0.5 * np.pi,
    },
    'angle1': {
        'nb': 51,
        'lim': np.r_[-1, 1] * 0.5 * np.pi,
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
):

    # -------------
    # check inputs
    # -------------

    key, cent, nin, e0, e1, angle0, angle1 = _check(
        coll=coll,
        key=key,
        cent=cent,
        nin=nin,
        e0=e0,
        e1=e1,
        angle0=angle0,
        angle1=angle1,
        config=config,
    )

    # -------------
    # prepare
    # -------------

    # -----------
    # angles ref

    ref_rays = (f'{key}_nangle0', f'{key}_nangle1')
    nrays = (angle0.size, angle1.size)
    coll.add_ref(ref_rays[0], size=nrays[0])
    coll.add_ref(ref_rays[1], size=nrays[1])

    # ------------
    # angles data

    kangle0 = f"{key}_angle0"
    coll.add_data(
        kangle0,
        data=angle0*180/np.pi,
        units='deg',
        ref=ref_rays[0],
    )

    kangle1 = f"{key}_angle1"
    coll.add_data(
        kangle1,
        data=angle1*180/np.pi,
        units='deg',
        ref=ref_rays[1],
    )

    # -------------
    # unit vectors
    # -------------

    # angles, full shape
    angle0f = angle0[:, None]
    angle1f = angle1[None, :]

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
        ref=ref_rays,
        # config
        config=config,
        strict=False,
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
    config=None,
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
    )

    # --------------
    # angle0, angle1 angles
    # --------------

    # angle0
    angle0 = _check_angles(
        ang=angle0,
        ang_name='angle0',
    )

    # angle1
    angle1 = _check_angles(
        ang=angle1,
        ang_name='angle1',
    )

    # --------------
    # options
    # --------------

    return key, cent, nin, e0, e1, angle0, angle1


def _check_angles(
    ang=None,
    ang_name=None,
):

    # --------------
    # default values
    # --------------

    if ang is None:
        ang = _DANGLES[ang_name]['nb']

    if isinstance(ang, (float, int)):
        ang = np.linspace(
            _DANGLES[ang_name]['lim'][0],
            _DANGLES[ang_name]['lim'][1],
            int(ang),
        )

    # --------------
    # flat unique array
    # --------------

    ang = ds._generic_check._check_flat1darray(
        ang, ang_name,
        dtype=float,
        unique=True,
    )

    return ang
