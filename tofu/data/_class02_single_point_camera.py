

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
        # specific to ptcam
        angles=(dangles['angle0']['key'], dangles['angle1']['key']),
        nin=nin,
        e0=e0,
        e1=e1,
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


# ########################################################
# ########################################################
#              Get rays angles
# ########################################################


def _get_rays_angles(
    coll=None,
    key_single_pt_cam=None,
    # rays to get angles of
    key_rays=None,
    segment=None,
    # max tolerance
    tol_radius=None,
):

    # -------------
    # check inputs
    # -------------

    key_single_pt_cam, key_rays, segment, tol_radius = _check_rays_angles(
        coll=coll,
        key_single_pt_cam=key_single_pt_cam,
        key_rays=key_rays,
        segment=segment,
        tol_radius=tol_radius,
    )

    # -------------
    # prepare
    # -------------

    wrays = coll._which_rays
    ref = coll.dobj[wrays][key_single_pt_cam]['ref'][1:]
    shape = coll.dobj[wrays][key_single_pt_cam]['shape'][1:]
    kang0, kang1 = coll.dobj[wrays][key_single_pt_cam]['angles']
    units = coll.ddata[kang0]['units']

    # unit vectors
    e0 = coll.dobj[wrays][key_single_pt_cam]['e0']
    e1 = coll.dobj[wrays][key_single_pt_cam]['e1']

    ptsx, ptsy, ptsz = coll.get_rays_pts(key_single_pt_cam)
    cent = np.r_[ptsx[0, 0, 0], ptsy[0, 0, 0], ptsz[0, 0, 0]]

    # -------------
    # get impact factor vs tol_radius
    # -------------

    ptsx, ptsy, ptsz = coll.get_rays_pts(key_rays, segment=segment)
    vx, vy, vz = coll.get_rays_vect(key_rays, segment=segment, norm=True)

    p0c_x = (ptsx[0, ...] - cent[0])
    p0c_y = (ptsy[0, ...] - cent[1])
    p0c_z = (ptsz[0, ...] - cent[2])

    crossx = p0c_y * vz - p0c_z * vy
    crossy = p0c_z * vx - p0c_x * vz
    crossz = p0c_x * vy - p0c_y * vx

    impact = np.sqrt(crossx**2 + crossy**2 + crossz**2)

    if tol_radius is not None:
        iok = impact < tol_radius
    else:
        iok = np.ones(shape, dtype=bool)

    # -------------
    # compute
    # -------------

    # intialize
    ang0 = np.full(shape, np.nan)
    ang1 = np.full(shape, np.nan)

    # compute
    if np.any(iok):

        # vect from cent
        vx = ptsx[1, iok] - cent[0]
        vy = ptsy[1, iok] - cent[1]
        vz = ptsz[1, iok] - cent[2]
        vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
        vx = vx / vnorm
        vy = vy / vnorm
        vz = vz / vnorm

        # derive angles
        ang1[iok] = np.arcsin(vx * e1[0] + vy * e1[1] + vz * e1[2])
        cos1_sin0 = vx * e0[0] + vy * e0[1] + vz * e0[2]
        ang0[iok] = np.arcsin(cos1_sin0 / np.cos(ang1[iok]))

    # -------------
    # output
    # -------------

    dout = {
        'angle0': {
            'key': f'{key_rays}_{key_single_pt_cam}_ang0',
            'data': ang0,
            'units': units,
            'ref': ref,
            'dim': 'angle',
        },
        'angle1': {
            'key': f'{key_rays}_{key_single_pt_cam}_ang1',
            'data': ang1,
            'units': units,
            'ref': ref,
            'dim': 'angle',
        },
        'impact': {
            'key': f'{key_rays}_{key_single_pt_cam}_impact',
            'data': impact,
            'ref': ref,
            'units': 'm',
            'dim': 'distance',
        },
    }

    return dout


# ########################################################
# ########################################################
#              check inputs rays angles
# ########################################################


def _check_rays_angles(
    coll=None,
    key_single_pt_cam=None,
    # rays to get angles of
    key_rays=None,
    segment=None,
    # max tolerance
    tol_radius=None,
):

    # -------------
    # keys
    # -------------

    wrays = coll._which_rays

    # key_rays
    lok = list(coll.dobj.get(wrays, {}).keys())
    key_rays = ds._generic_check._check_var(
        key_rays, 'key_rays',
        types=str,
        allowed=lok,
    )

    # key_single_pt_cam
    lok = [
        k0 for k0 in lok
        if isinstance(coll.dobj[wrays][k0].get('angles'), tuple)
        and len(coll.dobj[wrays][k0]['angles']) == 2
        and all([
            ss in coll.ddata.keys()
            for ss in coll.dobj[wrays][k0].get('angles')
        ])
        and isinstance(coll.dobj[wrays][k0].get('nin'), np.ndarray)
        and isinstance(coll.dobj[wrays][k0].get('e0'), np.ndarray)
        and isinstance(coll.dobj[wrays][k0].get('e1'), np.ndarray)
    ]
    key_single_pt_cam = ds._generic_check._check_var(
        key_single_pt_cam, 'key_single_pt_cam',
        types=str,
        allowed=lok,
    )

    # -----------------
    # segment
    # -----------------

    segment = int(ds._generic_check._check_var(
        segment, 'segment',
        types=(int, float),
        default=0,
    ))

    # -----------------
    # tol_radius
    # -----------------

    if tol_radius is not None:
        tol_radius = float(ds._generic_check._check_var(
            tol_radius, 'tol_radius',
            types=(int, float),
            sign='>0',
        ))

    return (
        key_single_pt_cam, key_rays,
        segment, tol_radius,
    )
