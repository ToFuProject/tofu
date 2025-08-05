# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt     # DB


import datastock as ds


from ..geom import CamLOS1D
from . import _class8_vos_utilities as _utilities


__all__ = ['compute_los_angles']


# ##############################################################
# ##############################################################
#                       Main
# ##############################################################


def compute_los_angles(
    coll=None,
    key=None,
    # los
    dcompute=None,
    # for storing los
    config=None,
    strict=None,
    length=None,
    reflections_nb=None,
    reflections_type=None,
    key_nseg=None,
    # for vos based on los
    res=None,
    compute_vos_from_los=None,
    # overwrite
    overwrite=None,
    **kwdargs,
):

    # ------------
    # check inputs
    # ------------

    key, is2d, compute_vos_from_los = _check(
        coll=coll,
        key=key,
        compute_vos_from_los=compute_vos_from_los,
    )

    # ---------------
    # loop on cameras
    # ---------------

    print(f"\tComputing los for diag '{key}'")
    for ii, (key_cam, v0) in enumerate(dcompute.items()):

        if v0['los_x'] is None or not np.any(np.isfinite(v0['los_x'])):
            continue

        # klos
        klos = f'{key}_{key_cam}_los'

        # ref
        ref = coll.dobj['camera'][key_cam]['dgeom']['ref']

        # ------------
        # add los

        cx2, cy2, cz2 = coll.get_camera_cents_xyz(key=key_cam)

        # check overwrite
        if klos in coll.dobj.get('rays', {}).keys():
            if overwrite is True:
                coll.remove_rays(klos)
            else:
                msg = (
                    f"Rays '{klos}' already exists!\n"
                    "Use overwrite=True to force overwrite!\n"
                )
                raise Exception(msg)

        coll.add_rays(
            key=klos,
            start_x=cx2,
            start_y=cy2,
            start_z=cz2,
            vect_x=v0['los_x'],
            vect_y=v0['los_y'],
            vect_z=v0['los_z'],
            ref=ref,
            diag=key,
            key_cam=key_cam,
            config=config,
            length=length,
            strict=False,
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
            key_nseg=key_nseg,
        )

        coll._dobj['diagnostic'][key]['doptics'][key_cam]['los'] = klos

        # ---------------------
        # rough estimate of vos

        if compute_vos_from_los is True:

            if ii == 0:
                print(f"\tComputing vos from los for diag '{key}'")

            _vos_from_los(
                coll=coll,
                key=key,
                key_cam=key_cam,
                v0=v0,
                config=config,
                strict=strict,
                res=res,
                overwrite=overwrite,
            )

        # ----------------------------------------
        # for spectro => estimate angle variations

        if v0['spectro'] is True:

            _angle_spectro(
                coll=coll,
                key=key,
                key_cam=key_cam,
                v0=v0,
                is2d=is2d,
                ref=ref,
                i0=ii,
                overwrite=overwrite,
            )


# ###########################################################
# ###########################################################
#               check
# ###########################################################


def _check(
    coll=None,
    key=None,
    compute_vos_from_los=None,
):

    # ----------
    # key

    lok = list(coll.dobj.get('diagnostic', {}))
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    is2d = coll.dobj['diagnostic'][key]['is2d']

    # ---------------------
    # compute_vos_from_los

    compute_vos_from_los = ds._generic_check._check_var(
        compute_vos_from_los, 'compute_vos_from_los',
        types=bool,
        default=True,
    )

    return key, is2d, compute_vos_from_los


# ###########################################################
# ###########################################################
#               VOS from LOS
# ###########################################################


def _vos_from_los(
    coll=None,
    key=None,
    key_cam=None,
    v0=None,
    config=None,
    strict=None,
    res=None,
    overwrite=None,
):

    # --------------
    # check inputs
    # --------------

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        default=0.005,
        sign='>0',
    )

    # -----------
    # prepare
    # -----------

    # optics
    optics = coll.dobj['diagnostic'][key]['doptics'][key_cam]['optics']
    pinhole = coll.dobj['diagnostic'][key]['doptics'][key_cam]['pinhole']

    # lspectro
    lspectro = [
        oo for oo in optics
        if oo in coll.dobj.get('crystal', {}).keys()
        or oo in coll.dobj.get('grating', {}).keys()
    ]

    # delta around cents, shape (shape_cam, npts)
    dx, dy, dz = coll.get_camera_dxyz(
        key=key_cam,
        include_center=True,
        kout=[0.3, 0.7, 1],
    )

    if pinhole is True:
        iref = v0['iref']

    # ----------
    # poly_cross
    # ----------

    shape_cam = v0['cx'].shape
    dphi = np.full(((2,) + shape_cam), np.nan)
    douti = {}
    indok = np.ones(shape_cam, dtype=bool)
    for ii, ind in enumerate(np.ndindex(shape_cam)):

        if not v0['iok'][ind]:
            indok[ind] = False
            continue

        sli = ind + (slice(None),)
        if pinhole is False:
            iref = v0['iref'][ind]

        # -----------------------
        # get start / end points

        x0 = np.r_[
            v0['x0'][sli],
            0.7*v0['x0'][sli],
            0.3*v0['x0'][sli],
            v0['cents0'][ind],
        ]
        x1 = np.r_[
            v0['x1'][sli],
            0.7*v0['x1'][sli],
            0.3*v0['x1'][sli],
            v0['cents1'][ind],
        ]

        ptsx, ptsy, ptsz = _get_rays_from_pix(
            coll=coll,
            # start points
            cx=v0['cx'][ind],
            cy=v0['cy'][ind],
            cz=v0['cz'][ind],
            dx=dx[sli],  # np.r_[0],
            dy=dy[sli],  # np.r_[0],
            dz=dz[sli],  # np.r_[0],
            # end points
            x0=x0,
            x1=x1,
            coords=coll.get_optics_x01toxyz(key=optics[iref]),
            lspectro=lspectro,
            config=config,
            strict=strict,
            # debug
            key=key,
            key_cam=key_cam,
            ii=ii,
            ind=ind,
        )

        # ---------
        # sampling

        length = np.sqrt(
            np.diff(ptsx, axis=0)**2
            + np.diff(ptsy, axis=0)**2
            + np.diff(ptsz, axis=0)**2
        )

        npts = int(np.ceil(np.nanmax(length) / res)) + 2
        indi = np.linspace(0, 1, npts)
        iok = np.all(np.isfinite(ptsx), axis=0)

        # -------------
        # interpolate

        ptsx = scpinterp.interp1d(
            [0, 1],
            ptsx[:, iok],
            kind='linear',
            axis=0,
        )(indi).ravel()

        ptsy = scpinterp.interp1d(
            [0, 1],
            ptsy[:, iok],
            kind='linear',
            axis=0,
        )(indi).ravel()

        ptsz = scpinterp.interp1d(
            [0, 1],
            ptsz[:, iok],
            kind='linear',
            axis=0,
        )(indi).ravel()

        # -------
        # dphi

        phi = np.arctan2(ptsy, ptsx)
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)

        # across pi?
        if phimax - phimin > np.pi:
            phimin = np.min(phi[phi > 0])
            phimax = np.max(phi[phi < 0])
            phimax = phimax + 2*np.pi

        dphi[(0,) + ind] = phimin
        dphi[(1,) + ind] = phimax

        # -----------
        # poly_cross

        ptsr = np.hypot(ptsx, ptsy)

        convh = ConvexHull(np.array([ptsr, ptsz]).T)
        pcross0 = ptsr[convh.vertices]
        pcross1 = ptsz[convh.vertices]

        # -----------
        # poly_hor

        convh = ConvexHull(np.array([ptsx, ptsy]).T)
        phor0 = ptsx[convh.vertices]
        phor1 = ptsy[convh.vertices]

        # ------------
        # add to douti

        douti[ind] = {
            'pcross0': pcross0,
            'pcross1': pcross1,
            'phor0': phor0,
            'phor1': phor1,
        }

    # ------------------------
    # reshape / harmonize
    # ------------------------

    ddata, dref = _utilities._harmonize_reshape(
        douti=douti,
        indok=indok,
        key_diag=key,
        key_cam=key_cam,
        ref_cam=coll.dobj['camera'][key_cam]['dgeom']['ref'],
    )

    if coll._dobj['diagnostic'][key]['doptics'][key_cam].get('dvos') is None:
        coll._dobj['diagnostic'][key]['doptics'][key_cam]['dvos'] = {}

    # ------------------------
    # dphi
    # ------------------------

    coll._dobj['diagnostic'][key]['doptics'][key_cam]['dvos']['dphi'] = dphi

    # ----------
    # store
    # ----------

    _utilities._store_dvos(
        coll=coll,
        key_diag=key,
        dvos={key_cam: ddata},
        dref={key_cam: dref},
        overwrite=overwrite,
        replace_poly=True,
        # optional
        keym=None,
        res_RZ=None,
        res_phi=None,
        res_lamb=None,
    )

    return

# ###########################################################
# ###########################################################
#               get multiple rays
# ###########################################################


def _get_rays_from_pix(
    coll=None,
    kref=None,
    # starting points
    cx=None,
    cy=None,
    cz=None,
    dx=None,
    dy=None,
    dz=None,
    # end points
    x0=None,
    x1=None,
    coords=None,
    lspectro=None,
    config=None,
    strict=None,
    # debug
    key=None,
    key_cam=None,
    ii=None,
    ind=None,
):

    # pixels points (start)
    cxi = cx + dx
    cyi = cy + dy
    czi = cz + dz
    nc = cxi.size

    # end points
    exi, eyi, ezi = coords(x0, x1)
    ne = exi.size

    # final shape
    shape = (2, nc*ne)
    pts_x = np.full(shape, np.nan)
    pts_y = np.full(shape, np.nan)
    pts_z = np.full(shape, np.nan)

    # adjust shapes
    pts_x[0, ...] = np.repeat(cxi, ne)
    pts_y[0, ...] = np.repeat(cyi, ne)
    pts_z[0, ...] = np.repeat(czi, ne)

    vect_x = np.tile(exi, nc) - pts_x[0, ...]
    vect_y = np.tile(eyi, nc) - pts_y[0, ...]
    vect_z = np.tile(ezi, nc) - pts_z[0, ...]

    # ----------
    # diag

    for jj, oo in enumerate(lspectro):

        reflect_ptsvect = coll.get_optics_reflect_ptsvect(oo)
        (
            pts_x[0, ...],
            pts_y[0, ...],
            pts_z[0, ...],
            vect_x, vect_y, vect_z,
            _,
            iok,
        ) = reflect_ptsvect(
            pts_x=pts_x[0, ...],
            pts_y=pts_y[0, ...],
            pts_z=pts_z[0, ...],
            vect_x=vect_x,
            vect_y=vect_y,
            vect_z=vect_z,
            strict=True,
            return_x01=False,
        )

    # ----------
    # config

    # prepare D
    D = np.array([pts_x[0, ...], pts_y[0, ...], pts_z[0, ...]])
    uu = np.array([vect_x, vect_y, vect_z])

    mask = np.all(np.isfinite(uu), axis=0)

    # call legacy code
    cam = CamLOS1D(
        dgeom=(D[:, mask], uu[:, mask]),
        config=config,
        Name='',
        Diag='',
        Exp='',
        strict=strict,
    )

    # pin
    pin = cam.dgeom['PkIn']

    pts_x[0, mask] = pin[0, :]
    pts_y[0, mask] = pin[1, :]
    pts_z[0, mask] = pin[2, :]

    # pout
    pout = cam.dgeom['PkOut']

    pts_x[-1, mask] = pout[0, :]
    pts_y[-1, mask] = pout[1, :]
    pts_z[-1, mask] = pout[2, :]

    return pts_x, pts_y, pts_z


# ###########################################################
# ###########################################################
#               Angles variations for spectro
# ###########################################################


def _angle_spectro(
    coll=None,
    key=None,
    key_cam=None,
    v0=None,
    is2d=None,
    ref=None,
    i0=None,
    overwrite=None,
):

    # ------------
    # prepare
    # ------------

    angmin = np.full(v0['cx'].shape, np.nan)
    angmax = np.full(v0['cx'].shape, np.nan)

    kref = v0['optics'][v0['iref']]
    ptsvect = coll.get_optics_reflect_ptsvect(key=kref)
    coords = coll.get_optics_x01toxyz(key=kref)

    # dx, dy, dz = coll.get_camera_dxyz(
    #     key=key_cam,
    #     include_center=True,
    # )

    # ------
    # loop
    # ------

    if i0 == 0:
        msg = (
            f"\tComputing angles for spectro diag '{key}':\n"
            f"\t\t- cam '{key_cam}':"
        )
    else:
        msg = f"\t\t- cam '{key_cam}':"
    print(msg)

    # langles = []        # DB
    linds = [range(ss) for ss in v0['cx'].shape]
    for ii, ij in enumerate(itt.product(*linds)):

        # verb
        msg = f"\t\t\tpixel {ii+1} / {v0['cx'].size}"
        end = "\n" if ii == v0['cx'].size - 1 else "\r"
        print(msg, end=end, flush=True)

        if not v0['iok'][ij]:
            continue

        # get 3d coordiantes of points on pixel
        cxi = v0['cx'][ij]  # + dx
        cyi = v0['cy'][ij]  # + dy
        czi = v0['cz'][ij]  # + dz

        nc = 1  # cxi.size

        # slice for x0
        sli = tuple(list(ij) + [slice(None)])

        # get 3d coords of points on crystal
        exi, eyi, ezi = coords(
            v0['x0'][sli],
            v0['x1'][sli],
        )
        ne = exi.size

        # cross
        cxi = np.repeat(cxi, ne)
        cyi = np.repeat(cyi, ne)
        czi = np.repeat(czi, ne)

        # get angles of incidence on crystal
        angles = ptsvect(
            pts_x=cxi,
            pts_y=cyi,
            pts_z=czi,
            vect_x=np.tile(exi, nc) - cxi,
            vect_y=np.tile(eyi, nc) - cyi,
            vect_z=np.tile(ezi, nc) - czi,
            strict=True,
            return_x01=False,
        )[6]

        # Correct for approximation of using
        # the same projected reflection from the center for all
        # ang0 = np.nanmean(angles[:ne])
        # angles[ne:] = ang0 + 0.5*(angles[ne:] - ang0)

        angmin[ij] = np.nanmin(angles[:ne])
        angmax[ij] = np.nanmax(angles[:ne])
        # langles.append(angles)      # DB

    # ddata
    kamin = f'{key}_{key_cam}_amin'
    kamax = f'{key}_{key_cam}_amax'
    ddata = {
        kamin: {
            'data': angmin,
            'ref': ref,
            'dim': 'angle',
            'quant': 'angle',
            'name': 'alpha',
            'units': 'rad',
        },
        kamax: {
            'data': angmax,
            'ref': ref,
            'dim': 'angle',
            'quant': 'angle',
            'name': 'alpha',
            'units': 'rad',
        },
    }

    # update
    for k0 in ddata.keys():
        if k0 in coll.ddata.keys():
            if overwrite is True:
                coll.remove_data(k0, propagate=False)
            else:
                msg = (
                    "Storing vos from los for:\n"
                    f"\t- diag '{key}'\n"
                    f"\t- camera '{key_cam}':\n"
                    f"\t- data '{k0}' already exists!\n"
                    "Use overwrite=True to force overwriting\n"
                )
                raise Exception(msg)

    coll.update(ddata=ddata)

    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amin'] = kamin
    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amax'] = kamax

    return
