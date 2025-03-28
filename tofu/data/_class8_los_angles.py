# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt     # DB


import datastock as ds


from ..geom import CamLOS1D


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

    dgeom = coll.dobj['camera'][key_cam]['dgeom']
    par = dgeom['parallel']

    if par:
        dx, dy, dz = coll.get_camera_dxyz(
            key=key_cam,
            include_center=True,
        )

    else:
        out0 = coll.ddata[dgeom['outline'][0]]['data']
        out1 = coll.ddata[dgeom['outline'][1]]['data']
        ke0 = dgeom['e0']
        ke1 = dgeom['e1']

    if pinhole is True:
        iref = v0['iref']

    # ----------
    # poly_cross
    # ----------

    lpoly_cross = []
    lpoly_hor = []

    shape0 = v0['cx'].shape
    dphi = np.full(((2,) + shape0), np.nan)
    linds = [range(ss) for ss in shape0]

    for ii, ind in enumerate(itt.product(*linds)):

        if not v0['iok'][ind]:
            lpoly_cross.append(None)
            lpoly_hor.append(None)
            continue

        sli = ind + (slice(None),)
        if pinhole is False:
            iref = v0['iref'][ind]

        if not par:
            e0 = [coll.ddata[kk]['data'][ind] for kk in ke0]
            e1 = [coll.ddata[kk]['data'][ind] for kk in ke1]
            dx = out0 * e0[0] + out1 * e1[0]
            dy = out0 * e0[1] + out1 * e1[1]
            dz = out0 * e0[2] + out1 * e1[2]

        # -----------------------
        # get start / end points

        ptsx, ptsy, ptsz = _get_rays_from_pix(
            coll=coll,
            # start points
            cx=v0['cx'][ind],
            cy=v0['cy'][ind],
            cz=v0['cz'][ind],
            dx=dx,  # np.r_[0],
            dy=dy,  # np.r_[0],
            dz=dz,  # np.r_[0],
            # end points
            x0=np.r_[v0['x0'][sli], v0['cents0'][ind]],
            x1=np.r_[v0['x1'][sli], v0['cents1'][ind]],
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
        conv0 = ptsr[convh.vertices]
        conv1 = ptsz[convh.vertices]

        lpoly_cross.append((conv0, conv1))

        # -----------
        # poly_hor

        convh = ConvexHull(np.array([ptsx, ptsy]).T)
        conv0 = ptsx[convh.vertices]
        conv1 = ptsy[convh.vertices]

        lpoly_hor.append((conv0, conv1))

    # ------------------------
    # poly_cross harmonization
    # ------------------------

    lnc = [0 if pp is None else pp[0].size for pp in lpoly_cross]
    lnh = [0 if pp is None else pp[0].size for pp in lpoly_hor]
    nmaxc = np.max(lnc)
    nmaxh = np.max(lnh)

    # ------------
    # prepare

    shc = tuple([nmaxc] + list(shape0))
    shh = tuple([nmaxh] + list(shape0))
    pcross0 = np.full(shc, np.nan)
    pcross1 = np.full(shc, np.nan)
    phor0 = np.full(shh, np.nan)
    phor1 = np.full(shh, np.nan)

    for ii, ind in enumerate(itt.product(*linds)):

        if not v0['iok'][ind]:
            continue

        sli = (slice(None),) + ind

        # --------
        # cross

        if lnc[ii] < nmaxc:
            iextra = np.linspace(0.1, 0.9, nmaxc - lnc[ii])
            indi = np.r_[0, iextra, np.arange(1, lnc[ii])].astype(int)

            pcross0[sli] = scpinterp.interp1d(
                range(lnc[ii]),
                lpoly_cross[ii][0],
                kind='linear',
                axis=0,
            )(indi)

            pcross1[sli] = scpinterp.interp1d(
                range(lnc[ii]),
                lpoly_cross[ii][1],
                kind='linear',
                axis=0,
            )(indi)

        else:
            pcross0[sli] = lpoly_cross[ii][0]
            pcross1[sli] = lpoly_cross[ii][1]

        # --------
        # hor

        if lnh[ii] < nmaxh:
            iextra = np.linspace(0.1, 0.9, nmaxh - lnh[ii])
            indi = np.r_[0, iextra, np.arange(1, lnh[ii])].astype(int)

            phor0[sli] = scpinterp.interp1d(
                range(lnh[ii]),
                lpoly_hor[ii][0],
                kind='linear',
                axis=0,
            )(indi)

            phor1[sli] = scpinterp.interp1d(
                range(lnh[ii]),
                lpoly_hor[ii][1],
                kind='linear',
                axis=0,
            )(indi)

        else:
            phor0[sli] = lpoly_hor[ii][0]
            phor1[sli] = lpoly_hor[ii][1]

    # ----------
    # store
    # ----------

    _vos_from_los_store(
        coll=coll,
        key=key,
        key_cam=key_cam,
        pcross0=pcross0,
        pcross1=pcross1,
        phor0=phor0,
        phor1=phor1,
        dphi=dphi,
        overwrite=overwrite,
    )


def _vos_from_los_store(
    coll=None,
    key=None,
    key_cam=None,
    pcross0=None,
    pcross1=None,
    phor0=None,
    phor1=None,
    dphi=None,
    overwrite=None,
):

    # --------
    # dref

    # keys
    knc = f'{key}_{key_cam}_vos_pc_n'
    knh = f'{key}_{key_cam}_vos_ph_n'

    # dict
    dref = {}
    if pcross0 is not None:
        dref[knc] = {'size': pcross0.shape[0]}
    if phor0 is not None:
        dref[knh] = {'size': phor0.shape[0]}

    # -------------
    # data

    # keys
    kpc0 = f'{key}_{key_cam}_vos_pc0'
    kpc1 = f'{key}_{key_cam}_vos_pc1'
    kph0 = f'{key}_{key_cam}_vos_ph0'
    kph1 = f'{key}_{key_cam}_vos_ph1'

    # reshape for 2d camera
    if coll.dobj['camera'][key_cam]['dgeom']['nd'] == '2d':
        shape0 = coll.dobj['camera'][key_cam]['dgeom']['shape']

        if pcross0 is not None:
            shape = tuple(np.r_[pcross0.shape[0], shape0])
            pcross0 = pcross0.reshape(shape)
            pcross1 = pcross1.reshape(shape)

        if phor0 is not None:
            shape = tuple(np.r_[phor0.shape[0], shape0])
            phor0 = phor0.reshape(shape)
            phor1 = phor1.reshape(shape)

    # ref
    refc = tuple([knc] + list(coll.dobj['camera'][key_cam]['dgeom']['ref']))
    refh = tuple([knh] + list(coll.dobj['camera'][key_cam]['dgeom']['ref']))

    # dict
    ddata = {}

    if pcross0 is not None:
        ddata.update({
            kpc0: {
                'data': pcross0,
                'ref': refc,
                'units': 'm',
                'dim': 'length',
                'quant': 'R',
            },
            kpc1: {
                'data': pcross1,
                'ref': refc,
                'units': 'm',
                'dim': 'length',
                'quant': 'Z',
            },
        })

    if phor0 is not None:
        ddata.update({
            kph0: {
                'data': phor0,
                'ref': refh,
                'units': 'm',
                'dim': 'length',
                'quant': 'X',
            },
            kph1: {
                'data': phor1,
                'ref': refh,
                'units': 'm',
                'dim': 'length',
                'quant': 'Y',
            },
        })

    # ----------
    # store

    # update
    for k0 in dref.keys():
        if k0 in coll.dref.keys():
            if overwrite is True:
                coll.remove_ref(k0, propagate=False)
            else:
                msg = (
                    "Storing vos from los for diag '{key}', camera '{keycam}':\n"
                    f"\t- ref '{k0}' already exists!\n"
                    "\t- Use overwrite=True to force overwriting\n"
                )
                raise Exception(msg)

    for k0 in ddata.keys():
        if k0 in coll.ddata.keys():
            if overwrite is True:
                coll.remove_data(k0, propagate=False)
            else:
                msg = (
                    "Storing vos from los for diag '{key}', camera '{keycam}':\n"
                    f"\t- data '{k0}' already exists!\n"
                    "\t- Use overwrite=True to force overwriting\n"
                )
                raise Exception(msg)

    coll.update(dref=dref, ddata=ddata)

    # add pcross
    doptics = coll._dobj['diagnostic'][key]['doptics']
    doptics[key_cam]['dvos'] = {
        'pcross': None if pcross0 is None else (kpc0, kpc1),
        'phor': None if phor0 is None else (kph0, kph1),
        'dphi': dphi,
    }


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

    if i0 == 0:
        msg = f"\tComputing angles for spectro diag '{key}':\n\t\t- cam '{key_cam}':"
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
        cxi = v0['cx'][ij] # + dx
        cyi = v0['cy'][ij] # + dy
        czi = v0['cz'][ij] # + dz

        nc = 1 # cxi.size

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
                    "Storing vos from los for diag '{key}', camera '{keycam}':\n"
                    f"\t- data '{k0}' already exists!\n"
                    "\t- Use overwrite=True to force overwriting\n"
                )
                raise Exception(msg)

    coll.update(ddata=ddata)

    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amin'] = kamin
    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amax'] = kamax
