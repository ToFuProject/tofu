# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt     # DB


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
    length=None,
    reflections_nb=None,
    reflections_type=None,
    key_nseg=None,
    # for vos based on los
    res=None,
    compute_vos_from_los=None,
    **kwdargs,
):

    # ------------
    # check inputs

    # key
    lok = list(coll.dobj.get('diagnostic', {}))
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    is2d = coll.dobj['diagnostic'][key]['is2d']

    # compute_vos_from_los
    compute_vos_from_los = ds._generic_check._check_var(
        compute_vos_from_los, 'compute_vos_from_los',
        types=bool,
        default=True,
    )

    # ---------------
    # loop on cameras

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
                res=res,
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
            )


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
    res=None,
):

    # --------------
    # check inputs

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        default=0.01,
        sign='>0',
    )

    # --------------
    # prepare

    # lspectro
    lspectro = [
        oo for oo in coll.dobj['diagnostic'][key]['doptics'][key_cam]['optics']
        if oo in coll.dobj.get('crystal', {}).keys()
        or oo in coll.dobj.get('grating', {}).keys()
    ]
    par = coll.dobj['camera'][key_cam]['dgeom']['parallel']

    if par:
        dx, dy, dz = coll.get_camera_dxyz(
            key=key_cam,
            include_center=True,
        )
    else:
        dgeom = coll.dobj['camera'][key_cam]['dgeom']
        out0 = coll.ddata[dgeom['outline'][0]]['data']
        out1 = coll.ddata[dgeom['outline'][1]]['data']
        ke0 = dgeom['e0']
        ke1 = dgeom['e1']


    # ----------
    # poly_cross

    lpoly_cross = []
    lpoly_hor = []
    lind = []
    npix = v0['cx'].size
    dphi = np.full((2, npix), np.nan)
    for ii in range(v0['cx'].size):

        if not v0['iok'][ii]:
            continue

        lind.append(ii)

        if not par:
            e0 = [coll.ddata[kk]['data'][ii] for kk in ke0]
            e1 = [coll.ddata[kk]['data'][ii] for kk in ke1]
            dx = out0 * e0[0] + out1 * e1[0]
            dy = out0 * e0[1] + out1 * e1[1]
            dz = out0 * e0[2] + out1 * e1[2]

        # get start / end points
        ptsx, ptsy, ptsz = _get_rays_from_pix(
            coll=coll,
            cx=v0['cx'][ii],
            cy=v0['cy'][ii],
            cz=v0['cz'][ii],
            x0=np.r_[v0['x0'][ii, :], v0['cents0'][ii]],
            x1=np.r_[v0['x1'][ii, :], v0['cents1'][ii]],
            dx=np.r_[0],
            dy=np.r_[0],
            dz=np.r_[0],
            coords=coll.get_optics_x01toxyz(key=v0['kref']),
            lspectro=lspectro,
            config=config,
        )

        # sampling
        length = np.sqrt(
            np.diff(ptsx, axis=0)**2
            + np.diff(ptsy, axis=0)**2
            + np.diff(ptsz, axis=0)**2
        )
        npts = int(np.ceil(np.nanmax(length) / res))
        ind = np.linspace(0, 1, npts)
        iok = np.all(np.isfinite(ptsx), axis=0)
        ptsx = scpinterp.interp1d(
            [0, 1],
            ptsx[:, iok],
            kind='linear',
            axis=0,
        )(ind).ravel()
        ptsy = scpinterp.interp1d(
            [0, 1],
            ptsy[:, iok],
            kind='linear',
            axis=0,
        )(ind).ravel()
        ptsz = scpinterp.interp1d(
            [0, 1],
            ptsz[:, iok],
            kind='linear',
            axis=0,
        )(ind).ravel()

        # dphi
        phi = np.arctan2(ptsy, ptsx)
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)
        if phimax - phimin > np.pi:
            phimin, phimax = phimax, phimin + 2.*np.pi
        dphi[:, ii] = (phimin, phimax)

        # poly_cross
        ptsr = np.hypot(ptsx, ptsy)
        convh = ConvexHull(np.array([ptsr, ptsz]).T)
        conv0 = ptsr[convh.vertices]
        conv1 = ptsz[convh.vertices]
        lpoly_cross.append((conv0, conv1))

        # poly_hor
        convh = ConvexHull(np.array([ptsx, ptsy]).T)
        conv0 = ptsx[convh.vertices]
        conv1 = ptsy[convh.vertices]
        lpoly_hor.append((conv0, conv1))

    # ------------------------
    # poly_cross harmonization

    lnc = [pp[0].size for pp in lpoly_cross]
    lnh = [pp[0].size for pp in lpoly_hor]
    nmaxc = np.max(lnc)
    nmaxh = np.max(lnh)
    pcross0 = np.full((nmaxc, npix), np.nan)
    pcross1 = np.full((nmaxc, npix), np.nan)
    phor0 = np.full((nmaxh, npix), np.nan)
    phor1 = np.full((nmaxh, npix), np.nan)
    for ii, indi in enumerate(lind):

        # cross
        if lnc[ii] < nmaxc:
            iextra = np.linspace(0.1, 0.9, nmaxc - lnc[ii])
            ind = np.r_[0, iextra, np.arange(1, lnc[ii])].astype(int)
            pcross0[:, indi] = scpinterp.interp1d(
                range(lnc[ii]),
                lpoly_cross[ii][0],
                kind='linear',
                axis=0,
            )(ind)
            pcross1[:, indi] = scpinterp.interp1d(
                range(lnc[ii]),
                lpoly_cross[ii][1],
                kind='linear',
                axis=0,
            )(ind)
        else:
            pcross0[:, indi] = lpoly_cross[ii][0]
            pcross1[:, indi] = lpoly_cross[ii][1]

        # hor
        if lnh[ii] < nmaxh:
            iextra = np.linspace(0.1, 0.9, nmaxh - lnh[ii])
            ind = np.r_[0, iextra, np.arange(1, lnh[ii])].astype(int)
            phor0[:, indi] = scpinterp.interp1d(
                range(lnh[ii]),
                lpoly_hor[ii][0],
                kind='linear',
                axis=0,
            )(ind)
            phor1[:, indi] = scpinterp.interp1d(
                range(lnh[ii]),
                lpoly_hor[ii][1],
                kind='linear',
                axis=0,
            )(ind)
        else:
            phor0[:, indi] = lpoly_hor[ii][0]
            phor1[:, indi] = lpoly_hor[ii][1]

    # ----------
    # store

    _vos_from_los_store(
        coll=coll,
        key=key,
        key_cam=key_cam,
        pcross0=pcross0,
        pcross1=pcross1,
        phor0=phor0,
        phor1=phor1,
        dphi=dphi,
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
    cx=None,
    cy=None,
    cz=None,
    x0=None,
    x1=None,
    dx=None,
    dy=None,
    dz=None,
    lspectro=None,
    coords=None,
    config=None,
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

    for ii, oo in enumerate(lspectro):

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
        strict=True,
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
):

    # ------------
    # prepare

    angmin = np.full(v0['cx'].size, np.nan)
    angmax = np.full(v0['cx'].size, np.nan)

    ptsvect = coll.get_optics_reflect_ptsvect(key=v0['kref'])
    coords = coll.get_optics_x01toxyz(key=v0['kref'])

    # dx, dy, dz = coll.get_camera_dxyz(
    #     key=key_cam,
    #     include_center=True,
    # )

    # ------
    # loop

    print(f"\tComputing angles for spectro diag '{key}':")

    # langles = []        # DB
    for ii in range(v0['cx'].size):

        # verb
        msg = f"\t\tpixel {ii+1} / {v0['cx'].size}"
        end = "\n" if ii == v0['cx'].size - 1 else "\r"
        print(msg, end=end, flush=True)

        if not v0['iok'][ii]:
            continue

        # get 3d coordiantes of points on pixel
        cxi = v0['cx'][ii] # + dx
        cyi = v0['cy'][ii] # + dy
        czi = v0['cz'][ii] # + dz

        nc = 1 # cxi.size

        # get 3d coords of points on crystal
        exi, eyi, ezi = coords(
            v0['x0'][ii, :],
            v0['x1'][ii, :],
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
        ang0 = np.nanmean(angles[:ne])
        # angles[ne:] = ang0 + 0.5*(angles[ne:] - ang0)

        angmin[ii] = np.nanmin(angles[:ne])
        angmax[ii] = np.nanmax(angles[:ne])
        # langles.append(angles)      # DB

    if is2d:
        angmin = angmin.reshape(v0['shape0'])
        angmax = angmax.reshape(v0['shape0'])

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
    coll.update(ddata=ddata)

    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amin'] = kamin
    coll._dobj['diagnostic'][key]['doptics'][key_cam]['amax'] = kamax