# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds


from ..geom import CamLOS1D


# ############################################################
# ############################################################
#                   Rays
# ############################################################


def _start_vect(ax=None, ay=None, az=None, name=None):
    dk = {
        'x': ax,
        'y': ay,
        'z': az,
    }
    sh = None
    for k0, v0 in dk.items():

        if v0 is None:
            msg = f"Arg '{name}_{k0}' cannot be None!"
            raise Exception(msg)

        dk[k0] = np.atleast_1d(v0).astype(float)
        if sh is None:
            sh = dk[k0].shape
        elif dk[k0].shape != sh:
            ls = [f"\t- {name}{k1}: {v1.shape}" for k1, v1 in dk.items()]
            msg = (
                f"Args {name}_x, {name}_y, {name}_z must have same shape!\n"
                "\n".join(ls)
            )
            raise Exception(msg)

    return dk['x'], dk['y'], dk['z']


def _check_inputs(
    coll=None,
    key=None,
    # start points
    start_x=None,
    start_y=None,
    start_z=None,
    # wavelength
    lamb=None,
    # ref
    ref=None,
    # from pts
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # from ray-tracing (vect + length or config or diag)
    vect_x=None,
    vect_y=None,
    vect_z=None,
    length=None,
    config=None,
    reflections_nb=None,
    reflections_type=None,
    key_nseg=None,
    diag=None,
    key_cam=None,
):

    # -------------
    # key

    wray = coll._which_rays
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get(wray, {}),
        short='ray',
        key=key,
    )

    # ------------
    # start points

    start_x, start_y, start_z = _start_vect(
        ax=start_x, ay=start_y, az=start_z, name='start',
    )
    shapes = [start_x.shape]

    # ------------
    # lamb

    if lamb is not None:
        lamb = np.atleast_1d(lamb).astype(float)

        if np.any(~np.isfinite(lamb)):
            msg = "Arg lamb must be all finite!"
            raise Exception(msg)

        if np.any(lamb <= 0.):
            msg = "Arg lamb must be all positive!"
            raise Exception(msg)

        shapes.append(lamb.shape)

    # ---------
    # shapes 1

    shref = None
    if len(shapes) == 1:
        if shapes[0] == (1,):
            pass
        else:
            shref = shapes[0]

    else:
        if shapes[0] == (1,) and shapes[1] == (1,):
            pass
        elif shapes[0] == shapes[1]:
            shref = shapes[0]
        elif (1,) in shapes:
            shref = shapes[1 - shapes.index((1,))]
        else:
            msg = (
                "Inconsistent shapes:\n"
                "\t- start_x: {shapes[0]}\n"
                "\t- lamb: {shapes[1]}\n"
            )
            raise Exception(msg)

    # ---------------------
    # pts vs config vs diag

    lkvpts = [('pts_x', pts_x), ('pts_y', pts_y), ('pts_z', pts_z)]
    lkvvect = [('vect_x', vect_x), ('vect_y', vect_y), ('vect_z', vect_z)]
    lkv_rt = [('config', config), ('length', length), ('diag', diag)]

    lc = [
        all([vv is not None for (kk, vv) in lkvpts]),
        all([vv is not None for (kk, vv) in lkvvect])
        and any([vv is not None for (kk, vv) in lkv_rt])
    ]
    if np.sum(lc) != 1:
        lstr0 = [f"\t\t- {kk} is None: {vv is None}" for (kk, vv) in lkvpts]
        lstr1 = [f"\t\t- {kk} is None: {vv is None}" for (kk, vv) in lkvvect]
        lstr2 = [f"\t\t- {kk} is None: {vv is None}" for (kk, vv) in lkv_rt]
        msg = (
            "Please provide either:\n"
            "\t- pts_x, pts_y, pts_z: to directly specify end points\n"
            + "\n".join(lstr0)
            + "\n\t- vect_x, vect_y, vect_z"
            + " and (config or length or diag): for ray-tracing\n"
            + "\n".join(lstr1)
            + "\n" + "\n".join(lstr2)
        )
        raise Exception(msg)

    # ------------
    # pts

    if lc[0]:

        pts_x, pts_y, pts_z = _start_vect(
            ax=pts_x, ay=pts_y, az=pts_z, name='pts',
        )

        if shref is None:
            if pts_x.ndim == 1:
                shref = pts_x.shape
                sh = tuple(np.r_[1, shref])
                pts_x = pts_x.reshape(sh)
                pts_y = pts_y.reshape(sh)
                pts_z = pts_z.reshape(sh)
            else:
                shref = pts_x.shape[1:]
                reflections_nb = pts_x.shape[0] - 1

        else:
            sh = tuple(np.r_[1, shref])
            if pts_x.shape == shref:
                pts_x = pts_x.reshape(sh)
                pts_y = pts_y.reshape(sh)
                pts_z = pts_z.reshape(sh)
            elif pts_x.ndim == len(shref) + 1 and pts_x.shape[1:] == shref:
                reflections_nb = pts_x.shape[0] - 1
            elif pts_x.shape == (1,):
                pts_x = np.full(shref, pts_x[0])
                pts_y = np.full(shref, pts_y[0])
                pts_z = np.full(shref, pts_z[0])
            else:
                msg = (
                    f"Arg pts_x has inconsistent shape:\n"
                    f"\t- pts_x.shape = {pts_x.shape}\n"
                    f"\t- shref: {shref}"
                )
                raise Exception(msg)

    # ------------
    # vectors

    elif lc[1]:

        vect_x, vect_y, vect_z = _start_vect(
            ax=vect_x, ay=vect_y, az=vect_z, name='vect',
        )

        # normalize
        norm = np.sqrt(vect_x**2 + vect_y**2 + vect_z**2)
        vect_x = vect_x / norm
        vect_y = vect_y / norm
        vect_z = vect_z / norm

        if shref is None:
            shref = vect_x.shape
        elif vect_x.shape not in [(1,), shref]:
            msg = (
                "Arg vect_x has inconsistent shape:\n"
                f"\t- vect_x.shape = {vect_x.shape}\n"
                f"\t- vs shape ref: {shref}\n"
            )
            raise Exception(msg)

    # ----------
    # config

    # ---------
    # reflections

    # reflections_nb
    if reflections_nb is None:
        reflections_nb = 0

    reflections_nb = ds._generic_check._check_var(
        reflections_nb, 'reflections_nb',
        types=int,
        sign='>= 0',
    )

    # reflections_type
    reflections_type = ds._generic_check._check_var(
        reflections_type, 'reflections_type',
        types=str,
        default='specular',
        allowed=['specular', 'diffusive'],
    )

    if reflections_nb > 0 and config is None and not lc[0]:
        msg = "reflections can only be handled if config is provided"
        raise Exception(msg)

    # key_nseg
    if key_nseg is not None:
        lok = list(coll.dref.keys())
        key_nseg = ds._generic_check._check_var(
            key_nseg, 'key_nseg',
            types=str,
            allowed=lok,
        )

    # ------
    # diag

    lspectro = None
    if diag is not None:
        lok = list(coll.dobj.get('diagnostic', {}).keys())
        diag = ds._generic_check._check_var(
            diag, 'diag',
            types=str,
            allowed=lok,
        )

        # key_cam
        lok = coll.dobj['diagnostic'][diag]['camera']
        key_cam = ds._generic_check._check_var(
            key_cam, 'key_cam',
            types=str,
            allowed=lok,
        )

        # lspectro
        loptics = coll.dobj['diagnostic'][diag]['doptics'][key_cam]['optics']
        lspectro = [
            oo for oo in loptics
            if oo in coll.dobj.get('crystal', {}).keys()
            or oo in coll.dobj.get('grating', {}).keys()
        ]

    return (
        key,
        start_x, start_y, start_z,
        lamb, ref,
        pts_x, pts_y, pts_z,
        vect_x, vect_y, vect_z,
        shref,
        config, reflections_nb, reflections_type,
        key_nseg,
        diag, key_cam, lspectro,
    )


# ############################################################
# ############################################################
#                   Main
# ############################################################


def _rays(
    coll=None,
    key=None,
    # start points
    start_x=None,
    start_y=None,
    start_z=None,
    # wavelength
    lamb=None,
    # ref
    ref=None,
    # from pts
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # from ray-tracing (vect + length or config or diag)
    vect_x=None,
    vect_y=None,
    vect_z=None,
    length=None,
    config=None,
    strict=None,
    # angles (if pre-computed)
    alpha=None,
    dalpha=None,
    dbeta=None,
    # reflections
    reflections_nb=None,
    reflections_type=None,
    key_nseg=None,
    diag=None,
    key_cam=None,
    # kwdargs
    kwdargs=None,
):

    # -------------
    # check inputs
    # -------------

    (
        key,
        start_x, start_y, start_z,
        lamb, ref,
        pts_x, pts_y, pts_z,
        vect_x, vect_y, vect_z,
        shaperef,
        config, reflections_nb, reflections_type,
        key_nseg,
        diag, key_cam, lspectro,
    ) = _check_inputs(
        coll=coll,
        key=key,
        # start points
        start_x=start_x,
        start_y=start_y,
        start_z=start_z,
        # wavelength
        lamb=lamb,
        # ref
        ref=ref,
        # from pts
        pts_x=pts_x,
        pts_y=pts_y,
        pts_z=pts_z,
        # from ray-tracing (vect + length or config or diag)
        vect_x=vect_x,
        vect_y=vect_y,
        vect_z=vect_z,
        length=length,
        config=config,
        reflections_nb=reflections_nb,
        reflections_type=reflections_type,
        key_nseg=key_nseg,
        diag=diag,
        key_cam=key_cam,
    )

    # ----------------
    # prepare
    # -------------

    # -----------------------------
    # compute from pts
    # ----------------------------

    if pts_x is not None:

        if pts_x.shape[0] == 1:
            pass
        else:
            v0x = pts_x[0:1, ...] - start_x
            v0y = pts_y[0:1, ...] - start_y
            v0z = pts_z[0:1, ...] - start_z
            vx = np.concatenate((v0x, np.diff(pts_x, axis=0)), axis=0)
            vy = np.concatenate((v0y, np.diff(pts_y, axis=0)), axis=0)
            vz = np.concatenate((v0z, np.diff(pts_z, axis=0)), axis=0)

            norm = np.sqrt(vx**2 + vy**2 + vz**2)
            vx = vx / norm
            vy = vy / norm
            vz = vz / norm

    # -----------------------------
    # compute ray-tracing
    # -----------------------------

    else:

        # final shape
        i0 = 0
        nbref = reflections_nb + 1
        if lspectro is not None and len(lspectro) > 0:
            i0 = len(lspectro)
            nbref += i0
        shape = tuple(np.r_[nbref, shaperef])

        # extract angles and pts
        pts_x = np.full(shape, np.nan)
        pts_y = np.full(shape, np.nan)
        pts_z = np.full(shape, np.nan)

        pts_x[0, ...] = start_x
        pts_y[0, ...] = start_y
        pts_z[0, ...] = start_z

        if reflections_nb >= 0 or diag is not None:
            alpha = np.full(shape, np.nan)
            dalpha = np.full(shape, np.nan)
            dbeta = np.full(shape, np.nan)

        stx = np.copy(start_x)
        sty = np.copy(start_y)
        stz = np.copy(start_z)

        # ----------
        # diag

        if diag is not None:

            for ii, oo in enumerate(lspectro):

                reflect_ptsvect = coll.get_optics_reflect_ptsvect(oo)
                (
                    pts_x[ii, ...],
                    pts_y[ii, ...],
                    pts_z[ii, ...],
                    vect_x, vect_y, vect_z,
                    alpha[ii, ...],
                    iok,
                ) = reflect_ptsvect(
                    pts_x=stx,
                    pts_y=sty,
                    pts_z=stz,
                    vect_x=vect_x,
                    vect_y=vect_y,
                    vect_z=vect_z,
                    strict=True,
                    return_x01=False,
                )

                # update start
                stx[...] = pts_x[ii, ...]
                sty[...] = pts_y[ii, ...]
                stz[...] = pts_z[ii, ...]

        # ----------
        # config

        if config is not None:

            # prepare D
            D = np.array([
                stx.ravel(),
                sty.ravel(),
                stz.ravel(),
            ])

            # prepare u
            if vect_x.size == 1 and stx.size > 1:
                uu = np.array([
                    np.full(stx.shape, vect_x[0]),
                    np.full(stx.shape, vect_y[0]),
                    np.full(stx.shape, vect_z[0]),
                ])

            else:
                uu = np.array([
                    vect_x.ravel(),
                    vect_y.ravel(),
                    vect_z.ravel(),
                ])

            mask = np.all(np.isfinite(uu), axis=0)
            maskre = np.reshape(mask, shaperef)

            # call legacy code
            cam = CamLOS1D(
                dgeom=(D[:, mask], uu[:, mask]),
                config=config,
                Name='',
                Diag='',
                Exp='',
                strict=strict,
            )

            # add reflections
            if reflections_nb > 0:
                cam.add_reflections(nb=reflections_nb)

            if cam.dgeom['dreflect'] is not None:
                dref = cam.dgeom['dreflect']
                pts_x[i0:-1, maskre] = dref['Ds'][0, ...].T
                pts_y[i0:-1, maskre] = dref['Ds'][1, ...].T
                pts_z[i0:-1, maskre] = dref['Ds'][2, ...].T

                pout = (
                    dref['Ds'][:, :, -1]
                    + dref['kouts'][None, :, -1] * dref['us'][:, :, -1]
                )
            else:
                pout = cam.dgeom['PkOut']

            pts_x[-1, maskre] = pout[0, :]
            pts_y[-1, maskre] = pout[1, :]
            pts_z[-1, maskre] = pout[2, :]

            # reset to nan where relevant
            pts_x[:, ~maskre] = np.nan
            pts_y[:, ~maskre] = np.nan
            pts_z[:, ~maskre] = np.nan

            # RMin
            # kRMin = _comp.LOS_PRMin(cam.D, cam.u, kOut=None)
            # PRMin = cam.D + kRMin[None, :]*cam.u
            # Rmin[maskre] = np.hypot(PRMin[0, :], PRMin[1, :])

            vperp = cam.dgeom['vperp']
            u_perp = np.sum(cam.u*vperp, axis=0)

            alpha[i0, maskre] = np.arcsin(-u_perp)

            if cam.dgeom['dreflect'] is not None:
                us = dref['us'][..., 0]

                u_perp = np.sum(us * vperp, axis=0)
                dalpha[i0, maskre] = np.arcsin(u_perp) - alpha[0, maskre]

                # beta
                u0 = us - u_perp[None, ...] * vperp
                e0 = u0 / np.linalg.norm(u0)[None, ...]
                e1 = np.array([
                    vperp[1]*e0[2] - vperp[2]*e0[1],
                    vperp[2]*e0[0] - vperp[0]*e0[2],
                    vperp[0]*e0[1] - vperp[1]*e0[0],
                ])
                v0 = np.sum(us * e0, axis=0)
                v1 = np.sum(us * e1, axis=0)
                dbeta[i0, maskre] = np.arctan2(v1, v0)

        # ----------
        # length

        elif length is not None:

            shape = tuple(np.r_[1., shaperef])
            pts_x = (start_x + length * vect_x).reshape(shape)
            pts_y = (start_y + length * vect_y).reshape(shape)
            pts_z = (start_z + length * vect_z).reshape(shape)

    # --------------------------------
    # store
    # --------------------------------

    return _make_dict(
        coll=coll,
        key=key,
        pts_x=pts_x,
        pts_y=pts_y,
        pts_z=pts_z,
        start_x=start_x,
        start_y=start_y,
        start_z=start_z,
        alpha=alpha,
        dalpha=dalpha,
        dbeta=dbeta,
        lamb=lamb,
        lspectro=lspectro,
        diag=diag,
        reflections_nb=reflections_nb,
        key_nseg=key_nseg,
        ref=ref,
        shaperef=shaperef,
        kwdargs=kwdargs,
    )


# ###########################################################
# ###########################################################
#                   Rays - check key
# ###########################################################


def _check_key(coll=None, key=None, key_cam=None):

    # check key
    lrays = list(coll.dobj.get(coll._which_rays, {}).keys())
    ldiag = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([
                v1.get('los') is not None
                and v1['los'] in lrays
                for k1, v1 in v0['doptics'].items()
            ])
    ]

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lrays + ldiag,
    )

    # Derive kray
    if key in lrays:
        kray = key
    else:

        # key_cam
        lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
        key_cam = ds._generic_check._check_var(
            key_cam, 'key_cam',
            types=str,
            allowed=lok,
        )

        kray = coll.dobj['diagnostic'][key]['doptics'][key_cam]['los']

    return kray


# #########################################################
# #########################################################
#                   Rays - check segment
# #########################################################


def _check_segment(coll, key, segment):

    if segment is not None:

        wrays = coll._which_rays
        nseg = coll.dobj[wrays][key]['shape'][0]
        lok = list(range(nseg)) + [-1]
        segment = int(ds._generic_check._check_var(
            segment, 'segment',
            types=(int, float),
        ))
        if segment not in lok:
            msg = (
                f"Arg segment for rays '{key}' must be in:\n"
                f"\t- allowed: {lok}\n"
                f"\t- Provided: {segment}\n"
            )
            raise Exception(msg)

    return segment


# #########################################################
# #########################################################
#                   Rays - get start and vect
# #########################################################


def _get_start(
    coll=None,
    key=None,
    key_cam=None,
):

    # ---------
    # check key

    key = _check_key(coll=coll, key=key, key_cam=key_cam)

    # ---------------
    # get start

    wrays = coll._which_rays
    stx, sty, stz = coll.dobj[wrays][key]['start']
    if isinstance(stx, str):
        stx = coll.ddata[stx]['data']
        sty = coll.ddata[sty]['data']
        stz = coll.ddata[stz]['data']

    return stx, sty, stz


def _get_pts(
    coll=None,
    key=None,
    key_cam=None,
    segment=None,
):

    # ---------
    # check key

    key = _check_key(coll=coll, key=key, key_cam=key_cam)
    segment = _check_segment(coll, key, segment)

    # ---------
    # get start

    stx, sty, stz = _get_start(coll=coll, key=key)

    # ---------------
    # get other pts

    wrays = coll._which_rays
    ptsx, ptsy, ptsz = coll.dobj[wrays][key]['pts']
    ptsx = coll.ddata[ptsx]['data']
    ptsy = coll.ddata[ptsy]['data']
    ptsz = coll.ddata[ptsz]['data']

    # concatenate
    if segment is None:
        ptsx = np.insert(ptsx, 0, stx, axis=0)
        ptsy = np.insert(ptsy, 0, sty, axis=0)
        ptsz = np.insert(ptsz, 0, stz, axis=0)

    elif segment == 0:
        ptsx = np.insert(ptsx[0:1, ...], 0, stx, axis=0)
        ptsy = np.insert(ptsy[0:1, ...], 0, sty, axis=0)
        ptsz = np.insert(ptsz[0:1, ...], 0, stz, axis=0)

    else:
        ptsx = ptsx[segment-1:segment+1, ...]
        ptsy = ptsy[segment-1:segment+1, ...]
        ptsz = ptsz[segment-1:segment+1, ...]

    return ptsx, ptsy, ptsz


def _get_vect(
    coll=None,
    key=None,
    key_cam=None,
    norm=None,
    segment=None,
):

    # ---------
    # check key

    key = _check_key(coll=coll, key=key, key_cam=key_cam)
    segment = _check_segment(coll, key, segment)

    # norm
    norm = ds._generic_check._check_var(
        norm, 'norm',
        types=bool,
        default=True,
    )

    # ---------------
    # get start

    stx, sty, stz = _get_start(coll=coll, key=key)

    wrays = coll._which_rays
    ptsx, ptsy, ptsz = coll.dobj[wrays][key]['pts']
    ptsx = coll.ddata[ptsx]['data']
    ptsy = coll.ddata[ptsy]['data']
    ptsz = coll.ddata[ptsz]['data']

    vx = (ptsx[0, ...] - stx)[None, ...]
    vy = (ptsy[0, ...] - sty)[None, ...]
    vz = (ptsz[0, ...] - stz)[None, ...]
    if ptsx.shape[0] > 1:
        vx = np.concatenate((vx, np.diff(ptsx, axis=0)), axis=0)
        vy = np.concatenate((vy, np.diff(ptsy, axis=0)), axis=0)
        vz = np.concatenate((vz, np.diff(ptsz, axis=0)), axis=0)

    # normalize
    if norm is True:
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        vx = vx / norm
        vy = vy / norm
        vz = vz / norm

    # segment
    if segment is not None:
        vx = vx[segment, ...]
        vy = vy[segment, ...]
        vz = vz[segment, ...]

    return vx, vy, vz


# ##########################################################
# ##########################################################
#                   store
# ##########################################################


def _make_dict(
    coll=None,
    key=None,
    pts_x=None,
    pts_y=None,
    pts_z=None,
    start_x=None,
    start_y=None,
    start_z=None,
    alpha=None,
    dalpha=None,
    dbeta=None,
    lamb=None,
    lspectro=None,
    diag=None,
    reflections_nb=None,
    key_nseg=None,
    ref=None,
    shaperef=None,
    # kwdargs
    kwdargs=None,
):

    # --------
    # dref

    shape = pts_x.shape
    nseg = shape[0]
    nextra = len(lspectro) if diag is not None else 0
    ntot = reflections_nb + 1 + nextra

    if nseg != ntot:
        msg = (
            f"Trying to add_rays('{key}')\n"
            "Mismatch between number of segments and pts_x.shape!\n"
            f"\t- pts_x.shape[0] = {pts_x.shape[0]}\n"
            f"\t- reflections_nb + 1 + len(lspectro) = {ntot}\n"
        )
        raise Exception(msg)

    assert nseg == reflections_nb + 1 + nextra

    # key_nseg
    if key_nseg is None:
        knseg = f'{key}_nseg'
        dref = {
            knseg: {'size': nseg},
        }
    else:
        if coll.dref[key_nseg]['size'] != nseg:
            msg = (
                "Wrong size of key_nseg:\n"
                f"\t- dref['key_nseg']['size] = coll.dref[key_nseg]['size']\n"
                f"\t- nseg = {nseg}"
            )
            raise Exception(msg)

        knseg = key_nseg
        dref = {}

    # if ref is None
    if ref is None:
        ref = []
        for ii, ss in enumerate(shaperef):
            kr = f'{key}_n{ii}'
            dref.update({
                kr: {'size': ss}
            })
            ref.append(kr)
        ref = tuple(ref)

    refpts = tuple(np.r_[(knseg,), ref])

    #  --------
    # ddata

    # pts
    kpx = f'{key}_ptx'
    kpy = f'{key}_pty'
    kpz = f'{key}_ptz'
    # kRmin = f'{key}-Rmin'

    ddata = {
        kpx: {
            'data': pts_x,
            'ref': refpts,
            'dim': 'distance',
            'quant': 'distance',
            'name': 'x',
            'units': 'm',
        },
        kpy: {
            'data': pts_y,
            'ref': refpts,
            'dim': 'distance',
            'quant': 'distance',
            'name': 'y',
            'units': 'm',
        },
        kpz: {
            'data': pts_z,
            'ref': refpts,
            'dim': 'distance',
            'quant': 'distance',
            'name': 'z',
            'units': 'm',
        },
        # kRmin: {
        #     'data': Rmin,
        #     'ref': refpts[1:],
        #     'dim': 'distance',
        #     'quant': 'distance',
        #     'name': 'R',
        #     'units': 'm',
        # },
    }

    # start
    if start_x.shape != (1,):
        ksx = f'{key}_startx'
        ksy = f'{key}_starty'
        ksz = f'{key}_startz'
        if ref is None:
            ref = None

        ddata.update({
            ksx: {
                'data': start_x,
                'ref': ref,
                'dim': 'distance',
                'quant': 'distance',
                'name': 'x',
                'units': 'm',
            },
            ksy: {
                'data': start_y,
                'ref': ref,
                'dim': 'distance',
                'quant': 'distance',
                'name': 'y',
                'units': 'm',
            },
            ksz: {
                'data': start_z,
                'ref': ref,
                'dim': 'distance',
                'quant': 'distance',
                'name': 'z',
                'units': 'm',
            },
        })

    # alpha
    kalpha = None
    if alpha is not None:
        kalpha = f'{key}_alpha'
        ddata.update({
            kalpha: {
                'data': alpha,
                'ref': refpts,
                'dim': 'angle',
                'quant': 'angle',
                'name': 'alpha',
                'units': 'rad',
            },
        })

    # dalpha
    kdalpha = None
    if dalpha is not None:
        kdalpha = f'{key}_dalpha'
        ddata.update({
            kdalpha: {
                'data': dalpha,
                'ref': refpts,
                'dim': 'angle',
                'quant': 'angle',
                'name': 'alpha',
                'units': 'rad',
            },
        })

    # dbeta
    kdbeta = None
    if dbeta is not None:
        kdbeta = f'{key}_dbeta'
        ddata.update({
            kdbeta: {
                'data': dbeta,
                'ref': refpts,
                'dim': 'angle',
                'quant': 'angle',
                'name': 'beta',
                'units': 'rad',
            },
        })

    # ------
    # dobj

    # start
    if start_x.shape == (1,):
        start = np.r_[start_x, start_y, start_z]
    else:
        start = (ksx, ksy, ksz)

    # lamb
    if lamb is not None:
        if lamb.shape == (1,):
            ll = lamb
        else:
            ll = lamb
    else:
        ll = lamb

    # pts
    pts = (kpx, kpy, kpz)

    # dobj
    dobj = {
        coll._which_rays: {
            key: {
                'start': start,
                'pts': pts,
                'lamb': ll,
                'shape': shape,
                'ref': refpts,
                # 'Rmin': kRmin,
                'alpha': kalpha,
                'reflect_dalpha': kdalpha,
                'reflect_dbeta': kdbeta,
                **kwdargs,
            },
        },
    }

    return dref, ddata, dobj


# ##################################################################
# ##################################################################
#                   Rays - remove
# ##################################################################


def _remove(
    coll=None,
    key=None,
):

    # ----------
    # check

    wrays = coll._which_rays
    lok = list(coll.dobj.get(wrays, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # ------------
    # list data

    lkd = [
        'start', 'pts', 'lamb',
        'alpha', 'reflect_dalpha', 'reflect_dbeta',
    ]
    ld = []
    for k0 in lkd:
        v0 = coll._dobj[wrays][key][k0]
        if isinstance(v0, str):
            ld.append(v0)
        elif isinstance(v0, (tuple, list)):
            ld += list(v0)
        elif v0 is None:
            pass
        else:
            msg = "How to handle rays['{key}']['{k0}'] = {v0}"
            raise Exception(msg)

    # -----------
    # remove data

    coll.remove_data(ld, propagate=True)

    # -----------
    # remove rays

    del coll._dobj[wrays][key]
