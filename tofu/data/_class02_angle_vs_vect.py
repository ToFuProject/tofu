

import numpy as np
from matplotlib.path import Path
import datastock as ds
import bsplines2d as bs2


# #################################################################
# #################################################################
#               Main
# #################################################################


def main(
    coll=None,
    # rays
    key_rays=None,
    res=None,
    segment=None,
    # vector components
    key_XR=None,
    key_YZ=None,
    key_Zphi=None,
    geometry=None,
    # optional separatrix
    key_sepR=None,
    key_sepZ=None,
    # options
    verb=None,
):

    # ----------------
    # check inputs
    # ----------------

    (
        key_rays,
        dkeys, geometry,
        key_sepR, key_sepZ,
        verb,
    ) = _check(
        coll=coll,
        key_rays=key_rays,
        # vector components
        key_XR=key_XR,
        key_YZ=key_YZ,
        key_Zphi=key_Zphi,
        geometry=geometry,
        # optional separatrix
        key_sepR=key_sepR,
        key_sepZ=key_sepZ,
    )
    wrays = coll._which_rays

    # ----------------
    # verb
    # ----------------

    if verb >= 1:
        lk = ['key_XR', 'key_YZ', 'key_Zphi']
        lstr = [
            f"\t - {kk}: {dkeys[kk]} {coll.ddata[dkeys[kk]]['shape']}"
            for kk in lk
        ]
        msg = "\nComputing rays_angle_vs_vect for:\n" + "\n".join(lstr) + "\n"
        print(msg)

    # ----------------
    # prepare sepR, sepZ
    # ----------------

    nstep = 4
    if key_sepR is not False:
        sepR, sepZ, sli_sep, ind_sep = _prepare_sep(
            coll=coll,
            key_sepR=key_sepR,
            key_sepZ=key_sepZ,
            dkeys=dkeys,
        )
        nstep += 1

    # ----------------
    # loop on rays
    # ----------------

    ddata = {}
    nrays = len(key_rays)
    for iray, kray in enumerate(key_rays):

        # ------------
        # verb

        if verb >= 1:
            sh = coll.dobj[wrays][kray]['shape']
            msg = f"\tFor kray = {kray} (shape {sh})... ({iray+1} / {nrays})"
            print(msg)

        # -------------
        # sample LOS

        # verb
        if verb >= 2:
            msg = f"\t\t- sampling... (1/{nstep})"
            print(msg)

        # sampel
        R, Z, phi, length = coll.sample_rays(
            key=kray,
            res=res,
            mode='abs',
            segment=segment,
            return_coords=['R', 'z', 'phi', 'l'],
        )

        # -------------
        # interpolate

        # verb
        if verb >= 2:
            msg = f"\t\t- interpolating... (2/{nstep})"
            print(msg)

        # interpolate
        dinterp = coll.interpolate(
            keys=[dkeys['key_XR'], dkeys['key_YZ'], dkeys['key_Zphi']],
            ref_key=None,
            x0=R,
            x1=Z,
            grid=False,
            details=False,
            val_out=np.nan,
            log_log=False,
            nan0=False,
            returnas=dict,
            store=False,
        )

        # ------------------
        # local unit vectors

        # verb
        if verb >= 2:
            msg = f"\t\t- deriving angle... (3/{nstep})"
            print(msg)

        # unit vectors
        ux, uy, uz = _local_unit_vect(
            phi=phi,
            vXR=dinterp[dkeys['key_XR']]['data'],
            vYZ=dinterp[dkeys['key_YZ']]['data'],
            vZphi=dinterp[dkeys['key_Zphi']]['data'],
            geometry=geometry,
        )

        # --------------
        # los vect

        vx, vy, vz = coll.get_rays_vect(kray, segment=segment)

        # reshape for broadcast
        axis = tuple(np.arange(0, ux.ndim - vx.ndim))
        vx = np.expand_dims(vx, axis)
        vy = np.expand_dims(vy, axis)
        vz = np.expand_dims(vz, axis)

        # --------------
        # compute angles

        angle = np.arccos(ux * (-vx) + uy * (-vy) + uz * (-vz))

        # --------------
        # sep ?

        if key_sepR is not None:

            # verb
            if verb >= 2:
                msg = f"\t\t- separatrix... (4/{nstep})"
                print(msg)

            # apply separatrix
            ref = _apply_sep(
                coll=coll,
                dinterp=dinterp,
                sepR=sepR,
                sepZ=sepZ,
                sli_sep=sli_sep,
                ind_sep=ind_sep,
                dkeys=dkeys,
                R=R,
                Z=Z,
                kray=kray,
                angle=angle,
            )

        # --------------
        # cleanup

        # verb
        if verb >= 2:
            msg = f"\t\t- cleanup... (5/{nstep})"
            print(msg)

        # angle
        axis = tuple([ii for ii, rr in enumerate(ref) if rr is not None])
        iok = np.any(np.isfinite(angle), axis=axis)
        sli = tuple([
            iok if rr is None else slice(None)
            for ii, rr in enumerate(ref)
        ])
        angle = angle[sli]

        # R, Z, phi, length
        R = R[iok, ...]
        Z = Z[iok, ...]
        phi = phi[iok, ...]
        length = length[iok, ...]

        # --------------
        # store

        refpts = (None,) + coll.dobj[wrays][kray]['ref'][1:]

        ddata[kray] = {
            'angle': {
                'key': None,
                'data': angle,
                'ref': ref,
                'dim': 'angle',
                'units': 'rad',
            },
            'length': {
                'key': None,
                'data': length,
                'ref': refpts,
                'dim': 'distance',
                'units': 'm',
            },
            'R': {
                'key': None,
                'data': R,
                'units': 'm',
                'dim': 'distance',
                'ref': refpts,
            },
            'Z': {
                'key': None,
                'data': Z,
                'units': 'm',
                'dim': 'distance',
                'ref': refpts,
            },
            'phi': {
                'key': None,
                'data': phi,
                'units': 'm',
                'dim': 'distance',
                'ref': refpts,
            },
        }

    return ddata


# #################################################################
# #################################################################
#               check
# #################################################################


def _check(
    coll=None,
    # rays
    key_rays=None,
    res=None,
    # vector components
    key_XR=None,
    key_YZ=None,
    key_Zphi=None,
    geometry=None,
    # optional separatrix
    key_sepR=None,
    key_sepZ=None,
    # optional
    verb=None,
):

    # -------------
    # key_rays
    # -------------

    wrays = coll._which_rays
    lok = list(coll.dobj.get(wrays, {}).keys())
    if isinstance(key_rays, str):
        key_rays = [key_rays]
    key_rays = ds._generic_check._check_var_iter(
        key_rays, 'key_rays',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
        default=lok,
    )

    # ------------------
    # geometry
    # ------------------

    geometry = ds._generic_check._check_var(
        geometry, 'geometry',
        types=str,
        default='toroidal',
        allowed=['toroidal'],
    )

    # ------------------
    # key vect coordinates
    # ------------------

    dkey = bs2._class02_line_tracing._check_keys_components(
        coll=coll,
        # 3 componants
        key_XR=key_XR,
        key_YZ=key_YZ,
        key_Zphi=key_Zphi,
    )

    # -------------------
    # Optional separatrix
    # -------------------

    # default => False
    if key_sepR is None:
        key_sepR = False
    if key_sepZ is None:
        key_sepZ = False

    # both xor none
    lc = [key_sepR is not False, key_sepZ is not False]
    if np.sum(lc) == 1:
        msg = (
            "Please provide either (xor):\n"
            "\t- both key_sepR and key_sepZ (True or str)\n"
            "\t- none of them (None or False)"
        )
        raise Exception(msg)

    # key_sepR
    key_sepR = _check_key_sep(coll, key_sepR, 'key_sepR')
    key_sepZ = _check_key_sep(coll, key_sepZ, 'key_sepZ')

    # cross-compatibility
    if key_sepR is not False:

        # same ref with each other
        ref_sep = coll.ddata[key_sepR]['ref']
        if ref_sep != coll.ddata[key_sepZ]['ref']:
            lstr = [
                f"\t- '{kk}': {coll.ddata[kk]['ref']}"
                for kk in [key_sepR, key_sepZ]
            ]
            msg = (
                "Args 'key_sepR' and 'key_sepZ' must share the same 'ref'!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        # shared ref with vector field
        ref_vect = coll.ddata[dkey['key_XR']]['ref']
        lout = [rr for rr in ref_sep if rr not in ref_vect]
        if len(lout) > 1:
            msg = (
            )
            raise Exception(msg)

    # -------------------
    # verb
    # -------------------

    lok = [False, True, 0, 1, 2]
    verb = int(ds._generic_check._check_var(
        verb, 'verb',
        types=(bool, int),
        default=lok[-1],
        allowed=lok,
    ))

    return (
        key_rays,
        dkey, geometry,
        key_sepR, key_sepZ,
        verb,
    )


def _check_key_sep(coll, key, keyn):

    # -----------------
    # True => automatic
    # -----------------

    if key is True:
        lk = [
            kk for kk in coll.ddata.keys()
            if kk.endswith(f"{keyn.split('_')[-1]}")
        ]
        if len(lk) == 1:
            key = lk[0]
        else:
            lstr = [f"\t- {kk}" for kk in lk]
            msg = (
                f"Arg '{keyn}' (True) could not be automatically identified\n"
                "No / several options:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # -----------------
    # str => check vs ddata
    # -----------------

    if key is not False:
        lok = [
            kk for kk, vv in coll.ddata.items()
        ]
        key = ds._generic_check._check_var(
            key, keyn,
            types=str,
            allowed=lok,
        )

    return key


# #################################################################
# #################################################################
#               Prepare sepR, sepZ
# #################################################################


def _prepare_sep(
    coll=None,
    key_sepR=None,
    key_sepZ=None,
    dkeys=None,
):

    # ------------
    # refs
    # ------------

    ref_sep = coll.ddata[key_sepR]['ref']
    ref_vect = coll.ddata[dkeys['key_XR']]['ref']

    # ------------
    # axis
    # ------------

    sepR = coll.ddata[key_sepR]['data']
    sepZ = coll.ddata[key_sepZ]['data']

    # -----------
    # slicing
    # -----------

    axis_pts = [ii for ii, rr in enumerate(ref_sep) if rr not in ref_vect][0]
    ind_sep = np.delete(np.arange(0, sepR.ndim), axis_pts).astype(int)
    sli_sep = np.array([
        0 if rr in ref_vect else slice(None)
        for rr in ref_sep
    ])

    return sepR, sepZ, sli_sep, ind_sep


# #################################################################
# #################################################################
#               get local unit vectors
# #################################################################


def _local_unit_vect(
    phi=None,
    vXR=None,
    vYZ=None,
    vZphi=None,
    geometry=None,
):

    # --------
    # linear
    # --------

    if geometry == 'linear':

        un = np.sqrt(vXR**2 + vYZ**2 + vZphi**2)
        ux = vXR / un
        uy = vYZ / un
        uz = vZphi / un

    # ----------
    # toroidal
    # ----------

    else:

        # --------------------
        # ux, uy from vR, vphi

        # associated unit vectors
        cosphif = np.cos(phi)[None, ...]
        sinphif = np.sin(phi)[None, ...]

        uX = vXR * cosphif - vZphi * sinphif
        uY = vXR * sinphif + vZphi * cosphif

        # ---------------
        # normalize

        un = np.sqrt(uX**2 + uY**2 + vYZ**2)
        ux = uX / un
        uy = uY / un
        uz = vYZ / un

    return ux, uy, uz


# #################################################################
# #################################################################
#               Apply sepR, sepZ
# #################################################################


def _apply_sep(
    coll=None,
    dinterp=None,
    dkeys=None,
    R=None,
    Z=None,
    sepR=None,
    sepZ=None,
    sli_sep=None,
    ind_sep=None,
    kray=None,
    angle=None,
):

    # -----------------------
    # prepare ref, shape, sli
    # -----------------------

    ref_interp = dinterp[dkeys['key_XR']]['ref']
    shape_interp = dinterp[dkeys['key_XR']]['data'].shape
    shape = tuple([
        ss for ii, ss in enumerate(shape_interp)
        if ref_interp[ii] is not None
    ])
    pts = np.array([R.ravel(), Z.ravel()]).T

    # sli
    refn = []
    for ii, rr in enumerate(ref_interp):
        if rr is not None or None not in refn:
            refn.append(rr)
    iang0 = tuple([ii for ii, rr in enumerate(refn) if rr is not None])
    iang1 = refn.index(None)
    sli_angle = [0 for rr in refn]

    # ref
    wrays = coll._which_rays
    ref_rays = coll.dobj[wrays][kray]['ref']
    ref, iN = [], 0
    for rr in ref_interp:
        if rr is None:
            if iN == 0:
                ref.append(rr)
            else:
                ref.append(ref_rays[iN])
            iN += 1
        else:
            ref.append(rr)

    # -------------------------------------
    # loop on indices to compute pts in sep
    # -------------------------------------

    for ii, ind in enumerate(np.ndindex(shape)):

        sli_sep[ind_sep] = ind
        sep = np.array([sepR[tuple(sli_sep)], sepZ[tuple(sli_sep)]]).T
        indout = ~Path(sep).contains_points(pts).reshape(R.shape)

        for ia, iin in zip(iang0, ind):
            sli_angle[ia] = iin
        sli_angle[iang1] = indout
        angle[tuple(sli_angle)] = np.nan

    return ref
