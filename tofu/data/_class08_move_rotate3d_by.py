

import numpy as np
import datastock as ds


from . import _class08_move3d_check as _check


# #################################################
# #################################################
#           Main
# #################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    # new diag
    key_new=None,
    # move params
    axis_pt=None,
    axis_vect=None,
    angle=None,
    # computing
    compute=None,
    strict=None,
    # los
    config=None,
    key_nseg=None,
    # equivalent aperture
    add_points=None,
    convex=None,
    # etendue
    margin_par=None,
    margin_perp=None,
    verb=None,
    # unused
    **kwdargs,
):

    # ----------
    # dinput
    # ----------

    (
        key, key_cam, key_new,
        angle, axis_pt, axis_vect,
    ) = _check.main(move='rotate', **locals())

    # ----------
    # prepare
    # ----------

    wdiag = coll._which_diagnostic
    doptics0 = coll.dobj[wdiag][key]['doptics']

    # ----------
    # compute
    # ----------

    doptics = {}
    for kcam in key_cam:

        # ------------
        # camera

        # translate
        kcam_new = _rotate_camera(
            coll=coll,
            kcam=kcam,
            axis_pt=axis_pt[kcam],
            axis_vect=axis_vect[kcam],
            angle=angle[kcam],
            key_new=key_new,
        )

        # ------------
        # optics

        lop_new = []
        doptics[kcam_new] = {}
        for kop in doptics0[kcam]['optics']:

            # translate
            kop_new = _rotate_optics(
                coll=coll,
                kop=kop,
                axis_pt=axis_pt[kcam],
                axis_vect=axis_vect[kcam],
                angle=angle[kcam],
                key_new=key_new,
            )
            lop_new.append(kop_new)

        # doptics
        doptics[kcam_new]['optics'] = lop_new
        doptics[kcam_new]['paths'] = doptics0[kcam]['paths']

    # ------------------
    # add diagnostic
    # ------------------

    coll.add_diagnostic(
        key=key_new,
        doptics=doptics,
        compute=compute,
        strict=strict,
        config=config,
        key_nseg=key_nseg,
        # equivalent aperture
        add_points=add_points,
        convex=convex,
        # etendue
        margin_par=margin_par,
        margin_perp=margin_perp,
        verb=verb,
    )

    return


# ###########################################
# ###########################################
#          Translate optics
# ###########################################


def _rotate_optics(
    coll=None,
    kop=None,
    axis_pt=None,
    axis_vect=None,
    angle=None,
    key_new=None,
):

    # --------------
    # extract
    # --------------

    kop, opcls = coll.get_optics_cls(kop)
    kop, opcls = kop[0], opcls[0]
    dgeom0 = coll.dobj[opcls][kop]['dgeom']

    # asis
    lk_asis = []

    # -----------
    # dgeom
    # -----------

    dgeom = {}
    if dgeom0.get('cent') is not None:
        dgeom['cent'] = np.r_[_rotate_pts(
            axis_pt,
            axis_vect,
            angle,
            *dgeom0['cent'],
        )]
        lk_asis.append(('outline_x0', 'outline_x1'))

    if dgeom0.get('poly_x') is not None:
        dgeom['poly_x'], dgeom['poly_y'], dgeom['poly_z'] = _rotate_pts(
            axis_pt,
            axis_vect,
            angle,
            dgeom0['poly_x'],
            dgeom0['poly_y'],
            dgeom0['poly_z'],
        )

    # unit vects
    for kk in ['nin', 'e0', 'e1']:
        if dgeom0.get(kk) is not None:
            dgeom[kk] = np.r_[_rotate_pts(
                axis_pt,
                axis_vect,
                angle,
                *dgeom0[kk],
                isvect=True,
            )]

    # ----------------
    # add as-is
    # ----------------

    _check._add_asis(
        coll=coll,
        dgeom0=dgeom0,
        dgeom=dgeom,
        lk_asis=lk_asis,
    )

    # ------------
    # key
    # ------------

    key = f'{kop}_{key_new}'

    # ------------
    # add to coll
    # ------------

    if opcls == 'aperture':
        coll.add_aperture(key=key, **dgeom)
    else:
        getattr(coll, f"add_{opcls}")(
            key=key,
            dgeom=dgeom,
        )

    return key


# #############################################
# #############################################
#              Translate camera
# #############################################


def _rotate_camera(
    coll=None,
    kcam=None,
    axis_pt=None,
    axis_vect=None,
    angle=None,
    key_new=None,
):

    # --------------
    # extract
    # --------------

    wcam = coll._which_cam
    dgeom0 = coll.dobj[wcam][kcam]['dgeom']

    # asis
    lk_asis = [
        ('outline_x0', 'outline_x1'),
    ]

    # -----------
    # dgeom
    # -----------

    dgeom = {}
    if dgeom0.get('cent') is not None:
        dgeom['cent'] = np.r_[_rotate_pts(
            axis_pt,
            axis_vect,
            angle,
            *dgeom0['cent'],
        )]
        lk_asis += ['cents_x0', 'cents_x1']

    if dgeom0.get('cents') is not None:
        dgeom['cents_x'], dgeom['cents_y'], dgeom['cents_z'] = _rotate_pts(
            axis_pt,
            axis_vect,
            angle,
            coll.ddata[dgeom0['cents'][0]]['data'],
            coll.ddata[dgeom0['cents'][1]]['data'],
            coll.ddata[dgeom0['cents'][2]]['data'],
        )

    # unit vects
    lk_vect = ['nin', 'e0', 'e1']
    for kk in lk_vect:
        if dgeom0.get(kk) is not None:
            dgeom[kk] = np.r_[_rotate_pts(
                axis_pt,
                axis_vect,
                angle,
                *dgeom0[kk],
                isvect=True,
            )]

        if dgeom0.get(f"{kk}_x") is not None:
            kx, ky, kz = f"{kk}_x", f"{kk}_y", f"{kk}_z"
            dgeom[kx], dgeom[ky], dgeom[kz] = _rotate_pts(
                axis_pt,
                axis_vect,
                angle,
                dgeom0[kx],
                dgeom0[ky],
                dgeom0[kz],
                isvect=True,
            )

    # ----------------
    # add as-is
    # ----------------

    _check._add_asis(
        coll=coll,
        dgeom0=dgeom0,
        dgeom=dgeom,
        lk_asis=lk_asis,
    )

    # ------------
    # key
    # ------------

    key = f'{kcam}_{key_new}'

    # ------------
    # add to coll
    # ------------

    if dgeom0['nd'] == '1d':
        coll.add_camera_1d(
            key=key,
            dgeom=dgeom,
        )
    else:
        coll.add_camera_2d(
            key=key,
            dgeom=dgeom,
        )

    return key


# #############################################
# #############################################
#           Rotate pts
# #############################################


def _rotate_pts(axis_pt, axis_vect, angle, xx, yy, zz, isvect=None):

    # --------
    # isvect
    # --------

    isvect = ds._generic_check._check_var(
        isvect, 'isvect',
        types=bool,
        default=False,
    )

    # --------
    # local unit vects
    # --------

    axis_vect = axis_vect / np.linalg.norm(axis_vect)

    if np.abs(axis_vect[2]) > 0.90:
        ecross = np.r_[1, 0, 0]
    else:
        ecross = np.r_[0, 0, 1]
    e0 = np.cross(axis_vect, ecross)

    e1 = np.cross(axis_vect, e0)
    e1 = e1 / np.linalg.norm(e1)

    # --------
    # local coords
    # --------

    if isvect is True:
        cent0 = np.r_[0, 0, 0]
    else:
        cent0 = axis_pt

    # axis
    axial = (
        (xx - cent0[0]) * axis_vect[0]
        + (yy - cent0[1]) * axis_vect[1]
        + (zz - cent0[2]) * axis_vect[2]
    )

    # x0
    x0 = (
        (xx - cent0[0]) * e0[0]
        + (yy - cent0[1]) * e0[1]
        + (zz - cent0[2]) * e0[2]
    )

    # x1
    x1 = (
        (xx - cent0[0]) * e1[0]
        + (yy - cent0[1]) * e1[1]
        + (zz - cent0[2]) * e1[2]
    )

    rr = np.hypot(x0, x1)
    theta = np.arctan2(x1, x0)

    # --------
    # rotate
    # --------

    x02 = rr * np.cos(theta + angle)
    x12 = rr * np.sin(theta + angle)

    # --------
    # pts new
    # --------

    xx_new = cent0[0] + axial * axis_vect[0] + x02 * e0[0] + x12 * e1[0]
    yy_new = cent0[1] + axial * axis_vect[1] + x02 * e0[1] + x12 * e1[1]
    zz_new = cent0[2] + axial * axis_vect[2] + x02 * e0[2] + x12 * e1[2]

    if isvect is True:
        assert np.allclose(np.sqrt(xx_new**2 + yy_new**2 + zz_new**2), 1.)

    return xx_new, yy_new, zz_new
