

import numpy as np
import datastock as ds


from . import _class8_check


# ##############################################################
# ##############################################################
#                 Main
# ##############################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    # new diag
    key_new=None,
    # move params
    vect_xyz=None,
    length=None,
    # computing
    compute=None,
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
        length, vect_xyz,
    ) = _check(**locals())

    # ----------
    # prepare
    # ----------

    wdiag = coll._which_diagnostic
    doptics0 = coll.dobj[wdiag][key]['doptics']

    # ----------
    # compute
    # ----------

    doptics = {kcam: [] for kcam in key_cam}
    for kcam in key_cam:

        # ------------
        # optics

        for kop in doptics0[kcam]['optics']:

            # translate
            key = _translate_optics(
                coll=coll,
                kop=kop,
                length=length[kcam],
                vect_xyz=vect_xyz[kcam],
            )

            # doptics
            doptics[kcam].append(key)

        # ------------
        # camera

        # translate
        key = _translate_camera(
            coll=coll,
            kcam=kcam,
            length=length[kcam],
            vect_xyz=vect_xyz[kcam],
            key_new=key_new,
        )

    # ------------------
    # add diagnostic
    # ------------------

    coll.add_diagnostic(
        key=key_new,
        doptics=doptics,
        compute=compute,
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


# ##############################################################
# ##############################################################
#                 Check
# ##############################################################


def _check(
    coll=None,
    key=None,
    key_cam=None,
    # new diag
    key_new=None,
    # move params
    vect_xyz=None,
    length=None,
    # unused
    **kwdargs,
):

    # --------------
    # key, key_cam
    # --------------

    key, key_cam = _class8_check._get_default_cam(
        coll=coll,
        key=key,
        key_cam=key_cam,
        default='all',
    )

    # --------------
    # key_new
    # --------------

    wdiag = coll._which_diagnostic
    lout = list(coll.dobj.get(wdiag, {}).keys())
    key_new = ds._generic_check._check_var(
        key_new, 'key_new',
        types=str,
        default=f"{key}_translate",
        excluded=lout,
    )

    # --------------
    # translation
    # --------------

    # length
    length = _check_length_vect(
        din=length,
        key_cam=key_cam,
        dval=np.r_[0.],
        name='length',
    )

    # vect_xyz
    vect_xyz = _check_length_vect(
        din=vect_xyz,
        key_cam=key_cam,
        dval=np.r_[0., 0., 0.],
        name='vect_xyz',
    )

    return (
        key, key_cam, key_new,
        length, vect_xyz,
    )


def _check_length_vect(
    din=None,
    key_cam=None,
    dval=None,
    name=None,
):

    # -----------
    # scalar
    # -----------

    size = dval.size

    if din is None:
        din = dval

    if np.isscalar(din):
        if not np.isfinite(din):
            msg = "Arg din must be a finite scalar!\nProvided: {din}\n"
            raise Exception(msg)
        din = {kcam: np.full((size,), din) for kcam in key_cam}

    elif isinstance(din, (np.ndarray, list, tuple)):
        if len(din) != size:
            msg = (
                f"Arg '{name}' must be of size = {size}\n"
                f"Provided: {din}\n"
            )
            raise Exception(msg)
        din = np.r_[din]

    # -----------
    # check dict
    # -----------

    c0 = (
        isinstance(din, dict)
        and all([
            kk in key_cam
            and (
                din[kk] is None
                or (
                    np.all(np.isfinite(din[kk]))
                    and np.r_[din[kk]].size == size
                )
            )
            for kk in din.keys()
        ])
    )
    if not c0:
        msg = (
            f"Arg '{name}' must be a dict with:\n"
            f"\t - keys in {key_cam}\n"
            f"\t- values ({size},) np.ndarray"
        )
        if size == 1:
            msg += "(or scalar)"
        msg += f"\nProvided: {din}\n"
        raise Exception(msg)

    # -----------
    # fill dict
    # -----------

    for kcam in key_cam:
        if din.get(kcam) is None:
            din[kcam] = dval

        # scalar
        if size == 1:
            din[kcam] = din[kcam][0]

    return din


# ##############################################################
# ##############################################################
#                 Translate optics
# ##############################################################


def _translate_optics(
    coll=None,
    kop=None,
    length=None,
    vect_xyz=None,
    key_new=None,
):

    # --------------
    # extract
    # --------------

    kop, opcls = coll.get_optics_cls(kop)
    kop, opcls = kop[0], opcls[0]
    dgeom0 = coll.dobj[opcls][kop]['dgeom']

    # asis
    lk_asis = [
        'nin', 'e0', 'e1',
    ]

    # -----------
    # dgeom
    # -----------

    dgeom = {}
    if dgeom0.get('cent') is not None:
        dgeom['cent'] = dgeom0['cent'] + length * vect_xyz
        lk_asis.append(('outline_x0', 'outline_x1'))

    if dgeom0.get('poly_x') is not None:
        for kk in ['poly_x', 'poly_y', 'poly_z']:
            dgeom[kk] = (
                dgeom0[kk] + length * vect_xyz
            )

    # ----------------
    # add as-is
    # ----------------

    _add_asis(
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


# ##############################################################
# ##############################################################
#                 Translate camera
# ##############################################################


def _translate_camera(
    coll=None,
    kcam=None,
    length=None,
    vect_xyz=None,
    key_new=None,
):

    # --------------
    # extract
    # --------------

    wcam = coll._which_cam
    dgeom0 = coll.dobj[wcam][kcam]['dgeom']

    # asis
    lk_asis = [
        'nin', 'e0', 'e1',
        'nin_x', 'nin_y', 'nin_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
        ('outline_x0', 'outline_x1'),
    ]

    # -----------
    # dgeom
    # -----------

    dgeom = {}
    if dgeom0.get('cent') is not None:
        dgeom['cent'] = dgeom0['cent'] + length * vect_xyz

        lk_asis += ['cents_x0', 'cents_x1']

    if dgeom0.get('cents') is not None:
        for ik, kk in enumerate(['cents_x', 'cents_y', 'cents_z']):
            dgeom[kk] = (
                coll.ddata[dgeom0['cents'][ik]]['data']
                + length * vect_xyz[ik]
            )

    # ----------------
    # add as-is
    # ----------------

    _add_asis(
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


def _add_asis(
    coll=None,
    dgeom0=None,
    dgeom=None,
    lk_asis=None,
):

    for kk in lk_asis:
        if isinstance(kk, tuple):
            for ii, ki in enumerate(kk):
                k0 = ki.split('_')[0]
                if dgeom0.get(k0) is not None:
                    dgeom[ki] = coll.ddata[dgeom0[k0][ii]]['data']

        elif dgeom0.get(kk) is not None:

            if isinstance(dgeom0[kk], str):
                dgeom[kk] = coll.ddata[kk]['data']

            else:
                if dgeom0.get(kk) is not None:
                    dgeom[kk] = dgeom0[kk]

    return
