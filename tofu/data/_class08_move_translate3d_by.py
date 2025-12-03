

from . import _class08_move3d_check as _check


# ###########################################
# ###########################################
#            Main
# ###########################################


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
        length, vect_xyz,
    ) = _check.main(move='translate', **locals())

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
        kcam_new = _translate_camera(
            coll=coll,
            kcam=kcam,
            length=length[kcam],
            vect_xyz=vect_xyz[kcam],
            key_new=key_new,
        )

        # ------------
        # optics

        lop_new = []
        doptics[kcam_new] = {}
        for kop in doptics0[kcam]['optics']:

            # translate
            kop_new = _translate_optics(
                coll=coll,
                kop=kop,
                length=length[kcam],
                vect_xyz=vect_xyz[kcam],
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
        for ii, kk in enumerate(['poly_x', 'poly_y', 'poly_z']):
            dgeom[kk] = (
                dgeom0[kk] + length * vect_xyz[ii]
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
