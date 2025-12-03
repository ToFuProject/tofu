

import numpy as np
import datastock as ds


from . import _class8_check


# ##################################################
# ##################################################
#           Check
# ##################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    # new diag
    key_new=None,
    # move - translate
    vect_xyz=None,
    length=None,
    # move - rotate
    axis_pt=None,
    axis_vect=None,
    angle=None,
    # move type
    move=None,
    # unused
    **kwdargs,
):

    # --------------
    # move
    # --------------

    move = ds._generic_check._check_var(
        move, 'move',
        types=str,
        allowed=['rotate', 'translate'],
    )

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

    out = (key, key_cam, key_new)

    # --------------
    # move params
    # --------------

    # --------------
    # translation

    if move == 'translate':
        # length
        length = _check_length_angle_vect(
            din=length,
            key_cam=key_cam,
            dval=np.r_[0.],
            name='length',
        )

        # vect_xyz
        vect_xyz = _check_length_angle_vect(
            din=vect_xyz,
            key_cam=key_cam,
            dval=np.r_[0., 0., 0.],
            name='vect_xyz',
        )

        # update out
        out = out + (length, vect_xyz)

    # --------------
    # rotation

    else:
        # axis_pt
        axis_pt = _check_length_angle_vect(
            din=axis_pt,
            key_cam=key_cam,
            dval=np.r_[0., 0., 0.],
            name='axis_pt',
        )

        # axis_vect
        axis_vect = _check_length_angle_vect(
            din=axis_vect,
            key_cam=key_cam,
            dval=np.r_[0., 0., 1.],
            name='axis_vect',
        )

        # angle
        angle = _check_length_angle_vect(
            din=angle,
            key_cam=key_cam,
            dval=np.r_[0.],
            name='angle',
        )

        # update out
        out = out + (angle, axis_pt, axis_vect)

    return out


# ##################################################
# ##################################################
#           subroutine
# ##################################################


def _check_length_angle_vect(
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
        if size == 1 and not np.isscalar(din[kcam]):
            din[kcam] = din[kcam][0]

    return din


# ##################################################
# ##################################################
#           add as-is
# ##################################################


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
