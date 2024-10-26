# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:39:20 2024

@author: dvezinet
"""


import json


import numpy as np
import datastock as ds


# #################################################################
# #################################################################
#          Main
# #################################################################


def main(
    # ---------------
    # input from tofu
    pfe=None,
    coll=None,
    fname=None,
    returnas=None,
):

    # ----------------
    # check inputs
    # --------------

    dout, returnas = _check(
        coll=coll,
        pfe=pfe,
        returnas=returnas,
    )

    # ----------------
    # get file content
    # ----------------

    # ------------
    # fill optics

    for kcls, vcls in dout.items():

        if kcls == 'diagnostic':
            continue

        for k1, v1 in vcls.items():
            _add_optics(
                coll=coll,
                which=kcls,
                din=v1,
                fname=fname,
            )

    # ---------------
    # add diagnostic

    _add_diagnostic(
        coll=coll,
        din=dout['diagnostic'],
    )

    if returnas is True:
        return coll


# #################################################################
# #################################################################
#          check
# #################################################################


def _check(
    coll=None,
    pfe=None,
    returnas=None,
):

    # --------------------
    # load dict from file
    # --------------------

    with open(pfe, 'r') as fn:
        dout = json.load(fn)

    # --------------------
    # check content
    # --------------------

    c0 = (
        isinstance(dout, dict)
        and all([ss in dout.keys() for ss in ['diagnostic', 'camera']])
    )
    if not c0:
        msg = (
            "File does not seem to hold a tofu-compatible diagnostic!\n"
            f"\t- Provided: {pfe}\n"
        )
        raise Exception(msg)

    # -------------
    # returnas
    # -------------

    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        types=bool,
        default=True,
    )

    return dout, returnas


# #################################################################
# #################################################################
#          add_optics
# #################################################################


def _add_optics(
    coll=None,
    which=None,
    din=None,
    fname=None,
):

    # ----------------
    # key
    # ----------------

    lout = list(coll.dobj.get(which, {}).keys())
    if din['key'] in lout:
        din['key'] = f"{fname}_{din['key']}"

    # ----------------
    # dgeom
    # ----------------

    dgeom = _get_dgeom(din)

    # ----------------
    # dmat
    # ----------------

    dmat = _get_dmat(din)

    # ----------------
    # add to coll
    # ----------------

    if which == 'aperture':
        coll.add_aperture(
            key=din['key'],
            **dgeom,
        )

    else:
        if which == 'camera':
            func = f"add_camera_{dgeom['nd']}"
            del dgeom['nd']
        elif which in ['aperture', 'filter', 'crystal', 'grating']:
            func = f'add_{which}'
        else:
            msg = f"Unknow optics class: {which}"
            raise Exception(msg)

        # add
        getattr(coll, func)(
            key=din['key'],
            dgeom=dgeom,
            dmat=dmat,
        )

    return


# #################################################################
# #################################################################
#          _get_dgeom
# #################################################################


def _get_dgeom(din=None):

    # --------------
    # prepare
    # --------------

    larray = [
        'cent',
        'cents_x0', 'cents_x1',
        'cents_x', 'cents_y', 'cents_z',
        'nin', 'e0', 'e1',
        'nin_x', 'nin_y', 'nin_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
        'outline_x0', 'outline_x1',
        'poly_x', 'poly_y', 'poly_z',
    ]
    lex = ['type', 'extenthalf', 'shape', 'parallel']

    # --------------
    # loop
    # --------------

    dgeom = {}
    for k0, v0 in din['dgeom'].items():

        if k0 in lex:
            continue

        if k0 in larray:
            v0 = np.asarray(v0['data'])
        elif isinstance(v0, dict):
            v0 = v0['data']

        dgeom[k0] = v0

    return dgeom


# #################################################################
# #################################################################
#          _get_dmat
# #################################################################


def _get_dmat(din=None):

    dmat = din.get('dmat')

    return dmat


# #################################################################
# #################################################################
#          _get_diagnostic
# #################################################################



def _add_diagnostic(
    coll=None,
    din=None,
):

    # ----------------
    # prepare doptics
    # ----------------

    doptics = din['doptics']

    # --------------
    # add
    # --------------

    coll.add_diagnostic(
        key=din['key'],
        doptics=doptics,
        compute=False,
    )

    # force camera order (to maintain arrays order)
    coll._dobj['diagnostic'][din['key']]['camera'] = din['camera']

    # ------------------
    # add computed data
    # -------------------

    lcam = din['camera']

    # -----------
    # los

    for kcam in lcam:

        # add ray
        klos = doptics[kcam].get('los_key')
        if klos is not None:
            coll.add_rays(
                key=klos,
                # start
                start_x=doptics[kcam]['los_x_start']['data'],
                start_y=doptics[kcam]['los_y_start']['data'],
                start_z=doptics[kcam]['los_z_start']['data'],
                # pts
                pts_x=doptics[kcam]['los_x_end']['data'],
                pts_y=doptics[kcam]['los_y_end']['data'],
                pts_z=doptics[kcam]['los_z_end']['data'],
                # angles
                alpha=doptics[kcam]['los_alpha']['data'],
                dalpha=doptics[kcam]['los_dalpha']['data'],
                dbeta=doptics[kcam]['los_dbeta']['data'],
            )

            # adjust ref to match camera
            ref = tuple(
                [coll.dobj['rays'][klos]['ref'][0]]
                + list(coll.dobj['camera'][kcam]['dgeom']['ref'])
            )
            coll._dobj['rays'][klos]['ref'] = ref

        # store in diag
        coll._dobj['diagnostic'][din['key']]['doptics'][kcam]['los'] = klos

    # -----------
    # etendue

    for kcam in lcam:

        # add data
        ketend = doptics[kcam].get('etendue_key')
        etend_type = doptics[kcam].get('etend_type')
        if ketend is not None:
            ref = coll.dobj['camera'][kcam]['dgeom']['ref']
            coll.add_data(
                key=ketend,
                data=doptics[kcam]['etendue']['data'],
                units=doptics[kcam]['etendue']['units'],
                ref=ref,
            )

        # store in diag
        coll._dobj['diagnostic'][din['key']]['doptics'][kcam]['etendue'] = ketend
        coll._dobj['diagnostic'][din['key']]['doptics'][kcam]['etend_type'] = etend_type

    # -----------
    # vos
    # -----------

    for kcam in lcam:

        # --------
        # add data

        for kp in ['pcross_x0', 'pcross_x1', 'phor_x0', 'phor_x1']:

            kpi = doptics[kcam][kp]['key']
            ref = doptics[kcam][kp]['ref']

            # add ref if needed
            if ref is not None:
                lr = [rr for rr in ref if rr not in coll.dref.keys()]
                for rr in lr:
                    ii = ref.index(rr)
                    coll.add_ref(
                        key=rr,
                        size=np.array(doptics[kcam][kp]['data']).shape[ii],
                    )

            # add data
            if kpi is not None:
                coll.add_data(
                    key=kpi,
                    data=doptics[kcam][kp]['data'],
                    units=doptics[kcam][kp]['units'],
                    ref=tuple(ref),
                )

        # -----------------
        # store vos in diag

        pcross0 = doptics[kcam].get('pcross_x0', {}).get('key')
        pcross1 = doptics[kcam].get('pcross_x1', {}).get('key')
        phor0 = doptics[kcam].get('phor_x0', {}).get('key')
        phor1 = doptics[kcam].get('phor_x1', {}).get('key')
        dphi = doptics[kcam].get('dphi', {}).get('data')

        if pcross0 is not None:
            pcross = (pcross0, pcross1)
            phor = (phor0, phor1)
            dphi = np.array(dphi)
        else:
            pcross = None
            phor = None
            dphi = None

        # -----------------
        # store phor in diag

        coll._dobj['diagnostic'][din['key']]['doptics'][kcam]['dvos'] = {
            'pcross': pcross,
            'phor': phor,
            'dphi': dphi,
        }

    return