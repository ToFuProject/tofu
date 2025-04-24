

import warnings


import numpy as np
from matplotlib.path import Path


from . import _utils
from . import _equilibrium
from . import _core_profiles


# ###########################################################
# ###########################################################
#              DEFAULTS
# ###########################################################


_DSUBKEY = {
    'equilibrium': _equilibrium._get_subkey,
    'core_profiles': _core_profiles._get_subkey,
}


# ###########################################################
# ###########################################################
#              Equilibrium
# ###########################################################


def main(
    din=None,
    ids=None,
    coll=None,
    key=None,
    prefix=None,
    dshort=None,
    strict=None,
    warn=None,
):

    # ------------------------
    # data as ref0 first
    # ------------------------

    lref0, dfail_ref0 = _add_ref0(
        coll=coll,
        din=din,
        dshort=dshort,
        ids=ids,
        prefix=prefix,
        strict=strict,
    )

    # ------------------------
    # data dependent only on ref0
    # ------------------------

    dfail_data_ref0 = _add_data_ref0(
        coll=coll,
        din=din,
        dshort=dshort,
        ids=ids,
        prefix=prefix,
        strict=strict,
        lref0=lref0,
    )

    # -----------------------------
    # mesh 2d second
    # -----------------------------

    dbsplines = _add_mesh_2d(
        din=din,
        coll=coll,
        ids=ids,
        prefix=prefix,
        # unused ?
        dshort=dshort,
    )

    # ---------------
    # 2d data
    # ---------------

    dfail_data2d, lk2d = _add_data_2d(
        din=din,
        ids=ids,
        coll=coll,
        dshort=dshort,
        prefix=prefix,
        strict=strict,
        # bsplines
        dbsplines=dbsplines,
    )

    # ---------------------
    # add mesh and data 1d
    # ---------------------

    dfail_data1d = _add_mesh_data_1d(
        din=din,
        ids=ids,
        coll=coll,
        dshort=dshort,
        prefix=prefix,
        strict=strict,
        # bsplines
        dbsplines=dbsplines,
        lk2d=lk2d,
    )

    # -------------
    # warnings
    # -------------

    dfail = dfail_ref0 | dfail_data_ref0 | dfail_data2d | dfail_data1d
    if warn is True and len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following data could not be loaded:\n"
            f"From ids = {ids}\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    return


# ###########################################################
# ###########################################################
#           Ref0
# ###########################################################


def _add_ref0(
    coll=None,
    din=None,
    dshort=None,
    ids=None,
    prefix=None,
    strict=None,
):

    # --------------
    # prepare
    # --------------

    # ex: time
    lshortref = [
        k0 for k0, v0 in dshort[ids].items()
        if v0.get('ref0') is not None
    ]

    # order to start with 1d vectors
    # ex: sepR also depends on nt
    nref = [dshort[ids][k0]['long'].count('[') for k0 in lshortref]
    lshortref = [lshortref[ii] for ii in np.argsort(nref)]
    nref = [dshort[ids][k0]['long'].count('[') for k0 in lshortref]

    # initialize
    ddata = {}
    dref = {}
    dfail = {}

    # --------------
    # must have
    # --------------

    lref0 = []
    for ii, k0 in enumerate(lshortref):
        ddatai, drefi = _utils._get_short(
            din=din,
            ids=ids,
            short=k0,
            dshort=dshort,
            prefix=prefix,
            strict=strict,
        )

        # ---------------------
        # check is several ref

        if nref[ii] > 1:
            refi = [
                rr for rr in dshort[ids][k0]['ref']
                if rr not in list(drefi.keys())[0]
            ]
            assert all([rr in lref0 for rr in refi])

        # -----
        # store

        if isinstance(ddatai, dict):
            dref.update(drefi)
            ddata.update(ddatai)
            lref0.append(dshort[ids][k0]['ref0'])

        else:
            dfail[k0] = str(ddatai)

    # --------------
    # fail
    # --------------

    if len(dfail) > 0 and strict is True:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "IDS '{ids}', the following keys could not be loaded:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------------
    # store
    # --------------

    # dref
    for kr, vr in dref.items():
        coll.add_ref(**vr)

    # data
    lk = ['key', 'data', 'units', 'ref', 'dim', 'quant', 'name']
    for kd, vd in ddata.items():
        coll.add_data(
            **{k0: v0 for k0, v0 in vd.items() if k0 in lk}
        )

    return lref0, dfail


# ###########################################################
# ###########################################################
#           data ref0
# ###########################################################


def _add_data_ref0(
    coll=None,
    din=None,
    dshort=None,
    ids=None,
    prefix=None,
    strict=None,
    lref0=None,
):

    # --------------
    # prepare
    # --------------

    # ex: time
    lshort = [
        k0 for k0, v0 in dshort[ids].items()
        if v0.get('ref0') is None
        and v0.get('ref') is not None
        and all([rr in lref0 for rr in v0['ref']])
    ]

    # initialize
    ddata = {}
    dfail = {}

    # --------------
    # must have
    # --------------

    for k0 in lshort:
        ddatai, drefi = _utils._get_short(
            din=din,
            ids=ids,
            short=k0,
            dshort=dshort,
            prefix=prefix,
            strict=strict,
        )

        # -----
        # store

        if isinstance(ddatai, dict):
            assert drefi is None
            ddata.update(ddatai)

        else:
            dfail[k0] = str(ddatai)

    # --------------
    # fail
    # --------------

    if len(dfail) > 0 and strict is True:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "IDS '{ids}', the following keys could not be loaded:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------------
    # store
    # --------------

    # data
    lk = ['key', 'data', 'units', 'ref', 'dim', 'quant', 'name']
    for kd, vd in ddata.items():
        coll.add_data(
            **{k0: v0 for k0, v0 in vd.items() if k0 in lk}
        )

    return dfail

# ###########################################################
# ###########################################################
#           Mesh 2d
# ###########################################################


def _add_mesh_2d(
    din=None,
    coll=None,
    ids=None,
    prefix=None,
    # unused ?
    dshort=None,
):
    # ----------------
    # initialize
    # ----------------

    dbsplines = {}

    # ----------------
    # list of mesh 2d
    # ----------------

    if ids == 'equilibrium':

        p2d = din[ids]['time_slice'][0]['profiles_2d']
        nmesh = len(p2d)
        for im in range(nmesh):
            mtype = p2d[im]['grid_type']['name']

            if mtype == 'rectangular':
                R = p2d[im]['grid']['dim1']
                Z = p2d[im]['grid']['dim2']

                km = _utils._make_key(
                    prefix=prefix,
                    ids=ids,
                    short=f'm2d{im}',
                )

                coll.add_mesh_2d_rect(
                    key=km,
                    knots0=R,
                    knots1=Z,
                    units='m',
                    deg=1,
                )

                dbsplines[im] = f"{km}_bs1"

    return dbsplines


# ###########################################################
# ###########################################################
#           data 2d
# ###########################################################


def _add_data_2d(
    din=None,
    ids=None,
    coll=None,
    dshort=None,
    prefix=None,
    strict=None,
    # bsplines
    dbsplines=None,
):

    # ------------------
    # get list of data
    # ------------------

    lids = [
        'wall',
        'summary',
        'pulse_schedule',
        'equilibrium',
        'core_profiles',
    ]
    if ids not in lids:
        msg = f"Not sure how to deal with 2d data for ids '{ids}'"
        raise Exception(msg)

    ldata = [
        k0 for k0, v0 in dshort[ids].items()
        if '[im2d]' in v0['long']
        and 'grid' not in dshort[ids][k0]['long']
    ]

    # -------------
    # loop on data
    # -------------

    dk2d = {}
    dfail = {}
    for kd in ldata:

        ddatai, drefi = _utils._get_short(
            din=din,
            ids=ids,
            short=kd,
            dshort=dshort,
            prefix=prefix,
            strict=strict,
        )

        # ----------
        # exception catching

        if not isinstance(ddatai, dict):
            dfail[kd] = ddatai
            continue

        # ------------
        # safety check

        c0 = (
            drefi is None
            and 'im2d' in ddatai[kd]['ref']
        )
        if not c0:
            msg = (
                "Inconsistent data from:\n"
                f"\t- ids = {ids}\n"
                f"\t- short = {kd}\n"
                f"\t- drefi = {drefi}\n"
                f"\t- ddatai[{kd}]['ref'] = {ddatai[kd]['ref']}\n"
            )
            raise Exception(msg)

        # ----------
        # ref

        kbs = dbsplines[0]

        ref = tuple([
            kbs if rr == 'im2d'
            else rr for rr in ddatai[kd]['ref']
        ])
        ddatai[kd]['ref'] = ref

        axis = ref.index(kbs)

        # -------------
        # safety check on shape
        # -------------

        wbs = coll._which_bsplines
        shape_bs = coll.dobj[wbs][kbs]['shape']
        shape = ddatai[kd]['data'].shape

        if shape[axis:axis+len(shape_bs)] == shape_bs:
            pass

        elif shape[axis:axis+len(shape_bs)] == shape_bs[::-1]:
            assert len(shape_bs) == 2
            ddatai[kd]['data'] = np.swapaxes(
                ddatai[kd]['data'],
                axis,
                axis + 1,
            )

        else:
            msg = (
                "2d data has unknow shape vs bsplines!\n"
                f"\t- ids = {ids}\n"
                f"\t- short = {kd}\n"
                f"\t- ref = {ref}\n"
                f"\t- kbs = {kbs}\n"
                f"\t- shape = {shape}\n"
                f"\t- coll.dobj[{wbs}][{kbs}]['shape'] = {shape_bs}\n"
            )
            raise Exception(msg)

        # -------------
        # store
        # -------------

        coll.add_data(**ddatai[kd])

        # store in list
        dk2d[kd] = ddatai[kd]['key']

    # -------------
    # update ldata and complete
    # -------------

    ldata = [kk for kk in ldata if kk not in dfail.keys()]
    lpsi = [kk for kk in ldata if 'psi' in kk]
    lphi = [kk for kk in ldata if 'phi' in kk]

    # -----------------
    # poloidal flux

    if len(lpsi) > 0:
        kpsi = [kk for kk in lpsi if kk.endswith('2dpsi')]
        if len(kpsi) == 1:
            kpsi = kpsi[0]

            kpsin = [kk for kk in lpsi if kk.endswith('psin')]
            if len(kpsin) == 0:
                pass
                # _add_psin_from_psi()

            krhopn = [kk for kk in ldata if kk.endswith('rhopn')]
            if len(krhopn) == 0:
                try:
                    _add_rhopn_from_psi(
                        coll=coll,
                        kpsi=kpsi,
                        din=din,
                        ids=ids,
                        dk2d=dk2d,
                    )
                except Exception:
                    dfail['2drhopn'] = "failed"

    # -----------------
    # toroidal flux

    if len(lphi) > 0:
        kphi = [kk for kk in lphi if kk.endswith('2dphi')]
        if len(kphi) == 1:
            kphi = kphi[0]

            kphin = [kk for kk in lphi if kk.endswith('phin')]
            if len(kphin) == 0:
                pass
                # _add_psin_from_psi()

            krhotn = [kk for kk in ldata if kk.endswith('rhotn')]
            if len(krhotn) == 0:
                try:
                    _add_rhotn_from_phi(
                        coll=coll,
                        kphi=kphi,
                        din=din,
                        ids=ids,
                        dk2d=dk2d,
                        kbs=kbs,
                    )
                except Exception:
                    dfail['2drhotn'] = "failed"

    # -------------
    # dfail
    # -------------

    if len(dfail) > 0 and strict is True:
        lstr = [f"\t- {k0}: {str(v0)}" for k0, v0 in dfail.items()]
        msg = (
            "The following 2d data could not be loaded:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    return dfail, list(dk2d.values())


def _add_rhopn_from_psi(
    coll=None,
    kpsi=None,
    din=None,
    ids=None,
    dk2d=None,
):
    """

    """

    if ids != 'equilibrium':
        return

    # ---------
    # extract

    psi = coll.ddata[dk2d[kpsi]]['data']
    psi0 = np.array([
        ts['global_quantities']['psi_axis']
        for ts in din[ids]['time_slice']
    ])[:, None, None]
    psi1 = np.array([
        ts['global_quantities']['psi_boundary']
        for ts in din[ids]['time_slice']
    ])[:, None, None]

    # ---------
    # compute

    rhopn = np.sqrt((psi - psi0) / (psi1 - psi0))

    # ---------
    # add data

    key = dk2d[kpsi].replace('psi', 'rhopn')
    coll.add_data(
        key=key,
        data=rhopn,
        ref=coll.ddata[dk2d[kpsi]]['ref'],
        units=None,
        dim='rho',
        quant='rhopn',
        name='rhopn',
    )

    dk2d['rhotpn'] = key

    return


def _add_rhotn_from_phi(
    coll=None,
    kphi=None,
    din=None,
    ids=None,
    dk2d=None,
    kbs=None,
):
    """

    """

    if ids != 'equilibrium':
        return

    # ---------
    # extract

    phi = coll.ddata[dk2d[kphi]]['data']
    phi1 = np.array([
        ts['profiles_1d']['phi'][-1]
        for ts in din[ids]['time_slice']
    ])[:, None, None]

    # ---------
    # compute

    rhotn = np.sqrt(phi / phi1)

    # --------------------------
    # set pts outside sep to nan

    wbs = coll._which_bsplines
    ksepR = [kk for kk in coll.ddata.keys() if kk.endswith('sepR')]
    ksepZ = [kk for kk in coll.ddata.keys() if kk.endswith('sepZ')]
    if len(ksepR) == len(ksepZ) == 1:
        sepR = coll.ddata[ksepR[0]]['data']
        sepZ = coll.ddata[ksepZ[0]]['data']
        assert sepR.shape[0] == len(din[ids]['time_slice'])
        kx0, kx1 = coll.dobj[wbs][kbs]['apex']
        x0 = coll.ddata[kx0]['data']
        x1 = coll.ddata[kx1]['data']
        pts = np.array([
            np.repeat(x0[:, None], x1.size, axis=1).ravel(),
            np.repeat(x1[None, :], x0.size, axis=0).ravel(),
        ]).T
        for ii in range(sepR.shape[0]):
            path = Path(np.array([sepR[ii, :], sepZ[ii, :]]).T)
            iout = ~path.contains_points(pts).reshape((x0.size, x1.size))
            sli = (ii, iout)
            rhotn[sli] = np.nan

    # ---------
    # add data

    key = dk2d[kphi].replace('phi', 'rhotn')
    coll.add_data(
        key=key,
        data=rhotn,
        ref=coll.ddata[dk2d[kphi]]['ref'],
        units=None,
        dim='rho',
        quant='rhotn',
        name='rhotn',
    )

    dk2d['rhotn'] = key

    return


# ###########################################################
# ###########################################################
#           mesh 1d
# ###########################################################


def _add_mesh_data_1d(
    din=None,
    ids=None,
    coll=None,
    dshort=None,
    prefix=None,
    strict=None,
    # bsplines
    dbsplines=None,
    lk2d=None,
):

    # ------------------
    # get list of data
    # ------------------

    ldata = [
        k0 for k0, v0 in dshort[ids].items()
        if '[im1d]' in v0['long']
    ]

    # -------------
    # loop on data
    # -------------

    dfail = {}
    ddata = {}
    axis = None
    shape = None

    if len(ldata) == 0:
        return dfail

    # -------------
    # loop on data
    # -------------

    for kd in ldata:

        ddatai, drefi = _utils._get_short(
            din=din,
            ids=ids,
            short=kd,
            dshort=dshort,
            prefix=prefix,
            strict=strict,
        )

        # ----------
        # exception catching

        if not isinstance(ddatai, dict):
            dfail[kd] = ddatai
            continue

        # ------------
        # safety check

        c0 = (
            drefi is None
            and np.sum(['im1d' in rr for rr in ddatai[kd]['ref']])
        )
        if not c0:
            msg = (
                "Inconsistent data from:\n"
                f"\t- ids = {ids}\n"
                f"\t- short = {kd}\n"
                f"\t- drefi = {drefi}\n"
                f"\t- ddatai[{kd}]['ref'] = {ddatai[kd]['ref']}\n"
            )
            raise Exception(msg)

        # shape
        shapei = ddatai[kd]['data'].shape
        if shape is None:
            shape = shapei
        else:
            assert shapei == shape

        # axis
        r1d = [rr for rr in ddatai[kd]['ref'] if 'm1d' in rr][0]
        axisi = ddatai[kd]['ref'].index(r1d)
        if axis is None:
            axis = axisi
        else:
            assert axis == axisi

        # ------------------
        # aggregate to ddata

        ddata[kd] = ddatai[kd]

    # update ldata
    ldata = [kk for kk in ldata if kk in ddata.keys()]

    # -------------------------------
    # identify 1d knots and 2d subkey
    # -------------------------------

    try:
        k1d, q1d, k2d = _get_subkey(
            coll=coll,
            ids=ids,
            shape=shape,
            axis=axis,
            ddata=ddata,
            ldata=ldata,
            lk2d=lk2d,
        )
    except Exception:
        dfail['subkey'] = 'could not identify'
        k1d, q1d, k2d = None, None, None

    # --------------------
    # add mesh
    # --------------------

    if k1d is not None:
        km = _utils._make_key(
            prefix=prefix,
            ids=ids,
            short='m1d',
        )

        lk = ['units', 'dim', 'quant', 'name']
        coll.add_mesh_1d(
            key=km,
            knots=q1d,
            subkey=k2d,
            deg=1,
            **{k0: ddata[k1d].get(k0) for k0 in lk},
        )

        kbs = f"{km}_bs1"

        # ----------------
        # add data 1d
        # ----------------

        for kd in ldata:

            if kd == k1d:
                continue

            # ----------
            # ref

            ref = tuple([
                kbs if 'im1d' in rr
                else rr for rr in ddata[kd]['ref']
            ])
            ddata[kd]['ref'] = ref

            axis = ref.index(kbs)

            # -------------
            # store
            # -------------

            coll.add_data(**ddata[kd])

    # -------------
    # dfail
    # -------------

    if len(dfail) > 0 and strict is True:
        lstr = [f"\t- {k0}: {str(v0)}" for k0, v0 in dfail.items()]
        msg = (
            "The following 1d data could not be loaded:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    return dfail


# ################################################################
# ################################################################
#        Identify subkey
# ################################################################


def _get_subkey(
    coll=None,
    ids=None,
    shape=None,
    axis=None,
    ddata=None,
    ldata=None,
    lk2d=None,
):

    # ----------------
    # identify mesh 1d
    # ----------------

    sli = tuple([
        slice(None) if ii == axis
        else slice(0, 1, 1)
        for ii in range(len(shape))
    ])

    l1d = [
        k0 for k0, v0 in ddata.items()
        if np.allclose(v0['data'], v0['data'][sli])
    ]
    if len(l1d) != 1:
        msg = (
            "No / multiple constant 1d mesh data identified:\n"
            f"\t- ids = {ids}\n"
            f"\t- l1d = {l1d}\n"
        )
        raise Exception(msg)

    k1d = l1d[0]
    key_1d = ddata[k1d]['key']
    q1d = ddata[k1d]['data'][sli].ravel()

    # ------------------
    # Identify 2d subkey
    # ------------------

    k2d_name = [
        kk for kk in lk2d
        if ddata[k1d]['name'] == coll.ddata[kk]['name']
        or ddata[k1d]['quant'] == coll.ddata[kk]['quant']
    ]

    if len(k2d_name) > 1:
        msg = (
            "Several 2d data identified to match 1d mesh:\n"
            "\t- ids = {ids}\n"
            "\t- k1d = {k1d}\n"
            "\t- k2d = {k2d}\n"
        )
        raise Exception(msg)

    # --------------------
    # no match => call specialized routine
    # --------------------

    if len(k2d_name) == 1:

        k2dn = k2d_name[0]

    elif ids in _DSUBKEY.keys():
        k1d, q1d, k2dn = _DSUBKEY[ids](
            coll=coll,
            ids=ids,
            shape=shape,
            axis=axis,
            ddata=ddata,
            ldata=ldata,
            lk2d=lk2d,
            k1d=k1d,
            key_1d=key_1d,
            q1d=q1d,
        )

    else:
        raise NotImplementedError(ids)

    return k1d, q1d, k2dn
