

import numpy as np


from . import _utils


# ###########################################################
# ###########################################################
#              DEFAULTS
# ###########################################################


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

    print(coll)

    # ---------------
    # 2d data
    # ---------------

    lk2d = _add_data_2d(
        din=din,
        ids=ids,
        coll=coll,
        dshort=dshort,
        prefix=prefix,
        strict=strict,
        # bsplines
        dbsplines=dbsplines,
    )

    # -------------
    # add mesh 1d
    # -------------

    # key_mesh_1d = _add_mesh_1d(
        # din=din,
        # coll=coll,
        # key=key,
        # ref_nt=ref_nt,
        # key_mesh=key_mesh,
        # key_psi2d=key_psi2d,
    # )

    # -------------
    # add data 1d
    # -------------

    # _add_data_1d(
        # din=din,
        # coll=coll,
        # key=key,
        # ref_nt=ref_nt,
        # key_mesh=key_mesh,
        # key_mesh_1d=key_mesh_1d,
    # )

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

    # initialize
    ddata = {}
    dref = {}
    dfail = {}

    # --------------
    # must have
    # --------------

    lref0 = []
    for k0 in lshortref:
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
    lk = ['key', 'data', 'units', 'ref']
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
        and not '1d' in k0
    ]

    # initialize
    ddata = {}
    dref = {}
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
    lk = ['key', 'data', 'units', 'ref']
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

    if ids == 'equilibrium':
        ldata_ind = [
            k0 for k0, v0 in dshort[ids].items()
            if '[im2d]' in v0['long']
            and 'grid' not in dshort[ids][k0]['long']
        ]

    # -------------
    # loop on data
    # -------------

    lk2d = []
    dfail = {}
    for kd in ldata_ind:

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

        ref = tuple([
            dbsplines[0] if rr == 'im2d'
            else rr for rr in ddatai[kd]['ref']
        ])
        ddatai[kd]['ref'] = ref

        # -------------
        # safety check on shape
        # -------------

        import pdb; pdb.set_trace()     # DB
        kbs = f"{key_mesh}_bs1"
        wbs = coll._which_bsplines
        shape_bs = coll.dobj[wbs][kbs]['shape']
        if psi2d.shape[1:] == shape_bs:
            pass
        elif psi2d.shape[1:] == shape_bs[::-1]:
            psi2d = np.swapaxes(psi2d, 1, 2)
        else:
            msg = (
                "2d data 'psi' has unknow shape vs bsplines!\n"
                f"\t- psi2d.shape = {psi2d.shape}\n"
                f"\t- coll.dobj[{wbs}][{kbs}]['shape'] = {shape_bs}\n"
            )
            raise Exception(msg)

        # -------------
        # store
        # -------------

        coll.add_data(**ddatai)

        # store in list
        lk2d.append(list(ddatai.keys())[0])

    # -------------
    # dfail
    # -------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {str(v0)}" for k0, v0 in dfail.items()]
        msg = (
            "The following 2d data could not be loaded:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    return lk2d


# ###########################################################
# ###########################################################
#           mesh 1d
# ###########################################################


def _add_mesh_1d(
    din=None,
    coll=None,
    key=None,
    ref_nt=None,
    key_mesh=None,
    key_psi2d=None,
):

    # ---------------
    # safety check
    # ---------------

    npsi = np.array([
        len(din['time_slice'][ii]['profiles_1d']['psi_norm'])
        for ii in range(coll.dref[ref_nt]['size'])
    ])
    if not np.all(npsi == npsi[0]):
        msg = (
            "size of profiles_1d['psi_norm'] is not homogeneous in time!\n"
            f"npsi = {npsi}\n"
        )
        raise Exception(msg)

    lpsin = np.array([
        din['time_slice'][ii]['profiles_1d']['psi_norm']
        for ii in range(coll.dref[ref_nt]['size'])
    ])
    if not np.allclose(lpsin, lpsin[0:1, :]):
        msg = (
            "Values of profiles_1d['psi'] is not homogeneous in time!\n"
            f"{lpsin}\n"
        )
        raise Exception(msg)

    psin1d = lpsin[0, :]

    # ---------------
    # add psin2d
    # ---------------

    psi = np.array([
        din['time_slice'][ii]['profiles_1d']['psi']
        for ii in range(coll.dref[ref_nt]['size'])
    ])
    psi0 = psi[:, 0][:, None, None]
    psi1 = psi[:, -1][:, None, None]

    psi2d = coll.ddata[key_psi2d]['data']
    psin2d = (psi2d - psi0) / (psi1 - psi0)

    kpsin2d = f"{key}_eq_psin2d"
    coll.add_data(
        kpsin2d,
        data=psin2d,
        units=None,
        dim='mag flux norm',
        name='psin',
        ref=coll.ddata[key_psi2d]['ref'],
    )

    # ---------------
    # psi
    # ---------------

    key_mesh_1d = f"{key}_eq_m1d"
    coll.add_mesh_1d(
        key=key_mesh_1d,
        knots=psin1d,
        dim='mag flux norm',
        name='psin',
        units=None,
        subkey=kpsin2d,
        deg=1,
    )

    return key_mesh_1d


# ###########################################################
# ###########################################################
#       data 1d
# ###########################################################


def _add_data_1d(
    din=None,
    coll=None,
    key=None,
    ref_nt=None,
    key_mesh=None,
    key_mesh_1d=None,
):

    dk = {
        'dpressure_dpsi': {'dpdpsi': 'Pa/Wb'},
        'dvolume_dpsi': {'dVdpsi': 'm3/Wb'},
        'elongation': {'kappa': None},
        # 'geometric_axis': {'geomAx': 'm'},
        'j_tor': {'jtor': 'A/m2'},
        'pressure': {'p': 'Pa'},
        'psi': {'psi': 'Wb'},
        'q': {'q': None},
        'r_inboard': {'rin': 'm'},
        'r_outboard': {'rout': 'm'},
        'rho_tor': {'rhot': None},
        'rho_tor_norm': {'rhotn': None},
        'surface': {'S': 'm2'},
        'triangularity_lower': {'triangLow': None},
        'triangularity_upper': {'triangUp': None},
        'volume': {'V': 'm3'},
    }

    kbs1d = f"{key_mesh_1d}_bs1"
    for ii, (k0, v0) in enumerate(dk.items()):

        data = np.array([
            din['time_slice'][ii]['profiles_1d'][k0]
            for ii in range(coll.dref[ref_nt]['size'])
        ])

        k1 = list(v0.keys())[0]
        keyi = f"{key}_eq_{k1}"
        coll.add_data(
            key=keyi,
            data=data,
            units=v0[k1],
            ref=(ref_nt, kbs1d),
        )

    return
