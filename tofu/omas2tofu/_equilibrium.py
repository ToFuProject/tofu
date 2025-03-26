

import numpy as np


from . import _ddef
from . import _utils


# ###########################################################
# ###########################################################
#              DEFAULTS
# ###########################################################


_IDS = 'eq'


_DIN = {
    'time': {
        'type': list,
    },
    'time_slice': {
        'type': dict,
        'len': 'time',
    },
}


# ###########################################################
# ###########################################################
#              Equilibrium
# ###########################################################


def main(
    din=None,
    coll=None,
    key=None,
    prefix=None,
):
    # ---------------
    # check
    # ---------------

    if prefix is None:
        prefix = ''
    elif not isinstance(prefix, str):
        msg = (
            "Arg prefix must be a str!\n"
            f"Provided: {prefix}\n"
        )
        raise Exception(msg)

    # ---------------
    # time
    # ---------------

    _get_time(coll)

    ref_nt = f"{key}_eq_nt"
    coll.add_ref(ref_nt, size=len(din['time']))

    kt = f"{key}_eq_t"
    coll.add_data(kt, data=din['time'], units='s', ref=ref_nt)

    # ---------------
    # 2d mesh
    # ---------------

    key_mesh = _add_mesh_2d(
        din=din,
        coll=coll,
        key=key,
    )

    # ---------------
    # 2d data
    # ---------------

    key_psi2d = _add_data_2d(
        din=din,
        coll=coll,
        key=key,
        ref_nt=ref_nt,
        key_mesh=key_mesh,
    )

    # -------------
    # add mesh 1d
    # -------------

    key_mesh_1d = _add_mesh_1d(
        din=din,
        coll=coll,
        key=key,
        ref_nt=ref_nt,
        key_mesh=key_mesh,
        key_psi2d=key_psi2d,
    )

    # -------------
    # add data 1d
    # -------------

    _add_data_1d(
        din=din,
        coll=coll,
        key=key,
        ref_nt=ref_nt,
        key_mesh=key_mesh,
        key_mesh_1d=key_mesh_1d,
    )

    return


# ###########################################################
# ###########################################################
#           Time
# ###########################################################


def _get_time(coll=None, din=None):

    # --------------
    # must have
    # --------------

    dout = _try_get(
        din,
        ddef['t'],
        must_have=True,
    )

    # --------------
    # add
    # --------------

    # ref
    reft = f"eq_nt"
    coll.add_ref(reft, size=dout['t']['data'].size)

    # data
    kt = "eq_t"
    coll.add_data()

    return


# ###########################################################
# ###########################################################
#           Mesh 2d
# ###########################################################


def _add_mesh_2d(
    din=None,
    coll=None,
    key=None,
):

    mtype = din['time_slice'][0]['profiles_2d'][0]['grid_type']['name']
    km = f"{key}_eq_m2d"

    # -------------
    # rectangular
    # -------------

    if mtype == 'rectangular':
        R = din['time_slice'][0]['profiles_2d'][0]['grid']['dim1']
        Z = din['time_slice'][0]['profiles_2d'][0]['grid']['dim2']

        coll.add_mesh_2d_rect(
            key=km,
            knots0=R,
            knots1=Z,
            units='m',
            deg=1,
        )

    # -------------
    # others
    # -------------

    else:
        raise NotImplementedError()

    return km


# ###########################################################
# ###########################################################
#           data 2d
# ###########################################################


def _add_data_2d(
    din=None,
    coll=None,
    key=None,
    ref_nt=None,
    key_mesh=None,
):

    # -------------
    # psi2d
    # -------------

    psi2d = np.array([
        din['time_slice'][ii]['profiles_2d'][0]['psi']
        for ii in range(coll.dref[ref_nt]['size'])
    ])

    # -------------
    # safety check on shape
    # -------------

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
    # safety check on shape
    # -------------

    key_psi2d = f"{key}_eq_psi2d"
    coll.add_data(
        key=key_psi2d,
        data=psi2d,
        ref=(ref_nt, kbs),
        dim='mag flux',
        name='psi',
        units='Wb',
    )

    return key_psi2d


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
