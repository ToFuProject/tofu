

import copy


# ################################################
# ################################################
#              IDS Names
# ################################################


_DIDS = {
    'equilibrium': 'eq',
}


# ################################################
# ################################################
#              SHORTS
# ################################################


_DSHORT = {

    # ---------------
    # equilibrium
    # ---------------

    "equilibrium": {

        # -----------
        # time traces

        't': {
            'dim': 'time',
            'long': 'time',
            'units': 's',
            'ref0': 'nt',
        },
        'Ip': {
            'dim': 'current',
            'long': 'time_slice[nt].global_quantities.ip',
            'units': 'A',
        },

        # --------
        # mesh 2d

        'm2dR': {
            'dim': 'distance',
            'long': 'time_slice[0].profiles_2d[im2d].grid.dim1',
            'units': 'm',
            'mesh': 'm2d',
        },
        'm2dZ': {
            'dim': 'distance',
            'long': 'time_slice[0].profiles_2d[im2d].grid.dim2',
            'units': 'm',
            'mesh': 'm2d',
        },

        # ----------
        # 2d data

        '2dpsi': {
            'dim': 'B flux',
            'long': 'time_slice[nt].profiles_2d[im2d].psi',
            'units': 'Wb',
        },
        '2dphi': {
            'dim': 'B flux',
            'long': 'time_slice[nt].profiles_2d[im2d].phi',
            'units': 'Wb',
        },
        '2dBR': {
            'dim': 'B',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_r',
            'units': 'T',
        },
        '2dBZ': {
            'dim': 'B',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_z',
            'units': 'T',
        },
        '2dBphi': {
            'dim': 'B',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_tor',
            'units': 'T',
        },

        # ----------
        # 1d data

        '1ddpdpsi': {
            'dim': 'p / flux',
            'long': 'time_slice[nt].profiles_1d.dpressure_dpsi',
            'units': 'Pa/Wb',
        },
        '1ddVdpsi': {
            'dim': 'vol / flux',
            'long': 'time_slice[nt].profiles_1d.dvolume_dpsi',
            'units': 'm3/Wb',
        },
        '1dkappa': {
            'dim': 'adim',
            'long': 'time_slice[nt].profiles_1d.elongation',
            'units': None,
        },
        '1djtor': {
            'dim': 'current density',
            'long': 'time_slice[nt].profiles_1d.j_tor',
            'units': 'A/m2',
        },
        '1dp': {
            'dim': 'pressure',
            'long': 'time_slice[nt].profiles_1d.pressure',
            'units': 'Pa',
        },
        '1dpsi': {
            'dim': 'B flux',
            'long': 'time_slice[nt].profiles_1d.psi',
            'units': 'Wb',
        },
        '1dq': {
            'dim': 'q',
            'long': 'time_slice[nt].profiles_1d.q',
            'units': None,
        },
        '1drin': {
            'dim': 'distance',
            'description': 'LCFS major radius, inboard',
            'long': 'time_slice[nt].profiles_1d.r_inboard',
            'units': 'm',
        },
        '1drout': {
            'dim': 'distance',
            'description': 'LCFS major radius, outboard',
            'long': 'time_slice[nt].profiles_1d.r_outboard',
            'units': 'm',
        },
        '1drhot': {
            'dim': 'rho',
            'long': 'time_slice[nt].profiles_1d.rho_tor',
            'units': None,
        },
        '1drhotn': {
            'dim': 'rho',
            'long': 'time_slice[nt].profiles_1d.rho_tor_norm',
            'units': None,
        },
        '1d': {
            'dim': '',
            'long': 'time_slice[nt].profiles_1d.',
            'units': '',
        },
        '1dS': {
            'dim': 'surface',
            'long': 'time_slice[nt].profiles_1d.surface',
            'units': 'm2',
        },
        '1dtrianglow': {
            'dim': 'triangularity',
            'long': 'time_slice[nt].profiles_1d.triangularity_lower',
            'units': None,
        },
        '1dtriangup': {
            'dim': 'triangularity',
            'long': 'time_slice[nt].profiles_1d.triangularity_upper',
            'units': None,
        },
        '1dV': {
            'dim': 'volume',
            'long': 'time_slice[nt].profiles_1d.volume',
            'units': 'm3',
        },
    },
}


# ###########################################
# ###########################################
#            get_dshort
# ###########################################


def get_dshort():

    dshort = copy.deepcopy(_DSHORT)

    for ids, vids in dshort.items():

        # ------------------
        # list of references

        lkref0 = [
            k0 for k0, v0 in vids.items()
            if v0.get('ref0') is not None
        ]
        lref0 = [dshort[ids][k0]['ref0'] for k0 in lkref0]

        for k0 in lkref0:
            dshort[ids][k0]['ref'] = (dshort[ids][k0]['ref0'],)

        # ---------------
        # list of meshes

        # lm = sorted(set([
        #     v0['mesh'] for k0, v0 in vids.items()
        #     if v0.get('mesh') is not None
        # ]))

        # -------------
        # list if data

        ldata = [
            k0 for k0, v0 in vids.items()
            if all([v0.get(ss) is None for ss in ['ref0', 'mesh']])
        ]

        # ---------
        # add ref

        for i0, k0 in enumerate(ldata):

            ref = []
            long = vids[k0]['long']
            ss = str(long)
            nr = long.count('[')

            for ir in range(nr):

                kr = ss[ss.index('[')+1:ss.index(']')]

                if kr.isnumeric():
                    continue

                if kr in lref0:
                    ref.append(kr)
                elif kr == 'im2d':
                    ref.append(kr)

                ss = ss[ss.index(']')+1:]

            dshort[ids][k0]['ref'] = tuple(ref)

    return dshort
