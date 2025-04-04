

import copy


# ################################################
# ################################################
#              IDS Names
# ################################################


_DIDS = {
    "pulse_schedule": 'ps',
    "summary": 'sum',
    'equilibrium': 'eq',
    'core_profiles': 'cprof',
}


# ################################################
# ################################################
#              SHORTS
# ################################################


_DSHORT = {

    # ---------------
    # pulse_schedule
    # ---------------

    "pulse_schedule": {
        "flux_t": {
            'dim': 'time',
            'name': 'time',
            'long': 'flux_control.time[flux_nt]',
            'units': 's',
            'ref0': 'flux_nt',
        },
        "flux_ip": {
            'dim': 'current',
            'long': 'flux_control.i_plasma.reference[flux_nt]',
            'units': 'A',
        },
        "flux_li3": {
            'dim': 'inductance',
            'long': 'flux_control.li_3.reference[flux_nt]',
            'units': 'H',
        },
        "ne_t": {
            'dim': 'time',
            'name': 'time',
            'long': 'density_control.time[ne_nt]',
            'units': 's',
            'ref0': 'ne_nt',
        },
        "ne_neV": {
            'dim': 'density',
            'long': 'density_control.n_e_volume_average.reference[ne_nt]',
            'units': '1/m3',
        },
        "ic_t": {
            'dim': 'time',
            'name': 'time',
            'long': 'ic.time[ic_nt]',
            'units': 's',
            'ref0': 'ic_nt',
        },
        "ic_ic": {
            'dim': 'power',
            'long': 'ic.power.reference[ic_nt]',
            'units': 'W',
        },
    },

    # ---------------
    # summary
    # ---------------

    "summary": {

        # -----------
        # time traces

        't': {
            'dim': 'time',
            'name': 'time',
            'long': 'time[nt]',
            'units': 's',
            'ref0': 'nt',
        },
        'Q': {
            'dim': 'gain',
            'long': 'global_quantities.fusion_gain.value[nt]',
            'units': None,
        },
        'H98': {
            'dim': None,
            'long': 'global_quantities.h_98.value[nt]',
            'units': None,
        },
    },

    # ---------------
    # equilibrium
    # ---------------

    "equilibrium": {

        # -----------
        # time traces

        't': {
            'dim': 'time',
            'name': 'time',
            'long': 'time[nt]',
            'units': 's',
            'ref0': 'nt',
        },
        'Ip': {
            'dim': 'current',
            'long': 'time_slice[nt].global_quantities.ip',
            'units': 'A',
        },

        # -----------
        # boundary

        'sepR': {
            'dim': 'distance',
            'long': 'time_slice[nt].boundary.outline.r[nsep]',
            'units': 'm',
            'ref0': 'nsep',
        },
        'sepZ': {
            'dim': 'distance',
            'long': 'time_slice[nt].boundary.outline.z[nsep]',
            'units': 'm',
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
            'name': 'psi',
            'long': 'time_slice[nt].profiles_2d[im2d].psi',
            'units': 'Wb',
        },
        '2dphi': {
            'dim': 'B flux',
            'name': 'phi',
            'long': 'time_slice[nt].profiles_2d[im2d].phi',
            'units': 'Wb',
        },
        '2dBR': {
            'dim': 'B',
            'name': 'BR',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_r',
            'units': 'T',
        },
        '2dBZ': {
            'dim': 'B',
            'name': 'BZ',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_z',
            'units': 'T',
        },
        '2dBphi': {
            'dim': 'B',
            'name': 'Btor',
            'long': 'time_slice[nt].profiles_2d[im2d].b_field_tor',
            'units': 'T',
        },

        # ----------
        # 1d data

        '1ddpdpsi': {
            'dim': 'p / flux',
            'long': 'time_slice[nt].profiles_1d.dpressure_dpsi[im1d]',
            'units': 'Pa/Wb',
        },
        '1ddVdpsi': {
            'dim': 'vol / flux',
            'long': 'time_slice[nt].profiles_1d.dvolume_dpsi[im1d]',
            'units': 'm3/Wb',
        },
        '1dkappa': {
            'dim': 'adim',
            'long': 'time_slice[nt].profiles_1d.elongation[im1d]',
            'units': None,
        },
        '1djtor': {
            'dim': 'current density',
            'long': 'time_slice[nt].profiles_1d.j_tor[im1d]',
            'units': 'A/m2',
        },
        '1dp': {
            'dim': 'pressure',
            'long': 'time_slice[nt].profiles_1d.pressure[im1d]',
            'units': 'Pa',
        },
        '1dpsi': {
            'dim': 'B flux',
            'name': 'psi',
            'long': 'time_slice[nt].profiles_1d.psi[im1d]',
            'units': 'Wb',
        },
        '1dpsin': {
            'dim': 'B flux norm',
            'name': 'psin',
            'long': 'time_slice[nt].profiles_1d.psi_norm[im1d]',
            'units': None,
        },
        '1dphi': {
            'dim': 'B flux',
            'name': 'phi',
            'long': 'time_slice[nt].profiles_1d.phi[im1d]',
            'units': 'Wb',
        },
        '1dq': {
            'dim': 'q',
            'long': 'time_slice[nt].profiles_1d.q[im1d]',
            'units': None,
        },
        '1drin': {
            'dim': 'distance',
            'description': 'LCFS major radius, inboard',
            'long': 'time_slice[nt].profiles_1d.r_inboard[im1d]',
            'units': 'm',
        },
        '1drout': {
            'dim': 'distance',
            'description': 'LCFS major radius, outboard',
            'long': 'time_slice[nt].profiles_1d.r_outboard[im1d]',
            'units': 'm',
        },
        '1drhot': {
            'dim': 'rho',
            'name': 'rhot',
            'long': 'time_slice[nt].profiles_1d.rho_tor[im1d]',
            'units': None,
        },
        '1drhotn': {
            'dim': 'rho',
            'name': 'rhotn',
            'long': 'time_slice[nt].profiles_1d.rho_tor_norm[im1d]',
            'units': None,
        },
        '1dS': {
            'dim': 'surface',
            'long': 'time_slice[nt].profiles_1d.surface[im1d]',
            'units': 'm2',
        },
        '1dtrianglow': {
            'dim': 'triangularity',
            'long': 'time_slice[nt].profiles_1d.triangularity_lower[im1d]',
            'units': None,
        },
        '1dtriangup': {
            'dim': 'triangularity',
            'long': 'time_slice[nt].profiles_1d.triangularity_upper[im1d]',
            'units': None,
        },
        '1dV': {
            'dim': 'volume',
            'long': 'time_slice[nt].profiles_1d.volume[im1d]',
            'units': 'm3',
        },
    },

    # ---------------
    # core_profiles
    # ---------------

    "core_profiles": {

        # -----------
        # time traces

        't': {
            'dim': 'time',
            'name': 'time',
            'long': 'time[nt]',
            'units': 's',
            'ref0': 'nt',
        },
        'Vloop': {
            'dim': 'voltage',
            'name': 'Vloop',
            'long': 'global_quantities.v_loop[nt]',
            'units': 'V',
        },
        'TeVol': {
            'dim': 'Te',
            'name': 'Te',
            'long': 'global_quantities.t_e_volume_average[nt]',
            'units': 'eV',
        },
        'neVol': {
            'dim': 'density',
            'name': 'ne',
            'long': 'global_quantities.n_e_volume_average[nt]',
            'units': '1/m3',
        },
        'Ip': {
            'dim': 'current',
            'name': 'Ip',
            'long': 'global_quantities.ip[nt]',
            'units': 'A',
        },
        'betap': {
            'dim': 'beta',
            'name': 'betap',
            'long': 'global_quantities.beta_pol[nt]',
            'units': None,
        },
        'betat': {
            'dim': 'beta',
            'name': 'betat',
            'long': 'global_quantities.beta_tor[nt]',
            'units': None,
        },
        'betatn': {
            'dim': 'beta',
            'name': 'betatn',
            'long': 'global_quantities.beta_tor_norm[nt]',
            'units': None,
        },
        'Wdia': {
            'dim': 'Energy',
            'name': 'Wdia',
            'long': 'global_quantities.energy_diamagnetic[nt]',
            'units': 'J',
        },

        # -----------
        # profiles 1d

        '1dpsi': {
            'dim': 'B flux',
            'name': 'psi',
            'long': 'profiles_1d[nt].grid.psi[im1d]',
            'units': 'Wb',
        },
        '1dphi': {
            'dim': 'B flux',
            'name': 'phi',
            'long': 'profiles_1d[nt].grid.phi[im1d]',
            'units': 'Wb',
        },
        '1drhotn': {
            'dim': 'rho',
            'name': 'rhotn',
            'long': 'profiles_1d[nt].grid.rho_tor_norm[im1d]',
            'units': None,
        },
        '1drhopn': {
            'dim': 'rho',
            'name': 'rhopn',
            'long': 'profiles_1d[nt].grid.rho_pol_norm[im1d]',
            'units': None,
        },
        '1dV': {
            'dim': 'rho',
            'name': 'rhopn',
            'long': 'profiles_1d[nt].grid.rho_pol_norm[im1d]',
            'units': None,
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
            if all([v0.get(ss) is None for ss in ['mesh']])
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

                if kr in lref0 + ['im1d', 'im2d']:
                    ref.append(kr)

                ss = ss[ss.index(']')+1:]

            dshort[ids][k0]['ref'] = tuple(ref)

    return dshort
