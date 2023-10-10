"""
That's where the default shortcuts for imas2tofu are defined
Shortcuts allow you (via the class MulTiIDSLoader) to define short str
    for some important data contained in IMAS ids
The are stored as a large dict, with:
    {ids: {
        shortcut1: long_version1,
        ...
        shortcutN: long_version1}
        }

There is a default copy of this file in the tfu install, but each user can
(re)define his/her own shortcuts using a local copy that will take precedence
at import time.

To customize your tofu install with user-specific parameters, run in terminal:

    tofu-custom

This will create a .tofu/ in your home (~) with a local copy that you can edit
To see the shortcuts available from your running ipython console do:

    > import tof as tf
    > tf.imas2tofu.MultiIDSLoader.get_shortcuts()

Available since tofu 1.4.3
"""

import numpy as np


# ############################################################################
#
#           General imas2tofu parameters
#
# ############################################################################


# public imas user (used for checking if can be saved)
_IMAS_USER_PUBLIC = 'imas_public'

# generic imas parameters dict
_IMAS_DIDD = {
    'shot': 0,
    'run': 0,
    'refshot': -1,
    'refrun': -1,
    'user': _IMAS_USER_PUBLIC,
    'database': 'west',
    'version': '3',
    'backend': 'hdf5',
}

_T0 = False

# ############################################################################
#
#           short versions of ids names
#
# ############################################################################


_dshortids = {
    'wall': 'wall',
    'pulse_schedule': 'sched',
    'summary': 'sum',
    'equilibrium': 'eq',
    'core_profiles': 'corprof',
    'edge_profiles': 'edgprof',
    'core_sources': 'corsour',
    'edge_sources': 'edgsour',
    'lh_antennas': 'lh',
    'ic_antennas': 'ic',
    'magnetics': 'mag',
    'barometry': 'baro',
    'calorimetry': 'calo',
    'neutron_diagnostic': 'neutrons',
    'ece': 'ece',
    'reflectometer_profile': 'reflecto',
    'interferometer': 'interf',
    'polarimeter': 'pola',
    'bolometer': 'bolo',
    'soft_x_rays': 'sxr',
    'spectrometer_visible': 'spectrovis',
    'bremsstrahlung_visible': 'brem',
}


# ############################################################################
#
#           shortcuts for imas2tofu interface (MultiIDSLoader class)
#
# ############################################################################


_dshort = {
    'wall': {
        'wallR': {'str': 'description_2d[0].limiter.unit[0].outline.r',
                  'units': 'm'},
        'wallZ': {'str': 'description_2d[0].limiter.unit[0].outline.z',
                  'units': 'm'},
    },

    'pulse_schedule': {
        'events_times': {'str': 'event[].time_stamp',
                         'units': 's'},
        'events_names': {'str': 'event[].identifier'},
    },

    'summary': {
        't': {
            'str': 'time',
            'units': 's',
        },
        'power_ic': {
            'str': 'heating_current_drive.power_ic.value',
            'units': 'W',
        },
        'power_lh': {
            'str': 'heating_current_drive.power_lh.value',
            'units': 'W',
        },
        'power_ec': {
            'str': 'heating_current_drive.power_ec.value',
            'units': 'W',
        },
        'power_nbi': {
            'str': 'heating_current_drive.power_nbi.value',
            'units': 'W',
        },
        'ip': {
            'str': 'global_quantities.ip.value',
            'units': 'A',
        },
        'v_loop': {
            'str': 'global_quantities.v_loop.value',
            'units': 'V',
        },
        'beta_tor_norm': {
            'str': 'global_quantities.beta_tor_norm.value',
            'units': None,
        },
        'beta_tor': {
            'str': 'global_quantities.beta_tor.value',
            'units': None,
        },
        'beta_pol': {
            'str': 'global_quantities.beta_pol.value',
            'units': None,
        },
        'energy_total': {
            'str': 'global_quantities.energy_total.value',
            'units': 'J',
        },
        'volume': {
            'str': 'global_quantities.volume.value',
            'units': 'm^3',
        },
        'fusion_gain': {
            'str': 'global_quantities.fusion_gain.value',
            'units': None,
        },
        'tau_resistive': {
            'str': 'global_quantities.tau_resistive.value',
            'units': 's',
        },
        'tau_energy': {
            'str': 'global_quantities.tau_energy.value',
            'units': 's',
        },
        'fusion_fluence': {
            'str': 'global_quantities.fusion_fluence.value',
            'units': 'J',
        },
        'ng_fraction': {
            'str': 'global_quantities.greenwald_fraction.value',
            'units': None,
        },
        'q95': {
            'str': 'global_quantities.q_95.value',
            'units': None,
        },
        'li': {
            'str': 'global_quantities.li.value',
            'units': None,
        },
        'r0': {
            'str': 'global_quantities.r0.value',
            'units': 'm',
        },
        'b0': {
            'str': 'global_quantities.b0.value',
            'units': 'T',
        },
        'fusion_power': {
            'str': 'fusion.power.value',
            'units': 'W',
        },
        'neutrons_flux': {
            'str': 'fusion.neutron_fluxes.total.value',
            'units': 'Hz',
        },
        'neutrons_power': {
            'str': 'fusion.neutron_power_total.value',
            'units': 'W',
        },
        'runaway_current': {
            'str': 'runaways.current.value',
            'units': 'A',
        },
        'runaway_particles': {
            'str': 'runaways.particles.value',
            'units': None,
        },
    },

    'equilibrium': {
        't': {'str': 'time', 'units': 's'},
        'ip': {'str': 'time_slice[time].global_quantities.ip',
               'dim': 'current', 'quant': 'Ip', 'units': 'A'},
        'q0': {'str': 'time_slice[time].global_quantities.q_axis',
               'units': None},
        'qmin': {'str': 'time_slice[time].global_quantities.q_min.value',
                 'units': None},
        'q95': {'str': 'time_slice[time].global_quantities.q_95',
                'units': None},
        'volume': {'str': 'time_slice[time].global_quantities.volume',
                   'dim': 'volume', 'quant': 'pvol', 'units': 'm^3'},
        'psiaxis': {'str': 'time_slice[time].global_quantities.psi_axis',
                    'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        'psisep': {'str': 'time_slice[time].global_quantities.psi_boundary',
                   'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        'BT0': {'str': ('time_slice[time].global_quantities'
                        + '.magnetic_axis.b_field_tor'),
                'dim': 'B', 'quant': 'BT', 'units': 'T'},
        'axR': {'str': 'time_slice[time].global_quantities.magnetic_axis.r',
                'dim': 'distance', 'quant': 'R', 'units': 'm'},
        'axZ': {'str': 'time_slice[time].global_quantities.magnetic_axis.z',
                'dim': 'distance', 'quant': 'Z', 'units': 'm'},
        'x0R': {'str': 'time_slice[time].boundary.x_point[0].r', 'units': 'm'},
        'x0Z': {'str': 'time_slice[time].boundary.x_point[0].z', 'units': 'm'},
        'x1R': {'str': 'time_slice[time].boundary.x_point[1].r', 'units': 'm'},
        'x1Z': {'str': 'time_slice[time].boundary.x_point[1].z', 'units': 'm'},
        'strike0R': {'str': 'time_slice[time].boundary.strike_point[0].r',
                     'units': 'm'},
        'strike0Z': {'str': 'time_slice[time].boundary.strike_point[0].z',
                     'units': 'm'},
        'strike1R': {'str': 'time_slice[time].boundary.strike_point[1].r',
                     'units': 'm'},
        'strike1Z': {'str': 'time_slice[time].boundary.strike_point[1].z',
                     'units': 'm'},
        'sepR': {'str': 'time_slice[time].boundary_separatrix.outline.r',
                 'units': 'm'},
        'sepZ': {'str': 'time_slice[time].boundary_separatrix.outline.z',
                 'units': 'm'},

        '1drhotn': {'str': 'time_slice[time].profiles_1d.rho_tor_norm',
                    'dim': 'rho', 'quant': 'rhotn', 'units': None},
        '1dphi': {'str': 'time_slice[time].profiles_1d.phi',
                  'dim': 'B flux', 'quant': 'phi', 'units': 'Wb'},
        '1dpsi': {'str': 'time_slice[time].profiles_1d.psi',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1dq': {'str': 'time_slice[time].profiles_1d.q',
                'dim': 'safety factor', 'quant': 'q', 'units': None},
        '1dpe': {'str': 'time_slice[time].profiles_1d.pressure',
                 'dim': 'pressure', 'quant': 'pe', 'units': 'Pa'},
        '1djT': {'str': 'time_slice[time].profiles_1d.j_tor',
                 'dim': 'vol. current dens.', 'quant': 'jT',
                 'units': 'A.m^-2'},

        '2dphi': {'str': 'time_slice[time].ggd[0].phi[0].values',
                  'dim': 'B flux', 'quant': 'phi', 'units': 'Wb'},
        '2dpsi': {'str': 'time_slice[time].ggd[0].psi[0].values',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '2djT': {'str': 'time_slice[time].ggd[0].j_tor[0].values',
                 'dim': 'vol. current dens.', 'quant': 'jT',
                 'units': 'A.m^-2'},
        '2dBR': {'str': 'time_slice[time].ggd[0].b_field_r[0].values',
                 'dim': 'B', 'quant': 'BR', 'units': 'T'},
        '2dBT': {'str': 'time_slice[time].ggd[0].b_field_tor[0].values',
                 'dim': 'B', 'quant': 'BT', 'units': 'T'},
        '2dBZ': {'str': 'time_slice[time].ggd[0].b_field_z[0].values',
                 'dim': 'B', 'quant': 'BZ', 'units': 'T'},
        '2dmeshNodes': {'str': ('grids_ggd[0].grid[0].space[0]'
                                + '.objects_per_dimension[0]'
                                + '.object[].geometry'),
                        'units': 'mixed'},
        '2dmeshFaces': {'str': ('grids_ggd[0].grid[0].space[0]'
                                + '.objects_per_dimension[2]'
                                + '.object[].nodes')},
        '2dmeshR': {'str': 'time_slice[0].profiles_2d[0].r', 'units': 'm'},
        '2dmeshZ': {'str': 'time_slice[0].profiles_2d[0].z', 'units': 'm'},
        },

    'core_profiles': {
        't': {'str': 'time', 'units': 's'},
        'ip': {'str': 'global_quantities.ip',
               'dim': 'current', 'quant': 'Ip', 'units': 'A'},
        'vloop': {'str': 'global_quantities.v_loop',
                  'dim': 'voltage', 'quant': 'Vloop', 'units': 'V'},

        '1dTe': {'str': 'profiles_1d[time].electrons.temperature',
                 'dim': 'temperature',  'quant': 'Te', 'units': 'eV'},
        '1dne': {'str': 'profiles_1d[time].electrons.density',
                 'dim': 'density', 'quant': 'ne', 'units': 'm^-3'},
        '1dzeff': {'str': 'profiles_1d[time].zeff',
                   'dim': 'charge', 'quant': 'zeff', 'units': None},
        '1dpsi': {'str': 'profiles_1d[time].grid.psi',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1drhotn': {'str': 'profiles_1d[time].grid.rho_tor_norm',
                    'dim': 'rho', 'quant': 'rhotn', 'units': None},
        '1drhopn': {'str': 'profiles_1d[time].grid.rho_pol_norm',
                    'dim': 'rho', 'quant': 'rhopn', 'units': None},
        '1dnW': {'str': 'profiles_1d[time].ion[identifier.label=W].density',
                 'dim': 'density', 'quant': 'nI', 'units': 'm^-3'},
        '1dTi_av': {
            'str': 'profiles_1d[time].t_i_average',
            'units': 'eV',
            'dim': 'temperature',
            'quant': 'Ti',
        },
    },

    'edge_profiles': {
        't': {'str': 'time', 'units': 's'},
    },

    'core_sources': {
        't': {'str': 'time', 'units': 's'},
        '1dpsi': {'str': ('source[identifier.name=lineradiation]'
                          + '.profiles_1d[time].grid.psi'),
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1drhotn': {'str': ('source[identifier.name=lineradiation]'
                            + '.profiles_1d[time].grid.rho_tor_norm'),
                    'dim': 'rho', 'quant': 'rhotn', 'units': None},
        '1dbrem': {'str': ('source[identifier.name=bremsstrahlung]'
                           + '.profiles_1d[time].electrons.energy'),
                   'dim': 'vol.emis.', 'quant': 'brem.', 'units': 'W.m^-3'},
        '1dline': {'str': ('source[identifier.name=lineradiation]'
                           + '.profiles_1d[time].electrons.energy'),
                   'dim': 'vol. emis.', 'quant': 'lines', 'units': 'W.m^-3'}},

    'edge_sources': {
        't': {'str': 'time', 'units': 's'},
        '2dmeshNodes': {'str': ('grid_ggd[0].space[0].objects_per_dimension[0]'
                                + '.object[].geometry'),
                        'units': 'mixed'},
        '2dmeshFaces': {'str': ('grid_ggd[0].space[0].objects_per_dimension[2]'
                                + '.object[].nodes')},
        '2dradiation': {'str': 'source[13].ggd[0].electrons.energy[0].values',
                        'dim': 'vol. emis.', 'quant': 'vol.emis.',
                        'name': 'tot. vol. emis.', 'units': 'W.m^-3'}},

    'lh_antennas': {
        't': {'str': 'antenna[chan].power_launched.time', 'units': 's'},
        'power0': {'str': 'antenna[0].power_launched.data',
                   'dim': 'power', 'quant': 'lh power', 'units': 'W',
                   'pos': True},
        'power1': {'str': 'antenna[1].power_launched.data',
                   'dim': 'power', 'quant': 'lh power', 'units': 'W',
                   'pos': True},
        'power': {'str': 'antenna[chan].power_launched.data',
                  'dim': 'power', 'quant': 'lh power', 'units': 'W',
                  'pos': True},
        'R': {'str': 'antenna[chan].position.r.data',
              'dim': 'distance', 'quant': 'R', 'units': 'm'}},

    'ic_antennas': {
        't': {'str': 'antenna[chan].module[0].power_forward.time',
              'units': 's'},
        'power0mod_fwd': {'str': 'antenna[0].module[].power_forward.data',
                          'dim': 'power', 'quant': 'ic power', 'units': 'W'},
        'power0mod_reflect': {'str': ('antenna[0].module[]'
                                      + '.power_reflected.data'),
                              'dim': 'power', 'quant': 'ic power',
                              'units': 'W'},
        'power1mod_fwd': {'str': 'antenna[1].module[].power_forward.data',
                          'dim': 'power', 'quant': 'ic power', 'units': 'W'},
        'power1mod_reflect': {'str': ('antenna[1].module[]'
                                      + '.power_reflected.data'),
                              'dim': 'power', 'quant': 'ic power',
                              'units': 'W'},
        'power2mod_fwd': {'str': 'antenna[2].module[].power_forward.data',
                          'dim': 'power', 'quant': 'ic power', 'units': 'W'},
        'power2mod_reflect': {'str': ('antenna[2].module[]'
                                      + '.power_reflected.data'),
                              'dim': 'power', 'quant': 'ic power',
                              'units': 'W'}},

    'magnetics': {
        't': {'str': 'time', 'units': 's'},
        'ip': {'str': 'method[0].ip.data', 'units': 'A'},
        'diamagflux': {'str': 'method[0].diamagnetic_flux.data',
                       'units': 'Wb'},
        'bpol_B': {'str': 'bpol_probe[chan].field.data',
                   'dim': 'B', 'quant': 'Bpol', 'units': 'T'},
        'bpol_name': {'str': 'bpol_probe[chan].name'},
        'bpol_R': {'str': 'bpol_probe[chan].position.r',
                   'dim': 'distance', 'quant': 'R', 'units': 'm'},
        'bpol_Z': {'str': 'bpol_probe[chan].position.z',
                   'dim': 'distance', 'quant': 'Z', 'units': 'm'},
        'bpol_angpol': {'str': 'bpol_probe[chan].poloidal_angle',
                        'dim': 'angle', 'quant': 'angle_pol', 'units': 'rad'},
        'bpol_angtor': {'str': 'bpol_probe[chan].toroidal_angle',
                        'dim': 'angle', 'quant': 'angle_tor', 'units': 'rad'},
        'floop_flux': {'str': 'flux_loop[chan].flux.data',
                       'dim': 'B flux', 'quant': 'B flux', 'units': 'Wb'},
        'floop_name': {'str': 'flux_loop[chan].name'},
        'floop_R': {'str': 'flux_loop[chan].position.r',
                    'dim': 'distance', 'quant': 'R', 'units': 'm'},
        'floop_Z': {'str': 'flux_loop[chan].position.z',
                    'dim': 'distance', 'quant': 'Z', 'units': 'm'}},

    'barometry': {
        't': {'str': 'gauge[chan].pressure.time', 'units': 's'},
        'names': {'str': 'gauge[chan].name'},
        'p': {'str': 'gauge[chan].pressure.data',
              'dim': 'pressure', 'quant': 'p', 'units': 'Pa'}},

    'calorimetry': {
        't': {'str': 'group[chan].component[0].power.time', 'units': 's'},
        'names': {'str': 'group[chan].name'},
        'power': {'str': 'group[chan].component[0].power.data',
                  'dim': 'power', 'quant': 'extracted power',
                  'units': 'W'}},

    'neutron_diagnostic': {
        't': {'str': 'time', 'units': 's'},
        'flux_total': {'str': 'synthetic_signals.total_neutron_flux',
                       'dim': 'particle flux', 'quant': 'particle flux',
                       'units': 's^-1'}},

    'ece': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'freq': {'str': 'channel[chan].frequency.data',
                 'dim': 'freq', 'quant': 'freq', 'units': 'Hz'},
        'Te': {'str': 'channel[chan].t_e.data',
               'dim': 'temperature', 'quant': 'Te', 'units': 'eV'},
        'R': {'str': 'channel[chan].position.r',
              'dim': 'distance', 'quant': 'R', 'units': 'm'},
        'rhotn': {'str': 'channel[chan].position.rho_tor_norm',
                  'dim': 'rho', 'quant': 'rhotn', 'units': None},
        'theta': {'str': 'channel[chan].position.theta',
                  'dim': 'angle', 'quant': 'theta', 'units': 'rad'},
        'tau1keV': {'str': 'channel[chan].optical_depth.data',
                    'dim': 'optical_depth', 'quant': 'tau', 'units': None},
        'validity_timed': {'str': 'channel[chan].t_e.validity_timed'},
        'names': {'str': 'channel[chan].name'},
        'Te0': {'str': 't_e_central.data',
                'dim': 'temperature', 'quant': 'Te', 'units': 'eV'}},

    'reflectometer_profile': {
        't': {'str': 'time', 'units': 's'},
        'ne': {'str': 'channel[chan].n_e.data',
               'dim': 'density', 'quant': 'ne', 'units': 'm^-3'},
        'R': {'str': 'channel[chan].position.r',
              'dim': 'distance', 'quant': 'R', 'units': 'm'},
        'Z': {'str': 'channel[chan].position.z',
              'dim': 'distance', 'quant': 'Z', 'units': 'm'},
        'phi': {'str': 'channel[chan].position.phi',
                'dim': 'angle', 'quant': 'phi', 'units': 'rad'},
        'names': {'str': 'channel[chan].name'},
        'mode': {'str': 'mode'},
        'sweep': {'str': 'sweep_time'}},

    'interferometer': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'names': {'str': 'channel[chan].name'},
        'ne_integ': {'str': 'channel[chan].n_e_line.data',
                     'dim': 'ne_integ', 'quant': 'ne_integ',
                     'units': 'm^-2', 'Brightness': True}},

    'polarimeter': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'lamb': {'str': 'channel[chan].wavelength',
                 'dim': 'distance', 'quant': 'wavelength',
                 'units': 'm'},
        'fangle': {'str': 'channel[chan].faraday_angle.data',
                   'dim': 'angle', 'quant': 'faraday angle',
                   'units': 'rad', 'Brightness': True},
        'names': {'str': 'channel[chan].name'}},

    'bolometer': {
        't': {'str': 'channel[chan].power.time',
              'quant': 't', 'units': 's'},
        'power': {'str': 'channel[chan].power.data',
                  'dim': 'power', 'quant': 'power radiative',
                  'units': 'W', 'Brightness': False},
        'etendue': {'str': 'channel[chan].etendue',
                    'dim': 'etendue', 'quant': 'etendue',
                    'units': 'm^2.sr'},
        'names': {'str': 'channel[chan].name'},
        'tpower': {'str': 'time', 'quant': 't', 'units': 's'},
        'prad': {'str': 'power_radiated_total',
                 'dim': 'power', 'quant': 'power radiative',
                 'units': 'W'},
        'pradbulk': {'str': 'power_radiated_inside_lcfs',
                     'dim': 'power', 'quant': 'power radiative',
                     'units': 'W'}},

    'soft_x_rays': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'power': {'str': 'channel[chan].power.data',
                  'dim': 'power', 'quant': 'power radiative',
                  'units': 'W', 'Brightness': False},
        'brightness': {'str': 'channel[chan].brightness.data',
                       'dim': 'brightness', 'quant': 'brightness',
                       'units': 'W.m^-2.sr^-1', 'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'etendue': {'str': 'channel[chan].etendue',
                    'dim': 'etendue', 'quant': 'etendue',
                    'units': 'm^2.sr'}},

    'spectrometer_visible': {
        't': {'str': ('channel[chan].grating_spectrometer'
                      + '.radiance_spectral.time'),
              'quant': 't', 'units': 's'},
        'spectra': {'str': ('channel[chan].grating_spectrometer'
                            + '.radiance_spectral.data'),
                    'dim': 'radiance_spectral',
                    'quant': 'radiance_spectral',
                    'units': '(photons).m^-2.s^-1.sr^-1.m^-1',
                    'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'lamb': {'str': 'channel[chan].grating_spectrometer.wavelengths',
                 'dim': 'wavelength', 'quant': 'wavelength', 'units': 'm'}},

    'bremsstrahlung_visible': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'radiance': {'str': 'channel[chan].radiance_spectral.data',
                     'dim': 'radiance_spectral',
                     'quant': 'radiance_spectral',
                     'units': '(photons).m^-2.s^-1.sr^-1.m^-1',
                     'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'lamb_up': {'str': 'channel[chan].filter.wavelength_upper',
                    'units': 'm'},
        'lamb_lo': {'str': 'channel[chan].filter.wavelength_lower',
                    'units': 'm'},
    },

    'spectrometer_x_ray_crystal': {
        't': {
            'str': 'time',
            'dim': 'time',
            'quant': 'time',
            'units': 's',
        },
        'Te': {
            'str': 'channel[0].profiles_line_integrated.t_e.data',
            'dim': 'temperature',
            'quant': 'Ti',
            'units': 'eV',
        },
        'Te_valid': {
            'str': 'channel[0].profiles_line_integrated.t_e.validity_timed',
            'dim': '',
            'quant': '',
            'units': '',
        },
        'Ti': {
            'str': 'channel[0].profiles_line_integrated.t_i.data',
            'dim': 'temperature',
            'quant': 'Ti',
            'units': 'eV',
        },
        'Ti_valid': {
            'str': 'channel[0].profiles_line_integrated.t_i.validity_timed',
            'dim': '',
            'quant': '',
            'units': '',
        },
        'vi': {
            'str': 'channel[0].profiles_line_integrated.velocity_tor.data',
            'dim': 'velocity',
            'quant': 'velocity',
            'units': 'm/s',
        },
        'vi_valid': {
            'str': 'channel[0].profiles_line_integrated.velocity_tor.validity_timed',
            'dim': '',
            'quant': '',
            'units': '',
        },
        'rhotn_sign': {
            'str': (
                'channel[0].profiles_line_integrated.'
                'lines_of_sight_rho_tor_norm.data'
            ),
            'dim': 'rho',
            'quant': 'rhotn',
            'units': None,
        },
        'rhotn_sign_valid': {
            'str': (
                'channel[0].profiles_line_integrated.'
                'lines_of_sight_rho_tor_norm.validity_timed'
            ),
            'dim': '',
            'quant': '',
            'units': '',
        },
        'data_raw': {
            'str': 'channel[0].frame[time].counts_n',
            'dim': '',
            'quant': 'counts',
            'units': 'counts',
        },
        'code_parameters': {
            'str': 'code.parameters',
            'dim': '',
            'quant': '',
            'units': '',
        },
    },
    }


# ############################################################################
#
#           default data for each ids (not used yet)
#
# ############################################################################


_didsdiag = {
    'lh_antennas': {'datacls': 'DataCam1D',
                    'geomcls': False,
                    'sig': {'data': 'power',
                            't': 't'}},
    'ic_antennas': {'datacls': 'DataCam1D',
                    'geomcls': False,
                    'sig': {'data': 'power',
                            't': 't'}},
    'magnetics': {'datacls': 'DataCam1D',
                  'geomcls': False,
                  'sig': {'data': 'bpol_B',
                          't': 't'}},
    'barometry': {'datacls': 'DataCam1D',
                  'geomcls': False,
                  'sig': {'data': 'p',
                          't': 't'}},
    'calorimetry': {'datacls': 'DataCam1D',
                    'geomcls': False,
                    'sig': {'data': 'power',
                            't': 't'}},
    'ece': {'datacls': 'DataCam1D',
            'geomcls': False,
            'sig': {'t': 't',
                    'X': 'rhotn_sign',
                    'data': 'Te'},
            'stack': True},
    'neutron_diagnostic': {'datacls': 'DataCam1D',
                           'geomcls': False,
                           'sig': {'t': 't',
                                   'data': 'flux_total'}},
    'reflectometer_profile': {'datacls': 'DataCam1D',
                              'geomcls': False,
                              'sig': {'t': 't',
                                      'X': 'R',
                                      'data': 'ne'}},
    'interferometer': {'datacls': 'DataCam1D',
                       'geomcls': 'CamLOS1D',
                       'sig': {'t': 't',
                               'data': 'ne_integ'},
                       'synth': {'dsynth': {
                           'quant': 'core_profiles.1dne',
                           'ref1d': 'core_profiles.1drhotn',
                           'ref2d': 'equilibrium.2drhotn'},
                               'dsig': {'core_profiles': ['t'],
                                        'equilibrium': ['t']},
                               'Brightness': True},
                       'stack': True},
    'polarimeter': {'datacls': 'DataCam1D',
                    'geomcls': 'CamLOS1D',
                    'sig': {'t': 't',
                            'data': 'fangle'},
                    'synth': {'dsynth': {
                        'fargs': ['core_profiles.1dne',
                                  'equilibrium.2dBR',
                                  'equilibrium.2dBT',
                                  'equilibrium.2dBZ',
                                  'core_profiles.1drhotn',
                                  'equilibrium.2drhotn']},
                              'dsig': {'core_profiles': ['t'],
                                       'equilibrium': ['t']},
                              'Brightness': True},
                    'stack': True},
    'bolometer': {'datacls': 'DataCam1D',
                  'geomcls': 'CamLOS1D',
                  'sig': {'t': 't',
                          'data': 'power'},
                  'synth': {'dsynth': {
                      'quant': 'core_sources.1dprad',
                      'ref1d': 'core_sources.1drhotn',
                      'ref2d': 'equilibrium.2drhotn'},
                            'dsig': {'core_sources': ['t'],
                                     'equilibrium': ['t']},
                            'Brightness': False},
                  'stack': True},
    'soft_x_rays': {'datacls': 'DataCam1D',
                    'geomcls': 'CamLOS1D',
                    'sig': {'t': 't',
                            'data': 'power'},
                    'stack': True},
    'spectrometer_visible': {'datacls': 'DataCam1DSpectral',
                             'geomcls': 'CamLOS1D',
                             'sig': {'data': 'spectra',
                                     't': 't',
                                     'lamb': 'lamb'}},
    'bremsstrahlung_visible': {
        'datacls': 'DataCam1D',
        'geomcls': 'CamLOS1D',
        'sig': {
            't': 't',
            'data': 'radiance',
        },
        'synth': {
            'dsynth': {
                'quant': [
                    'core_profiles.1dTe',
                    'core_profiles.1dne',
                    'core_profiles.1dzeff',
                ],
                'ref1d': 'core_profiles.1drhotn',
                'ref2d': 'equilibrium.2drhotn',
            },
            'dsig': {
                'core_profiles': ['t'],
                'equilibrium': ['t'],
            },
            'Brightness': True,
        },
        'stack': True
    },
    'spectrometer_x_ray_crystal': {
        'geomcls': 'CamLOS1D',
        'sig': {
            't': 't',
        },
    },
    }


# ############################################################################
#
#           Complete dshort and didsdiag
#
# ############################################################################


_lidsconfig = ['wall']
_lidsdiag = sorted([kk for kk, vv in _didsdiag.items() if 'sig' in vv.keys()])
_lidslos = list(_lidsdiag)
for ids_ in _lidsdiag:
    if _didsdiag[ids_]['geomcls'] not in ['CamLOS1D']:
        _lidslos.remove(ids_)

for ids in _lidslos:
    dlos = {}
    strlos = 'line_of_sight'
    if ids == 'reflectometer_profile':
        strlos += '_detection'

    if ids == 'spectrometer_x_ray_crystal':
        strbase = 'channel[0].profiles_line_integrated'
        dlos = {
            'los_pt1R': {
                'str': 'channel[0].crystal.centre.r',
                'units': 'm',
            },
            'los_pt1Z': {
                'str': 'channel[0].crystal.centre.z',
                'units': 'm',
            },
            'los_pt1Phi': {
                'str': 'channel[0].crystal.centre.phi',
                'units': 'm',
            },
            'los_pt2R': {
                'str': f'{strbase}.lines_of_sight_second_point.r',
                'units': 'm',
            },
            'los_pt2Z': {
                'str': f'{strbase}.lines_of_sight_second_point.z',
                'units': 'm',
            },
            'los_pt2Phi': {
                'str': f'{strbase}.lines_of_sight_second_point.phi',
                'units': 'm',
            },
        }

    else:
        dlos['los_pt1R'] = {
            'str': 'channel[chan].{}.first_point.r'.format(strlos),
            'units': 'm'}
        dlos['los_pt1Z'] = {
            'str': 'channel[chan].{}.first_point.z'.format(strlos),
            'units': 'm'}
        dlos['los_pt1Phi'] = {
            'str': 'channel[chan].{}.first_point.phi'.format(strlos),
            'units': 'rad'}
        dlos['los_pt2R'] = {
            'str': 'channel[chan].{}.second_point.r'.format(strlos),
            'units': 'm'}
        dlos['los_pt2Z'] = {
            'str': 'channel[chan].{}.second_point.z'.format(strlos),
            'units': 'm'}
        dlos['los_pt2Phi'] = {
            'str': 'channel[chan].{}.second_point.phi'.format(strlos),
            'units': 'rad'}
    _dshort[ids].update(dlos)


_lidssynth = sorted([kk for kk, vv in _didsdiag.items()
                     if 'synth' in vv.keys()])
for ids_ in _lidssynth:
    for kk, vv in _didsdiag[ids_]['synth']['dsynth'].items():
        if type(vv) is str:
            vv = [vv]
        for ii in range(0, len(vv)):
            v0, v1 = vv[ii].split('.')
            if v0 not in _didsdiag[ids_]['synth']['dsig'].keys():
                _didsdiag[ids_]['synth']['dsig'][v0] = [v1]
            elif v1 not in _didsdiag[ids_]['synth']['dsig'][v0]:
                _didsdiag[ids_]['synth']['dsig'][v0].append(v1)
        _didsdiag[ids_]['synth']['dsynth'][kk] = vv


# ############################################################################
#
#           Dict for computing signals from loaded signals
#
# ############################################################################


# -------------
# Functions

def _events(names, t):
    ustr = 'U{}'.format(np.nanmax(np.char.str_len(np.char.strip(names))))
    return np.array([(nn, tt)
                     for nn, tt in zip(*[np.char.strip(names), t])],
                    dtype=[('name', ustr), ('t', np.float)])


def _RZ2array(ptsR, ptsZ):
    out = np.array([ptsR, ptsZ]).T
    if out.ndim == 1:
        out = out[None, :]
    return out


def _losptsRZP(*pt12RZP):
    return np.swapaxes([pt12RZP[:3], pt12RZP[3:]], 0, 1).T


def _losptsRZP2(*pt12RZP):
    nlos = pt12RZP[3].size
    return np.swapaxes(
        [
            (
                np.full((nlos,), pt12RZP[0]),
                np.full((nlos,), pt12RZP[1]),
                np.full((nlos,), pt12RZP[2]),
            ),
            pt12RZP[3:]
        ], 0, 1).T


def _add(a0, a1):
    return np.abs(a0 + a1)


def _eqB(BT, BR, BZ):
    return np.sqrt(BT**2 + BR**2 + BZ**2)


def _icmod(al, ar, axis=0):
    return np.sum(al - ar, axis=axis)


def _icmodadd(al0, ar0, al1, ar1, al2, ar2, axis=0):
    return (np.sum(al0 - ar0, axis=axis)
            + np.sum(al1 - ar1, axis=axis)
            + np.sum(al2 - ar2, axis=axis))


def _rhopn1d(psi):
    return np.sqrt((psi - psi[:, 0:1]) / (psi[:, -1] - psi[:, 0])[:, None])


def _rhopn2d(psi, psi0, psisep):
    return np.sqrt(
        (psi - psi0[:, None]) / (psisep[:, None] - psi0[:, None]))


def _rhotn2d(phi):
    return np.sqrt(np.abs(phi) / np.nanmax(np.abs(phi), axis=1)[:, None])


def _eqSep(sepR, sepZ, npts=100):
    nt = len(sepR)
    assert len(sepZ) == nt
    sep = np.full((nt, npts, 2), np.nan)
    pts = np.linspace(0, 100, npts)
    for ii in range(0, nt):
        ptsii = np.linspace(0, 100, sepR[ii].size)
        sep[ii, :, 0] = np.interp(pts, ptsii, sepR[ii])
        sep[ii, :, 1] = np.interp(pts, ptsii, sepZ[ii])
    return sep


def _eqtheta(axR, axZ, nodes, cocos=11):
    theta = np.arctan2(nodes[:, 0][None, :] - axZ[:, None],
                       nodes[:, 1][None, :] - axR[:, None])
    if cocos == 1:
        theta = -theta
    return theta


def _rhosign(rho, theta):
    if isinstance(theta, np.ndarray):
        rhotns = np.array(rho)
        ind = ~np.isnan(theta)
        ind[ind] &= np.cos(theta[ind]) < 0.
        rhotns[ind] = -rho[ind]
    else:
        rhotns = [None for ii in range(len(theta))]
        for ii in range(len(theta)):
            rhotns[ii] = _rhosign(rho[ii], theta[ii])
    return rhotns


def _lamb(lamb_up, lamb_lo):
    return 0.5*(lamb_up + lamb_lo)


# ----------
# dict


_dcomp = {
          'pulse_schedule':
          {'events': {'lstr': ['events_names', 'events_times'],
                      'func': _events}},

          'wall':
          {'wall': {'lstr': ['wallR', 'wallZ'],
                    'func': _RZ2array}},

          'equilibrium':
          {'ax': {'lstr': ['axR', 'axZ'], 'func': _RZ2array, 'units': 'm'},
           'sep': {'lstr': ['sepR', 'sepZ'],
                   'func': _eqSep, 'kargs': {'npts': 100}, 'units': 'm'},
           '2dB': {'lstr': ['2dBT', '2dBR', '2dBZ'], 'func': _eqB,
                   'dim': 'B', 'quant': 'B', 'units': 'T'},
           '1drhopn': {'lstr': ['1dpsi', 'psiaxis', 'psisep'],
                       'func': _rhopn2d,
                       'dim': 'rho', 'quant': 'rhopn', 'units': None},
           '2drhopn': {'lstr': ['2dpsi', 'psiaxis', 'psisep'],
                       'func': _rhopn2d,
                       'dim': 'rho', 'quant': 'rhopn', 'units': None},
           '2drhotn': {'lstr': ['2dphi'], 'func': _rhotn2d,
                       'dim': 'rho', 'quant': 'rhotn', 'units': None},
           'x0': {'lstr': ['x0R', 'x0Z'], 'func': _RZ2array, 'units': 'm'},
           'x1': {'lstr': ['x1R', 'x1Z'], 'func': _RZ2array, 'units': 'm'},
           'strike0': {'lstr': ['strike0R', 'strike0Z'], 'func': _RZ2array,
                       'units': 'm'},
           'strike1': {'lstr': ['strike1R', 'strike1Z'], 'func': _RZ2array,
                       'units': 'm'},
           '2dtheta': {'lstr': ['axR', 'axZ', '2dmeshNodes'],
                       'func': _eqtheta, 'kargs': {'cocos': 11},
                       'units': 'rad'}},

          'core_profiles':
          {'1drhopn': {'lstr': ['1dpsi'], 'func': _rhopn1d,
                       'dim': 'rho', 'quant': 'rhopn', 'units': None}},

          'core_sources':
          {'1drhopn': {'lstr': ['1dpsi'], 'func': _rhopn1d,
                       'dim': 'rho', 'quant': 'rhopn', 'units': None},
           '1dprad': {'lstr': ['1dbrem', '1dline'], 'func': _add,
                      'dim': 'vol. emis.', 'quant': 'prad',
                      'units': 'W.m^-3'}},

          'magnetics':
          {'bpol_pos': {'lstr': ['bpol_R', 'bpol_Z'], 'func': _RZ2array},
           'floop_pos': {'lstr': ['floop_R', 'floop_Z'], 'func': _RZ2array}},

          'ic_antennas': {
              'power0': {'lstr': ['power0mod_fwd', 'power0mod_reflect'],
                         'func': _icmod, 'kargs': {'axis': 0},
                         'pos': True, 'units': 'W'},
              'power1': {'lstr': ['power1mod_fwd', 'power1mod_reflect'],
                         'func': _icmod, 'kargs': {'axis': 0},
                         'pos': True, 'units': 'W'},
              'power2': {'lstr': ['power2mod_fwd', 'power2mod_reflect'],
                         'func': _icmod, 'kargs': {'axis': 0},
                         'pos': True, 'units': 'W'},
              'power': {'lstr': ['power0mod_fwd', 'power0mod_reflect',
                                 'power1mod_fwd', 'power1mod_reflect',
                                 'power2mod_fwd', 'power2mod_reflect'],
                        'func': _icmodadd, 'kargs': {'axis': 0},
                        'pos': True, 'units': 'W'}},

          'ece':
          {'rhotn_sign': {'lstr': ['rhotn', 'theta'], 'func': _rhosign,
                          'units': None}},

          'bremsstrahlung_visible':
          {'lamb': {'lstr': ['lamb_up', 'lamb_lo'], 'func': _lamb,
                    'dim': 'distance',
                    'quantity': 'wavelength',
                    'units': 'm'}}
          }

# Complete los
_lstr = ['los_pt1R', 'los_pt1Z', 'los_pt1Phi',
         'los_pt2R', 'los_pt2Z', 'los_pt2Phi']
for ids in _lidslos:
    _dcomp[ids] = _dcomp.get(ids, {})
    if ids == 'spectrometer_x_ray_crystal':
        _dcomp[ids]['los_ptsRZPhi'] = {'lstr': _lstr, 'func': _losptsRZP2}
    else:
        _dcomp[ids]['los_ptsRZPhi'] = {'lstr': _lstr, 'func': _losptsRZP}


# Uniformize
_lids = set(_dshort.keys()).union(_dcomp.keys())
for ids in _lids:
    _dshort[ids] = _dshort.get(ids, {})
    _dcomp[ids] = _dcomp.get(ids, {})


# ############################################################################
#
#           Dict of signals that, by default are not loaded
#         (because replaced by more complete computed signals)
#                         (avoids redundancy)
#
# ############################################################################


# All except (for when sig not specified in get_data())
_dall_except = {}
for ids in _lidslos:
    _dall_except[ids] = _lstr
_dall_except['equilibrium'] = ['axR', 'axZ', 'sepR', 'sepZ',
                               '2dBT', '2dBR', '2dBZ',
                               'x0R', 'x0Z', 'x1R', 'x1Z',
                               'strike0R', 'strike0Z', 'strike1R', 'strike1Z']
_dall_except['magnetics'] = ['bpol_R', 'bpol_Z', 'floop_R', 'floop_Z']
_dall_except['ic_antennas'] = ['power0mod_launched', 'power0mod_reflected',
                               'power1mod_launched', 'power1mod_reflected',
                               'power2mod_launched', 'power2mod_reflected']


# ############################################################################
#
#           Dict of default pre-defined sets of signals
#
# ############################################################################


_dpreset = {
            'overview':
            {'wall': None,
             'pulse_schedule': None,
             'equilibrium': None},

            'plasma2d':
            {'wall': ['domainR', 'domainZ'],
             'equilibrium': ['t', 'ax', 'sep'],
             'core_profiles': ['t', '1dTe', '1dne', '1dzeff',
                               '1drhotn', '1dphi'],
             'core_sources': ['t', '1dprad'],
             'edge_profiles': ['t'],
             'edge_sources': ['t']},

            'ece':
            {'wall': ['domainR', 'domainZ'],
             'ece': None,
             'core_profiles': ['t', 'Te', 'ne']}
           }


# ############################################################################
#
#       List of ids considered as basis and automatically loaded by default
#
# ############################################################################


_IDS_BASE = ['wall', 'pulse_schedule']


# ############################################################################
#
#       Default parameters for exporting to tofu objects (_comp_toobjects.py)
#
# ############################################################################

_INDEVENT = 0
_DTLIM = {'west': ['IGNITRON', 'PLUSDIP']}
