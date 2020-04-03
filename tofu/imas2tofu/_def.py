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

# ############################################################################
#
#           shortcuts for imas2tofu interface (MultiIDSLoader class)
#
# ############################################################################

_dshort = {
    'wall': {
        'wallR': {'str': 'description_2d[0].limiter.unit[0].outline.r'},
        'wallZ': {'str': 'description_2d[0].limiter.unit[0].outline.z'}},

    'pulse_schedule': {
        'events_times': {'str': 'event[].time_stamp'},
        'events_names': {'str': 'event[].identifier'}},

    'equilibrium': {
        't': {'str': 'time'},
        'ip': {'str': 'time_slice[time].global_quantities.ip',
               'dim': 'current', 'quant': 'Ip', 'units': 'A'},
        'q0': {'str': 'time_slice[time].global_quantities.q_axis'},
        'qmin': {'str': 'time_slice[time].global_quantities.q_min.value'},
        'q95': {'str': 'time_slice[time].global_quantities.q_95'},
        'volume': {'str': 'time_slice[time].global_quantities.volume',
                   'dim': 'volume', 'quant': 'pvol', 'units': 'm3'},
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
        'x0R': {'str': 'time_slice[time].boundary.x_point[0].r'},
        'x0Z': {'str': 'time_slice[time].boundary.x_point[0].z'},
        'x1R': {'str': 'time_slice[time].boundary.x_point[1].r'},
        'x1Z': {'str': 'time_slice[time].boundary.x_point[1].z'},
        'strike0R': {'str': 'time_slice[time].boundary.strike_point[0].r'},
        'strike0Z': {'str': 'time_slice[time].boundary.strike_point[0].z'},
        'strike1R': {'str': 'time_slice[time].boundary.strike_point[1].r'},
        'strike1Z': {'str': 'time_slice[time].boundary.strike_point[1].z'},
        'sepR': {'str': 'time_slice[time].boundary_separatrix.outline.r'},
        'sepZ': {'str': 'time_slice[time].boundary_separatrix.outline.z'},

        '1drhotn': {'str': 'time_slice[time].profiles_1d.rho_tor_norm',
                    'dim': 'rho', 'quant': 'rhotn', 'units': 'adim.'},
        '1dphi': {'str': 'time_slice[time].profiles_1d.phi',
                  'dim': 'B flux', 'quant': 'phi', 'units': 'Wb'},
        '1dpsi': {'str': 'time_slice[time].profiles_1d.psi',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1dq': {'str': 'time_slice[time].profiles_1d.q',
                'dim': 'safety factor', 'quant': 'q', 'units': 'adim.'},
        '1dpe': {'str': 'time_slice[time].profiles_1d.pressure',
                 'dim': 'pressure', 'quant': 'pe', 'units': 'Pa'},
        '1djT': {'str': 'time_slice[time].profiles_1d.j_tor',
                 'dim': 'vol. current dens.', 'quant': 'jT', 'units': 'A/m^2'},

        '2dphi': {'str': 'time_slice[time].ggd[0].phi[0].values',
                  'dim': 'B flux', 'quant': 'phi', 'units': 'Wb'},
        '2dpsi': {'str': 'time_slice[time].ggd[0].psi[0].values',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '2djT': {'str': 'time_slice[time].ggd[0].j_tor[0].values',
                 'dim': 'vol. current dens.', 'quant': 'jT', 'units': 'A/m^2'},
        '2dBR': {'str': 'time_slice[time].ggd[0].b_field_r[0].values',
                 'dim': 'B', 'quant': 'BR', 'units': 'T'},
        '2dBT': {'str': 'time_slice[time].ggd[0].b_field_tor[0].values',
                 'dim': 'B', 'quant': 'BT', 'units': 'T'},
        '2dBZ': {'str': 'time_slice[time].ggd[0].b_field_z[0].values',
                 'dim': 'B', 'quant': 'BZ', 'units': 'T'},
        '2dmeshNodes': {'str': ('grids_ggd[0].grid[0].space[0]'
                                + '.objects_per_dimension[0]'
                                + '.object[].geometry')},
        '2dmeshFaces': {'str': ('grids_ggd[0].grid[0].space[0]'
                                + '.objects_per_dimension[2]'
                                + '.object[].nodes')},
        '2dmeshR': {'str': 'time_slice[0].profiles_2d[0].r'},
        '2dmeshZ': {'str': 'time_slice[0].profiles_2d[0].z'}},

    'core_profiles': {
        't': {'str': 'time'},
        'ip': {'str': 'global_quantities.ip',
               'dim': 'current', 'quant': 'Ip', 'units': 'A'},
        'vloop': {'str': 'global_quantities.v_loop',
                  'dim': 'voltage', 'quant': 'Vloop', 'units': 'V/m'},

        '1dTe': {'str': 'profiles_1d[time].electrons.temperature',
                 'dim': 'temperature',  'quant': 'Te', 'units': 'eV'},
        '1dne': {'str': 'profiles_1d[time].electrons.density',
                 'dim': 'density', 'quant': 'ne', 'units': '/m^3'},
        '1dzeff': {'str': 'profiles_1d[time].zeff',
                   'dim': 'charge', 'quant': 'zeff', 'units': 'adim.'},
        '1dphi': {'str': 'profiles_1d[time].grid.phi',
                  'dim': 'B flux', 'quant': 'phi', 'units': 'Wb'},
        '1dpsi': {'str': 'profiles_1d[time].grid.psi',
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1drhotn': {'str': 'profiles_1d[time].grid.rho_tor_norm',
                    'dim': 'rho', 'quant': 'rhotn', 'units': 'adim.'},
        '1drhopn': {'str': 'profiles_1d[time].grid.rho_pol_norm',
                    'dim': 'rho', 'quant': 'rhopn', 'units': 'adim.'},
        '1dnW': {'str': 'profiles_1d[time].ions[identifier.label=W].density',
                 'dim': 'density', 'quant': 'nI', 'units': '/m^3'}},

    'edge_profiles': {
        't': {'str': 'time'}},

    'core_sources': {
        't': {'str': 'time'},
        '1dpsi': {'str': ('source[identifier.name=lineradiation]'
                          + '.profiles_1d[time].grid.psi'),
                  'dim': 'B flux', 'quant': 'psi', 'units': 'Wb'},
        '1drhotn': {'str': ('source[identifier.name=lineradiation]'
                            + '.profiles_1d[time].grid.rho_tor_norm'),
                    'dim': 'rho', 'quant': 'rhotn', 'units': 'Wb'},
        '1dbrem': {'str': ('source[identifier.name=bremsstrahlung]'
                           + '.profiles_1d[time].electrons.energy'),
                   'dim': 'vol.emis.', 'quant': 'brem.', 'units': 'W/m^3'},
        '1dline': {'str': ('source[identifier.name=lineradiation]'
                           + '.profiles_1d[time].electrons.energy'),
                   'dim': 'vol. emis.', 'quant': 'lines', 'units': 'W/m^3'}},

    'edge_sources': {
        't': {'str': 'time'},
        '2dmeshNodes': {'str': ('grid_ggd[0].space[0].objects_per_dimension[0]'
                                + '.object[].geometry')},
        '2dmeshFaces': {'str': ('grid_ggd[0].space[0].objects_per_dimension[2]'
                                + '.object[].nodes')},
        '2dradiation': {'str': 'source[13].ggd[0].electrons.energy[0].values',
                        'dim': 'vol. emis.', 'quant': 'vol.emis.',
                        'name': 'tot. vol. emis.', 'units': 'W/m^3'}},

    'lh_antennas': {
        't': {'str': 'antenna[chan].power_launched.time'},
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
        't': {'str': 'antenna[chan].module[0].power_forward.time'},
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
        't': {'str': 'time'},
        'ip': {'str': 'method[0].ip.data'},
        'diamagflux': {'str': 'method[0].diamagnetic_flux.data'},
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
        't': {'str': 'gauge[chan].pressure.time'},
        'names': {'str': 'gauge[chan].name'},
        'p': {'str': 'gauge[chan].pressure.data',
              'dim': 'pressure', 'quant': 'p', 'units': 'Pa?'}},

    'calorimetry': {
        't': {'str': 'group[chan].component[0].power.time'},
        'names': {'str': 'group[chan].name'},
        'power': {'str': 'group[chan].component[0].power.data',
                  'dim': 'power', 'quant': 'extracted power',
                  'units': 'W'}},

    'neutron_diagnostic': {
        't': {'str': 'time', 'units': 's'},
        'flux_total': {'str': 'synthetic_signals.total_neutron_flux',
                       'dim': 'particle flux', 'quant': 'particle flux',
                       'units': 'Hz'}},

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
                  'dim': 'rho', 'quant': 'rhotn', 'units': 'adim.'},
        'theta': {'str': 'channel[chan].position.theta',
                  'dim': 'angle', 'quant': 'theta', 'units': 'rad.'},
        'tau1keV': {'str': 'channel[chan].optical_depth.data',
                    'dim': 'optical_depth', 'quant': 'tau', 'units': 'adim.'},
        'validity_timed': {'str': 'channel[chan].t_e.validity_timed'},
        'names': {'str': 'channel[chan].name'},
        'Te0': {'str': 't_e_central.data',
                'dim': 'temperature', 'quant': 'Te', 'units': 'eV'}},

    'reflectometer_profile': {
        't': {'str': 'time'},
        'ne': {'str': 'channel[chan].n_e.data',
               'dim': 'density', 'quant': 'ne', 'units': '/m^3'},
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
                     'units': '/m2', 'Brightness': True}},

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
                    'units': 'm2.sr'},
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
                       'units': 'W/(m2.sr)', 'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'etendue': {'str': 'channel[chan].etendue',
                    'dim': 'etendue', 'quant': 'etendue',
                    'units': 'm2.sr'}},

    'spectrometer_visible': {
        't': {'str': ('channel[chan].grating_spectrometer'
                      + '.radiance_spectral.time'),
              'quant': 't', 'units': 's'},
        'spectra': {'str': ('channel[chan].grating_spectrometer'
                            + '.radiance_spectral.data'),
                    'dim': 'radiance_spectral',
                    'quant': 'radiance_spectral',
                    'units': 'ph/s/(m2.sr)/m', 'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'lamb': {'str': 'channel[chan].grating_spectrometer.wavelengths',
                 'dim': 'wavelength', 'quant': 'wavelength', 'units': 'm'}},

    'bremsstrahlung_visible': {
        't': {'str': 'time',
              'quant': 't', 'units': 's'},
        'radiance': {'str': 'channel[chan].radiance_spectral.data',
                     'dim': 'radiance_spectral',
                     'quant': 'radiance_spectral',
                     'units': 'ph/s/(m2.sr)/m',
                     'Brightness': True},
        'names': {'str': 'channel[chan].name'},
        'lamb_up': {'str': 'channel[chan].filter.wavelength_upper'},
        'lamb_lo': {'str': 'channel[chan].filter.wavelength_lower'}}
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
                    'data': 'Te'}},
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
                               'Brightness': True}},
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
                              'Brightness': True}},
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
                            'Brightness': False}},
    'soft_x_rays': {'datacls': 'DataCam1D',
                    'geomcls': 'CamLOS1D',
                    'sig': {'t': 't',
                            'data': 'power'}},
    'spectrometer_visible': {'datacls': 'DataCam1DSpectral',
                             'geomcls': 'CamLOS1D',
                             'sig': {'data': 'spectra',
                                     't': 't',
                                     'lamb': 'lamb'}},
    'bremsstrahlung_visible': {
        'datacls': 'DataCam1D',
        'geomcls': 'CamLOS1D',
        'sig': {'t': 't',
                'data': 'radiance'},
        'synth': {
            'dsynth': {
                'quant': ['core_profiles.1dTe',
                          'core_profiles.1dne',
                          'core_profiles.1dzeff'],
                'ref1d': 'core_profiles.1drhotn',
                'ref2d': 'equilibrium.2drhotn'},
            'dsig': {'core_profiles': ['t'],
                     'equilibrium': ['t']},
            'Brightness': True}}}
