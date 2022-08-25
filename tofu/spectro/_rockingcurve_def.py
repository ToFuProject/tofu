
import numpy as np
import scipy.interpolate
import scipy.constants as scpct
import astropy.units as u

# #############################################################################
# #############################################################################
#                                   _DCRYST
# #############################################################################
# #############################################################################

_DCRYST = {
    'Quartz_110': {
        'name': 'Quartz_110',
        'symbol': 'Qz110',
        'target': {
            'ion': 'ArXVII',
            'wavelength': 3.96e-10,
            'unit': u.m,
        },
        'atoms': ['Si', 'O'],
        'atoms_Z': [14., 8.],
        'atoms_nb': [3., 6.],
        'miller': np.r_[1., 1., 0.],
        'volume': {
            'value': None,
            'unit': 1/u.m**3,
        },
        'd_hkl': {
            'value': None,
            'unit': u.m,
        },
        'mesh': {
            'type': 'hexagonal',
            'positions': {
                'Si': {
                    'u': np.r_[0.465],
                    'x': None,
                    'y': None,
                    'z': None,
                    'N': None,
                },
                'O': {
                    'u': np.r_[0.415, 0.272, 0.120],
                    'x': None,
                    'y': None,
                    'z': None,
                    'N': None,
                },
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures (1963)',
        },
        'phases': {
            'Si': None,
            'O': None,
        },
        'inter_atomic': {
            'distances': {
                'a0': 4.91304e-10,
                'c0': 5.40463e-10,
            },
            'unit': u.m,
            'comments': 'within the unit cell',
            'Tref': {
                'data': 25.,
                'unit': u.deg_C,
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': 1/u.deg_C,
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin_theta_lambda': {
            'values': {
                'Si': np.r_[
                    0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                    0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
                ]*1e10,
                'O': np.r_[
                    0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
                ]*1e10,
            },
            'unit': 1/u.m,
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985)',
        },
        'atomic_scattering': {
            'factors': {
                'Si': np.r_[
                    12., 11., 9.5, 8.8, 8.3, 7.7, 7.27, 6.25, 5.3,
                    4.45, 3.75, 3.15, 2.7, 2.35, 2.07, 1.87, 1.71, 1.6,
                ],
                'O': np.r_[
                    9., 7.836, 5.756, 4.068, 2.968, 2.313, 1.934, 1.710, 1.566,
                    1.462, 1.373, 1.294,
                ],
            },
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985)',
        },
    },
    'Quartz_102': {
        'name': 'Quartz_102',
        'symbol': 'Qz102',
        'target': {
            'ion': 'ArXVIII',
            'wavelength': 3.75e-10,
            'unit': u.m,
        },
        'atoms': ['Si', 'O'],
        'atoms_Z': [14., 8.],
        'atoms_nb': [3., 6.],
        'miller': np.r_[1., 0., 2.],
        'volume': {
            'value': None,
            'unit': 1/u.m**3,
        },
        'd_hkl': {
            'value': None,
            'unit': u.m,
        },
        'mesh': {
            'type': 'hexagonal',
            'positions': {
                'Si': {
                    'u': np.r_[0.465],
                    'x': None,
                    'y': None,
                    'z': None,
                    'N': None,
                },
                'O': {
                    'u': np.r_[0.415, 0.272, 0.120],
                    'x': None,
                    'y': None,
                    'z': None,
                    'N': None,
                },
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures (1963)',
        },
        'phases': {
            'Si': None,
            'O': None,
        },
        'inter_atomic': {
            'distances': {
                'a0': 4.91304e-10,
                'c0': 5.40463e-10,
            },
            'unit': u.m,
            'comments': 'within the unit cell',
            'Tref': {
                'data': 25.,
                'unit': u.deg_C,
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': 1/u.deg_C,
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin_theta_lambda': {
            'values': {
                'Si': np.r_[
                    0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                    0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
                ]*1e10,
                'O': np.r_[
                    0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
                ]*1e10,
            },
            'unit': 1/u.m,
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985)',
        },
        'atomic_scattering': {
            'factors': {
                'Si': np.r_[
                    12., 11., 9.5, 8.8, 8.3, 7.7, 7.27, 6.25, 5.3,
                    4.45, 3.75, 3.15, 2.7, 2.35, 2.07, 1.87, 1.71, 1.6,
                ],
                'O': np.r_[
                    9., 7.836, 5.756, 4.068, 2.968, 2.313, 1.934, 1.710, 1.566,
                    1.462, 1.373, 1.294,
                ],
            },
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985)',
        },
    },

    'Germanium_XXX': {
        'name': None,
        'symbol': None,
        'target': {
            'ion': None,
            'wavelength': None,
            'unit': u.m,
        },
        'atoms': None,
        'atoms_Z': None,
        'atoms_nb': None,
        'miller': None,
        'volume': {
            'value': None,
            'unit': 1/u.m**3,
        },
        'd_hkl': {
            'value': None,
            'unit': u.m,
        },
        'mesh': {
            'type': None,
            'positions': None,
            'sources': None,
        },
        'phases': None,
        'inter_atomic': {
            'distances': None,
            'unit': u.m,
            'comments': None,
            'Tref': {
                'data': None,
                'unit': u.deg_C,
            },
            'sources': None,
        },
        'thermal_expansion': {
            'coefs': None,
            'unit': 1/u.deg_C,
            'comments': None,
            'sources': None,
        },
        'sin_theta_lambda': {
            'values': None,
            'unit': 1/u.m,
            'sources': None,
        },
        'atomic_scattering': {
            'factors': None,
            'sources': None,
        },
    },
}

# #############################################################################
# #############################################################################
#                         Atoms positions in mesh
# #############################################################################
# #############################################################################

# -------------------------
# Positions from literature
# -------------------------

# Si and O positions for alpha-Quartz crystal
# From R.W.G. Wyckoff, Crystal Structures (1963)
# xsi = np.r_[-u, u, 0.]
# ysi = np.r_[-u, 0., u]
# zsi = np.r_[1./3., 0., 2./3.]
# xo = np.r_[x, y - x, -y, x - y, y, -x]
# yo = np.r_[y, -x, x - y, -y, x, y - x]
# zo = np.r_[z, z + 1./3., z + 2./3., -z, 2./3. - z, 1./3. - z]

# Atoms positions for Germanium crystal

# -----------
# Attribution
# -----------

for ii in _DCRYST.keys():
    if "Quartz" in ii:
        uSi = _DCRYST[ii]['mesh']['positions']['Si']['u'][0]
        uOx = _DCRYST[ii]['mesh']['positions']['O']['u'][0]
        uOy = _DCRYST[ii]['mesh']['positions']['O']['u'][1]
        uOz = _DCRYST[ii]['mesh']['positions']['O']['u'][2]
        _DCRYST[ii]['mesh']['positions']['Si']['x'] = np.r_[
            -uSi,
            uSi,
            0.
        ]
        _DCRYST[ii]['mesh']['positions']['Si']['y'] = np.r_[
            -uSi,
            0.,
            uSi
        ]
        _DCRYST[ii]['mesh']['positions']['Si']['z'] = np.r_[
            1./3.,
            0.,
            2./3.
        ]
        _DCRYST[ii]['mesh']['positions']['Si']['N'] = np.size(
            _DCRYST[ii]['mesh']['positions']['Si']['x']
        )
        _DCRYST[ii]['mesh']['positions']['O']['x'] = np.r_[
            uOx,
            uOy - uOx,
            -uOy,
            uOx - uOy,
            uOy,
            -uOx
        ]
        _DCRYST[ii]['mesh']['positions']['O']['y'] = np.r_[
            uOy,
            -uOx,
            uOx - uOy,
            -uOy,
            uOx,
            uOy - uOx
        ]
        _DCRYST[ii]['mesh']['positions']['O']['z'] = np.r_[
            uOz,
            uOz + 1./3.,
            uOz + 2./3.,
            -uOz,
            2./3. - uOz,
            1./3. - uOz
        ]
        _DCRYST[ii]['mesh']['positions']['O']['N'] = np.size(
            _DCRYST[ii]['mesh']['positions']['O']['x']
        )
    elif "Ge" in ii:
        None

# #############################################################################
# #############################################################################
#                         Elementary box volume and
#                       inter-reticular spacing d_hkl
# #############################################################################
# #############################################################################

# --------------------------------------------------------------------------
# Definition of volume and inter-reticular spacing, function of the meshtype
# --------------------------------------------------------------------------

def hexa_volume(aa, cc):
    return (aa**2) * cc * (np.sqrt(3.)/2.)

def hexa_spacing(hh, kk, ll, aa, cc):
    return (3.*aa**2*cc**2)**(1/2) / (
        4.*cc**2*(hh**2 + kk**2 + hh*kk) + 3.*ll**2*aa**2
    )**(1/2)

# -----------
# Attribution
# -----------

for ii in _DCRYST.keys():
    if "Quartz" in ii and "hexagonal" in _DCRYST[ii]['mesh']['type']:
        a = _DCRYST[ii]['inter_atomic']['distances']['a0']  # m
        c = _DCRYST[ii]['inter_atomic']['distances']['c0']  # m
        h = _DCRYST[ii]['miller'][0]
        k = _DCRYST[ii]['miller'][1]
        l = _DCRYST[ii]['miller'][2]
        _DCRYST[ii]['volume']['value'] = hexa_volume(a, c)
        _DCRYST[ii]['d_hkl']['value'] = hexa_spacing(h, k, l, a, c)
    elif "Ge" in ii:
        None


# #############################################################################
# #############################################################################
#                               Structure factor
# #############################################################################
# #############################################################################

# -----------
# Attribution
# -----------
# From W. Zachariasen, Theory of X-ray Diffraction in Crystals
# (Wiley, New York, 1945)

# Linear absorption coefficient
# -----------------------------

for ii in _DCRYST.keys():
    if "Quartz" in ii:
        Zsi, Zo = _DCRYST[ii]['atoms_Z'][0], _DCRYST[ii]['atoms_Z'][1]
        # For wavelength in Angstroms (ex: lamb=3.96A)
        def mu_si(lamb):
            return 1.38e-2*(lamb**2.79)*(Zsi**2.73)
        def mu_si1(lamb):
            return 5.33e-4*(lamb**2.74)*(Zsi**3.03)
        def mu_o(lamb):
            return 5.4e-3*(lamb**2.92)*(Zo**3.07)
        def mu(lamb, mu_si, mu_o):
            return 2.65e-8*(7.*mu_si + 8.*mu_o)/15.

# Atomic scattering factor, real and imaginary parts
# --------------------------------------------------

for ii in _DCRYST.keys():
    if "Quartz" in ii:
        sol_si = _DCRYST[ii]['sin_theta_lambda']['values']['Si']
        sol_o = _DCRYST[ii]['sin_theta_lambda']['values']['O']
        asf_si = _DCRYST[ii]['atomic_scattering']['factors']['Si']
        asf_o = _DCRYST[ii]['atomic_scattering']['factors']['O']
        interp_si = scipy.interpolate.interp1d(sol_si, asf_si)
        interp_o = scipy.interpolate.interp1d(sol_o, asf_o)

        # For wavelength in Angstroms (ex: lamb=3.96A)
        def dfsi_re(lamb):
            return 0.1335*lamb - 6e-3
        def fsi_re(lamb, sol):
            return interp_si(sol) + dfsi_re(lamb)
        def fsi_im(lamb, mu_si):
            return 5.936e-4*Zsi*(mu_si/lamb)
        def dfo_re(lamb):
            return 0.1335*lamb - 0.206
        def fo_re(lamb, sol):
            return interp_o(sol) + dfo_re(lamb)
        def fo_im(lamb, mu_o):
            return 5.936e-4*Zo*(mu_o/lamb)

# Phases
# ------

def phasesi(hh, kk, ll, xsi, ysi, zsi):
    return hh*xsi + kk*ysi + ll*zsi
def phaseo(hh, kk, ll, xo, yo, zo):
    return hh*xo + kk*yo + ll*zo


for ii in _DCRYST.keys():
    if "Quartz" in ii:
        h = _DCRYST[ii]['miller'][0]
        k = _DCRYST[ii]['miller'][1]
        l = _DCRYST[ii]['miller'][2]
        xsi = _DCRYST[ii]['mesh']['positions']['Si']['x']
        ysi = _DCRYST[ii]['mesh']['positions']['Si']['y']
        zsi = _DCRYST[ii]['mesh']['positions']['Si']['z']
        Nsi = _DCRYST[ii]['mesh']['positions']['Si']['N']
        xo = _DCRYST[ii]['mesh']['positions']['O']['x']
        yo = _DCRYST[ii]['mesh']['positions']['O']['y']
        zo = _DCRYST[ii]['mesh']['positions']['O']['z']
        No = _DCRYST[ii]['mesh']['positions']['O']['N']

        phaseSi = np.full((Nsi), np.nan)
        phaseO = np.full((No), np.nan)

        if "110" in ii:
            for i in range(Nsi):
                phaseSi[i] = phasesi(h, k, l, xsi[i], ysi[i], zsi[i])
            for i in range(No):
                phaseO[i] = phaseo(h, k, l, xo[i], yo[i], zo[i])
            _DCRYST[ii]['phases']['Si'] = phaseSi
            _DCRYST[ii]['phases']['O'] = phaseO
        elif "102" in ii:
            for i in range(Nsi):
                phaseSi[i] = phasesi(h, k, l, xsi[i], ysi[i], zsi[i])
            for i in range(No):
                phaseO[i] = phaseo(h, k, l, xo[i], yo[i], zo[i])
            _DCRYST[ii]['phases']['Si'] = phaseSi
            _DCRYST[ii]['phases']['O'] = phaseO
