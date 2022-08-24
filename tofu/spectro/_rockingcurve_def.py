
import numpy as np
import scipy.interpolate


# #############################################################################
# #############################################################################
#                                   _DCRYST
# #############################################################################
# #############################################################################

_DCRYST = {
    'Quartz_110': {
        'name': 'Quartz_110',
        'symbol': 'Qz110',
        'target_ion': 'Ar16+',
        'target_lamb': 3.96e-10,
        'atoms': ['Si', 'O'],
        'atoms_Z': [14., 8.],
        'atoms_nb': [3., 6.],
        'miller': np.r_[1., 1., 0.],
        'volume': None,
        'd_hkl': None,
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
                'a0': 4.91304,
                'c0': 5.40463,
            },
            'unit': 'A',
            'comments': 'within the unit cell',
            'Tref': {
                'data': 25.,
                'unit': 'C',
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': '1/C',
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin_theta_lambda': {
            'Si': np.r_[
                0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
            ],
            'O': np.r_[
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
            ],
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
        'target_ion': 'Ar17+',
        'target_lamb': 3.75e-10,
        'atoms': ['Si', 'O'],
        'atoms_Z': [14., 8.],
        'atoms_nb': [3., 6.],
        'miller': np.r_[1., 0., 2.],
        'volume': None,
        'd_hkl': None,
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
                'a0': 4.91304,
                'c0': 5.40463,
            },
            'unit': 'A',
            'comments': 'within the unit cell',
            'Tref': {
                'data': 25.,
                'unit': 'C',
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': '1/C',
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin_theta_lambda': {
            'values': {
                'Si': np.r_[
                    0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                    0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
                ],
                'O': np.r_[
                    0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
                ],
            },
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
        'target': None,
        'atoms': None,
        'atoms_Z': None,
        'atoms_nb': None,
        'miller': None,
        'volume': None,
        'd_hkl': None,
        'mesh': {
            'type': None,
            'positions': None,
            'sources': None,
        },
        'phases': None,
        'inter_atomic': {
            'distances': None,
            'unit': None,
            'comments': None,
            'Tref': None,
            'sources': None,
        },
        'thermal_expansion': {
            'coefs': None,
            'unit': None,
            'comments': None,
            'sources': None,
        },
        'sin_theta_lambda': {
            'values': None,
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

# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: Quartz_110 and Quartz_102
# ---------------------------------------------------------------

# Position of the 3 Si atoms in the unit cell
# -------------------------------------------

# Quartz_110
uSi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['u'][0]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['x'] = np.r_[
    -uSi,
    uSi,
    0.
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['y'] = np.r_[
    -uSi,
    0.,
    uSi
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['z'] = np.r_[
    1./3.,
    0.,
    2./3.
]
_DCRYST['Quartz_110']['mesh']['positions']['Si']['N'] = np.size(
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['x']
)

# Quartz_102
_DCRYST['Quartz_102']['mesh']['positions']['Si']['x'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['x']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['y'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['y']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['z'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['z']
)
_DCRYST['Quartz_102']['mesh']['positions']['Si']['N'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['Si']['N']
)

# Position of the 6 O atoms in the unit cell
# ------------------------------------------

# Quartz_110
uOx = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][0]
uOy = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][1]
uOz = _DCRYST['Quartz_110']['mesh']['positions']['O']['u'][2]
_DCRYST['Quartz_110']['mesh']['positions']['O']['x'] = np.r_[
    uOx,
    uOy - uOx,
    -uOy,
    uOx - uOy,
    uOy,
    -uOx
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['y'] = np.r_[
    uOy,
    -uOx,
    uOx - uOy,
    -uOy,
    uOx,
    uOy - uOx
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['z'] = np.r_[
    uOz,
    uOz + 1./3.,
    uOz + 2./3.,
    -uOz,
    2./3. - uOz,
    1./3. - uOz
]
_DCRYST['Quartz_110']['mesh']['positions']['O']['N'] = np.size(
    _DCRYST['Quartz_110']['mesh']['positions']['O']['x']
)

# Quartz_102
_DCRYST['Quartz_102']['mesh']['positions']['O']['x'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['x']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['y'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['y']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['z'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['z']
)
_DCRYST['Quartz_102']['mesh']['positions']['O']['N'] = (
    _DCRYST['Quartz_110']['mesh']['positions']['O']['N']
)

# ---------------------------------
# Attribution to Germanium crystals
# ---------------------------------

# #############################################################################
# #############################################################################
#                         Elementary box volume and
#                       inter-reticular spacing d_hkl
# #############################################################################
# #############################################################################

# -----------------------------------------------------------------------
# Definition of volume and inter-reticular spacing relations, func(meshtype)
# -----------------------------------------------------------------------


def hexa_volume(aa, cc):
    return (aa**2) * cc * (np.sqrt(3.)/2.)


def hexa_spacing(hh, kk, ll, aa, cc):
    return np.sqrt(
        (3.*(aa**2)*(cc**2))
        / (4.*(hh**2 + kk**2 + hh*kk)*(cc**2) + 3.*(ll**2)*(aa**2))
    )


# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: Quartz_110 and Quartz_102
# ---------------------------------------------------------------


# Same values for 110- and Quartz_102
a = _DCRYST['Quartz_110']['inter_atomic']['distances']['a0']
c = _DCRYST['Quartz_110']['inter_atomic']['distances']['c0']

h110 = _DCRYST['Quartz_110']['miller'][0]
k110 = _DCRYST['Quartz_110']['miller'][1]
l110 = _DCRYST['Quartz_110']['miller'][2]
h102 = _DCRYST['Quartz_102']['miller'][0]
k102 = _DCRYST['Quartz_102']['miller'][1]
l102 = _DCRYST['Quartz_102']['miller'][2]

_DCRYST['Quartz_110']['volume'] = hexa_volume(a, c)
_DCRYST['Quartz_110']['d_hkl'] = hexa_spacing(h110, k110, l110, a, c) * 1.e-10
_DCRYST['Quartz_102']['volume'] = hexa_volume(a, c)
_DCRYST['Quartz_102']['d_hkl'] = hexa_spacing(h102, k102, l102, a, c) * 1e-10


# ---------------------------------
# Attribution to Germanium crystals
# ---------------------------------

# #############################################################################
# #############################################################################
#                               Structure factor
# #############################################################################
# #############################################################################

# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: Quartz_110 and Quartz_102
# ---------------------------------------------------------------
# From W. Zachariasen, Theory of X-ray Diffraction in Crystals
# (Wiley, New York, 1945)

# Linear absorption coefficient
# -----------------------------

# Same values for 110- and Quartz_102
Zsi = _DCRYST['Quartz_110']['atoms_Z'][0]
Zo = _DCRYST['Quartz_110']['atoms_Z'][1]


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


# Same values for 110- and Quartz_102
sol_si = _DCRYST['Quartz_110']['sin_theta_lambda']['Si']
sol_o = _DCRYST['Quartz_110']['sin_theta_lambda']['O']
asf_si = _DCRYST['Quartz_110']['atomic_scattering']['factors']['Si']
asf_o = _DCRYST['Quartz_110']['atomic_scattering']['factors']['O']
interp_si = scipy.interpolate.interp1d(sol_si, asf_si)
interp_o = scipy.interpolate.interp1d(sol_o, asf_o)


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


h110 = _DCRYST['Quartz_110']['miller'][0]
k110 = _DCRYST['Quartz_110']['miller'][1]
l110 = _DCRYST['Quartz_110']['miller'][2]
h102 = _DCRYST['Quartz_102']['miller'][0]
k102 = _DCRYST['Quartz_102']['miller'][1]
l102 = _DCRYST['Quartz_102']['miller'][2]


# Same values for 110- and Quartz_102
xsi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['x']
ysi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['y']
zsi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['z']
Nsi = _DCRYST['Quartz_110']['mesh']['positions']['Si']['N']
xo = _DCRYST['Quartz_110']['mesh']['positions']['O']['x']
yo = _DCRYST['Quartz_110']['mesh']['positions']['O']['y']
zo = _DCRYST['Quartz_110']['mesh']['positions']['O']['z']
No = _DCRYST['Quartz_110']['mesh']['positions']['O']['N']


def phasesi(hh, kk, ll, xsi, ysi, zsi):
    return hh*xsi + kk*ysi + ll*zsi


def phaseo(hh, kk, ll, xo, yo, zo):
    return hh*xo + kk*yo + ll*zo


phaseSi_110 = np.full((Nsi), np.nan)
phaseO_110 = np.full((No), np.nan)
phaseSi_102 = np.full((Nsi), np.nan)
phaseO_102 = np.full((No), np.nan)

for i in range(Nsi):
    phaseSi_110[i] = phasesi(h110, k110, l110, xsi[i], ysi[i], zsi[i])
    phaseSi_102[i] = phasesi(h102, k102, l102, xsi[i], ysi[i], zsi[i])
_DCRYST['Quartz_110']['phases']['Si'] = phaseSi_110
_DCRYST['Quartz_102']['phases']['Si'] = phaseSi_102

for i in range(No):
    phaseO_110[i] = phaseo(h110, k110, l110, xo[i], yo[i], zo[i])
    phaseO_102[i] = phaseo(h102, k102, l102, xo[i], yo[i], zo[i])
_DCRYST['Quartz_110']['phases']['O'] = phaseO_110
_DCRYST['Quartz_102']['phases']['O'] = phaseO_102
