
import numpy as np
import scipy.interpolate


# #############################################################################
# #############################################################################
#                                   _DCRYST
# #############################################################################
# #############################################################################

_DCRYST = {
    '110-Quartz': {
        'name': '110-Qz',
        'symbol': 'Qz',
        'target': 'ArXVII ion (3.96 A)',
        'atoms': ['Si', 'O'],
        'atomic number': [14., 8.],
        'number of atoms': [3., 6.],
        'Miller indices': np.r_[1., 1., 0.],
        'Volume': None,
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
        'Inter-atomic': {
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
        'Thermal expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': '1/C',
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin(theta)/lambda': {
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
        'atomic scattering': {
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
    '102-Quartz': {
        'name': '102-Qz',
        'symbol': 'Qz',
        'target': 'ArXVIII ion (3.75 A)',
        'atoms': ['Si', 'O'],
        'atomic number': [14., 8.],
        'number of atoms': [3., 6.],
        'Miller indices': np.r_[1., 0., 2.],
        'Volume': None,
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
        'Inter-atomic': {
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
        'Thermal expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': '1/C',
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin(theta)/lambda': {
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
        'atomic scattering': {
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
    'XXX-Germanium': {
        'name': None,
        'symbol': None,
        'target': None,
        'atoms': None,
        'atomic number': None,
        'number of atoms': None,
        'Miller indices': None,
        'Volume': None,
        'd_hkl': None,
        'mesh': {
            'type': None,
            'positions': None,
            'sources': None,
        },
        'phases': None,
        'Inter-atomic': {
            'distances': None,
            'unit': None,
            'comments': None,
            'Tref': None,
            'sources': None,
        },
        'Thermal expansion': {
            'coefs': None,
            'unit': None,
            'comments': None,
            'sources': None,
        },
        'sin(theta)/lambda': {
            'values': None,
            'sources': None,
        },
        'atomic scattering': {
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
# Attribution to alpha-Quartz crystals: 110-Quartz and 102-Quartz
# ---------------------------------------------------------------

# Position of the 3 Si atoms in the unit cell
# -------------------------------------------

# 110-Quartz
uSi = _DCRYST['110-Quartz']['mesh']['positions']['Si']['u'][0]
_DCRYST['110-Quartz']['mesh']['positions']['Si']['x'] = np.r_[
    -uSi,
    uSi,
    0.
]
_DCRYST['110-Quartz']['mesh']['positions']['Si']['y'] = np.r_[
    -uSi,
    0.,
    uSi
]
_DCRYST['110-Quartz']['mesh']['positions']['Si']['z'] = np.r_[
    1./3.,
    0.,
    2./3.
]
_DCRYST['110-Quartz']['mesh']['positions']['Si']['N'] = np.size(
    _DCRYST['110-Quartz']['mesh']['positions']['Si']['x']
)

# 102-Quartz
_DCRYST['102-Quartz']['mesh']['positions']['Si']['x'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['Si']['x']
)
_DCRYST['102-Quartz']['mesh']['positions']['Si']['y'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['Si']['y']
)
_DCRYST['102-Quartz']['mesh']['positions']['Si']['z'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['Si']['z']
)
_DCRYST['102-Quartz']['mesh']['positions']['Si']['N'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['Si']['N']
)

# Position of the 6 O atoms in the unit cell
# ------------------------------------------

# 110-Quartz
uOx = _DCRYST['110-Quartz']['mesh']['positions']['O']['u'][0]
uOy = _DCRYST['110-Quartz']['mesh']['positions']['O']['u'][1]
uOz = _DCRYST['110-Quartz']['mesh']['positions']['O']['u'][2]
_DCRYST['110-Quartz']['mesh']['positions']['O']['x'] = np.r_[
    uOx,
    uOy - uOx,
    -uOy,
    uOx - uOy,
    uOy,
    -uOx
]
_DCRYST['110-Quartz']['mesh']['positions']['O']['y'] = np.r_[
    uOy,
    -uOx,
    uOx - uOy,
    -uOy,
    uOx,
    uOy - uOx
]
_DCRYST['110-Quartz']['mesh']['positions']['O']['z'] = np.r_[
    uOz,
    uOz + 1./3.,
    uOz + 2./3.,
    -uOz,
    2./3. - uOz,
    1./3. - uOz
]
_DCRYST['110-Quartz']['mesh']['positions']['O']['N'] = np.size(
    _DCRYST['110-Quartz']['mesh']['positions']['O']['x']
)

# 102-Quartz
_DCRYST['102-Quartz']['mesh']['positions']['O']['x'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['O']['x']
)
_DCRYST['102-Quartz']['mesh']['positions']['O']['y'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['O']['y']
)
_DCRYST['102-Quartz']['mesh']['positions']['O']['z'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['O']['z']
)
_DCRYST['102-Quartz']['mesh']['positions']['O']['N'] = (
    _DCRYST['110-Quartz']['mesh']['positions']['O']['N']
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

def hexa_volume(a, c):
    return (a**2)*c*(np.sqrt(3.)/2.)

def hexa_spacing(h, k, l, a, c):
    return np.sqrt(
        (3.*(a**2)*(c**2))/(4.*(h**2 + k**2 + h*k)*(c**2) + 3.*(l**2)*(a**2))
    )

# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: 110-Quartz and 102-Quartz
# ---------------------------------------------------------------

# Same values for 110- and 102-Quartz
a = _DCRYST['110-Quartz']['Inter-atomic']['distances']['a0']
c = _DCRYST['110-Quartz']['Inter-atomic']['distances']['c0']

h110 = _DCRYST['110-Quartz']['Miller indices'][0]
k110 = _DCRYST['110-Quartz']['Miller indices'][1]
l110 = _DCRYST['110-Quartz']['Miller indices'][2]
h102 = _DCRYST['102-Quartz']['Miller indices'][0]
k102 = _DCRYST['102-Quartz']['Miller indices'][1]
l102 = _DCRYST['102-Quartz']['Miller indices'][2]

_DCRYST['110-Quartz']['Volume'] = hexa_volume(a=a, c=c)
_DCRYST['110-Quartz']['d_hkl'] = hexa_spacing(
    h=h110, k=k110, l=l110, a=a, c=c,
)
_DCRYST['102-Quartz']['Volume'] = hexa_volume(a=a, c=c)
_DCRYST['102-Quartz']['d_hkl'] = hexa_spacing(
    h=h102, k=k102, l=l102, a=a, c=c,
)

# ---------------------------------
# Attribution to Germanium crystals
# ---------------------------------

# #############################################################################
# #############################################################################
#                               Structure factor
# #############################################################################
# #############################################################################

# ---------------------------------------------------------------
# Attribution to alpha-Quartz crystals: 110-Quartz and 102-Quartz
# ---------------------------------------------------------------
# From W. Zachariasen, Theory of X-ray Diffraction in Crystals
# (Wiley, New York, 1945)

# Linear absorption coefficient
# -----------------------------

# Same values for 110- and 102-Quartz
Zsi = _DCRYST['110-Quartz']['atomic number'][0]
Zo = _DCRYST['110-Quartz']['atomic number'][1]

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

# Same values for 110- and 102-Quartz
sol_si = _DCRYST['110-Quartz']['sin(theta)/lambda']['Si']
sol_o = _DCRYST['110-Quartz']['sin(theta)/lambda']['O']
asf_si = _DCRYST['110-Quartz']['atomic scattering']['factors']['Si']
asf_o = _DCRYST['110-Quartz']['atomic scattering']['factors']['O']
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

h110 = _DCRYST['110-Quartz']['Miller indices'][0]
k110 = _DCRYST['110-Quartz']['Miller indices'][1]
l110 = _DCRYST['110-Quartz']['Miller indices'][2]
h102 = _DCRYST['102-Quartz']['Miller indices'][0]
k102 = _DCRYST['102-Quartz']['Miller indices'][1]
l102 = _DCRYST['102-Quartz']['Miller indices'][2]

# Same values for 110- and 102-Quartz
xsi = _DCRYST['110-Quartz']['mesh']['positions']['Si']['x']
ysi = _DCRYST['110-Quartz']['mesh']['positions']['Si']['y']
zsi = _DCRYST['110-Quartz']['mesh']['positions']['Si']['z']
Nsi = _DCRYST['110-Quartz']['mesh']['positions']['Si']['N']
xo = _DCRYST['110-Quartz']['mesh']['positions']['O']['x']
yo = _DCRYST['110-Quartz']['mesh']['positions']['O']['y']
zo = _DCRYST['110-Quartz']['mesh']['positions']['O']['z']
No = _DCRYST['110-Quartz']['mesh']['positions']['O']['N']

def phasesi(h, k, l, xsi, ysi, zsi):
    return h*xsi + k*ysi + l*zsi

def phaseo(h, k, l, xo, yo, zo):
    return h*xo + k*yo + l*zo

phaseSi_110 = np.full((Nsi), np.nan)
phaseO_110 = np.full((No), np.nan)
phaseSi_102 = np.full((Nsi), np.nan)
phaseO_102 = np.full((No), np.nan)

for i in range(Nsi):
    phaseSi_110[i] = phasesi(h110, k110, l110, xsi[i], ysi[i], zsi[i])
    phaseSi_102[i] = phasesi(h102, k102, l102, xsi[i], ysi[i], zsi[i])
_DCRYST['110-Quartz']['phases']['Si'] = phaseSi_110
_DCRYST['102-Quartz']['phases']['Si'] = phaseSi_102

for i in range(No):
    phaseO_110[i] = phaseo(h110, k110, l110, xo[i], yo[i], zo[i])
    phaseO_102[i] = phaseo(h102, k102, l102, xo[i], yo[i], zo[i])
_DCRYST['110-Quartz']['phases']['O'] = phaseO_110
_DCRYST['102-Quartz']['phases']['O'] = phaseO_102
