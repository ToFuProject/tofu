

import copy


import numpy as np
import scipy.interpolate


# ################################################################
# ################################################################
#                     _DCRYST_MAT
# ################################################################
# ################################################################


_DCRYST_MAT = {

    # ------
    # Quartz

    'Quartz': {
        'material_symbol': 'Qz',
        'atoms': ['Si', 'O'],
        'atoms_Z': [14., 8.],
        'atoms_nb': [3., 6.],
        'volume': None,
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
        'inter_atomic': {
            'distances': {
                'a0': 4.91304, # e-10,
                'c0': 5.40463, # e-10,
            },
            'unit': 'A',
            'comments': 'within the unit cell',
            'Tref': {
                'data': 25. + 273.15,
                'unit': 'K',
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'phases': { # Will be populated
            'Si': None,
            'O': None,
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 1.337e-5,
                'alpha_c': 7.97e-6,
            },
            'unit': '1/K',
            'comments': 'in parallel directions to a0 and c0',
            'sources': 'R.W.G. Wyckoff, Crystal Structures',
        },
        'sin_theta_lambda': {
            'Si': np.r_[
                0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
            ], # *1e10,
            'O': np.r_[
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
            ], # *1e10,
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

    # ----------
    # Germanium

    'Germanium': {
        'material_symbol': 'Ge',
        'atoms': ['Ge'],
        'atoms_Z': [32.],
        'atoms_nb': [8.],
        'volume': None,
        'mesh': {
            'type': 'diamond',
            'positions': {
                'Ge': {
                    'u': np.r_[0.25],
                    'x': None,
                    'y': None,
                    'z': None,
                    'N': None,
                },
            },
            'sources':  'R.W.G. Wyckoff, Crystal Structures (1963), Eqn 8a',
        },
        'inter_atomic': {
            'distances': {
                'a0': 5.65735, # e-10
            },
            'unit': 'A',
            'comments': None,
            'Tref': {
                'data': 20. + 273.15,
                'unit': 'K',
            },
            'sources': 'R.W.G. Wyckoff, Crystal Structures (1963), Table II.5',
        },
        'phases': { # Will be populated
            'Ge': None,
        },
        'thermal_expansion': {
            'coefs': {
                'alpha_a': 6.12e-6
            },
            'unit': '1/K',
            'comments': 'only one unique direction',
            'sources': 'H.P. Singh, Acta Cryst. (1968). A24, 469',
        },
        'sin_theta_lambda': {
            'Ge': np.r_[
                0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2,
                1.3, 1.4, 1.5
            ], # [1/AA],
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985), Table 3.3.1A',
        },
        'atomic_scattering': {
            'factors': {
                'Ge': np.r_[
                    32.0, 31.28, 29.52, 27.48, 25.53, 23.76,
                    22.11, 20.54, 19.02, 16.19, 13.72, 11.68,
                    10.08, 8.87, 7.96, 7.29, 6.77, 6.37,
                    6.02, 5.72
                ]
            },
            'sources':
                'Int. Tab. X-Ray Crystallography, Vol.I,II,III,IV (1985), Table 3.3.1A',
        },
    },
}


# ##############################################################
# ##############################################################
#           Complement materials dict
# ##############################################################


def _complement_dict_mat(dcryst_mat=None):
    """ Add what is missing to material dict """

    for k0, v0 in dcryst_mat.items():

        # ---------
        # positions
        # ---------

        # Quartz
        if k0 == 'Quartz':
            _positions_quartz(
                dcryst_mat=dcryst_mat,
                k0=k0,
                v0=v0,
            )

        # Germanium
        elif k0 == 'Germanium':
            _positions_germanium(
                dcryst_mat=dcryst_mat,
                k0=k0,
                v0=v0,
            )

        # other
        else:
            msg = "Unknown material"
            raise NotImplementedError(msg)

        # ------
        # volume
        # ------

        # hexagonal
        if v0['mesh']['type'] == 'hexagonal':
            dcryst_mat[k0]['volume'] = hexa_volume(
                v0['inter_atomic']['distances']['a0'],
                v0['inter_atomic']['distances']['c0'],
            )

        # diamond
        elif v0['mesh']['type'] == 'diamond':
            dcryst_mat[k0]['volume'] = diam_volume(
                v0['inter_atomic']['distances']['a0']
            )

        # other
        else:
            msg = f"Mesh volume not implemented for '{k0}'"
            raise NotImplementedError(msg)

        # -----------------------------------------------------
        # atomic absorption coefficients and scattering factor
        # -----------------------------------------------------

        if k0 == 'Quartz':
            _atomic_coefs_factor_Quartz(
                dcryst_mat=dcryst_mat,
                k0=k0,
                v0=v0,
            )

        elif k0 == 'Germanium':
            _atomic_coefs_factor_Germanium(
                dcryst_mat=dcryst_mat,
                k0=k0,
                v0=v0,
            )

        # other
        else:
            msg = "Unknown material"
            raise NotImplementedError(msg)


# ###############################################################
# ###############################################################
#                   Positions
# ##############################################################
# ###############################################################


def _positions_quartz(
    dcryst_mat=None,
    k0=None,
    v0=None,
):

    # Si and O positions for alpha-Quartz crystal
    # From R.W.G. Wyckoff, Crystal Structures (1963)
    # xsi = np.r_[-u, u, 0.]
    # ysi = np.r_[-u, 0., u]
    # zsi = np.r_[1./3., 0., 2./3.]
    # xo = np.r_[x, y - x, -y, x - y, y, -x]
    # yo = np.r_[y, -x, x - y, -y, x, y - x]
    # zo = np.r_[z, z + 1./3., z + 2./3., -z, 2./3. - z, 1./3. - z]

    # Quartz
    uSi = v0['mesh']['positions']['Si']['u'][0]
    dcryst_mat[k0]['mesh']['positions']['Si']['x'] = np.r_[
        -uSi,
        uSi,
        0.
    ]
    dcryst_mat[k0]['mesh']['positions']['Si']['y'] = np.r_[
        -uSi,
        0.,
        uSi
    ]
    dcryst_mat[k0]['mesh']['positions']['Si']['z'] = np.r_[
        1./3.,
        0.,
        2./3.
    ]
    dcryst_mat[k0]['mesh']['positions']['Si']['N'] = np.size(
        dcryst_mat[k0]['mesh']['positions']['Si']['x']
    )

    # Oxygen
    uOx = v0['mesh']['positions']['O']['u'][0]
    uOy = v0['mesh']['positions']['O']['u'][1]
    uOz = v0['mesh']['positions']['O']['u'][2]
    dcryst_mat[k0]['mesh']['positions']['O']['x'] = np.r_[
        uOx,
        uOy - uOx,
        -uOy,
        uOx - uOy,
        uOy,
        -uOx
    ]
    dcryst_mat[k0]['mesh']['positions']['O']['y'] = np.r_[
        uOy,
        -uOx,
        uOx - uOy,
        -uOy,
        uOx,
        uOy - uOx
    ]
    dcryst_mat[k0]['mesh']['positions']['O']['z'] = np.r_[
        uOz,
        uOz + 1./3.,
        uOz + 2./3.,
        -uOz,
        2./3. - uOz,
        1./3. - uOz
    ]
    dcryst_mat[k0]['mesh']['positions']['O']['N'] = np.size(
        dcryst_mat[k0]['mesh']['positions']['O']['x']
    )


def _positions_germanium(
    dcryst_mat=None,
    k0=None,
    v0=None,
):

    # Ge positions for Ge crystal
    # From R.W.G. Wyckoff, Crystal Structures (1963)

    # Germanium
    uGe = v0['mesh']['positions']['Ge']['u'][0]
    dcryst_mat[k0]['mesh']['positions']['Ge']['x'] = np.r_[
        0.,
        0.,
        2*uGe,
        2*uGe,
        uGe,
        uGe,
        3*uGe,
        3*uGe
    ]
    dcryst_mat[k0]['mesh']['positions']['Ge']['y'] = np.r_[
        0.,
        2*uGe,
        0.,
        2*uGe,
        uGe,
        3*uGe,
        uGe,
        3*uGe
    ]
    dcryst_mat[k0]['mesh']['positions']['Ge']['z'] = np.r_[
        0.,
        2*uGe,
        2*uGe,
        0.,
        uGe,
        3*uGe,
        3*uGe,
        uGe

    ]
    dcryst_mat[k0]['mesh']['positions']['Ge']['N'] = np.size(
        dcryst_mat[k0]['mesh']['positions']['Ge']['x']
    )



# ##################################################################
# ##################################################################
#                   Elementary box volume
# ##################################################################
# ##################################################################


def hexa_volume(aa, cc):
    return (aa**2) * cc * (np.sqrt(3.)/2.)

def diam_volume(aa):
    return aa**3


# ###############################################################
# ###############################################################
#       atomic absorption coefficients and scattering factor
# ##############################################################
# ###############################################################


def _atomic_coefs_factor_Quartz(
    dcryst_mat=None,
    k0=None,
    v0=None,
):

    Zsi, Zo = v0['atoms_Z']

    # -----------------------------------
    # linear atomic absorption coefficients 'mu'
    # From W. Zachariasen, Theory of X-ray Diffraction in Crystals
    # (Wiley, New York, 1945)

    # Data file for anomalous corrections to atomic scattering factor & attenuation
    # From Lawrence Berkeley National Lab, Center for X-Ray Optics
    # https://henke.lbl.gov/optical_constants/asf.html

    def mu_si(lamb, Zsi=Zsi):
        if lamb > 6.74:     # e-10 ?
            return 5.33e-4*(lamb**2.74)*(Zsi**3.03)
        else:
            return 1.38e-2*(lamb**2.79)*(Zsi**2.73)

    def mu_o(lamb, Z0=Zo):
        return 5.4e-3*(lamb**2.92)*(Zo**3.07)

    def mu(lamb, mu_si=mu_si, mu_o=mu_o):
        return 2.65e-8*(7.*mu_si(lamb) + 8.*mu_o(lamb))/15.

    # store in dict
    dcryst_mat[k0]['mu'] = mu

    # ----------------------------
    # Atomic scattering factor 'f'

    # Same values for 110- and Quartz_102
    sol_si = v0['sin_theta_lambda']['Si']
    sol_o = v0['sin_theta_lambda']['O']
    asf_si = v0['atomic_scattering']['factors']['Si']
    asf_o = v0['atomic_scattering']['factors']['O']
    interp_si = scipy.interpolate.interp1d(sol_si, asf_si)
    interp_o = scipy.interpolate.interp1d(sol_o, asf_o)

    def dfsi_re(lamb):
        return 0.1335*lamb - 6e-3

    def fsi_re(lamb, sol, dfsi_re=dfsi_re, interp_si=interp_si):
        return interp_si(sol) + dfsi_re(lamb)

    def fsi_im(lamb, Zsi=Zsi, mu_si=mu_si):
        return 5.936e-4*Zsi*(mu_si(lamb)/lamb)

    def dfo_re(lamb):
        return 0.1335*lamb - 0.206

    def fo_re(lamb, sol, dfo_re=dfo_re, interp_o=interp_o):
        return interp_o(sol) + dfo_re(lamb)

    def fo_im(lamb, Zo=Zo, mu_o=mu_o):
        return 5.936e-4*Zo*(mu_o(lamb)/lamb)

    # store in dict
    dcryst_mat[k0]['dfsi_re'] = dfsi_re
    dcryst_mat[k0]['dfo_re'] = dfo_re
    dcryst_mat[k0]['fsi_re'] = fsi_re
    dcryst_mat[k0]['fsi_im'] = fsi_im
    dcryst_mat[k0]['fo_re'] = fo_re
    dcryst_mat[k0]['fo_im'] = fo_im

def _atomic_coefs_factor_Germanium(
    dcryst_mat=None,
    k0=None,
    v0=None,
):

    # Useful constant
    import scipy.constants as cnt
    hc = cnt.h*cnt.c/cnt.e*1e10 # [eV*AA]

    # Charge of Germanium
    Zge = v0['atoms_Z']

    # Density of Germanium
    rho_Ge = 5.307 # [g/cm^3]

    # Data file for anomalous corrections to atomic scattering factor & attenuation
    # From Lawrence Berkeley National Lab, Center for X-Ray Optics
    # https://henke.lbl.gov/optical_constants/asf.html

    import os
    table_path = os.path.join(
        os.path.dirname(__file__),
        'Ge_AtomicScatteringFactors_LBL.txt'
        )

    tables = np.loadtxt(table_path)

    # Table values
    E_LBL  = tables[:,0] # [eV]
    f1_LBL = tables[:,1] # [e/atom]
    f2_LBL = tables[:,2] # [e/atom]

    # ----------------------------
    # Atomic scattering factor 'f'

    # Interpolates mean scattering factors
    sol_ge = v0['sin_theta_lambda']['Ge'] # [1/AA]
    asf_ge = v0['atomic_scattering']['factors']['Ge'] # [e/atom]
    interp_ge_f0 = scipy.interpolate.interp1d(sol_ge, asf_ge)

    # Interpolates NIST anomalous scattering factor corrections
    interp_ge_f1 = scipy.interpolate.interp1d(E_LBL, f1_LBL)
    interp_ge_f2 = scipy.interpolate.interp1d(E_LBL, f2_LBL)

    def dfge_re(lamb, Zge=Zge):
        return interp_ge_f1(hc/lamb) - Zge

    def fge_re(lamb, sol, dfge_re=dfge_re, interp_ge_f0=interp_ge_f0):
        return interp_ge_f0(sol) + dfge_re(lamb)

    def fge_im(lamb):
        return interp_ge_f2(hc/lamb)


    # store in dict
    dcryst_mat[k0]['dfge_re'] = dfge_re
    dcryst_mat[k0]['fge_re'] = fge_re
    dcryst_mat[k0]['fge_im'] = fge_im

    # -----------------------------------
    # linear atomic absorption coefficients 'mu'

    # Material-dependent conversion from f2 to mu
    # From NIST X-ray Form Factor, Attenuation, and Scattering Tables
    # https://physics.nist.gov/PhysRefData/FFast/html/form.html
    def mu_ge(lamb):
        return 5.7969e5*fge_im(lamb)*(lamb/hc) * rho_Ge *1e-8 # [1/AA]

    # store in dict
    dcryst_mat[k0]['mu'] = mu_ge



# ###############################################################
# ###############################################################
#                     _DCRYST
# ##############################################################
# ###############################################################


_DCRYST = {
    'Quartz_110': {
        'material': 'Quartz',
        'name': 'Quartz_110',
        'symbol': 'Qz110',
        'miller': np.r_[1., 1., 0.],
        'target': {
            'ion': 'Ar16+',
            'lamb': 3.96, # e-10,
            'units': 'm',
        },
        'd_hkl': None,
    },
    'Quartz_102': {
        'material': 'Quartz',
        'name': 'Quartz_102',
        'symbol': 'Qz102',
        'miller': np.r_[1., 0., 2.],
        'target': {
            'ion': 'Ar17+',
            'lamb': 3.75, # e-10,
            'units': 'm',
        },
        'd_hkl': None,
    },

    # 'Germanium_XXX': {
        # 'material': 'Germanium',
        # 'name': None,
        # 'symbol': None,
        # 'miller': None,
        # 'target': {
            # 'ion': 'Ar16+',
            # 'wavelength': 3.96e-10,
        # }
        # 'd_hkl': None,
    # },
}


# ##############################################################
# ##############################################################
#           Complement cut dict
# ##############################################################


def _complement_dict_cut(dcryst_mat=None, dcryst_cut=None):
    """ Add what is missing to material dict """

    for k0, v0 in dcryst_cut.items():

        # ------------------------
        # complement with material

        # safety check on material
        if v0.get('material') not in dcryst_mat.keys():
            lstr = [f"\t- '{k0}'" for k0 in dcryst_mat.keys()]
            msg = (
                f"Please, for '{k0}', select a material from:\n"
                + "\n".join(lstr)
                + f"\nProvided: {v0.get('material')}\n"
            )
            raise Exception(msg)

        # complement dict
        for k1, v1 in dcryst_mat[v0['material']].items():
            dcryst_cut[k0][k1] = copy.deepcopy(v1)

        # -----------
        # preparation

        hh, kk, ll = v0['miller']

        # ------------------------------
        # inter-reticular distance d_hkl

        # hexagonal
        if v0['mesh']['type'] == 'hexagonal':
            dcryst_cut[k0]['d_hkl'] = hexa_spacing(
                hh,
                kk,
                ll,
                v0['inter_atomic']['distances']['a0'],
                v0['inter_atomic']['distances']['c0'],
            )

        # diamond
        elif v0['mesh']['type'] == 'diamond':
            dcryst_cut[k0]['d_hkl'] = diam_spacing(
                hh,
                kk,
                ll,
                v0['inter_atomic']['distances']['a0'],
            )

        # other
        else:
            msg = f"Mesh d_hkl not implemented for '{k0}'"
            raise NotImplementedError(msg)

        # ------------
        # phases

        # Quartz
        if v0['material'] == 'Quartz':

            Nsi = v0['mesh']['positions']['Si']['N']
            No = v0['mesh']['positions']['O']['N']

            def phasesi(hh, kk, ll, xsi, ysi, zsi):
                return hh*xsi + kk*ysi + ll*zsi

            def phaseo(hh, kk, ll, xo, yo, zo):
                return hh*xo + kk*yo + ll*zo

            # initiate
            dcryst_cut[k0]['phases']['Si'] = np.full((Nsi,), np.nan)
            dcryst_cut[k0]['phases']['O'] = np.full((No,), np.nan)

            # Quartz
            for ii in range(Nsi):
                dcryst_cut[k0]['phases']['Si'][ii] = phasesi(
                    hh,
                    kk,
                    ll,
                    v0['mesh']['positions']['Si']['x'][ii],
                    v0['mesh']['positions']['Si']['y'][ii],
                    v0['mesh']['positions']['Si']['z'][ii],
                )

            # Oxygen
            for ii in range(No):
                dcryst_cut[k0]['phases']['O'][ii] = phaseo(
                    hh,
                    kk,
                    ll,
                    v0['mesh']['positions']['O']['x'][ii],
                    v0['mesh']['positions']['O']['y'][ii],
                    v0['mesh']['positions']['O']['z'][ii],
                )

        # Germanium
        elif v0['material'] == 'Germanium':

            Nge = v0['mesh']['positions']['Ge']['N']

            # dot(s, r_atom); s = lattice vector, r_atom = atom position
            def phasege(hh, kk, ll, xge, yge, zge):
                return hh*xge + kk*yge + ll*zge

            # initiate
            dcryst_cut[k0]['phases']['Ge'] = np.full((Nge,), np.nan)

            # Germanium
            for ii in range(Nge):
                dcryst_cut[k0]['phases']['Ge'][ii] = phasege(
                    hh,
                    kk,
                    ll,
                    v0['mesh']['positions']['Ge']['x'][ii],
                    v0['mesh']['positions']['Ge']['y'][ii],
                    v0['mesh']['positions']['Ge']['z'][ii],
                )

        # Other
        else:
            msg = f"Phases not implemented for '{k0}'"
            raise NotImplementedError(msg)

    return dcryst_cut


# #################################################################
# #################################################################
#               Inter-reticular spacing d_hkl
# #################################################################
# #################################################################


def hexa_spacing(hh, kk, ll, aa, cc):
    return np.sqrt(
        (3.*(aa**2)*(cc**2))
        / (4.*(hh**2 + kk**2 + hh*kk)*(cc**2) + 3.*(ll**2)*(aa**2))
    )

def diam_spacing(hh, kk, ll, aa):
    return aa / np.sqrt(hh**2 + kk**2 + ll**2)


# ###############################################################
# ###############################################################
#                   Run complement routines
# ###############################################################
# ###############################################################


_complement_dict_mat(_DCRYST_MAT)
_complement_dict_cut(dcryst_mat=_DCRYST_MAT, dcryst_cut=_DCRYST)


def _build_cry(
    crystal=None,
    din=None,
):
    '''
    _build_cry is a function meant to populate a crystal dictionary
    using either hardcoded WEST default crystal names ['Quartz_110', 'Quartz_102']
    or a user-defined crystal as a dictionary in the form of --

    # C-Mod's crystal
    dcry = {
        'material': 'Quartz',
        'name': 'HIREX',
        'symbol': 'Qz102',
        'miller': np.r_[1., 0., 2.,],
        'target': {
            'ion': 'Ar16+',
            'lamb': 3.96,
            'units': 'AA',
            },
        'd_hkl': None,
        }

    '''

    # Hardcoded defaults
    if crystal in ['Quartz_110', 'Quartz_102']:
        dcry = _DCRYST

    # Else if user-defined crystal
    elif isinstance(din, dict):
        dcry = {
            crystal: din
        }

    # Builds the crystal dictionary
    _complement_dict_cut(dcryst_mat=_DCRYST_MAT, dcryst_cut=dcry)

    return dcry[crystal]
