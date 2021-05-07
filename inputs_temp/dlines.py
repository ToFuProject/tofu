import scipy.constants as scpct

_DSOURCES = {
    'Kallne': {
        'long': (
            'Kallne et al., '
            + 'High Resolution X-Ray Spectroscopy Diagnostics'
            + ' of High Temperature Plasmas'
            + ', Physica Scripta, vol. 31, 6, pp. 551-564, 1985'
        ),
    },
    'Bitter': {
        'long': (
            'Bitter et al., '
            + 'XRay diagnostics of tokamak plasmas'
            + ', Physica Scripta, vol. 47, pp. 87-95, 1993'
        ),
    },
    'Gabriel': {
        'long': (
            'Gabriel,'
            + 'Mon. Not. R. Astro. Soc., vol. 160, pp 99-119, 1972'
        ),
    },
    'NIST': {
        'long': 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html',
    },
    'Vainshtein 85': {
        'long': (
            'Vainshtein and Safranova, '
            + 'Energy Levels of He-like and Li-like Ions, '
            + 'Physica Scripta, vol. 31, pp 519-532, 1985'
        ),
    },
    'Goryaev 17': {
        'long': (
            "Goryaev et al., "
            + "Atomic data for doubly-excited states 2lnl' of He-like "
            + "ions and 1s2lnl' of Li-like ions with Z=6-36 and n=2,3, "
            + "Atomic Data and Nuclear Data Tables, vol. 113, "
            + "pp 117-257, 2017"
        ),
    },
    'Bruhns 07': {
        'long': (
            'Bruhns et al.,'
            + '"Testing QED Screening and Two-Loop Contributions "'
            + '"with He-Like Ions", '
            + 'Physical Review Letters, vol. 99, 113001, 2007'
        ),
    },
    'Amaro 12': {
        'long': (
            'Amaro et al.,'
            + '"Absolute Measurement of the Relativistic Magnetic"'
            + '" Dipole Transition Energy in Heliumlike Argon", '
            + 'Physical Review Letters, vol. 109, 043005, 2012'
        ),
    },
    'Adhoc 200408': {
        'long': (
            'Wavelength computed from the solid references '
            + 'ArXVII_w_Bruhns and ArXVII_z_Amaro and from the '
            + 'detector position optimized from them using shots='
            +  '[54044, 54045, 54046, 54047, 54049, 54061, 55076], '
            + 'indt=[2,4,5,6,8], indxj=None on 08.04.2020, using '
            + 'Vainshtein for x, y and Goryaev for k, j, q, r, a'
        ),
    },
    'Adhoc 200513': {
        'long': (
            'Same as Adhoc 200408 but n3, n4 and y corrected by'
            + ' individual vims computed from C3, C4 campaigns, '
            + 'as presented in CTVD on 14.05.2020'
        ),
    },
}


delements = {
    'Ar': {'Z': 18, 'A': 39.948},
    'Fe': {'Z': 26, 'A': 55.845},
    'W': {'Z': 74, 'A': 183.84}
}
for k0, v0 in delements.items():
    delements[k0]['m'] = v0['Z']*scpct.m_p + v0['A']*scpct.m_n

# In dtransitions: ['lower state', 'upper state']
# Source: Gabriel
dtransitions = {
    # 1s^22p(^2P^0) - 1s2p^2(^2P)
    'Li-a': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2P_{3/2})'],
    },
    'Li-b': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2P_{3/2})'],
    },
    'Li-c': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2P_{1/2})'],
    },
    'Li-d': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2P_{1/2})'],
    },

    # 1s^22p(^2P^0) - 1s2p^2(^4P)
    'Li-e': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{5/2})'],
    },
    'Li-f': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{3/2})'],
    },
    'Li-g': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^4P_{3/2})'],
    },
    'Li-h': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{1/2})'],
    },
    'Li-i': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^4P_{1/2})'],
    },

    # 1s^22p(^2P^0) - 1s2p^2(^2D)
    'Li-j': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2D_{5/2})'],
    },
    'Li-k': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2D_{3/2})'],
    },
    'Li-l': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2D_{3/2})'],
    },

    # 1s^22p(^2P^0) - 1s2p^2(^2S)
    'Li-m': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2S_{1/2})'],
    },
    'Li-n': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2S_{1/2})'],
    },

    # 1s^22p(^2P^0) - 1s2s^2(^2S)
    'Li-o': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{3/2})', '1s2s^2(^2S_{1/2})'],
    },
    'Li-p': {
        'isoel': 'Li-like',
        'trans': ['1s^22p(^2P^0_{1/2})', '1s2s^2(^2S_{1/2})'],
    },

    # 1s^22s(^2S) - 1s2s2p(^1P)(^2P^0)
    'Li-q': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^1P^0)(^2P^0_{3/2})'],
    },
    'Li-r': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^1P^0)(^2P^0_{1/2})'],
    },

    # 1s^22s(^2S) - 1s2s2p(^3P)(^2P^0)
    'Li-s': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^3P^0)(^2P^0_{3/2})'],
    },
    'Li-t': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^3P^0)(^2P^0_{1/2})'],
    },

    # 1s^22s(^2S) - 1s2s2p(^4P^0)
    'Li-u': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^4P^0_{3/2})'],
    },
    'Li-v': {
        'isoel': 'Li-like',
        'trans': ['1s^22s(^2S_{1/2})', '1s2s2p(^4P^0_{1/2})'],
    },

    # Satellites of ArXVII w from n = 3
    'Li-n3-a1': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{1/2})', '1s2p3p(^2S_{1/2})'],
    },
    'Li-n3-a2': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2S_{1/2})'],
    },
    'Li-n3-b1': {
        'isoel': 'Li-like',
        'trans': ['1s^23d(^2D_{3/2})', '1s2p3d(^2F_{5/2})'],
    },
    'Li-n3-b2': {
        'isoel': 'Li-like',
        'trans': ['1s^23d(^2D_{5/2})', '1s2p3d(^2F_{5/2})'],
    },
    'Li-n3-b3': {
        'isoel': 'Li-like',
        'trans': ['1s^23d(^2D_{5/2})', '1s2p3d(^2F_{7/2})'],
    },
    'Li-n3-b4': {
        'isoel': 'Li-like',
        'trans': ['1s^23d(^2D_{5/2})', '1s2p3d(^2D_{5/2})'],
    },
    'Li-n3-c1': {
        'isoel': 'Li-like',
        'trans': ['1s^23s(^2S_{1/2})', '1s2p3s(^2P_{1/2})'],
    },
    'Li-n3-c2': {
        'isoel': 'Li-like',
        'trans': ['1s^23s(^2S_{1/2})', '1s2p3s(^2P_{3/2})'],
    },
    'Li-n3-d1': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2P_{3/2})'],
    },
    'Li-n3-d2': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{1/2})', '1s2p3p(^2D_{3/2})'],
    },
    'Li-n3-d3': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2D_{5/2})'],
    },
    'Li-n3-e1': {
        'isoel': 'Li-like',
        'trans': ['1s^23s(^2S_{1/2})', '1s2p3s(^2P_{3/2})'],
    },
    'Li-n3-f1': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2S_{1/2})'],
    },
    'Li-n3-e2': {
        'isoel': 'Li-like',
        'trans': ['1s^23s(^2S_{1/2})', '1s2p3s(^2P_{1/2})'],
    },
    'Li-n3-f2': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2D_{5/2})'],
    },
    'Li-n3-g1': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{1/2})', '1s2s3d(^2D_{3/2})'],
    },
    'Li-n3-f3': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3p(^2D_{3/2})'],
    },
    'Li-n3-g2': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2s3d(^2D_{5/2})'],
    },
    'Li-n3-g3': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2s3d(^2D_{3/2})'],
    },
    'Li-n3-f4': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2p3d(^4P_{5/2})'],
    },
    'Li-n3-h1': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{1/2})', '1s2s3s(^2S_{1/2})'],
    },
    'Li-n3-h2': {
        'isoel': 'Li-like',
        'trans': ['1s^23p(^2P_{3/2})', '1s2s3s(^2S_{1/2})'],
    },

    # He-like

    # 1s^2(^1S) - 1s2p(^1P^0)  -  Resonance
    'He-w': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2p(^1P^0_{1})'],
    },

    # 1s^2(^1S) - 1s2p(^3P^0)  -  Forbidden
    'He-x': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2p(^3P^0_{2})'],
    },
    'He-y': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2p(^3P^0_{1})'],
    },
    'He-y2': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2p(^3P^0_{0})'],
    },

    # 1s^2(^1S) - 1s2s(^3S)  -  Forbidden
    'He-z': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2s(^3S_{1})'],
    },
    'He-z2': {
        'isoel': 'He-like',
        'trans': ['1s^2(^1S_{0})', '1s2s(^1S_{0})'],
    },
}




dlines = {
    # --------------------------
    # Ar
    # --------------------------

    'ArXIV_n4_Adhoc200408': {'charge': 13, 'ION': 'ArXIV',
                             'symbol': 'n4', 'lambda0': 3.9530e-10,
                             'transition': ['unknown', 'unknwown'],
                             'source': 'Adhoc 200408'},
    'ArXIV_n4_Adhoc200513': {'charge': 13, 'ION': 'ArXIV',
                             'symbol': 'n4', 'lambda0': 3.9528e-10,
                             'transition': ['unknown', 'unknwown'],
                             'source': 'Adhoc 200513'},
    'ArXV_n3_Adhoc200408': {'charge': 14, 'ION': 'ArXV',
                            'symbol': 'n3', 'lambda0': 3.9560e-10,
                            'transition': ['unknown', 'unknwown'],
                            'source': 'Adhoc 200408'},
    'ArXV_n3_Adhoc200513': {'charge': 14, 'ION': 'ArXV',
                            'symbol': 'n3', 'lambda0': 3.9562e-10,
                            'transition': ['unknown', 'unknwown'],
                            'source': 'Adhoc 200513'},

    'ArXV_1': {'charge': 14, 'ION': 'ArXV',
               'symbol': '1', 'lambda0': 4.0096e-10,
               'transition': ['1s2s^22p(^1P_1)', '1s^22s^2(^1S_0)'],
               'source': 'Kallne', 'innershell': True},
    'ArXV_2-1': {'charge': 14, 'ION': 'ArXV',
                 'symbol': '2-1', 'lambda0': 4.0176e-10,
                 'transition': ['1s2p^22s(^4P^3P_1)', '1s^22s2p(^3P_1)'],
                 'source': 'Kallne'},
    'ArXV_2-2': {'charge': 14, 'ION': 'ArXV',
                 'symbol': '2-2', 'lambda0': 4.0179e-10,
                 'transition': ['1s2s2p^2(^3D_1)', '1s^22s2p(^3P_0)'],
                 'source': 'Kallne'},
    'ArXV_2-3': {'charge': 14, 'ION': 'ArXV',
                 'symbol': '2-3', 'lambda0': 4.0180e-10,
                 'transition': ['1s2p^22s(^4P^3P_2)', '1s^22s2p(^3P_2)'],
                 'source': 'Kallne'},
    'ArXV_3': {'charge': 14, 'ION': 'ArXV',
               'symbol': '3', 'lambda0': 4.0192e-10,
               'transition': ['1s2s2p^2(^3D_2)', '1s^22s2p(^3P_1)'],
               'source': 'Kallne'},
    'ArXV_4': {'charge': 14, 'ION': 'ArXV',
               'symbol': '4', 'lambda0': 4.0219e-10,
               'transition': ['1s2s2p^2(^3D_5)', '1s^22s2p(^3P_2)'],
               'source': 'Kallne'},
    'ArXV_5': {'charge': 14, 'ION': 'ArXV',
               'symbol': '5', 'lambda0': 4.0291e-10,
               'transition': ['1s2s2p^2(^1D_5)', '1s^22s2p(^1P_1)'],
               'source': 'Kallne'},

    'ArXVI_a_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9852e-10,
                       'transition': ('Li-like', 'a'),
                       'source': 'Kallne'},
    'ArXVI_a_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.98573e-10,
                     'transition': ('Li-like', 'a'),
                     'source': 'NIST'},
    'ArXVI_a_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9858e-10,
                        'transition': ('Li-like', 'a'),
                        'source': 'Goryaev 17'},
    'ArXVI_a_Adhoc200408': {'charge': 15, 'ION': 'ArXVI',
                            'lambda0': 3.9848e-10,
                            'transition': ('Li-like', 'a'),
                            'source': 'Adhoc 200408'},
    'ArXVI_b_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9818e-10,
                        'transition': ('Li-like', 'b'),
                        'source': 'Goryaev 17'},
    'ArXVI_c_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9899e-10,
                        'transition': ('Li-like', 'c'),
                        'source': 'Goryaev 17'},
    'ArXVI_d_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9858e-10,
                        'transition': ('Li-like', 'd'),
                        'source': 'Goryaev 17'},
    'ArXVI_e_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0126e-10,
                        'transition': ('Li-like', 'e'),
                        'source': 'Goryaev 17'},
    'ArXVI_f_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0146e-10,
                        'transition': ('Li-like', 'f'),
                        'source': 'Goryaev 17'},
    'ArXVI_g_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0105e-10,
                        'transition': ('Li-like', 'g'),
                        'source': 'Goryaev 17'},
    'ArXVI_h_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0164e-10,
                        'transition': ('Li-like', 'h'),
                        'source': 'Goryaev 17'},
    'ArXVI_i_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0123e-10,
                        'transition': ('Li-like', 'i'),
                        'source': 'Goryaev 17'},
    'ArXVI_j_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9932e-10,
                       'transition': ('Li-like', 'j'),
                       'source': 'Kallne'},
    'ArXVI_j_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9938e-10,
                     'transition': ('Li-like', 'j'),
                     'source': 'NIST'},
    'ArXVI_j_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9939e-10,
                        'transition': ('Li-like', 'j'),
                        'source': 'Goryaev 17'},
    'ArXVI_j_Adhoc200408': {'charge': 15, 'ION': 'ArXVI',
                            'lambda0': 3.9939e-10,
                            'transition': ('Li-like', 'j'),
                            'source': 'Adhoc 200408'},
    'ArXVI_k_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9892e-10,
                       'transition': ('Li-like', 'k'),
                       'source': 'Kallne', 'comment': 'Dielect. recomb. from ArXVII'},
    'ArXVI_k_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9898e-10,
                     'transition': ('Li-like', 'k'),
                     'source': 'NIST', 'comment': 'Dielect. recomb. from ArXVII'},
    'ArXVI_k_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9899e-10,
                        'transition': ('Li-like', 'k'),
                        'source': 'Goryaev 17'},
    'ArXVI_k_Adhoc200408': {'charge': 15, 'ION': 'ArXVI',
                            'lambda0': 3.9897e-10,
                            'transition': ('Li-like', 'k'),
                            'source': 'Adhoc 200408'},
    'ArXVI_l_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9939e-10,
                        'transition': ('Li-like', 'l'),
                        'source': 'Goryaev 17'},
    'ArXVI_m_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9562e-10,
                       'transition': ('Li-like', 'm'),
                       'source': 'Kallne'},
    'ArXVI_m_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.96561e-10,
                     'transition': ('Li-like', 'm'),
                     'source': 'NIST'},
    'ArXVI_m_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9656e-10,
                        'transition': ('Li-like', 'm'),
                        'source': 'Goryaev 17'},
    'ArXVI_n_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9616e-10,
                        'transition': ('Li-like', 'n'),
                        'source': 'Goryaev 17'},
    'ArXVI_o_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0730e-10,
                        'transition': ('Li-like', 'o'),
                        'source': 'Goryaev 17'},
    'ArXVI_p_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0688e-10,
                        'transition': ('Li-like', 'p'),
                        'source': 'Goryaev 17'},
    'ArXVI_q_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9806e-10,
                       'transition': ('Li-like', 'q'),
                       'source': 'Kallne', 'innershell': True},
    'ArXVI_q_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9676e-10,
                     'transition': ('Li-like', 'q'),
                     'source': 'NIST', 'innershell': True},
    'ArXVI_q_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9815e-10,
                        'transition': ('Li-like', 'q'),
                        'source': 'Goryaev 17'},
    'ArXVI_q_Adhoc200408': {'charge': 15, 'ION': 'ArXVI',
                            'lambda0': 3.9814e-10,
                            'transition': ('Li-like', 'q'),
                            'source': 'Adhoc 200408'},
    'ArXVI_r_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9827e-10,
                       'transition': ('Li-like', 'r'),
                       'source': 'Kallne'},
    'ArXVI_r_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9685e-10,
                     'transition': ('Li-like', 'r'),
                     'source': 'NIST'},
    'ArXVI_r_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9835e-10,
                        'transition': ('Li-like', 'r'),
                        'source': 'Goryaev 17'},
    'ArXVI_r_Adhoc200408': {'charge': 15, 'ION': 'ArXVI',
                            'lambda0': 3.9833e-10,
                            'transition': ('Li-like', 'r'),
                            'source': 'Adhoc 200408'},
    'ArXVI_s_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9669e-10,
                       'transition': ('Li-like', 's'),
                       'source': 'Kallne'},
    'ArXVI_s_NIST': {'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9813e-10,
                     'transition': ('Li-like', 's'),
                     'source': 'NIST'},
    'ArXVI_s_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9677e-10,
                        'transition': ('Li-like', 's'),
                        'source': 'Goryaev 17'},
    'ArXVI_t_Kallne': {'charge': 15, 'ION': 'ArXVI',
                       'lambda0': 3.9677e-10,
                       'transition': ('Li-like', 't'),
                       'source': 'Kallne'},
    'ArXVI_t_NIST': {'Z': 18, 'charge': 15, 'ION': 'ArXVI',
                     'lambda0': 3.9834e-10,
                     'transition': ('Li-like', 't'),
                     'source': 'NIST'},
    'ArXVI_t_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 3.9686e-10,
                        'transition': ('Li-like', 't'),
                        'source': 'Goryaev 17'},
    'ArXVI_u_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0150e-10,
                        'transition': ('Li-like', 'u'),
                        'source': 'Goryaev 17'},
    'ArXVI_v_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                        'lambda0': 4.0161e-10,
                        'transition': ('Li-like', 'v'),
                        'source': 'Goryaev 17'},

    # Li-like n=3 satellites
    'ArXVI_n3a1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9473e-10,
                           'transition': ('Li-like', 'n3-a1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3a2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9484e-10,
                           'transition': ('Li-like', 'n3-a2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3b1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9512e-10,
                           'transition': ('Li-like', 'n3-b1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3b2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9515e-10,
                           'transition': ('Li-like', 'n3-b2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3b3_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9527e-10,
                           'transition': ('Li-like', 'n3-b3'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3b4_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9542e-10,
                           'transition': ('Li-like', 'n3-b4'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3c1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9546e-10,
                           'transition': ('Li-like', 'n3-c1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3c2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9551e-10,
                           'transition': ('Li-like', 'n3-c2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3d1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9556e-10,
                           'transition': ('Li-like', 'n3-d1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3d2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9559e-10,
                           'transition': ('Li-like', 'n3-d2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3d3_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9568e-10,
                           'transition': ('Li-like', 'n3-d3'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3e1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9626e-10,
                           'transition': ('Li-like', 'n3-e1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3f1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9643e-10,
                           'transition': ('Li-like', 'n3-f1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3e2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9661e-10,
                           'transition': ('Li-like', 'n3-e2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3f2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9680e-10,
                           'transition': ('Li-like', 'n3-f2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3g1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9706e-10,
                           'transition': ('Li-like', 'n3-g1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3f3_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9711e-10,
                           'transition': ('Li-like', 'n3-f3'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3g2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9712e-10,
                           'transition': ('Li-like', 'n3-g2'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3g3_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9718e-10,
                           'transition': ('Li-like', 'n3-g3'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3f4_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9740e-10,
                           'transition': ('Li-like', 'n3-f4'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3h1_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9922e-10,
                           'transition': ('Li-like', 'n3-h1'),
                           'source': 'Goryaev 17'},
    'ArXVI_n3h2_Goryaev': {'charge': 15, 'ION': 'ArXVI',
                           'lambda0': 3.9934e-10,
                           'transition': ('Li-like', 'n3-h2'),
                           'source': 'Goryaev 17'},

    # He-like
    'ArXVII_w_Kallne': {'charge': 16, 'ION': 'ArXVII',
                        'lambda0': 3.9482e-10,
                        'transition': ('He-like', 'w'),
                        'source': 'Kallne'},
    'ArXVII_w_NIST': {'charge': 16, 'ION': 'ArXVII',
                      'lambda0': 3.94906e-10,
                      'transition': ('He-like', 'w'),
                      'source': 'NIST'},
    'ArXVII_w_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                            'lambda0': 3.9492e-10,
                            'transition': ('He-like', 'w'),
                            'source': 'Vainshtein 85'},
    'ArXVII_w_Goryaev': {'charge': 16, 'ION': 'ArXVII',
                         'lambda0': 3.9493e-10,
                         'transition': ('He-like', 'w'),
                         'source': 'Goryaev 17'},
    'ArXVII_w_Bruhns': {'charge': 16, 'ION': 'ArXVII',
                        'lambda0': 3.94906e-10,
                        'transition': ('He-like', 'w'),
                        'source': 'Bruhns 07'},
    'ArXVII_x_Kallne': {'charge': 16, 'ION': 'ArXVII',
                        'lambda0': 3.9649e-10,
                        'transition': ('He-like', 'x'),
                        'source': 'Kallne'},
    'ArXVII_x_NIST': {'charge': 16, 'ION': 'ArXVII',
                      'lambda0': 3.965857e-10,
                      'transition': ('He-like', 'x'),
                      'source': 'NIST'},
    'ArXVII_x_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                            'lambda0': 3.9660e-10,
                            'transition': ('He-like', 'x'),
                            'source': 'Vainshtein 85'},
    'ArXVII_x_Adhoc200408': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.9658e-10,
                             'transition': ('He-like', 'x'),
                             'source': 'Adhoc 200408'},
    'ArXVII_x_Adhoc200513': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.9659e-10,
                             'transition': ('He-like', 'x'),
                             'source': 'Adhoc 200513'},
    'ArXVII_y_Kallne': {'charge': 16, 'ION': 'ArXVII',
                        'lambda0': 3.9683e-10,
                        'transition': ('He-like', 'y'),
                        'source': 'Kallne'},
    'ArXVII_y_NIST': {'charge': 16, 'ION': 'ArXVII',
                      'lambda0': 3.969355e-10,
                      'transition': ('He-like', 'y'),
                      'source': 'NIST'},
    'ArXVII_y_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                            'lambda0': 3.9694e-10,
                            'transition': ('He-like', 'y'),
                            'source': 'Vainshtein 85'},
    'ArXVII_y_Goryaev': {'charge': 16, 'ION': 'ArXVII',
                         'lambda0': 3.9696e-10,
                         'transition': ('He-like', 'y'),
                         'source': 'Goryaev 17'},
    'ArXVII_y_Adhoc200408': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.9692e-10,
                             'transition': ('He-like', 'y'),
                             'source': 'Adhoc 200408'},
    'ArXVII_y_Adhoc200513': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.96933e-10,
                             'transition': ('He-like', 'y'),
                             'source': 'Adhoc 200513'},
    'ArXVII_y2_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.9703e-10,
                             'transition': ('He-like', 'y2'),
                             'source': 'Vainshtein 85'},
    'ArXVII_z_Kallne': {'charge': 16, 'ION': 'ArXVII',
                        'lambda0': 3.9934e-10,
                        'transition': ('He-like', 'z'),
                        'source': 'Kallne'},
    'ArXVII_z_NIST': {'charge': 16, 'ION': 'ArXVII',
                      'lambda0': 3.99414e-10,
                      'transition': ('He-like', 'z'),
                      'source': 'NIST'},
    'ArXVII_z_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                            'lambda0': 3.9943e-10,
                            'transition': ('He-like', 'z'),
                            'source': 'Vainshtein 85'},
    'ArXVII_z_Goryaev': {'charge': 16, 'ION': 'ArXVII',
                         'lambda0': 3.9944e-10,
                         'transition': ('He-like', 'z'),
                         'source': 'Goryaev 17'},
    'ArXVII_z2_Vainshtein': {'charge': 16, 'ION': 'ArXVII',
                             'lambda0': 3.9682e-10,
                             'transition': ('He-like', 'z2'),
                             'source': 'Vainshtein 85'},
    'ArXVII_z_Amaro': {'charge': 16, 'ION': 'ArXVII',
                       'lambda0': 3.994129e-10,
                       'transition': ('He-like', 'z'),
                       'source': 'Amaro 12'},
    'ArXVII_T': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'T', 'lambda0': 3.7544e-10,
                 'transition': [r'$2s2p(^1P_1)$', r'$1s2s(^1S_0)$'],
                 'source': 'Kallne'},
    'ArXVII_K': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'K', 'lambda0': 3.7557e-10,
                 'transition': [r'$2p^2(^1D_2)$', r'$1s2p(^3P_2)$'],
                 'source': 'Kallne'},
    'ArXVII_Q': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'Q', 'lambda0': 3.7603e-10,
                 'transition': [r'$2s2p(^3P_2)$', r'$1s2s(^3S_1)$'],
                 'source': 'Kallne'},
    'ArXVII_B': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'B', 'lambda0': 3.7626e-10,
                 'transition': [r'$2p^2(^3P_2)$', r'$1s2p(^3P_1)$'],
                 'source': 'Kallne'},
    'ArXVII_R': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'R', 'lambda0': 3.7639e-10,
                 'transition': [r'$2s2p(^3P_1)$', r'$1s2s(^3S_1)$'],
                 'source': 'Kallne'},
    'ArXVII_A': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'A', 'lambda0': 3.7657e-10,
                 'transition': [r'$2p^2(^3P_2)$', r'$1s2p(^3P_2)$'],
                 'source': 'Kallne'},
    'ArXVII_J': {'charge': 16, 'ION': 'ArXVII',
                 'symbol': 'J', 'lambda0': 3.7709e-10,
                 'transition': [r'$2p^2(^1D_2)$', r'$1s2p(^1P_1)$'],
                 'source': 'Kallne'},

    'ArXVIII_W1': {'charge': 17, 'ION': 'ArXVIII',
                   'symbol': 'W_1', 'lambda0': 3.7300e-10,
                   'transition': [r'$2p(^2P_{3/2})$', r'$1s(^2S_{1/2})$'],
                   'source': 'Kallne'},
    'ArXVIII_W2': {'charge': 17, 'ION': 'ArXVIII',
                   'symbol': 'W_2', 'lambda0': 3.7352e-10,
                   'transition': [r'$2p(^2P_{1/2})$', r'$1s(^2S_{1/2})$'],
                   'source': 'Kallne'},

    # --------------------------
    # Fe
    # --------------------------

    'FeXXIII_beta': {'charge': 22, 'ION': 'FeXXIII',
                     'symbol': 'beta', 'lambda0': 1.87003e-10,
                     'transition': [r'$1s^22s^2(^1S_0)$',
                                    r'$1s2s^22p(^1P_1)$'],
                     'source': 'Bitter'},

    'FeXXIV_t_Bitter': {'charge': 23, 'ION': 'FeXXIV',
                        'lambda0': 1.8566e-10,
                        'transition': ('Li-like', 't'),
                        'source': 'Bitter'},
    'FeXXIV_q_Bitter': {'charge': 23, 'ION': 'FeXXIV',
                        'lambda0': 1.8605e-10,
                        'transition': ('Li-like', 'q'),
                        'source': 'Bitter'},
    'FeXXIV_k_Bitter': {'charge': 23, 'ION': 'FeXXIV',
                        'lambda0': 1.8626e-10,
                        'transition': ('Li-like', 'k'),
                        'source': 'Bitter'},
    'FeXXIV_r_Bitter': {'charge': 23, 'ION': 'FeXXIV',
                        'lambda0': 1.8631e-10,
                        'transition': ('Li-like', 'r'),
                        'source': 'Bitter'},
    'FeXXIV_j_Bitter': {'charge': 23, 'ION': 'FeXXIV',
                        'lambda0': 1.8654e-10,
                        'transition': ('Li-like', 'j'),
                        'source': 'Bitter'},

    'FeXXV_w_Bitter': {'charge': 24, 'ION': 'FeXXV',
                       'lambda0': 1.8498e-10,
                       'transition': ('He-like', 'w'),
                       'source': 'Bitter'},
    'FeXXV_x_Bitter': {'charge': 24, 'ION': 'FeXXV',
                       'lambda0': 1.85503e-10,
                       'transition': ('He-like', 'x'),
                       'source': 'Bitter'},
    'FeXXV_y_Bitter': {'charge': 24, 'ION': 'FeXXV',
                       'lambda0': 1.8590e-10,
                       'transition': ('He-like', 'y'),
                       'source': 'Bitter'},
    'FeXXV_z_Bitter': {'charge': 24, 'ION': 'FeXXV',
                       'lambda0': 1.8676e-10,
                       'transition': ('He-like', 'z'),
                       'source': 'Bitter'},

    # --------------------------
    # W
    # --------------------------

    'W_adhoc_Adhoc200513': {'charge': 43, 'ION': 'WXLIV',
                            'symbol':'adhoc', 'lambda0': 3.97509e-10,
                            'transition': ['?', '?'],
                            'source': 'Adhoc 200513'},
    'WXLIV_0_NIST': {'charge': 43, 'ION': 'WXLIV',
                     'symbol':'0', 'lambda0': 3.9635e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3d^94s^24p(3/2,1/2)^0_16f(1,5/2)3/2'],
                     'source': 'NIST'},
    'WXLIV_1_NIST': {'charge': 43, 'ION': 'WXLIV',
                     'symbol':'1', 'lambda0': 3.9635e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3d^94s^24p(3/2,1/2)^0_26f(2,5/2)1/2'],
                     'source': 'NIST'},
    'WXLIV_2_NIST': {'charge': 43, 'ION': 'WXLIV',
                     'symbol':'2', 'lambda0': 4.017e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3p^53d^{10}4s^24p(3/2,1/2)_25d(2,5/2)3/2'],
                     'source': 'NIST'},
    'WXLIV_3_NIST': {'charge': 43, 'ION': 'WXLIV',
                     'symbol':'3', 'lambda0': 4.017e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3p^53d^{10}4s^24p(3/2,1/2)_25d(2,5/2)1/2'],
                     'source': 'NIST'},
    'WXLV_0_NIST': {'charge': 44, 'ION': 'WXLV',
                    'symbol':'0', 'lambda0': 3.9730e-10,
                    'transition': ['3d^{10}4s^2(^1S_{0})',
                                   '3p^5(^2P^0_{3/2})3d^{10}4s^25d(3/2,5/2)^01'],
                    'source': 'NIST'},
    'WXLV_1_NIST': {'charge': 44, 'ION': 'WXLV',
                    'symbol':'1', 'lambda0': 3.9895e-10,
                    'transition': ['3d^{10}4s^2(^1S_{0})',
                                   '3d^9(^2D_{5/2})4s^26f(5/2,7/2)^01'],
                    'source': 'NIST'},
    'WLIII_0_NIST': {'charge': 52, 'ION': 'WLIII',
                    'symbol':'0', 'lambda0': 4.017e-10,
                    'transition': ['3d^{10}4s^24p^2(^3P_{0})',
                                   '3d^9(^2D_{3/2})4s^24p^2(^3P^0)(3/2,0)_{3/2}6f(3/2,5/2)^01'],
                    'source': 'NIST'},
}


for k0, v0 in dlines.items():
    elem = v0['ION'][:2]
    if elem[1].isupper():
        elem = elem[0]
    dlines[k0]['element'] = elem
    for k1, v1 in delements[elem].items():
        dlines[k0][k1] = v1
    if isinstance(v0['transition'], tuple):
        trans = v0['transition'][0].split('-')[0] + '-' + v0['transition'][1]
        #trans = dtransitions[v0['transition'][0]][v0['transition'][1]]
        dlines[k0]['symbol'] = trans[trans.index('-')+1:]
        dlines[k0]['transition'] = trans
