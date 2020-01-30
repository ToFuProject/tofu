import scipy.constants as scpct

_DSOURCES = {
    'Kallne': ('Kallne et al., '
               + 'High Resolution X-Ray Spectroscopy Diagnostics'
               + ' of High Temperature Plasmas'
               + ', Physica Scripta, vol. 31, 6, pp. 551-564, 1985'),
    'Bitter': ('Bitter et al., '
               + 'XRay diagnostics of tokamak plasmas'
               + ', Physica Scripta, vol. 47, pp. 87-95, 1993'),
    'Gabriel': ('Gabriel,'
               + 'Mon. Not. R. Astro. Soc., vol. 160, pp 99-119, 1972'),
    'NIST': 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html'
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
    'Li-like': {
        # 1s^22p(^2P^0) - 1s2p^2(^2P)
        'a': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2P_{3/2})'],
        'b': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2P_{3/2})'],
        'c': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2P_{1/2})'],
        'd': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2P_{1/2})'],
        # 1s^22p(^2P^0) - 1s2p^2(^4P)
        'e': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{5/2})'],
        'f': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{3/2})'],
        'g': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^4P_{3/2})'],
        'h': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^4P_{1/2})'],
        'i': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^4P_{1/2})'],
        # 1s^22p(^2P^0) - 1s2p^2(^2D)
        'j': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2D_{5/2})'],
        'k': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2D_{3/2})'],
        'l': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2D_{3/2})'],
        # 1s^22p(^2P^0) - 1s2p^2(^2S)
        'm': ['1s^22p(^2P^0_{3/2})', '1s2p^2(^2S_{1/2})'],
        'n': ['1s^22p(^2P^0_{1/2})', '1s2p^2(^2S_{1/2})'],
        # 1s^22p(^2P^0) - 1s2s^2(^2S)
        'o': ['1s^22p(^2P^0_{3/2})', '1s2s^2(^2S_{1/2})'],
        'p': ['1s^22p(^2P^0_{1/2})', '1s2s^2(^2S_{1/2})'],
        # 1s^22s(^2S) - 1s2s2p(^1P)(^2P^0)
        'q': ['1s^22s(^2S_{1/2})', '1s2s2p(^1P^0)(^2P^0_{3/2})'],
        'r': ['1s^22s(^2S_{1/2})', '1s2s2p(^1P^0)(^2P^0_{1/2})'],
        # 1s^22s(^2S) - 1s2s2p(^3P)(^2P^0)
        's': ['1s^22s(^2S_{1/2})', '1s2s2p(^3P^0)(^2P^0_{3/2})'],
        't': ['1s^22s(^2S_{1/2})', '1s2s2p(^3P^0)(^2P^0_{1/2})'],
        # 1s^22s(^2S) - 1s2s2p(^4P^0)
        'u': ['1s^22s(^2S_{1/2})', '1s2s2p(^4P^0_{3/2})'],
        'v': ['1s^22s(^2S_{1/2})', '1s2s2p(^4P^0_{1/2})'],
    },
    'He-like':{
        # 1s^2(^1S) - 1s2p(^1P^0)
        'w': ['1s^2(^1S_{0})', '1s2p(^1P^0_{1})'],
        # 1s^2(^1S) - 1s2p(^3P^0)
        'x': ['1s^2(^1S_{0})', '1s2p(^3P^0_{2})'],
        'y': ['1s^2(^1S_{0})', '1s2p(^3P^0_{1})'],
        # 1s^2(^1S) - 1s2s(^3S)
        'z': ['1s^2(^1S_{0})', '1s2s(^3S_{1})'],
    }
}




dlines = {
    # --------------------------
    # Ar
    # --------------------------

    'ArXIV_n>3': {'q': 13, 'ION': 'ArXIV',
                  'symbol': 'n>3', 'lambda': 3.9495e-10,
                  'transition': ['unknown', 'unknwown'],
                  'source': 'adhoc'},

    'ArXV_1': {'q': 14, 'ION': 'ArXV',
               'symbol': '1', 'lambda': 4.0096e-10,
               'transition': ['1s2s^22p(^1P_1)', '1s^22s^2(^1S_0)'],
               'source': 'Kallne', 'innershell': True},
    'ArXV_2-1': {'q': 14, 'ION': 'ArXV',
                 'symbol': '2-1', 'lambda': 4.0176e-10,
                 'transition': ['1s2p^22s(^4P^3P_1)', '1s^22s2p(^3P_1)'],
                 'source': 'Kallne'},
    'ArXV_2-2': {'q': 14, 'ION': 'ArXV',
                 'symbol': '2-2', 'lambda': 4.0179e-10,
                 'transition': ['1s2s2p^2(^3D_1)', '1s^22s2p(^3P_0)'],
                 'source': 'Kallne'},
    'ArXV_2-3': {'q': 14, 'ION': 'ArXV',
                 'symbol': '2-3', 'lambda': 4.0180e-10,
                 'transition': ['1s2p^22s(^4P^3P_2)', '1s^22s2p(^3P_2)'],
                 'source': 'Kallne'},
    'ArXV_3': {'q': 14, 'ION': 'ArXV',
               'symbol': '3', 'lambda': 4.0192e-10,
               'transition': ['1s2s2p^2(^3D_2)', '1s^22s2p(^3P_1)'],
               'source': 'Kallne'},
    'ArXV_4': {'q': 14, 'ION': 'ArXV',
               'symbol': '4', 'lambda': 4.0219e-10,
               'transition': ['1s2s2p^2(^3D_5)', '1s^22s2p(^3P_2)'],
               'source': 'Kallne'},
    'ArXV_5': {'q': 14, 'ION': 'ArXV',
               'symbol': '5', 'lambda': 4.0291e-10,
               'transition': ['1s2s2p^2(^1D_5)', '1s^22s2p(^1P_1)'],
               'source': 'Kallne'},
    'ArXV_n3': {'q': 14, 'ION': 'ArXV',
                'symbol': 'n=3', 'lambda': 3.9550e-10,
                'transition': ['unknown', 'unknwown'],
                'source': 'adhoc'},


    'ArXVI_a_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9852e-10,
                       'transition': ('Li-like', 'a'),
                       'source': 'Kallne'},
    'ArXVI_a_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.98573e-10,
                     'transition': ('Li-like', 'a'),
                     'source': 'NIST'},
    'ArXVI_j_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9932e-10,
                       'transition': ('Li-like', 'j'),
                       'source': 'Kallne'},
    'ArXVI_j_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9938e-10,
                     'transition': ('Li-like', 'j'),
                     'source': 'NIST'},
    'ArXVI_k_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9892e-10,
                       'transition': ('Li-like', 'k'),
                       'source': 'Kallne', 'comment': 'Dielect. recomb. from ArXVII'},
    'ArXVI_k_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9898e-10,
                     'transition': ('Li-like', 'k'),
                     'source': 'NIST', 'comment': 'Dielect. recomb. from ArXVII'},
    'ArXVI_m_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9562e-10,
                       'transition': ('Li-like', 'm'),
                       'source': 'Kallne'},
    'ArXVI_m_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.96561e-10,
                     'transition': ('Li-like', 'm'),
                     'source': 'NIST'},
    'ArXVI_q_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9806e-10,
                       'transition': ('Li-like', 'q'),
                       'source': 'Kallne', 'innershell': True},
    'ArXVI_q_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9676e-10,
                     'transition': ('Li-like', 'q'),
                     'source': 'NIST', 'innershell': True},
    'ArXVI_r_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9827e-10,
                       'transition': ('Li-like', 'r'),
                       'source': 'Kallne'},
    'ArXVI_r_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9685e-10,
                     'transition': ('Li-like', 'r'),
                     'source': 'NIST'},
    'ArXVI_s_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9669e-10,
                       'transition': ('Li-like', 's'),
                       'source': 'Kallne'},
    'ArXVI_s_NIST': {'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9813e-10,
                     'transition': ('Li-like', 's'),
                     'source': 'NIST'},
    'ArXVI_t_Kallne': {'q': 15, 'ION': 'ArXVI',
                       'lambda': 3.9677e-10,
                       'transition': ('Li-like', 't'),
                       'source': 'Kallne'},
    'ArXVI_t_NIST': {'Z': 18, 'q': 15, 'ION': 'ArXVI',
                     'lambda': 3.9834e-10,
                     'transition': ('Li-like', 't'),
                     'source': 'NIST'},

    'ArXVII_w_Kallne': {'q': 16, 'ION': 'ArXVII',
                        'lambda': 3.9482e-10,
                        'transition': ('He-like', 'w'),
                        'source': 'Kallne'},
    'ArXVII_w_NIST': {'q': 16, 'ION': 'ArXVII',
                      'lambda': 3.94906e-10,
                      'transition': ('He-like', 'w'),
                      'source': 'NIST'},
    'ArXVII_x_Kallne': {'q': 16, 'ION': 'ArXVII',
                        'lambda': 3.9649e-10,
                        'transition': ('He-like', 'x'),
                        'source': 'Kallne'},
    'ArXVII_x_NIST': {'q': 16, 'ION': 'ArXVII',
                      'lambda': 3.965857e-10,
                      'transition': ('He-like', 'x'),
                      'source': 'NIST'},
    'ArXVII_y_Kallne': {'q': 16, 'ION': 'ArXVII',
                        'lambda': 3.9683e-10,
                        'transition': ('He-like', 'y'),
                        'source': 'Kallne'},
    'ArXVII_y_NIST': {'q': 16, 'ION': 'ArXVII',
                      'lambda': 3.969355e-10,
                      'transition': ('He-like', 'y'),
                      'source': 'NIST'},
    'ArXVII_z_Kallne': {'q': 16, 'ION': 'ArXVII',
                        'lambda': 3.9934e-10,
                        'transition': ('He-like', 'z'),
                        'source': 'Kallne'},
    'ArXVII_z_NIST': {'q': 16, 'ION': 'ArXVII',
                      'lambda': 3.99414e-10,
                      'transition': ('He-like', 'z'),
                      'source': 'NIST'},
    'ArXVII_T': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'T', 'lambda': 3.7544e-10,
                 'transition': [r'$2s2p(^1P_1)$', r'$1s2s(^1S_0)$'],
                 'source': 'Kallne'},
    'ArXVII_K': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'K', 'lambda': 3.7557e-10,
                 'transition': [r'$2p^2(^1D_2)$', r'$1s2p(^3P_2)$'],
                 'source': 'Kallne'},
    'ArXVII_Q': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'Q', 'lambda': 3.7603e-10,
                 'transition': [r'$2s2p(^3P_2)$', r'$1s2s(^3S_1)$'],
                 'source': 'Kallne'},
    'ArXVII_B': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'B', 'lambda': 3.7626e-10,
                 'transition': [r'$2p^2(^3P_2)$', r'$1s2p(^3P_1)$'],
                 'source': 'Kallne'},
    'ArXVII_R': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'R', 'lambda': 3.7639e-10,
                 'transition': [r'$2s2p(^3P_1)$', r'$1s2s(^3S_1)$'],
                 'source': 'Kallne'},
    'ArXVII_A': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'A', 'lambda': 3.7657e-10,
                 'transition': [r'$2p^2(^3P_2)$', r'$1s2p(^3P_2)$'],
                 'source': 'Kallne'},
    'ArXVII_J': {'q': 16, 'ION': 'ArXVII',
                 'symbol': 'J', 'lambda': 3.7709e-10,
                 'transition': [r'$2p^2(^1D_2)$', r'$1s2p(^1P_1)$'],
                 'source': 'Kallne'},

    'ArXVIII_W1': {'q': 17, 'ION': 'ArXVIII',
                   'symbol': 'W_1', 'lambda': 3.7300e-10,
                   'transition': [r'$2p(^2P_{3/2})$', r'$1s(^2S_{1/2})$'],
                   'source': 'Kallne'},
    'ArXVIII_W2': {'q': 17, 'ION': 'ArXVIII',
                   'symbol': 'W_2', 'lambda': 3.7352e-10,
                   'transition': [r'$2p(^2P_{1/2})$', r'$1s(^2S_{1/2})$'],
                   'source': 'Kallne'},

    # --------------------------
    # Fe
    # --------------------------

    'FeXXIII_beta': {'q': 22, 'ION': 'FeXXIII',
                     'symbol': 'beta', 'lambda': 1.87003e-10,
                     'transition': [r'$1s^22s^2(^1S_0)$',
                                    r'$1s2s^22p(^1P_1)$'],
                     'source': 'Bitter'},

    'FeXXIV_t_Bitter': {'q': 23, 'ION': 'FeXXIV',
                        'lambda': 1.8566e-10,
                        'transition': ('Li-like', 't'),
                        'source': 'Bitter'},
    'FeXXIV_q_Bitter': {'q': 23, 'ION': 'FeXXIV',
                        'lambda': 1.8605e-10,
                        'transition': ('Li-like', 'q'),
                        'source': 'Bitter'},
    'FeXXIV_k_Bitter': {'q': 23, 'ION': 'FeXXIV',
                        'lambda': 1.8626e-10,
                        'transition': ('Li-like', 'k'),
                        'source': 'Bitter'},
    'FeXXIV_r_Bitter': {'q': 23, 'ION': 'FeXXIV',
                        'lambda': 1.8631e-10,
                        'transition': ('Li-like', 'r'),
                        'source': 'Bitter'},
    'FeXXIV_j_Bitter': {'q': 23, 'ION': 'FeXXIV',
                        'lambda': 1.8654e-10,
                        'transition': ('Li-like', 'j'),
                        'source': 'Bitter'},

    'FeXXV_w_Bitter': {'q': 24, 'ION': 'FeXXV',
                       'lambda': 1.8498e-10,
                       'transition': ('He-like', 'w'),
                       'source': 'Bitter'},
    'FeXXV_x_Bitter': {'q': 24, 'ION': 'FeXXV',
                       'lambda': 1.85503e-10,
                       'transition': ('He-like', 'x'),
                       'source': 'Bitter'},
    'FeXXV_y_Bitter': {'q': 24, 'ION': 'FeXXV',
                       'lambda': 1.8590e-10,
                       'transition': ('He-like', 'y'),
                       'source': 'Bitter'},
    'FeXXV_z_Bitter': {'q': 24, 'ION': 'FeXXV',
                       'lambda': 1.8676e-10,
                       'transition': ('He-like', 'z'),
                       'source': 'Bitter'},

    # --------------------------
    # W
    # --------------------------

    'WXLIV_0_NIST': {'q': 43, 'ION': 'WXLIV',
                     'symbol':'0', 'lambda': 3.9635e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3d^94s^24p(3/2,1/2)^0_16f(1,5/2)3/2'],
                     'source': 'NIST'},
    'WXLIV_1_NIST': {'q': 43, 'ION': 'WXLIV',
                     'symbol':'1', 'lambda': 3.9635e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3d^94s^24p(3/2,1/2)^0_26f(2,5/2)1/2'],
                     'source': 'NIST'},
    'WXLIV_2_NIST': {'q': 43, 'ION': 'WXLIV',
                     'symbol':'2', 'lambda': 4.017e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3p^53d^{10}4s^24p(3/2,1/2)_25d(2,5/2)3/2'],
                     'source': 'NIST'},
    'WXLIV_3_NIST': {'q': 43, 'ION': 'WXLIV',
                     'symbol':'3', 'lambda': 4.017e-10,
                     'transition': ['3d^{10}4s^24p(^2P^0_{1/2})',
                                    '3p^53d^{10}4s^24p(3/2,1/2)_25d(2,5/2)1/2'],
                     'source': 'NIST'},
    'WXLV_0_NIST': {'q': 44, 'ION': 'WXLV',
                    'symbol':'0', 'lambda': 3.9730e-10,
                    'transition': ['3d^{10}4s^2(^1S_{0})',
                                   '3p^5(^2P^0_{3/2})3d^{10}4s^25d(3/2,5/2)^01'],
                    'source': 'NIST'},
    'WXLV_1_NIST': {'q': 44, 'ION': 'WXLV',
                    'symbol':'1', 'lambda': 3.9895e-10,
                    'transition': ['3d^{10}4s^2(^1S_{0})',
                                   '3d^9(^2D_{5/2})4s^26f(5/2,7/2)^01'],
                    'source': 'NIST'},
    'WLIII_0_NIST': {'q': 52, 'ION': 'WLIII',
                    'symbol':'0', 'lambda': 4.017e-10,
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
        trans = dtransitions[v0['transition'][0]][v0['transition'][1]]
        dlines[k0]['symbol'] = v0['transition'][1]
        dlines[k0]['transition'] = trans
