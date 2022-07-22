

import os


import numpy as np
import scipy.io as scpio


import tofu as tf


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUTS = os.path.dirname(_PATH_HERE)
_PATH_SAVE = _PATH_INPUTS


# #############################################################################
# #############################################################################
#                       routines
# #############################################################################


def extract(save=False):

    lf = [
        ff for ff in os.listdir(_PATH_HERE)
        if ff.endswith('.mat')
        and 'coordinates' if ff
        and any([ss in ff for ss in ['vessel', 'limiter']])
        and ff.startswith('COMPASS')
    ]
    if len(lf) == 0:
        return

    # ----------------
    # Extract all data

    dout = {'Ves': {}, 'PFC': {}}
    for ff in lf:

        pfe = os.path.join(_PATH_HERE, ff)
        out = scpio.loadmat(pfe)

        if 'vessel' in ff:
            kR, kZ = 'R1', 'Z1'
            name = 'InnerV1'
        else:
            kR, kZ = 'R', 'Z'
            name = 'V0'

        R = out[kR].ravel()
        Z = out[kZ].ravel()

        dout['Ves'][name] = tf.geom.Ves(
            Poly=np.array([R, Z]),
            Name=name,
            Exp='COMPASS',
            SavePath=_PATH_SAVE,
        )

    # ---------------
    # Derive PFCs

    dind = {
        'lower': {
            'V0': np.arange(129, 194),
            'InnerV1': np.arange(9, 20)[::-1],
        },
        'upper': {
            'V0': np.arange(39, 61),
            'InnerV1': np.arange(36, 46)[::-1],
        },
        'inner': {
            'V0': np.arange(72, 119),
            'InnerV1': np.r_[4, 3, 2, 1, 0, 51, 50],
        },
        'outer': {
            'V0': np.r_[np.arange(197, 231), np.arange(0, 35)],
            'InnerV1': np.arange(21, 33)[::-1],
        },
    }
    for k0, v0 in dind.items():
        poly = np.concatenate(
            (
                dout['Ves']['V0'].Poly[:, v0['V0']],
                dout['Ves']['InnerV1'].Poly[:, v0['InnerV1']],
            ),
            axis=1,
        )
        dout['PFC'][k0] = tf.geom.PFC(
            Poly=poly,
            Name=k0,
            Exp='COMPASS',
            SavePath=_PATH_SAVE,
        )

    # ---------------
    # Format output

    if save:
        for cc in dout.keys():
            for nn in dout[cc].keys():
                dout[cc][nn].save_to_txt(path=None)

    return dout
