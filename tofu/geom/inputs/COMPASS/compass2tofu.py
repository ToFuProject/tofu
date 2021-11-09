

import os


import numpy as np
import scipy.io as scpio


_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUTS = os.path.dirname(_PATH_HERE)



# #############################################################################
# #############################################################################
#                       routines
# #############################################################################


def extract(pfe=None):

    if not pfe.endswith('.mat'):
        return

    out = scpio.loadmat(pfe)

    fname = os.path.split(pfe)[1][:-4]

    dout = None
    if fname in ['wall_01', 'wall_02', 'wall_03']:
        R = out['wall'][0, 0][0].ravel()
        Z = out['wall'][0, 0][1].ravel()
        vv = int(fname[-2:])

        poly = np.array(R, Z)
        cls = 'Ves'
        name = f'V{vv}'

    elif all([ss in fname for ss in ['COMPASS', 'coordinates']]):
        kR, kZ = ('R', 'Z') if 'limiter' in fname else ('R1', 'Z1')
        R = out[kR].ravel()
        Z = out[kZ].ravel()

        if 'vessel' in fname:
            dout = {
                'Ves': {
                    'Name': 'InnerV1',
                    'poly': np.array(R, Z),
                },
            }

        else:

            # First extract limiter as vessel V0
            dout = {
                'Ves': {
                    'Name': 'V0',
                    'poly': np.array(R, Z),
                },
            }

            # Then extract PFC from limiter / vessel
            ind = []


    elif 'magnetics' in fname:
        # TBC
        pass

    elif fname == 'TK3X_D_8871':
        # TBC
        pass

    else:
        msg = "Unknown file"
        raise Exception(msg)

    # ---------------
    # Format output

    if dout is not None:
        for cc in dout.keys():
            for nn in dout[cc].keys();
                dout[cc][nn]['Exp'] = 'COMPASS'

    return dout
