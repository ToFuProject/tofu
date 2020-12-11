
import os
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tofu as tf



# -----------------------------------------------------------------------------
#                   Default parameters
# -----------------------------------------------------------------------------


_HERE = os.path.abspath(os.path.dirname(__file__))
_PFE = os.path.join(_HERE, 'WEST_Geometry_Rev_Nov2016_V2.xlsx')

# Toroidal width of a sector
_DTHETA = 2*np.pi / 18.

# 4 corners of a Casing Cover PJ from Creoview
_PJ_PTS = np.array([[313.728, -988.400, 2255.087],
                    [476.477, -988.400, 2226.390],
                    [313.728, -988.400, 1938.128],
                    [368.071, -988.400, 1928.547]])
_PJ_THETA = np.arctan2(_PJ_PTS[:, 2], _PJ_PTS[:, 0])
_PJ_DTHETA = np.abs(0.5*((_PJ_THETA[1]-_PJ_THETA[0])
                         + (_PJ_THETA[3]-_PJ_THETA[2])))
_PJ_POS0 = (0.5*((_PJ_THETA[1]+_PJ_THETA[0])/2 + (_PJ_THETA[3]+_PJ_THETA[2])/2)
            - _DTHETA)
_PJ_POS = (_PJ_POS0
           + np.r_[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]*_DTHETA)
_CASING_DTHETA = [_DTHETA - _PJ_DTHETA, 2*_DTHETA - _PJ_DTHETA]
_CASING_POS = [_PJ_POS0 + _DTHETA/2 + np.arange(6)*3*_DTHETA,
               _PJ_POS0 + 2*_DTHETA + np.arange(6)*3*_DTHETA]
_CASING_DTHETA = np.tile(_CASING_DTHETA, 6)
_CASING_POS = np.array(_CASING_POS).T.ravel()


_EXP = 'WEST'
_DNAMES = {
    'Baffle': {'name': None},
    'IVPP HFS': {'name': None},
    'IVPP LFS': {
        'name': 'ThermalShieldLFSV0',
        'class': 'PFC',
        'thick': 0.008,
        'save': True,
    },
    'Inner VV': {
        'name': 'Inner00V0',
        'class': 'Ves',
        'save': True,
    },
    'LDiv CasingCover': {
        'name': 'CasingCoverLDivV0',
        'class': 'PFC',
        'save': True,
    },
    'LDiv PFUs': {'name': None},
    'LPA': {'name': None},
    'Ldiv Casing': {
        'name': 'CasingLDivV0',
        'class': 'PFC',
        'extent': _CASING_DTHETA,
        'pos': _CASING_POS,
        'save': True,
    },
    'Ldiv Casing PJ': {
        'name': 'CasingPJLDivV0',
        'class': 'PFC',
        'extent': _PJ_DTHETA,
        'pos': _PJ_POS,
        'save': True,
    },
    'Ldiv PFU Plate': {
        'name': 'CasingPFUPlateLDivV0',
        'class': 'PFC',
        'save': True,
    },
    'Outer VV': {
        'name': 'Inner01V0',
        'class': 'Ves',
        'save': True,
    },
    'UDiv CasingCover': {
        'name': 'CasingCoverUDivV0',
        'class': 'PFC',
        'save': True,
    },
    'UDiv PFUs': {'name': None},
    'Udiv Casing': {
        'name': 'CasingUDivV0',
        'class': 'PFC',
        'extent': _CASING_DTHETA,
        'pos': _CASING_POS,
        'save': True,
    },
    'Udiv Casing PJ': {
        'name': 'CasingPJUDivV0',
        'class': 'PFC',
        'save': True,
    },
    'Udiv PFU Plate': {
        'name': 'CasingPFUPlateUDivV0',
        'class': 'PFC',
        'save': True,
    },
    'VDE': {'name': None},
}


# -----------------------------------------------------------------------------
#           Extract and plot geometry
# -----------------------------------------------------------------------------

def get_all(pfe=None, dnames=None, ax=None, plot=None, save=None):

    #--------------
    #   Plot
    if pfe is None:
        pfe = _PFE
    if dnames is None:
        dnames = _DNAMES
    if plot is None:
        plot = True
    if save is None:
        save = False


    #--------------
    #   Extract
    out = pd.read_excel(pfe, sheet_name='Main', header=[0,1])

    ls = list(out.columns.levels[0])
    for ss in ls:
        nn = ss.split('\n')[0]
        if nn not in dnames.keys():
            continue
        poly = np.array([out[ss]['R (m)'], out[ss]['Z (m)']])
        if 'thick' in dnames[nn].keys():
            dv = poly[:, 1:] - poly[:, :-1]
            vout = np.array([dv[1, :], -dv[0, :]])
            if np.mean(vout[0, :]) < 0:
                vout = -vout
            vout = np.concatenate((vout[:, 0:1], vout), axis=1)
            vout = vout / np.sqrt(np.sum(vout**2, axis=0))[None, :]
            poly = np.concatenate(
                (poly,
                 poly[:, ::-1] + dnames[nn]['thick']*vout[:, ::-1]),
                axis=1
            )
        dnames[nn]['poly'] = poly
        indok = ~np.any(np.isnan(dnames[nn]['poly']), axis=0)
        if 'class' in dnames[nn].keys():
            try:
                dnames[nn]['obj'] = getattr(tf.geom, dnames[nn]['class'])(
                    Name=dnames[nn]['name'],
                    Poly=dnames[nn]['poly'][:, indok],
                    Exp=_EXP,
                )
            except Exception as err:
                msg = (str(err)
                       + "\ntofu object {} failed".format(nn))
                warnings.warn(str(err))

    #--------------
    #   Plot
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], aspect='equal')

    for nn in dnames.keys():
        ll, = ax.plot(dnames[nn]['poly'][0, :], dnames[nn]['poly'][1, :],
                      label=nn)
        if dnames[nn].get('obj') is not None:
            ax = dnames[nn]['obj'].plot(lax=ax, proj='cross',
                                        element='P', indices=False)

    ax.legend()

    #--------------
    #   save
    if save is True:
        for nn in dnames.keys():
            c0 = (dnames[nn].get('save') is True
                  and dname[nn].get('obj') is not None)
            if c0:
                dname[nn]['obj'].save_to_txt(path=_HERE)
    return dnames, ax
