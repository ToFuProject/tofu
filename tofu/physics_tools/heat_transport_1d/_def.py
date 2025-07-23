

# ########################################################
# ########################################################
#           DEFAULT MATERIAL PROPERTIES
# ########################################################


_DMAT = {

    # Gold
    'Au': {
        'kappa': {
            'data': 315,
            'units': 'W/m/K',
            'name': 'heat conductivity',
        },
        'rho': {
            'data': 19.3e3,
            'units': 'kg/m3',
            'name': 'mass density',
        },
        'c': {
            'data': 126,
            'units': 'J/kg/K',
            'name': 'specific heat capacity',
        },
        'Tfus': {
            'data': 1064,
            'units': 'C',
            'name': 'fusion temperature',
        },
    },

    # Silicon Nitride
    'Si3N4': {
        'kappa': {
            'data': 20,
            'units': 'W/m/K',
            'name': 'heat conductivity',
        },
        'rho': {
            'data': 3.17e3,
            'units': 'kg/m3',
            'name': 'mass density',
        },
        'c': {
            'data': 700,
            'units': 'J/kg/K',
            'name': 'specific heat capacity',
        },
        'Tfus': {
            'data': 1900,
            'units': 'C',
            'name': 'fusion temperature',
        },
    },
}


# ##################################################
# ##################################################
#               Complement
# ##################################################


for k0, v0 in _DMAT.items():
    _DMAT[k0]['gamma'] = {
        'data': v0['kappa']['data'] / (v0['c']['data'] * v0['rho']['data']),
        'units': 'W/m2',
        'name': 'thermal diffusivity',
    }
