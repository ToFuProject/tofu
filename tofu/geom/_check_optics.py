

import inspect
import warnings


import numpy as np
import matplotlib as mpl


from . import _comp_optics


# ####################################################################
# ####################################################################
#               common routines for checking
# ####################################################################
# ####################################################################


def _check_dict_valid_keys(var=None, varname=None, valid_keys=None):
    c0 = (
        isinstance(var, dict)
        and all([
            isinstance(ss, str) and ss in valid_keys for ss in var.keys()
        ])
    )
    if not c0:
        msg = (
            """
            Arg {} must be:
                - dict
                - with valid keys in {}

            Provided:
                - type: {}
                - keys: {}
            """.format(
                varname,
                valid_keys,
                type(var),
                sorted(var.keys()) if isinstance(var, dict) else None)
        )
        raise Exception(msg)


def _check_flat1darray_size(var=None, varname=None, size=None, norm=None):

    # Default inputs
    if norm is None:
        norm = False

    # Return None of None
    if var is None:
        return None

    # Format to flat 1d array and check size
    var = np.atleast_1d(var).ravel()
    if var.size != size:
        msg = (
            """
            Var {} should be convertible to a 1d np.ndarray of size {}

            Provided:
                {}
            """.format(varname, size, var)
        )
        raise Exception(msg)

    # Normalize ?
    if norm is True:
        var = var / np.linalg.norm(var)
    return var


def _check_orthornormaldirect(e1=None, e2=None, var=None, varname=None):
    c0 = (
        np.abs(np.sum(e1*var)) < 1.e-12
        and np.abs(np.sum(e2*var)) < 1.e-12
        and np.linalg.norm(np.cross(e1, e2) - var) < 1.e-12
    )
    if not c0:
        msg = (
            """
            Unit vector basis (e1, e2, {0}) must be orthonormal direct!

            Provided:
                - norm(e1.{0}) = {}
                - norm(e2.{0}) = {}
                - norm((e1xe2) - {0}) = {}
            """.format(
                varname,
                np.abs(np.sum(e1*var)),
                np.abs(np.sum(e2*var)),
                np.linalg.norm(np.cross(e1, e2) - var),
            )
        )
        raise Exception(msg)


def _check_dict_unitvector(dd=None, dd_name=None):

    # Check all unit vectors
    for k0 in ['nout', 'nin', 'e1', 'e2']:
        dd[k0] = _check_flat1darray_size(
            var=dd.get(k0), varname=k0, size=3, norm=True)

    # Check consistency between unit vectors
    if dd['e1'] is not None:
        c0 = (
            dd['e2'] is not None
            and np.abs(np.sum(dd['e1']*dd['e2'])) < 1.e-12
        )
        if not c0:
            msg = (
                """
                If {0}['e1'] is provided, then:
                    - {0}['e2'] must be provided too
                    - {0}['e1'] must be perpendicular to {0}['e2']

                Provided:
                    {1}
                """.format(dd_name, dd['e2'])
            )
            raise Exception(msg)

        if dd['nout'] is not None:
            _check_orthornormaldirect(
                e1=dd['e1'], e2=dd['e2'],
                var=dd['nout'], varname='nout',
            )
        if dd['nin'] is not None:
            _check_orthornormaldirect(
                e1=dd['e1'], e2=dd['e2'],
                var=-dd['nin'], varname='-nin',
            )


# ####################################################################
# ####################################################################
#               check dgeom
# ####################################################################
# ####################################################################


def _checkformat_dgeom(dgeom=None, ddef=None, valid_keys=None):
    """
    Check the content of dgeom

    center, summit and rcurve

    extenthalf

    basis of unit vectors (e1, e2, nout) + nin
    """

    if dgeom is None:
        return

    # --------------------
    # Check dict integrity
    # --------------------

    # Check dict type and content (each key is a valid str)
    _check_dict_valid_keys(var=dgeom, varname='dgeom', valid_keys=valid_keys)

    # Set default values if any
    for kk in ddef.keys():
        dgeom[kk] = dgeom.get(kk, ddef[kk])

    # Set default values to None of not provided
    for kk in valid_keys:
        dgeom[kk] = dgeom.get(kk, None)

    # ------------------------------
    # Check each value independently
    # ------------------------------

    # Complementarity (center, rcurve) <=> (summit, rcurve)
    dgeom['center'] = _check_flat1darray_size(
        var=dgeom.get('center'), varname='center', size=3)
    dgeom['summit'] = _check_flat1darray_size(
        var=dgeom.get('summit'), varname='summit', size=3)

    lc = [dgeom['center'] is not None, dgeom['summit'] is not None]
    if not any(lc):
        msg = "Please provide at least dgeom['center'] and/or dgeom['summit']"
        raise Exception(msg)

    if dgeom['rcurve'] is None:
        msg = "Arg dgeom['rcurve'] must be convertible to a float"
        raise Exception(msg)
    dgeom['rcurve'] = float(dgeom['rcurve'])

    # Check dimensions (half extent)
    dgeom['extenthalf'] = _check_flat1darray_size(
        var=dgeom.get('extenthalf'), varname='extenthalf', size=2)

    # ----------------------------------
    # Add missing vectors and parameters
    # ----------------------------------

    if dgeom['e1'] is not None:
        if dgeom['nout'] is None:
            dgeom['nout'] = np.cross(dgeom['e1'], dgeom['e2'])
        if dgeom['nin'] is None:
            dgeom['nin'] = -dgeom['nout']
        if dgeom['center'] is None:
            dgeom['center'] = (dgeom['summit']
                               + dgeom['nin']*dgeom['rcurve'])
        if dgeom['summit'] is None:
            dgeom['summit'] = (dgeom['center']
                               + dgeom['nout']*dgeom['rcurve'])
    elif dgeom['center'] is not None and dgeom['summit'] is not None:
        if dgeom['nout'] is None:
            nout = (dgeom['summit'] - dgeom['center'])
            dgeom['nout'] = nout / np.linalg.norm(nout)
        if dgeom['e1'] is None:
            dgeom['e1'] = np.cross(np.r_[0, 0, 1], dgeom['nout'])
            dgeom['e1'] = dgeom['e1'] / np.linalg.norm(dgeom['e1'])
            msg = (
                "dgeom['e1'] was not provided!\n"
                + "  => setting e1 to horizontal by default!\n"
            )
            warnings.warn(msg)
        if dgeom['e2'] is None:
            dgeom['e2'] = np.cross(dgeom['nout'], dgeom['e1'])
            dgeom['e2'] = dgeom['e2'] / np.linalg.norm(dgeom['e2'])

    if dgeom['extenthalf'] is not None:
        if dgeom['Type'] == 'sph' and dgeom['Typeoutline'] == 'rect':
            ind = np.argmax(dgeom['extenthalf'])
            dphi = dgeom['extenthalf'][ind]
            sindtheta = np.sin(dgeom['extenthalf'][ind-1])
            dgeom['surface'] = 4.*dgeom['rcurve']**2*dphi*sindtheta

    # Check orthonormal direct basis
    _check_dict_unitvector(dd=dgeom, dd_name='dgeom')

    return dgeom


# ####################################################################
# ####################################################################
#               check dmat
# ####################################################################
# ####################################################################


def _checkformat_dmat(dmat=None, dgeom=None, ddef=None, valid_keys=None):
    """
    Check the content of dmat

    Crystal parameters : d, formula, density, lengths, angles, cut

    New basis of unit vectors due to miscut (e1, e2, nout)
    + nin + alpha, beta
    """

    if dmat is None:
        return

    # ---------------------
    # check dict integrity
    # ---------------------

    # Check dict typeand content (each key is a valid string)
    _check_dict_valid_keys(var=dmat, varname='dmat', valid_keys=valid_keys)

    # Set default values if any
    for kk in ddef.keys():
        dmat[kk] = dmat.get(kk, ddef[kk])

    # -------------------------------
    # check each value independently
    # -------------------------------

    # Check dimension of array and its size
    dmat['lengths'] = _check_flat1darray_size(
        var=dmat.get('lengths'), varname='lengths', size=3)
    dmat['angles'] = _check_flat1darray_size(
        var=dmat.get('angles'), varname='angles', size=3)

    if dmat['d'] is None:
        msg = "Arg dmat['d'] must be convertible to a float."
        raise Exception(msg)
    dmat['d'] = float(dmat['d'])

    if dmat['density'] is None:
        msg = "Arg dmat['density'] must be convertible to a float."
        raise Exception(msg)
    dmat['density'] = float(dmat['density'])

    if isinstance(dmat['formula'], str) is False:
        msg = (
            """
            Var {} must be a valid string.

            Provided
                - type: {}
            """.format('formula', type(dmat['formula']))
        )
        raise Exception(msg)

    if dmat.get('cut') is not None:
        dmat['cut'] = np.atleast_1d(dmat['cut']).ravel().astype(int)
        if dmat['cut'].size == 3:
            pass
        elif dmat['cut'].size == 4:
            pass
        else:
            msg = (
                "Var 'cut' should be convertible to a 1d array of size 3 or 4"
                f"\nProvided: {dmat.get('cut')}"
            )
            raise Exception(msg)

    # Check orthonormal direct basis
    _check_dict_unitvector(dd=dmat, dd_name='dmat')

    # Check all additionnal angles to define the new basis
    for k0 in ['alpha', 'beta']:
        dmat[k0] = _check_flat1darray_size(
            var=dmat.get(k0), varname=k0, size=1, norm=False)

    # -------------------------------------------------------------
    # Add missing vectors and parameters according to the new basis
    # -------------------------------------------------------------

    if all([dgeom[kk] is not None for kk in ['nout', 'e1', 'e2']]):

        # dict of value, comment, default value and type of
        #  alpha and beta angles in dmat
        dpar = {
            'alpha': {
                # 'alpha': alpha,
                'com': 'miscut amplitude',
                'default': 0.,
                'type': float,
            },
            'beta': {
                # 'beta': beta,
                'com': 'miscut orientation',
                'default': 0.,
                'type': float,
            },
        }

        # setting to default value if any is None
        lparNone = [aa for aa in dpar.keys() if dmat.get(aa) is None]

        # if any is None, assigning default value and send a warning message
        if len(lparNone) > 0:
            msg = "The following parameters were set to their default values:"
            for aa in lparNone:
                dmat[aa] = dpar[aa]['default']
                msg += "\n\t - {} = {} ({})".format(
                    aa, dpar[aa]['default'], dpar[aa]['com'],
                )
            warnings.warn(msg)

        # check conformity of type of angles
        lparWrong = []
        for aa in dpar.keys():
            try:
                dmat[aa] = float(dmat[aa])
            except Exception as err:
                lparWrong.append(aa)

        if len(lparWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for aa in lparWrong:
                msg += "\n\t - {} = {} ({})".format(
                    aa, dpar[aa]['type'], type(dmat[aa]),
                )
            raise Exception(msg)

        # Check value of alpha
        if dmat['alpha'] < 0 or dmat['alpha'] > np.pi/2:
            msg = (
                "Arg dmat['alpha'] must be an angle (radians) in [0; pi/2]"
                + "\nProvided:\n\t{}".format(dmat['alpha'])
            )
            raise Exception(msg)

        if dmat['beta'] < -np.pi or dmat['beta'] > np.pi:
            msg = (
                "Arg dmat['beta'] must be an angle (radians) in [-pi; pi]"
                + "\nProvided:\n\t{}".format(dmat['beta'])
            )
            raise Exception(msg)

        # dict of value, comment, default value and type of unit vectors in
        #  dmat, computting default value of unit vectors, corresponding to
        #  dgeom values if angles are 0.
        dvec = {
            'e1': {
                # 'e1': e1,
                'com': 'unit vector (miscut)',
                'default': (
                    np.cos(dmat['alpha'])*(
                        np.cos(dmat['beta'])*dgeom['e1']
                        + np.sin(dmat['beta'])*dgeom['e2']
                    )
                    - np.sin(dmat['alpha'])*dgeom['nout']
                ),
                'type': float,
            },
            'e2': {
                # 'e2': e2,
                'com': 'unit vector (miscut)',
                'default': (
                    np.cos(dmat['beta'])*dgeom['e2']
                    - np.sin(dmat['beta'])*dgeom['e1']
                ),
                'type': float,
            },
            'nout': {
                # 'nout': nout,
                'com': 'outward unit vector (normal to non-parallel mesh)',
                'default': (
                    np.cos(dmat['alpha'])*dgeom['nout']
                    + np.sin(dmat['alpha']) * (
                        np.cos(dmat['beta'])*dgeom['e1']
                        + np.sin(dmat['beta'])*dgeom['e2']
                    )
                ),
                'type': float,
            }
        }

        # setting to default value if any is None
        lvecNone = [bb for bb in dvec.keys() if dmat.get(bb) is None]

        # if any is None, assigning default value
        if len(lvecNone) > 0:
            for bb in lvecNone:
                dmat[bb] = dvec[bb]['default']

        # check conformity of type of unit vectors
        lvecWrong = []
        for bb in dvec.keys():
            try:
                dmat[bb] = np.atleast_1d(dmat[bb]).ravel().astype(float)
                dmat[bb] = dmat[bb] / np.linalg.norm(dmat[bb])
                assert np.allclose(dmat[bb], dvec[bb]['default'])
            except Exception as err:
                lvecWrong.append(bb)

        if len(lvecWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for bb in lvecWrong:
                msg += "\n\t - {} = {} ({})".format(
                    bb, dvec[bb]['type'], type(dmat[bb]),
                )
            msg += "\nAnd (nout , e1, e2) must be an orthonormal direct basis"
            raise Exception(msg)

        # Computation of unit vector nin
        dmat['nin'] = -dmat['nout']

    return dmat


# ####################################################################
# ####################################################################
#               check dbragg
# ####################################################################
# ####################################################################


def _checkformat_dbragg(dbragg=None, ddef=None, valid_keys=None, dmat=None):

    """
    rocking curve, wavelenght of reference and its bragg angle associated
    """
    if dbragg is None:
        dbragg = dict.fromkeys(valid_keys)

    # ---------------------
    # check dict integrity
    # ---------------------

    # Check type dict and content (each key is a valid string)
    _check_dict_valid_keys(var=dbragg, varname='dbragg', valid_keys=valid_keys)

    # Set default values when necessary
    for kk in ddef.keys():
        dbragg[kk] = dbragg.get(kk, ddef[kk])

    # ------------------------------------------------
    # Check braggref and lambref type and computation
    # ------------------------------------------------

    # Check braggref
    ltypes = [int, float, np.int_, np.float64]
    c0 = bool(
        type(dbragg.get('braggref')) in ltypes
        and dbragg['braggref'] >= 0
        and dbragg['braggref'] <= np.pi/2.
    )
    if not c0:
        msg = (
            """
            Var {} is not valid!
            Value should be in [0; pi/2]

            Provided: {}
            """.format('braggref', dbragg['braggref'])
            )
        raise Exception(msg)

    # Set default lambda if necessary
    user_prov = True
    if dbragg.get('lambref') is None:
        # set to bragg = braggref
        dbragg['lambref'] = _comp_optics.get_lamb_from_bragg(
            np.r_[dbragg['braggref']],
            dmat['d'],
            n=1,
        )[0]
        user_prov = False

    # Set default bragg angle if necessary
    braggref = _comp_optics.get_bragg_from_lamb(
                            np.r_[dbragg['lambref']], dmat['d'], n=1)[0]
    if np.isnan(braggref):
        lambok = []
        msg = (
            """
            Var {} is not valid!
            Please check your arguments to calculate the Bragg's law correctly!
            Provided:
                - crystal inter-plane d [m] = {}
                - wavelenght interval [m] : {}
                - lambref = {}
            """.format('lambref', dmat['d'], lambok, dbragg['lambref'])
        )
        raise Exception(msg)

    # Update braggref according to lambref, if lambref was user-provided
    if user_prov is True:
        dbragg['braggref'] = braggref

    # ------------------------------------------------
    # Check rocking curve value
    # ------------------------------------------------

    # Check type dict and content (each key is a valid string)
    drock = dbragg.get('rockingcurve')
    if drock is not None:
        lkeyok = [
            'sigma', 'deltad', 'Rmax', 'dangle', 'lamb',
            'value', 'type', 'source',
        ]
        _check_dict_valid_keys(var=drock, varname='drock', valid_keys=lkeyok)

        # check type, size and content of each key in drock
        # Rocking curve can be provided as:
        # - analytical form (Lorentzian-log)
        # - tabulated in 2d
        # - tabulated in 1d
        try:
            if drock.get('sigma') is not None:
                dbragg['rockingcurve']['sigma'] = float(drock['sigma'])
                dbragg['rockingcurve']['deltad'] = float(
                    drock.get('deltad', 0.),
                )
                dbragg['rockingcurve']['Rmax'] = float(drock.get('Rmax', 1.))
                dbragg['rockingcurve']['type'] = 'lorentz-log'

            elif drock.get('dangle') is not None:
                c2d = (drock.get('lamb') is not None
                       and drock.get('value').ndim == 2)
                if c2d:
                    if drock['value'].shape != (drock['dangle'].size,
                                                drock['lamb'].size):
                        msg = (
                            """ Tabulated 2d rocking curve should be:
                                shape = (dangle.size, lamb.size)
                            """)
                        raise Exception(msg)
                    dbragg['rockingcurve']['dangle'] = np.r_[drock['dangle']]
                    dbragg['rockingcurve']['lamb'] = np.r_[drock['lamb']]
                    dbragg['rockingcurve']['value'] = drock['value']
                    dbragg['rockingcurve']['type'] = 'tabulated-2d'

                else:
                    if drock.get('lamb') is None:
                        msg = (
                            """Please also specify the lamb for which
                                the rocking curve was tabulated""")
                        raise Exception(msg)
                    dbragg['rockingcurve']['lamb'] = float(drock['lamb'])
                    dbragg['rockingcurve']['dangle'] = np.r_[drock['dangle']]
                    dbragg['rockingcurve']['value'] = np.r_[drock['value']]
                    dbragg['rockingcurve']['type'] = 'tabulated-1d'

                if drock.get('source') is None:
                    msg = "Unknown source for the tabulated rocking curve!"
                    warnings.warn(msg)
                dbragg['rockingcurve']['source'] = drock.get('source')

        except Exception as err:
            msg = (
                """
                Provide the rocking curve as a dictionnary with either:
                    - parameters of a lorentzian in log10:
                      'sigma': float, 'deltad':float, 'Rmax': float
                    - tabulated (dangle, value) with source (url...):
                      'dangle': np.darray, 'value': np.darray, 'source': str
                """
            )
            raise Exception(msg)
    else:
        dbragg['rockingcurve'] = None
    return dbragg


# ####################################################################
# ####################################################################
#               check colors
# ####################################################################
# ####################################################################


def _checkformat_inputs_dmisc(color=None, ddef=None):
    if color is None:
        color = mpl.colors.to_rgba(ddef['dmisc']['color'])
    assert mpl.colors.is_color_like(color)
    return tuple(mpl.colors.to_rgba(color))


# ####################################################################
# ####################################################################
#               check synthetic diags
# ####################################################################
# ####################################################################


def _check_config_get_Ves(config=None, struct=None):

    if config.__class__.__name__ != 'Config':
        msg = (
            "Arg config must be a Config object "
            f"Provided:\n\t- config: {type(config)}"
        )
        raise Exception(msg)

    lok = list(config.dStruct['dObj']['Ves'].keys())
    if struct is None and len(lok) == 1:
        struct = lok[0]
    elif struct not in lok:
        msg = (
            "Arg struct must be the name of a StructIn in config!\n"
            f"Provided:\n\t- Available: {lok}\n\t- struct: {struct}"
        )
        raise Exception(msg)
    return struct


def _check_calc_signal_from_emissivity(
    emis=None,
    config=None,
    struct=None,
    lamb=None,
    det=None,
    binning=None,
):

    # config
    struct = _check_config_get_Ves(config=config, struct=struct)

    # lamb
    lamb = np.atleast_1d(lamb).ravel()

    # det
    assert det is not None

    # emis
    if not callable(emis):
        msg = "Arg emis must be a callable!"
        raise Exception(msg)

    argspec = inspect.getfullargspec(emis)
    c0 = (
        all([ss in argspec.args for ss in ['r', 'z', 'phi', 'lamb', 't']])
    )
    if not c0:
        msg = (
            "Arg emis must take at least the following keyword args:\n"
            "\t- r: major radius\n"
            "\t- z: height\n"
            "\t- phi: toroidal angle\n"
            "\t- lambda: wavelength\n"
            "\t- t: time\n"
            f"\nProvided: {argspec.args}"
        )
        raise Exception(msg)

    # binning
    if binning is None:
        binning = False
    if binning is not False:
        c0 = (
            isinstance(binning, dict)
            and sorted(binning.keys()) == ['xi', 'xj']
            and all([isinstance(v0, np.ndarray) for k0, v0 in binning.items()])
        )
        if c0:
            binning = (binning['xi'], binning['xj'])
        c0 = (
            hasattr(binning, '__iter__')
            and len(binning) == 2
            and all([isinstance(bb, np.ndarray) for bb in binning])
            and all([np.allclose(bb, np.unique(bb)) for bb in binning])
        )
        if not c0:
            msg = (
                "Arg binning must be either:\n"
                "\t- dict of keys ('xi', 'xj') with bin sorted edges arrays\n"
                "\t- list of 2 'xi' and 'xj' bin sorted edges arrays\n"
            )
            raise Exception(msg)

    return struct, lamb, binning