
import warnings

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
    for kk in lkok:
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

    # Check all unit vectors
    for k0, v0 in ['nout', 'nin', 'e1', 'e2']:
        dgeom[k0] = _check_flat1darray_size(
            var=dgeom.get(k0), varname=k0, size=3, norm=True)

    # Check consistency between unit vectors
    if dgeom['e1'] is not None:
        c0 = (
            dgeom['e2'] is not None
            and np.abs(np.sum(dgeom['e1']*dgeom['e2'])) < 1.e-12
        )
        if not c0:
            msg = (
                """
                If dgeom['e1'] is provided, then:
                    - dgeom['e2'] must be provided too
                    - dgeom['e1'] must be perpendicular to dgeom['e2']

                Provided:
                    {}
                """.format(dgeom['e2'])
            )
            raise Exception(msg)

        if dgeom['nout'] is not None:
            _check_orthornormaldirect(
                e1=dgeom['e1'], e2=dgeom['e2'],
                var=dgeom['nout'], varname='nout',
            )
        if dgeom['nin'] is not None:
            _check_orthornormaldirect(
                e1=dgeom['e1'], e2=dgeom['e2'],
                var=-dgeom['nin'], varname='-nin',
            )

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

    if dgeom['extenthalf'] is not None:
        if dgeom['Type'] == 'sph' and dgeom['Typeoutline'] == 'rect':
            ind = np.argmax(dgeom['extenthalf'])
            dphi = dgeom['extenthalf'][ind]
            sindtheta = np.sin(dgeom['extenthalf'][ind-1])
            dgeom['surface'] = 4.*dgeom['rcurve']**2*dphi*sindtheta
    return dgeom


# ####################################################################
# ####################################################################
#               check dmat
# ####################################################################
# ####################################################################


def _checkformat_dmat(dmat=None, ddef=None, valid_keys=None):
    """
    Check the content of dmat

    Crystal parameters : d, formula, density, lengths, angles, cut

    New basis of unit vectors due to non-parallelism (e1, e2, nout) + nin + alpha, beta
    """	


    if dmat is None:
        return

    #---------------------
    # check dict integrity
    #---------------------

    # Check dict typeand content (each key is a valid string)
    _check_dict_valid_keys(var=dmat, varname='dmat', valid_keys=valid_keys)

    # Set default values if any
    for kk in ddef.keys():
        dmat[kk] = dmat.get(kk, ddef[kk]) 	

    #-------------------------------
    # check each value independently
    #-------------------------------

    # Check dimension of array and its size 	
    dmat['lengths'] = _check_flat1darray_size(
        var=dmat.get('lengths'), varname='lengths', size=1)
    dmat['angles'] = _check_flat1darray_size(
        var=dmat.get('angles'), varname='angles', size=1)	

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
        if dmat['cut'].size <= 4:
            msg = (
                """
                Var {} should be convertible to a 1d np.ndarray of minimal size of {}.
                
                Provided: {}
                """.format('cut', 5, dmat.get('cut'))
            )
            raise Exception(msg)

    # Check all unit vectors
    for k0 in ['nout', 'nin', 'e1', 'e2',]:
        dmat[k0] = _check_flat1darray_size(
            var=dmat.get(k0), varname=k0, size=3, norm=True)
    
    # Check consistency between unit vectors
    if dmat['e1'] is not None:
        c0 = (
            dmat['e2'] is not None
            and np.abs(np.sum(dmat['e1']*mat['e2'])) < 1.e-12
        )
        if not c0:
            msg = (
                """
                If dmat['e1'] is provided, then:
                    - dmat['e2'] must be provided too
                    - dmat['e1'] must be perpendicular to dmat['e2']

                Provided:
                    {}
                """.format(dmat['e2'])
            )
            raise Exception(msg)

        if dmat['nout'] is not None:
            _check_orthornormaldirect(
                e1=dmat['e1'], e2=dmat['e2'],
                var=dmat['nout'], varname='nout',
            )
        if dmat['nin'] is not None:
            _check_orthornormaldirect(
                e1=dmat['e1'], e2=dmat['e2'],
                var=-dmat['nin'], varname='-nin',
            )
    # Check all additionnal angles to define the new basis
    for k0 in ['alpha', 'beta',]:
        dmat[k0] = _check_flat1darray_size(
            var=dmat.get(k0), varname=k0, size=1, norm=False)

    # -------------------------------------------------------------
    # Add missing vectors and parameters according to the new basis 
    # -------------------------------------------------------------

    if all([dgeom[kk] is not None for kk in ['nout', 'e1', 'e2']]):

        # dict of value, comment, default value and type of alpha and beta angles in dmat
        dpar = {'alpha':
             {'alpha':alpha, 'com':'non-parallelism amplitude', 'default':0, 'type':float},
             'beta':
             {'beta':beta, 'com':'non-parallelism orientation', 'default':0, 'type':float}
        }

        # dict of value, comment, default value and type of unit vectors in dmat
        dvec = {'e1':
             {'e1':e1, 'com':'unit vector (non-parallelism)', 'default':0, 'type':float},
             'e2':
             {'e2':e2, 'com':'unit vector (non-parallelism)', 'default':0, 'type':float},
             'nout':
             {'nout':nout, 'com':'unit vector (non-parallelism)', 'default':0, 'type':float}
        }

        # setting to default value if any is None
        lparNone = [aa for aa in dpar.keys() if dmat(aa) is None]
        lvecNone = [bb for bb in dvec.keys() if dmat(bb) is None]

        # if any is None, assigning default value and send a warning message 
        if len(lparNone) > 0:
            msg = "The following parameters were set to their default values:"
            for aa in lparNone:
                dmat[aa] = dpar[aa]['default']
                msg += "\n\t - {} = {} ({})".format(aa, dpar[aa]['default'], dpar[aa]['com'])
            warnings.warn(msg)

        if len(lvecNone) > 0:
            msg = "The following parameters were set to their default values:"
            for bb in lvecNone:
                dmat[bb] = dvec[bb]['default']
                msg += "\n\t - {} = {} ({})".format(bb, dvec[bb]['default'], dvec[bb]['com'])
            warnings.warn(msg)

        # check conformity of type of parameters
        lparWrong = []; lvecWrong = []
        for aa in dpar.keys():
            try:
                dmat[aa] = float(dmat[aa])
            except Exception as err:
                lparWrong.append(aa)
        
        if len(lparWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for aa in lparWrong:
                msg += "\n\t - {} = {} ({})".format(aa, dpar[aa]['type'], type(dpar[aa]['value']))
            raise Exception(msg)

        for bb in dvec.keys():
            try:
                dmat[bb] = np.atleast_1d(dmat[bb]).ravel().astype(int)
            except Exception as err:
                lvecWrong.append(bb)

        if len(lvecWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for bb in lvecWrong:
                msg += "\n\t - {} = {} ({})".format(bb, dvec[bb]['type'], type(dvec[bb]['value']))
            raise Exception(msg)

    if all([dgeom[kk] is not None for kk in ['nout', 'e1', 'e2']]):
        # Computation of unit vectors nout, nin, e1 and e2  
        dmat['nout'] = (dgeom['nout']*np.cos(dmat['alpha'])
                     +np.sin(dmat['alpha'])*(np.cos(dmat['beta'])*dgeom['e1']
                     +np.sin(dmat['beta'])*dgeom['e2'])
        )
        dmat['nin'] = -dmat['nout']
        dmat['e1'] = np.sin(dmat['alpha'])*(np.cos(dmat['beta'])*dgeom['e1']
                   +np.sin(dmat['beta'])*dgeom['e2'])
        dmat['e2'] = np.cross(dmat['nout'], dmat['e1'])

        # 0 < alpha < pi/2
        if dgeom['e1'] and dgeom['nout'] >= 0:
            dmat['alpha'] = np.abs(np.arctan(dgeom['e1'] / dgeom['nout']))
        if dgeom['e1'] >= 0 and dgeom['nout'] < 0:
            dmat['alpha'] = np.abs(np.arctan(dgeom['e1'] / dgeom['nout'])+(np.pi/2))
        if dgeom['e1'] < 0 and dgeom['nout'] >= 0:
            dmat['alpha'] = np.abs(np.arctan(dgeom['e1'] / dgeom['nout'])+(np.pi/2))
        if dgeom['e1'] and dgeom['nout'] < 0:
            dmat['alpha'] = np.abs(np.arctan(dgeom['e1'] / dgeom['nout'])+(2*np.pi))
        # 0 < beta < 2pi
        dmat['beta'] = np.arctan2(dgeom['e2'], dgeom['e1'])+(np.pi)

        # check if input parameters verify trigonometry relations between each bases
        try:
            np.cos(dmat['alpha']) = np.abs(dgeom['nout'] / dmat['nout'])
            np.cos(dmat['beta']) = dgeom['e1'] / dmat['e1']
        except Exception as err:
            msg = (
                """
                Please check your args in input, something seems wrong with conversions:
                - np.cos(dmat['alpha']) = dgeom['nout'] / dmat['nout']
                - np.cos(dmat['beta']) = dgeom['e1'] / dmat['e1']

                If angles are provided, alpha should be in [0; pi/2] range and 
                beta in [0; 2pi] range.
                If unit vectors are, we've first computed these angles from dgeom[nout, e1, e2]
                and checked with your inputs dmat[nout, e1, e2].

                Provide:
                    - {} = {} 
                    - {} = {} 
                    - {} = {} 
                    - {} = {} 
                """.format(
                    'alpha', dmat['alpha'],
                    'beta', dmat['beta'],
                    'dmat[nout]', dmat['nout'],
                    'dmat[e1]', dmat['e1'])
            )
            raise Exception(msg)
    return dmat


# ####################################################################
# ####################################################################
#               check dbragg
# ####################################################################
# ####################################################################


def _checkformat_dbragg(dbragg=None, ddef=None, valid_keys=None):
    
    """	
    rocking curve, wavelenght of reference and its bragg angle associated
    """
    if dbragg is None:
        dbragg = dict.fromkeys(_core_optics._get_keys_dbragg())


    #---------------------
    # check dict integrity
    #---------------------

    # Check type dict and content (each key is a valid string)
    _check_dict_valid_keys(var=dbragg, varname='dbragg', valid_keys=valid_keys)

    # Set default values when necessary
    for kk in ddef.keys():
        dbragg[kk] = dbragg.get(kk, ddef[kk])

    #------------------------------------------------
    # Check braggref and lambref type and computation
    #------------------------------------------------

    # Check braggref
    ltypes = [int, float, np.int_, np.float_]
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

    # Update braggref according to lambref, if lambref was user-provided
    if user_prov is True:
        dbragg['braggref'] = braggref

    # Set default lambda if necessary
    user_prov = True
    if dbragg.get('lambref') is None:
        # set to bragg = braggref
        dbragg['lambref'] = _comp_optics.get_lamb_from_bragg(
                            np.r_[dbragg['braggref']], _core_optics._dmat['d'], n=1)[0]
        user_prov = False

    # Set default bragg angle if necessary
    braggref = _comp_optics.get_bragg_from_lamb(
                            np.r_[dbragg['lambref']], _core_optics._dmat['d'], n=1)[0]
    if np.isnan(braggref):
        lambok = []
        msg = (
            """
            Var {} is not valid!
            Please check your arguments to calculate the Bragg's law correctly!
            Provided:
                - crystal inter-plane d [A] = {}
                - wavelenght interval [m] : {}
                - lambref = {}
            """.format('lambref', dmat['d'], lambok, dbragg['lambref'])
        )
        raise Exception(msg)

    #------------------------------------------------
    # Check rocking curve value 
    #------------------------------------------------

    # Check type dict and content (each key is a valid string)
    drock=dbragg['rockingcurve']
    lkeyok = ['sigma', 'deltad', 'Rmax', 'dangle', 'lamb', 'value', 'type',
              'source']
    _check_dict_valid_keys(var=drock, varname='drock', valid_keys=lkeyok)

    # check type, size and content of each key in drock
    try:
        if drock.get('sigma') is not None:
            dbragg['rockingcurve']['sigma'] = float(drock['sigma'])
            dbragg['rockingcurve']['deltad'] = float(drock.get('deltad', 0.))
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
    return dbragg


# ####################################################################
# ####################################################################
#               check colors
# ####################################################################
# ####################################################################


@staticmethod
def _checkformat_inputs_dmisc(cls, color=None):
    if color is None:
        color = mpl.colors.to_rgba(cls._ddef['dmisc']['color'])
    assert mpl.colors.is_color_like(color)
    return tuple(mpl.colors.to_rgba(color))
