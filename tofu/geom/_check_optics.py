
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
    
    # Check all additionnal angles to define the new basis
    for k0 in ['alpha', 'beta',]:
        dmat[k0] = _check_flat1darray_size(
            var=dmat.get(k0), varname=k0, size=1, norm=False)

    # -------------------------------------------------------------
    # Add missing vectors and parameters according to the new basis 
    # -------------------------------------------------------------

    if all([dgeom[aa] is not None for aa in ['nout', 'e1', 'e2']]):
        # If alpha & beta not given, parallelism into crystal is imposed on.
        # If any value is missing, possibility of non-parallelism and error message is
        #  shown to remedy the situation.
        # If all values are given, the new basis computation can be done.
        if all([dmat[aa] is None for aa in ['alpha', 'beta']]):
            dmat['nout'] = None
            warnings.warn("Parallelism into crystal will be imposed here if alpha and beta are None.")
        if any([dmat[aa] is None for aa in ['alpha', 'beta']]):
            msg = (
                """
                Please give two valid values for alpha and beta angles to compute the new basis!
                Provided:
                    -{} = {}
                    -{} = {}
                """.format('alpha', dmat['alpha'], 'beta', dmat['beta'])
            )
            raise Exception(msg)
        if all([dmat[aa] is not None for aa in ['alpha', 'beta']]):
            if dmat['nout'] is None:
	            dmat['nout'] = dgeom['nout']*np.cos(dmat['alpha'])
                               +np.sin(dmat['alpha'])*(np.cos(dmat['beta'])*dgeom['e1']
                               +np.sin(dmat['beta'])*dgeom['e2'])

    if dmat['nin'] is None and dmat['nout'] is not None:
        dmat['nin'] = -dmat['nout']

    # warning still to be insered 
    if all([dgeom[aa] is not None for aa in ['e1', 'nout']]):
        if dmat['alpha'] is None:
	    # 0 < alpha < pi/2
            if dgeom['e1'] and dgeom['nout'] >= 0:
                dmat['alpha'] = np.arctan(dgeom['e1'] / dgeom['nout'])
            if dgeom['e1'] >= 0 and dgeom['nout'] < 0:
                dmat['alpha'] = np.arctan(dgeom['e1'] / dgeom['nout'])+(np.pi/2)
            if dgeom['e1'] < 0 and dgeom['nout'] >= 0:
                dmat['alpha'] = np.arctan(dgeom['e1'] / dgeom['nout'])+(np.pi/2)
            if dgeom['e1'] and dgeom['nout'] < 0:
                dmat['alpha'] = np.arctan(dgeom['e1'] / dgeom['nout'])+(2*np.pi)

    # warning still to be insered
    if all([dgeom[aa] is not None for aa in ['e1', 'e2']]):
        if dmat['beta'] is None:
	    # 0 < beta < 2pi
	        dmat['beta'] = np.arctan2(dgeom['e2'] / dgeom['e1'])

    if all([dgeom[aa] is not None for aa in ['nout', 'e1', 'e2']]):
        if all([dmat[aa] is None for aa in ['alpha', 'beta', 'nout']]):
            dmat['e1'] = None
            dmat['e2'] = None
            warnings.warn("Parallelism into crystal will be imposed here if neither alpha, nor beta nor nout are known.")
        if any([dmat[aa] is None for aa in ['alpha', 'beta', 'nout']]):
            msg = (
                """
                Please give three valid values for alpha and beta angles plus nout unit vector to compute the new basis!
                Provided:
                    -{} = {}
                    -{} = {}
                    -{} = {}
                """.format('alpha', dmat['alpha'], 'beta', dmat['beta'], 'nout', dmat['nout'])
            )
            raise Exception(msg)
        if all([dmat[aa] is not None for aa in ['alpha', 'beta', 'nout']]):
	        dmat['e1'] = np.cos(dmat['beta'])*dgeom['e1']+np.sin(dmat['beta'])*dgeom['e2']
	        dmat['e2'] = np.cross(dmat['nout'], dmat['e1'])
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
                - crystal inter-plane d = {}
                - wavelenght interval [m] : {}
                - lambref = {}
            """.format('lambref', dmat['d'], lambok, dbragg['lambref'])
        )
        raise Exception(msg)

    #------------------------------------------------
    # Check rocking curve value 
    #------------------------------------------------

    # Check type dict and content (each key is a valid string)
    #_check_dict_valid_keys(var=dbragg['rockingcurve'], varname='rockingcurve', valid_keys=valid_keys)
