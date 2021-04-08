

# ####################################################################
# ####################################################################
#               common routines for checking
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
    if norm is True
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

    # Check dict type and content (each key is a valid str)
    _check_dict_valid_keys(var=dgeom, varname='dgeom', valid_keys=valid_keys)

    # Set default values if any
    for kk in ddef.keys():
        dgeom[kk] = dgeom.get(kk, ddef[kk])

    # Set default values to None of not provided
    for kk in lkok:
        dgeom[kk] = dgeom.get(kk, None)

    # --------------------
    # Check each value independently

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

    # ----------------------
    # Add missing vectors and parameters

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
