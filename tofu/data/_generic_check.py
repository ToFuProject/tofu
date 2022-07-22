# -*- coding: utf-8 -*-


# common
import matplotlib.pyplot as plt


_LALLOWED_AXESTYPES = [
    None,
    'cross', 'hor',
    'matrix',
    'timetrace',
    'profile1d',
    'image',
    'text',
    'misc',
]


# #############################################################################
# #############################################################################
#                           Utilities
# #############################################################################


def _check_var(
    var,
    varname,
    types=None,
    default=None,
    allowed=None,
    excluded=None,
):
    """ Check a variable, with options

    - check is instance of any types
    - check belongs to list of allowed values
    - check does not belong to list of excluded values

    if None:
        - set to default if provided
        - set to allowed value if only has 1 element

    Print proper error message if necessary, return the variable itself

    """

    # set to default
    if var is None:
        var = default
    if var is None and allowed is not None and len(allowed) == 1:
        var = allowed[0]

    # check type
    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(msg)

    # check if allowed
    if allowed is not None:
        if var not in allowed:
            msg = (
                f"Arg {varname} must be in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    # check if excluded
    if excluded is not None:
        if var in excluded:
            msg = (
                f"Arg {varname} must not be in {excluded}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    return var


def _check_var_iter(
    var,
    varname,
    types=None,
    types_iter=None,
    default=None,
    allowed=None,
    excluded=None,
):
    """ Check a variable supposed to be an iterable, with options

    - check is instance of any types
    - check each element is instance of types_iter
    - check each element belongs to list of allowed values
    - check each element does not belong to list of excluded values

    if var is not iterable, turned into a list of itself

    if None:
        - set to default if provided
        - set to list of allowed if provided

    Print proper error message if necessary, return the variable itself

    """

    # set to default
    if var is None:
        var = default
    if var is None and allowed is not None:
        var = allowed

    if var is not None and not hasattr(var, '__iter__'):
        var = [var]

    # check type
    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(msg)

    # check types_iter
    if types_iter is not None and var is not None:
        if not all([isinstance(vv, types_iter) for vv in var]):
            msg = (
                f"Arg {varname} must be an iterable of types {types_iter}\n"
                f"Provided: {[type(vv) for vv in var]}"
            )
            raise Exception(msg)

    # check if allowed
    if allowed is not None:
        if any([vv not in allowed for vv in var]):
            msg = (
                f"Arg {varname} must contain elements in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    # check if excluded
    if excluded is not None:
        if any([vv in excluded for vv in var]):
            msg = (
                f"Arg {varname} must contain elements not in {excluded}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    return var


# #############################################################################
# #############################################################################
#                   Utilities for naming keys
# #############################################################################


def _name_key(dd=None, dd_name=None, keyroot='key'):
    """ Return existing default keys and their number as a dict

    Used to automatically iterate on dict keys

    """

    dk = {
        kk: int(kk[len(keyroot):])
        for kk in dd.keys()
        if kk.startswith(keyroot)
        and kk[len(keyroot):].isnumeric()
    }
    if len(dk) == 0:
        nmax = 0
    else:
        nmax = max([v0 for v0 in dk.values()]) + 1
    return dk, nmax


# #############################################################################
# #############################################################################
#                   Utilities for plotting
# #############################################################################


def _check_dax(dax=None, main=None):

    # None
    if dax is None:
        return dax

    # Axes
    if issubclass(dax.__class__, plt.Axes):
        if main is None:
            msg = (
            )
            raise Exception(msg)
        else:
            return {main: dax}

    # dict
    c0 = (
        isinstance(dax, dict)
        and all([
            isinstance(k0, str)
            and (
                (
                    issubclass(v0.__class__, plt.Axes)
                )
                or (
                    isinstance(v0, dict)
                    and issubclass(v0.get('handle').__class__, plt.Axes)
                    and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
        )
        import pdb; pdb.set_trace()     # DB
        raise Exception(msg)

    for k0, v0 in dax.items():
        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'handle': v0, 'type': k0}
        if isinstance(v0, dict):
            dax[k0]['type'] = v0.get('type')

    return dax
