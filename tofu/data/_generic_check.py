# -*- coding: utf-8 -*-


# common
import matplotlib.pyplot as plt


_LALLOWED_AXESTYPES = [
    'cross', 'hor',
    'matrix',
    'timetrace',
    'profile1d',
    'image',
    'misc'
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
                    k0 in _LALLOWED_AXESTYPES
                    and issubclass(v0.__class__, plt.Axes)
                )
                or (
                    isinstance(v0, dict)
                    and issubclass(v0.get('ax').__class__, plt.Axes)
                    and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
        )
        raise Exception(msg)

    for k0, v0 in dax.items():
        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'ax': v0, 'type': k0}

    return dax
