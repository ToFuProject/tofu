# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import scipy.sparse as scpsp


_LOPERATORS_INT = [
    'D0',
    'D0N2',
    'D1N2',
    'D2N2',
]


# #############################################################################
# #############################################################################
#                           Utilities
# #############################################################################


def _check_var(var, varname, types=None, default=None, allowed=None):
    if var is None:
        var = default

    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(msg)

    if allowed is not None:
        if var not in allowed:
            msg = (
                f"Arg {varname} must be in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)
    return var


# #############################################################################
# #############################################################################
#                   Mesh2DRect - bsplines - operators
# #############################################################################


def _get_mesh2dRect_operators_check(operator=None, geometry=None):

    # operator
    operator = _check_var(
        operator, 'operator',
        default='D0',
        types=str,
        allowed=_LOPERATORS_INT,
    )

    # geometry
    geometry = _check_var(
        geometry, 'geometry',
        default='toroidal',
        types=str,
        allowed=['toroidal', 'linear'],
    )


def get_mesh2dRect_operators(
    mesh=None,
    operator=None,
    geometry=None,
    deg=None,
    knotsR=None,
    knotsZ=None,
):

    # ------------
    # check inputs

    operator, geometry = _get_mesh2dRect_operators_check(
        operator=operator,
        geometry=geometry,
    )

    # ------------
    # compute

    # D0N2
    if operator == 'D0N2':

        if deg == 0:

            if geometry == 'linear':
                op_matrix = None # scpsp.diags()
            else:
                op_matrix = None

        if deg == 1:

            if geometry == 'linear':
                op_matrix = None # scpsp.diags()
            else:
                op_matrix = None




    return operator, op_matrix, components
