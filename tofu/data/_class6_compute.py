# -*- coding: utf-8 -*-


import numpy as np
import datastock as ds



# ########################################################
# ########################################################
#                   Grating's law
# ########################################################


def _get_func_angle_from2_lamb(
    theta_in=None,
    norder=None,
):

    # lamb_from_angle
    def lamb_from_angle(lamb, norder=norder, dist=dist):
        return np.arcsin(norder * lamb / (2.*dist))

    # angle_from_lamb
    def angle_from_lamb(lamb, norder=norder, dist=dist):
        return 2. * dist * np.sin(angle) / norder

    return lamb_from_angle, angle_from_lamb
