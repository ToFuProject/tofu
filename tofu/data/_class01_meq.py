# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:24:53 2025

@author: dvezinet
"""



# Built-in
import os
import warnings


# Common
import numpy as np


__all_ = ['load_meq']


# ########################################################
# ########################################################
#               Units
# ########################################################


# ########################################################
# ########################################################
#               check shapes
# ########################################################


def get_load_pfe():

    # -----------------
    # check dependency
    # -----------------

    try:
        import scipy.io as scpio
    except Exception as err:
        msg = (
            "loading an mat file requires an optional dependency:\n"
            "\t- file trying to load: {pfe}\n"
            "\t- required dependency: scipy.io"
        )
        err.args = (msg,)
        raise err

    # -----------------
    # define load_pfe
    # -----------------

    def func(pfe):

        dout = {
            k0: v0
            for k0, v0 in scpio.loadmat(pfe).items()
            if (
                (not k0.startswith('__'))
                and (
                    (isinstance(v0, np.ndarray) and v0.size > 0)
                    or not isinstance(v0, np.ndarray)
                )
            )
        }

        return dout

    return func