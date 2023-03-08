# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import datastock as ds

# specific
from . import _spectralunits


# ########################################################
# ########################################################
#               units
# ########################################################


def convert_spectral_units(
    coll=None,
    data=None,
    units=None,
    units_in=None,
):

    # ----------
    # check data

    if isinstance(data, str):

        # data
        lok = list(coll.ddata.keys())
        data = ds._generic_checks._check_var(
            data, 'data',
            types=str,
            allowed=lok,
        )

        # extract
        units_in = str(coll.ddata[data]['units'])
        data = coll.ddata[data]['data']

    # ----------
    # convert

    return _spectralunits.convert_spectral(
        data_in=data,
        units_in=units_in,
        units_out=units,
    )