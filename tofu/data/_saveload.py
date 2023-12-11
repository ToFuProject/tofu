# -*- coding: utf-8 -*-


import os
import bsplines2d as bs2


__all__ = ['load']


# ##################################################################
# ##################################################################
#                   Save
# ##################################################################


# ###################################################################
# ###################################################################
#               load
# ###################################################################


def load(
    pfe=None,
    cls=None,
    allow_pickle=None,
    sep=None,
    verb=None,
):

    # --------------------
    # use datastock.load()

    from ._class10_Inversion import Inversion as Collection

    coll = bs2.load(
        pfe=pfe,
        cls=Collection,
        allow_pickle=allow_pickle,
        sep=sep,
        verb=verb,
    )

    return coll