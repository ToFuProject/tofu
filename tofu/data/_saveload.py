# -*- coding: utf-8 -*-


import os
import datastock as ds


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

    coll = ds.load(
        pfe=pfe,
        cls=Collection,
        allow_pickle=allow_pickle,
        sep=sep,
        verb=verb,
    )

    return coll
