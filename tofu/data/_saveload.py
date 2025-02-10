# -*- coding: utf-8 -*-


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
    coll=None,
    allow_pickle=None,
    sep=None,
    verb=None,
):

    # --------------------
    # use datastock.load()

    if cls is None:
        from ._class10_Inversion import Inversion as Collection
        cls = Collection

    coll = bs2.load(
        pfe=pfe,
        cls=cls,
        coll=coll,
        allow_pickle=allow_pickle,
        sep=sep,
        verb=verb,
    )

    return coll