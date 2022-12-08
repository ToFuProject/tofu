# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


# tofu
# from tofu import __version__ as __version__


__all__ = ['Config']


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class Config(ds.DataStock):

    _show_in_summary = 'all'
    _dshow = dict(ds.DataStock._dshow)
    _dshow.update({
        'structure': [
        ],
        'config': [
        ],
    })
