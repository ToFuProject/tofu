# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds
import bsplines2d as bs2


# tofu
# from tofu import __version__ as __version__


__all__ = ['Config']


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class Config(bs2.BSplines2D):

    _show_in_summary = 'all'
    _dshow = dict(bs2.BSplines2D._dshow)
    _dshow.update({
        'structure': [
        ],
        'config': [
        ],
    })
