# -*- coding: utf-8 -*-


# Common
import spectrally as sp


# tofu
# from tofu import __version__ as __version__


__all__ = ['Config']


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class Config(sp.Collection):

    _show_in_summary = 'all'
    _dshow = dict(sp.Collection._dshow)
    _dshow.update({
        'structure': [
        ],
        'config': [
        ],
    })