
# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class1_Plasma2D
# from . import _class1_check
# from . import _class1_compute


__all__ = ['Rays']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Rays(_class1_Plasma2D.Plasma2D):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _dshow = dict(_class1_Plasma2D.Plasma2D._dshow)
    _dshow.update({
        'rays': [
            'shape',
            'pts', 'segments', 'lamb',
            'touch', 'angles',
        ],
    })

    def add_rays(
        self,
        key=None,
        # from pts
        pts_x=None,
        pts_y=None,
        pts_z=None,
        # from ray-tracing (start + vect + length)
        start_x=None,
        start_y=None,
        start_z=None,
        vect_x=None,
        vect_y=None,
        vect_z=None,
        length=None,
        segments=None,
        config=None,
        diag=None,
    ):
        """ Add a set of rays

        Can be defined from:
            - pts: explicit
            - starting pts + vectors

        """

        # check / format input
        dref, ddata, dobj = _class1_check._rays(
            coll=self,
            key=key,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_refelections(
        self,
        key=None,
        nb=None,
    ):
        pass

    # --------------
    # plotting
    # --------------

    def plot_rays(
        self,
        key=None,
    ):
        pass
