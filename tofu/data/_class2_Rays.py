
# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class1_Plasma2D
from . import _class2_check as _check
# from . import _class1_compute


__all__ = ['Rays']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Rays(_class1_Plasma2D.Plasma2D):

    _show_in_summary = 'all'

    _dshow = dict(_class1_Plasma2D.Plasma2D._dshow)
    _dshow.update({
        'rays': [
            'shape',
            'ref',
            'pts',
            'lamb',
            'alpha',
            'dalpha',
            'dbeta',
        ],
    })

    def add_rays(
        self,
        key=None,
        # start points
        start_x=None,
        start_y=None,
        start_z=None,
        # wavelength
        lamb=None,
        # ref
        ref=None,
        # from pts
        pts_x=None,
        pts_y=None,
        pts_z=None,
        # from ray-tracing (vect + length or config or diag)
        vect_x=None,
        vect_y=None,
        vect_z=None,
        length=None,
        config=None,
        reflections_nb=None,
        reflections_type=None,
        diag=None,
    ):
        """ Add a set of rays

        Can be defined from:
            - pts: explicit
            - starting pts + vectors

        """

        # check / format input
        dref, ddata, dobj = _check._rays(
            coll=self,
            key=key,
            # start points
            start_x=start_x,
            start_y=start_y,
            start_z=start_z,
            # wavelength
            lamb=lamb,
            # ref
            ref=ref,
            # from pts
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # from ray-tracing (vect + length or config or diag)
            vect_x=vect_x,
            vect_y=vect_y,
            vect_z=vect_z,
            length=length,
            config=config,
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
            diag=diag,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # --------------
    # plotting
    # --------------

    def plot_rays(
        self,
        key=None,
    ):
        pass
