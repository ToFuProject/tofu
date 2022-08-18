# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class3_Crystal
# from . import _class4_check


__all__ = ['Grating']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Grating(_class3_Crystal.Crystal):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = {
        'grating': [
            'type', 'material',
            'rcurve', 'miller',
            'cent',
        ],
    }

    def add_grating(
        self,
        key=None,
        # 2d outline
        outline_x0=None,
        outline_x1=None,
        cent=None,
        # 3d outline
        poly_x=None,
        poly_y=None,
        poly_z=None,
        # normal vector
        nin=None,
        e0=None,
        e1=None,
    ):
        """ Add a grating

        Can be defined from:
            - 2d outline + 3d center + unit vectors (nin, e0, e1)
            - 3d polygon + nin

        Unit vectors will be checked and normalized
        If planar, area will be computed
        Outline will be made counter-clockwise

        """

        # check / format input
        dref, ddata, dobj = _class4_check._grating(
            coll=self,
            key=key,
            # 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            cent=cent,
            # 3d outline
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            # normal vector
            nin=nin,
            e0=e0,
            e1=e1,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

