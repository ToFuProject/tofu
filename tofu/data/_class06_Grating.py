# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class05_Crystal import Crystal as Previous
from . import _class3_check
# from . import _class6_check as _check
from . import _class06_compute as _compute


__all__ = ['Grating']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Grating(Previous):

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'crystal': [
            'dgeom.type',
            'dgeom.curve_r',
            'dgeom.area',
            'dmat.name',
            'dmat.miller',
            'dgeom.outline',
            'dgeom.poly',
            'dgeom.cent',
            # 'dmisc.color',
        ],
    })

    def add_grating(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        # dmisc
        color=None,
    ):
        """ Add a crystal

        Can be defined from:
            - 2d outline + 3d center + unit vectors (nin, e0, e1)
            - 3d polygon + nin

        Unit vectors will be checked and normalized
        If planar, area will be computed
        Outline will be made counter-clockwise

        """

        # check / format input
        dref, ddata, dobj = _class3_check._add_surface3d(
            coll=self,
            key=key,
            which='grating',
            which_short='grat',
            # 2d outline
            **dgeom,
        )

        key = list(dobj['grating'].keys())[0]

        # material
        dobj['grating'][key]['dmat'] = _check._dmat(
            dgeom=dobj['grating'][key]['dgeom'],
            dmat=dmat,
            alpha=alpha,
            beta=beta,
        )

        # dmisc
        dobj['grating'][key]['dmisc'] = _class3_check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # ------------
    # utilities
    # ------------

    def get_optics_isconvex(self, keys=None):
        """ return list of bool flags indicating if each optics is convex """
        return _compute._isconvex(coll=self, keys=keys)
