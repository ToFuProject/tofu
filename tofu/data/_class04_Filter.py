# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class03_Aperture import Aperture as Previous
from . import _class3_check
from . import _class4_check as _check


__all__ = ['Filter']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Filter(Previous):

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'filter': [
            'dgeom.type',
            'dgeom.curve_r',
            'dgeom.area',
            'dmat.name',
            'dgeom.outline',
            'dgeom.poly',
            'dgeom.cent',
            'dmat.energy',
            'dmat.transmission',
            # 'dmisc.color',
        ],
    })

    def add_filter(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        # dmisc
        color=None,
    ):
        """ Add a filter

        dgeom is a dict holding the geometry:
            - 2d outline + 3d center + unit vectors (nin, e0, e1)


        dmat is a dict holding the material properties


        Unit vectors will be checked and normalized
        If planar, area will be computed
        Outline will be made counter-clockwise

        """

        # check / format input
        dref, ddata, dobj = _class3_check._add_surface3d(
            coll=self,
            key=key,
            which='filter',
            which_short='filt',
            # 2d outline
            **dgeom,
        )

        key = list(dobj['filter'].keys())[0]

        # material
        dref2, ddata2, dmat = _check._dmat(
            key=key,
            dmat=dmat,
        )

        if dmat is not None:
            dref.update(dref2)
            ddata.update(ddata2)
            dobj['filter'][key]['dmat'] = dmat

        # dmisc
        dobj['filter'][key]['dmisc'] = _class3_check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)
