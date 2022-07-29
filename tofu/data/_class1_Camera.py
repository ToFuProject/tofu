# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class0_Plasma2D
from . import _class1_check


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Tokamak
# #############################################################################


class Diagnostic(_class0_Plasma2D.Plasma2D):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
        # 'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    def add_aperture(
        self,
        key=None,
        # 3d outline
        poly_x=None,
        poly_y=None,
        poly_z=None,
        # normal vector
        nin=None,
    ):
        # check / format input
        dref, ddata, dobj = _class1_check._aperture(
            key=key,
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            nin=nIn,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_camera(
        self,
        key=None,
        # common 2d outline
        outline_x0=None,
        outline_x1=None,
        # centers of all pixels
        cents_x=None,
        cents_y=None,
        cents_z=None,
        # inwards normal vectors
        nin_x=None,
        nin_y=None,
        nin_z=None,
        # orthonormal direct base
        e0_x=None,
        e0_y=None,
        e0_z=None,
        e1_x=None,
        e1_y=None,
        e1_z=None,
        # quantum efficiency
        lamb=None,
        energy=None,
        qeff=None,
    ):
        # check / format input
        dref, ddata, dobj = _class1_check._camera(
            key=key,
            # common 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            # centers of all pixels
            cents_x=cents_x,
            cents_y=cents_y,
            cents_z=cents_z,
            # inwards normal vectors
            nin_x=nin_x,
            nin_y=nin_y,
            nin_z=nin_z,
            # orthonormal direct base
            e0_x=e0_x,
            e0_y=e0_y,
            e0_z=e0_z,
            e1_x=e1_x,
            e1_y=e1_y,
            e1_z=e1_z,
            # quantum efficiency
            lamb=lamb,
            energy=energy,
            qeff=qeff,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_diagnostic(
        self,
        key=None,
        cameras=None.
        apertures=None,
        **kwdargs,
    ):
        # check / format input
        dref, ddata, dobj = _class1_check._camera(
            key=key,
            cameras=cameras,
            apertures=apertures,
            **kwdargs,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)
