# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class6_Grating
from . import _class3_check
from . import _class7_check as _check


__all__ = ['Camera']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Camera(_class6_Grating.Grating):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _dshow = dict(_class6_Grating.Grating._dshow)
    _dshow.update({
        'camera': [
            'type', 'parallel',
            'shape', 'ref',
            'pix area', 'pix nb',
            'outline',
            'cent', 'cents',
            'qeff_energy',
            'qeff',
            'model',
        ],
    })

    def add_camera_1d(
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
        dref, ddata, dobj = _check._camera_1d(
            coll=self,
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

    def add_camera_2d(
        self,
        key=None,
        # common 2d outline
        outline_x0=None,
        outline_x1=None,
        # centers of all pixels
        cent=None,
        cents_x0=None,
        cents_x1=None,
        # inwards normal vectors
        nin=None,
        e0=None,
        e1=None,
        # quantum efficiency
        lamb=None,
        energy=None,
        qeff=None,
    ):
        # check / format input
        dref, ddata, dobj = _check._camera_2d(
            coll=self,
            key=key,
            # common 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            # centers of all pixels
            cent=cent,
            cents_x0=cents_x0,
            cents_x1=cents_x1,
            # inwards normal vectors
            nin=nin,
            e0=e0,
            e1=e1,
            # quantum efficiency
            lamb=lamb,
            energy=energy,
            qeff=qeff,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # ---------------
    # utilities
    # ---------------

    def get_camera_unit_vectors(self, key=None):
        """ Return unit vectors components as dict """
        return _check.get_camera_unitvectors(
            coll=self,
            key=key,
        )

    def get_camera_cents_xyz(self, key=None):
        """ Return cents_x, cents_y, cents_z """
        return _check.get_camera_cents_xyz(
            coll=self,
            key=key,
        )

    def get_as_dict(self, key=None):
        """ Return the desired object as a dict (input to some routines) """

        return _class3_check._return_as_dict(
            coll=self,
            key=key,
        )
