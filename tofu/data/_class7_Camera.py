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
            'dgeom.type',
            'dgeom.parallel',
            'dgeom.shape',
            'dgeom.ref',
            'dgeom.pix_area',
            'dgeom.pix_nb',
            'dgeom.outline',
            'dgeom.cent',
            'dgeom.cents',
            'dmat.qeff_energy',
            'dmat.qeff',
            'dmisc.color',
        ],
    })

    def _add_camera(
        self,
        dref=None,
        ddata=None,
        dobj=None,
        dmat=None,
        color=None,
    ):
        key = list(dobj['camera'].keys())[0]

        # material
        dref2, ddata2, dmat = _check._dmat(
            key=key,
            dmat=dmat,
        )

        if dmat is not None:
            if dref2 is not None:
                dref.update(dref2)
                ddata.update(ddata2)
            dobj['camera'][key]['dmat'] = dmat

        # dmisc
        dobj['camera'][key]['dmisc'] = _class3_check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_camera_1d(
        self,
        key=None,
        # geometry
        dgeom=None,
        # quantum efficiency
        dmat=None,
        # dmisc
        color=None,
    ):
        """ add a 1d camera

        A 1d camera is an unordered set of pixels of indentical outline
        Its geometry os defined by dgeom
        Its material properties (i.e: quantum efficiency) in dmat

        The geometry in dgeom must contain:
            - 'outline_x0': 1st coordinate of planar outline of a single pixel
            - 'outline_x1': 1st coordinate of planar outline of a single pixel
            - 'cents_x': x coordinate of the centers of ll pixels
            - 'cents_y': y coordinate of the centers of ll pixels
            - 'cents_z': z coordinate of the centers of ll pixels
            - 'nin_x': x coordinate of inward normal unit vector of all pixels
            - 'nin_y': y coordinate of inward normal unit vector of all pixels
            - 'nin_z': z coordinate of inward normal unit vector of all pixels
            - 'e0_x': x coordinate of e0 unit vector of all pixels
            - 'e0_y': y coordinate of e0 unit vector of all pixels
            - 'e0_z': z coordinate of e0 unit vector of all pixels
            - 'e1_x': x coordinate of e1 unit vector of all pixels
            - 'e1_y': y coordinate of e1 unit vector of all pixels
            - 'e1_z': z coordinate of e1 unit vector of all pixels

        The material dict, dmat can contain:
            - 'energy': a 1d energy vector , in eV
            - 'qeff': a 1d vector, same size as energy, with values in [0; 1]

        """
        # check / format input
        dref, ddata, dobj = _check._camera_1d(
            coll=self,
            key=key,
            **dgeom,
        )

        # add generic parts
        self._add_camera(
            dref=dref,
            ddata=ddata,
            dobj=dobj,
            dmat=dmat,
            color=color,
        )

    def add_camera_2d(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        # dmisc
        color=None,
    ):
        """ add a 2d camera

        A 2d camera is an ordered 2d grid of pixels of indentical outline
        Its geometry os defined by dgeom
        Its material properties (i.e: quantum efficiency) in dmat

        The geometry in dgeom must contain:
            - 'outline_x0': 1st coordinate of planar outline of a single pixel
            - 'outline_x1': 1st coordinate of planar outline of a single pixel
            - 'cent': (x, y, z) coordinate of the center of the camera
            - 'cents_x0': x0 coordinate of the centers of all pixels
            - 'cents_x1': x1 coordinate of the centers of all pixels
            - 'nin': x coordinate of inward normal unit vector of all pixels
            - 'e0': x coordinate of e0 unit vector of all pixels
            - 'e1': x coordinate of e1 unit vector of all pixels

        The material dict, dmat can contain:
            - 'energy': a 1d energy vector , in eV
            - 'qeff': a 1d vector, same size as energy, with values in [0; 1]

        """
        # check / format input
        dref, ddata, dobj = _check._camera_2d(
            coll=self,
            key=key,
            **dgeom,
        )

        # add generic parts
        self._add_camera(
            dref=dref,
            ddata=ddata,
            dobj=dobj,
            dmat=dmat,
            color=color,
        )

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
