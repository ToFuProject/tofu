# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class2_Camera
from . import _class3_check


__all__ = ['Crystal']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Crystal(_class2_Camera.Camera):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = {
        'crystal': [
            'type', 'material',
            'rcurve', 'miller',
            'cent',
        ],
    }

    def add_crystal(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        # spectro
        dspectro=None,
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
        dref, ddata, dobj = _class2_check._add_surface3d(
            coll=self,
            key=key,
            which='crystal',
            which_short='cryst',
            # 2d outline
            **dgeom,
        )

        # material
        _class3_check._dmat(
            dobj=dobj,
            **dmat,
        )

        # spectro
        _class3_check._dspectro(
            dobj=dobj,
            dspectro=dspectro,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def get_crystal_ideal_configuration(
        self,
        key=None,
        configuration=None,
        lamb=None,
        bragg=None,
    ):
        """ Return the ideal positions of other elements for a configuration

        'ideal' means:
            - maximizing focalization
            - for the chosen wavelength / bragg angle

        'johann': for spherical crystals only
            build the rowland circle and return the ideal detector position

        'von hamos': for cylindrical only
            build the axis and return
             - slit position
             - detector position given its height

        'pinhole': for cylindrical only
            buids the axis and return
             - pinhole position
             - detector position given its height
        """

        return _class3_compute._ideal_configuration(
            coll=self,
            key=key,
            configuration=configuration,
            lamb=lamb,
            bragg=bragg,
        )
