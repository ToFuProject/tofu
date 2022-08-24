# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class3_Aperture
from . import _class3_check as _check
from . import _class3_compute as _compute


__all__ = ['Filter']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Filter(_class3_Aperture.Aperture):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = dict(_class3_Aperture.Aperture._dshow)
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
        ],
    })

    def add_filter(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
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
        dref, ddata, dobj = _check._add_surface3d(
            coll=self,
            key=key,
            which='crystal',
            which_short='cryst',
            # 2d outline
            **dgeom,
        )

        key = list(dobj['crystal'].keys())[0]

        # material
        dobj['crystal'][key]['dmat'] = _check._dmat(
            dgeom=dobj['crystal'][key]['dgeom'],
            dmat=dmat,
            alpha=alpha,
            beta=beta,
        )

        # spectro
        # dspectro = _check._dspectro(
        # dobj=dobj,
        # dspectro=dspectro,
        # )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # compute rocking curve
        # if dmat['ready_to_compute'] is True:
        # self.set_crystal_rocking_curve()
