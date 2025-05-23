# -*- coding: utf-8 -*-


# Built-in
# import copy


# tofu
from ._class02_Rays import Rays as Previous
from . import _class3_check as _check
from . import _class03_save2stp as _save2stp


__all__ = ['Aperture']


# #########################################################################
# #########################################################################
#                           Diagnostic
# #########################################################################


class Aperture(Previous):

    # _ddef = copy.deepcopy(ds.DataStock._ddef)
    # _ddef['params']['ddata'].update({
    #       'bsplines': (str, ''),
    # })
    # _ddef['params']['dobj'] = None
    # _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'aperture': [
            'dgeom.type',
            'dgeom.curve_r',
            'dgeom.area',
            'dgeom.outline',
            'dgeom.poly',
            'dgeom.cent',
            # 'dmisc.color',
        ],
    })

    def add_aperture(
        self,
        key=None,
        # 2d outline
        outline_x0=None,
        outline_x1=None,
        cent=None,
        extenthalf=None,
        # 3d outline
        poly_x=None,
        poly_y=None,
        poly_z=None,
        # normal vector
        nin=None,
        e0=None,
        e1=None,
        # curvature
        curve_r=None,
        make_planar=None,
        # dmisc
        color=None,
    ):
        """ Add an aperture

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
            which='aperture',
            which_short='ap',
            # 2d outline
            outline_x0=outline_x0,
            outline_x1=outline_x1,
            cent=cent,
            extenthalf=extenthalf,
            # 3d outline
            poly_x=poly_x,
            poly_y=poly_y,
            poly_z=poly_z,
            # normal vector
            nin=nin,
            e0=e0,
            e1=e1,
            # curvature
            curve_r=curve_r,
            make_planar=make_planar,
        )

        # dmisc
        key = list(dobj['aperture'].keys())[0]
        dobj['aperture'][key]['dmisc'] = _check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # ---------------
    # utilities
    # ---------------

    def get_as_dict(self, which=None, key=None):
        """ Return the desired object as a dict (input to some routines) """

        return _check._return_as_dict(
            coll=self,
            which=which,
            key=key,
        )

    # ----------------
    # save as stp file
    # ----------------

    def save_optics_to_stp(
        self,
        keys=None,
        # options
        factor=None,
        color=None,
        chain=None,
        # saving
        pfe=None,
        overwrite=None,
        verb=None,
    ):
        """ Save the selected optics 3d outlines to a stp file

        Optionally chain them to be a single POLYLINE

        """

        return _save2stp.main(
            coll=self,
            keys=keys,
            # options
            factor=factor,
            color=color,
            chain=chain,
            # saving
            pfe=pfe,
            overwrite=overwrite,
            verb=verb,
        )
